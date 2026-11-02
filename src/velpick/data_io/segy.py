"""Read-only SEG-Y access built on a numpy memmap.

Scope (Stage 2): big-endian SEG-Y rev 0/1 with fixed-length traces and sample formats
1 (IBM float), 2 (int32), 3 (int16), 5 (IEEE float), 8 (int8).  Anything else raises
``SegyFormatError`` with an explanation instead of guessing.

The source file is opened read-only and is never modified.  Trace-header byte positions
come from ``DEFAULT_TRACE_HEADER_MAP`` (SEG-Y standard locations) and can be overridden
per project, because field files often put values in non-standard bytes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

TEXT_HEADER_BYTES = 3200
BINARY_HEADER_BYTES = 400
FILE_HEADER_BYTES = TEXT_HEADER_BYTES + BINARY_HEADER_BYTES
TRACE_HEADER_BYTES = 240


class SegyError(Exception):
    """Base class for SEG-Y problems."""


class SegyFormatError(SegyError):
    """The file is not in a layout this reader supports (or is inconsistent)."""


@dataclass(frozen=True)
class HeaderField:
    """A trace-header field. ``byte`` is the 1-based first byte (SEG-Y convention)."""
    name: str
    byte: int
    nbytes: int
    signed: bool = True

    @property
    def be_dtype(self) -> np.dtype:
        return np.dtype(f">{'i' if self.signed else 'u'}{self.nbytes}")


def _fields(*items) -> dict:
    return {f.name: f for f in (HeaderField(*i) for i in items)}


DEFAULT_TRACE_HEADER_MAP: Mapping[str, HeaderField] = _fields(
    ("tracl", 1, 4), ("tracr", 5, 4), ("fldr", 9, 4), ("tracf", 13, 4),
    ("ep", 17, 4), ("cdp", 21, 4), ("cdpt", 25, 4), ("trid", 29, 2),
    ("offset", 37, 4), ("scalel", 69, 2), ("scalco", 71, 2),
    ("sx", 73, 4), ("sy", 77, 4), ("gx", 81, 4), ("gy", 85, 4),
    ("cunits", 89, 2), ("delrt", 109, 2), ("ns", 115, 2, False), ("dt", 117, 2, False),
)

# format code -> (numpy big-endian dtype or None for IBM, bytes/sample, description)
_SAMPLE_FORMATS = {
    1: (None, 4, "IBM 32-bit float"),
    2: (">i4", 4, "32-bit integer"),
    3: (">i2", 2, "16-bit integer"),
    5: (">f4", 4, "IEEE 32-bit float"),
    8: ("i1", 1, "8-bit integer"),
}


def ibm_to_ieee(words) -> np.ndarray:
    """Convert IBM System/360 32-bit floats (as uint32 words) to float32."""
    w = np.asarray(words).astype(np.uint32, copy=False)
    sign = (w >> 31).astype(np.float64)
    exp = ((w >> 24) & 0x7F).astype(np.int32) - 64
    frac = (w & 0x00FFFFFF).astype(np.float64) / 16777216.0
    return ((1.0 - 2.0 * sign) * frac * np.power(16.0, exp)).astype(np.float32)


def _be(buf: bytes, byte: int, nbytes: int, signed: bool = False) -> int:
    return int.from_bytes(buf[byte - 1: byte - 1 + nbytes], "big", signed=signed)


def _decode_text_header(raw: bytes) -> list:
    candidates = [raw.decode("cp500", "replace"), raw.decode("ascii", "replace")]

    def score(t):
        return sum(1 for i in range(0, 3200, 80) if t[i:i + 1] == "C")

    text = max(candidates, key=score)
    return [text[i:i + 80].rstrip() for i in range(0, 3200, 80)]


class SegyFile:
    """Read-only view of one SEG-Y file."""

    def __init__(self, path, header_map: Optional[Mapping[str, HeaderField]] = None):
        self.path = Path(path)
        self.header_map = dict(header_map or DEFAULT_TRACE_HEADER_MAP)
        self.warnings: list = []
        self._mm = None
        self._parse()

    # ------------------------------------------------------------------ parsing
    def _parse(self) -> None:
        if not self.path.is_file():
            raise SegyError(f"File not found: {self.path}")
        st = self.path.stat()
        self.file_size, self.file_mtime_ns = st.st_size, st.st_mtime_ns
        if self.file_size < FILE_HEADER_BYTES + TRACE_HEADER_BYTES:
            raise SegyFormatError("File is too small to contain SEG-Y headers and one trace.")
        with open(self.path, "rb") as f:
            head = f.read(FILE_HEADER_BYTES + TRACE_HEADER_BYTES)
        self.text_header = _decode_text_header(head[:TEXT_HEADER_BYTES])

        fmt = _be(head, 3225, 2)
        if fmt not in _SAMPLE_FORMATS:
            if int.from_bytes(head[3224:3226], "little") in _SAMPLE_FORMATS:
                raise SegyFormatError("Binary header looks little-endian; only big-endian "
                                      "SEG-Y is supported in this version.")
            raise SegyFormatError(f"Unsupported or invalid sample format code {fmt} "
                                  f"(supported: {sorted(_SAMPLE_FORMATS)}).")
        self.sample_format = fmt
        self._np_dtype, self.bytes_per_sample, self.format_name = _SAMPLE_FORMATS[fmt]

        self.revision_major = _be(head, 3501, 2) >> 8
        n_ext = _be(head, 3505, 2, signed=True)
        if self.revision_major >= 2:
            self.warnings.append("SEG-Y revision 2 flagged; read using rev 0/1 layout rules.")
        if self.revision_major >= 1 and n_ext < 0:
            raise SegyFormatError("Variable number of extended textual headers is not supported.")
        self.data_start = FILE_HEADER_BYTES + (TEXT_HEADER_BYTES * n_ext
                                               if self.revision_major >= 1 and n_ext > 0 else 0)

        bin_dt = _be(head, 3217, 2)
        bin_ns = _be(head, 3221, 2)
        f_ns, f_dt = self.header_map["ns"], self.header_map["dt"]
        off = FILE_HEADER_BYTES
        t0_ns = int.from_bytes(head[off + f_ns.byte - 1: off + f_ns.byte - 1 + f_ns.nbytes], "big")
        t0_dt = int.from_bytes(head[off + f_dt.byte - 1: off + f_dt.byte - 1 + f_dt.nbytes], "big")

        payload = self.file_size - self.data_start
        tried = []
        self.ns = 0
        for cand in (bin_ns, t0_ns):
            if cand <= 0 or cand in tried:
                continue
            tried.append(cand)
            tsize = TRACE_HEADER_BYTES + cand * self.bytes_per_sample
            if payload % tsize == 0:
                self.ns = cand
                break
        if not self.ns:
            raise SegyFormatError(
                f"File size is not a whole number of traces. Samples/trace candidates "
                f"(binary header {bin_ns}, first trace header {t0_ns}) do not divide the "
                f"{payload} data bytes. The file may be truncated, variable-length, or use "
                f"extended headers.")
        if bin_ns != t0_ns:
            self.warnings.append(f"Binary-header ns ({bin_ns}) differs from first trace "
                                 f"header ns ({t0_ns}); using {self.ns} (matches file size).")
        self.dt_us = bin_dt or t0_dt
        if not self.dt_us:
            self.warnings.append("Sample interval is zero in both binary and trace header.")
        elif bin_dt and t0_dt and bin_dt != t0_dt:
            self.warnings.append(f"Binary-header dt ({bin_dt} us) differs from first trace "
                                 f"header dt ({t0_dt} us); using {bin_dt}.")
        self.trace_size = TRACE_HEADER_BYTES + self.ns * self.bytes_per_sample
        self.n_traces = payload // self.trace_size
        if self.n_traces == 0:
            raise SegyFormatError("No traces in file.")

    # ------------------------------------------------------------------ access
    def _raw(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.data_start,
                                 shape=(self.n_traces, self.trace_size))
        return self._mm

    def close(self) -> None:
        self._mm = None

    @property
    def dt_ms(self) -> float:
        return self.dt_us / 1000.0

    def scan_headers(self, fields: Optional[Sequence[str]] = None,
                     chunk_traces: int = 50_000):
        """Read trace-header fields for every trace, in chunks.

        Returns ``(arrays, sha256_hex)``; ``arrays[name]`` is int64 of length n_traces and
        the hash covers all raw 240-byte trace headers (a content identifier of the input).
        """
        names = list(fields) if fields else list(self.header_map)
        out = {n: np.empty(self.n_traces, dtype=np.int64) for n in names}
        h = hashlib.sha256()
        raw = self._raw()
        for i in range(0, self.n_traces, chunk_traces):
            j = min(i + chunk_traces, self.n_traces)
            block = np.array(raw[i:j, :TRACE_HEADER_BYTES])
            h.update(block.tobytes())
            for n in names:
                f = self.header_map[n]
                col = np.ascontiguousarray(block[:, f.byte - 1: f.byte - 1 + f.nbytes])
                out[n][i:j] = col.view(f.be_dtype).ravel()
        return out, h.hexdigest()

    def read_traces(self, positions) -> np.ndarray:
        """Return samples as float32 [len(positions), ns]; positions are 0-based trace numbers."""
        pos = np.asarray(positions, dtype=np.int64).ravel()
        if pos.size and (pos.min() < 0 or pos.max() >= self.n_traces):
            raise IndexError(f"Trace position out of range 0..{self.n_traces - 1}")
        if pos.size == 0:
            return np.empty((0, self.ns), dtype=np.float32)
        body = np.ascontiguousarray(self._raw()[pos, TRACE_HEADER_BYTES:])
        if self._np_dtype is None:
            return ibm_to_ieee(body.view(">u4"))
        return body.view(self._np_dtype).astype(np.float32)

    def info(self) -> dict:
        return {
            "path": str(self.path), "size_bytes": self.file_size, "n_traces": self.n_traces,
            "samples_per_trace": self.ns, "dt_us": self.dt_us,
            "record_length_ms": (self.ns - 1) * self.dt_ms,
            "sample_format": f"{self.sample_format} ({self.format_name})",
            "revision_major": self.revision_major, "warnings": list(self.warnings),
        }
