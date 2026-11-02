"""Trace-header index: CMP -> trace positions, offsets, coordinates, validation.

One scan pass over the trace headers builds the index.  Gathers are then located through
the *actual* CMP ids present in the file (no continuity assumption).  Header offsets are
cross-checked against source-receiver distance computed from scaled coordinates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from velpick.data_io.segy import SegyFile

REQUIRED_FIELDS = ("cdp", "offset", "sx", "sy", "gx", "gy", "scalco")


class IndexStaleError(Exception):
    """Cached index does not match the SEG-Y file it is being used with."""


def apply_scalar(values: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    """SEG-Y coordinate scalar: >0 multiply, <0 divide by |scalar|, 0 treated as 1."""
    s = np.asarray(scalar, dtype=np.float64)
    factor = np.where(s > 0, s, np.where(s < 0, 1.0 / np.abs(np.where(s == 0, 1, s)), 1.0))
    return np.asarray(values, dtype=np.float64) * factor


@dataclass(frozen=True)
class Issue:
    severity: str          # "error" | "warning" | "info"
    code: str
    message: str
    count: Optional[int] = None


@dataclass
class ValidationReport:
    issues: List[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def codes(self) -> set:
        return {i.code for i in self.issues}

    def text(self) -> str:
        if not self.issues:
            return "No issues found."
        return "\n".join(f"[{i.severity.upper():7s}] {i.code}: {i.message}" for i in self.issues)


@dataclass
class TraceIndex:
    cdp: np.ndarray
    offset: np.ndarray            # header offset, file units (not scaled)
    computed_offset: np.ndarray   # hypot(sx-gx, sy-gy) from scaled coordinates
    sx: np.ndarray
    sy: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    scalco: np.ndarray
    cunits: np.ndarray
    delrt: np.ndarray             # recording delay (ms) per trace
    trace_ns_values: np.ndarray
    trace_dt_values: np.ndarray
    file_ns: int
    file_dt_us: int
    file_size: int
    file_mtime_ns: int
    headers_sha256: str
    sample_format: int

    def __post_init__(self) -> None:
        order = np.argsort(self.cdp, kind="stable")
        ids, starts, counts = np.unique(self.cdp[order], return_index=True, return_counts=True)
        self._order, self._ids, self._starts, self._counts = order, ids, starts, counts

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, segy: SegyFile) -> "TraceIndex":
        missing = [n for n in REQUIRED_FIELDS if n not in segy.header_map]
        if missing:
            raise ValueError(f"Header map lacks required fields: {missing}")
        h, sha = segy.scan_headers()
        sx = apply_scalar(h["sx"], h["scalco"])
        sy = apply_scalar(h["sy"], h["scalco"])
        gx = apply_scalar(h["gx"], h["scalco"])
        gy = apply_scalar(h["gy"], h["scalco"])
        return cls(
            cdp=h["cdp"], offset=h["offset"].astype(np.float64),
            computed_offset=np.hypot(sx - gx, sy - gy), sx=sx, sy=sy, gx=gx, gy=gy,
            scalco=h["scalco"].astype(np.int32),
            cunits=h["cunits"].astype(np.int32) if "cunits" in h else np.zeros(segy.n_traces, np.int32),
            delrt=h["delrt"].astype(np.int32) if "delrt" in h else np.zeros(segy.n_traces, np.int32),
            trace_ns_values=np.unique(h["ns"]), trace_dt_values=np.unique(h["dt"]),
            file_ns=segy.ns, file_dt_us=segy.dt_us, file_size=segy.file_size,
            file_mtime_ns=segy.file_mtime_ns, headers_sha256=sha,
            sample_format=segy.sample_format)

    # ------------------------------------------------------------------ lookup
    @property
    def n_traces(self) -> int:
        return int(self.cdp.size)

    @property
    def cmp_ids(self) -> np.ndarray:
        return self._ids

    def has_cmp(self, cmp_id: int) -> bool:
        return bool(np.isin(cmp_id, self._ids))

    def positions(self, cmp_id: int) -> np.ndarray:
        """File trace positions (ascending) for one CMP."""
        k = np.searchsorted(self._ids, cmp_id)
        if k >= self._ids.size or self._ids[k] != cmp_id:
            raise KeyError(cmp_id)
        s = self._starts[k]
        return self._order[s: s + self._counts[k]]

    def fold(self, cmp_id: int) -> int:
        return int(self._counts[np.searchsorted(self._ids, cmp_id)])

    def missing(self, requested: Iterable[int]) -> List[int]:
        return sorted(int(c) for c in requested if not self.has_cmp(int(c)))

    def nearest_ids(self, cmp_id: int, k: int = 2) -> List[int]:
        d = np.abs(self._ids - cmp_id)
        return [int(x) for x in np.sort(self._ids[np.argsort(d, kind="stable")[:k]])]

    def cmp_summary(self, cmp_id: int) -> dict:
        p = self.positions(cmp_id)
        off = np.abs(self.offset[p])
        return {
            "cmp": int(cmp_id), "fold": int(p.size),
            "offset_min": float(off.min()), "offset_max": float(off.max()),
            "duplicate_offsets": int(p.size - np.unique(off).size),
            "mid_x": float(((self.sx[p] + self.gx[p]) / 2).mean()),
            "mid_y": float(((self.sy[p] + self.gy[p]) / 2).mean()),
        }

    # ------------------------------------------------------------------ validation
    def validate(self, offset_tol: float = 1.0, min_fold: int = 6) -> ValidationReport:
        """Report header problems that matter for velocity picking. Warnings, not verdicts."""
        r: List[Issue] = []
        add = lambda sev, code, msg, n=None: r.append(Issue(sev, code, msg, n)) # noqa: E731

        if self.file_dt_us == 0:
            add("error", "DT_ZERO", "Sample interval is zero; time axis cannot be built.")
        if self.trace_dt_values.size > 1 or (self.trace_dt_values.size == 1
                                             and self.trace_dt_values[0] not in (0, self.file_dt_us)):
            add("warning", "DT_MISMATCH", f"Trace-header dt values {self.trace_dt_values.tolist()} "
                f"differ from file dt {self.file_dt_us} us.")
        if self.trace_ns_values.size > 1 or (self.trace_ns_values.size == 1
                                             and self.trace_ns_values[0] not in (0, self.file_ns)):
            add("warning", "NS_MISMATCH", f"Trace-header ns values {self.trace_ns_values.tolist()} "
                f"differ from file samples/trace {self.file_ns}.")

        if np.all(self.cdp == 0):
            add("error", "NO_CMP_IDS", "All CMP/CDP header values are zero; gathers cannot be located "
                "(CMP binning from geometry is not implemented yet).")
        else:
            nz = int((self.cdp == 0).sum())
            if nz:
                add("warning", "CMP_ZERO", "Some traces have CMP id 0 and form a 'CMP 0' gather.", nz)
            ids = self._ids
            if ids.size > 1:
                diffs = np.unique(np.diff(ids))
                span = int(ids[-1] - ids[0] + 1)
                if span != ids.size:
                    add("info", "CMP_NONCONTIGUOUS",
                        f"{ids.size} CMP ids between {ids[0]} and {ids[-1]}; {span - ids.size} integer "
                        f"ids in that range have no traces (id steps seen: {diffs[:8].tolist()}).")
            if not np.all(np.diff(self.cdp) >= 0):
                add("info", "NOT_CMP_SORTED", "Traces are not stored in CMP order; gathers are "
                    "located through the index.")
            low = int((self._counts < min_fold).sum())
            if low:
                add("warning", "LOW_FOLD", f"{low} CMP(s) have fold below {min_fold} "
                    f"(min fold {int(self._counts.min())}).", low)

        absoff = np.abs(self.offset)
        if np.all(absoff == 0):
            if np.median(self.computed_offset) > 0:
                add("warning", "OFFSET_HEADER_ZERO", "Header offsets are all zero; offsets can be "
                    "computed from coordinates (select offset source 'computed').")
            else:
                add("error", "NO_OFFSET", "Offsets are zero in the header and coordinates give no "
                    "separation; NMO and semblance are impossible.")
        else:
            zero = int((absoff == 0).sum())
            if zero:
                add("warning", "OFFSET_ZERO_TRACES", "Some traces have zero offset.", zero)
            if (self.offset < 0).any():
                add("info", "OFFSET_SIGNED", "Negative offsets present; |offset| is used (hyperbolic "
                    "NMO depends on x squared).", int((self.offset < 0).sum()))
            if np.median(self.computed_offset) > 0:
                diff = np.abs(self.computed_offset - absoff)
                bad = int((diff > offset_tol).sum())
                if bad:
                    add("warning", "OFFSET_MISMATCH", f"Header offset differs from coordinate distance "
                        f"by more than {offset_tol} (max {diff.max():.2f}). Check coordinate scalar, "
                        f"units, or header byte positions.", bad)

        if np.unique(self.delrt).size > 1:
            add("warning", "DELRT_VARIES", "Recording delay differs between traces; gathers mixing "
                "different delays cannot be extracted.")
        elif self.delrt.size and self.delrt[0] != 0:
            add("info", "DELRT_NONZERO", f"Recording delay is {int(self.delrt[0])} ms; the time axis "
                "starts at that time.")
        vals = np.unique(self.scalco)
        if vals.size > 1:
            add("warning", "SCALCO_MIXED", f"Coordinate scalars differ across traces: {vals[:6].tolist()}.")
        elif vals.size == 1 and vals[0] == 0:
            add("info", "SCALCO_ZERO", "Coordinate scalar is 0; treated as 1.")
        if np.all(self.cunits == 0):
            add("info", "COORD_UNITS_UNKNOWN", "Coordinate-units flag is 0 (unknown). Units are only "
                "inferred through the offset cross-check.")

        dup = sum(1 for c in self._ids if self.cmp_summary(int(c))["duplicate_offsets"] > 0)
        if dup:
            add("info", "DUPLICATE_OFFSETS", f"{dup} CMP(s) contain repeated offsets (expected for "
                "wide-azimuth data); the engine uses offset only and ignores azimuth.", dup)
        return ValidationReport(r)

    # ------------------------------------------------------------------ cache
    _ARRAYS = ("cdp", "offset", "computed_offset", "sx", "sy", "gx", "gy", "scalco", "cunits", "delrt",
               "trace_ns_values", "trace_dt_values")
    _META = ("file_ns", "file_dt_us", "file_size", "file_mtime_ns", "headers_sha256", "sample_format")

    def save(self, path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        meta = json.dumps({k: getattr(self, k) for k in self._META})
        with open(tmp, "wb") as f:
            np.savez_compressed(f, meta=np.array(meta), **{k: getattr(self, k) for k in self._ARRAYS})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, segy: SegyFile) -> "TraceIndex":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in cls._ARRAYS}
        if (meta["file_size"], meta["file_mtime_ns"]) != (segy.file_size, segy.file_mtime_ns):
            raise IndexStaleError("SEG-Y file size or modification time changed since the index was "
                                  "built; rebuild the index.")
        if arrays["cdp"].size != segy.n_traces:
            raise IndexStaleError("Trace count differs from the cached index; rebuild the index.")
        return cls(**arrays, **meta)

    @classmethod
    def load_or_build(cls, segy: SegyFile, cache_path) -> "TraceIndex":
        try:
            return cls.load(cache_path, segy)
        except (FileNotFoundError, IndexStaleError, KeyError, ValueError):
            idx = cls.build(segy)
            idx.save(cache_path)
            return idx
