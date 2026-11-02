"""CMP gather extraction (single CMPs; supergathers come later)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from velpick.data_io.segy import SegyFile
from velpick.geometry.trace_index import TraceIndex


class CmpNotFoundError(KeyError):
    """Requested CMP id has no traces in the index."""


@dataclass
class Gather:
    cmp_id: int
    data: np.ndarray               # float32 [ntr, ns], sorted by increasing |offset|
    offsets: np.ndarray            # float64 [ntr], |offset| in file units (metres for the test file)
    dt_ms: float
    t_start_ms: float              # time of first sample (recording delay)
    trace_positions: np.ndarray    # file trace numbers, same order as data
    dead: np.ndarray               # bool [ntr]: all-zero (after cleaning) traces
    n_nonfinite_samples: int
    offset_source: str
    contributing_cmps: Tuple[int, ...]

    @property
    def ntr(self) -> int:
        return int(self.data.shape[0])

    @property
    def ns(self) -> int:
        return int(self.data.shape[1])

    @property
    def times_ms(self) -> np.ndarray:
        return self.t_start_ms + np.arange(self.ns) * self.dt_ms


def extract_gather(segy: SegyFile, index: TraceIndex, cmp_id: int,
                   offset_source: str = "header") -> Gather:
    """Read one CMP gather. Non-finite samples are set to 0 and counted; dead traces are flagged,
    not removed. Source data is never modified."""
    if offset_source not in ("header", "computed"):
        raise ValueError("offset_source must be 'header' or 'computed'")
    if not index.has_cmp(cmp_id):
        raise CmpNotFoundError(f"CMP {cmp_id} not in file; nearest available ids: "
                               f"{index.nearest_ids(cmp_id)}")
    pos = index.positions(cmp_id)
    delays = np.unique(index.delrt[pos])
    if delays.size > 1:
        raise ValueError(f"CMP {cmp_id}: traces have different recording delays {delays.tolist()} ms; "
                         "not supported.")
    t_start = float(delays[0])
    raw = index.offset[pos] if offset_source == "header" else index.computed_offset[pos]
    offsets = np.abs(raw).astype(np.float64)
    order = np.argsort(offsets, kind="stable")
    pos, offsets = pos[order], offsets[order]
    data = segy.read_traces(pos)
    finite = np.isfinite(data)
    n_bad = int(data.size - finite.sum())
    if n_bad:
        data = np.where(finite, data, 0.0).astype(np.float32)
    dead = ~np.any(data != 0, axis=1)
    return Gather(cmp_id=int(cmp_id), data=data, offsets=offsets, dt_ms=segy.dt_ms, t_start_ms=t_start,
                  trace_positions=pos, dead=dead, n_nonfinite_samples=n_bad,
                  offset_source=offset_source, contributing_cmps=(int(cmp_id),))


def semblance_from_gather(gather: Gather, params, **kw):
    """Convenience: run the engine on a Gather (dead traces excluded)."""
    from velpick.velocity_analysis import compute_semblance
    return compute_semblance(gather.data, gather.offsets, gather.dt_ms, params,
                             t_start_ms=gather.t_start_ms, live=~gather.dead,
                             cmp_id=gather.cmp_id, **kw)
