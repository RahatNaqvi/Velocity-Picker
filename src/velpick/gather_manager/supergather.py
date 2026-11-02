"""Supergather construction from neighbouring CMP gathers (display / analysis input).

Spacing between analysis CMPs and supergather width are separate parameters (the caller chooses the
CMP list). Which CMPs contribute is always recorded in ``contributing_cmps``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .gather import Gather


def build_supergather(gathers: Sequence[Gather], max_traces: Optional[int] = None) -> Gather:
    """Merge gathers into one, sorted by |offset|. ``max_traces`` keeps the nearest-offset traces
    (GeoEast caps the supergather at 15-500 traces; which traces it keeps is not documented)."""
    if not gathers:
        raise ValueError("no gathers")
    base = gathers[0]
    for g in gathers:
        if (g.dt_ms, g.ns, g.t_start_ms) != (base.dt_ms, base.ns, base.t_start_ms):
            raise ValueError("gathers differ in sampling or start time")
    data = np.concatenate([g.data for g in gathers])
    off = np.concatenate([g.offsets for g in gathers])
    pos = np.concatenate([g.trace_positions for g in gathers])
    dead = np.concatenate([g.dead for g in gathers])
    order = np.argsort(off, kind="stable")
    if max_traces is not None:
        order = order[:max_traces]
    ids = tuple(sorted(int(c) for g in gathers for c in g.contributing_cmps))
    return Gather(cmp_id=int(np.median(ids)), data=data[order], offsets=off[order], dt_ms=base.dt_ms,
                  t_start_ms=base.t_start_ms, trace_positions=pos[order], dead=dead[order],
                  n_nonfinite_samples=int(sum(g.n_nonfinite_samples for g in gathers)),
                  offset_source=base.offset_source, contributing_cmps=ids)
