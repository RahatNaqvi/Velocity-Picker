"""Constant-velocity stacks (CVS): each strip is the stack of NMO-corrected traces for ONE constant velocity.

Used to judge picks visually: where a reflection is flat and strongest in a strip, that strip's velocity is a good
stacking velocity at that time. One stack trace is produced per input CMP gather and per velocity (GeoEast shows
several stack traces per strip, one per CMP). Stack = mean over live (non-muted) traces; zero where fewer than
``min_live`` traces are live. Same NMO and stretch-mute rules as the spectra. Works on copies; no side effects.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .nmo import nmo_correct
from .semblance import AnalysisCancelled


def constant_velocity_stacks(gathers: Sequence, dt_ms: float, velocities: Sequence[float], t_axis_ms: np.ndarray,
                             stretch_pct: float = 50.0, min_live: int = 3, t_start_ms: float = 0.0,
                             progress: Optional[Callable[[int, int], None]] = None,
                             cancelled: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """Return float32 array [n_velocities, n_gathers, n_times]. Gathers need ``.data``, ``.offsets``, ``.dead``."""
    vel = np.asarray(velocities, dtype=float)
    t = np.asarray(t_axis_ms, dtype=float)
    out = np.zeros((vel.size, len(gathers), t.size), dtype=np.float32)
    prepared = [(g.data[~g.dead], g.offsets[~g.dead]) for g in gathers]
    for k, v in enumerate(vel):
        if cancelled is not None and cancelled():
            raise AnalysisCancelled()
        for j, (d, x) in enumerate(prepared):
            if d.shape[0] == 0:
                continue
            a, valid = nmo_correct(d, x, dt_ms, t, float(v), stretch_pct, t_start_ms)
            n = (valid & (a != 0)).sum(axis=0)
            s = a.sum(axis=0, dtype=np.float64) / np.maximum(n, 1)
            out[k, j] = np.where(n >= min_live, s, 0.0)
        if progress is not None:
            progress(k + 1, vel.size)
    return out
