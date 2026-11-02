"""Hyperbolic NMO correction (vectorised, linear interpolation, stretch mute).

Model: tau(x) = sqrt(t0^2 + x^2 / v^2), v = stacking (NMO) velocity under the hyperbolic
assumption.  Units: time ms, offset m, velocity m/s.  Sample j of the input is at
``t_start_ms + j*dt_ms``.  Pure function; the input array is never modified.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def nmo_correct(data: np.ndarray, offsets: np.ndarray, dt_ms: float, t0_ms: np.ndarray,
                velocity: Union[float, np.ndarray], stretch_pct: float,
                t_start_ms: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(corrected, valid)``, both ``[ntr, n_t0]``.

    ``velocity`` is a scalar or an array aligned with ``t0_ms`` (v as a function of t0).
    ``valid`` is False where the source time falls outside the record or where the NMO stretch
    (tau - t0)/t0 exceeds ``stretch_pct``/100; ``corrected`` is 0 there.
    """
    data = np.asarray(data)
    ntr, ns = data.shape
    t0 = np.asarray(t0_ms, dtype=np.float64)
    if np.any(t0 <= 0):
        raise ValueError("t0 values must be > 0")
    v = np.broadcast_to(np.asarray(velocity, dtype=np.float64), t0.shape)
    if np.any(v <= 0):
        raise ValueError("velocities must be > 0")
    x = np.abs(np.asarray(offsets, dtype=np.float64))[:, None]
    tau = np.sqrt(t0[None, :] ** 2 + (1000.0 * x / v[None, :]) ** 2)
    idx = (tau - t_start_ms) / dt_ms
    i0 = np.clip(np.floor(idx), 0, ns - 2).astype(np.int64)
    fr = idx - i0
    valid = (idx >= 0) & (idx <= ns - 1) & ((tau / t0[None, :] - 1.0) <= stretch_pct / 100.0)
    rows = np.arange(ntr)[:, None]
    out = data[rows, i0] * (1.0 - fr) + data[rows, i0 + 1] * fr
    out = np.where(valid, out, 0.0).astype(np.float32)
    return out, valid
