"""Trace conditioning applied to *copies* of the data before velocity analysis.

Order used by GeoEast VelocityAna (manual sec. 5.2f): filter first, then equalise.  Mute is applied
as a top mute.  None of these functions modify their input.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d


def ormsby_bandpass(data: np.ndarray, dt_ms: float, f1: float, f2: float, f3: float, f4: float) -> np.ndarray:
    """Zero-phase Ormsby trapezoid (low cut f1, low pass f2, high pass f3, high cut f4, Hz), via FFT.

    Not modelled: GeoEast's finite 'filter operator length' (ringing / edge effects differ slightly).
    """
    if not (0 <= f1 < f2 <= f3 < f4):
        raise ValueError("need 0 <= f1 < f2 <= f3 < f4")
    d = np.asarray(data, dtype=np.float64)
    n = d.shape[-1]
    nf = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nf, dt_ms / 1000.0)
    w = np.interp(f, [0, f1, f2, f3, f4, f[-1] + 1.0], [0, 0, 1, 1, 0, 0])
    out = np.fft.irfft(np.fft.rfft(d, nf, axis=-1) * w, nf, axis=-1)[..., :n]
    return out.astype(np.float32)


def equalize(data: np.ndarray, dt_ms: float, window_ms: float) -> np.ndarray:
    """Divide every trace by its sliding RMS over ``window_ms`` (centred). 0 = no equalisation;
    values below 200 ms are raised to 200 ms, as in the GeoEast manual. All-zero traces stay zero."""
    if window_ms <= 0:
        return np.asarray(data, dtype=np.float32).copy()
    n = max(3, int(round(max(window_ms, 200.0) / dt_ms)))
    d = np.asarray(data, dtype=np.float64)
    rms = np.sqrt(np.maximum(uniform_filter1d(d ** 2, n, axis=-1, mode="nearest"), 0.0))
    floor = 1e-6 * max(float(rms.max()), 1e-30)
    return np.where(rms > floor, d / np.maximum(rms, floor), 0.0).astype(np.float32)


def top_mute(data: np.ndarray, offsets: np.ndarray, dt_ms: float,
             points: Sequence[Tuple[float, float]], taper_ms: float = 0.0,
             t_start_ms: float = 0.0) -> np.ndarray:
    """Zero samples above the mute time for each offset.

    ``points`` = [(offset_m, time_ms), ...]; linear interpolation inside, first value before, linear
    extrapolation from the last two points after (the rule given in the GeoEast manual). A linear
    taper of ``taper_ms`` below the mute time ramps the amplitude from 0 to 1.
    """
    p = sorted(points)
    xo = np.array([a for a, _ in p], dtype=float)
    to = np.array([b for _, b in p], dtype=float)
    x = np.abs(np.asarray(offsets, dtype=float))
    tm = np.interp(x, xo, to)
    if len(p) >= 2:
        s = (to[-1] - to[-2]) / max(xo[-1] - xo[-2], 1e-9)
        beyond = x > xo[-1]
        tm[beyond] = to[-1] + s * (x[beyond] - xo[-1])
    t = t_start_ms + np.arange(np.asarray(data).shape[-1]) * dt_ms
    if taper_ms > 0:
        w = np.clip((t[None, :] - tm[:, None]) / taper_ms, 0.0, 1.0)
    else:
        w = (t[None, :] >= tm[:, None]).astype(float)
    return (np.asarray(data, dtype=np.float64) * w).astype(np.float32)
