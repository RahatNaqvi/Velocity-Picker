"""Hyperbolic-NMO semblance (Neidell & Taner, 1971 definition).

For each trial velocity v and zero-offset time t0 the gather is NMO-corrected (see nmo.py) and

    S(t0, v) = sum_{j in gate} ( sum_i a_ij )^2  /  sum_{j in gate} ( n_j * sum_i a_ij^2 )

where i runs over live (non-muted, non-dead) traces, n_j is the number of live traces at gate
sample j, and the gate is a boxcar centred on t0 of about ``window_ms`` (2h+1 output samples,
h = round(window_ms / (2*step))), truncated at the ends of the analysed time range.
By Cauchy-Schwarz 0 <= S <= 1.  Where the number of live traces at the centre sample is below
``min_live_traces`` S is set to 0 (with one trace S would be exactly 1, which is meaningless).

Assumptions / limits: constant-velocity hyperbolic moveout (stacking velocity, not interval
velocity); azimuth ignored; linear interpolation (attenuates high frequencies, affects peak
amplitude but not peak position); first sample at ``t_start_ms``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from .nmo import nmo_correct
from .params import AnalysisParams, ParamValidationError, validate_params


class AnalysisCancelled(Exception):
    """Raised when the caller's ``cancelled`` callback returns True."""


@dataclass
class Semblance:
    values: np.ndarray            # float32 [n_t0, n_v], in [0, 1]
    t0_ms: np.ndarray
    velocities: np.ndarray        # m/s
    params: AnalysisParams
    params_hash: str
    n_traces_used: int
    cmp_id: Optional[int] = None
    elapsed_s: float = 0.0

    def peak_velocity(self, t0_ms: float) -> Tuple[float, float]:
        """(velocity, semblance) of the maximum on the row nearest ``t0_ms``."""
        r = int(np.argmin(np.abs(self.t0_ms - t0_ms)))
        k = int(np.argmax(self.values[r]))
        return float(self.velocities[k]), float(self.values[r, k])


def _boxsum(x: np.ndarray, h: int) -> np.ndarray:
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    j = np.arange(n)
    return c[np.clip(j + h + 1, 0, n)] - c[np.clip(j - h, 0, n)]


def compute_semblance(data: np.ndarray, offsets: np.ndarray, dt_ms: float, params: AnalysisParams,
                      *, t_start_ms: float = 0.0, live: Optional[np.ndarray] = None,
                      cmp_id: Optional[int] = None,
                      progress: Optional[Callable[[int, int], None]] = None,
                      cancelled: Optional[Callable[[], bool]] = None) -> Semblance:
    """Compute the semblance panel for one gather. Raises ParamValidationError on invalid input."""
    t_begin = time.perf_counter()
    data = np.asarray(data)
    offsets = np.abs(np.asarray(offsets, dtype=np.float64))
    keep = np.ones(data.shape[0], dtype=bool) if live is None else np.asarray(live, dtype=bool).copy()
    if params.offset_min is not None:
        keep &= offsets >= params.offset_min
    if params.offset_max is not None:
        keep &= offsets <= params.offset_max
    issues = validate_params(params, dt_ms, data.shape[1], offsets[keep], t_start_ms)
    if any(i.severity == "error" for i in issues):
        raise ParamValidationError(issues)
    d, x = data[keep], offsets[keep]

    t0 = params.t0_axis(dt_ms)
    vel = params.velocities()
    step = params.step_ms(dt_ms)
    h = int(round(params.window_ms / (2.0 * step)))
    out = np.zeros((t0.size, vel.size), dtype=np.float32)

    for k, v in enumerate(vel):
        if cancelled is not None and cancelled():
            raise AnalysisCancelled()
        a, valid = nmo_correct(d, x, dt_ms, t0, float(v), params.stretch_mute_pct, t_start_ms)
        a64 = a.astype(np.float64)
        n = valid.sum(axis=0).astype(np.float64)
        num = _boxsum(a64.sum(axis=0) ** 2, h)
        den = _boxsum(n * (a64 ** 2).sum(axis=0), h)
        s = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        s[n < params.min_live_traces] = 0.0
        out[:, k] = np.clip(s, 0.0, 1.0)
        if progress is not None:
            progress(k + 1, vel.size)

    nt = int(round(params.smooth_t_ms / step))
    if nt > 1:
        out = uniform_filter1d(out, size=nt, axis=0, mode="nearest")
    if params.smooth_v_samples > 1:
        out = uniform_filter1d(out, size=params.smooth_v_samples, axis=1, mode="nearest")
    return Semblance(values=out.astype(np.float32), t0_ms=t0, velocities=vel, params=params,
                     params_hash=params.hash(), n_traces_used=int(keep.sum()), cmp_id=cmp_id,
                     elapsed_s=time.perf_counter() - t_begin)
