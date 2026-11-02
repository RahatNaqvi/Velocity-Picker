"""GeoEast-style stack-power velocity spectrum (structure taken from the VelocityAna 3.43 manual).

What the manual states (and what is implemented here):
  * traces are NMO-corrected with a series of scan velocities and stacked; the stack power is the result;
  * stacking is two-step: 3 adjacent CMP gathers are added and the ABSOLUTE VALUE of the sum forms a
    matrix; several such matrices are added to form the spectrum (reduces dip influence);
  * input is band-passed, then equalised (window 300 ms by default); a 2D filter can smooth the power
    matrix; the spectrum is sampled every ``line_interval_ms`` (> 6 x data sample interval).

What the manual does NOT state, so is an explicit assumption here (all exposed as parameters):
  * the stack normalisation (mean over live traces is used), the smoothing window (one line interval),
    the 2D-filter kernel (3 x 3 matrix samples), the NMO-stretch limit (``stretch_mute_pct``), and any
    display gain (see ``row_normalise``).
This is therefore a GeoEast-*like* spectrum, NOT a reproduction of GeoEast numerically.
Picked quantity: stacking (NMO) velocity, hyperbolic model, as in the manual's isotropic P-wave formula.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d

from .nmo import nmo_correct
from .semblance import AnalysisCancelled


@dataclass(frozen=True)
class StackPowerParams:
    vmin: float
    vmax: float
    dv: float
    t0_start_ms: float
    t0_end_ms: float
    line_interval_ms: float = 25.0
    group_size: int = 3                      # CMPs summed before taking |.| (manual: 3)
    stretch_mute_pct: float = 50.0           # assumption (not given in the manual)
    min_live_traces: int = 3
    smooth_ms: Optional[float] = None        # None -> one line interval (assumption)
    filter2d: Optional[Tuple[int, int]] = (3, 3)   # (time lines, velocity samples); None = off

    def velocities(self) -> np.ndarray:
        n = int(np.floor((self.vmax - self.vmin) / self.dv + 1e-9)) + 1
        return self.vmin + self.dv * np.arange(n)

    def lines(self) -> np.ndarray:
        n = int(np.floor((self.t0_end_ms - self.t0_start_ms) / self.line_interval_ms + 1e-9)) + 1
        return self.t0_start_ms + self.line_interval_ms * np.arange(n)

    def validate(self, dt_ms: float) -> None:
        if not (self.vmin > 0 and self.dv > 0 and self.vmax > self.vmin):
            raise ValueError("need 0 < vmin < vmax and dv > 0")
        if not (self.t0_start_ms > 0 and self.t0_end_ms > self.t0_start_ms):
            raise ValueError("need 0 < t0_start < t0_end")
        if self.line_interval_ms <= 6 * dt_ms:
            raise ValueError("line_interval_ms must exceed 6 x the sample interval (GeoEast rule)")
        if self.group_size < 1 or self.min_live_traces < 2:
            raise ValueError("group_size >= 1 and min_live_traces >= 2 required")


@dataclass
class StackPowerSpectrum:
    amplitude: np.ndarray        # [n_lines, n_v]  sum over groups of |group stack amplitude| (after smoothing)
    t0_ms: np.ndarray
    velocities: np.ndarray
    params: StackPowerParams
    cmp_groups: List[Tuple[int, ...]]
    n_traces_used: int
    elapsed_s: float = 0.0

    @property
    def power(self) -> np.ndarray:
        return self.amplitude ** 2

    def row_normalised(self, mode: str = "mean") -> np.ndarray:
        """Display helper: divide each time line by its mean (or max). Not part of the computation."""
        a = self.amplitude
        d = a.mean(axis=1, keepdims=True) if mode == "mean" else a.max(axis=1, keepdims=True)
        return a / np.maximum(d, 1e-30)


def select_spectrum_cmps(available: Sequence[int], center: int, n_requested: int,
                         interval: Optional[int] = None) -> List[int]:
    """CMPs for one spectrum, following the manual: capped by the CMP interval to the next spectrum
    (sec. 5.6b); above 3, a multiple of 3 is used. Neighbours are taken symmetrically about ``center``
    from the ids actually available (no continuity assumed). Exact rule for even counts is unknown;
    ties go to the lower id."""
    n = n_requested if interval is None else min(n_requested, interval)
    if n > 3:
        n = (n // 3) * 3
    n = max(n, 1)
    ids = np.array(sorted(set(int(a) for a in available)))
    order = ids[np.lexsort((ids, np.abs(ids - center)))]
    return sorted(int(x) for x in order[:n])


def select_neighbour_cmps(available: Sequence[int], center: int, n: int) -> List[int]:
    """The ``n`` CMPs nearest to ``center`` among the ids actually available (symmetric about it where possible; ties go to the
    lower id). Used for the traces of one stack segment (GeoEast: 'traces per stack segment')."""
    ids = np.array(sorted(set(int(a) for a in available)))
    order = ids[np.lexsort((ids, np.abs(ids - center)))]
    return sorted(int(x) for x in order[: max(1, int(n))])


def compute_stack_power(gathers: Sequence, dt_ms: float, params: StackPowerParams,
                        *, t_start_ms: float = 0.0,
                        progress: Optional[Callable[[int, int], None]] = None,
                        cancelled: Optional[Callable[[], bool]] = None) -> StackPowerSpectrum:
    """``gathers``: objects with ``.data`` [ntr, ns], ``.offsets``, ``.dead`` and ``.cmp_id`` (see Gather),
    already conditioned by the caller if desired. CMPs are split in consecutive groups of
    ``params.group_size`` (a short last group is kept)."""
    t_begin = time.perf_counter()
    params.validate(dt_ms)
    gs = sorted(gathers, key=lambda g: g.cmp_id)
    k = params.group_size
    groups = [gs[i:i + k] for i in range(0, len(gs), k)]
    vel, lines = params.velocities(), params.lines()
    per_line = 6
    fine = params.t0_start_ms + (params.line_interval_ms / per_line) * np.arange((lines.size - 1) * per_line + 1)
    acc = np.zeros((fine.size, vel.size))
    n_used = 0
    prepared = []
    for grp in groups:
        d = np.concatenate([g.data[~g.dead] for g in grp]).astype(np.float32)
        x = np.concatenate([g.offsets[~g.dead] for g in grp])
        prepared.append((d, x))
        n_used += d.shape[0]
    for iv, v in enumerate(vel):
        if cancelled is not None and cancelled():
            raise AnalysisCancelled()
        for d, x in prepared:
            a, valid = nmo_correct(d, x, dt_ms, fine, float(v), params.stretch_mute_pct, t_start_ms)
            live = valid & (a != 0)
            n = live.sum(axis=0)
            s = np.where(n >= params.min_live_traces, a.sum(axis=0, dtype=np.float64) / np.maximum(n, 1), 0.0)
            acc[:, iv] += np.abs(s)
        if progress is not None:
            progress(iv + 1, vel.size)
    sm = params.smooth_ms if params.smooth_ms is not None else params.line_interval_ms
    nsm = max(1, int(round(sm / (params.line_interval_ms / per_line))))
    if nsm > 1:
        acc = uniform_filter1d(acc, nsm, axis=0, mode="nearest")
    amp = acc[::per_line][: lines.size]
    if params.filter2d:
        amp = uniform_filter(amp, size=params.filter2d, mode="nearest")
    amp = np.maximum(amp, 0.0)          # sums of |.| are >= 0; remove float round-off from the filters
    return StackPowerSpectrum(amplitude=amp.astype(np.float32), t0_ms=lines, velocities=vel, params=params,
                              cmp_groups=[tuple(g.cmp_id for g in grp) for grp in groups],
                              n_traces_used=n_used, elapsed_s=time.perf_counter() - t_begin)
