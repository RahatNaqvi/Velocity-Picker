"""Analysis parameters, validation and cost estimation.

Units everywhere: time in ms, offsets in metres, velocities in m/s.  The engine cannot know the
unit of file offsets; convert (e.g. feet -> metres) before analysis.

There are deliberately no default velocity range, time range or gate length: those depend on the
survey and must be chosen by the processor.  Only technical safeguards have defaults.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AnalysisParams:
    vmin: float                        # m/s
    vmax: float                        # m/s
    dv: float                          # m/s
    t0_start_ms: float                 # first zero-offset time analysed (must be > 0)
    t0_end_ms: float
    window_ms: float                   # semblance gate length, centred on each t0
    t0_step_ms: Optional[float] = None  # output time step; None -> data sample interval
    stretch_mute_pct: float = 30.0     # mute where (tau - t0)/t0 exceeds this percentage
    min_live_traces: int = 3           # semblance forced to 0 where fewer traces are live
    offset_min: Optional[float] = None  # optional |offset| limits (m) for traces used
    offset_max: Optional[float] = None
    smooth_t_ms: float = 0.0           # optional boxcar smoothing of the result along time
    smooth_v_samples: int = 1          # optional boxcar smoothing along velocity (odd; 1 = off)
    method: str = "hyperbolic_semblance"
    interpolation: str = "linear"

    def velocities(self) -> np.ndarray:
        n = int(np.floor((self.vmax - self.vmin) / self.dv + 1e-9)) + 1
        return self.vmin + self.dv * np.arange(n)

    def step_ms(self, dt_ms: float) -> float:
        return float(self.t0_step_ms) if self.t0_step_ms else float(dt_ms)

    def t0_axis(self, dt_ms: float) -> np.ndarray:
        step = self.step_ms(dt_ms)
        n = int(np.floor((self.t0_end_ms - self.t0_start_ms) / step + 1e-9)) + 1
        return self.t0_start_ms + step * np.arange(n)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ParamIssue:
    severity: str   # "error" | "warning" | "info"
    code: str
    message: str


class ParamValidationError(ValueError):
    def __init__(self, issues: Sequence[ParamIssue]):
        self.issues = list(issues)
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in issues if i.severity == "error"))


@dataclass(frozen=True)
class CostEstimate:
    n_velocities: int
    n_t0: int
    n_traces: int
    n_interpolations: int          # trace-samples interpolated in total
    result_bytes: int              # float32 semblance panel
    working_bytes: int             # approx. peak transient memory for one trial velocity


def estimate_cost(params: AnalysisParams, dt_ms: float, n_traces: int) -> CostEstimate:
    nv, nt = params.velocities().size, params.t0_axis(dt_ms).size
    return CostEstimate(nv, nt, n_traces, nv * nt * n_traces, nv * nt * 4,
                        n_traces * nt * 8 * 6)


def _tau(t0, x, v):
    return float(np.sqrt(t0 ** 2 + (1000.0 * x / v) ** 2))


def validate_params(p: AnalysisParams, dt_ms: float, ns: int, offsets: Sequence[float],
                    t_start_ms: float = 0.0) -> List[ParamIssue]:
    """Check parameter combinations against the data. Errors block the run; warnings do not."""
    out: List[ParamIssue] = []
    add = lambda s, c, m: out.append(ParamIssue(s, c, m))  # noqa: E731
    off = np.abs(np.asarray(offsets, dtype=float))
    rec_end = t_start_ms + (ns - 1) * dt_ms

    if p.method != "hyperbolic_semblance":
        add("error", "METHOD", f"Unknown method '{p.method}'.")
    if p.interpolation != "linear":
        add("error", "INTERP", f"Interpolation '{p.interpolation}' not implemented (only 'linear').")
    if not (p.vmin > 0):
        add("error", "VMIN", "vmin must be > 0.")
    if not (p.dv > 0):
        add("error", "DV", "dv must be > 0.")
    if not (p.vmax > p.vmin):
        add("error", "VRANGE", "vmax must be greater than vmin.")
    if p.dv > 0 and p.vmax > p.vmin and p.velocities().size < 2:
        add("error", "VCOUNT", "Fewer than two trial velocities.")
    if not (p.t0_start_ms > 0):
        add("error", "T0_START", "t0_start must be > 0 (stretch is undefined at t0 = 0).")
    if p.t0_end_ms <= p.t0_start_ms:
        add("error", "T0_RANGE", "t0_end must be greater than t0_start.")
    if p.t0_start_ms < t_start_ms or p.t0_end_ms > rec_end:
        add("error", "T0_OUTSIDE", f"t0 range must lie within the record ({t_start_ms:g}-{rec_end:g} ms).")
    step = p.step_ms(dt_ms)
    if not (step > 0):
        add("error", "T0_STEP", "t0 step must be > 0.")
    elif step < dt_ms:
        add("info", "T0_STEP_FINE", "t0 step is smaller than the sample interval (oversampled output).")
    if not (p.window_ms > 0):
        add("error", "WINDOW", "window_ms must be > 0.")
    elif step > 0 and p.window_ms < 2 * step:
        add("warning", "WINDOW_SHORT", "Gate is shorter than two output samples; semblance will be "
            "extremely noisy.")
    elif p.window_ms < 4 * dt_ms:
        add("warning", "WINDOW_SHORT", "Gate is shorter than four sample intervals; semblance may be noisy.")
    if not (p.stretch_mute_pct > 0):
        add("error", "STRETCH", "stretch_mute_pct must be > 0.")
    elif p.stretch_mute_pct > 100:
        add("warning", "STRETCH_LARGE", "Stretch mute above 100 % admits heavily stretched far offsets.")
    if p.min_live_traces < 2:
        add("error", "MINTR", "min_live_traces must be at least 2 (semblance of one trace is always 1).")
    if p.smooth_v_samples < 1 or p.smooth_v_samples % 2 == 0:
        add("error", "SMOOTH_V", "smooth_v_samples must be a positive odd integer.")
    if p.smooth_t_ms < 0:
        add("error", "SMOOTH_T", "smooth_t_ms must be >= 0.")
    if p.offset_min is not None and p.offset_max is not None and p.offset_max <= p.offset_min:
        add("error", "OFFSET_RANGE", "offset_max must be greater than offset_min.")

    sel = off.copy()
    if p.offset_min is not None:
        sel = sel[sel >= p.offset_min]
    if p.offset_max is not None:
        sel = sel[sel <= p.offset_max]
    if sel.size < p.min_live_traces:
        add("error", "TOO_FEW_TRACES", f"Only {sel.size} trace(s) inside the offset limits; need at least "
            f"{p.min_live_traces}.")
    elif np.all(sel == 0):
        add("error", "NO_OFFSET", "All offsets are zero; moveout analysis is impossible.")

    if not any(i.severity == "error" for i in out) and sel.size:
        xmax, m = float(sel.max()), p.stretch_mute_pct / 100.0
        # velocity resolution: moveout difference between adjacent trial velocities at the mute limit
        t0, v = p.t0_end_ms, p.vmin
        x = min(xmax, v * (t0 / 1000.0) * np.sqrt((1 + m) ** 2 - 1))
        d = abs(_tau(t0, x, v) - _tau(t0, x, v + p.dv))
        if d > p.window_ms / 2:
            add("warning", "DV_COARSE", f"At t0={t0:g} ms, v={v:g} m/s the moveout step between adjacent "
                f"trial velocities is {d:.1f} ms, more than half the gate ({p.window_ms / 2:g} ms); "
                "peaks may be skipped. Reduce dv or lengthen the gate.")
        far = _tau(t0, xmax, p.vmax) - t0
        if far < p.window_ms / 2:
            add("warning", "LOW_RESOLUTION", f"At t0={t0:g} ms and v={p.vmax:g} m/s the far-offset "
                f"moveout is only {far:.1f} ms (< half the gate); velocity discrimination will be poor "
                "at the deep end.")
    return out
