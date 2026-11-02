"""Run an analysis for one CMP (no Qt imports; executed on a worker thread by the window)."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from velpick.data_io import SegyFile
from velpick.gather_manager import Gather, extract_gather
from velpick.geometry import TraceIndex
from velpick.velocity_analysis import (AnalysisParams, StackPowerParams, compute_semblance, compute_stack_power,
                                       constant_velocity_stacks, equalize, ormsby_bandpass, select_neighbour_cmps,
                                       select_spectrum_cmps)
from .settings import AnalysisSettings


@dataclass
class AnalysisResult:
    cmp_id: int
    method: str
    display: np.ndarray            # [n_lines, n_v] float32, what the spectrum panel shows
    t0_ms: np.ndarray
    velocities: np.ndarray
    cmps_used: List[int]
    gather: Gather                 # raw centre gather (never modified)
    settings: dict
    elapsed_s: float
    notes: List[str] = field(default_factory=list)
    cvs: Optional[np.ndarray] = None           # [n_strips, n_cmps, n_t] constant-velocity stacks
    cvs_t: Optional[np.ndarray] = None
    cvs_vels: Optional[np.ndarray] = None
    cvs_cmps: List[int] = field(default_factory=list)


def condition(g: Gather, s: AnalysisSettings) -> Gather:
    """Bandpass then equalise a COPY; samples that are zero in the input (data mute) stay zero."""
    out = copy.copy(g)
    d, zero = g.data, (g.data == 0)
    if s.bandpass:
        d = ormsby_bandpass(d, g.dt_ms, s.f1, s.f2, s.f3, s.f4)
    if s.eq_ms > 0:
        d = equalize(d, g.dt_ms, s.eq_ms)
    d = np.array(d, dtype=np.float32, copy=True)
    d[zero] = 0.0
    out.data = d
    if s.max_offset and s.max_offset > 0:                  # traces beyond the offset limit are excluded, not deleted
        out.dead = g.dead | (g.offsets > s.max_offset)
    return out


def run_analysis(sf: SegyFile, idx: TraceIndex, cmp_id: int, s: AnalysisSettings,
                 progress: Optional[Callable[[int, int], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None) -> AnalysisResult:
    t_begin = time.perf_counter()
    centre = extract_gather(sf, idx, cmp_id)
    rec_end = centre.t_start_ms + (centre.ns - 1) * centre.dt_ms
    t_end = min(s.t0_end, rec_end - 2 * centre.dt_ms)
    notes: List[str] = []
    if t_end < s.t0_end:
        notes.append(f"t0 end limited to {t_end:.0f} ms (record length).")
    cvs_on = s.cvs_enabled and s.cvs_strips >= 2
    share = 0.8 if cvs_on else 1.0

    def spec_progress(i, n):
        if progress is not None:
            progress(int(100 * share * i / max(n, 1)), 100)

    ids = select_spectrum_cmps(idx.cmp_ids, cmp_id, s.n_cmps)
    cache = {cmp_id: condition(centre, s)}

    def cond(c):
        if c not in cache:
            cache[c] = condition(extract_gather(sf, idx, c), s)
        return cache[c]

    if s.method == "stackpower":
        gathers = [cond(c) for c in ids]
        p = StackPowerParams(vmin=s.vmin, vmax=s.vmax, dv=s.dv, t0_start_ms=s.t0_start, t0_end_ms=t_end,
                             line_interval_ms=s.line_interval, group_size=3, stretch_mute_pct=s.stretch,
                             filter2d=(3, 3) if s.filter2d else None)
        sp = compute_stack_power(gathers, centre.dt_ms, p, t_start_ms=centre.t_start_ms, progress=spec_progress,
                                 cancelled=cancelled)
        disp = sp.row_normalised("mean") if s.row_norm else sp.amplitude
        t0, vel, used = sp.t0_ms, sp.velocities, list(ids)
        if len(ids) < s.n_cmps:
            notes.append(f"Only {len(ids)} CMP(s) available around {cmp_id} (asked for {s.n_cmps}).")
    elif s.method == "semblance":
        g = cond(cmp_id)
        p = AnalysisParams(vmin=s.vmin, vmax=s.vmax, dv=s.dv, t0_start_ms=s.t0_start, t0_end_ms=t_end,
                           window_ms=s.window_ms, t0_step_ms=s.line_interval, stretch_mute_pct=s.sem_stretch)
        sem = compute_semblance(g.data, g.offsets, g.dt_ms, p, t_start_ms=g.t_start_ms, live=~g.dead, cmp_id=cmp_id,
                                progress=spec_progress, cancelled=cancelled)
        v = sem.values
        disp = v / np.maximum(v.max(axis=1, keepdims=True), 1e-12) if s.row_norm else v
        t0, vel, used = sem.t0_ms, sem.velocities, [cmp_id]
    else:
        raise ValueError(f"unknown method {s.method!r}")

    cvs = cvs_t = cvs_v = None
    cvs_ids: List[int] = []
    if cvs_on:
        cvs_ids = select_neighbour_cmps(idx.cmp_ids, cmp_id, s.cvs_cmps)
        v_lo = s.cvs_vmin if s.cvs_vmin > 0 else s.vmin
        v_hi = s.cvs_vmax if s.cvs_vmax > v_lo else s.vmax
        cvs_v = np.linspace(v_lo, v_hi, int(s.cvs_strips))
        cvs_t = np.arange(max(s.t0_start, centre.t_start_ms + 2 * centre.dt_ms), t_end + 1e-6, s.cvs_dt_ms)

        def cvs_progress(i, n):
            if progress is not None:
                progress(int(100 * (share + (1 - share) * i / max(n, 1))), 100)

        cvs = constant_velocity_stacks([cond(c) for c in cvs_ids], centre.dt_ms, cvs_v, cvs_t,
                                       stretch_pct=s.stretch, t_start_ms=centre.t_start_ms,
                                       progress=cvs_progress, cancelled=cancelled)
    return AnalysisResult(cmp_id=cmp_id, method=s.method, display=np.asarray(disp, np.float32), t0_ms=t0,
                          velocities=vel, cmps_used=list(used), gather=centre, settings=s.to_dict(),
                          elapsed_s=time.perf_counter() - t_begin, notes=notes, cvs=cvs, cvs_t=cvs_t,
                          cvs_vels=cvs_v, cvs_cmps=cvs_ids)
