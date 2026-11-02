import numpy as np
import pytest

from synthetic_gathers import make_gather
from velpick.velocity_analysis import (AnalysisCancelled, AnalysisParams, ParamValidationError,
                                       compute_semblance, estimate_cost, nmo_correct, validate_params)

OFF = np.arange(100.0, 3100.0, 100.0)          # 30 traces, 100..3000 m
DT, NS = 4.0, 1000                             # 4 ms, 3996 ms


def P(**kw):
    base = dict(vmin=1500, vmax=4000, dv=25, t0_start_ms=200, t0_end_ms=3000, window_ms=40)
    base.update(kw)
    return AnalysisParams(**base)


# ------------------------------------------------------------------ NMO
def test_nmo_flattens_event_at_true_velocity():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    t0 = np.array([1000.0])
    out, valid = nmo_correct(d, OFF, DT, t0, 2500.0, 100.0)
    assert valid.all()
    # corrected amplitude at t0 equals the wavelet peak on every trace (linear interp tolerance)
    # linear interpolation of a 30 Hz wavelet at 4 ms loses up to ~10 % of the peak
    assert out[:, 0].min() > 0.85
    wrong, _ = nmo_correct(d, OFF, DT, t0, 3500.0, 100.0)
    assert out[:, 0].sum() > wrong[:, 0].sum() + 5


def test_nmo_velocity_function_and_input_untouched():
    d = make_gather([(500, 2000, 1.0), (1500, 3000, 1.0)], OFF, DT, NS)
    keep = d.copy()
    t0 = np.array([500.0, 1500.0])
    out, valid = nmo_correct(d, OFF, DT, t0, np.array([2000.0, 3000.0]), 100.0)
    assert valid[:, 0].sum() >= 15 and valid[:, 1].sum() >= 25      # far offsets are stretch-muted
    assert out[:, 0][valid[:, 0]].min() > 0.85 and out[:, 1][valid[:, 1]].min() > 0.85
    assert np.all(out[~valid] == 0)
    assert np.array_equal(d, keep)


def test_nmo_stretch_mute_and_record_end():
    d = np.ones((3, 300), dtype=np.float32)               # record 0..1196 ms
    x = np.array([0.0, 1000.0, 3000.0])
    out, valid = nmo_correct(d, x, 4.0, np.array([100.0]), 1500.0, 30.0)
    assert valid[0, 0] and not valid[1, 0] and not valid[2, 0]   # tau 674 ms: stretch 574 % > 30 %
    assert out[1, 0] == 0.0
    out2, valid2 = nmo_correct(d, x, 4.0, np.array([100.0]), 1500.0, 1e6)
    assert valid2[1, 0] and not valid2[2, 0]                     # tau 2002 ms is beyond the record
    with pytest.raises(ValueError):
        nmo_correct(d, x, 4.0, np.array([0.0]), 1500.0, 30.0)
    with pytest.raises(ValueError):
        nmo_correct(d, x, 4.0, np.array([100.0]), -1.0, 30.0)


def test_nmo_negative_offsets_treated_as_abs():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    a, _ = nmo_correct(d, OFF, DT, np.array([1000.0]), 2500.0, 100.0)
    b, _ = nmo_correct(d, -OFF, DT, np.array([1000.0]), 2500.0, 100.0)
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ semblance known answers
def test_single_event_peak_at_true_t0_and_v():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    p = P(dv=25)
    s = compute_semblance(d, OFF, DT, p)
    assert s.values.shape == (s.t0_ms.size, s.velocities.size)
    assert s.values.min() >= 0 and s.values.max() <= 1.0 + 1e-6
    v, val = s.peak_velocity(1000.0)
    assert abs(v - 2500) <= p.dv and val > 0.6
    # the global maximum of the panel is at the event
    r, k = np.unravel_index(np.argmax(s.values), s.values.shape)
    # in time the maximum is only defined to about one gate (flat event => plateau of ~1)
    assert abs(s.t0_ms[r] - 1000) <= p.window_ms and abs(s.velocities[k] - 2500) <= p.dv


def test_three_events_increasing_velocity():
    ev = [(500, 2000, 1.0), (1500, 2800, 1.0), (2500, 3500, 1.0)]
    d = make_gather(ev, OFF, DT, NS)
    s = compute_semblance(d, OFF, DT, P(dv=25, window_ms=40))
    for t0, v_true, _ in ev:
        v, val = s.peak_velocity(t0)
        assert abs(v - v_true) <= 50, (t0, v, v_true)
        assert val > 0.5


def test_velocity_resolution_degrades_with_depth_as_expected():
    """With a fixed offset range (stretch mute off) the velocity peak widens with t0."""
    ev = [(600, 2200, 1.0), (2600, 2200, 1.0)]
    s = compute_semblance(make_gather(ev, OFF, DT, NS), OFF, DT, P(dv=25, stretch_mute_pct=1000))

    def width(t0):
        r = int(np.argmin(np.abs(s.t0_ms - t0)))
        row = s.values[r]
        return (row > 0.5 * row.max()).sum()
    assert width(2600) > width(600)


def test_coherent_beats_noise_and_noise_stays_low():
    noise = make_gather([], OFF, DT, NS, noise=1.0, seed=3)
    s_n = compute_semblance(noise, OFF, DT, P())
    assert s_n.values.mean() < 0.15
    ev = make_gather([(1000, 2500, 1.0)], OFF, DT, NS, noise=0.3, seed=4)
    v, val = compute_semblance(ev, OFF, DT, P()).peak_velocity(1000)
    assert abs(v - 2500) <= 50 and val > s_n.values.mean() * 3


def test_identical_traces_give_semblance_one():
    d = np.tile(np.random.default_rng(0).normal(size=(1, NS)).astype(np.float32), (10, 1))
    x = np.zeros(10)
    x[:] = 1e-9   # zero moveout; keep >0 offsets for the validator
    s = compute_semblance(d, x, DT, P(vmin=2000, vmax=2100, dv=50))
    assert np.allclose(s.values, 1.0, atol=1e-5)


def test_polarity_reversal_reduces_semblance():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    d2 = d.copy(); d2[::2] *= -1
    a = compute_semblance(d, OFF, DT, P()).peak_velocity(1000)[1]
    b = compute_semblance(d2, OFF, DT, P()).peak_velocity(1000)[1]
    assert b < 0.3 * a


def test_two_traces_never_give_spurious_one():
    d = make_gather([(1000, 2500, 1.0)], OFF[:2], DT, NS)
    with pytest.raises(ParamValidationError, match="TOO_FEW_TRACES"):
        compute_semblance(d, OFF[:2], DT, P())
    s = compute_semblance(d, OFF[:2], DT, P(min_live_traces=2))
    assert s.values.max() <= 1.0


def test_dead_trace_exclusion_equals_subset():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    d[7] = 0.0
    live = np.ones(len(OFF), bool); live[7] = False
    a = compute_semblance(d, OFF, DT, P(), live=live)
    b = compute_semblance(np.delete(d, 7, 0), np.delete(OFF, 7), DT, P())
    assert np.array_equal(a.values, b.values) and a.n_traces_used == 29


def test_offset_limits_select_traces():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    s = compute_semblance(d, OFF, DT, P(offset_min=500, offset_max=2000))
    assert s.n_traces_used == 16


def test_smoothing_is_explicit_and_changes_result():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    a = compute_semblance(d, OFF, DT, P())
    b = compute_semblance(d, OFF, DT, P(smooth_v_samples=5, smooth_t_ms=16))
    assert not np.array_equal(a.values, b.values)
    assert abs(b.peak_velocity(1000)[0] - 2500) <= 50


def test_t0_step_coarser_than_sampling():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    s = compute_semblance(d, OFF, DT, P(t0_step_ms=8.0))
    assert np.allclose(np.diff(s.t0_ms), 8.0) and abs(s.peak_velocity(1000)[0] - 2500) <= 25


# ------------------------------------------------------------------ independent slow reference
def brute_force(data, offsets, dt, p, t_start=0.0):
    ns = data.shape[1]
    t0s, vs = p.t0_axis(dt), p.velocities()
    step = p.step_ms(dt)
    h = int(round(p.window_ms / (2 * step)))
    S = np.zeros((t0s.size, vs.size))
    for kv, v in enumerate(vs):
        amp = np.zeros((len(offsets), t0s.size)); ok = np.zeros_like(amp, bool)
        for i, x in enumerate(offsets):
            for j, t0 in enumerate(t0s):
                tau = np.sqrt(t0 ** 2 + (1000 * x / v) ** 2)
                pos = (tau - t_start) / dt
                if pos < 0 or pos > ns - 1 or (tau / t0 - 1) > p.stretch_mute_pct / 100:
                    continue
                i0 = min(int(np.floor(pos)), ns - 2)
                f = pos - i0
                amp[i, j] = float(data[i, i0]) * (1 - f) + float(data[i, i0 + 1]) * f
                ok[i, j] = True
        for j in range(t0s.size):
            lo, hi = max(0, j - h), min(t0s.size, j + h + 1)
            num = sum(amp[:, jj].sum() ** 2 for jj in range(lo, hi))
            den = sum(ok[:, jj].sum() * (amp[:, jj] ** 2).sum() for jj in range(lo, hi))
            n_c = ok[:, j].sum()
            S[j, kv] = 0.0 if (den == 0 or n_c < p.min_live_traces) else min(max(num / den, 0), 1)
    return S


def test_vectorised_engine_matches_brute_force_reference():
    off = np.arange(100.0, 1600.0, 150.0)
    d = make_gather([(400, 2200, 1.0), (900, 3000, 0.7)], off, DT, 400, noise=0.05, seed=9)
    p = P(vmin=1800, vmax=3600, dv=100, t0_start_ms=100, t0_end_ms=1200, window_ms=32, t0_step_ms=8)
    ref = brute_force(d, off, DT, p)
    got = compute_semblance(d, off, DT, p).values
    assert np.allclose(got, ref, atol=2e-4)


# ------------------------------------------------------------------ validation, cost, control
def codes(issues, sev=None):
    return {i.code for i in issues if sev is None or i.severity == sev}


def test_param_validation_errors():
    kw = dict(dt_ms=DT, ns=NS, offsets=OFF)
    assert codes(validate_params(P(dv=0), **kw), "error") >= {"DV"}
    assert "VRANGE" in codes(validate_params(P(vmax=1500), **kw), "error")
    assert "T0_START" in codes(validate_params(P(t0_start_ms=0), **kw), "error")
    assert "T0_OUTSIDE" in codes(validate_params(P(t0_end_ms=5000), **kw), "error")
    assert "T0_RANGE" in codes(validate_params(P(t0_end_ms=100), **kw), "error")
    assert "WINDOW" in codes(validate_params(P(window_ms=0), **kw), "error")
    assert "STRETCH" in codes(validate_params(P(stretch_mute_pct=0), **kw), "error")
    assert "MINTR" in codes(validate_params(P(min_live_traces=1), **kw), "error")
    assert "SMOOTH_V" in codes(validate_params(P(smooth_v_samples=4), **kw), "error")
    assert "NO_OFFSET" in codes(validate_params(P(), DT, NS, np.zeros(10)), "error")
    assert "METHOD" in codes(validate_params(P(method="foo"), **kw), "error")
    assert not codes(validate_params(P(), **kw), "error")


def test_param_validation_warnings_do_not_block():
    kw = dict(dt_ms=DT, ns=NS, offsets=OFF)
    assert "WINDOW_SHORT" in codes(validate_params(P(window_ms=8), **kw), "warning")
    assert "STRETCH_LARGE" in codes(validate_params(P(stretch_mute_pct=150), **kw), "warning")
    assert "DV_COARSE" in codes(validate_params(P(dv=500, window_ms=20), **kw), "warning")
    short = np.array([100.0, 200.0, 300.0, 400.0])
    assert "LOW_RESOLUTION" in codes(validate_params(P(vmax=6000), DT, NS, short), "warning")
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    compute_semblance(d, OFF, DT, P(window_ms=8))      # warning only: runs


def test_cost_estimate_counts_and_axes():
    p = P(vmin=1500, vmax=4000, dv=50, t0_start_ms=200, t0_end_ms=3000)
    c = estimate_cost(p, DT, 30)
    assert c.n_velocities == 51 and c.n_t0 == 701 and c.n_interpolations == 51 * 701 * 30
    assert c.result_bytes == 51 * 701 * 4
    assert p.velocities()[0] == 1500 and p.velocities()[-1] == 4000


def test_params_hash_stable_and_sensitive():
    assert P().hash() == P().hash()
    assert P().hash() != P(dv=50).hash()


def test_progress_and_cancellation():
    d = make_gather([(1000, 2500, 1.0)], OFF, DT, NS)
    calls = []
    s = compute_semblance(d, OFF, DT, P(dv=100), progress=lambda i, n: calls.append((i, n)))
    assert calls[-1] == (s.velocities.size, s.velocities.size)
    n = {"c": 0}

    def cancel():
        n["c"] += 1
        return n["c"] > 3
    with pytest.raises(AnalysisCancelled):
        compute_semblance(d, OFF, DT, P(dv=100), cancelled=cancel)


def test_stretch_mute_narrows_aperture_so_shallow_peaks_broaden():
    """Documented behaviour: the default mute removes far offsets at shallow times."""
    ev = [(600, 2200, 1.0)]
    d = make_gather(ev, OFF, DT, NS)

    def width(pct):
        s = compute_semblance(d, OFF, DT, P(dv=25, stretch_mute_pct=pct))
        row = s.values[int(np.argmin(np.abs(s.t0_ms - 600)))]
        return (row > 0.5 * row.max()).sum()
    assert width(30) > width(1000)
