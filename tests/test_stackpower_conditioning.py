import numpy as np
import pytest

from synthetic_gathers import make_gather, ricker
from velpick.gather_manager import Gather, build_supergather
from velpick.velocity_analysis import (StackPowerParams, compute_stack_power, equalize, ormsby_bandpass,
                                       select_spectrum_cmps, top_mute, AnalysisCancelled)

OFF = np.arange(200.0, 3200.0, 100.0)
DT, NS = 4.0, 1000


def G(cmp_id, data, off=OFF, dead=None):
    n = data.shape[0]
    return Gather(cmp_id=cmp_id, data=data, offsets=np.asarray(off, float), dt_ms=DT, t_start_ms=0.0,
                  trace_positions=np.arange(n) + 1000 * cmp_id,
                  dead=np.zeros(n, bool) if dead is None else dead, n_nonfinite_samples=0,
                  offset_source="header", contributing_cmps=(cmp_id,))


# ------------------------------------------------------------------ conditioning
def test_bandpass_passes_inband_and_rejects_outband():
    t = np.arange(2000) * 4.0 / 1000.0
    x = np.stack([np.sin(2 * np.pi * 25 * t), np.sin(2 * np.pi * 3 * t), np.sin(2 * np.pi * 100 * t)])
    y = ormsby_bandpass(x, 4.0, 7, 11, 45, 55)
    core = slice(300, 1700)
    rms = lambda a: np.sqrt((a[core] ** 2).mean())
    assert rms(y[0]) == pytest.approx(rms(x[0]), rel=0.02)
    assert rms(y[1]) < 0.02 * rms(x[1]) and rms(y[2]) < 0.02 * rms(x[2])
    with pytest.raises(ValueError):
        ormsby_bandpass(x, 4.0, 11, 7, 45, 55)


def test_bandpass_is_zero_phase_and_does_not_modify_input():
    d = make_gather([(1000, 2500, 1.0)], OFF[:3], DT, NS)
    keep = d.copy()
    y = ormsby_bandpass(d, DT, 7, 11, 45, 55)
    assert np.array_equal(d, keep)
    assert np.all(np.argmax(np.abs(y), axis=1) == np.argmax(np.abs(d), axis=1))   # no time shift of the peak


def test_equalize_flattens_amplitude_trend_and_keeps_zero_traces_zero():
    rng = np.random.default_rng(0)
    ramp = np.linspace(1, 50, NS)
    d = (rng.normal(size=(4, NS)) * ramp).astype(np.float32)
    d[3] = 0.0
    y = equalize(d, DT, 300)
    early, late = y[:3, 100:300].std(), y[:3, 700:900].std()
    assert 0.7 < early / late < 1.4
    assert np.all(y[3] == 0)
    assert np.array_equal(equalize(d, DT, 0), d)


def test_top_mute_rule_and_taper():
    d = np.ones((3, 500), np.float32)
    x = np.array([0.0, 1500.0, 9000.0])
    y = top_mute(d, x, DT, [(0, 100), (1500, 1000), (3500, 2000)])
    first = (y != 0).argmax(axis=1) * DT
    assert first[0] == 100 and first[1] == 1000
    assert y[2].sum() == 0                    # extrapolated mute (~4600 ms) is beyond the record (1996 ms)
    yt = top_mute(d[:2], x[:2], DT, [(0, 100), (1500, 1000)], taper_ms=100)
    assert yt[1, int(1000 / DT)] == 0 and 0 < yt[1, int(1050 / DT)] < 1 and yt[1, int(1200 / DT)] == 1


# ------------------------------------------------------------------ CMP selection / supergather
def test_select_spectrum_cmps_follows_manual_rules():
    avail = list(range(200, 226))
    assert select_spectrum_cmps(avail, 210, 1) == [210]
    assert select_spectrum_cmps(avail, 210, 3) == [209, 210, 211]
    assert select_spectrum_cmps(avail, 210, 21, interval=5) == [209, 210, 211]      # capped by interval, multiple of 3
    assert select_spectrum_cmps(avail, 210, 21) == list(range(200, 221))
    assert select_spectrum_cmps([210, 220, 230], 220, 3) == [210, 220, 230]          # non-contiguous ids honoured
    assert select_spectrum_cmps([210, 211], 210, 3) == [210, 211]                    # fewer available than asked


def test_supergather_merges_sorts_and_records_contributors():
    g1 = G(10, make_gather([(1000, 2500, 1.0)], OFF, DT, NS), OFF)
    g2 = G(11, make_gather([(1000, 2500, 1.0)], OFF + 50, DT, NS), OFF + 50)
    sg = build_supergather([g2, g1])
    assert sg.ntr == 60 and sg.contributing_cmps == (10, 11)
    assert np.all(np.diff(sg.offsets) >= 0)
    assert build_supergather([g1, g2], max_traces=10).ntr == 10
    assert build_supergather([g1, g2], max_traces=10).offsets.max() <= np.sort(sg.offsets)[9]
    bad = G(12, np.zeros((5, NS + 10), np.float32))
    with pytest.raises(ValueError):
        build_supergather([g1, bad])


# ------------------------------------------------------------------ stack power
def P(**kw):
    base = dict(vmin=1500, vmax=4000, dv=50, t0_start_ms=300, t0_end_ms=2500, filter2d=None)
    base.update(kw)
    return StackPowerParams(**base)


def three_cmps(events, noise=0.0, dip_ms=0.0):
    gs = []
    for k in range(3):
        ev = [(t0 + dip_ms * (k - 1), v, a) for t0, v, a in events]
        gs.append(G(100 + k, make_gather(ev, OFF, DT, NS, noise=noise, seed=k)))
    return gs


def peak_v(spec, t0):
    r = int(np.argmin(np.abs(spec.t0_ms - t0)))
    return float(spec.velocities[int(np.argmax(spec.amplitude[r]))])


def test_stack_power_peaks_at_true_velocities():
    ev = [(600, 2200, 1.0), (1400, 2900, 1.0), (2200, 3600, 1.0)]
    s = compute_stack_power(three_cmps(ev), DT, P())
    assert s.cmp_groups == [(100, 101, 102)] and s.n_traces_used == 90
    for t0, v in ((600, 2200), (1400, 2900), (2200, 3600)):
        assert abs(peak_v(s, t0) - v) <= 100, (t0, peak_v(s, t0), v)
    assert s.amplitude.min() >= 0 and np.allclose(s.power, s.amplitude ** 2)


def test_stack_power_with_noise_and_row_normalisation():
    s = compute_stack_power(three_cmps([(1200, 2800, 1.0)], noise=0.4), DT, P())
    assert abs(peak_v(s, 1200) - 2800) <= 100
    rn = s.row_normalised()
    assert np.allclose(rn.mean(axis=1), 1.0, atol=1e-5)


def test_two_step_stack_survives_dip_better_than_one_big_coherent_stack():
    """6 CMPs with 14 ms/CMP dip: two groups of 3 (|sum| added) vs one group of 6 (sum then |.|)."""
    ev = [(1200, 2800, 1.0)]
    gs = []
    for k in range(6):
        gs.append(G(100 + k, make_gather([(1200 + 14.0 * k, 2800, 1.0)], OFF, DT, NS)))
    def peak_amp(group_size):
        s = compute_stack_power(gs, DT, P(group_size=group_size))
        r = int(np.argmin(np.abs(s.t0_ms - 1200 + 0)))
        return s.amplitude[r - 2:r + 6].max()
    assert peak_amp(3) > 1.3 * peak_amp(6)


def test_dead_traces_are_excluded_and_input_untouched():
    gs = three_cmps([(1200, 2800, 1.0)])
    dead = np.zeros(len(OFF), bool); dead[4] = True
    gs[1].data[4] = 0.0; gs[1].dead = dead
    before = gs[0].data.copy()
    s = compute_stack_power(gs, DT, P())
    assert s.n_traces_used == 3 * len(OFF) - 1
    assert np.array_equal(gs[0].data, before)


def test_parameter_rules_progress_and_cancel():
    gs = three_cmps([(1200, 2800, 1.0)])
    with pytest.raises(ValueError, match="6 x"):
        compute_stack_power(gs, DT, P(line_interval_ms=20))          # 20 <= 6*4
    with pytest.raises(ValueError):
        compute_stack_power(gs, DT, P(vmax=1000))
    calls = []
    compute_stack_power(gs, DT, P(dv=500), progress=lambda i, n: calls.append((i, n)))
    assert calls[-1][0] == calls[-1][1]
    k = {"n": 0}
    def cancel():
        k["n"] += 1
        return k["n"] > 2
    with pytest.raises(AnalysisCancelled):
        compute_stack_power(gs, DT, P(dv=100), cancelled=cancel)


def test_filter2d_smooths_and_changes_result():
    gs = three_cmps([(1200, 2800, 1.0)], noise=0.3)
    a = compute_stack_power(gs, DT, P()).amplitude
    b = compute_stack_power(gs, DT, P(filter2d=(3, 3))).amplitude
    assert not np.allclose(a, b) and b.std() < a.std()


# ------------------------------------------------------------------ constant velocity stacks
def test_cvs_event_is_strongest_in_strip_nearest_true_velocity():
    from velpick.velocity_analysis import constant_velocity_stacks
    gs = three_cmps([(1000, 2800, 1.0)])
    vels = np.arange(2000, 3700, 200.0)
    t = np.arange(400.0, 1600.0, 8.0)
    cvs = constant_velocity_stacks(gs, DT, vels, t, stretch_pct=100.0)
    assert cvs.shape == (vels.size, 3, t.size) and cvs.dtype == np.float32
    r = int(np.argmin(np.abs(t - 1000)))
    peak = np.abs(cvs[:, :, r - 2:r + 3]).max(axis=(1, 2))
    assert abs(vels[int(np.argmax(peak))] - 2800) <= 200
    assert peak.max() > 2 * np.median(peak)


def test_cvs_cancel_and_dead_gather():
    from velpick.velocity_analysis import constant_velocity_stacks
    gs = three_cmps([(1000, 2800, 1.0)])
    gs[0].dead[:] = True                                  # a gather with no live traces gives a zero stack trace
    out = constant_velocity_stacks(gs, DT, [2800.0], np.arange(400.0, 1600.0, 8.0), 100.0)
    assert np.all(out[0, 0] == 0) and np.abs(out[0, 1]).max() > 0
    n = {"c": 0}
    def cancel():
        n["c"] += 1
        return n["c"] > 1
    with pytest.raises(AnalysisCancelled):
        constant_velocity_stacks(gs, DT, [2000.0, 2200.0, 2400.0], np.arange(400.0, 800.0, 8.0), 100.0, cancelled=cancel)


def test_select_neighbour_cmps_exact_count_symmetric_and_non_contiguous():
    from velpick.velocity_analysis import select_neighbour_cmps
    assert select_neighbour_cmps(range(200, 226), 210, 5) == [208, 209, 210, 211, 212]            # exact count (no multiples of 3)
    assert select_neighbour_cmps(range(210, 216), 210, 25) == [210, 211, 212, 213, 214, 215]       # fewer available than asked
    assert select_neighbour_cmps([210, 220, 230], 220, 2) == [210, 220]                            # ties go to the lower id
    assert select_neighbour_cmps([5, 6, 7], 6, 0) == [6]
