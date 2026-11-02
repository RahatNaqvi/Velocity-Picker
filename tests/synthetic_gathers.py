"""Synthetic CMP gathers with analytic hyperbolic events (Ricker wavelets evaluated exactly)."""
import numpy as np


def ricker(t_ms, f_hz):
    a = (np.pi * f_hz * t_ms / 1000.0) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_gather(events, offsets, dt_ms=4.0, ns=1000, f_hz=30.0, noise=0.0, seed=0):
    """events: list of (t0_ms, v_mps, amplitude). Returns data [ntr, ns]."""
    rng = np.random.default_rng(seed)
    t = np.arange(ns) * dt_ms
    d = np.zeros((len(offsets), ns))
    for i, x in enumerate(offsets):
        for t0, v, amp in events:
            tau = np.sqrt(t0 ** 2 + (1000.0 * x / v) ** 2)
            d[i] += amp * ricker(t - tau, f_hz)
    if noise:
        d += rng.normal(0, noise, d.shape)
    return d.astype(np.float32)
