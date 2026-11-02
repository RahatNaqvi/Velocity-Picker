"""Compute and plot semblance for one CMP (read-only on the SEG-Y).

Example:
  python scripts/run_semblance.py CMP700-750.sgy --cmp 700 --vmin 1500 --vmax 6000 --dv 50 \
      --t0-start 100 --t0-end 4000 --window 40 --stretch 30 --offset-max 4000 --out cmp700.png
Needs matplotlib (pip install matplotlib). All values are in ms, m and m/s.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from velpick.data_io import SegyFile
from velpick.gather_manager import extract_gather, semblance_from_gather
from velpick.geometry import TraceIndex
from velpick.velocity_analysis import AnalysisParams, validate_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segy")
    ap.add_argument("--cmp", type=int, required=True)
    ap.add_argument("--vmin", type=float, required=True)
    ap.add_argument("--vmax", type=float, required=True)
    ap.add_argument("--dv", type=float, required=True)
    ap.add_argument("--t0-start", type=float, required=True)
    ap.add_argument("--t0-end", type=float, required=True)
    ap.add_argument("--window", type=float, required=True)
    ap.add_argument("--stretch", type=float, default=30.0)
    ap.add_argument("--offset-min", type=float)
    ap.add_argument("--offset-max", type=float)
    ap.add_argument("--out", default="semblance.png")
    a = ap.parse_args()

    sf = SegyFile(a.segy)
    idx = TraceIndex.build(sf)
    g = extract_gather(sf, idx, a.cmp)
    p = AnalysisParams(vmin=a.vmin, vmax=a.vmax, dv=a.dv, t0_start_ms=a.t0_start, t0_end_ms=a.t0_end,
                       window_ms=a.window, stretch_mute_pct=a.stretch,
                       offset_min=a.offset_min, offset_max=a.offset_max)
    for i in validate_params(p, g.dt_ms, g.ns, g.offsets, g.t_start_ms):
        print(f"[{i.severity.upper()}] {i.code}: {i.message}")
    s = semblance_from_gather(g, p)
    print(f"CMP {a.cmp}: {s.n_traces_used} traces used, {s.elapsed_s:.2f} s, "
          f"semblance max {s.values.max():.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    clip = np.percentile(np.abs(g.data), 98)
    ax[0].imshow(g.data.T, aspect="auto", cmap="gray_r", vmin=-clip, vmax=clip,
                 extent=[g.offsets[0], g.offsets[-1], g.times_ms[-1], 0])
    ax[0].set_xlabel("|offset| (m)"); ax[0].set_ylabel("time (ms)")
    ax[0].set_title(f"CMP {a.cmp} gather")
    im = ax[1].imshow(s.values, aspect="auto", cmap="jet", vmin=0,
                      vmax=np.percentile(s.values, 99.5),
                      extent=[s.velocities[0], s.velocities[-1], s.t0_ms[-1], s.t0_ms[0]])
    ax[1].set_xlabel("trial velocity (m/s)"); ax[1].set_ylabel("t0 (ms)")
    ax[1].set_title("Semblance"); plt.colorbar(im, ax=ax[1])
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
