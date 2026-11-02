"""GeoEast-style stack-power velocity spectrum for chosen CMPs (read-only on the SEG-Y).

Example (CMPs 210-212 form one group of 3):
  python scripts/run_stackpower.py CMP-For-Vel210-215.sgy --cmps 210 211 212 \
      --vmin 1500 --vmax 6000 --dv 40 --t0-start 100 --t0-end 2500 --picks examples/picks_cmp210.txt --out cmp210.png

Units: ms, m, m/s. Conditioning (applied to copies): Ormsby bandpass, then equalisation; samples that are
zero in the input (the data's own top mute) stay zero. Needs matplotlib. See docs/ENGINE.md for assumptions.
"""
import argparse
import copy
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from velpick.data_io import SegyFile
from velpick.gather_manager import extract_gather
from velpick.geometry import TraceIndex
from velpick.velocity_analysis import StackPowerParams, compute_stack_power, equalize, ormsby_bandpass


def read_picks(path):
    """Lines containing two numbers (time ms, velocity m/s), separated by spaces, tabs, commas or semicolons."""
    out = []
    for line in open(path, errors="replace"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if len(nums) >= 2 and re.fullmatch(r"[\s,;\t\d.\-]+", line.strip()):
            out.append((float(nums[0]), float(nums[1])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segy")
    ap.add_argument("--cmps", type=int, nargs="+", required=True, help="CMP ids to use (split into groups of --group-size)")
    ap.add_argument("--vmin", type=float, required=True)
    ap.add_argument("--vmax", type=float, required=True)
    ap.add_argument("--dv", type=float, required=True)
    ap.add_argument("--t0-start", type=float, required=True)
    ap.add_argument("--t0-end", type=float, required=True)
    ap.add_argument("--line-interval", type=float, default=25.0, help="spectrum line spacing, ms (must exceed 6 x sample interval)")
    ap.add_argument("--group-size", type=int, default=3)
    ap.add_argument("--stretch", type=float, default=50.0, help="NMO stretch mute %% (assumption, not from the manual)")
    ap.add_argument("--bandpass", type=float, nargs=4, default=[7, 11, 45, 55], metavar=("F1", "F2", "F3", "F4"))
    ap.add_argument("--no-bandpass", action="store_true")
    ap.add_argument("--eq", type=float, default=300.0, help="equalisation window ms (0 = off)")
    ap.add_argument("--no-filter2d", action="store_true")
    ap.add_argument("--picks", help="text file with 'time velocity' lines to overlay")
    ap.add_argument("--display", choices=["row-mean", "global"], default="row-mean")
    ap.add_argument("--out", default="stackpower.png")
    a = ap.parse_args()

    sf = SegyFile(a.segy)
    idx = TraceIndex.build(sf)
    missing = idx.missing(a.cmps)
    if missing:
        raise SystemExit(f"CMPs not in file: {missing}. Available ids: {idx.cmp_ids[:20].tolist()}")
    gathers = []
    for c in a.cmps:
        g = extract_gather(sf, idx, c)
        d, zero = g.data, (g.data == 0)
        if not a.no_bandpass:
            d = ormsby_bandpass(d, g.dt_ms, *a.bandpass)
        if a.eq > 0:
            d = equalize(d, g.dt_ms, a.eq)
        d = np.array(d, copy=True)
        d[zero] = 0.0
        g = copy.copy(g)
        g.data = d
        gathers.append(g)
    p = StackPowerParams(vmin=a.vmin, vmax=a.vmax, dv=a.dv, t0_start_ms=a.t0_start, t0_end_ms=a.t0_end,
                         line_interval_ms=a.line_interval, group_size=a.group_size, stretch_mute_pct=a.stretch,
                         filter2d=None if a.no_filter2d else (3, 3))
    s = compute_stack_power(gathers, sf.dt_ms, p)
    print(f"CMP groups {s.cmp_groups}: {s.n_traces_used} traces, {s.elapsed_s:.1f} s, "
          f"{s.velocities.size} velocities x {s.t0_ms.size} lines")
    img = s.row_normalised("mean") if a.display == "row-mean" else s.amplitude
    fig, ax = plt.subplots(figsize=(6, 9))
    half = a.line_interval / 2
    im = ax.imshow(img, aspect="auto", cmap="jet", vmin=0, vmax=np.percentile(img, 99.5),
                   extent=[s.velocities[0], s.velocities[-1], s.t0_ms[-1] + half, s.t0_ms[0] - half])
    if a.picks:
        pk = read_picks(a.picks)
        if len(pk):
            ax.plot(pk[:, 1], pk[:, 0], "w-o", ms=4, lw=1.4, label="picks")
            ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(s.velocities[0], s.velocities[-1])        # keep the spectrum range even if picks go beyond it
    ax.set_ylim(s.t0_ms[-1] + half, s.t0_ms[0] - half)
    ax.set_xlabel("velocity (m/s)")
    ax.set_ylabel("t0 (ms)")
    ax.set_title(f"stack power, CMPs {a.cmps}\n{'row-mean normalised' if a.display == 'row-mean' else 'global scale'}", fontsize=9)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
