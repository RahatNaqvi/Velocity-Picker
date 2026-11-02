"""Read-only inspection: python -m velpick.data_io.inspect_cli FILE.sgy [--text] [--cache index.npz]"""
import argparse

from velpick.data_io.segy import SegyFile
from velpick.geometry.trace_index import TraceIndex


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a SEG-Y file (never modifies it).")
    ap.add_argument("segy")
    ap.add_argument("--text", action="store_true", help="print the textual header")
    ap.add_argument("--cache", help="optional index cache (.npz) to load/save")
    a = ap.parse_args(argv)
    sf = SegyFile(a.segy)
    for k, v in sf.info().items():
        print(f"{k}: {v}")
    if a.text:
        print("\n".join(sf.text_header))
    idx = TraceIndex.load_or_build(sf, a.cache) if a.cache else TraceIndex.build(sf)
    print(f"\nCMPs: {idx.cmp_ids.size}  ids {idx.cmp_ids[:10].tolist()}"
          f"{' ...' if idx.cmp_ids.size > 10 else ''}")
    print(f"{'cmp':>8} {'fold':>6} {'off_min':>9} {'off_max':>9} {'dup_off':>8} {'mid_x':>12} {'mid_y':>12}")
    for c in idx.cmp_ids[:50]:
        s = idx.cmp_summary(int(c))
        print(f"{s['cmp']:>8} {s['fold']:>6} {s['offset_min']:>9.1f} {s['offset_max']:>9.1f} "
              f"{s['duplicate_offsets']:>8} {s['mid_x']:>12.1f} {s['mid_y']:>12.1f}")
    print("\nValidation:\n" + idx.validate().text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
