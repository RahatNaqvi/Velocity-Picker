"""Writer for small synthetic SEG-Y files used in tests (rev 0 layout, big-endian)."""
import numpy as np

from velpick.data_io.segy import DEFAULT_TRACE_HEADER_MAP as MAP


def ieee_to_ibm(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.uint32)
    nz = x != 0
    a = np.abs(x[nz])
    e = np.floor(np.log(a) / np.log(16.0)).astype(np.int64) + 1
    frac = a / np.power(16.0, e)
    hi = frac >= 1.0
    frac[hi] /= 16.0; e[hi] += 1
    lo = frac < 1.0 / 16.0
    frac[lo] *= 16.0; e[lo] -= 1
    mant = np.round(frac * 16777216.0).astype(np.int64)
    ov = mant >= 16777216
    mant[ov] = 1048576; e[ov] += 1
    word = ((x[nz] < 0).astype(np.uint32) << 31) | ((e + 64).astype(np.uint32) << 24) | mant.astype(np.uint32)
    out[nz] = word
    return out


def _encode(samples, fmt):
    if fmt == 1:
        return ieee_to_ibm(samples).astype(">u4").tobytes()
    if fmt == 5:
        return samples.astype(">f4").tobytes()
    if fmt == 2:
        return np.round(samples).astype(">i4").tobytes()
    if fmt == 3:
        return np.round(samples).astype(">i2").tobytes()
    if fmt == 8:
        return np.round(samples).astype("i1").tobytes()
    raise ValueError(fmt)


def write_segy(path, traces, headers, dt_us=4000, fmt=1, binary_ns=None, binary_fmt_bytes=None):
    traces = np.asarray(traces, dtype=np.float64)
    n, ns = traces.shape
    text = "".join(f"C{i + 1:2d} synthetic test file".ljust(80) for i in range(40)).encode("cp500")
    b = bytearray(400)
    b[16:18] = int(dt_us).to_bytes(2, "big")
    b[20:22] = int(ns if binary_ns is None else binary_ns).to_bytes(2, "big")
    b[24:26] = int(fmt).to_bytes(2, "big") if binary_fmt_bytes is None else binary_fmt_bytes
    with open(path, "wb") as f:
        f.write(text); f.write(bytes(b))
        for i in range(n):
            h = np.zeros(240, dtype=np.uint8)
            vals = dict(headers[i]); vals.setdefault("ns", ns); vals.setdefault("dt", dt_us)
            for name, v in vals.items():
                fld = MAP[name]
                h[fld.byte - 1: fld.byte - 1 + fld.nbytes] = np.frombuffer(
                    np.array(v).astype(fld.be_dtype).tobytes(), dtype=np.uint8)
            f.write(h.tobytes()); f.write(_encode(traces[i], fmt))


def make_cmp_dataset(cmps=(100, 110, 130), fold=6, ns=40, scalar=-10, shuffle=True, seed=0,
                     offsets=None, header_offset_bias=0):
    """Traces whose samples encode (cmp, offset) so extraction can be verified; geometry on the x axis."""
    rng = np.random.default_rng(seed)
    rows, heads = [], []
    for ci, c in enumerate(cmps):
        offs = np.array(offsets if offsets is not None else [100 + 200 * k for k in range(fold)], float)
        for o in offs:
            mid = 1000.0 + 6.25 * c
            sx, gx = mid - o / 2, mid + o / 2
            f = 10.0 if scalar == -10 else (1.0 if scalar in (0, 1) else float(scalar))
            heads.append(dict(cdp=c, offset=int(o) + header_offset_bias, scalco=scalar,
                              sx=int(sx * f), sy=0, gx=int(gx * f), gy=0, cunits=0))
            rows.append(np.full(ns, c * 1000.0 + o / 10.0) + rng.normal(0, 0.0, ns))
    rows = np.array(rows)
    if shuffle:
        p = rng.permutation(len(rows))
        rows = rows[p]; heads = [heads[i] for i in p]
    return rows, heads
