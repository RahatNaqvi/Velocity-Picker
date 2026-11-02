import hashlib
import numpy as np
import pytest

from synthetic_segy import ieee_to_ibm, make_cmp_dataset, write_segy
from velpick.data_io import SegyFile, SegyFormatError, ibm_to_ieee


def test_ibm_known_values():
    assert ibm_to_ieee(np.array([0xC276A000], dtype=np.uint32))[0] == pytest.approx(-118.625)
    assert ibm_to_ieee(np.array([0x41100000], dtype=np.uint32))[0] == 1.0
    assert ibm_to_ieee(np.array([0], dtype=np.uint32))[0] == 0.0
    assert ieee_to_ibm(np.array([-118.625]))[0] == 0xC276A000


def test_ibm_roundtrip_random():
    rng = np.random.default_rng(1)
    x = (rng.normal(size=5000) * 10.0 ** rng.integers(-6, 6, 5000)).astype(np.float32)
    y = ibm_to_ieee(ieee_to_ibm(x))
    assert np.allclose(y, x, rtol=2e-6, atol=0)


@pytest.mark.parametrize("fmt,tol", [(1, 2e-6), (5, 0), (2, 0), (3, 0), (8, 0)])
def test_sample_formats_roundtrip(tmp_path, fmt, tol):
    rows, heads = make_cmp_dataset(ns=30, shuffle=False)
    if fmt in (3, 8):
        rows = np.clip(np.round(rows / 100.0), -100, 100)
    p = tmp_path / "t.sgy"
    write_segy(p, rows, heads, fmt=fmt)
    sf = SegyFile(p)
    assert (sf.n_traces, sf.ns, sf.dt_us, sf.sample_format) == (len(rows), 30, 4000, fmt)
    out = sf.read_traces(np.arange(len(rows)))
    assert np.allclose(out, rows, rtol=tol, atol=0 if fmt != 1 else 1e-12)


def test_header_scan_and_selected_traces_in_requested_order(tmp_path):
    rows, heads = make_cmp_dataset(shuffle=True)
    p = tmp_path / "t.sgy"; write_segy(p, rows, heads)
    sf = SegyFile(p)
    h, sha = sf.scan_headers()
    assert h["cdp"].tolist() == [x["cdp"] for x in heads]
    assert h["offset"].tolist() == [x["offset"] for x in heads]
    assert len(sha) == 64
    pick = [7, 2, 2, 0]
    assert np.allclose(sf.read_traces(pick), rows[pick], rtol=2e-6)
    assert sf.read_traces([]).shape == (0, sf.ns)
    with pytest.raises(IndexError):
        sf.read_traces([sf.n_traces])


def test_text_header_decoded(tmp_path):
    rows, heads = make_cmp_dataset()
    p = tmp_path / "t.sgy"; write_segy(p, rows, heads)
    assert SegyFile(p).text_header[0].startswith("C 1")


def test_truncated_file_gives_clear_error(tmp_path):
    rows, heads = make_cmp_dataset()
    p = tmp_path / "t.sgy"; write_segy(p, rows, heads)
    data = p.read_bytes()
    q = tmp_path / "trunc.sgy"; q.write_bytes(data[:-100])
    with pytest.raises(SegyFormatError, match="whole number of traces"):
        SegyFile(q)


def test_binary_ns_wrong_falls_back_to_trace_header_with_warning(tmp_path):
    rows, heads = make_cmp_dataset(ns=40)
    p = tmp_path / "t.sgy"; write_segy(p, rows, heads, binary_ns=0)
    sf = SegyFile(p)
    assert sf.ns == 40 and any("differs" in w for w in sf.warnings)  # binary ns of 0 is reported
    q = tmp_path / "t2.sgy"; write_segy(q, rows, heads, binary_ns=41)
    sf2 = SegyFile(q)
    assert sf2.ns == 40 and any("differs" in w for w in sf2.warnings)


def test_little_endian_and_bad_format_rejected(tmp_path):
    rows, heads = make_cmp_dataset()
    p = tmp_path / "le.sgy"; write_segy(p, rows, heads, binary_fmt_bytes=(5).to_bytes(2, "little"))
    with pytest.raises(SegyFormatError, match="little-endian"):
        SegyFile(p)
    q = tmp_path / "bad.sgy"; write_segy(q, rows, heads, binary_fmt_bytes=(4).to_bytes(2, "big"))
    with pytest.raises(SegyFormatError, match="Unsupported"):
        SegyFile(q)


def test_source_file_never_modified(tmp_path):
    rows, heads = make_cmp_dataset()
    p = tmp_path / "t.sgy"; write_segy(p, rows, heads)
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    mt = p.stat().st_mtime_ns
    sf = SegyFile(p); sf.scan_headers(); sf.read_traces([0, 1, 2]); sf.close()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before
    assert p.stat().st_mtime_ns == mt
