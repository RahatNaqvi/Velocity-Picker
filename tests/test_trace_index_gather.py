import numpy as np
import pytest

from synthetic_segy import make_cmp_dataset, write_segy
from velpick.data_io import SegyFile
from velpick.gather_manager import CmpNotFoundError, extract_gather
from velpick.geometry import IndexStaleError, TraceIndex, apply_scalar


def build(tmp_path, **kw):
    rows, heads = make_cmp_dataset(**kw)
    p = tmp_path / "t.sgy"
    write_segy(p, rows, heads)
    sf = SegyFile(p)
    return rows, heads, sf, TraceIndex.build(sf)


def test_apply_scalar_cases():
    v = np.array([1000, 1000, 1000, 1000])
    s = np.array([-10, 10, 0, 1])
    assert apply_scalar(v, s).tolist() == [100.0, 10000.0, 1000.0, 1000.0]


def test_noncontiguous_unsorted_cmps_located_by_actual_ids(tmp_path):
    rows, heads, sf, idx = build(tmp_path, cmps=(130, 100, 110))
    assert idx.cmp_ids.tolist() == [100, 110, 130]
    assert idx.fold(110) == 6
    assert idx.missing([100, 105, 130, 131]) == [105, 131]
    assert idx.nearest_ids(105, 2) == [100, 110]
    assert "CMP_NONCONTIGUOUS" in idx.validate().codes()
    assert "NOT_CMP_SORTED" in idx.validate().codes()
    with pytest.raises(KeyError):
        idx.positions(105)


def test_gather_sorted_by_offset_and_data_matches_cmp(tmp_path):
    rows, heads, sf, idx = build(tmp_path, cmps=(100, 110, 130), shuffle=True)
    g = extract_gather(sf, idx, 110)
    assert g.ntr == 6 and g.ns == 40 and g.dt_ms == 4.0
    assert np.all(np.diff(g.offsets) >= 0) and g.offsets[0] == 100 and g.offsets[-1] == 1100
    assert np.allclose(g.data[:, 0], 110 * 1000.0 + g.offsets / 10.0, rtol=2e-6)
    assert g.contributing_cmps == (110,) and not g.dead.any()
    with pytest.raises(CmpNotFoundError, match="nearest"):
        extract_gather(sf, idx, 120)


def test_offset_cross_check_consistent_and_biased(tmp_path):
    _, _, _, idx = build(tmp_path)
    assert "OFFSET_MISMATCH" not in idx.validate().codes()
    assert np.allclose(idx.computed_offset, np.abs(idx.offset), atol=0.11)
    _, _, _, idx2 = build(tmp_path, header_offset_bias=50)
    rep = idx2.validate()
    assert "OFFSET_MISMATCH" in rep.codes()
    assert [i for i in rep.issues if i.code == "OFFSET_MISMATCH"][0].count == idx2.n_traces


def test_wrong_scalar_detected_through_offset_check(tmp_path):
    rows, heads = make_cmp_dataset()
    for h in heads:
        h["scalco"] = 0          # coordinates were written x10, scalar says "no scaling"
    p = tmp_path / "t.sgy"; write_segy(p, rows, heads)
    assert "OFFSET_MISMATCH" in TraceIndex.build(SegyFile(p)).validate().codes()


def test_zero_offsets_and_missing_cmp_ids(tmp_path):
    rows, heads = make_cmp_dataset()
    for h in heads:
        h["offset"] = 0; h["sx"] = h["gx"] = 0; h["sy"] = h["gy"] = 0
    p = tmp_path / "t.sgy"; write_segy(p, rows, heads)
    assert "NO_OFFSET" in TraceIndex.build(SegyFile(p)).validate().codes()
    rows, heads = make_cmp_dataset()
    for h in heads:
        h["cdp"] = 0
    q = tmp_path / "t2.sgy"; write_segy(q, rows, heads)
    rep = TraceIndex.build(SegyFile(q)).validate()
    assert "NO_CMP_IDS" in rep.codes() and rep.has_errors


def test_header_offsets_zero_but_coordinates_valid_suggests_computed(tmp_path):
    rows, heads = make_cmp_dataset()
    for h in heads:
        h["offset"] = 0
    p = tmp_path / "t.sgy"; write_segy(p, rows, heads)
    sf = SegyFile(p); idx = TraceIndex.build(sf)
    assert "OFFSET_HEADER_ZERO" in idx.validate().codes()
    g = extract_gather(sf, idx, 100, offset_source="computed")
    assert g.offsets[0] == pytest.approx(100, abs=0.1) and g.offset_source == "computed"


def test_dead_and_nonfinite_traces_flagged_not_dropped(tmp_path):
    rows, heads = make_cmp_dataset(shuffle=False, cmps=(100,))
    rows[2, :] = 0.0
    p = tmp_path / "t.sgy"
    write_segy(p, rows, heads, fmt=5)
    sf = SegyFile(p)
    idx = TraceIndex.build(sf)
    g = extract_gather(sf, idx, 100)
    assert g.ntr == 6 and g.dead.sum() == 1
    rows[3, 5] = np.nan
    q = tmp_path / "n.sgy"; write_segy(q, rows, heads, fmt=5)
    sf2 = SegyFile(q); g2 = extract_gather(sf2, TraceIndex.build(sf2), 100)
    assert g2.n_nonfinite_samples == 1 and np.isfinite(g2.data).all()


def test_duplicate_offsets_reported_for_wide_azimuth_like_gathers(tmp_path):
    _, _, _, idx = build(tmp_path, offsets=[100, 100, 300, 300, 500, 700])
    assert idx.cmp_summary(100)["duplicate_offsets"] == 2
    assert "DUPLICATE_OFFSETS" in idx.validate().codes()


def test_index_cache_roundtrip_and_staleness(tmp_path):
    rows, heads, sf, idx = build(tmp_path)
    c = tmp_path / "idx.npz"
    idx.save(c)
    idx2 = TraceIndex.load(c, sf)
    assert np.array_equal(idx.cdp, idx2.cdp) and idx.headers_sha256 == idx2.headers_sha256
    assert idx2.positions(110).tolist() == idx.positions(110).tolist()
    # change the source file -> cache must be rejected, load_or_build rebuilds
    write_segy(sf.path, rows[:-1], heads[:-1])
    sf_new = SegyFile(sf.path)
    with pytest.raises(IndexStaleError):
        TraceIndex.load(c, sf_new)
    rebuilt = TraceIndex.load_or_build(sf_new, c)
    assert rebuilt.n_traces == sf_new.n_traces
