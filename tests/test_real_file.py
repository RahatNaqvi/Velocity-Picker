"""Checks against the real 6-CMP test file (line 42, CMPs 700-750). Skipped if the file is absent.
Set VELPICK_TEST_SEGY to its path. The file is not stored in the repository."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from velpick.data_io import SegyFile
from velpick.gather_manager import extract_gather
from velpick.geometry import TraceIndex

PATH = Path(os.environ.get("VELPICK_TEST_SEGY", "/mnt/user-data/uploads/CMP700-750.sgy"))
pytestmark = pytest.mark.skipif(not PATH.is_file(), reason="real test file not available")


@pytest.fixture(scope="module")
def sf_idx():
    sf = SegyFile(PATH)
    return sf, TraceIndex.build(sf)


def test_file_level(sf_idx):
    sf, _ = sf_idx
    assert (sf.n_traces, sf.ns, sf.dt_us, sf.sample_format, sf.revision_major) == (2201, 1626, 4000, 1, 0)
    assert sf.warnings == []


def test_cmps_and_fold(sf_idx):
    _, idx = sf_idx
    assert idx.cmp_ids.tolist() == [700, 710, 720, 730, 740, 750]
    assert [idx.fold(c) for c in idx.cmp_ids] == [344, 353, 359, 373, 374, 398]
    assert idx.missing([700, 705, 750]) == [705]
    assert abs(idx.offset).min() == 644 and abs(idx.offset).max() == 7270


def test_offsets_agree_with_coordinates(sf_idx):
    _, idx = sf_idx
    assert np.abs(idx.computed_offset - np.abs(idx.offset)).max() < 1.0
    rep = idx.validate()
    assert not rep.has_errors and "OFFSET_MISMATCH" not in rep.codes()


def test_gather_700_and_750(sf_idx):
    sf, idx = sf_idx
    for cmp_id, omin, omax in ((700, 1269, 7270), (750, 644, 6645)):
        g = extract_gather(sf, idx, cmp_id)
        assert g.ntr == idx.fold(cmp_id) and g.ns == 1626
        assert g.offsets[0] == omin and g.offsets[-1] == omax
        assert np.all(np.diff(g.offsets) >= 0)
        assert np.isfinite(g.data).all() and g.dead.sum() == 0
        assert 1e3 < np.sqrt((g.data ** 2).mean()) < 1e5   # RMS about 5e3 measured independently


def test_reading_does_not_modify_file(sf_idx):
    before = hashlib.sha256(PATH.read_bytes()).hexdigest()
    sf = SegyFile(PATH); idx = TraceIndex.build(sf); extract_gather(sf, idx, 720)
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == before
