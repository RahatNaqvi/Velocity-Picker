import sqlite3
import pytest

from velpick.database import ProjectDB
from velpick.export import export_csv
from velpick.picking import Navigator, PickModel, regular_sequence


@pytest.fixture
def db(tmp_path):
    d = ProjectDB(tmp_path / "p.db")
    yield d
    d.close()


def setup_loc(db, cmp_id=210):
    sid = db.add_source("/x/a.sgy", 100, 1, "abc", 10, 100, 2000, 1)
    return sid, db.ensure_location(sid, 0, cmp_id)


# ------------------------------------------------------------------ database
def test_schema_version_and_idempotent_open(tmp_path):
    d = ProjectDB(tmp_path / "p.db"); assert d.schema_version() == 2; d.close()
    d2 = ProjectDB(tmp_path / "p.db"); assert d2.schema_version() == 2; d2.close()


def test_newer_schema_is_refused(tmp_path):
    p = tmp_path / "p.db"; ProjectDB(p).close()
    c = sqlite3.connect(p); c.execute("PRAGMA user_version=99"); c.commit(); c.close()
    with pytest.raises(RuntimeError, match="newer"):
        ProjectDB(p)


def test_versions_are_immutable_and_history_kept(db):
    _, loc = setup_loc(db)
    v1, n1 = db.save_version(loc, [(100, 1500), (1000, 2500)], status="approved")
    v2, n2 = db.save_version(loc, [(100, 1500), (1000, 2600)])
    assert (n1, n2) == (1, 2) and v1 != v2
    assert db.get_picks(v1) == [(100, 1500), (1000, 2500)]          # approved version untouched
    assert db.get_version(v2)["parent_version_id"] == v1
    assert db.latest_version(loc)["id"] == v2
    assert [r["version_no"] for r in db.versions(loc)] == [1, 2]
    assert db.get_version(v1)["status"] == "approved"                # saving did not downgrade/alter it


def test_review_changes_status_only(db):
    _, loc = setup_loc(db)
    v, _ = db.save_version(loc, [(100, 1500)])
    db.set_review(v, "reviewed", "looks fine")
    r = db.get_version(v)
    assert r["status"] == "reviewed" and r["note"] == "looks fine" and db.get_picks(v) == [(100, 1500)]
    with pytest.raises(ValueError):
        db.set_review(v, "bogus")


def test_restore_makes_new_version_needing_review(db):
    _, loc = setup_loc(db)
    v1, _ = db.save_version(loc, [(100, 1500), (500, 2000)], status="approved")
    db.save_version(loc, [(100, 1600)])
    vr, nr = db.restore_version(v1, loc)
    assert nr == 3 and db.get_picks(vr) == [(100, 1500), (500, 2000)]
    row = db.get_version(vr)
    assert row["provenance"] == "restored" and row["status"] == "needs_review"


def test_invalid_input_rejected_without_partial_write(db):
    _, loc = setup_loc(db)
    for bad in ([(100, 1500), (100, 1600)], [(200, 1500), (100, 1600)], [(100, -5)]):
        with pytest.raises(ValueError):
            db.save_version(loc, bad)
    with pytest.raises(ValueError):
        db.save_version(loc, [(100, 1500)], status="nope")
    assert db.versions(loc) == []


def test_status_listing_provenance_and_config(db):
    sid, loc = setup_loc(db, 210)
    loc2 = db.ensure_location(sid, 0, 215)
    db.save_version(loc, [(100, 1500)], provenance="copied", source_info={"from_cmp": 205}, status="needs_review",
                    analysis_config={"method": "stackpower", "dv": 40})
    db.save_version(loc2, [(100, 1500)], status="reviewed")
    st = db.location_status(sid, 0)
    assert st[210]["status"] == "needs_review" and st[210]["provenance"] == "copied" and st[215]["status"] == "reviewed"
    assert 220 not in st
    cfg = db.conn.execute("SELECT COUNT(*) FROM analysis_config").fetchone()[0]
    db.save_version(loc, [(100, 1500)], analysis_config={"method": "stackpower", "dv": 40})
    assert db.conn.execute("SELECT COUNT(*) FROM analysis_config").fetchone()[0] == cfg      # same config reused


def test_drafts_roundtrip_and_cleared_on_save(db):
    _, loc = setup_loc(db)
    assert db.load_draft(loc) is None
    db.save_draft(loc, [(100, 1500), (900, 2400)], provenance="copied", source_info={"from_cmp": 1})
    d = db.load_draft(loc)
    assert d["picks"] == [(100, 1500), (900, 2400)] and d["provenance"] == "copied"
    db.save_draft(loc, [(100, 1500)])
    assert db.load_draft(loc)["picks"] == [(100, 1500)]
    db.save_version(loc, [(100, 1500)])
    assert db.load_draft(loc) is None


def test_compare_versions_and_meta(db):
    _, loc = setup_loc(db)
    a, _ = db.save_version(loc, [(100, 1500), (1000, 2500)])
    b, _ = db.save_version(loc, [(100, 1500), (500, 2300), (1000, 2600)])
    rows = db.compare_versions(a, b)
    assert [r["time_ms"] for r in rows] == [100, 500, 1000]

    assert rows[2]["diff"] == pytest.approx(100)
    db.set_meta("line", 42); assert db.get_meta("line") == 42 and db.get_meta("nope", 7) == 7


def test_persistence_across_reopen(tmp_path):
    p = tmp_path / "p.db"
    d = ProjectDB(p); sid, loc = setup_loc(d); vid, _ = d.save_version(loc, [(100, 1500), (900, 2400)]); d.close()
    d2 = ProjectDB(p)
    assert d2.get_picks(vid) == [(100, 1500), (900, 2400)]
    d2.close()


# ------------------------------------------------------------------ pick model
def test_add_move_delete_keep_sorted_and_unique():
    m = PickModel()
    for t, v in ((1000, 2500), (100, 1500), (500, 2000)):
        m.add(t, v)
    assert m.picks == [(100, 1500), (500, 2000), (1000, 2500)]
    i = m.add(500.4, 2100)                         # within 1 ms: replaces velocity, no duplicate time
    assert len(m) == 3 and m.picks[i] == (500, 2100)
    m.move(1, 5000, 2200)                          # clamped between neighbours (index stable)
    assert m.picks[1][0] == pytest.approx(999.0) and m.picks[1][1] == 2200
    m.move(1, -50, 2200)
    assert m.picks[1][0] == pytest.approx(101.0)
    m.delete(0); assert len(m) == 2
    with pytest.raises(ValueError):
        m.add(100, 0)


def test_undo_redo_and_drag_checkpoint():
    m = PickModel(); m.add(100, 1500); m.add(900, 2400)
    m.checkpoint(); m.move(1, 800, 2300); m.move(1, 700, 2250); m.move(1, 600, 2200)   # one drag = one undo step
    assert m.picks[1] == (600, 2200)
    assert m.undo() and m.picks[1] == (900, 2400)
    assert m.redo() and m.picks[1] == (600, 2200)
    m.undo(); m.undo(); m.undo()
    assert m.picks == [] and not m.undo()


def test_dirty_tracking_and_reset():
    m = PickModel(); assert not m.dirty
    m.add(100, 1500); assert m.dirty
    m.mark_saved(); assert not m.dirty
    m.add(900, 2400); assert m.dirty
    m.reset([(100, 1500)], "copied", {"from_cmp": 5})
    assert not m.dirty and m.provenance == "copied" and not m.undo()


def test_copied_picks_carry_provenance_and_original_untouched():
    src = PickModel(); src.add(100, 1500); src.add(900, 2400)
    dst = PickModel(); dst.set_all(src.picks, provenance="copied", source_info={"from_cmp": 210})
    dst.move(0, 120, 1550)
    assert src.picks[0] == (100, 1500) and dst.provenance == "copied" and dst.dirty


def test_warnings_are_informational():
    m = PickModel()
    m.add(100, 1500); m.add(500, 3000); m.add(900, 2800); m.add(1500, 7000)
    w = " ".join(m.warnings(vmin=1500, vmax=6000, max_jump_pct=25))
    assert "Large velocity jump" in w and "reversal" in w and "outside the analysed velocity range" in w
    assert len(m) == 4                                              # nothing was changed or blocked
    assert PickModel().warnings() == []


def test_velocity_function_interpolation():
    m = PickModel(); m.add(100, 1500); m.add(1100, 2500)
    assert m.velocity_at([100, 600, 1100, 2000, 0]).tolist() == [1500, 2000, 2500, 2500, 1500]
    assert PickModel().velocity_at([1, 2]).tolist() != [1, 2]
    assert m.nearest_index(105, 1510, t_scale=50, v_scale=50, max_dist=1) == 0
    assert m.nearest_index(600, 9000, 50, 50, 1) is None


# ------------------------------------------------------------------ navigation
def test_navigator_and_regular_sequence_with_gaps():
    ids, missing = regular_sequence([200, 225, 250, 300, 350], 200, 350, 50)
    assert ids == [200, 250, 300, 350] and missing == []
    ids, missing = regular_sequence([200, 250, 350], 200, 350, 50)
    assert ids == [200, 250, 350] and missing == [300]
    nav = Navigator([300, 100, 200])
    assert nav.current == 100 and nav.previous() is None and nav.next() == 200 and nav.next() == 300 and nav.next() is None
    assert nav.jump(100) == 100 and nav.next_id() == 200
    with pytest.raises(KeyError):
        nav.jump(150)
    with pytest.raises(ValueError):
        regular_sequence([1], 1, 5, 0)


# ------------------------------------------------------------------ export
def test_csv_export_latest_versions(db, tmp_path):
    sid, loc = setup_loc(db, 210)
    loc2 = db.ensure_location(sid, 0, 215)
    db.save_version(loc, [(100, 1500)])
    db.save_version(loc, [(100, 1500), (900, 2400)], status="approved")
    db.save_version(loc2, [(100, 1700)], status="needs_review")
    out = tmp_path / "o.csv"
    assert export_csv(db, sid, out) == 2
    text = out.read_text()
    assert "velocity_type: stacking" in text
    rows = [r for r in text.splitlines() if not r.startswith("#")][1:]
    assert len(rows) == 3 and rows[1].startswith("0,210,900.0,2400.0,stacking,2,approved,manual")
    assert export_csv(db, sid, tmp_path / "a.csv", statuses=["approved"]) == 1


# ------------------------------------------------------------------ named velocities, overwrite, migration
def test_default_named_velocity_exists_and_is_active(db):
    sets = db.list_sets()
    assert [s["name"] for s in sets] == ["default"] and db.active_set_id == 1


def test_same_cmp_can_hold_different_functions_in_different_names(db):
    _, loc = setup_loc(db)
    db.save_version(loc, [(100, 1500), (900, 2400)])                       # in 'default'
    velo1 = db.create_set("velo1", "first pass")
    db.save_version(loc, [(100, 1600), (900, 2600)], set_id=velo1)
    assert db.get_picks(db.latest_version(loc, set_id=1)["id"]) == [(100, 1500), (900, 2400)]
    assert db.get_picks(db.latest_version(loc, set_id=velo1)["id"]) == [(100, 1600), (900, 2600)]
    db.set_active(velo1)                                                   # default for reads/writes follows the active name
    assert db.latest_version(loc)["version_no"] == 1
    _, vno = db.save_version(loc, [(100, 1650)])
    assert vno == 2 and len(db.versions(loc, set_id=1)) == 1 and len(db.versions(loc)) == 2
    st = db.location_status(1, 0)
    assert st[210]["version_no"] == 2
    assert db.location_status(1, 0, set_id=1)[210]["version_no"] == 1


def test_velocity_names_are_validated_and_unique_case_insensitively(db):
    db.create_set("velo1")
    for bad in ("", "   ", "velo1", "VELO1", "a/b", "x" * 81, "what?"):
        with pytest.raises(ValueError):
            db.create_set(bad)
    sid = db.create_set("  velo2  ")
    assert db.get_set(sid)["name"] == "velo2"
    db.rename_set(sid, "final")
    assert db.set_by_name("final")["id"] == sid
    with pytest.raises(ValueError):
        db.rename_set(sid, "default")
    db.rename_set(sid, "FINAL")                                            # renaming to itself in another case is fine


def test_active_name_is_remembered_across_sessions(tmp_path):
    d = ProjectDB(tmp_path / "p.db"); sid = d.create_set("velo9"); d.set_active(sid); d.close()
    d2 = ProjectDB(tmp_path / "p.db"); assert d2.active_set_id == sid; d2.close()


def test_overwrite_replaces_picks_in_place_and_keeps_an_audit_trail(db):
    _, loc = setup_loc(db)
    vid, vno = db.save_version(loc, [(100, 1500), (900, 2400)], status="picked", note="first")
    n = db.overwrite_version(vid, [(100, 1550), (500, 2000), (900, 2450)], status="reviewed", note="fixed shallow")
    assert n == vno == 1 and len(db.versions(loc)) == 1                    # same version, no new row
    assert db.get_picks(vid) == [(100, 1550), (500, 2000), (900, 2450)]
    row = db.get_version(vid)
    assert row["status"] == "reviewed" and row["note"] == "fixed shallow"
    log = db.audit_log(vid)
    assert len(log) == 1 and log[0]["old_picks"] == [(100, 1500), (900, 2400)] and log[0]["old_status"] == "picked"
    db.overwrite_version(vid, [(100, 1600)])
    assert [l["old_picks"] for l in db.audit_log(vid)][-1] == [(100, 1550), (500, 2000), (900, 2450)]   # nothing is lost


def test_approved_versions_cannot_be_overwritten(db):
    _, loc = setup_loc(db)
    vid, _ = db.save_version(loc, [(100, 1500)], status="approved")
    with pytest.raises(ValueError, match="approved"):
        db.overwrite_version(vid, [(100, 1700)])
    assert db.get_picks(vid) == [(100, 1500)] and db.audit_log(vid) == []
    db.set_review(vid, "reviewed")                                         # an explicit status change unlocks it
    db.overwrite_version(vid, [(100, 1700)])
    assert db.get_picks(vid) == [(100, 1700)]


def test_overwrite_validation_leaves_the_version_untouched(db):
    _, loc = setup_loc(db)
    vid, _ = db.save_version(loc, [(100, 1500), (900, 2400)])
    with pytest.raises(ValueError):
        db.overwrite_version(vid, [(100, 1500), (100, 1600)])
    with pytest.raises(ValueError):
        db.overwrite_version(vid, [(100, 1500)], status="bogus")
    with pytest.raises(KeyError):
        db.overwrite_version(99999, [(100, 1500)])
    assert db.get_picks(vid) == [(100, 1500), (900, 2400)] and db.audit_log(vid) == []


def test_copy_set_copies_latest_versions_marks_review_and_leaves_source_alone(db):
    sid, loc = setup_loc(db, 210)
    loc2 = db.ensure_location(sid, 0, 215)
    db.save_version(loc, [(100, 1500)], status="approved")
    db.save_version(loc, [(100, 1500), (900, 2400)], status="reviewed")    # latest of 210
    db.save_version(loc2, [(100, 1700)], status="approved")
    new = db.copy_set(1, "velo2")
    assert db.get_set(new)["description"].startswith("Copy of 'default'")
    v210, v215 = db.latest_version(loc, set_id=new), db.latest_version(loc2, set_id=new)
    assert db.get_picks(v210["id"]) == [(100, 1500), (900, 2400)] and v210["version_no"] == 1
    assert v210["provenance"] == "copied" and v210["status"] == "needs_review" and v215["status"] == "needs_review"
    assert len(db.versions(loc, set_id=1)) == 2 and db.latest_version(loc2, set_id=1)["status"] == "approved"   # source intact
    kept = db.copy_set(1, "velo3", keep_status=True)
    assert db.latest_version(loc2, set_id=kept)["status"] == "approved"
    with pytest.raises(ValueError):
        db.copy_set(1, "velo2")
    assert db.set_by_name("velo2") is not None and len(db.list_sets()) == 3   # a failed copy created nothing extra
    stats = {s["name"]: (s["n_functions"], s["n_versions"]) for s in db.list_sets()}
    assert stats["default"] == (2, 3) and stats["velo2"] == (2, 2)


def test_drafts_are_kept_per_named_velocity(db):
    _, loc = setup_loc(db)
    other = db.create_set("velo1")
    db.save_draft(loc, [(100, 1500)])
    db.save_draft(loc, [(100, 1999)], set_id=other)
    assert db.load_draft(loc)["picks"] == [(100, 1500)] and db.load_draft(loc, set_id=other)["picks"] == [(100, 1999)]
    db.save_version(loc, [(100, 1500)])                                    # saving clears only that name's draft
    assert db.load_draft(loc) is None and db.load_draft(loc, set_id=other) is not None


def test_csv_export_is_per_named_velocity_and_names_it(db, tmp_path):
    sid, loc = setup_loc(db, 210)
    db.save_version(loc, [(100, 1500)])
    velo1 = db.create_set("velo1"); db.save_version(loc, [(100, 1800), (900, 2600)], set_id=velo1)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert export_csv(db, sid, a, set_id=1) == 1 and export_csv(db, sid, b, set_id=velo1) == 1
    assert "# velocity_name: default" in a.read_text() and "# velocity_name: velo1" in b.read_text()
    rows = [r for r in b.read_text().splitlines() if not r.startswith("#")]
    assert rows[0].endswith("velocity_name") and rows[1].endswith(",velo1") and len(rows) == 3
    db.set_active(velo1)
    c = tmp_path / "c.csv"; export_csv(db, sid, c)                          # default = active name
    assert "# velocity_name: velo1" in c.read_text()


def test_migration_from_schema_1_keeps_existing_picks_and_makes_a_backup(tmp_path):
    from velpick.database.db import _V1
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.executescript(_V1); c.execute("PRAGMA user_version=1")                # a project made by the previous version
    c.execute("INSERT INTO source_file(path,size_bytes,mtime_ns,headers_sha256,n_traces,samples_per_trace,dt_us,sample_format,added_utc) "
              "VALUES('/x/a.sgy',1,1,'abc',10,100,2000,1,'2026-10-01')")
    c.execute("INSERT INTO location(source_file_id,line,cmp) VALUES(1,0,210)")
    c.execute("INSERT INTO velocity_function(location_id,velocity_type) VALUES(1,'stacking')")
    c.execute("INSERT INTO function_version(function_id,version_no,created_utc,provenance,status,note) VALUES(1,1,'2026-10-02','manual','approved','old')")
    c.execute("INSERT INTO function_version(function_id,version_no,parent_version_id,created_utc,provenance,status,note) VALUES(1,2,1,'2026-10-03','copied','needs_review','')")
    c.executemany("INSERT INTO pick VALUES(?,?,?,?)", [(1, 0, 100.0, 1500.0), (1, 1, 900.0, 2400.0), (2, 0, 100.0, 1600.0)])
    c.execute("INSERT INTO draft VALUES(1,'stacking','[[100,1700]]','manual',NULL,'2026-10-04')")
    c.execute("INSERT INTO meta VALUES('last_cmp','210')")
    c.commit(); c.close()
    d = ProjectDB(p)
    assert d.schema_version() == 2 and (tmp_path / "old.db.bak-v1").exists()
    assert [s["name"] for s in d.list_sets()] == ["default"]
    loc = d.conn.execute("SELECT id FROM location").fetchone()["id"]
    assert [v["version_no"] for v in d.versions(loc)] == [1, 2]
    assert d.get_picks(d.versions(loc)[0]["id"]) == [(100.0, 1500.0), (900.0, 2400.0)]
    assert d.latest_version(loc)["provenance"] == "copied" and d.get_version(1)["status"] == "approved"
    assert d.load_draft(loc)["picks"] == [(100.0, 1700.0)] and d.get_meta("last_cmp") == 210
    assert d.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    d.save_version(loc, [(100, 1800)])                                      # and it keeps working
    assert len(d.versions(loc)) == 3
    d.close()
    sqlite3.connect(tmp_path / "old.db.bak-v1").execute("SELECT 1 FROM velocity_function").fetchall()   # backup is the old schema


# ------------------------------------------------------------------ file identity (moving / renaming the SEG-Y)
def _add(db, path, sha="h1", size=100):
    return db.add_source(path, size, 1, sha, 10, 100, 2000, 1)


def test_same_data_opened_from_another_path_reattaches_the_picks(db):
    a = _add(db, "/data/line42.sgy")
    loc = db.ensure_location(a, 0, 210); db.save_version(loc, [(100, 1500)])
    assert db.source_moved_from is None
    b = _add(db, "/mnt/other/renamed.sgy")                                   # moved / renamed / another mount point
    assert b == a and db.source_moved_from == "/data/line42.sgy"
    assert db.conn.execute("SELECT path FROM source_file WHERE id=?", (a,)).fetchone()[0] == "/mnt/other/renamed.sgy"
    assert db.get_picks(db.latest_version(db.ensure_location(b, 0, 210))["id"]) == [(100, 1500)]
    assert _add(db, "/mnt/other/renamed.sgy") == a and db.source_moved_from is None   # same path again: nothing to report
    assert _add(db, "/data/other.sgy", sha="h2") != a                        # different content = a different file
    assert _add(db, "/data/line42_cut.sgy", size=999) != a                   # same headers but another size = different file


def test_duplicate_source_rows_made_by_older_versions_are_merged(db):
    old = _add(db, "/old/path.sgy"); new_loc_old = [db.ensure_location(old, 0, c) for c in (210, 211, 212)]
    db.save_version(new_loc_old[0], [(100, 1500)]); db.save_version(new_loc_old[2], [(100, 1700)])      # 211 has no picks
    db.conn.execute("INSERT INTO source_file(path,size_bytes,mtime_ns,headers_sha256,n_traces,samples_per_trace,dt_us,sample_format,added_utc)"
                    " VALUES('/new/path.sgy',100,1,'h1',10,100,2000,1,'x')"); db.conn.commit()
    new = db.conn.execute("SELECT id FROM source_file WHERE path='/new/path.sgy'").fetchone()[0]
    loc211 = db.ensure_location(new, 0, 211); db.save_version(loc211, [(100, 1900)])
    assert _add(db, "/new/path.sgy") == new                                    # exact path wins; the older entry is folded in
    st = db.location_status(new, 0)
    assert set(st) == {210, 211, 212}
    assert db.get_picks(db.latest_version(db.ensure_location(new, 0, 210))["id"]) == [(100, 1500)]
    assert db.get_picks(db.latest_version(db.ensure_location(new, 0, 211))["id"]) == [(100, 1900)]   # nothing overwritten
    assert db.conn.execute("SELECT COUNT(*) FROM source_file").fetchone()[0] == 1
    assert db.conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_merge_leaves_real_conflicts_alone(db):
    old = _add(db, "/old.sgy"); db.save_version(db.ensure_location(old, 0, 210), [(100, 1500)])
    db.conn.execute("INSERT INTO source_file(path,size_bytes,mtime_ns,headers_sha256,n_traces,samples_per_trace,dt_us,sample_format,added_utc)"
                    " VALUES('/new.sgy',100,1,'h1',10,100,2000,1,'x')"); db.conn.commit()
    new = db.conn.execute("SELECT id FROM source_file WHERE path='/new.sgy'").fetchone()[0]
    db.save_version(db.ensure_location(new, 0, 210), [(100, 2222)])            # both hold picks for CMP 210
    assert _add(db, "/new.sgy") == new
    assert db.get_picks(db.latest_version(db.ensure_location(new, 0, 210))["id"]) == [(100, 2222)]
    assert db.get_picks(db.latest_version(db.ensure_location(old, 0, 210))["id"]) == [(100, 1500)]   # both kept
    assert db._merge_sources(new) == (0, 1)
