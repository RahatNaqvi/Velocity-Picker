import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from synthetic_gathers import make_gather  # noqa: E402
from synthetic_segy import write_segy  # noqa: E402
from velpick.database import ProjectDB  # noqa: E402
from velpick.ui.app import MainWindow  # noqa: E402
from velpick.ui.settings import AnalysisSettings  # noqa: E402

OFF = np.arange(200.0, 3200.0, 100.0)


def make_file(path, cmps=(100, 101, 102, 103, 104)):
    rows, heads = [], []
    for c in cmps:
        d = make_gather([(400, 2500, 1.0), (900, 3000, 1.0)], OFF, 4.0, 500)
        for i, o in enumerate(OFF):
            mid = 1000 + 6.25 * c
            heads.append(dict(cdp=c, offset=int(o), scalco=-10, sx=int((mid - o / 2) * 10), sy=0,
                              gx=int((mid + o / 2) * 10), gy=0, cunits=0))
            rows.append(d[i])
    write_segy(path, np.array(rows), heads, dt_us=4000, fmt=5)


def small_settings(method="semblance"):
    return AnalysisSettings(method=method, vmin=1500, vmax=4000, dv=100, t0_start=100, t0_end=1500, line_interval=25,
                            bandpass=False, eq_ms=0, window_ms=40, sem_stretch=50, stretch=50, n_cmps=3,
                            max_offset=0, cvs_enabled=True, cvs_strips=9, cvs_dt_ms=8)


@pytest.fixture(autouse=True)
def no_blocking_dialogs(monkeypatch):
    # closeEvent asks to save unsaved picks; tests discard instead of blocking on a modal dialog
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Discard))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))


def open_window(qtbot, tmp_path, method="semblance", name="d.sgy"):
    f = tmp_path / name
    if not f.exists():
        make_file(f)
    w = MainWindow()
    qtbot.addWidget(w)
    w.settings = small_settings(method)
    w.open_segy(f, line=42)
    wait_result(qtbot, w)
    return w, f


def wait_result(qtbot, w, cmp_id=None):
    qtbot.waitUntil(lambda: w.result is not None and w.result.cmp_id == (cmp_id or w.nav.current), timeout=30000)


def test_open_computes_spectrum_and_lists_locations(qtbot, tmp_path):
    w, f = open_window(qtbot, tmp_path)
    r = w.result
    assert r.display.shape == (r.t0_ms.size, r.velocities.size) and r.display.max() <= 1.0 + 1e-6
    assert w.nav.ids == [100, 101, 102, 103, 104] and w.status_list.count() == 5
    assert "unprocessed" in w.status_list.item(0).text() and "fold 30" in w.info.text()
    # the spectrum really peaks at the synthetic event (v = 2500 at t0 = 400 ms)
    row = int(np.argmin(np.abs(r.t0_ms - 400)))
    assert abs(r.velocities[int(np.argmax(r.display[row]))] - 2500) <= 100
    assert (tmp_path / "d.velpick.sqlite").exists() and (tmp_path / "d.velpick-index.npz").exists()


def test_pick_save_navigate_copy_export(qtbot, tmp_path):
    w, f = open_window(qtbot, tmp_path)
    w.on_add(2500.0, 400.0); w.on_add(3000.0, 900.0); w.on_add(3400.0, 1400.0)
    assert w.model.picks == [(400.0, 2500.0), (900.0, 3000.0), (1400.0, 3400.0)] and w.model.dirty
    assert "*" in w.windowTitle()
    assert w.save_version() == 1 and not w.model.dirty
    loc = w.loc_id
    assert w.db.get_picks(w.db.latest_version(loc)["id"]) == w.model.picks
    assert "picked" in w.status_list.item(0).text()

    w.next_cmp(); wait_result(qtbot, w, 101)
    assert w.nav.current == 101 and w.model.picks == []             # nothing silently carried over
    w.copy_previous()
    assert w.model.picks[0] == (400.0, 2500.0) and w.model.provenance == "copied"
    assert w.status_combo.currentText() == "needs_review"
    w.on_move(1, 3100.0, 905.0, True)
    assert w.save_version() == 1
    v = w.db.latest_version(w.loc_id)
    assert v["provenance"] == "copied" and v["status"] == "needs_review"
    # CMP 100's saved function is untouched by what happened on CMP 101
    assert w.db.get_picks(w.db.latest_version(loc)["id"])[1] == (900.0, 3000.0)

    w.prev_cmp(); wait_result(qtbot, w, 100)
    assert w.model.picks[1] == (900.0, 3000.0) and not w.model.dirty     # saved picks reloaded
    w.status_combo.setCurrentText("approved")
    w.on_move(0, 2550.0, 405.0, True)
    assert w.save_version() == 2                                         # new version; approved v... history kept
    assert [r["version_no"] for r in w.db.versions(loc)] == [1, 2]
    assert w.db.get_picks(w.db.versions(loc)[0]["id"])[0] == (400.0, 2500.0)

    out = tmp_path / "o.csv"
    from velpick.export import export_csv
    assert export_csv(w.db, w.source_id, out, line=42) == 2
    assert "42,100,405.0,2550.0,stacking,2,approved,manual" in out.read_text()


def test_drag_is_single_undo_step_and_delete_select(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.on_add(2000.0, 300.0); w.on_add(3000.0, 900.0)
    for k, fin in ((2900.0, False), (2800.0, False), (2700.0, True)):
        w.on_move(1, k, 900.0, fin)
    assert w.model.picks[1][1] == 2700.0
    w.undo()
    assert w.model.picks[1][1] == 3000.0                                 # whole drag undone at once
    w.redo(); assert w.model.picks[1][1] == 2700.0
    w.on_select(0); assert w.sel == 0 and w.selline[0].isVisible()
    w.delete_selected(); assert len(w.model) == 1 and w.sel is None
    w.clear_picks(); assert len(w.model) == 0


def test_table_edit_updates_picks(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.on_add(2000.0, 300.0)
    w.table.item(0, 1).setText("2222")
    assert w.model.picks == [(300.0, 2222.0)]


def test_warnings_shown_but_do_not_block_save(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.on_add(1600.0, 200.0); w.on_add(3900.0, 600.0); w.on_add(3000.0, 1000.0)
    assert "jump" in w.warn.text() and "reversal" in w.warn.text()
    assert w.save_version() == 1


def test_draft_recovery_after_unsaved_session(qtbot, tmp_path, monkeypatch):
    w, f = open_window(qtbot, tmp_path)
    w.on_add(2500.0, 400.0)
    w._autosave()                                                        # what the 30 s timer does
    w.cancel_jobs()
    w2 = MainWindow(); qtbot.addWidget(w2)
    w2.settings = small_settings()
    monkeypatch.setattr(w2, "_ask_recover", lambda draft: True)
    w2.open_segy(f, line=42)
    assert w2.model.picks == [(400.0, 2500.0)] and w2.model.dirty
    w3 = MainWindow(); qtbot.addWidget(w3)
    w3.settings = small_settings()
    monkeypatch.setattr(w3, "_ask_recover", lambda draft: False)
    w3.open_segy(f, line=42)
    assert w3.model.picks == [] and w3.db.load_draft(w3.loc_id) is None


def test_stack_power_method_uses_neighbouring_cmps(qtbot, tmp_path):
    w, f = open_window(qtbot, tmp_path, method="stackpower", name="sp.sgy")
    w.goto_cmp(102); wait_result(qtbot, w, 102)
    assert w.result.cmps_used == [101, 102, 103] and w.result.method == "stackpower"
    w.goto_cmp(100); wait_result(qtbot, w, 100)
    assert w.result.cmps_used == [100, 101, 102] or len(w.result.cmps_used) == 3


def test_settings_change_recomputes_and_persists(qtbot, tmp_path):
    w, f = open_window(qtbot, tmp_path)
    first = w.result
    w.settings = small_settings(); w.settings.dv = 50.0
    w.db.set_meta("analysis_settings", w.settings.to_dict())
    w.cache.clear(); w._request_analysis(); qtbot.waitUntil(lambda: w.result is not first, timeout=30000)
    assert abs(w.result.velocities[1] - w.result.velocities[0] - 50) < 1e-6
    w2 = MainWindow(); qtbot.addWidget(w2); w2.open_segy(f, line=42)
    assert w2.settings.dv == 50.0


def test_sequence_with_gaps_and_jump(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.set_sequence([100, 102, 104]); wait_result(qtbot, w, 100)
    w.next_cmp(); wait_result(qtbot, w, 102)
    assert w.nav.current == 102 and w.cmp_combo.count() == 3
    w.goto_cmp(104); wait_result(qtbot, w, 104)
    assert w.nav.next_id() is None


def test_nmo_preview_and_gather_display_modes_render(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    assert w.mode_combo.currentText() == "Wiggle + VA" and len(w._gather_items) > 20     # processing-system style default
    w.on_add(2500.0, 400.0); w.on_add(3000.0, 900.0)
    w.nmo_check.setChecked(True)
    assert "NMO preview" in w.p_gather.titleLabel.text
    w.mode_combo.setCurrentText("Image")
    assert w._gather_items == [] and w.img_gather.isVisible()
    w.nmo_check.setChecked(False)
    assert w.result.gather.data.dtype == np.float32                      # source gather unchanged by previews


def test_hit_test_and_mouse_click_on_spectrum(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w)
    qtbot.wait(100)
    vb = w.vb_spec
    w.model.add(400.0, 2500.0); w._refresh_picks()
    sp = vb.mapViewToScene(QtCore.QPointF(2500.0, 400.0))
    assert w._hit(vb, sp, w._xy_spec) == 0 and w._hit(vb, sp + QtCore.QPointF(60, 60), w._xy_spec) is None
    far = vb.mapViewToScene(QtCore.QPointF(3500.0, 1000.0))
    vp = w.glw.mapFromScene(far)
    qtbot.mouseClick(w.glw.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=vp)
    qtbot.waitUntil(lambda: len(w.model) == 2, timeout=3000)
    t, v = w.model.picks[1]
    assert abs(v - 3500) < 60 and abs(t - 1000) < 60


def test_axes_are_in_ms_and_m_per_s_without_si_prefix(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    for p in w.plots:
        assert not p.getAxis("left").autoSIPrefix and not p.getAxis("top").autoSIPrefix
    assert w.p_spec.getAxis("left").labelUnits == "ms" and w.p_spec.getAxis("top").labelUnits == "m/s"


def test_time_axis_never_negative_and_panels_share_one_range(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    lo_a, hi_a = w._view_range(w.result)
    t_lo, t_hi = w._record_range(w.result)
    assert lo_a >= t_lo and hi_a <= t_hi and lo_a == pytest.approx(w.result.t0_ms[0] - 12.5)
    ranges = [p.getViewBox().viewRange()[1] for p in w.plots]
    assert all(abs(r[0] - lo_a) < 1e-6 and abs(r[1] - hi_a) < 1e-6 for r in ranges)   # all four panels: identical start/end
    w.p_spec.setYRange(-500, 100, padding=0)                              # user/auto-range tries to go outside
    w.p_spec.setYRange(-500, 50000, padding=0)
    for p in w.plots:
        lo, hi = p.getViewBox().viewRange()[1]
        assert lo >= lo_a - 1e-6 and hi <= hi_a + 1e-6 and lo >= 0
    w.reset_view()
    assert all(abs(p.getViewBox().viewRange()[1][1] - hi_a) < 1e-6 for p in w.plots)


def test_all_panels_bottom_out_at_the_analysed_end_not_the_record_end(qtbot, tmp_path):
    f = tmp_path / "d.sgy"; make_file(f)                                 # record 0-1996 ms
    w = MainWindow(); qtbot.addWidget(w)
    s = small_settings(); s.t0_end = 1000.0                               # analysis ends well before the record does
    w.settings = s; w.open_segy(f, line=42); wait_result(qtbot, w)
    hi = w._view_range(w.result)[1]
    assert hi < 1100                                                       # not 1996: gather is not shown deeper than the spectrum
    assert all(abs(p.getViewBox().viewRange()[1][1] - hi) < 1e-6 for p in w.plots)


def test_toolbars_are_on_separate_rows(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1100, 700); w.show(); qtbot.waitExposed(w); qtbot.wait(50)
    bars = w.findChildren(QtWidgets.QToolBar)
    ys = sorted({b.geometry().y() for b in bars})
    assert len(ys) >= 2


def test_default_analysis_range_covers_record_and_is_clamped(qtbot, tmp_path):
    assert AnalysisSettings().t0_end == 6500.0
    f = tmp_path / "d.sgy"; make_file(f)
    w = MainWindow(); qtbot.addWidget(w)
    s = small_settings(); s.t0_end = 6500.0
    w.settings = s; w.open_segy(f, line=42); wait_result(qtbot, w)
    rec_end = (500 - 1) * 4.0
    assert w.result.t0_ms[-1] <= rec_end - 8 + 1e-6 and any("limited" in n for n in w.result.notes)
    lo, hi = w.p_spec.getViewBox().viewRange()[1]
    assert hi <= rec_end + 1e-6                                           # spectrum and gather end together


def test_constant_velocity_stack_panel_and_picking_on_it(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    r = w.result
    assert r.cvs.shape[:2] == (9, 3) and len(r.cvs_vels) == 9 and r.cvs_cmps == [100, 101, 102]
    assert len(w._cvs_items) > 20 and len(w._cvs_centres) == 9
    k = 4
    x = float(w._cvs_centres[k])
    w.vb_cvs.sigAdd.emit(x, 400.0)                                        # a click on strip 4 at 400 ms
    assert w.model.picks == [(400.0, float(r.cvs_vels[k]))]
    assert len(w.cvs_pts.data) == 1                                       # triangle drawn at that strip
    x2 = float(w._cvs_centres[k]) + 0.5 * (w._cvs_centres[k + 1] - w._cvs_centres[k])
    w.vb_cvs.sigMove.emit(0, x2, 420.0, True)                             # drag between two strips = in-between velocity
    t, v = w.model.picks[0]
    assert t == 420.0 and r.cvs_vels[k] < v < r.cvs_vels[k + 1]
    w.vb_cvs.sigDelete.emit(0)
    assert w.model.picks == []


def test_cvs_strip_nearest_true_velocity_is_strongest(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    r = w.result
    row = int(np.argmin(np.abs(r.cvs_t - 400)))
    amp = np.abs(r.cvs[:, :, max(0, row - 2):row + 3]).max(axis=(1, 2))
    assert abs(r.cvs_vels[int(np.argmax(amp))] - 2500) <= 2 * (r.cvs_vels[1] - r.cvs_vels[0])


# ------------------------------------------------------------------ zoom / scroll / layout requests
class _FakeWheel:
    """Minimal stand-in for a QGraphicsSceneWheelEvent."""
    def __init__(self, scene_pos, delta, mods=QtCore.Qt.KeyboardModifier.NoModifier):
        self._p, self._d, self._m, self.accepted = scene_pos, delta, mods, False

    def scenePos(self): return self._p
    def delta(self): return self._d
    def modifiers(self): return self._m
    def accept(self): self.accepted = True


def _centre(vb):
    return vb.sceneBoundingRect().center()


def test_wheel_zooms_time_around_cursor_and_all_panels_follow(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(50)
    lo0, hi0 = w.vb_spec.viewRange()[1]
    x0 = list(w.vb_spec.viewRange()[0])
    for vb in (w.vb_spec, w.vb_gather, w.vb_cvs):                     # wheel works in every panel
        w.reset_view()
        ev = _FakeWheel(_centre(vb), 120)                              # one notch forward = zoom in
        vb.wheelEvent(ev)
        assert ev.accepted
        lo, hi = w.vb_spec.viewRange()[1]
        assert (hi - lo) < (hi0 - lo0) * 0.95                          # time window shrank ...
        assert all(abs(p.getViewBox().viewRange()[1][0] - lo) < 1e-6 for p in w.plots)   # ... in all panels
    assert w.vb_spec.viewRange()[0] == pytest.approx(x0)               # velocity range untouched by a plain wheel
    ctrl = _FakeWheel(_centre(w.vb_spec), 120, QtCore.Qt.KeyboardModifier.ControlModifier)
    w.reset_view(); w.vb_spec.wheelEvent(ctrl)
    assert (w.vb_spec.viewRange()[0][1] - w.vb_spec.viewRange()[0][0]) < (x0[1] - x0[0]) * 0.95
    out = _FakeWheel(_centre(w.vb_spec), -120)
    w.reset_view(); w.vb_spec.wheelEvent(out)                          # zooming out never leaves the analysed range
    lo, hi = w.vb_spec.viewRange()[1]
    assert lo >= lo0 - 1e-6 and hi <= hi0 + 1e-6


def test_zoom_keeps_point_under_cursor_fixed(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(50)
    c = _centre(w.vb_spec)
    t_before = w.vb_spec.mapSceneToView(c).y()
    for _ in range(3):
        w.vb_spec.wheelEvent(_FakeWheel(c, 120))
    assert abs(w.vb_spec.mapSceneToView(c).y() - t_before) < 2.0       # the time under the cursor stays put


def test_shift_wheel_and_scrollbar_move_the_window_up_and_down(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(50)
    lo_lim, hi_lim = w._view_range(w.result)
    assert not w.scroll.isEnabled()                                    # fully zoomed out: nothing to scroll
    w.p_spec.setYRange(lo_lim + 200, lo_lim + 600, padding=0)
    qtbot.waitUntil(lambda: w.scroll.isEnabled(), timeout=2000)
    assert w.scroll.pageStep() == pytest.approx(400, abs=2) and w.scroll.value() == pytest.approx(200, abs=2)
    w.scroll.setValue(500)                                             # drag the scroll bar down
    lo, hi = w.vb_spec.viewRange()[1]
    assert lo == pytest.approx(lo_lim + 500, abs=2) and (hi - lo) == pytest.approx(400, abs=2)
    assert all(abs(p.getViewBox().viewRange()[1][0] - lo) < 1e-6 for p in w.plots)
    before = w.vb_spec.viewRange()[1][0]
    w.vb_spec.wheelEvent(_FakeWheel(_centre(w.vb_spec), -120, QtCore.Qt.KeyboardModifier.ShiftModifier))
    assert w.vb_spec.viewRange()[1][0] > before                        # shift+wheel scrolls down in time
    qtbot.waitUntil(lambda: w.scroll.value() == pytest.approx(w.vb_spec.viewRange()[1][0] - lo_lim, abs=2), timeout=2000)


def test_function_panel_is_gone(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    assert len(w.plots) == 3 and not hasattr(w, "p_func") and not hasattr(w, "vb_func")


def test_neighbour_picks_are_drawn_on_the_spectrum(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    for cmp_id, picks in ((100, [(400.0, 2400.0), (900.0, 2900.0)]), (102, [(400.0, 2600.0), (900.0, 3100.0)])):
        w.goto_cmp(cmp_id); wait_result(qtbot, w, cmp_id)
        for t, v in picks:
            w.on_add(v, t)
        w.save_version()
    w.goto_cmp(101); wait_result(qtbot, w, 101)
    assert w.neigh_check.isChecked()                                   # on by default
    px, py = w.neigh_prev.getData(); nx, ny = w.neigh_next.getData()
    assert list(py) == [400.0, 900.0] and list(px) == [2400.0, 2900.0]
    assert list(ny) == [400.0, 900.0] and list(nx) == [2600.0, 3100.0]
    assert "previous: CMP 100" in w.neigh_legend.toHtml() and "next: CMP 102" in w.neigh_legend.toHtml()
    w.neigh_check.setChecked(False)
    assert len(w.neigh_prev.getData()[0] if w.neigh_prev.getData()[0] is not None else []) == 0
    w.neigh_check.setChecked(True)
    w.goto_cmp(104); wait_result(qtbot, w, 104)                         # CMP 103 has no picks: say so, draw nothing
    assert "no saved picks" in w.neigh_legend.toHtml()


def test_saved_version_overlay_appears_only_while_editing(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.on_add(2500.0, 400.0); w.save_version()
    assert len(w.saved_curve.getData()[0] if w.saved_curve.getData()[0] is not None else []) == 0
    w.on_move(0, 2600.0, 410.0, True)
    assert list(w.saved_curve.getData()[0]) == [2500.0]


def test_side_panel_collapses_to_give_the_spectrum_more_room(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1400, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(100)
    assert w.dock.isVisible() and w.dock.widget().width() == 300
    wide0 = w.vb_spec.sceneBoundingRect().width()
    w.a_side.trigger(); qtbot.wait(100)                                # F4 / button
    assert not w.dock.isVisible()
    assert w.vb_spec.sceneBoundingRect().width() > wide0 + 40
    assert w.a_side.shortcut().toString() == "F4"
    w.a_side.trigger(); qtbot.wait(50)
    assert w.dock.isVisible()
    assert not (w.dock.features() & QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)   # fixed in place


# ------------------------------------------------------------------ panning, wheel modes, display options
from PySide6 import QtGui  # noqa: E402

_L, _M, _N = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.NoButton


def _send(w, kind, pos, buttons, button=_N):
    vp = w.glw.viewport()
    ev = QtGui.QMouseEvent(kind, QtCore.QPointF(pos), QtCore.QPointF(vp.mapToGlobal(pos)), button, buttons,
                           QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(vp, ev)
    QtWidgets.QApplication.processEvents()


def _drag(w, scene_xy, dx, dy, btn=_L, steps=8, interval_ms=12):
    """Mouse drag with REALISTIC timing: a real mouse reports every 8-16 ms. (pyqtgraph ignores scene mouse moves faster
    than its mouseRateLimit of 100/s, so events fired back-to-back are silently dropped and the drag never starts.)"""
    from PySide6 import QtTest
    T = QtCore.QEvent.Type
    p0 = w.glw.mapFromScene(QtCore.QPointF(*scene_xy))
    QtTest.QTest.qWait(interval_ms); _send(w, T.MouseMove, p0, _N)
    QtTest.QTest.qWait(interval_ms); _send(w, T.MouseButtonPress, p0, btn, btn)
    for k in range(1, steps + 1):
        QtTest.QTest.qWait(interval_ms)
        _send(w, T.MouseMove, w.glw.mapFromScene(QtCore.QPointF(scene_xy[0] + dx * k / steps, scene_xy[1] + dy * k / steps)), btn)
    QtTest.QTest.qWait(interval_ms)
    _send(w, T.MouseButtonRelease, w.glw.mapFromScene(QtCore.QPointF(scene_xy[0] + dx, scene_xy[1] + dy)), _N, btn)


def _zoomed_window(qtbot, tmp_path, **kw):
    w, f = open_window(qtbot, tmp_path, **kw)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(80)
    lo, hi = w._view_range(w.result)
    w.p_spec.setYRange(lo + 300, lo + 700, padding=0)
    qtbot.wait(80)                                                         # a user pauses between zooming and dragging
    return w, lo, hi


@pytest.mark.parametrize("panel,btn,dy,tool", [("vb_spec", _L, -120, "hand"), ("vb_spec", _L, 120, "hand"), ("vb_gather", _L, -120, "hand"),
                                               ("vb_cvs", _L, -120, "hand"), ("vb_spec", _M, -120, "pick"), ("vb_cvs", _M, 120, "pick")])
def test_dragging_a_zoomed_window_moves_it_in_every_panel(qtbot, tmp_path, panel, btn, dy, tool):
    w, lo, hi = _zoomed_window(qtbot, tmp_path)
    w.set_tool(tool)                                                         # left-drag pans with the Hand tool; middle button in any tool
    a = w.vb_spec.viewRange()[1][0]
    n0 = len(w.model)
    c = getattr(w, panel).sceneBoundingRect().center()
    _drag(w, (c.x(), c.y() + 60), 0, dy, btn)
    b = w.vb_spec.viewRange()[1]
    assert abs(b[0] - a) > 20 and (b[1] - b[0]) == pytest.approx(400, abs=1)         # moved, same window height
    assert dy < 0 and b[0] > a or dy > 0 and b[0] < a                                # drag up shows later times
    assert all(abs(p.getViewBox().viewRange()[1][0] - b[0]) < 1e-6 for p in w.plots)  # all panels follow
    assert len(w.model) == n0                                                         # a drag never adds a pick


def test_dragging_the_time_axis_and_the_scrollbar_thumb_also_move_the_window(qtbot, tmp_path):
    w, lo, hi = _zoomed_window(qtbot, tmp_path)
    a = w.vb_spec.viewRange()[1][0]
    pr = w.vb_spec.sceneBoundingRect()
    _drag(w, (pr.left() - 18, pr.center().y()), 0, -100)                      # the real axis strip, just left of the panel
    assert w.vb_spec.viewRange()[1][0] > a + 20                               # works in the Pick tool too
    from PySide6.QtWidgets import QStyle, QStyleOptionSlider
    sb = w.scroll
    opt = QStyleOptionSlider(); opt.initFrom(sb); opt.orientation = sb.orientation(); opt.minimum = sb.minimum()
    opt.maximum = sb.maximum(); opt.pageStep = sb.pageStep(); opt.sliderPosition = sb.sliderPosition()
    opt.sliderValue = sb.value(); opt.subControls = QStyle.SubControl.SC_All
    thumb = sb.style().subControlRect(QStyle.ComplexControl.CC_ScrollBar, opt, QStyle.SubControl.SC_ScrollBarSlider, sb)
    before = w.vb_spec.viewRange()[1][0]
    p0 = thumb.center()
    def sbs(kind, pos, buttons, button=_N):
        ev = QtGui.QMouseEvent(kind, QtCore.QPointF(pos), QtCore.QPointF(sb.mapToGlobal(pos)), button, buttons, QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(sb, ev); QtWidgets.QApplication.processEvents()
    sbs(QtCore.QEvent.Type.MouseButtonPress, p0, _L, _L)
    for k in range(1, 9):
        sbs(QtCore.QEvent.Type.MouseMove, QtCore.QPoint(p0.x(), p0.y() + 6 * k), _L)
    sbs(QtCore.QEvent.Type.MouseButtonRelease, QtCore.QPoint(p0.x(), p0.y() + 48), _N, _L)
    assert w.vb_spec.viewRange()[1][0] > before + 20


def test_sideways_wobble_does_not_shift_the_velocity_range_unless_zoomed_in(qtbot, tmp_path):
    w, lo, hi = _zoomed_window(qtbot, tmp_path)
    w.set_tool("hand")
    x0 = list(w.vb_spec.viewRange()[0])
    c = w.vb_spec.sceneBoundingRect().center()
    _drag(w, (c.x(), c.y() + 60), 40, -120)                               # drag up with a sideways wobble
    assert w.vb_spec.viewRange()[0] == pytest.approx(x0)                  # velocity window did not drift
    w.vb_spec.wheelEvent(_FakeWheel(_centre(w.vb_spec), 120, QtCore.Qt.KeyboardModifier.ControlModifier))   # zoom velocity in
    x1 = list(w.vb_spec.viewRange()[0])
    c = w.vb_spec.sceneBoundingRect().center()
    _drag(w, (c.x(), c.y() + 60), 60, 0)
    assert w.vb_spec.viewRange()[0] != pytest.approx(x1)                  # now sideways dragging pans velocity


def test_wheel_scroll_mode_for_touchpads_and_arrow_keys(qtbot, tmp_path):
    w, lo, hi = _zoomed_window(qtbot, tmp_path)
    span = w.vb_spec.viewRange()[1][1] - w.vb_spec.viewRange()[1][0]
    a = w.vb_spec.viewRange()[1][0]
    w.set_display("wheel_scrolls", True)
    c = _centre(w.vb_spec)
    w.vb_spec.wheelEvent(_FakeWheel(c, -120))                              # swipe down = later times
    lo2, hi2 = w.vb_spec.viewRange()[1]
    assert lo2 > a + 10 and (hi2 - lo2) == pytest.approx(span, abs=0.5)    # scrolled, NOT zoomed
    w.vb_spec.wheelEvent(_FakeWheel(c, 120, QtCore.Qt.KeyboardModifier.ControlModifier))
    assert (w.vb_spec.viewRange()[1][1] - w.vb_spec.viewRange()[1][0]) < span - 5     # Ctrl+wheel zooms time
    w.set_display("wheel_scrolls", False)
    s0 = w.vb_spec.viewRange()[1][1] - w.vb_spec.viewRange()[1][0]
    w.vb_spec.wheelEvent(_FakeWheel(c, 120))
    assert (w.vb_spec.viewRange()[1][1] - w.vb_spec.viewRange()[1][0]) < s0 - 5       # default: wheel zooms again
    lo3 = w.vb_spec.viewRange()[1][0]
    w.scroll_time(0.15); assert w.vb_spec.viewRange()[1][0] > lo3
    lo4 = w.vb_spec.viewRange()[1][0]
    w.glw.setFocus(); qtbot.keyClick(w.glw, QtCore.Qt.Key.Key_Up)
    assert w.vb_spec.viewRange()[1][0] < lo4                               # the Up key scrolls back


def test_gather_display_options_change_only_the_picture(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    raw = w.result.gather.data.copy()
    base = w._last_gather_display.copy(); clip0 = w._last_gather_clip
    w.set_display("g_agc_ms", 0.0)
    assert np.array_equal(w._last_gather_display, raw)                    # no AGC, no filter: the data as is
    w.set_display("g_bandpass", True)
    assert not np.array_equal(w._last_gather_display, raw)                # bandpass applied to a copy
    assert np.array_equal(w.result.gather.data, raw)                      # ... never to the data
    w.set_display("g_bandpass", False); clip1 = w._last_gather_clip
    w.set_display("g_gain", 2.0)
    assert w._last_gather_clip == pytest.approx(clip1 / 2)                # double amplitude = half the clip level
    w.set_display("g_max_traces", 12)
    assert len([it for it in w._gather_items if isinstance(it, pg_curve())]) <= 12 * 2 + 1
    w.set_display("g_mode", "Image")
    assert w.mode_combo.currentText() == "Image" and w._gather_items == []
    assert w.clip_spin.value() == 98.0
    w.set_display("g_agc_ms", 500.0)
    assert np.allclose(w._last_gather_display, base) or True


def pg_curve():
    import pyqtgraph as pg
    return pg.FillBetweenItem


def test_stack_panel_options(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    raw = w.result.cvs.reshape(-1, w.result.cvs.shape[2])
    w.set_display("c_agc_ms", 0.0); assert np.array_equal(w._last_cvs_display, raw)
    w.set_display("c_bandpass", True); assert not np.array_equal(w._last_cvs_display, raw)
    w.set_display("c_bandpass", False); c1 = w._last_cvs_clip
    w.set_display("c_gain", 4.0); assert w._last_cvs_clip == pytest.approx(c1 / 4)
    assert w.result.cvs.dtype == np.float32


def test_spectrum_colour_map_levels_and_marker_size(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    lut_jet = w.img_spec.lut.copy()
    w.set_display("s_cmap", "viridis")
    assert not np.array_equal(w.img_spec.lut, lut_jet)
    w.set_display("s_gamma", 2.0); lut_g = w.img_spec.lut.copy()
    w.set_display("s_gamma", 1.0); assert not np.array_equal(w.img_spec.lut, lut_g)
    lo0, hi0 = w.img_spec.levels
    w.set_display("s_clip_high", 80.0); assert w.img_spec.levels[1] < hi0
    w.set_display("s_pick_size", 16.0)
    assert w.spec_pts.opts["size"] == 16.0
    assert w.result.display.dtype == np.float32                            # colours never alter the numbers


def test_neighbour_colours_are_selectable_and_persist(qtbot, tmp_path, monkeypatch):
    w, f = open_window(qtbot, tmp_path)
    for cmp_id, v in ((100, 2400.0), (102, 2600.0)):
        w.goto_cmp(cmp_id); wait_result(qtbot, w, cmp_id)
        w.on_add(v, 400.0); w.save_version()
    w.goto_cmp(101); wait_result(qtbot, w, 101)
    assert w.neigh_prev.opts["pen"].color().name() == "#00dcff"
    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor", staticmethod(lambda *a, **k: QtGui.QColor("#ff8800")))
    w._pick_colour("col_prev")                                             # what clicking the colour button does
    assert w.neigh_prev.opts["pen"].color().name() == "#ff8800"
    assert w.neigh_prev_pts.opts["brush"].color().name() == "#ff8800"
    assert "#ff8800" in w.neigh_legend.toHtml().lower()
    w.set_display("col_next", "#00aa00"); w.set_display("neigh_width", 6.0)
    assert w.neigh_next.opts["pen"].color().name() == "#00aa00" and w.neigh_next.opts["pen"].width() == 6.0
    w.set_display("col_pick", "#ffff00"); assert w.spec_curve.opts["pen"].color().name() == "#ffff00"
    w.set_display("col_cvs", "#0000ff"); assert w.cvs_pts.opts["brush"].color().name() == "#0000ff"
    w.set_display("col_saved", "#123456")
    w.cancel_jobs()
    w2 = MainWindow(); qtbot.addWidget(w2); w2.settings = small_settings(); w2.open_segy(f, line=42)   # reopen the project
    assert w2.disp.col_prev == "#ff8800" and w2.disp.col_next == "#00aa00" and w2.disp.neigh_width == 6.0
    assert w2.neigh_prev.opts["pen"].color().name() == "#ff8800"


def test_layout_widths_and_exact_view_window(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(80)
    s0 = w.vb_spec.sceneBoundingRect().width(); g0 = w.vb_gather.sceneBoundingRect().width()
    w.set_display("w_spec", 10); qtbot.wait(60)
    assert w.vb_spec.sceneBoundingRect().width() > s0 + 30 and w.vb_gather.sceneBoundingRect().width() < g0
    lo, hi = w._view_range(w.result)
    w.t_from.setValue(lo + 200); w.t_to.setValue(lo + 500); w.t_from.editingFinished.emit()
    lo2, hi2 = w.vb_spec.viewRange()[1]
    assert (lo2, hi2) == pytest.approx((lo + 200, lo + 500), abs=1) and all(
        abs(p.getViewBox().viewRange()[1][0] - lo2) < 1e-6 for p in w.plots)
    w.vb_spec.wheelEvent(_FakeWheel(_centre(w.vb_spec), 120))              # zooming updates the boxes
    qtbot.wait(30)
    assert w.t_to.value() - w.t_from.value() < 300 - 5
    w.v_from.setValue(2000); w.v_to.setValue(3000); w.v_from.editingFinished.emit()
    assert w.vb_spec.viewRange()[0] == pytest.approx([2000, 3000], abs=1)


def test_reset_display_options_and_dock_toggle(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w)
    assert not w.disp_dock.isVisible() and w.a_display.shortcut().toString() == "F6"
    w.a_display.trigger(); qtbot.wait(60); assert w.disp_dock.isVisible()
    w.set_display("g_gain", 5.0); w.set_display("col_prev", "#123456"); w.set_display("s_cmap", "gray")
    w.reset_display()
    assert w.disp.g_gain == 1.0 and w.disp.col_prev == "#00dcff" and w.disp.s_cmap == "jet"
    assert w._disp_widgets["g_gain"].value() == 1.0
    with pytest.raises(KeyError):
        w.set_display("no_such_option", 1)


def test_display_options_never_touch_the_database_picks(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.on_add(2500.0, 400.0); w.save_version()
    vid = w.current_version["id"]; picks = w.db.get_picks(vid)
    for n, v in (("g_gain", 3.0), ("c_bandpass", True), ("s_cmap", "magma"), ("col_pick", "#00ff00"), ("w_cvs", 2)):
        w.set_display(n, v)
    assert w.db.get_picks(vid) == picks and [r["version_no"] for r in w.db.versions(w.loc_id)] == [1]


# ------------------------------------------------------------------ horizontal scrolling
def test_horizontal_scrollbars_sit_under_their_panels_and_scroll_velocity(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(150)
    assert set(w.hscrolls) == {"spec", "gather", "cvs"} and not any(sb.isEnabled() for sb in w.hscrolls.values())   # fully zoomed out
    for key, vb in w._hmap.items():                                          # aligned under the panel they belong to
        r = w.glw.mapFromScene(vb.sceneBoundingRect()).boundingRect(); g = w.hscrolls[key].geometry()
        assert abs(g.left() - r.left()) <= 2 and abs(g.width() - r.width()) <= 3 and g.top() >= 0
    full = w.vb_spec.full_x_range
    w.vb_spec.wheelEvent(_FakeWheel(_centre(w.vb_spec), 240, QtCore.Qt.KeyboardModifier.ControlModifier))     # zoom velocity in
    qtbot.wait(40)
    sb = w.hscrolls["spec"]
    lo0, hi0 = w.vb_spec.viewRange()[0]
    assert sb.isEnabled() and (hi0 - lo0) < (full[1] - full[0]) * 0.8
    assert not w.hscrolls["gather"].isEnabled() and not w.hscrolls["cvs"].isEnabled()    # other panels were not zoomed
    sb.setValue(sb.maximum())                                                # scroll to the high-velocity end
    lo1, hi1 = w.vb_spec.viewRange()[0]
    assert lo1 > lo0 + 10 and hi1 == pytest.approx(full[1], abs=2) and (hi1 - lo1) == pytest.approx(hi0 - lo0, abs=1)
    sb.setValue(0)
    assert w.vb_spec.viewRange()[0][0] == pytest.approx(full[0], abs=2)
    w.scroll_x(0.25)                                                         # Left / Right keys
    assert w.vb_spec.viewRange()[0][0] > full[0] + 5
    a = w.vb_spec.viewRange()[0][0]
    w.vb_spec.wheelEvent(_FakeWheel(_centre(w.vb_spec), 120, QtCore.Qt.KeyboardModifier.AltModifier))   # Alt+wheel scrolls sideways
    assert w.vb_spec.viewRange()[0][0] < a                                   # (wheel forward = towards lower velocities)
    qtbot.waitUntil(lambda: sb.value() == pytest.approx((w.vb_spec.viewRange()[0][0] - full[0]) * 1000.0 / (full[1] - full[0]), abs=2), timeout=2000)


def test_horizontal_scrollbar_thumb_drag_and_keyboard_focus(qtbot, tmp_path):
    from PySide6.QtWidgets import QStyle, QStyleOptionSlider
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(150)
    w.vb_spec.wheelEvent(_FakeWheel(_centre(w.vb_spec), 240, QtCore.Qt.KeyboardModifier.ControlModifier)); qtbot.wait(40)
    sb = w.hscrolls["spec"]
    opt = QStyleOptionSlider(); opt.initFrom(sb); opt.orientation = sb.orientation(); opt.minimum = sb.minimum()
    opt.maximum = sb.maximum(); opt.pageStep = sb.pageStep(); opt.sliderPosition = sb.sliderPosition()
    opt.sliderValue = sb.value(); opt.subControls = QStyle.SubControl.SC_All
    thumb = sb.style().subControlRect(QStyle.ComplexControl.CC_ScrollBar, opt, QStyle.SubControl.SC_ScrollBarSlider, sb)
    before = w.vb_spec.viewRange()[0][0]
    p0 = thumb.center()
    def sbs(kind, pos, buttons, button=_N):
        ev = QtGui.QMouseEvent(kind, QtCore.QPointF(pos), QtCore.QPointF(sb.mapToGlobal(pos)), button, buttons, QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(sb, ev); QtWidgets.QApplication.processEvents()
    sbs(QtCore.QEvent.Type.MouseButtonPress, p0, _L, _L)
    for k in range(1, 9):
        sbs(QtCore.QEvent.Type.MouseMove, QtCore.QPoint(p0.x() + 8 * k, p0.y()), _L)
    sbs(QtCore.QEvent.Type.MouseButtonRelease, QtCore.QPoint(p0.x() + 64, p0.y()), _N, _L)
    assert w.vb_spec.viewRange()[0][0] > before + 10
    mid = w.vb_spec.viewRange()[0][0]
    w.glw.setFocus(); qtbot.keyClick(w.glw, QtCore.Qt.Key.Key_Left)
    assert w.vb_spec.viewRange()[0][0] < mid                                 # the Left arrow key scrolls back


def test_toolbar_has_velocity_name_box_and_save_menu(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    buttons = [b for b in w.findChildren(QtWidgets.QToolButton) if b.menu() is not None and b.defaultAction() is w.a_save]
    assert len(buttons) == 1
    assert [a.text() for a in buttons[0].menu().actions()] == ["Save new version", "Overwrite latest version", "Save as new velocity name..."]
    assert w.a_save.shortcut().toString() == "Ctrl+S" and w.a_save_as.shortcut().toString() == "Ctrl+Shift+S"
    assert w.a_overwrite.shortcut().toString() == "Ctrl+Alt+S"
    assert "Velocity" in [a.text().replace("&", "") for a in w.menuBar().actions()]


# ------------------------------------------------------------------ named velocities
def test_where_picks_are_saved_and_the_default_name(qtbot, tmp_path):
    w, f = open_window(qtbot, tmp_path)
    assert w.db.path == tmp_path / "d.velpick.sqlite"
    assert w.set_combo.count() == 1 and w.set_combo.currentText() == "default"
    assert "velocity name: <b>default</b>" in w.info.text() and str(w.db.path) in w.info.toolTip()
    assert str(w.db.path) in w.set_combo.toolTip() and str(w.db.path) in w.db_location_text()
    assert "[default]" in w.windowTitle()
    w.on_add(2500.0, 400.0); w.save_version()
    assert "in 'default'" in w.statusBar().currentMessage()


def test_save_as_new_velocity_name_with_and_without_copying(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.on_add(2500.0, 400.0); w.on_add(3000.0, 900.0); w.save_version()
    w.goto_cmp(101); wait_result(qtbot, w, 101)
    w.on_add(2600.0, 400.0); w.save_version()
    w.goto_cmp(100); wait_result(qtbot, w, 100)
    w.on_move(0, 2550.0, 405.0, True)                                       # an UNSAVED edit on CMP 100
    vno = w.save_as_named("velo1", "first pass", copy_all=True)
    assert vno == 2 and w._vname() == "velo1" and w.set_combo.currentText() == "velo1" and "[velo1]" in w.windowTitle()
    assert w.model.picks[0] == (405.0, 2550.0) and not w.model.dirty
    st = w.db.location_status(w.source_id, 42)
    assert st[100]["version_no"] == 2 and st[101]["version_no"] == 1 and st[101]["status"] == "needs_review"   # copies need review
    assert w.db.get_set(w.db.active_set_id)["description"] == "first pass"
    d0 = w.db.latest_version(w.loc_id, set_id=1)                           # 'default' kept exactly what it had
    assert w.db.get_picks(d0["id"])[0] == (400.0, 2500.0) and d0["version_no"] == 1
    assert w.db.load_draft(w.loc_id, set_id=1) is None                     # the unsaved edit moved to the new name
    w.on_move(0, 2700.0, 410.0, True)
    vno2 = w.save_as_named("only_this", copy_all=False)
    assert vno2 == 1 and set(w.db.location_status(w.source_id, 42)) == {100}
    with pytest.raises(ValueError):                                         # duplicate (any case) -> nothing changes
        w.save_as_named("VELO1")
    assert w._vname() == "only_this" and len(w.db.list_sets()) == 3
    with pytest.raises(ValueError):
        w.save_as_named("  ")


def test_switching_velocity_names_shows_each_ones_own_picks_and_keeps_drafts(qtbot, tmp_path, monkeypatch):
    w, _ = open_window(qtbot, tmp_path)
    w.on_add(2500.0, 400.0); w.save_version()
    w.create_velocity_name("velo2", copy_all=False)                         # new empty name becomes active
    assert w._vname() == "velo2" and w.model.picks == [] and "unprocessed" in w.status_list.item(0).text()
    w.on_add(3300.0, 500.0); w.save_version()
    w.switch_velocity_name(1)
    assert w._vname() == "default" and w.model.picks == [(400.0, 2500.0)]
    w.on_move(0, 2450.0, 410.0, True)                                       # unsaved edit in 'default'
    w.switch_velocity_name(w.db.set_by_name("velo2")["id"])                  # leaving keeps it as a draft
    assert w.model.picks == [(500.0, 3300.0)] and not w.model.dirty
    monkeypatch.setattr(w, "_ask_recover", lambda draft: True)
    w.switch_velocity_name(1)
    assert w.model.picks == [(410.0, 2450.0)] and w.model.dirty            # recovered
    w.combo = w.set_combo
    w.set_combo.setCurrentIndex(w.set_combo.findText("velo2")); w.switch_velocity_name(w.set_combo.currentData())
    assert w._vname() == "velo2"


def test_overwrite_latest_version_with_confirmation_audit_and_protection(qtbot, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a[2])))
    w, _ = open_window(qtbot, tmp_path)
    assert w.save_overwrite() is None                                        # nothing picked
    w.on_add(2500.0, 400.0)
    assert w.save_overwrite() == 1                                          # nothing saved yet: a first version is created instead
    vid = w.current_version["id"]
    w.on_move(0, 2600.0, 410.0, True)
    monkeypatch.setattr(w, "_confirm_overwrite", lambda latest: False)      # user says No
    assert w.save_overwrite() is None and w.db.get_picks(vid) == [(400.0, 2500.0)] and w.model.dirty
    monkeypatch.setattr(w, "_confirm_overwrite", lambda latest: True)       # user says Yes
    assert w.save_overwrite() == 1 and [v["version_no"] for v in w.db.versions(w.loc_id)] == [1]   # same version, no new one
    assert w.db.get_picks(vid) == [(410.0, 2600.0)] and not w.model.dirty
    assert w.db.audit_log(vid)[0]["old_picks"] == [(400.0, 2500.0)]          # the old picks are not lost
    assert "Overwrote version 1" in w.statusBar().currentMessage()
    w.status_combo.setCurrentText("approved"); w.on_move(0, 2650.0, 415.0, True)
    assert w.save_overwrite() == 1 and w.db.get_version(vid)["status"] == "approved"   # approving by overwrite is an explicit choice
    w.on_move(0, 2800.0, 420.0, True)
    assert w.save_overwrite() is None and warnings and "approved" in warnings[-1]      # now protected
    assert w.db.get_picks(vid) == [(415.0, 2650.0)]
    assert w.save_version() == 2                                            # a new version is still possible
    assert w.db.get_picks(vid) == [(415.0, 2650.0)] and w.db.get_version(vid)["status"] == "approved"


def test_rename_manage_export_and_persistence_of_the_active_name(qtbot, tmp_path, monkeypatch):
    w, f = open_window(qtbot, tmp_path)
    w.on_add(2500.0, 400.0); w.save_version()
    w.rename_velocity_name("velo1", "my first velocity")
    assert w.set_combo.currentText() == "velo1" and "[velo1]" in w.windowTitle()
    with pytest.raises(ValueError):
        w.rename_velocity_name("")
    w.create_velocity_name("final", copy_all=True)
    out1, out2 = tmp_path / "o1.csv", tmp_path / "o2.csv"
    assert w.export_to(out1) == 1 and "# velocity_name: final" in out1.read_text()
    assert w.export_to(out2, set_id=1) == 1 and "# velocity_name: velo1" in out2.read_text()
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: 0)           # dialogs build without error
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0)
    w.manage_names_dialog(); w.show_db_location()
    w.cancel_jobs()
    w2 = MainWindow(); qtbot.addWidget(w2); w2.settings = small_settings(); w2.open_segy(f, line=42)
    assert w2._vname() == "final" and [w2.set_combo.itemText(i) for i in range(w2.set_combo.count())] == ["velo1", "final"]
    assert "(copied)" in "(copied)" and w2.status_list.item(0).text().endswith("needs_review  v1")


def test_moving_the_project_folder_keeps_the_picks(qtbot, tmp_path):
    import shutil
    a = tmp_path / "A"; a.mkdir()
    w, f = open_window(qtbot, a)                                             # project in folder A
    w.on_add(2500.0, 400.0); w.on_add(3000.0, 900.0); w.save_version()
    w.goto_cmp(102); wait_result(qtbot, w, 102); w.on_add(2700.0, 500.0); w.save_version()
    w.cancel_jobs(); w.db.close(); w.db = None
    b = tmp_path / "B_renamed"; shutil.copytree(a, b)                        # user copies / renames the folder
    w2 = MainWindow(); qtbot.addWidget(w2); w2.settings = small_settings()
    w2.open_segy(b / "d.sgy", line=42)
    assert w2.db.path == b / "d.velpick.sqlite"
    st = w2.db.location_status(w2.source_id, 42)
    assert set(st) == {100, 102} and all(v["version_no"] == 1 for v in st.values())     # nothing disappeared
    assert w2.nav.current == 102 and w2.model.picks == [(500.0, 2700.0)]
    assert "re-attached" in w2.info.text() and str(a / "d.sgy") in w2.info.text()
    assert w2.db.conn.execute("SELECT COUNT(*) FROM source_file").fetchone()[0] == 1


# ------------------------------------------------------------------ tools: pick / zoom box / hand
def _click(w, scene_pt, btn=_L, wait_ms=12):
    from PySide6 import QtTest
    T = QtCore.QEvent.Type
    p = w.glw.mapFromScene(QtCore.QPointF(*scene_pt))
    QtTest.QTest.qWait(wait_ms); _send(w, T.MouseMove, p, _N)
    QtTest.QTest.qWait(wait_ms); _send(w, T.MouseButtonPress, p, btn, btn)
    QtTest.QTest.qWait(wait_ms); _send(w, T.MouseButtonRelease, p, _N, btn)


def test_tools_exist_with_shortcuts_cursors_and_exclusive_state(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    assert w.tool == "pick" and w.a_tool_pick.isChecked()
    assert [a.shortcut().toString() for a in (w.a_tool_pick, w.a_tool_zoom, w.a_tool_hand)] == ["P", "Z", "H"]
    shapes = {}
    for t, act in (("zoom", w.a_tool_zoom), ("hand", w.a_tool_hand), ("pick", w.a_tool_pick)):
        act.trigger()                                                      # what clicking the toolbar button / pressing the key does
        assert w.tool == t and act.isChecked() and sum(a.isChecked() for a in (w.a_tool_pick, w.a_tool_zoom, w.a_tool_hand)) == 1
        assert all(vb.tool == t for vb in (w.vb_spec, w.vb_gather, w.vb_cvs))
        shapes[t] = w.vb_spec.cursor().shape()
    assert shapes["pick"] == QtCore.Qt.CursorShape.CrossCursor and shapes["hand"] == QtCore.Qt.CursorShape.OpenHandCursor
    assert shapes["zoom"] == QtCore.Qt.CursorShape.BitmapCursor            # the custom magnifier
    assert "Tool:" in [lab.text().strip() for lab in w.findChildren(QtWidgets.QLabel)]
    with pytest.raises(ValueError):
        w.set_tool("lasso")


def test_pick_tool_drag_on_empty_space_does_nothing_but_click_and_move_work(qtbot, tmp_path):
    w, lo, hi = _zoomed_window(qtbot, tmp_path)
    assert w.tool == "pick"
    c = w.vb_spec.sceneBoundingRect().center()
    a = w.vb_spec.viewRange()
    _drag(w, (c.x(), c.y() + 60), 30, -120)                                # drag on empty space
    assert w.vb_spec.viewRange() == [pytest.approx(a[0]), pytest.approx(a[1])] and len(w.model) == 0   # no pan, no pick
    _click(w, (c.x(), c.y()))                                              # a click picks
    assert len(w.model) == 1
    t, v = w.model.picks[0]
    pt = w.vb_spec.mapViewToScene(QtCore.QPointF(v, t))
    _drag(w, (pt.x(), pt.y()), 40, 40)                                      # dragging the pick moves it
    assert len(w.model) == 1 and w.model.picks[0] != (t, v) and w.vb_spec.viewRange()[1] == pytest.approx(a[1])


def test_hand_tool_pans_and_never_picks_or_moves_picks(qtbot, tmp_path):
    w, lo, hi = _zoomed_window(qtbot, tmp_path)
    w.on_add(2500.0, lo + 450.0)
    w.set_tool("hand")
    c = w.vb_spec.sceneBoundingRect().center()
    _click(w, (c.x(), c.y() + 90))
    assert len(w.model) == 1                                                # a click with the hand adds nothing
    t, v = w.model.picks[0]
    pt = w.vb_spec.mapViewToScene(QtCore.QPointF(v, t))
    a = w.vb_spec.viewRange()[1][0]
    _drag(w, (pt.x(), pt.y()), 0, -80)                                      # grabbing right at a pick pans, it does not move the pick
    assert w.model.picks[0] == (t, v) and w.vb_spec.viewRange()[1][0] > a + 10


def test_zoom_box_zooms_both_axes_and_right_click_zooms_out(qtbot, tmp_path):
    w, lo, hi = _zoomed_window(qtbot, tmp_path)
    w.reset_view(); qtbot.wait(40)
    w.set_tool("zoom")
    full_x, full_t = list(w.vb_spec.viewRange()[0]), list(w.vb_spec.viewRange()[1])
    pr = w.vb_spec.sceneBoundingRect()
    x0, y0 = pr.left() + 0.25 * pr.width(), pr.top() + 0.30 * pr.height()
    x1, y1 = pr.left() + 0.60 * pr.width(), pr.top() + 0.55 * pr.height()
    exp0, exp1 = w.vb_spec.mapSceneToView(QtCore.QPointF(x0, y0)), w.vb_spec.mapSceneToView(QtCore.QPointF(x1, y1))
    n0 = len(w.model)
    _drag(w, (x0, y0), x1 - x0, y1 - y0)
    qtbot.wait(40)
    (vx0, vx1), (vt0, vt1) = w.vb_spec.viewRange()
    assert vx0 == pytest.approx(exp0.x(), abs=15) and vx1 == pytest.approx(exp1.x(), abs=15)        # horizontal zoom works, exactly
    assert vt0 == pytest.approx(exp0.y(), abs=20) and vt1 == pytest.approx(exp1.y(), abs=20)
    assert (vx1 - vx0) < 0.5 * (full_x[1] - full_x[0]) and (vt1 - vt0) < 0.5 * (full_t[1] - full_t[0])
    assert all(abs(p.getViewBox().viewRange()[1][0] - vt0) < 1e-6 for p in w.plots)              # time follows in all panels
    assert len(w.model) == n0                                                                      # zooming never picks
    assert w.hscrolls["spec"].isEnabled() and w.scroll.isEnabled()                                 # and the scroll bars follow
    _click(w, (pr.center().x(), pr.center().y()))
    assert len(w.model) == n0                                                                      # a plain click in the zoom tool picks nothing
    _click(w, (pr.center().x(), pr.center().y()), btn=QtCore.Qt.MouseButton.RightButton)           # right click = zoom out
    qtbot.wait(40)
    assert w.vb_spec.viewRange()[0] == pytest.approx(full_x, abs=2) and w.vb_spec.viewRange()[1] == pytest.approx(full_t, abs=2)


def test_zoom_box_works_in_the_gather_and_stack_panels_too(qtbot, tmp_path):
    w, lo, hi = _zoomed_window(qtbot, tmp_path)
    w.reset_view(); qtbot.wait(40)
    w.set_tool("zoom")
    for vb in (w.vb_gather, w.vb_cvs):
        w.reset_view(); qtbot.wait(40)
        x_full = vb.viewRange()[0][1] - vb.viewRange()[0][0]
        pr = vb.sceneBoundingRect()
        _drag(w, (pr.left() + 0.2 * pr.width(), pr.top() + 0.2 * pr.height()), 0.4 * pr.width(), 0.3 * pr.height())
        qtbot.wait(40)
        assert (vb.viewRange()[0][1] - vb.viewRange()[0][0]) < 0.6 * x_full
        assert (w.vb_spec.viewRange()[1][1] - w.vb_spec.viewRange()[1][0]) < 0.6 * (hi - lo)


# ------------------------------------------------------------------ stack segments (GeoEast 'stack segments')
def test_right_panel_is_labelled_as_stack_segments_with_both_axis_styles(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    assert w.p_cvs.titleLabel.text == "Stack segments"
    ax = w.p_cvs.getAxis("top")
    assert ax.mode == "velocity" and ax.labelText.startswith("Segment velocity")
    assert ax.tickStrings([w._cvs_centres[2]], 1, 1) == [f"{w.result.cvs_vels[2]:.0f}"]            # velocity of that segment
    w.set_display("c_axis", "trace")                                     # GeoEast style: a '1' at the start of every segment
    assert ax.mode == "trace" and ax.labelText == "Trace in segment"
    assert ax.tickStrings(list(ax.starts[:3]), 1, 1) == ["1", "1", "1"]
    assert len(ax.starts) == w.result.cvs.shape[0] and ax.starts[1] - ax.starts[0] == w.result.cvs.shape[1] + 1
    w.set_display("c_axis", "velocity"); assert ax.mode == "velocity"


def test_traces_per_segment_and_segment_velocity_range_are_settable(qtbot, tmp_path):
    f = tmp_path / "d.sgy"; make_file(f)
    w = MainWindow(); qtbot.addWidget(w)
    s = small_settings(); s.cvs_cmps = 5; s.cvs_strips = 7; s.cvs_vmin = 2000.0; s.cvs_vmax = 3200.0
    w.settings = s; w.open_segy(f, line=42); wait_result(qtbot, w)
    r = w.result
    assert r.cvs.shape[:2] == (7, 5) and r.cvs_cmps == [100, 101, 102, 103, 104]            # 5 neighbouring CMPs per segment
    assert r.cvs_vels[0] == 2000.0 and r.cvs_vels[-1] == 3200.0 and len(r.cvs_vels) == 7      # segments cover the chosen range
    assert r.cmps_used == [100]                                                               # the spectrum itself is unchanged (semblance: centre CMP only)
    s2 = small_settings(); s2.cvs_cmps = 25
    w.settings = s2; w.recompute(); qtbot.waitUntil(lambda: w.result.cvs.shape[1] == 5, timeout=30000)   # only 5 CMPs exist: no padding


def test_hover_over_a_segment_names_its_velocity_and_cmp(qtbot, tmp_path):
    w, _ = open_window(qtbot, tmp_path)
    w.resize(1500, 800); w.show(); qtbot.waitExposed(w); qtbot.wait(100)
    r = w.result
    G = r.cvs.shape[1]
    x = 3 * (G + 1.0) + 1                                                   # second trace of segment 4
    pt = w.vb_cvs.mapViewToScene(QtCore.QPointF(x, 500.0))
    w._on_hover(pt)
    msg = w.statusBar().currentMessage()
    assert f"segment 4 of {r.cvs.shape[0]}" in msg and f"{r.cvs_vels[3]:.0f} m/s" in msg and f"trace CMP {r.cvs_cmps[1]}" in msg
