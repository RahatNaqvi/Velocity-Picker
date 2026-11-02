"""velpick main window, laid out like the usual processing-system display:
spectrum (pick here) | gather | constant-velocity stacks (pick here too). Neighbouring CMPs' saved picks are drawn on the
spectrum (previous = cyan dashed, next = magenta dashed).

Mouse (spectrum and constant-velocity-stack panels): left click = add pick, left drag on a pick = move,
right click on a pick = delete, left drag elsewhere = pan. Wheel = zoom TIME around the cursor (all panels follow),
Ctrl+wheel = zoom velocity, Shift+wheel or the scroll bar = move up/down in time. Keys: Delete removes the selected pick,
Ctrl+Z / Ctrl+Y undo/redo, PgDown / PgUp next / previous CMP, Ctrl+S saves a NEW version.
Picked velocity type: stacking (NMO) velocity. Time is always ms, velocity m/s.
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from velpick.data_io import SegyError, SegyFile
from velpick.database import STATUSES, ProjectDB
from velpick.export import export_csv
from velpick.geometry import TraceIndex
from velpick.picking import Navigator, PickModel, regular_sequence
from velpick.velocity_analysis import AnalysisCancelled, equalize, nmo_correct, ormsby_bandpass
from .runner import AnalysisResult, run_analysis
from .settings import AnalysisSettings, DisplaySettings
from .widgets import CMAP_NAMES, TOOLS, OffsetAxis, PickViewBox, StripAxis, jet_lut, make_lut, seismic_lut

_Q = QtCore.Qt
STATUS_COLOURS = {"needs_review": "#d98c00", "picked": "#1a6fd1", "reviewed": "#2e9e44", "approved": "#0b6e1f"}


class _Signals(QtCore.QObject):
    done = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)
    progress = QtCore.Signal(int, int, int)


class _Job(QtCore.QRunnable):
    def __init__(self, req: int, fn):
        super().__init__()
        self.req, self.fn, self.cancel = req, fn, False
        self.signals = _Signals()
        self.setAutoDelete(False)

    def run(self):
        try:
            res = self.fn(lambda i, n: self.signals.progress.emit(self.req, i, n), lambda: self.cancel)
            if not self.cancel:
                self.signals.done.emit(self.req, res)
        except AnalysisCancelled:
            pass
        except Exception as e:  # reported to the user, never silently swallowed
            if not self.cancel:
                self.signals.failed.emit(self.req, f"{type(e).__name__}: {e}")


class NameDialog(QtWidgets.QDialog):
    """Ask for a velocity name (+ description). Used for 'save as new name', 'new empty name' and 'rename'."""

    def __init__(self, title, parent=None, name="", description="", copy_text=None, ask_description=True, note=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(440)
        form = QtWidgets.QFormLayout(self)
        if note:
            lab = QtWidgets.QLabel(note); lab.setWordWrap(True); form.addRow(lab)
        self.name = QtWidgets.QLineEdit(name); self.name.setPlaceholderText("e.g. velo1, final_v2")
        form.addRow("Velocity name", self.name)
        self.desc = QtWidgets.QLineEdit(description); self.desc.setPlaceholderText("optional description")
        if ask_description:
            form.addRow("Description", self.desc)
        self.copy = QtWidgets.QCheckBox(copy_text) if copy_text else None
        if self.copy is not None:
            self.copy.setChecked(True)
            form.addRow(self.copy)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self):
        return self.name.text().strip(), self.desc.text().strip(), (self.copy.isChecked() if self.copy is not None else False)


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, s: AnalysisSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis settings")
        form = QtWidgets.QFormLayout(self)
        self.w = {}

        def dspin(name, lo, hi, dec=1, step=1.0, suffix=""):
            b = QtWidgets.QDoubleSpinBox(); b.setRange(lo, hi); b.setDecimals(dec); b.setSingleStep(step)
            b.setSuffix(suffix); b.setValue(getattr(s, name)); self.w[name] = b; return b

        self.method = QtWidgets.QComboBox(); self.method.addItems(["stackpower", "semblance"])
        self.method.setCurrentText(s.method)
        form.addRow("Method", self.method)
        form.addRow("Velocity min", dspin("vmin", 100, 20000, 0, 100, " m/s"))
        form.addRow("Velocity max", dspin("vmax", 100, 20000, 0, 100, " m/s"))
        form.addRow("Velocity step", dspin("dv", 1, 1000, 0, 5, " m/s"))
        form.addRow("t0 start", dspin("t0_start", 1, 60000, 0, 50, " ms"))
        form.addRow("t0 end", dspin("t0_end", 10, 60000, 0, 100, " ms"))
        form.addRow("Spectrum line spacing", dspin("line_interval", 1, 200, 1, 5, " ms"))
        self.n_cmps = QtWidgets.QSpinBox(); self.n_cmps.setRange(1, 21); self.n_cmps.setValue(s.n_cmps)
        form.addRow("CMPs per spectrum (stack power)", self.n_cmps)
        form.addRow("NMO stretch mute (stack power)", dspin("stretch", 1, 500, 0, 5, " %"))
        self.filter2d = QtWidgets.QCheckBox(); self.filter2d.setChecked(s.filter2d)
        form.addRow("2D smoothing of spectrum", self.filter2d)
        self.bandpass = QtWidgets.QCheckBox(); self.bandpass.setChecked(s.bandpass)
        form.addRow("Bandpass (Ormsby)", self.bandpass)
        for n, lab in (("f1", "  low cut"), ("f2", "  low pass"), ("f3", "  high pass"), ("f4", "  high cut")):
            form.addRow(lab, dspin(n, 0, 500, 1, 1, " Hz"))
        form.addRow("Equalisation window (0 = off)", dspin("eq_ms", 0, 1000, 0, 50, " ms"))
        form.addRow("Semblance gate", dspin("window_ms", 4, 500, 0, 10, " ms"))
        form.addRow("NMO stretch mute (semblance)", dspin("sem_stretch", 1, 500, 0, 5, " %"))
        self.row_norm = QtWidgets.QCheckBox(); self.row_norm.setChecked(s.row_norm)
        form.addRow("Normalise each time line (display)", self.row_norm)
        form.addRow("Maximum offset used (0 = all)", dspin("max_offset", 0, 100000, 0, 500, " m"))
        self.cvs_enabled = QtWidgets.QCheckBox(); self.cvs_enabled.setChecked(s.cvs_enabled)
        form.addRow("Stack-segment panel (right-hand panel)", self.cvs_enabled)
        self.cvs_strips = QtWidgets.QSpinBox(); self.cvs_strips.setRange(2, 61); self.cvs_strips.setValue(s.cvs_strips)
        form.addRow("  number of segments (constant velocities)", self.cvs_strips)
        self.cvs_cmps = QtWidgets.QSpinBox(); self.cvs_cmps.setRange(1, 25); self.cvs_cmps.setValue(s.cvs_cmps)
        form.addRow("  traces per segment (neighbouring CMPs)", self.cvs_cmps)
        form.addRow("  first segment velocity (0 = analysis min)", dspin("cvs_vmin", 0, 30000, 0, 100, " m/s"))
        form.addRow("  last segment velocity (0 = analysis max)", dspin("cvs_vmax", 0, 30000, 0, 100, " m/s"))
        form.addRow("  stack time sampling", dspin("cvs_dt_ms", 2, 100, 0, 2, " ms"))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                        QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> AnalysisSettings:
        d = {k: w.value() for k, w in self.w.items()}
        d.update(method=self.method.currentText(), n_cmps=self.n_cmps.value(), filter2d=self.filter2d.isChecked(),
                 bandpass=self.bandpass.isChecked(), row_norm=self.row_norm.isChecked(),
                 cvs_enabled=self.cvs_enabled.isChecked(), cvs_strips=self.cvs_strips.value(), cvs_cmps=self.cvs_cmps.value())
        return AnalysisSettings.from_dict(d)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.sf: Optional[SegyFile] = None
        self.idx: Optional[TraceIndex] = None
        self.db: Optional[ProjectDB] = None
        self.source_id: Optional[int] = None
        self.line = 0
        self.nav: Optional[Navigator] = None
        self.model = PickModel()
        self.settings = AnalysisSettings()
        self.disp = DisplaySettings()
        self._moved_note = None
        self.tool = "pick"
        self._disp_widgets = {}
        self._view_busy = False
        self._last_gather_display = None
        self._last_cvs_display = None
        self.result: Optional[AnalysisResult] = None
        self.cache: "OrderedDict[tuple, AnalysisResult]" = OrderedDict()
        self.loc_id: Optional[int] = None
        self.current_version = None
        self.sel: Optional[int] = None
        self._req = 0
        self._jobs = {}
        self._dragging = False
        self._need_range = True
        self._gather_items: list = []
        self._cvs_items: list = []
        self._neigh_items: list = []
        self._cvs_centres = None
        self._cvs_vels = None
        self._build_ui()
        self._build_actions()
        self.set_tool("pick")
        self.pool = QtCore.QThreadPool.globalInstance()
        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.setInterval(30000)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()
        self._refresh_picks()

    # ================================================================== UI construction
    def _make_plot(self, vb, axis_items=None):
        p = pg.PlotItem(viewBox=vb, axisItems=axis_items or {})
        p.invertY(True)
        p.showGrid(x=True, y=True, alpha=0.25)
        p.hideButtons()                                   # no 'A' auto-range button: ranges are controlled explicitly
        vb.disableAutoRange()
        p.showAxis("top"); p.hideAxis("bottom")           # velocity / offset axes on top, like the processing systems
        for ax in ("left", "top"):
            p.getAxis(ax).enableAutoSIPrefix(False)       # always ms and m/s, never 'kms' or 'km/s'
        return p

    def _build_ui(self):
        self.setWindowTitle("velpick - velocity picking workstation")
        self.resize(1700, 920)
        self.glw = pg.GraphicsLayoutWidget()
        self.vb_spec, self.vb_gather, self.vb_cvs = (PickViewBox() for _ in range(3))
        self.p_spec = self._make_plot(self.vb_spec)
        self.p_gather = self._make_plot(self.vb_gather, {"top": OffsetAxis("top")})
        self.p_cvs = self._make_plot(self.vb_cvs, {"top": StripAxis("top")})
        self.plots = [self.p_spec, self.p_gather, self.p_cvs]
        for c, p in enumerate(self.plots):
            self.glw.addItem(p, row=0, col=c)
        for c, w in enumerate((5, 5, 5)):
            self.glw.ci.layout.setColumnStretchFactor(c, w)
        self.vb_gather.picking_enabled = False
        for p in (self.p_gather, self.p_cvs):
            p.setYLink(self.p_spec)
        self.p_spec.setLabel("left", "Time", "ms")
        self.p_spec.setLabel("top", "Velocity", "m/s")
        self.p_gather.setLabel("top", "Offset", "m")
        self.p_cvs.setLabel("top", "Segment velocity", "m/s")
        for p in (self.p_gather, self.p_cvs):
            p.hideAxis("left")
        self.p_spec.setTitle("Spectrum", size="9pt")
        self.p_gather.setTitle("Gather", size="9pt")
        self.p_cvs.setTitle("Stack segments", size="9pt")
        for vb in (self.vb_spec, self.vb_gather, self.vb_cvs):
            vb.sigResetView.connect(self.reset_view)
        self.vb_spec.hit_test = lambda sp: self._hit(self.vb_spec, sp, self._xy_spec)
        self.vb_cvs.hit_test = lambda sp: self._hit(self.vb_cvs, sp, self._xy_cvs)
        self.vb_spec.sigAdd.connect(self.on_add); self.vb_spec.sigSelect.connect(self.on_select)
        self.vb_spec.sigDelete.connect(self.on_delete); self.vb_spec.sigMove.connect(self.on_move)
        self.vb_cvs.sigAdd.connect(lambda x, t: self.on_add(self._cvs_x_to_v(x), t))
        self.vb_cvs.sigSelect.connect(self.on_select); self.vb_cvs.sigDelete.connect(self.on_delete)
        self.vb_cvs.sigMove.connect(lambda i, x, t, fin: self.on_move(i, self._cvs_x_to_v(x), t, fin))

        # ---- central widget: panels + vertical time scroll bar (scroll the time window up/down)
        centre = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(centre); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        hl = QtWidgets.QHBoxLayout(); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(0)
        self.scroll = QtWidgets.QScrollBar(_Q.Orientation.Vertical)
        self.scroll.setToolTip("Scroll the time window (mouse wheel zooms time, Shift+wheel scrolls)")
        self.scroll.valueChanged.connect(self._on_scroll)
        self._sb_busy = False
        hl.addWidget(self.glw, 1); hl.addWidget(self.scroll, 0)
        outer.addLayout(hl, 1)
        # horizontal scroll bars, one under each panel (velocity / offset / strips), kept aligned with the panels
        row2 = QtWidgets.QHBoxLayout(); row2.setContentsMargins(0, 0, 0, 0); row2.setSpacing(0)
        self.hbar_box = QtWidgets.QWidget(); self.hbar_box.setFixedHeight(16)
        row2.addWidget(self.hbar_box, 1); row2.addSpacing(self.scroll.sizeHint().width())
        outer.addLayout(row2, 0)
        self._hmap = {"spec": self.vb_spec, "gather": self.vb_gather, "cvs": self.vb_cvs}
        self.hscrolls, self._hbusy = {}, {}
        tips = {"spec": "Scroll the velocity window (Ctrl+wheel zooms velocity)", "gather": "Scroll the offset window",
                "cvs": "Scroll the strips"}
        for key, vb in self._hmap.items():
            sb = QtWidgets.QScrollBar(_Q.Orientation.Horizontal, self.hbar_box)
            sb.setToolTip(tips[key]); sb.setEnabled(False)
            sb.valueChanged.connect(lambda v, k=key: self._on_hscroll(k, v))
            self.hscrolls[key], self._hbusy[key] = sb, False
            vb.sigXRangeChanged.connect(lambda *_, k=key: self._sync_hscroll(k))
        self.glw.installEventFilter(self)
        self.setCentralWidget(centre)
        self.vb_spec.sigYRangeChanged.connect(lambda *_: self._sync_scrollbar())
        self.vb_spec.sigRangeChanged.connect(lambda *_: self._place_legend() if self.neigh_legend.scene() is not None else None)

        self.img_gather = pg.ImageItem(); self.img_gather.setLookupTable(seismic_lut()); self.img_gather.setZValue(-10)
        self.p_gather.addItem(self.img_gather)
        self.img_spec = pg.ImageItem(); self.img_spec.setLookupTable(jet_lut()); self.img_spec.setZValue(-10)
        self.p_spec.addItem(self.img_spec)
        pen_w = pg.mkPen("w", width=2)
        red = pg.mkBrush((230, 30, 30))
        self.spec_curve = pg.PlotCurveItem(pen=pen_w); self.spec_curve.setZValue(5)
        self.spec_pts = pg.ScatterPlotItem(size=10, pen=pg.mkPen("k"), brush=pg.mkBrush("w")); self.spec_pts.setZValue(10)
        self.sel_spec = pg.ScatterPlotItem(size=16, pen=pg.mkPen((255, 200, 0), width=3), brush=pg.mkBrush(None)); self.sel_spec.setZValue(11)
        # neighbouring CMPs' saved picks and the last saved version of this CMP, drawn on the spectrum
        self.neigh_prev = pg.PlotCurveItem(pen=pg.mkPen((0, 220, 255), width=4, style=_Q.PenStyle.DashLine)); self.neigh_prev.setZValue(4)
        self.neigh_prev_pts = pg.ScatterPlotItem(size=9, symbol="o", pen=pg.mkPen("k"), brush=pg.mkBrush((0, 220, 255))); self.neigh_prev_pts.setZValue(6)
        self.neigh_next = pg.PlotCurveItem(pen=pg.mkPen((255, 0, 255), width=2, style=_Q.PenStyle.DashLine)); self.neigh_next.setZValue(4)
        self.neigh_next_pts = pg.ScatterPlotItem(size=9, symbol="s", pen=pg.mkPen("k"), brush=pg.mkBrush((255, 0, 255))); self.neigh_next_pts.setZValue(6)
        self.saved_curve = pg.PlotCurveItem(pen=pg.mkPen((255, 255, 255, 170), width=1.5, style=_Q.PenStyle.DotLine)); self.saved_curve.setZValue(4)
        self.neigh_legend = pg.TextItem("", anchor=(0, 1), color=(255, 255, 255), fill=pg.mkBrush(0, 0, 0, 190)); self.neigh_legend.setZValue(20)
        self.cvs_curve = pg.PlotCurveItem(pen=pg.mkPen((230, 30, 30, 160), width=1)); self.cvs_curve.setZValue(5)
        self.cvs_pts = pg.ScatterPlotItem(size=12, symbol="t", pen=pg.mkPen("k"), brush=red); self.cvs_pts.setZValue(10)
        self.sel_cvs = pg.ScatterPlotItem(size=18, symbol="t", pen=pg.mkPen((255, 140, 0), width=2), brush=pg.mkBrush(None)); self.sel_cvs.setZValue(11)
        for it in (self.neigh_prev, self.neigh_next, self.neigh_prev_pts, self.neigh_next_pts, self.saved_curve, self.spec_curve, self.spec_pts, self.sel_spec):
            self.p_spec.addItem(it)
        for it in (self.cvs_curve, self.cvs_pts, self.sel_cvs):
            self.p_cvs.addItem(it)
        dash = pg.mkPen((210, 120, 0, 150), width=1, style=_Q.PenStyle.DashLine)
        cur = pg.mkPen((120, 120, 120, 160), width=1)
        self.selline = [pg.InfiniteLine(angle=0, movable=False, pen=dash) for _ in self.plots]
        self.hline = [pg.InfiniteLine(angle=0, movable=False, pen=cur) for _ in self.plots]
        self.vline = {id(self.p_spec): pg.InfiniteLine(angle=90, movable=False, pen=cur)}
        for p, a, h in zip(self.plots, self.selline, self.hline):
            p.addItem(a, ignoreBounds=True); p.addItem(h, ignoreBounds=True)
            a.setVisible(False)
        self.p_spec.addItem(self.vline[id(self.p_spec)], ignoreBounds=True)
        self.glw.scene().sigMouseMoved.connect(self._on_hover)

        # ---- left side panel: fixed width, collapsible (F4) so the spectrum gets the room
        self.dock = QtWidgets.QDockWidget("Locations / picks", self)
        # Closable (so the View action / F4 can hide it) but not movable or floatable; no title bar = no stray buttons
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.dock.setTitleBarWidget(QtWidgets.QWidget())
        box = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(box)
        box.setFixedWidth(300)
        self.info = QtWidgets.QLabel("Open a SEG-Y file (File > Open SEG-Y...)"); self.info.setWordWrap(True)
        self.status_list = QtWidgets.QListWidget(); self.status_list.setMaximumHeight(170)
        self.status_list.itemClicked.connect(lambda it: self.goto_cmp(it.data(_Q.ItemDataRole.UserRole)))
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Time (ms)", "Velocity (m/s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellChanged.connect(self._on_cell_changed)
        self.table.itemSelectionChanged.connect(self._on_table_select)
        self.warn = QtWidgets.QLabel(""); self.warn.setWordWrap(True); self.warn.setStyleSheet("color:#b35900")
        self.note_edit = QtWidgets.QLineEdit(); self.note_edit.setPlaceholderText("Review note (saved with the version)")
        for w in (self.info, QtWidgets.QLabel("Analysis locations"), self.status_list, QtWidgets.QLabel("Picks (editable)"),
                  self.table, self.warn, self.note_edit):
            lay.addWidget(w)
        self.dock.setWidget(box)
        self.addDockWidget(_Q.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.progress = QtWidgets.QProgressBar(); self.progress.setMaximumWidth(180); self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self._build_display_dock()
        # Up / Down arrows scroll the time window while the plots have the focus
        self.glw.setFocusPolicy(_Q.FocusPolicy.ClickFocus)
        for key, frac in (("Up", -0.15), ("Down", 0.15)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.glw)
            sc.setContext(_Q.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda f=frac: self.scroll_time(f))
        for key, frac in (("Left", -0.15), ("Right", 0.15)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self.glw)
            sc.setContext(_Q.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda f=frac: self.scroll_x(f))
        self.vb_spec.sigRangeChanged.connect(lambda *_: self._sync_view_spins())
        self._apply_colours(); self._apply_layout(); self._apply_wheel(); self._apply_spectrum_style()

    def set_tool(self, name: str):
        """Pick / Zoom box / Hand. Clicks and drags mean exactly one thing in each tool."""
        if name not in TOOLS:
            raise ValueError(name)
        self.tool = name
        for vb in (self.vb_spec, self.vb_gather, self.vb_cvs):
            vb.set_tool(name)
        {"pick": self.a_tool_pick, "zoom": self.a_tool_zoom, "hand": self.a_tool_hand}[name].setChecked(True)
        hint = {"pick": "Pick tool: click = add pick, drag a pick = move it, right-click a pick = delete it.",
                "zoom": "Zoom tool: drag a rectangle to zoom in; right-click zooms out.",
                "hand": "Hand tool: drag to move the view (nothing is picked)."}[name]
        self.statusBar().showMessage(hint, 8000)

    def scroll_time(self, fraction: float):
        """Move the shared time window up (negative) or down (positive) by a fraction of its height."""
        self.vb_spec.scroll_time(fraction)

    def scroll_x(self, fraction: float):
        """Move the horizontal window (velocity / offset / strips) of every panel by a fraction of its width."""
        for vb in self._hmap.values():
            vb.scroll_x(fraction)

    # ---- horizontal scroll bars
    def eventFilter(self, obj, ev):
        if obj is self.glw and ev.type() == QtCore.QEvent.Type.Resize:
            QtCore.QTimer.singleShot(0, self._place_hscrolls)
        return super().eventFilter(obj, ev)

    def _place_hscrolls(self):
        """Put each horizontal scroll bar exactly under its panel."""
        h = self.hbar_box.height()
        for key, vb in self._hmap.items():
            r = self.glw.mapFromScene(vb.sceneBoundingRect()).boundingRect()
            self.hscrolls[key].setGeometry(int(r.left()), 0, max(20, int(r.width())), h)

    def _sync_hscroll(self, key):
        vb, sb = self._hmap[key], self.hscrolls[key]
        if self._hbusy[key] or vb.full_x_range is None:
            return
        f0, f1 = vb.full_x_range
        lo, hi = vb.viewRange()[0]
        total, page = f1 - f0, hi - lo
        if total <= 0:
            return
        k = 1000.0 / total
        self._hbusy[key] = True
        try:
            sb.setRange(0, max(0, int(round((total - page) * k))))
            sb.setPageStep(max(1, int(round(page * k))))
            sb.setValue(int(round((lo - f0) * k)))
            sb.setEnabled(bool(total - page > 0.005 * total))
        finally:
            self._hbusy[key] = False

    def _on_hscroll(self, key, value):
        vb = self._hmap[key]
        if self._hbusy[key] or vb.full_x_range is None:
            return
        f0, f1 = vb.full_x_range
        lo, hi = vb.viewRange()[0]
        new_lo = f0 + value * (f1 - f0) / 1000.0
        vb.setXRange(new_lo, new_lo + (hi - lo), padding=0)

    # ================================================================== display options (look only; data is never changed)
    def _build_display_dock(self):
        self.disp_dock = QtWidgets.QDockWidget("Display options", self)
        self.disp_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.disp_dock.setTitleBarWidget(QtWidgets.QWidget())
        area = QtWidgets.QScrollArea(); area.setWidgetResizable(True); area.setFixedWidth(330)
        area.setHorizontalScrollBarPolicy(_Q.ScrollBarPolicy.ScrollBarAlwaysOff)
        box = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(box)
        v.addWidget(QtWidgets.QLabel("<b>Display options</b> (look only - data and picks are never changed; saved with the project)"))
        v.itemAt(0).widget().setWordWrap(True)

        def group(title):
            g = QtWidgets.QGroupBox(title); f = QtWidgets.QFormLayout(g); v.addWidget(g); return f

        def dspin(name, lo, hi, dec=1, step=1.0, suffix=""):
            b = QtWidgets.QDoubleSpinBox(); b.setRange(lo, hi); b.setDecimals(dec); b.setSingleStep(step); b.setSuffix(suffix)
            b.setKeyboardTracking(False); b.setValue(getattr(self.disp, name))
            b.valueChanged.connect(lambda val, n=name: self.set_display(n, float(val))); self._disp_widgets[name] = b; return b

        def ispin(name, lo, hi, step=1):
            b = QtWidgets.QSpinBox(); b.setRange(lo, hi); b.setSingleStep(step); b.setKeyboardTracking(False); b.setValue(getattr(self.disp, name))
            b.valueChanged.connect(lambda val, n=name: self.set_display(n, int(val))); self._disp_widgets[name] = b; return b

        def check(name):
            b = QtWidgets.QCheckBox(); b.setChecked(getattr(self.disp, name))
            b.toggled.connect(lambda val, n=name: self.set_display(n, bool(val))); self._disp_widgets[name] = b; return b

        def combo(name, items):
            b = QtWidgets.QComboBox(); b.addItems(items); b.setCurrentText(getattr(self.disp, name))
            b.currentTextChanged.connect(lambda val, n=name: self.set_display(n, val)); self._disp_widgets[name] = b; return b

        def colour(name):
            b = QtWidgets.QPushButton(); b.setFixedHeight(22)
            b.clicked.connect(lambda _=False, n=name: self._pick_colour(n)); self._disp_widgets[name] = b; return b

        f = group("Gather")
        f.addRow("Bandpass (display)", check("g_bandpass"))
        for n, lab in (("g_f1", "  low cut"), ("g_f2", "  low pass"), ("g_f3", "  high pass"), ("g_f4", "  high cut")):
            f.addRow(lab, dspin(n, 0, 500, 1, 1, " Hz"))
        f.addRow("AGC window (0 = off)", dspin("g_agc_ms", 0, 2000, 0, 50, " ms"))
        f.addRow("Amplitude (gain)", dspin("g_gain", 0.05, 50, 2, 0.25, " x"))
        f.addRow("Clip", dspin("g_clip", 50, 100, 1, 0.5, " %"))
        self.clip_spin = self._disp_widgets["g_clip"]
        f.addRow("Wiggle traces shown", ispin("g_max_traces", 10, 600, 10))
        f = group("Stack segments")
        f.addRow("Top axis shows", combo("c_axis", ["velocity", "trace"]))
        f.addRow("Bandpass (display)", check("c_bandpass"))
        for n, lab in (("c_f1", "  low cut"), ("c_f2", "  low pass"), ("c_f3", "  high pass"), ("c_f4", "  high cut")):
            f.addRow(lab, dspin(n, 0, 500, 1, 1, " Hz"))
        f.addRow("AGC window (0 = off)", dspin("c_agc_ms", 0, 2000, 0, 50, " ms"))
        f.addRow("Amplitude (gain)", dspin("c_gain", 0.05, 50, 2, 0.25, " x"))
        f.addRow("Clip", dspin("c_clip", 50, 100, 1, 0.5, " %"))
        f = group("Spectrum")
        f.addRow("Colour map", combo("s_cmap", CMAP_NAMES))
        f.addRow("Lowest colour at", dspin("s_clip_low", 0, 50, 1, 1, " %"))
        f.addRow("Highest colour at", dspin("s_clip_high", 50, 100, 1, 0.5, " %"))
        f.addRow("Contrast (gamma)", dspin("s_gamma", 0.2, 5, 2, 0.1))
        f.addRow("Pick marker size", dspin("s_pick_size", 4, 30, 0, 1, " px"))
        f = group("Colours")
        for n, lab in (("col_pick", "Current picks"), ("col_prev", "Previous CMP picks"), ("col_next", "Next CMP picks"),
                       ("col_saved", "Last saved version"), ("col_cvs", "Stack triangles")):
            f.addRow(lab, colour(n))
        f.addRow("Neighbour line width", dspin("neigh_width", 1, 10, 1, 0.5, " px"))
        f = group("Layout and view")
        for n, lab in (("w_spec", "Spectrum width"), ("w_gather", "Gather width"), ("w_cvs", "Stacks width")):
            f.addRow(lab, ispin(n, 1, 20))
        f.addRow("Wheel scrolls time", check("wheel_scrolls"))
        f.addRow(QtWidgets.QLabel("(on: wheel scrolls, Ctrl+wheel zooms - for touch-pads)"))
        self.t_from = QtWidgets.QDoubleSpinBox(); self.t_to = QtWidgets.QDoubleSpinBox()
        self.v_from = QtWidgets.QDoubleSpinBox(); self.v_to = QtWidgets.QDoubleSpinBox()
        for b, suf, hi in ((self.t_from, " ms", 60000), (self.t_to, " ms", 60000), (self.v_from, " m/s", 30000), (self.v_to, " m/s", 30000)):
            b.setRange(0, hi); b.setDecimals(0); b.setSingleStep(50); b.setSuffix(suf); b.setKeyboardTracking(False)
            b.editingFinished.connect(self._apply_view_spins)
        f.addRow("Time window from", self.t_from); f.addRow("  to", self.t_to)
        f.addRow("Velocity window from", self.v_from); f.addRow("  to", self.v_to)
        reset = QtWidgets.QPushButton("Reset display options to defaults")
        reset.clicked.connect(self.reset_display)
        v.addWidget(reset); v.addStretch(1)
        area.setWidget(box)
        self.disp_dock.setWidget(area)
        self.addDockWidget(_Q.DockWidgetArea.RightDockWidgetArea, self.disp_dock)
        self.disp_dock.hide()
        self._sync_display_widgets()

    def _sync_display_widgets(self):
        """Show the current DisplaySettings in the widgets (without triggering changes)."""
        for name, wdg in self._disp_widgets.items():
            val = getattr(self.disp, name)
            wdg.blockSignals(True)
            if isinstance(wdg, QtWidgets.QComboBox):
                wdg.setCurrentText(val)
            elif isinstance(wdg, QtWidgets.QCheckBox):
                wdg.setChecked(bool(val))
            elif isinstance(wdg, QtWidgets.QPushButton):
                wdg.setStyleSheet(f"background-color: {val}; border: 1px solid #555;"); wdg.setText(val)
            else:
                wdg.setValue(val)
            wdg.blockSignals(False)

    def set_display(self, name: str, value, persist: bool = True):
        """Change one display option (from the dock, or from code) and redraw what it affects."""
        if not hasattr(self.disp, name):
            raise KeyError(name)
        setattr(self.disp, name, value)
        self._sync_display_widgets()
        if name.startswith("g_"):
            self._draw_gather()
        elif name.startswith("c_"):
            self._draw_cvs()
        elif name.startswith("s_"):
            self._apply_spectrum_style(); self._draw_spectrum()
        elif name.startswith("col_") or name == "neigh_width":
            self._apply_colours(); self._draw_neighbours()
        elif name.startswith("w_"):
            self._apply_layout()
        elif name == "wheel_scrolls":
            self._apply_wheel()
        if persist and self.db is not None:
            self.db.set_meta("display_settings", self.disp.to_dict())

    def reset_display(self):
        self.disp = DisplaySettings()
        for n in ("g_mode", "col_pick"):
            pass
        self._sync_display_widgets()
        for fn in (self._apply_colours, self._apply_layout, self._apply_wheel, self._apply_spectrum_style,
                   self._draw_gather, self._draw_spectrum, self._draw_cvs, self._draw_neighbours):
            fn()
        if self.db is not None:
            self.db.set_meta("display_settings", self.disp.to_dict())

    def _pick_colour(self, name: str):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(getattr(self.disp, name)), self, "Choose colour")
        if c.isValid():
            self.set_display(name, c.name())

    def _apply_colours(self):
        d = self.disp
        qc = lambda hexs, a=255: QtGui.QColor(hexs).toRgb() if a == 255 else (lambda c: (c.setAlpha(a), c)[1])(QtGui.QColor(hexs))  # noqa: E731
        self.spec_curve.setPen(pg.mkPen(d.col_pick, width=2)); self.spec_pts.setBrush(pg.mkBrush(d.col_pick))
        self.saved_curve.setPen(pg.mkPen(qc(d.col_saved, 190), width=1.5, style=_Q.PenStyle.DotLine))
        self.neigh_prev.setPen(pg.mkPen(d.col_prev, width=d.neigh_width + 1, style=_Q.PenStyle.DashLine))
        self.neigh_next.setPen(pg.mkPen(d.col_next, width=d.neigh_width, style=_Q.PenStyle.DashLine))
        self.neigh_prev_pts.setBrush(pg.mkBrush(d.col_prev)); self.neigh_next_pts.setBrush(pg.mkBrush(d.col_next))
        self.cvs_curve.setPen(pg.mkPen(qc(d.col_cvs, 160), width=1)); self.cvs_pts.setBrush(pg.mkBrush(d.col_cvs))

    def _apply_spectrum_style(self):
        self.spec_pts.setSize(self.disp.s_pick_size); self.sel_spec.setSize(self.disp.s_pick_size + 6)

    def _apply_layout(self):
        for c, n in enumerate(("w_spec", "w_gather", "w_cvs")):
            self.glw.ci.layout.setColumnStretchFactor(c, max(1, int(getattr(self.disp, n))))
        QtCore.QTimer.singleShot(0, self._place_hscrolls)

    def _apply_wheel(self):
        for vb in (self.vb_spec, self.vb_gather, self.vb_cvs):
            vb.wheel_scrolls = bool(self.disp.wheel_scrolls)

    def _sync_view_spins(self):
        if self._view_busy or not hasattr(self, "t_from"):
            return
        self._view_busy = True
        try:
            (x0, x1), (y0, y1) = self.vb_spec.viewRange()
            for b, val in ((self.t_from, y0), (self.t_to, y1), (self.v_from, x0), (self.v_to, x1)):
                b.blockSignals(True); b.setValue(val); b.blockSignals(False)
        finally:
            self._view_busy = False

    def _apply_view_spins(self):
        """Typed time / velocity window -> the panels (clamped to the analysed range by the view limits)."""
        if self.result is None or self._view_busy:
            return
        t0, t1, v0, v1 = self.t_from.value(), self.t_to.value(), self.v_from.value(), self.v_to.value()
        if t1 - t0 >= 10:
            self.p_spec.setYRange(t0, t1, padding=0)
        if v1 - v0 >= 10:
            self.vb_spec.setXRange(v0, v1, padding=0)
        self._sync_view_spins()

    # ---- time scroll bar (kept in sync with the shared time axis)
    def _sync_scrollbar(self):
        if self._sb_busy or self.result is None:
            return
        lo_lim, hi_lim = self._view_range(self.result)
        lo, hi = self.vb_spec.viewRange()[1]
        total, page = hi_lim - lo_lim, hi - lo
        self._sb_busy = True
        try:
            self.scroll.setRange(0, max(0, int(round(total - page))))
            self.scroll.setPageStep(max(1, int(round(page))))
            self.scroll.setValue(int(round(lo - lo_lim)))
            self.scroll.setEnabled(bool(total - page > 1))
        finally:
            self._sb_busy = False

    def _on_scroll(self, value: int):
        if self._sb_busy or self.result is None:
            return
        lo_lim, _ = self._view_range(self.result)
        lo, hi = self.vb_spec.viewRange()[1]
        self.vb_spec.setYRange(lo_lim + value, lo_lim + value + (hi - lo), padding=0)

    def _act(self, text, slot, shortcut=None, tip=None):
        a = QtGui.QAction(text, self)
        a.triggered.connect(lambda checked=False: slot())
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
        return a

    def _build_actions(self):
        self.a_open = self._act("Open SEG-Y...", self.open_dialog, "Ctrl+O")
        self.a_export = self._act("Export CSV...", self.export_dialog, "Ctrl+E")
        self.a_quit = self._act("Quit", self.close, "Ctrl+Q")
        self.a_undo = self._act("Undo", self.undo, "Ctrl+Z"); self.a_redo = self._act("Redo", self.redo, "Ctrl+Y")
        self.a_del = self._act("Delete selected pick", self.delete_selected, "Delete")
        self.a_clear = self._act("Clear all picks", self.clear_picks)
        self.a_prev = self._act("< Previous CMP", self.prev_cmp, "PgUp"); self.a_next = self._act("Next CMP >", self.next_cmp, "PgDown")
        self.a_jump = self._act("Jump to CMP...", self.jump_dialog, "Ctrl+G")
        self.a_seq = self._act("Set analysis sequence...", self.sequence_dialog)
        self.a_copyprev = self._act("Copy from previous CMP", self.copy_previous, "Ctrl+Shift+P",
                                    "Copies the previous location's saved function; marked needs_review")
        self.a_copysel = self._act("Copy from selected CMP...", self.copy_selected)
        self.a_save = self._act("Save new version", self.save_version, "Ctrl+S",
                                "Save the picks as a NEW version of this CMP in the active velocity name (nothing is overwritten)")
        self.a_overwrite = self._act("Overwrite latest version", self.save_overwrite, "Ctrl+Alt+S",
                                     "Replace the picks of this CMP's latest version (old picks are kept in an audit log; "
                                     "approved versions are protected)")
        self.a_save_as = self._act("Save as new velocity name...", self.save_as_new_name, "Ctrl+Shift+S",
                                   "Create a new named velocity and save the current picks into it")
        self.a_new_set = self._act("New empty velocity name...", self.new_velocity_name_dialog)
        self.a_rename_set = self._act("Rename velocity name...", self.rename_velocity_name_dialog)
        self.a_manage_sets = self._act("Manage velocity names...", self.manage_names_dialog)
        self.a_dbinfo = self._act("Show where picks are saved...", self.show_db_location)
        self.a_versions = self._act("Versions / restore...", self.versions_dialog)
        self.a_settings = self._act("Settings...", self.settings_dialog)
        self.a_recalc = self._act("Recompute", self.recompute, "F5")
        self.a_cancel = self._act("Cancel computation", self.cancel_jobs, "Esc")
        self.a_fit = self._act("Reset view", self.reset_view, "Home", "Undo zooming: show the whole analysed time and velocity range")
        self.tool_group = QtGui.QActionGroup(self)
        self.tool_group.setExclusive(True)

        def tool_action(name, text, key, tip):
            a = QtGui.QAction(text, self)
            a.setCheckable(True); a.setShortcut(QtGui.QKeySequence(key)); a.setToolTip(tip)
            a.triggered.connect(lambda checked=False, n=name: self.set_tool(n))
            self.tool_group.addAction(a)
            return a
        self.a_tool_pick = tool_action("pick", "Pick", "P", "Pick tool (P): click adds a pick, drag a pick to move it, right-click deletes it")
        self.a_tool_zoom = tool_action("zoom", "Zoom box", "Z", "Zoom tool (Z): drag a rectangle to zoom in (both axes); right-click zooms out")
        self.a_tool_hand = tool_action("hand", "Hand", "H", "Hand tool (H): drag to move the view; nothing is picked")
        self.a_tool_pick.setChecked(True)
        mb = self.menuBar()
        m = mb.addMenu("&File"); [m.addAction(a) for a in (self.a_open, self.a_export, self.a_dbinfo, self.a_quit)]
        m = mb.addMenu("&Edit"); [m.addAction(a) for a in (self.a_undo, self.a_redo, self.a_del, self.a_clear)]
        m = mb.addMenu("&Navigate"); [m.addAction(a) for a in (self.a_prev, self.a_next, self.a_jump, self.a_seq)]
        m = mb.addMenu("&Pick"); [m.addAction(a) for a in (self.a_copyprev, self.a_copysel, self.a_save, self.a_versions)]
        m = mb.addMenu("&Velocity")
        for a in (self.a_save, self.a_overwrite, self.a_save_as):
            m.addAction(a)
        m.addSeparator()
        for a in (self.a_new_set, self.a_rename_set, self.a_manage_sets):
            m.addAction(a)
        m.addSeparator(); m.addAction(self.a_versions)
        m = mb.addMenu("&Analysis"); [m.addAction(a) for a in (self.a_settings, self.a_recalc, self.a_cancel)]
        m = mb.addMenu("&View")
        for a in (self.a_tool_pick, self.a_tool_zoom, self.a_tool_hand):
            m.addAction(a)
        m.addSeparator(); m.addAction(self.a_fit)
        self.a_display = self.disp_dock.toggleViewAction()
        self.a_display.setText("Display options"); self.a_display.setShortcut(QtGui.QKeySequence("F6"))
        m.addAction(self.a_display)
        self.a_side = self.dock.toggleViewAction()
        self.a_side.setText("Side panel (locations / picks)"); self.a_side.setShortcut(QtGui.QKeySequence("F4"))
        m.addAction(self.a_side)
        tb = self.addToolBar("Main"); tb.setMovable(False)
        for a in (self.a_prev, self.a_next):
            tb.addAction(a)
        self.cmp_combo = QtWidgets.QComboBox(); self.cmp_combo.setMinimumWidth(90)
        self.cmp_combo.activated.connect(lambda i: self.goto_cmp(self.cmp_combo.itemData(i)))
        tb.addWidget(QtWidgets.QLabel(" CMP ")); tb.addWidget(self.cmp_combo); tb.addSeparator()
        for a in (self.a_undo, self.a_redo, self.a_copyprev): tb.addAction(a)
        tb.addSeparator()
        self.status_combo = QtWidgets.QComboBox(); self.status_combo.addItems(STATUSES); self.status_combo.setCurrentText("picked")
        self.set_combo = QtWidgets.QComboBox(); self.set_combo.setMinimumWidth(130)
        self.set_combo.activated.connect(lambda i: self.switch_velocity_name(self.set_combo.itemData(i)))
        tb.addWidget(QtWidgets.QLabel(" Velocity: ")); tb.addWidget(self.set_combo)
        tb.addWidget(QtWidgets.QLabel(" Status ")); tb.addWidget(self.status_combo)
        save_btn = QtWidgets.QToolButton(); save_btn.setDefaultAction(self.a_save)
        save_btn.setToolButtonStyle(_Q.ToolButtonStyle.ToolButtonTextOnly)
        save_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        save_menu = QtWidgets.QMenu(save_btn)
        for a in (self.a_save, self.a_overwrite, self.a_save_as):
            save_menu.addAction(a)
        save_btn.setMenu(save_menu); tb.addWidget(save_btn)
        tb.addSeparator(); tb.addAction(self.a_fit); tb.addAction(self.a_side)
        self.addToolBarBreak()
        tb2 = self.addToolBar("Display"); tb2.setMovable(False)
        tb2.addWidget(QtWidgets.QLabel(" Tool: "))
        for a in (self.a_tool_pick, self.a_tool_zoom, self.a_tool_hand):
            tb2.addAction(a)
        tb2.addSeparator()
        self.mode_combo = QtWidgets.QComboBox(); self.mode_combo.addItems(["Wiggle + VA", "Image"])
        self.nmo_check = QtWidgets.QCheckBox("NMO preview (display only)")
        self.neigh_check = QtWidgets.QCheckBox("Show neighbour picks"); self.neigh_check.setChecked(True)
        tb2.addWidget(QtWidgets.QLabel(" Gather: "))
        for w in (self.mode_combo, self.nmo_check, self.neigh_check):
            tb2.addWidget(w)
        self._disp_widgets["g_mode"] = self.mode_combo
        self.mode_combo.currentTextChanged.connect(lambda t: self.set_display("g_mode", t))
        self.nmo_check.toggled.connect(lambda _: self._draw_gather())
        self.neigh_check.toggled.connect(lambda _: self._draw_neighbours())
        tb2.addSeparator(); tb2.addAction(self.a_display)

    # ================================================================== file / project
    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open SEG-Y", "", "SEG-Y (*.sgy *.segy *.SGY *.SEGY);;All files (*)")
        if path:
            line, ok = QtWidgets.QInputDialog.getInt(self, "Line", "Line number of this data (project label):", 0, 0, 10**9)
            if ok:
                self.open_segy(path, line=line)

    def open_segy(self, path, line: int = 0, db_path=None):
        """Open a SEG-Y (read-only), build/load the header index, open/create the project database next to it."""
        path = Path(path)
        try:
            self.sf = SegyFile(path)
            cache = path.with_name(path.stem + ".velpick-index.npz")
            try:
                self.idx = TraceIndex.load_or_build(self.sf, cache)
            except OSError:
                self.idx = TraceIndex.build(self.sf)
            rep = self.idx.validate()
            if rep.has_errors:
                QtWidgets.QMessageBox.warning(self, "SEG-Y problems", rep.text())
            if self.db is not None:
                self.db.close()
            self.db = ProjectDB(db_path or path.with_name(path.stem + ".velpick.sqlite"))
        except (SegyError, OSError, RuntimeError) as e:
            QtWidgets.QMessageBox.critical(self, "Cannot open", str(e))
            return
        self.line = int(line)
        self.source_id = self.db.add_source(path, self.sf.file_size, self.sf.file_mtime_ns, self.idx.headers_sha256,
                                            self.sf.n_traces, self.sf.ns, self.sf.dt_us, self.sf.sample_format)
        self._moved_note = self.db.source_moved_from
        saved = self.db.get_meta("analysis_settings")
        if saved:
            self.settings = AnalysisSettings.from_dict(saved)
        dsaved = self.db.get_meta("display_settings")
        if dsaved:
            self.disp = DisplaySettings.from_dict(dsaved)
        self._sync_display_widgets()
        self._apply_colours(); self._apply_layout(); self._apply_wheel(); self._apply_spectrum_style()
        self.cache.clear()
        seq = self.db.get_meta("sequence")
        ids = [c for c in (seq or []) if self.idx.has_cmp(c)] or [int(c) for c in self.idx.cmp_ids]
        self.nav = Navigator(ids)
        last = self.db.get_meta("last_cmp")
        if last in ids:
            self.nav.jump(last)
        self._need_range = True
        self._fill_combo()
        self._refresh_set_combo()
        self._refresh_status_list()
        self.setWindowTitle(f"velpick - {path.name} (line {self.line})")
        self._load_location(self.nav.current)

    def export_dialog(self):
        if not self.db:
            return
        sid = self.db.active_set_id
        sets = self.db.list_sets()
        if len(sets) > 1:
            names = [r["name"] for r in sets]
            item, ok = QtWidgets.QInputDialog.getItem(self, "Export", "Velocity name to export:", names, names.index(self._vname()), False)
            if not ok:
                return
            sid = self.db.set_by_name(item)["id"]
        vname = self.db.get_set(sid)["name"]
        default = f"{Path(self.sf.path).stem}_{vname}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", default, "CSV (*.csv)")
        if path:
            self.export_to(path, sid)

    def export_to(self, path, set_id=None) -> int:
        n = export_csv(self.db, self.source_id, path, line=self.line, set_id=set_id)
        vname = self.db.get_set(self.db.active_set_id if set_id is None else set_id)["name"]
        self.statusBar().showMessage(f"Exported {n} velocity function(s) of '{vname}' to {path}", 8000)
        return n

    def db_location_text(self) -> str:
        if not self.db:
            return "No project is open."
        return (f"Picks are saved in the project database:\n{self.db.path}\n\n"
                f"Active velocity name: {self._vname()}\n"
                "It sits next to the SEG-Y (which is never modified). Each save creates a numbered version inside the named velocity.")

    def show_db_location(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Where picks are saved"); box.setText(self.db_location_text())
        copy = box.addButton("Copy path", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is copy and self.db:
            QtWidgets.QApplication.clipboard().setText(str(self.db.path))

    # ================================================================== navigation / location handling
    def _fill_combo(self):
        self.cmp_combo.blockSignals(True)
        self.cmp_combo.clear()
        for c in self.nav.ids:
            self.cmp_combo.addItem(str(c), c)
        self.cmp_combo.blockSignals(False)

    def goto_cmp(self, cmp_id):
        if self.nav is None or cmp_id is None or cmp_id == self.nav.current and self.loc_id is not None:
            return
        self._autosave()
        self.nav.jump(int(cmp_id))
        self._load_location(self.nav.current)

    def next_cmp(self):
        if self.nav and self.nav.next_id() is not None:
            self.goto_cmp(self.nav.next_id())

    def prev_cmp(self):
        if self.nav and self.nav.previous_id() is not None:
            self.goto_cmp(self.nav.previous_id())

    def jump_dialog(self):
        if not self.nav:
            return
        c, ok = QtWidgets.QInputDialog.getInt(self, "Jump to CMP", "CMP number:", self.nav.current, 0, 2**31 - 1)
        if ok:
            if c in self.nav.ids:
                self.goto_cmp(c)
            else:
                near = self.idx.nearest_ids(c, 2) if self.idx else []
                QtWidgets.QMessageBox.information(self, "Not in sequence", f"CMP {c} is not in the analysis sequence. Nearest: {near}")

    def sequence_dialog(self):
        if not self.idx:
            return
        ids = [int(c) for c in self.idx.cmp_ids]
        s, ok1 = QtWidgets.QInputDialog.getInt(self, "Sequence", "First CMP:", ids[0], 0, 2**31 - 1)
        if not ok1: return
        e, ok2 = QtWidgets.QInputDialog.getInt(self, "Sequence", "Last CMP:", ids[-1], 0, 2**31 - 1)
        if not ok2: return
        step, ok3 = QtWidgets.QInputDialog.getInt(self, "Sequence", "Interval (CMP numbers):", 1, 1, 100000)
        if not ok3: return
        have, missing = regular_sequence(ids, s, e, step)
        if not have:
            QtWidgets.QMessageBox.warning(self, "Empty sequence", "No requested location exists in the file."); return
        if missing:
            QtWidgets.QMessageBox.information(self, "Missing locations", f"{len(missing)} requested CMP(s) do not exist, e.g. {missing[:10]}")
        self.set_sequence(have)

    def set_sequence(self, ids: List[int]):
        self._autosave()
        cur = self.nav.current if self.nav.current in ids else ids[0]
        self.nav = Navigator(ids, cur)
        self.db.set_meta("sequence", ids)
        self._fill_combo(); self._refresh_status_list(); self._load_location(self.nav.current)

    def _load_location(self, cmp_id: int):
        s = self.idx.cmp_summary(cmp_id)
        self.loc_id = self.db.ensure_location(self.source_id, self.line, cmp_id, s["mid_x"], s["mid_y"])
        self.db.set_meta("last_cmp", cmp_id)
        saved = self.db.latest_version(self.loc_id)
        self.current_version = saved
        picks = self.db.get_picks(saved["id"]) if saved else []
        prov = saved["provenance"] if saved else "manual"
        self.model.reset(picks, prov, None)
        draft = self.db.load_draft(self.loc_id)
        if draft and (saved is None or draft["updated_utc"] >= saved["created_utc"]) and draft["picks"] != picks:
            if self._ask_recover(draft):
                self.model.set_all(draft["picks"], draft["provenance"], draft["source_info"], record=False)
            else:
                self.db.clear_draft(self.loc_id)
        self.status_combo.setCurrentText(saved["status"] if saved else "picked")
        self.note_edit.setText(saved["note"] if saved else "")
        self.sel = None
        self.cmp_combo.blockSignals(True)
        self.cmp_combo.setCurrentIndex(self.nav.ids.index(cmp_id)); self.cmp_combo.blockSignals(False)
        self._refresh_picks()
        self._request_analysis()

    def _ask_recover(self, draft) -> bool:
        r = QtWidgets.QMessageBox.question(self, "Recover unsaved picks?",
                                           f"Unsaved picks from {draft['updated_utc']} were found for this CMP. Recover them?")
        return r == QtWidgets.QMessageBox.StandardButton.Yes

    # ================================================================== analysis (background)
    def recompute(self):
        self.cache.clear(); self._request_analysis()

    def cancel_jobs(self):
        for j in self._jobs.values():
            j.cancel = True
        self.progress.setVisible(False)

    def _request_analysis(self):
        if self.sf is None or self.nav is None:
            return
        cmp_id, key = self.nav.current, (self.nav.current, self.settings.key())
        if key in self.cache:
            self.cache.move_to_end(key); self._apply_result(self.cache[key]); return
        self.cancel_jobs()
        self._req += 1
        req = self._req
        sf, idx, s = self.sf, self.idx, AnalysisSettings.from_dict(self.settings.to_dict())
        job = _Job(req, lambda progress, cancelled: run_analysis(sf, idx, cmp_id, s, progress, cancelled))
        job.signals.done.connect(self._on_done); job.signals.failed.connect(self._on_failed)
        job.signals.progress.connect(self._on_progress)
        self._jobs = {req: job}
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setVisible(True)
        self.statusBar().showMessage(f"Computing {s.method} spectrum for CMP {cmp_id}...")
        self.pool.start(job)

    def _on_progress(self, req, i, n):
        if req == self._req and n:
            self.progress.setValue(int(100 * i / n))

    def _on_done(self, req, res):
        if req != self._req:
            return
        self.progress.setVisible(False)
        self.cache[(res.cmp_id, AnalysisSettings.from_dict(res.settings).key())] = res
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        self._apply_result(res)
        self.statusBar().showMessage(f"CMP {res.cmp_id}: {res.method} computed in {res.elapsed_s:.1f} s", 6000)

    def _on_failed(self, req, msg):
        if req != self._req:
            return
        self.progress.setVisible(False)
        self.statusBar().showMessage("Analysis failed", 6000)
        QtWidgets.QMessageBox.warning(self, "Analysis failed", msg)

    def _apply_result(self, res: AnalysisResult):
        if res.cmp_id != self.nav.current:
            return
        self.result = res
        self._set_limits(res)
        self._draw_gather(); self._draw_spectrum(); self._draw_cvs(); self._draw_neighbours(); self._refresh_picks()
        if self._need_range:
            self.reset_view()
            self._need_range = False
        self._sync_scrollbar()
        for k in self._hmap:
            self._sync_hscroll(k)
        QtCore.QTimer.singleShot(0, self._place_hscrolls)

    def _record_range(self, res):
        g = res.gather
        return g.t_start_ms, g.t_start_ms + (g.ns - 1) * g.dt_ms

    def _view_range(self, res):
        """Time range shown in ALL panels: the analysed range (+/- half a spectrum line), inside the record.
        Every panel therefore starts and ends at the same time, with data (or picks) filling it."""
        t_lo, t_hi = self._record_range(res)
        half = float(res.t0_ms[1] - res.t0_ms[0]) / 2.0 if len(res.t0_ms) > 1 else 0.0
        return max(t_lo, float(res.t0_ms[0]) - half), min(t_hi, float(res.t0_ms[-1]) + half)

    def _set_limits(self, res: AnalysisResult):
        """Hard limits so the view can never leave the data: no negative times, no empty space around the panels."""
        t_lo, t_hi = self._view_range(res)
        dv = float(res.velocities[1] - res.velocities[0]) if len(res.velocities) > 1 else 1.0
        v_lo, v_hi = float(res.velocities[0] - dv), float(res.velocities[-1] + dv)
        ntr = res.gather.ntr
        for p in self.plots:
            p.getViewBox().setLimits(yMin=t_lo, yMax=t_hi)
        self.vb_spec.setLimits(xMin=v_lo, xMax=v_hi)
        self.vb_gather.setLimits(xMin=-1, xMax=ntr)
        self.vb_spec.full_x_range = (float(res.velocities[0] - dv / 2), float(res.velocities[-1] + dv / 2))
        self.vb_gather.full_x_range = (-0.5, ntr - 0.5)
        if res.cvs is not None:
            total = res.cvs.shape[0] * (res.cvs.shape[1] + 1)
            self.vb_cvs.setLimits(xMin=-1, xMax=total)
            self.vb_cvs.full_x_range = (-0.5, total - 1.5)

    def reset_view(self):
        """Show the analysed range in all panels (same time range everywhere)."""
        r = self.result
        if r is None:
            return
        t0, t1 = self._view_range(r)
        self.p_spec.setYRange(t0, t1, padding=0)
        dv = float(r.velocities[1] - r.velocities[0]) if len(r.velocities) > 1 else 1.0
        self.vb_spec.setXRange(r.velocities[0] - dv / 2, r.velocities[-1] + dv / 2, padding=0)
        self.vb_gather.setXRange(-0.5, r.gather.ntr - 0.5, padding=0)
        if r.cvs is not None:
            self.vb_cvs.setXRange(-0.5, r.cvs.shape[0] * (r.cvs.shape[1] + 1) - 1.5, padding=0)

    def settings_dialog(self):
        d = SettingsDialog(self.settings, self)
        if d.exec():
            self.settings = d.values()
            if self.db:
                self.db.set_meta("analysis_settings", self.settings.to_dict())
            self.cache.clear(); self._need_range = True; self._request_analysis()

    # ================================================================== drawing
    def _processed(self, data, dt_ms, pre: str):
        """Display-only processing of a COPY: optional bandpass, then AGC (filter first, then equalise)."""
        d = self.disp
        if getattr(d, pre + "_bandpass"):
            data = ormsby_bandpass(data, dt_ms, *(getattr(d, f"{pre}_f{k}") for k in (1, 2, 3, 4)))
        agc = getattr(d, pre + "_agc_ms")
        if agc > 0:
            data = equalize(data, dt_ms, agc)
        return data

    def _clip_level(self, data, pct, gain) -> float:
        nz = data[data != 0]
        clip = float(np.percentile(np.abs(nz), pct)) if nz.size else 1.0
        return (clip or 1.0) / max(float(gain), 1e-6)

    def _draw_gather(self):
        for it in self._gather_items:
            self.p_gather.removeItem(it)
        self._gather_items = []
        if self.result is None:
            return
        g, d = self.result.gather, self.disp
        data = g.data
        title = "Gather"
        if self.nmo_check.isChecked() and len(self.model) >= 1:
            t0 = g.t_start_ms + g.dt_ms * np.arange(1, g.ns)
            data, _ = nmo_correct(g.data, g.offsets, g.dt_ms, t0, self.model.velocity_at(t0), 50.0, g.t_start_ms)
            data = np.concatenate([np.zeros((data.shape[0], 1), np.float32), data], axis=1)
            title = "NMO preview"
        data = self._processed(data, g.dt_ms, "g")
        clip = self._clip_level(data, d.g_clip, d.g_gain)
        self._last_gather_display, self._last_gather_clip = data, clip
        self.p_gather.setTitle(title, size="9pt")
        self.p_gather.getAxis("top").offsets = g.offsets
        ntr = g.ntr
        if d.g_mode == "Image":
            self.img_gather.setVisible(True)
            self.img_gather.setImage(np.ascontiguousarray(data.T), autoLevels=False, levels=(-clip, clip))
            self.img_gather.setRect(QtCore.QRectF(-0.5, g.t_start_ms - g.dt_ms / 2, ntr, g.ns * g.dt_ms))
        else:
            self.img_gather.setVisible(False)
            pick = np.unique(np.linspace(0, ntr - 1, min(ntr, int(d.g_max_traces))).astype(int))
            spacing = ntr / max(len(pick), 1)
            t = g.t_start_ms + g.dt_ms * np.arange(g.ns)
            self._wiggles(self.p_gather, self._gather_items, [(float(k), data[k]) for k in pick], t, clip, spacing)

    def _wiggles(self, plot, store, traces, t, clip, spacing, wiggle_pen=0.6):
        """Variable-area wiggle display. traces = [(x_position, samples)]; amplitude `clip` = 1.5 x spacing."""
        gain = 1.5 * spacing / clip
        for x0, tr in traces:
            if not np.any(tr):
                continue
            x = np.clip(x0 + gain * tr, x0 - 1.5 * spacing, x0 + 1.5 * spacing)
            line = pg.PlotCurveItem(x=x, y=t, pen=pg.mkPen("k", width=wiggle_pen))
            a = pg.PlotCurveItem(x=np.maximum(x, x0), y=t, pen=None)
            b = pg.PlotCurveItem(x=np.full_like(t, x0), y=t, pen=None)
            fill = pg.FillBetweenItem(a, b, brush=pg.mkBrush("k"))
            for it in (fill, line):
                plot.addItem(it); store.append(it)

    def _draw_spectrum(self):
        r = self.result
        if r is None:
            return
        d, sd = r.display, self.disp
        dv = float(r.velocities[1] - r.velocities[0]) if len(r.velocities) > 1 else 1.0
        dt = float(r.t0_ms[1] - r.t0_ms[0]) if len(r.t0_ms) > 1 else 1.0
        lo, hi = float(np.percentile(d, sd.s_clip_low)), float(np.percentile(d, sd.s_clip_high))
        if hi <= lo:
            hi = lo + 1e-9
        self.img_spec.setLookupTable(make_lut(sd.s_cmap, sd.s_gamma))
        self.img_spec.setImage(d, autoLevels=False, levels=(lo, hi))
        self.img_spec.setRect(QtCore.QRectF(r.velocities[0] - dv / 2, r.t0_ms[0] - dt / 2, len(r.velocities) * dv, len(r.t0_ms) * dt))
        self.p_spec.setTitle(f"CMP {r.cmp_id} {r.method}", size="9pt")

    def _draw_cvs(self):
        for it in self._cvs_items:
            self.p_cvs.removeItem(it)
        self._cvs_items = []
        r = self.result
        if r is None or r.cvs is None:
            self._cvs_centres = self._cvs_vels = None
            self.p_cvs.setTitle("Stack segments (off)", size="9pt")
            self.p_cvs.getAxis("top").centres = None
            return
        S, G, nt = r.cvs.shape
        pitch = G + 1.0
        self._cvs_vels = np.asarray(r.cvs_vels, float)
        self._cvs_centres = np.arange(S) * pitch + (G - 1) / 2.0
        ax = self.p_cvs.getAxis("top")
        ax.centres, ax.labels = list(self._cvs_centres), list(self._cvs_vels)
        d = self.disp
        data = self._processed(r.cvs.reshape(S * G, nt), float(r.cvs_t[1] - r.cvs_t[0]), "c")
        clip = self._clip_level(data, d.c_clip, d.c_gain)
        self._last_cvs_display, self._last_cvs_clip = data, clip
        traces = [(k * pitch + j, data[k * G + j]) for k in range(S) for j in range(G)]
        self._wiggles(self.p_cvs, self._cvs_items, traces, r.cvs_t, clip, 0.6, wiggle_pen=0.5)
        ax.starts = [k * pitch for k in range(S)]
        ax.mode = d.c_axis if d.c_axis in ("velocity", "trace") else "velocity"
        if ax.mode == "trace":
            self.p_cvs.setLabel("top", "Trace in segment")
        else:
            self.p_cvs.setLabel("top", "Segment velocity", "m/s")
        self.p_cvs.setTitle("Stack segments", size="9pt")
        self._refresh_cvs_picks()

    def _cvs_x_to_v(self, x: float) -> float:
        if self._cvs_centres is None:
            return -1.0
        return float(np.interp(x, self._cvs_centres, self._cvs_vels))

    def _cvs_v_to_x(self, v):
        return np.interp(v, self._cvs_vels, self._cvs_centres)

    def _refresh_cvs_picks(self):
        pk = self.model.picks
        if self._cvs_centres is None or not pk:
            for it in (self.cvs_curve, self.cvs_pts, self.sel_cvs):
                it.setData(x=[], y=[])
            return
        x = [float(self._cvs_v_to_x(v)) for _, v in pk]
        t = [tt for tt, _ in pk]
        self.cvs_curve.setData(x=x, y=t); self.cvs_pts.setData(x=x, y=t)
        if self.sel is not None and 0 <= self.sel < len(pk):
            self.sel_cvs.setData(x=[x[self.sel]], y=[t[self.sel]])
        else:
            self.sel_cvs.setData(x=[], y=[])

    def _saved_picks_at(self, cmp_id):
        if cmp_id is None or not self.db:
            return None, None
        row = self.db.conn.execute("SELECT id FROM location WHERE source_file_id=? AND line=? AND cmp=?",
                                   (self.source_id, self.line, cmp_id)).fetchone()
        v = self.db.latest_version(row["id"]) if row else None
        return (self.db.get_picks(v["id"]), v) if v else (None, None)

    def _draw_neighbours(self):
        """Saved functions of the previous / next CMP in the sequence, drawn on the spectrum."""
        for curve in (self.neigh_prev, self.neigh_next, self.neigh_prev_pts, self.neigh_next_pts):
            curve.setData(x=[], y=[])
        if self.neigh_legend.scene() is not None:
            self.p_spec.removeItem(self.neigh_legend)
        if not (self.neigh_check.isChecked() and self.nav and self.db):
            return
        labels = []
        for curve, pts, nid, name, col in ((self.neigh_prev, self.neigh_prev_pts, self.nav.previous_id(), "previous", self.disp.col_prev),
                                           (self.neigh_next, self.neigh_next_pts, self.nav.next_id(), "next", self.disp.col_next)):
            pk, v = self._saved_picks_at(nid)
            if pk:
                curve.setData(x=[p[1] for p in pk], y=[p[0] for p in pk])
                pts.setData(x=[p[1] for p in pk], y=[p[0] for p in pk])
                labels.append(f'<span style="color:{col}; font-size:10pt"><b>{name}: CMP {nid} v{v["version_no"]} ({v["status"]})</b></span>')
            elif nid is not None:
                labels.append(f'<span style="color:#dddddd; font-size:10pt">{name}: CMP {nid} (no saved picks)</span>')
        if labels:
            self.neigh_legend.setHtml("<br>".join(labels))
            self.p_spec.addItem(self.neigh_legend, ignoreBounds=True)
            self._place_legend()

    def _place_legend(self):
        (x0, x1), (y0, y1) = self.vb_spec.viewRange()
        self.neigh_legend.setPos(x0 + 0.02 * (x1 - x0), y1 - 0.01 * (y1 - y0) - 0)
        self.neigh_legend.setAnchor((0, 1))

    def _draw_saved_overlay(self):
        """While editing, show the last saved version of this CMP as a white dotted line for comparison."""
        if self.current_version is not None and self.model.dirty:
            pk = self.db.get_picks(self.current_version["id"])
            self.saved_curve.setData(x=[p[1] for p in pk], y=[p[0] for p in pk])
        else:
            self.saved_curve.setData(x=[], y=[])

    def _refresh_picks(self):
        pk = self.model.picks
        x, t = [p[1] for p in pk], [p[0] for p in pk]
        self.spec_curve.setData(x=x, y=t)
        self.spec_pts.setData(x=x, y=t)
        self._draw_saved_overlay()
        if self.sel is not None and 0 <= self.sel < len(pk):
            self.sel_spec.setData(x=[x[self.sel]], y=[t[self.sel]])
            for l in self.selline:
                l.setPos(t[self.sel]); l.setVisible(True)
        else:
            self.sel = None
            self.sel_spec.setData(x=[], y=[])
            for l in self.selline:
                l.setVisible(False)
        self._refresh_cvs_picks()
        self._fill_table()
        s = self.settings
        w = self.model.warnings(vmin=s.vmin, vmax=s.vmax, t_min=s.t0_start, t_max=s.t0_end)
        self.warn.setText("\n".join("! " + m for m in w[:8]))
        self._update_info()
        star = " *" if self.model.dirty else ""
        base = f"velpick - {Path(self.sf.path).name} (line {self.line})" if self.sf else "velpick"
        vn = f" [{self._vname()}]" if self.db else ""
        self.setWindowTitle(base + (f" - CMP {self.nav.current}" if self.nav else "") + vn + star)

    def _fill_table(self):
        pk = self.model.picks
        self.table.blockSignals(True)
        self.table.setRowCount(len(pk))
        for i, (t, v) in enumerate(pk):
            for j, val in enumerate((t, v)):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{val:.1f}"))
        if self.sel is not None:
            self.table.selectRow(self.sel)
        self.table.blockSignals(False)

    def _update_info(self):
        if not self.nav or not self.idx:
            return
        s = self.idx.cmp_summary(self.nav.current)
        v = self.current_version
        saved = f"saved v{v['version_no']} ({v['status']}, {v['provenance']})" if v else "nothing saved yet"
        prov = self.model.provenance + (" - NEEDS REVIEW" if self.model.provenance in ("copied", "restored", "interpolated") else "")
        res = f"<br>Spectrum from CMPs {self.result.cmps_used}" if self.result else ""
        notes = ("<br>" + " ".join(self.result.notes)) if self.result and self.result.notes else ""
        self.info.setText(f"<b>CMP {self.nav.current}</b> (line {self.line}) - {self.nav.position + 1}/{len(self.nav.ids)}<br>"
                          f"fold {s['fold']}, offsets {s['offset_min']:.0f}-{s['offset_max']:.0f} m<br>"
                          f"velocity name: <b>{self._vname()}</b><br>"
                          f"{saved}<br>editing: {prov}{res}{notes}"
                          + (f"<br><i>This SEG-Y was moved (was {self._moved_note}); its picks were re-attached.</i>" if self._moved_note else ""))
        self.info.setToolTip(f"Saved in: {self.db.path}" if self.db else "")

    def _refresh_status_list(self):
        self.status_list.clear()
        if not self.nav:
            return
        st = self.db.location_status(self.source_id, self.line)
        for c in self.nav.ids:
            s = st.get(c)
            text = f"CMP {c}   " + (f"{s['status']}  v{s['version_no']}" if s else "unprocessed")
            it = QtWidgets.QListWidgetItem(text)
            it.setData(_Q.ItemDataRole.UserRole, c)
            it.setForeground(QtGui.QColor(STATUS_COLOURS.get(s["status"], "#777777") if s else "#777777"))
            self.status_list.addItem(it)

    # ================================================================== picking interaction
    def _xy_spec(self, v: float) -> float:
        return v

    def _xy_cvs(self, v: float) -> float:
        return float(self._cvs_v_to_x(v)) if self._cvs_centres is not None else v

    def _hit(self, vb: PickViewBox, scene_pos, to_x, radius: float = 10.0):
        best, bd = None, radius
        for i, (t, v) in enumerate(self.model.picks):
            p = vb.mapViewToScene(QtCore.QPointF(to_x(v), t))
            d = ((p.x() - scene_pos.x()) ** 2 + (p.y() - scene_pos.y()) ** 2) ** 0.5
            if d <= bd:
                best, bd = i, d
        return best

    def on_add(self, v: float, t: float):
        if self.nav is None or t <= 0 or v <= 0:
            return
        self.sel = self.model.add(t, v)
        self._refresh_picks(); self._redraw_nmo()

    def on_select(self, i: int):
        self.sel = i; self._refresh_picks()

    def on_delete(self, i: int):
        if 0 <= i < len(self.model):
            self.model.delete(i); self.sel = None
            self._refresh_picks(); self._redraw_nmo()

    def on_move(self, i: int, v: float, t: float, finished: bool):
        if not (0 <= i < len(self.model)):
            return
        if not self._dragging:
            self.model.checkpoint(); self._dragging = True
        self.model.move(i, t, max(v, 1.0))
        self.sel = i
        self._refresh_picks()
        if finished:
            self._dragging = False
            self._redraw_nmo()

    def _redraw_nmo(self):
        if self.nmo_check.isChecked():
            self._draw_gather()

    def delete_selected(self):
        if self.sel is not None:
            self.on_delete(self.sel)

    def clear_picks(self):
        self.model.clear(); self.sel = None; self._refresh_picks(); self._redraw_nmo()

    def undo(self):
        if self.model.undo():
            self.sel = None; self._refresh_picks(); self._redraw_nmo()

    def redo(self):
        if self.model.redo():
            self.sel = None; self._refresh_picks(); self._redraw_nmo()

    def _on_table_select(self):
        rows = {i.row() for i in self.table.selectedItems()}
        if len(rows) == 1:
            self.sel = rows.pop()
            self._refresh_selection_only()

    def _refresh_selection_only(self):
        pk = self.model.picks
        if self.sel is not None and 0 <= self.sel < len(pk):
            t, v = pk[self.sel]
            self.sel_spec.setData(x=[v], y=[t])
            if self._cvs_centres is not None:
                self.sel_cvs.setData(x=[float(self._cvs_v_to_x(v))], y=[t])
            for l in self.selline:
                l.setPos(t); l.setVisible(True)

    def _on_cell_changed(self, row, col):
        try:
            val = float(self.table.item(row, col).text())
            t, v = self.model.picks[row]
            self.model.checkpoint()
            if col == 0:
                self.model.move(row, val, v)
            else:
                self.model.move(row, t, val)
        except (ValueError, IndexError):
            pass
        self.sel = row
        self._refresh_picks(); self._redraw_nmo()

    def _on_hover(self, pos):
        for p in self.plots:
            vb = p.getViewBox()
            if not (p.isVisible() and vb.sceneBoundingRect().contains(pos)):
                continue
            pt = vb.mapSceneToView(pos)
            for h in self.hline:
                h.setPos(pt.y())
            vl = self.vline.get(id(p))
            if vl is not None:
                vl.setPos(pt.x())
            r = self.result
            if p is self.p_gather:
                k = int(round(pt.x()))
                if r is not None and 0 <= k < r.gather.ntr:
                    self.statusBar().showMessage(f"trace {k}  offset {r.gather.offsets[k]:.0f} m   t = {pt.y():.0f} ms")
            elif p is self.p_cvs:
                msg = f"v = {self._cvs_x_to_v(pt.x()):.0f} m/s   t = {pt.y():.0f} ms"
                if r is not None and r.cvs is not None and self._cvs_centres is not None:
                    S, G = r.cvs.shape[:2]
                    k = int(pt.x() // (G + 1.0)); j = int(round(pt.x() - k * (G + 1.0)))
                    if 0 <= k < S:
                        msg = (f"segment {k + 1} of {S} = {r.cvs_vels[k]:.0f} m/s"
                               + (f", trace CMP {r.cvs_cmps[j]}" if 0 <= j < G else "") + f"   t = {pt.y():.0f} ms")
                self.statusBar().showMessage(msg)
            else:
                msg = f"v = {pt.x():.0f} m/s   t = {pt.y():.0f} ms"
                if p is self.p_spec and r is not None:
                    a = int(np.argmin(np.abs(r.t0_ms - pt.y()))); b = int(np.argmin(np.abs(r.velocities - pt.x())))
                    msg += f"   value {r.display[a, b]:.2f}"
                self.statusBar().showMessage(msg)
            return

    # ================================================================== copying / saving / versions
    def _saved_picks_for(self, cmp_id: int):
        row = self.db.conn.execute("SELECT id FROM location WHERE source_file_id=? AND line=? AND cmp=?",
                                   (self.source_id, self.line, cmp_id)).fetchone()
        v = self.db.latest_version(row["id"]) if row else None
        return (self.db.get_picks(v["id"]), v) if v else (None, None)

    def _apply_copy(self, cmp_id: int, label: str) -> bool:
        pk, v = self._saved_picks_for(cmp_id)
        if not pk:
            QtWidgets.QMessageBox.information(self, "Nothing to copy", f"CMP {cmp_id} has no saved function.")
            return False
        self.model.set_all(pk, "copied", {"from_cmp": cmp_id, "from_version_no": v["version_no"], "how": label})
        self.status_combo.setCurrentText("needs_review"); self.sel = None
        self._refresh_picks(); self._redraw_nmo()
        self.statusBar().showMessage(f"Copied CMP {cmp_id} v{v['version_no']} - marked needs_review; edit, then save.", 8000)
        return True

    def copy_previous(self):
        if self.nav and self.nav.previous_id() is not None:
            self._apply_copy(self.nav.previous_id(), "previous")
        else:
            self.statusBar().showMessage("There is no previous CMP.", 4000)

    def copy_selected(self):
        if not self.nav:
            return
        st = self.db.location_status(self.source_id, self.line)
        options = [str(c) for c in sorted(st) if c != self.nav.current]
        if not options:
            QtWidgets.QMessageBox.information(self, "Nothing to copy", "No other CMP has a saved function yet."); return
        c, ok = QtWidgets.QInputDialog.getItem(self, "Copy from CMP", "Source CMP:", options, 0, False)
        if ok:
            self._apply_copy(int(c), "selected")

    def save_version(self):
        if self.db is None or self.loc_id is None:
            return None
        pk = self.model.picks
        if not pk:
            self.statusBar().showMessage("No picks to save.", 4000); return None
        cfg = dict(self.result.settings) if self.result else self.settings.to_dict()
        if self.result:
            cfg["cmps_used"] = self.result.cmps_used
        try:
            vid, vno = self.db.save_version(self.loc_id, pk, provenance=self.model.provenance, source_info=self.model.source_info,
                                            analysis_config=cfg, status=self.status_combo.currentText(), note=self.note_edit.text())
        except (ValueError, RuntimeError) as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e)); return None
        self.model.mark_saved()
        self.current_version = self.db.get_version(vid)
        self._refresh_status_list(); self._draw_neighbours(); self._refresh_picks()
        self.statusBar().showMessage(f"Saved CMP {self.nav.current} as version {vno} in '{self._vname()}' ({self.status_combo.currentText()}).", 8000)
        return vno

    def _save_args(self):
        cfg = dict(self.result.settings) if self.result else self.settings.to_dict()
        if self.result:
            cfg["cmps_used"] = self.result.cmps_used
        return dict(provenance=self.model.provenance, source_info=self.model.source_info, analysis_config=cfg,
                    status=self.status_combo.currentText(), note=self.note_edit.text())

    # ================================================================== named velocities (save as / overwrite / switch)
    def _vname(self) -> str:
        if not self.db:
            return ""
        r = self.db.get_set(self.db.active_set_id)
        return r["name"] if r else ""

    def _refresh_set_combo(self):
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        if self.db:
            for r in self.db.list_sets():
                self.set_combo.addItem(r["name"], r["id"])
            self.set_combo.setCurrentIndex(max(0, self.set_combo.findData(self.db.active_set_id)))
            self.set_combo.setToolTip(f"Active velocity name (choose another to switch).\nSaved in: {self.db.path}")
        self.set_combo.blockSignals(False)

    def switch_velocity_name(self, set_id):
        """Make another named velocity active; unsaved picks stay as a draft of the one you leave."""
        if not self.db or set_id is None or int(set_id) == self.db.active_set_id:
            return
        self._autosave()
        self.db.set_active(int(set_id))
        self.current_version = None
        self._refresh_set_combo(); self._refresh_status_list()
        if self.nav:
            self._load_location(self.nav.current)
        self.statusBar().showMessage(f"Active velocity name: {self._vname()}", 6000)

    def create_velocity_name(self, name, description="", copy_all=False, switch=True) -> int:
        """New named velocity: empty, or holding a copy of the latest function of every CMP of the active one."""
        if not self.db:
            raise RuntimeError("no project open")
        new_id = (self.db.copy_set(self.db.active_set_id, name, description) if copy_all
                  else self.db.create_set(name, description))
        if switch:
            self.switch_velocity_name(new_id)
        else:
            self._refresh_set_combo()
        return new_id

    def save_as_named(self, name, description="", copy_all=True):
        """'Save as': create the named velocity (optionally copying all functions), make it active and save the picks
        that are in the editor right now as a new version of this CMP in it. The old name keeps what it had."""
        if not self.db or self.loc_id is None:
            return None
        picks, args = self.model.picks, self._save_args()
        old_loc = self.loc_id
        new_id = (self.db.copy_set(self.db.active_set_id, name, description) if copy_all
                  else self.db.create_set(name, description))                    # raises ValueError for a bad / duplicate name
        self.db.clear_draft(old_loc)                                              # these edits now belong to the new name
        self.db.set_active(new_id)
        vno = None
        if picks:
            _, vno = self.db.save_version(old_loc, picks, **args)
        self._refresh_set_combo(); self._refresh_status_list()
        self._load_location(self.nav.current)
        self.statusBar().showMessage(f"Saved as new velocity name '{self._vname()}'" + (f" (CMP {self.nav.current} version {vno})" if vno else ""), 8000)
        return vno

    def save_as_new_name(self):
        if not self.db:
            return
        d = NameDialog("Save as new velocity name", self, copy_text="Copy all functions from the current velocity name "
                       f"('{self._vname()}') - uncheck to start with only this CMP",
                       note="The new name gets its own functions and version history. '" + self._vname() + "' stays exactly as it is.")
        if d.exec():
            name, desc, copy_all = d.values()
            try:
                self.save_as_named(name, desc, copy_all)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Cannot create the velocity name", str(e))

    def new_velocity_name_dialog(self):
        if not self.db:
            return
        d = NameDialog("New empty velocity name", self)
        if d.exec():
            name, desc, _ = d.values()
            try:
                self.create_velocity_name(name, desc, copy_all=False)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Cannot create the velocity name", str(e))

    def rename_velocity_name_dialog(self):
        if not self.db:
            return
        cur = self.db.get_set(self.db.active_set_id)
        d = NameDialog("Rename velocity name", self, name=cur["name"], description=cur["description"])
        if d.exec():
            name, desc, _ = d.values()
            try:
                self.rename_velocity_name(name, desc)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Cannot rename", str(e))

    def rename_velocity_name(self, name, description=None):
        self.db.rename_set(self.db.active_set_id, name)
        if description is not None:
            self.db.set_description(self.db.active_set_id, description)
        self._refresh_set_combo(); self._refresh_picks()

    def manage_names_dialog(self):
        if not self.db:
            return
        d = QtWidgets.QDialog(self); d.setWindowTitle("Velocity names in this project"); d.resize(760, 320)
        lay = QtWidgets.QVBoxLayout(d)
        tbl = QtWidgets.QTableWidget(0, 5)
        tbl.setHorizontalHeaderLabels(["Name", "Description", "CMPs with picks", "Versions", "Last changed (UTC)"])
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.horizontalHeader().setStretchLastSection(True)
        rows = self.db.list_sets()
        tbl.setRowCount(len(rows))
        for i, r in enumerate(rows):
            mark = "  (active)" if r["id"] == self.db.active_set_id else ""
            for j, val in enumerate((r["name"] + mark, r["description"], r["n_functions"], r["n_versions"], r["modified_utc"])):
                tbl.setItem(i, j, QtWidgets.QTableWidgetItem(str(val)))
        lay.addWidget(tbl)
        lay.addWidget(QtWidgets.QLabel(f"Database: {self.db.path}"))
        bb = QtWidgets.QDialogButtonBox()
        go = bb.addButton("Switch to selected", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close).clicked.connect(d.reject)
        go.clicked.connect(d.accept); lay.addWidget(bb)
        if d.exec() and tbl.currentRow() >= 0:
            self.switch_velocity_name(rows[tbl.currentRow()]["id"])

    def _confirm_overwrite(self, latest) -> bool:
        r = QtWidgets.QMessageBox.question(
            self, "Overwrite latest version?",
            f"Replace the picks of CMP {self.nav.current}, version {latest['version_no']} in '{self._vname()}'?\n\n"
            "The previous picks stay available in the audit log. (Use 'Save new version' to keep both as versions.)")
        return r == QtWidgets.QMessageBox.StandardButton.Yes

    def save_overwrite(self):
        """Replace the latest saved version of this CMP in the active velocity name (never an approved one)."""
        if self.db is None or self.loc_id is None:
            return None
        pk = self.model.picks
        if not pk:
            self.statusBar().showMessage("No picks to save.", 4000); return None
        latest = self.db.latest_version(self.loc_id)
        if latest is None:
            self.statusBar().showMessage("Nothing saved yet for this CMP - saving a first version instead.", 6000)
            return self.save_version()
        if latest["status"] == "approved":
            QtWidgets.QMessageBox.warning(self, "Protected version",
                                          f"Version {latest['version_no']} of CMP {self.nav.current} is approved and cannot be overwritten.\n"
                                          "Use 'Save new version', or change its review status first.")
            return None
        if not self._confirm_overwrite(latest):
            return None
        try:
            vno = self.db.overwrite_version(latest["id"], pk, **self._save_args())
        except (ValueError, RuntimeError) as e:
            QtWidgets.QMessageBox.critical(self, "Overwrite failed", str(e)); return None
        self.model.mark_saved()
        self.current_version = self.db.get_version(latest["id"])
        self._refresh_status_list(); self._draw_neighbours(); self._refresh_picks()
        self.statusBar().showMessage(f"Overwrote version {vno} of CMP {self.nav.current} in '{self._vname()}'.", 8000)
        return vno

    def versions_dialog(self):
        if self.db is None or self.loc_id is None:
            return
        rows = self.db.versions(self.loc_id)
        d = QtWidgets.QDialog(self); d.setWindowTitle(f"Versions - CMP {self.nav.current}"); d.resize(720, 320)
        lay = QtWidgets.QVBoxLayout(d)
        tbl = QtWidgets.QTableWidget(len(rows), 6)
        tbl.setHorizontalHeaderLabels(["Ver", "Saved (UTC)", "Status", "Provenance", "Picks", "Note"])
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        for i, r in enumerate(rows):
            for j, val in enumerate((r["version_no"], r["created_utc"], r["status"], r["provenance"], r["n_picks"], r["note"])):
                tbl.setItem(i, j, QtWidgets.QTableWidgetItem(str(val)))
        lay.addWidget(tbl)
        lay.addWidget(QtWidgets.QLabel("'Load into editor' copies the chosen version into the editor as needs_review. "
                                       "Nothing is overwritten; saving creates a new version."))
        bb = QtWidgets.QDialogButtonBox()
        load = bb.addButton("Load into editor", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close).clicked.connect(d.reject)
        load.clicked.connect(d.accept); lay.addWidget(bb)
        if d.exec() and tbl.currentRow() >= 0:
            r = rows[tbl.currentRow()]
            self.model.set_all(self.db.get_picks(r["id"]), "restored", {"restored_from_version_no": r["version_no"],
                                                                         "restored_from_version_id": r["id"]})
            self.status_combo.setCurrentText("needs_review"); self.sel = None
            self._refresh_picks(); self._redraw_nmo()

    # ================================================================== safety
    def _autosave(self):
        if self.db is None or self.loc_id is None:
            return
        if self.model.dirty:
            self.db.save_draft(self.loc_id, self.model.picks, self.model.provenance, self.model.source_info)
        else:
            self.db.clear_draft(self.loc_id)

    def closeEvent(self, ev):
        self.cancel_jobs()
        if self.model.dirty and self.db is not None:
            r = QtWidgets.QMessageBox.question(
                self, "Unsaved picks", "Save the current picks as a new version before closing?",
                QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard |
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if r == QtWidgets.QMessageBox.StandardButton.Cancel:
                ev.ignore(); return
            if r == QtWidgets.QMessageBox.StandardButton.Save:
                self.save_version()
            else:
                self.db.clear_draft(self.loc_id)
        if self.db is not None:
            self.db.close(); self.db = None
        ev.accept()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="velpick velocity picking workstation")
    ap.add_argument("segy", nargs="?", help="SEG-Y file with CMP gathers (opened read-only)")
    ap.add_argument("--line", type=int, default=0, help="line number label for this data (default 0)")
    ap.add_argument("--db", help="project database path (default: next to the SEG-Y)")
    a = ap.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    w = MainWindow()
    w.show()
    if a.segy:
        w.open_segy(a.segy, line=a.line, db_path=a.db)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
