"""Small pyqtgraph building blocks: pick-aware view box, offset axis, colour tables."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

# white background + black foreground: black variable-area wiggles must be visible (processing-system convention)
pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="w", foreground="k")

_MB = QtCore.Qt.MouseButton


def jet_lut(n: int = 256) -> np.ndarray:
    pos = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    col = [(0, 0, 128), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)]
    return pg.ColorMap(pos, col).getLookupTable(0.0, 1.0, n)


def seismic_lut(n: int = 256) -> np.ndarray:
    """White = negative, black = positive (variable-density convention)."""
    return pg.ColorMap([0.0, 1.0], [(255, 255, 255), (0, 0, 0)]).getLookupTable(0.0, 1.0, n)


def zoom_cursor() -> QtGui.QCursor:
    """A small magnifier, so the zoom-box tool is recognisable."""
    pm = QtGui.QPixmap(28, 28); pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor("black"), 2)); p.setBrush(QtGui.QColor(255, 255, 255, 150))
    p.drawEllipse(3, 3, 14, 14); p.drawLine(15, 15, 25, 25)
    p.setPen(QtGui.QPen(QtGui.QColor("black"), 1)); p.drawLine(10, 6, 10, 14); p.drawLine(6, 10, 14, 10)
    p.end()
    return QtGui.QCursor(pm, 10, 10)


TOOLS = ("pick", "zoom", "hand")


class PickViewBox(pg.ViewBox):
    """Mouse behaviour depends on the TOOL, so picking, zooming and panning can never be confused:
      pick : left click adds a pick (or selects one); left drag on a pick moves it; right click on a pick deletes it;
             left drag on empty space does nothing.
      zoom : left drag draws a rectangle and zooms exactly to it (both axes); right click zooms back out.
      hand : left drag pans (vertical only unless the horizontal axis is zoomed in); clicks never pick.
    In every tool: the wheel zooms / scrolls, middle-button drag and dragging an axis strip pan.
    Coordinates are view coordinates (x = velocity, y = time)."""
    sigResetView = QtCore.Signal()
    sigAdd = QtCore.Signal(float, float)
    sigSelect = QtCore.Signal(int)
    sigDelete = QtCore.Signal(int)
    sigMove = QtCore.Signal(int, float, float, bool)      # index, x, y, finished

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_test = None          # callable(scene_pos) -> index or None, set by the window
        self.picking_enabled = True
        self.tool = "pick"
        self.full_x_range = None      # (lo, hi) of the un-zoomed horizontal view, set by the window
        self.wheel_scrolls = False    # False: wheel zooms time (Shift scrolls); True: wheel scrolls (Ctrl zooms time)
        self._drag_idx = None
        self.setMenuEnabled(False)

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(tool)
        self.tool = tool
        self._drag_idx = None
        self.setMouseMode(pg.ViewBox.RectMode if tool == "zoom" else pg.ViewBox.PanMode)
        if tool == "zoom":
            self.setCursor(zoom_cursor())
        elif tool == "hand":
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.OpenHandCursor))
        else:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

    def showAxRect(self, ax, **kwargs):
        """Zoom exactly to the rectangle that was drawn (pyqtgraph adds ~5 % padding by default)."""
        kwargs.setdefault("padding", 0)
        super().showAxRect(ax, **kwargs)

    def x_is_zoomed(self) -> bool:
        """True when the horizontal axis shows less than the full range (set by the window as ``full_x_range``)."""
        full = self.full_x_range or tuple(self.state["limits"]["xLimits"])
        if full[0] is None or full[1] is None:
            return True
        lo, hi = self.viewRange()[0]
        return (hi - lo) < 0.99 * (full[1] - full[0])

    def scroll_x(self, fraction: float) -> None:
        """Move the horizontal window by a fraction of its width (positive = to the right)."""
        lo, hi = self.viewRange()[0]
        self.translateBy(x=(hi - lo) * fraction)

    def scroll_time(self, fraction: float) -> None:
        """Move the shared time window by a fraction of its height (positive = later times)."""
        lo, hi = self.viewRange()[1]
        self.translateBy(y=(hi - lo) * fraction)

    def _hit(self, scene_pos):
        return self.hit_test(scene_pos) if self.hit_test else None

    def wheelEvent(self, ev, axis=None):
        """Default: wheel = zoom TIME around the cursor, Ctrl+wheel = zoom the horizontal axis, Shift+wheel = scroll time.
        With ``wheel_scrolls`` (touch-pads): wheel = scroll time, Ctrl+wheel = zoom time, Shift+wheel = zoom horizontal."""
        mods = ev.modifiers()
        ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        alt = bool(mods & QtCore.Qt.KeyboardModifier.AltModifier)
        try:
            horizontal = ev.orientation() == QtCore.Qt.Orientation.Horizontal     # sideways swipe on a touch-pad
        except Exception:
            horizontal = False
        if alt or horizontal:                                                    # Alt+wheel / sideways swipe: scroll velocity
            self.scroll_x(-0.12 * ev.delta() / 120.0)
            ev.accept()
            return
        s = 1.02 ** (ev.delta() * self.state["wheelScaleFactor"])
        c = self.mapSceneToView(ev.scenePos())
        center = pg.Point(c.x(), c.y())
        notch = ev.delta() / 120.0
        if self.wheel_scrolls:
            if ctrl:
                self.scaleBy(y=s, center=center)
            elif shift:
                self.scaleBy(x=s, center=center)
            else:
                self.scroll_time(-0.12 * notch)
        else:
            if ctrl:
                self.scaleBy(x=s, center=center)
            elif shift:
                self.scroll_time(-0.15 * notch)
            else:
                self.scaleBy(y=s, center=center)
        ev.accept()

    def mouseClickEvent(self, ev):
        if self.tool == "zoom":
            if ev.button() == _MB.RightButton:
                self.sigResetView.emit()                       # right click = zoom back out
            ev.accept()
            return
        if self.tool == "hand" or not self.picking_enabled:
            ev.accept()                                        # a click with the hand never picks
            return
        if ev.button() == _MB.LeftButton:
            i = self._hit(ev.scenePos())
            if i is None:
                p = self.mapSceneToView(ev.scenePos())
                self.sigAdd.emit(p.x(), p.y())
            else:
                self.sigSelect.emit(i)
            ev.accept()
        elif ev.button() == _MB.RightButton:
            i = self._hit(ev.scenePos())
            if i is not None:
                self.sigDelete.emit(i)
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        left = ev.button() == _MB.LeftButton
        if self.tool == "zoom" and left and axis is None:
            return super().mouseDragEvent(ev, axis)            # pyqtgraph rubber-band zoom (RectMode)
        pan = (self.tool == "hand" and left) or ev.button() == _MB.MiddleButton or axis is not None
        if pan:
            if ev.isStart() and self.tool == "hand":
                self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ClosedHandCursor))
            if ev.isFinish() and self.tool == "hand":
                self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.OpenHandCursor))
            # vertical only unless the horizontal axis is zoomed in: no sideways drift of the velocity window
            return super().mouseDragEvent(ev, 1 if axis is None and not self.x_is_zoomed() else axis)
        if self.tool == "pick" and left and self.picking_enabled:
            if ev.isStart():
                self._drag_idx = self._hit(ev.buttonDownScenePos())
            if self._drag_idx is not None:
                p = self.mapSceneToView(ev.scenePos())
                self.sigMove.emit(self._drag_idx, p.x(), p.y(), bool(ev.isFinish()))
                ev.accept()
                if ev.isFinish():
                    self._drag_idx = None
                return
        ev.accept()                                            # a drag on empty space in the pick tool does nothing


class OffsetAxis(pg.AxisItem):
    """Bottom axis of the gather panel: x is the trace number (sorted by offset); labels show the offset in m."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offsets = None

    def tickStrings(self, values, scale, spacing):
        if self.offsets is None or len(self.offsets) == 0:
            return super().tickStrings(values, scale, spacing)
        out = []
        for v in values:
            i = int(round(v))
            out.append(f"{self.offsets[i]:.0f}" if 0 <= i < len(self.offsets) else "")
        return out


class StripAxis(pg.AxisItem):
    """Top axis of the stack-segment panel. x is a strip coordinate. Mode 'velocity': ticks at segment centres labelled with the
    segment's constant velocity (m/s). Mode 'trace' (GeoEast style): a '1' at the first trace of every segment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centres = None
        self.labels = None
        self.starts = None
        self.mode = "velocity"

    def tickValues(self, minVal, maxVal, size):
        pos = self.starts if self.mode == "trace" else self.centres
        if pos is None:
            return super().tickValues(minVal, maxVal, size)
        c = [x for x in pos if minVal <= x <= maxVal]
        step = max(1, int(np.ceil(len(c) / max(2, int(size / 45)))))
        return [(1.0, c[::step])]

    def tickStrings(self, values, scale, spacing):
        if self.centres is None:
            return super().tickStrings(values, scale, spacing)
        if self.mode == "trace":
            return ["1" for _ in values]
        cen = np.asarray(self.centres)
        return [f"{self.labels[int(np.argmin(np.abs(cen - v)))]:.0f}" for v in values]


def make_lut(name: str = "jet", gamma: float = 1.0, n: int = 256) -> np.ndarray:
    """Colour table for the spectrum: jet, viridis, inferno, magma, gray, seismic (blue-white-red). gamma > 1 darkens the
    low values (more contrast at the top), gamma < 1 lifts them."""
    if name == "jet":
        lut = jet_lut(n)
    elif name == "gray":
        lut = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]).getLookupTable(0.0, 1.0, n)
    elif name == "seismic":
        lut = pg.ColorMap([0.0, 0.5, 1.0], [(0, 0, 200), (255, 255, 255), (200, 0, 0)]).getLookupTable(0.0, 1.0, n)
    else:
        lut = pg.colormap.get(name).getLookupTable(0.0, 1.0, n)
    idx = np.clip((np.linspace(0.0, 1.0, n) ** max(gamma, 1e-3)) * (n - 1), 0, n - 1).astype(int)
    return np.asarray(lut)[idx]


CMAP_NAMES = ["jet", "viridis", "inferno", "magma", "gray", "seismic"]
