"""Pick editing model (no GUI): add / move / delete, undo / redo, warnings, copy sources.

Times in ms, velocities in m/s. Pick times are kept strictly increasing; moving a pick is clamped between its
neighbours so indices stay stable during a drag. Warnings never block editing (geological variation is allowed).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Pick = Tuple[float, float]


class PickModel:
    MIN_DT_MS = 1.0

    def __init__(self) -> None:
        self._picks: List[Pick] = []
        self._undo: List[tuple] = []
        self._redo: List[tuple] = []
        self.provenance = "manual"
        self.source_info: Optional[dict] = None
        self._saved: Optional[tuple] = tuple()

    # ------------------------------------------------------------------ state
    @property
    def picks(self) -> List[Pick]:
        return list(self._picks)

    def __len__(self) -> int:
        return len(self._picks)

    @property
    def dirty(self) -> bool:
        return tuple(self._picks) != self._saved

    def mark_saved(self) -> None:
        self._saved = tuple(self._picks)

    def _snapshot(self) -> tuple:
        return (tuple(self._picks), self.provenance, self.source_info)

    def checkpoint(self) -> None:
        """Push an undo state (call once at the start of a drag)."""
        self._undo.append(self._snapshot())
        self._redo.clear()

    def set_all(self, picks: Sequence[Pick], provenance: str = "manual", source_info: Optional[dict] = None,
                saved: bool = False, record: bool = True) -> None:
        if record:
            self.checkpoint()
        pk = sorted((float(t), float(v)) for t, v in picks)
        self._picks = pk
        self.provenance, self.source_info = provenance, source_info
        if saved:
            self.mark_saved()

    def reset(self, picks: Sequence[Pick] = (), provenance: str = "manual", source_info: Optional[dict] = None) -> None:
        """Load without undo history (navigating to another location)."""
        self._undo.clear(); self._redo.clear()
        self._picks = sorted((float(t), float(v)) for t, v in picks)
        self.provenance, self.source_info = provenance, source_info
        self.mark_saved()

    # ------------------------------------------------------------------ edits
    def add(self, t: float, v: float) -> int:
        """Insert a pick; if one exists within MIN_DT_MS in time its velocity is replaced. Returns its index."""
        if v <= 0 or t <= 0:
            raise ValueError("time and velocity must be > 0")
        self.checkpoint()
        for i, (ti, _) in enumerate(self._picks):
            if abs(ti - t) < self.MIN_DT_MS:
                self._picks[i] = (ti, float(v))
                return i
        self._picks.append((float(t), float(v)))
        self._picks.sort()
        return self._picks.index((float(t), float(v)))

    def move(self, i: int, t: float, v: float, checkpoint: bool = False) -> int:
        """Move pick i; time is clamped between neighbours (and above 0) so the index does not change.
        A non-positive velocity is rejected."""
        if v <= 0:
            raise ValueError("velocity must be > 0")
        if checkpoint:
            self.checkpoint()
        lo = self._picks[i - 1][0] + self.MIN_DT_MS if i > 0 else 0.0 + self.MIN_DT_MS
        hi = self._picks[i + 1][0] - self.MIN_DT_MS if i < len(self._picks) - 1 else float("inf")
        self._picks[i] = (float(min(max(t, lo), hi)), float(v))
        return i

    def delete(self, i: int) -> None:
        self.checkpoint()
        del self._picks[i]

    def clear(self) -> None:
        if self._picks:
            self.checkpoint()
            self._picks = []

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        p, self.provenance, self.source_info = self._undo.pop()
        self._picks = list(p)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        p, self.provenance, self.source_info = self._redo.pop()
        self._picks = list(p)
        return True

    # ------------------------------------------------------------------ queries
    def nearest_index(self, t: float, v: float, t_scale: float = 1.0, v_scale: float = 1.0,
                      max_dist: float = 1.0) -> Optional[int]:
        """Index of the pick nearest (t,v) in scaled units, or None if farther than max_dist."""
        if not self._picks:
            return None
        d = [np.hypot((pt - t) / t_scale, (pv - v) / v_scale) for pt, pv in self._picks]
        k = int(np.argmin(d))
        return k if d[k] <= max_dist else None

    def velocity_at(self, t) -> np.ndarray:
        """Piecewise-linear velocity function, constant outside the pick range (NaN if no picks)."""
        t = np.asarray(t, dtype=float)
        if not self._picks:
            return np.full(t.shape, np.nan)
        pt, pv = zip(*self._picks)
        return np.interp(t, pt, pv)

    def warnings(self, vmin: Optional[float] = None, vmax: Optional[float] = None, max_jump_pct: float = 25.0,
                 t_min: Optional[float] = None, t_max: Optional[float] = None) -> List[str]:
        """Suspicious-value messages. Informational only; nothing is blocked or changed."""
        w: List[str] = []
        p = self._picks
        if len(p) == 1:
            w.append("Only one pick: the function is constant in time.")
        for (t1, v1), (t2, v2) in zip(p, p[1:]):
            if t2 <= t1:
                w.append(f"Non-increasing times at {t1:.0f} ms / {t2:.0f} ms.")
            jump = abs(v2 - v1) / v1 * 100
            if jump > max_jump_pct:
                w.append(f"Large velocity jump {v1:.0f} -> {v2:.0f} m/s ({jump:.0f} %) between {t1:.0f} and {t2:.0f} ms.")
            elif v2 < v1:
                w.append(f"Velocity reversal {v1:.0f} -> {v2:.0f} m/s between {t1:.0f} and {t2:.0f} ms (may be geological).")
        for t, v in p:
            if vmin is not None and v < vmin or vmax is not None and v > vmax:
                w.append(f"Pick {t:.0f} ms / {v:.0f} m/s is outside the analysed velocity range {vmin:.0f}-{vmax:.0f}.")
            if t_min is not None and t < t_min or t_max is not None and t > t_max:
                w.append(f"Pick at {t:.0f} ms is outside the analysed time range {t_min:.0f}-{t_max:.0f}.")
        return w
