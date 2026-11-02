"""CMP navigation over the analysis sequence (actual CMP ids; gaps are reported, never assumed away)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def regular_sequence(available: Sequence[int], start: int, end: int, interval: int) -> Tuple[List[int], List[int]]:
    """Analysis locations start, start+interval, ... up to end, restricted to ids that exist.
    Returns (ids_present, ids_requested_but_missing)."""
    if interval < 1:
        raise ValueError("interval must be >= 1")
    have = set(int(a) for a in available)
    wanted = list(range(int(start), int(end) + 1, int(interval)))
    return [c for c in wanted if c in have], [c for c in wanted if c not in have]


class Navigator:
    def __init__(self, ids: Sequence[int], current: Optional[int] = None):
        self.ids = sorted(int(i) for i in ids)
        if not self.ids:
            raise ValueError("empty analysis sequence")
        self.current = self.ids[0] if current is None else self._check(current)

    def _check(self, cmp_id: int) -> int:
        if cmp_id not in self.ids:
            raise KeyError(f"CMP {cmp_id} is not in the analysis sequence")
        return cmp_id

    @property
    def position(self) -> int:
        return self.ids.index(self.current)

    def previous_id(self) -> Optional[int]:
        return self.ids[self.position - 1] if self.position > 0 else None

    def next_id(self) -> Optional[int]:
        return self.ids[self.position + 1] if self.position < len(self.ids) - 1 else None

    def next(self) -> Optional[int]:
        n = self.next_id()
        if n is not None:
            self.current = n
        return n

    def previous(self) -> Optional[int]:
        p = self.previous_id()
        if p is not None:
            self.current = p
        return p

    def jump(self, cmp_id: int) -> int:
        self.current = self._check(int(cmp_id))
        return self.current
