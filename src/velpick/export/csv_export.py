"""CSV export of saved velocity functions (latest version per location). Never alters the database.

Columns: line, cmp, time_ms, velocity_mps, velocity_type, version_no, status, provenance, velocity_name.
This is velpick's own documented CSV, not a package-specific format (those need real spec/sample files).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Iterable, Optional

from velpick.database import ProjectDB


def export_csv(db: ProjectDB, source_id: int, path, *, line: int = 0, velocity_type: str = "stacking",
               statuses: Optional[Iterable[str]] = None, set_id: Optional[int] = None) -> int:
    """Write the latest versions of one NAMED velocity (default: the active one), optionally only given statuses.
    Returns the number of functions written."""
    keep = set(statuses) if statuses else None
    sid = db.active_set_id if set_id is None else int(set_id)
    vset = db.get_set(sid)
    n = 0
    with open(path, "w", newline="") as f:
        f.write(f"# velpick velocity function export\n# generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        f.write(f"# velocity_name: {vset['name']}\n# velocity_type: {velocity_type} (time ms, velocity m/s)\n# source_file_id: {source_id}\n")
        w = csv.writer(f)
        w.writerow(["line", "cmp", "time_ms", "velocity_mps", "velocity_type", "version_no", "status", "provenance", "velocity_name"])
        rows = db.conn.execute("SELECT id, cmp FROM location WHERE source_file_id=? AND line=? ORDER BY cmp", (source_id, line)).fetchall()
        for r in rows:
            v = db.latest_version(r["id"], velocity_type, sid)
            if v is None or (keep and v["status"] not in keep):
                continue
            for t, vel in db.get_picks(v["id"]):
                w.writerow([line, r["cmp"], f"{t:.1f}", f"{vel:.1f}", velocity_type, v["version_no"], v["status"], v["provenance"], vset["name"]])
            n += 1
    return n
