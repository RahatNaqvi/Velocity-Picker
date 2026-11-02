"""SQLite project database: velocity functions with immutable versions, provenance, drafts.

Design rules
  * A velocity function has a stable identity: (location, velocity_type).
  * Every save creates a NEW version; picks of an existing version are never changed (approved or not).
    Only review status and note of a version may be updated (``set_review``).
  * Provenance records how a version came to be: manual | copied | interpolated | restored | imported.
  * A project holds one or more NAMED VELOCITIES ("velocity sets", e.g. ``velo1``). Each named velocity has its own
    functions and version history per location; the same CMP can carry different functions in different names.
  * "Overwrite" (``overwrite_version``) replaces the picks of an existing version in place, but never for an approved
    version, and the old picks are kept in ``version_audit``. ``save_version`` always creates a new version.
  * Schema changes go through ordered migrations keyed by ``PRAGMA user_version``; a backup copy of the
    file is made before migrating an existing database.
  * Units: time ms, velocity m/s.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

STATUSES = ("needs_review", "picked", "reviewed", "approved")
PROVENANCE = ("manual", "copied", "interpolated", "restored", "imported")
VELOCITY_TYPES = ("stacking", "rms", "interval")

_V1 = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE source_file (
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, size_bytes INTEGER, mtime_ns INTEGER,
    headers_sha256 TEXT, n_traces INTEGER, samples_per_trace INTEGER, dt_us INTEGER, sample_format INTEGER,
    added_utc TEXT NOT NULL, UNIQUE(path, headers_sha256));
CREATE TABLE location (
    id INTEGER PRIMARY KEY, source_file_id INTEGER NOT NULL REFERENCES source_file(id),
    line INTEGER NOT NULL DEFAULT 0, cmp INTEGER NOT NULL, x REAL, y REAL,
    UNIQUE(source_file_id, line, cmp));
CREATE TABLE analysis_config (
    id INTEGER PRIMARY KEY, config_hash TEXT NOT NULL UNIQUE, config_json TEXT NOT NULL, created_utc TEXT NOT NULL);
CREATE TABLE velocity_function (
    id INTEGER PRIMARY KEY, location_id INTEGER NOT NULL REFERENCES location(id),
    velocity_type TEXT NOT NULL, UNIQUE(location_id, velocity_type));
CREATE TABLE function_version (
    id INTEGER PRIMARY KEY, function_id INTEGER NOT NULL REFERENCES velocity_function(id),
    version_no INTEGER NOT NULL, parent_version_id INTEGER REFERENCES function_version(id),
    created_utc TEXT NOT NULL, provenance TEXT NOT NULL, source_info_json TEXT,
    analysis_config_id INTEGER REFERENCES analysis_config(id),
    status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', UNIQUE(function_id, version_no));
CREATE TABLE pick (
    version_id INTEGER NOT NULL REFERENCES function_version(id), ord INTEGER NOT NULL,
    time_ms REAL NOT NULL, velocity REAL NOT NULL, PRIMARY KEY(version_id, ord));
CREATE TABLE draft (
    location_id INTEGER NOT NULL, velocity_type TEXT NOT NULL, picks_json TEXT NOT NULL,
    provenance TEXT NOT NULL, source_info_json TEXT, updated_utc TEXT NOT NULL,
    PRIMARY KEY(location_id, velocity_type));
"""
_V2 = """
CREATE TABLE velocity_set (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, description TEXT NOT NULL DEFAULT '',
    velocity_type TEXT NOT NULL DEFAULT 'stacking', created_utc TEXT NOT NULL, modified_utc TEXT NOT NULL);
INSERT INTO velocity_set(id, name, description, velocity_type, created_utc, modified_utc)
    VALUES (1, 'default', 'Created automatically (functions saved before named velocities existed live here).',
            'stacking', strftime('%Y-%m-%dT%H:%M:%S+00:00','now'), strftime('%Y-%m-%dT%H:%M:%S+00:00','now'));
CREATE TABLE velocity_function_new (
    id INTEGER PRIMARY KEY, set_id INTEGER NOT NULL REFERENCES velocity_set(id),
    location_id INTEGER NOT NULL REFERENCES location(id), velocity_type TEXT NOT NULL,
    UNIQUE(set_id, location_id, velocity_type));
INSERT INTO velocity_function_new(id, set_id, location_id, velocity_type)
    SELECT id, 1, location_id, velocity_type FROM velocity_function;
DROP TABLE velocity_function;
ALTER TABLE velocity_function_new RENAME TO velocity_function;
CREATE TABLE draft_new (
    location_id INTEGER NOT NULL, velocity_type TEXT NOT NULL, set_id INTEGER NOT NULL DEFAULT 1,
    picks_json TEXT NOT NULL, provenance TEXT NOT NULL, source_info_json TEXT, updated_utc TEXT NOT NULL,
    PRIMARY KEY(location_id, velocity_type, set_id));
INSERT INTO draft_new(location_id, velocity_type, set_id, picks_json, provenance, source_info_json, updated_utc)
    SELECT location_id, velocity_type, 1, picks_json, provenance, source_info_json, updated_utc FROM draft;
DROP TABLE draft;
ALTER TABLE draft_new RENAME TO draft;
CREATE TABLE version_audit (
    id INTEGER PRIMARY KEY, version_id INTEGER NOT NULL REFERENCES function_version(id), changed_utc TEXT NOT NULL,
    action TEXT NOT NULL, old_picks_json TEXT NOT NULL, old_status TEXT NOT NULL, old_note TEXT NOT NULL,
    old_provenance TEXT NOT NULL);
"""
MIGRATIONS = {1: _V1, 2: _V2}      # version -> script; add new numbered scripts here, never edit old ones
FK_OFF = {2}                       # migrations that rebuild tables must run with foreign keys off
LATEST = max(MIGRATIONS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectDB:
    def __init__(self, path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        self.active_set_id = 1
        self.source_moved_from = None
        saved = self.get_meta("active_set")
        if saved is not None and self.get_set(int(saved)) is not None:
            self.active_set_id = int(saved)

    # ------------------------------------------------------------------ schema
    def schema_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self) -> None:
        v = self.schema_version()
        if v > LATEST:
            raise RuntimeError(f"Database schema v{v} is newer than this program (v{LATEST}).")
        if v == LATEST:
            return
        if v > 0:
            self.conn.commit()
            shutil.copyfile(self.path, self.path.with_name(self.path.name + f".bak-v{v}"))
        for n in range(v + 1, LATEST + 1):
            if n in FK_OFF:
                self.conn.execute("PRAGMA foreign_keys=OFF")
            try:
                self.conn.executescript("BEGIN;" + MIGRATIONS[n] + ";COMMIT;")
                bad = self.conn.execute("PRAGMA foreign_key_check").fetchall()
                if bad:
                    raise RuntimeError(f"Migration to schema v{n} left {len(bad)} broken reference(s); restore the .bak file.")
            except Exception:
                self.conn.rollback()
                raise
            finally:
                if n in FK_OFF:
                    self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute(f"PRAGMA user_version={n}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default=None):
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set_meta(self, key: str, value) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                              (key, json.dumps(value)))

    # ------------------------------------------------------------------ sources / locations
    def add_source(self, path, size_bytes, mtime_ns, headers_sha256, n_traces, samples_per_trace, dt_us,
                   sample_format) -> int:
        """Register the SEG-Y and return its id. A file's identity is its CONTENT (hash of all trace headers + size), not
        its path: if the same data is opened from another folder / mount point, the existing entry is re-attached (its
        path is updated and ``source_moved_from`` holds the old path) so no picks are lost. Entries for the same content
        that older versions created per path are merged (see ``_merge_sources``)."""
        self.source_moved_from = None
        path = str(path)
        exact = self.conn.execute("SELECT * FROM source_file WHERE path=? AND headers_sha256 IS ?", (path, headers_sha256)).fetchone()
        same = []
        if headers_sha256:
            same = self.conn.execute("SELECT * FROM source_file WHERE headers_sha256=? AND size_bytes=? ORDER BY id",
                                     (headers_sha256, size_bytes)).fetchall()
        if exact is not None:
            keep = int(exact["id"])
            with self.conn:
                self.conn.execute("UPDATE source_file SET mtime_ns=?, size_bytes=? WHERE id=?", (mtime_ns, size_bytes, keep))
        elif same:
            keep = int(same[0]["id"])
            self.source_moved_from = same[0]["path"]
            with self.conn:
                self.conn.execute("UPDATE source_file SET path=?, mtime_ns=? WHERE id=?", (path, mtime_ns, keep))
        else:
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO source_file(path,size_bytes,mtime_ns,headers_sha256,n_traces,samples_per_trace,dt_us,sample_format,"
                    "added_utc) VALUES(?,?,?,?,?,?,?,?,?)",
                    (path, size_bytes, mtime_ns, headers_sha256, n_traces, samples_per_trace, dt_us, sample_format, _now()))
            keep = int(cur.lastrowid)
        if headers_sha256:
            self._merge_sources(keep)
        return keep

    def _merge_sources(self, keep_id: int):
        """Fold other source entries with identical content into ``keep_id``. Locations that exist only in the other entry
        move over; an empty duplicate location is dropped; if both hold functions for the same CMP the other entry is left
        alone (nothing is ever overwritten). Returns (locations moved, true conflicts left)."""
        keep = self.conn.execute("SELECT * FROM source_file WHERE id=?", (keep_id,)).fetchone()
        dups = self.conn.execute("SELECT id FROM source_file WHERE headers_sha256=? AND size_bytes=? AND id!=?",
                                 (keep["headers_sha256"], keep["size_bytes"], keep_id)).fetchall()
        moved = conflicts = 0
        with self.conn:
            for d in dups:
                for loc in self.conn.execute("SELECT * FROM location WHERE source_file_id=?", (d["id"],)).fetchall():
                    tgt = self.conn.execute("SELECT id FROM location WHERE source_file_id=? AND line=? AND cmp=?",
                                            (keep_id, loc["line"], loc["cmp"])).fetchone()
                    if tgt is None:
                        self.conn.execute("UPDATE location SET source_file_id=? WHERE id=?", (keep_id, loc["id"])); moved += 1
                        continue
                    n_src = self.conn.execute("SELECT COUNT(*) FROM velocity_function WHERE location_id=?", (loc["id"],)).fetchone()[0]
                    n_tgt = self.conn.execute("SELECT COUNT(*) FROM velocity_function WHERE location_id=?", (tgt["id"],)).fetchone()[0]
                    if n_src == 0:
                        self.conn.execute("DELETE FROM draft WHERE location_id=?", (loc["id"],))
                        self.conn.execute("DELETE FROM location WHERE id=?", (loc["id"],))
                    elif n_tgt == 0:
                        self.conn.execute("UPDATE velocity_function SET location_id=? WHERE location_id=?", (tgt["id"], loc["id"]))
                        self.conn.execute("UPDATE draft SET location_id=? WHERE location_id=?", (tgt["id"], loc["id"]))
                        self.conn.execute("DELETE FROM location WHERE id=?", (loc["id"],)); moved += 1
                    else:
                        conflicts += 1
                if self.conn.execute("SELECT COUNT(*) FROM location WHERE source_file_id=?", (d["id"],)).fetchone()[0] == 0:
                    self.conn.execute("DELETE FROM source_file WHERE id=?", (d["id"],))
        return moved, conflicts

    def ensure_location(self, source_id: int, line: int, cmp_id: int, x=None, y=None) -> int:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO location(source_file_id,line,cmp,x,y) VALUES(?,?,?,?,?)",
                              (source_id, line, cmp_id, x, y))
        return int(self.conn.execute("SELECT id FROM location WHERE source_file_id=? AND line=? AND cmp=?",
                                     (source_id, line, cmp_id)).fetchone()["id"])

    # ------------------------------------------------------------------ versions
    def _sid(self, set_id) -> int:
        return self.active_set_id if set_id is None else int(set_id)

    def _function_id(self, location_id: int, velocity_type: str, set_id: int) -> int:
        self.conn.execute("INSERT OR IGNORE INTO velocity_function(set_id,location_id,velocity_type) VALUES(?,?,?)",
                          (set_id, location_id, velocity_type))
        return int(self.conn.execute("SELECT id FROM velocity_function WHERE set_id=? AND location_id=? AND velocity_type=?",
                                     (set_id, location_id, velocity_type)).fetchone()["id"])

    @staticmethod
    def _check_picks(picks, velocity_type, provenance, status):
        if velocity_type not in VELOCITY_TYPES:
            raise ValueError(f"unknown velocity type {velocity_type!r}")
        if provenance not in PROVENANCE:
            raise ValueError(f"unknown provenance {provenance!r}")
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        pk = [(float(t), float(v)) for t, v in picks]
        if any(t2 <= t1 for (t1, _), (t2, _) in zip(pk, pk[1:])):
            raise ValueError("pick times must be strictly increasing")
        if any(v <= 0 for _, v in pk):
            raise ValueError("pick velocities must be > 0")
        return pk

    def _config_id(self, analysis_config):
        if analysis_config is None:
            return None
        import hashlib
        cj = json.dumps(analysis_config, sort_keys=True)
        h = hashlib.sha256(cj.encode()).hexdigest()[:16]
        self.conn.execute("INSERT OR IGNORE INTO analysis_config(config_hash,config_json,created_utc) VALUES(?,?,?)", (h, cj, _now()))
        return self.conn.execute("SELECT id FROM analysis_config WHERE config_hash=?", (h,)).fetchone()["id"]

    def _insert_version(self, location_id, pk, velocity_type, provenance, source_info, analysis_config, status, note, set_id):
        """Insert a new version (caller owns the transaction)."""
        cfg_id = self._config_id(analysis_config)
        fid = self._function_id(location_id, velocity_type, set_id)
        last = self.conn.execute("SELECT id, version_no FROM function_version WHERE function_id=? ORDER BY version_no DESC LIMIT 1",
                                 (fid,)).fetchone()
        vno = (last["version_no"] + 1) if last else 1
        cur = self.conn.execute(
            "INSERT INTO function_version(function_id,version_no,parent_version_id,created_utc,provenance,"
            "source_info_json,analysis_config_id,status,note) VALUES(?,?,?,?,?,?,?,?,?)",
            (fid, vno, last["id"] if last else None, _now(), provenance, json.dumps(source_info) if source_info else None,
             cfg_id, status, note))
        vid = int(cur.lastrowid)
        self.conn.executemany("INSERT INTO pick(version_id,ord,time_ms,velocity) VALUES(?,?,?,?)",
                              [(vid, i, t, v) for i, (t, v) in enumerate(pk)])
        self.conn.execute("DELETE FROM draft WHERE location_id=? AND velocity_type=? AND set_id=?", (location_id, velocity_type, set_id))
        self.conn.execute("UPDATE velocity_set SET modified_utc=? WHERE id=?", (_now(), set_id))
        return vid, vno

    def save_version(self, location_id: int, picks: Sequence[Tuple[float, float]], *, velocity_type: str = "stacking",
                     provenance: str = "manual", source_info: Optional[dict] = None, analysis_config: Optional[dict] = None,
                     status: str = "picked", note: str = "", set_id: Optional[int] = None) -> Tuple[int, int]:
        """Create a new version in the named velocity (default: the active one). Returns (version_id, version_no).
        Never modifies earlier versions."""
        pk = self._check_picks(picks, velocity_type, provenance, status)
        sid = self._sid(set_id)
        try:
            with self.conn:
                return self._insert_version(location_id, pk, velocity_type, provenance, source_info, analysis_config, status, note, sid)
        except sqlite3.Error as e:
            raise RuntimeError(f"Database error while saving: {e}") from e

    def overwrite_version(self, version_id: int, picks: Sequence[Tuple[float, float]], *, provenance: Optional[str] = None,
                          source_info: Optional[dict] = None, analysis_config: Optional[dict] = None,
                          status: Optional[str] = None, note: Optional[str] = None) -> int:
        """Replace the picks of an existing version IN PLACE. Approved versions are protected (ValueError); the old
        picks, status, note and provenance are kept in ``version_audit``. Returns the version number."""
        old = self.get_version(version_id)
        if old is None:
            raise KeyError(version_id)
        if old["status"] == "approved":
            raise ValueError("This version is approved and protected. Save a new version instead (or change its review status first).")
        new_status = status or old["status"]
        new_prov = provenance or old["provenance"]
        pk = self._check_picks(picks, "stacking", new_prov, new_status)
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO version_audit(version_id,changed_utc,action,old_picks_json,old_status,old_note,old_provenance) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (version_id, _now(), "overwrite", json.dumps([[t, v] for t, v in self.get_picks(version_id)]),
                     old["status"], old["note"], old["provenance"]))
                self.conn.execute("DELETE FROM pick WHERE version_id=?", (version_id,))
                self.conn.executemany("INSERT INTO pick(version_id,ord,time_ms,velocity) VALUES(?,?,?,?)",
                                      [(version_id, i, t, v) for i, (t, v) in enumerate(pk)])
                cfg_id = self._config_id(analysis_config) if analysis_config is not None else old["analysis_config_id"]
                self.conn.execute(
                    "UPDATE function_version SET provenance=?, source_info_json=?, analysis_config_id=?, status=?, note=? WHERE id=?",
                    (new_prov, json.dumps(source_info) if source_info else old["source_info_json"], cfg_id, new_status,
                     old["note"] if note is None else note, version_id))
                fid = old["function_id"]
                sid = self.conn.execute("SELECT set_id FROM velocity_function WHERE id=?", (fid,)).fetchone()["set_id"]
                self.conn.execute("UPDATE velocity_set SET modified_utc=? WHERE id=?", (_now(), sid))
                loc = self.conn.execute("SELECT location_id, velocity_type FROM velocity_function WHERE id=?", (fid,)).fetchone()
                self.conn.execute("DELETE FROM draft WHERE location_id=? AND velocity_type=? AND set_id=?",
                                  (loc["location_id"], loc["velocity_type"], sid))
        except sqlite3.Error as e:
            raise RuntimeError(f"Database error while overwriting: {e}") from e
        return int(old["version_no"])

    def audit_log(self, version_id: int):
        return [{"changed_utc": r["changed_utc"], "action": r["action"], "old_picks": [tuple(p) for p in json.loads(r["old_picks_json"])],
                 "old_status": r["old_status"], "old_note": r["old_note"], "old_provenance": r["old_provenance"]}
                for r in self.conn.execute("SELECT * FROM version_audit WHERE version_id=? ORDER BY id", (version_id,))]

    # ------------------------------------------------------------------ named velocities
    def get_set(self, set_id: int):
        return self.conn.execute("SELECT * FROM velocity_set WHERE id=?", (int(set_id),)).fetchone()

    def set_by_name(self, name: str):
        return self.conn.execute("SELECT * FROM velocity_set WHERE name=?", (name.strip(),)).fetchone()

    def list_sets(self):
        return self.conn.execute(
            "SELECT s.*, (SELECT COUNT(DISTINCT f.location_id) FROM velocity_function f JOIN function_version v ON v.function_id=f.id "
            "WHERE f.set_id=s.id) AS n_functions, (SELECT COUNT(*) FROM velocity_function f JOIN function_version v ON v.function_id=f.id "
            "WHERE f.set_id=s.id) AS n_versions FROM velocity_set s ORDER BY s.id").fetchall()

    @staticmethod
    def _clean_name(name: str) -> str:
        n = (name or "").strip()
        if not n:
            raise ValueError("The velocity name must not be empty.")
        if len(n) > 80:
            raise ValueError("The velocity name is too long (80 characters maximum).")
        if any(c in n for c in '/\\:*?"<>|\n\r\t'):
            raise ValueError("The velocity name must not contain / \\ : * ? \" < > | or line breaks (it is used in file names).")
        return n

    def _create_set(self, name, description, velocity_type):
        n = self._clean_name(name)
        if self.set_by_name(n) is not None:
            raise ValueError(f"A velocity name '{n}' already exists in this project.")
        cur = self.conn.execute("INSERT INTO velocity_set(name,description,velocity_type,created_utc,modified_utc) VALUES(?,?,?,?,?)",
                                (n, description, velocity_type, _now(), _now()))
        return int(cur.lastrowid)

    def create_set(self, name: str, description: str = "", velocity_type: str = "stacking") -> int:
        with self.conn:
            return self._create_set(name, description, velocity_type)

    def rename_set(self, set_id: int, new_name: str) -> None:
        n = self._clean_name(new_name)
        other = self.set_by_name(n)
        if other is not None and other["id"] != int(set_id):
            raise ValueError(f"A velocity name '{n}' already exists in this project.")
        with self.conn:
            self.conn.execute("UPDATE velocity_set SET name=?, modified_utc=? WHERE id=?", (n, _now(), int(set_id)))

    def set_description(self, set_id: int, description: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE velocity_set SET description=?, modified_utc=? WHERE id=?", (description, _now(), int(set_id)))

    def set_active(self, set_id: int) -> None:
        if self.get_set(set_id) is None:
            raise KeyError(set_id)
        self.active_set_id = int(set_id)
        self.set_meta("active_set", int(set_id))

    def copy_set(self, src_id: int, new_name: str, description: str = "", keep_status: bool = False) -> int:
        """New named velocity containing the LATEST version of every function of ``src_id`` (provenance 'copied',
        status 'needs_review' unless ``keep_status``). The source is untouched; all-or-nothing."""
        src = self.get_set(src_id)
        if src is None:
            raise KeyError(src_id)
        with self.conn:
            new_id = self._create_set(new_name, description or f"Copy of '{src['name']}'", src["velocity_type"])
            funcs = self.conn.execute("SELECT id, location_id, velocity_type FROM velocity_function WHERE set_id=?", (int(src_id),)).fetchall()
            for f in funcs:
                v = self.conn.execute("SELECT * FROM function_version WHERE function_id=? ORDER BY version_no DESC LIMIT 1", (f["id"],)).fetchone()
                if v is None:
                    continue
                picks = self.get_picks(v["id"])
                self._insert_version(f["location_id"], picks, f["velocity_type"], "copied",
                                     {"from_set": src["name"], "from_version_no": v["version_no"]}, None,
                                     v["status"] if keep_status else "needs_review", f"copied from '{src['name']}' v{v['version_no']}", new_id)
        return new_id

    def latest_version(self, location_id: int, velocity_type: str = "stacking", set_id: Optional[int] = None):
        return self.conn.execute(
            "SELECT v.* FROM function_version v JOIN velocity_function f ON f.id=v.function_id "
            "WHERE f.location_id=? AND f.velocity_type=? AND f.set_id=? ORDER BY v.version_no DESC LIMIT 1",
            (location_id, velocity_type, self._sid(set_id))).fetchone()

    def versions(self, location_id: int, velocity_type: str = "stacking", set_id: Optional[int] = None):
        return self.conn.execute(
            "SELECT v.*, (SELECT COUNT(*) FROM pick p WHERE p.version_id=v.id) AS n_picks FROM function_version v "
            "JOIN velocity_function f ON f.id=v.function_id WHERE f.location_id=? AND f.velocity_type=? AND f.set_id=? "
            "ORDER BY v.version_no", (location_id, velocity_type, self._sid(set_id))).fetchall()

    def get_picks(self, version_id: int) -> List[Tuple[float, float]]:
        return [(r["time_ms"], r["velocity"]) for r in self.conn.execute(
            "SELECT time_ms, velocity FROM pick WHERE version_id=? ORDER BY ord", (version_id,))]

    def get_version(self, version_id: int):
        return self.conn.execute("SELECT * FROM function_version WHERE id=?", (version_id,)).fetchone()

    def restore_version(self, version_id: int, location_id: int, velocity_type: str = "stacking",
                        set_id: Optional[int] = None) -> Tuple[int, int]:
        """Restore = save a new version with the old picks (history is kept; needs review)."""
        old = self.get_version(version_id)
        if old is None:
            raise KeyError(version_id)
        return self.save_version(location_id, self.get_picks(version_id), velocity_type=velocity_type,
                                 provenance="restored", source_info={"restored_from_version_id": version_id,
                                                                     "restored_from_version_no": old["version_no"]},
                                 status="needs_review", note=f"restored from v{old['version_no']}", set_id=set_id)

    def set_review(self, version_id: int, status: str, note: Optional[str] = None) -> None:
        """Only status/note change; picks stay untouched."""
        if status not in STATUSES:
            raise ValueError(status)
        with self.conn:
            if note is None:
                self.conn.execute("UPDATE function_version SET status=? WHERE id=?", (status, version_id))
            else:
                self.conn.execute("UPDATE function_version SET status=?, note=? WHERE id=?", (status, note, version_id))

    def compare_versions(self, version_a: int, version_b: int) -> List[dict]:
        """Time-aligned comparison: velocity of each version at the union of pick times (linear interp)."""
        import numpy as np
        a, b = self.get_picks(version_a), self.get_picks(version_b)
        ts = sorted({t for t, _ in a} | {t for t, _ in b})
        f = lambda p, t: float(np.interp(t, [x for x, _ in p], [y for _, y in p])) if p else float("nan")  # noqa: E731
        return [{"time_ms": t, "v_a": f(a, t), "v_b": f(b, t), "diff": f(b, t) - f(a, t)} for t in ts]

    def location_status(self, source_id: int, line: int = 0, velocity_type: str = "stacking",
                        set_id: Optional[int] = None) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        for r in self.conn.execute("SELECT id, cmp FROM location WHERE source_file_id=? AND line=?", (source_id, line)):
            v = self.latest_version(r["id"], velocity_type, set_id)
            if v is not None:
                out[r["cmp"]] = {"status": v["status"], "version_no": v["version_no"], "provenance": v["provenance"]}
        return out

    # ------------------------------------------------------------------ drafts (autosave)
    def save_draft(self, location_id: int, picks, provenance: str = "manual", source_info=None,
                   velocity_type: str = "stacking", set_id: Optional[int] = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO draft(location_id,velocity_type,set_id,picks_json,provenance,source_info_json,updated_utc) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(location_id,velocity_type,set_id) DO UPDATE SET picks_json=excluded.picks_json,"
                "provenance=excluded.provenance, source_info_json=excluded.source_info_json, updated_utc=excluded.updated_utc",
                (location_id, velocity_type, self._sid(set_id), json.dumps([[float(t), float(v)] for t, v in picks]), provenance,
                 json.dumps(source_info) if source_info else None, _now()))

    def load_draft(self, location_id: int, velocity_type: str = "stacking", set_id: Optional[int] = None):
        r = self.conn.execute("SELECT * FROM draft WHERE location_id=? AND velocity_type=? AND set_id=?",
                              (location_id, velocity_type, self._sid(set_id))).fetchone()
        if r is None:
            return None
        return {"picks": [tuple(p) for p in json.loads(r["picks_json"])], "provenance": r["provenance"],
                "source_info": json.loads(r["source_info_json"]) if r["source_info_json"] else None,
                "updated_utc": r["updated_utc"]}

    def clear_draft(self, location_id: int, velocity_type: str = "stacking", set_id: Optional[int] = None) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM draft WHERE location_id=? AND velocity_type=? AND set_id=?",
                              (location_id, velocity_type, self._sid(set_id)))
