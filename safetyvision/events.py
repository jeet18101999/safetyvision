"""Event log (SQLite) with snapshots, operator review (correct / false alarm), occupancy history and export."""
import csv
import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import cv2


def day_start(ts=None) -> float:
    lt = time.localtime(ts or time.time())
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


class EventStore:
    def __init__(self, db_path: Path, snapshot_dir: Path):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, camera TEXT, type TEXT, message TEXT,
            snapshot TEXT, verified INTEGER, extra TEXT)""")
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(events)")}
        for col, typ in (("raw", "TEXT"), ("feedback", "TEXT"), ("feedback_ts", "REAL")):
            if col not in cols:
                self.db.execute(f"ALTER TABLE events ADD COLUMN {col} {typ}")
        self.db.execute("CREATE TABLE IF NOT EXISTS occupancy (ts REAL, camera TEXT, people INTEGER)")
        self.db.execute("CREATE INDEX IF NOT EXISTS occ_ts ON occupancy(ts)")
        self.db.execute("CREATE INDEX IF NOT EXISTS ev_ts ON events(ts)")
        self.db.commit()

    # ---------------------------------------------------------------- events
    def add(self, camera, etype, message, frame=None, verified=None, extra=None, raw_frame=None) -> dict:
        ts = time.time()
        snap = raw = ""
        if frame is not None or raw_frame is not None:
            day = self.snapshot_dir / time.strftime("%Y-%m-%d", time.localtime(ts))
            day.mkdir(parents=True, exist_ok=True)
            base = f"{time.strftime('%H%M%S', time.localtime(ts))}_{int(ts * 1000) % 1000:03d}_{camera}_{etype}"
            if frame is not None:
                snap = str(day / f"{base}.jpg")
                cv2.imwrite(snap, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if raw_frame is not None:  # unannotated copy, used for learning from the operator's review
                raw = str(day / f"{base}_raw.jpg")
                cv2.imwrite(raw, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO events (ts, camera, type, message, snapshot, verified, extra, raw) VALUES (?,?,?,?,?,?,?,?)",
                (ts, camera, etype, message, snap, None if verified is None else int(verified),
                 json.dumps(extra or {}), raw))
            self.db.commit()
            eid = cur.lastrowid
        return {"id": eid, "ts": ts, "camera": camera, "type": etype, "message": message, "snapshot": snap,
                "raw": raw, "verified": verified, "extra": extra or {}, "feedback": None}

    _COLS = "id, ts, camera, type, message, snapshot, verified, extra, raw, feedback"

    def _row(self, r) -> dict:
        return {"id": r[0], "ts": r[1], "camera": r[2], "type": r[3], "message": r[4], "snapshot": r[5],
                "verified": None if r[6] is None else bool(r[6]), "extra": json.loads(r[7] or "{}"),
                "raw": r[8] or "", "feedback": r[9]}

    def get(self, eid) -> dict | None:
        with self._lock:
            r = self.db.execute(f"SELECT {self._COLS} FROM events WHERE id = ?", (eid,)).fetchone()
        return self._row(r) if r else None

    def recent(self, limit=500, etype=None, camera=None, review=None, since=None) -> list[dict]:
        q, args = f"SELECT {self._COLS} FROM events WHERE 1=1", []
        if etype:
            q += " AND type = ?"
            args.append(etype)
        if camera:
            q += " AND camera = ?"
            args.append(camera)
        if review == "unreviewed":
            q += " AND feedback IS NULL"
        elif review:
            q += " AND feedback = ?"
            args.append(review)
        if since:
            q += " AND ts >= ?"
            args.append(since)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.db.execute(q, args).fetchall()
        return [self._row(r) for r in rows]

    def set_feedback(self, eid, feedback):
        with self._lock:
            self.db.execute("UPDATE events SET feedback = ?, feedback_ts = ? WHERE id = ?", (feedback, time.time(), eid))
            self.db.commit()

    def counts_by_type(self, since) -> dict:
        with self._lock:
            rows = self.db.execute("SELECT type, COUNT(*) FROM events WHERE ts >= ? GROUP BY type", (since,)).fetchall()
        return dict(rows)

    def unreviewed_count(self, since=0) -> int:
        with self._lock:
            return self.db.execute("SELECT COUNT(*) FROM events WHERE feedback IS NULL AND ts >= ? "
                                   "AND type NOT IN ('CAMERA_OFFLINE', 'SYSTEM')", (since,)).fetchone()[0]

    def accuracy(self, since=0) -> dict:
        """Operator-reviewed precision per event type: {type: (correct, false)}"""
        with self._lock:
            rows = self.db.execute("SELECT type, feedback, COUNT(*) FROM events WHERE feedback IS NOT NULL AND ts >= ? "
                                   "GROUP BY type, feedback", (since,)).fetchall()
        out = {}
        for t, fb, n in rows:
            c, f = out.get(t, (0, 0))
            out[t] = (c + n, f) if fb == "correct" else (c, f + n)
        return out

    # ---------------------------------------------------------------- occupancy
    def log_occupancy(self, counts: dict):
        ts = time.time()
        with self._lock:
            self.db.executemany("INSERT INTO occupancy VALUES (?,?,?)", [(ts, c, int(n)) for c, n in counts.items()])
            self.db.commit()

    def occupancy_series(self, since, bucket=900) -> list[tuple[float, int]]:
        """Total people on site, max per time bucket."""
        with self._lock:
            rows = self.db.execute("SELECT ts, SUM(people) FROM occupancy WHERE ts >= ? GROUP BY ts ORDER BY ts",
                                   (since,)).fetchall()
        buckets = {}
        for ts, n in rows:
            b = int(ts // bucket * bucket)
            buckets[b] = max(buckets.get(b, 0), n)
        return sorted(buckets.items())

    # ---------------------------------------------------------------- maintenance
    def export_csv(self, path, since=0):
        rows = self.recent(1_000_000, since=since)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "time", "camera", "type", "message", "ai_verified", "review", "snapshot"])
            for e in reversed(rows):
                w.writerow([e["id"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["ts"])), e["camera"],
                            e["type"], e["message"], {True: "yes", False: "no", None: ""}[e["verified"]],
                            e["feedback"] or "", e["snapshot"]])
        return len(rows)

    def purge(self, retention_days):
        if not retention_days:
            return
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            self.db.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            self.db.execute("DELETE FROM occupancy WHERE ts < ?", (cutoff,))
            self.db.commit()
        cut_day = time.strftime("%Y-%m-%d", time.localtime(cutoff))
        for d in self.snapshot_dir.iterdir():
            if d.is_dir() and d.name < cut_day:
                shutil.rmtree(d, ignore_errors=True)
