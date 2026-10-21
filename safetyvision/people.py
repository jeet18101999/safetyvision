"""Person re-identification by body appearance (no face recognition).

* ReIDModel      - body-appearance embedding (OpenCV Zoo 'Youtu ReID', Apache-2.0), run on the GPU through PyTorch.
                   An optional projection head, trained on the site's own named people, is applied on top.
* Gallery        - known people (named by the operator) and unknown people ("Person 17") with prototype embeddings.
* IdentityEngine - per camera track: collects good body crops, embeds them in batches, matches the gallery,
                   creates unknown people, asks the local AI about ambiguous matches, records sightings
                   (who / which camera / when / foot path) for the movement map, and trains the projection head.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import config as C

log = logging.getLogger(__name__)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ONNX_PATH = "models/reid/person_reid_youtu_2021nov.onnx"
ONNX_URL = "https://huggingface.co/opencv/person_reid_youtureid/resolve/main/person_reid_youtu_2021nov.onnx"


def people_dir() -> Path:
    d = C.DATA_DIR / "people"
    (d / "thumbs").mkdir(parents=True, exist_ok=True)
    (d / "crops").mkdir(parents=True, exist_ok=True)
    return d


def _norm(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


# ============================================================================ embedding model
class ReIDModel:
    def __init__(self, device="cuda:0"):
        path = C.resolve(ONNX_PATH)
        if not path.exists():
            import requests
            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(ONNX_URL, timeout=120, stream=True)
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(path)
        self.device = device
        self.torch = None
        try:
            import onnx
            import torch
            from onnx2torch import convert
            self.torch = torch
            self.net = convert(onnx.load(str(path))).eval()
            if device.startswith("cuda"):
                self.net = self.net.cuda().half()
        except Exception as e:  # no torch conversion: slower OpenCV DNN on CPU
            log.warning("ReID on OpenCV DNN (CPU): %s", e)
            self.net = cv2.dnn.readNet(str(path))
        self.head = None  # trained projection (numpy matrices), see train_head()
        self._lock = threading.Lock()

    def backbone(self, crops) -> np.ndarray:
        if not crops:
            return np.zeros((0, 768), np.float32)
        blob = np.stack([((cv2.resize(c, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD)
                         .transpose(2, 0, 1) for c in crops]).astype(np.float32)
        with self._lock:
            if self.torch is not None:
                t = self.torch.from_numpy(blob)
                if self.device.startswith("cuda"):
                    t = t.cuda().half()
                with self.torch.no_grad():
                    f = self.net(t).float().reshape(len(crops), -1).cpu().numpy()
            else:
                self.net.setInput(blob)
                f = self.net.forward().reshape(len(crops), -1)
        return _norm(f)

    def project(self, raw) -> np.ndarray:
        """Raw backbone embeddings -> matching space (identity until a site head is trained)."""
        raw = np.asarray(raw, np.float32)
        if self.head is None or raw.size == 0:
            return _norm(raw)
        w1, b1, w2 = self.head
        h = np.maximum(raw @ w1 + b1, 0)
        return _norm(np.concatenate([_norm(raw) * 0.5, _norm(h @ w2)], axis=-1))


# ============================================================================ gallery + sightings
class Gallery:
    MAX_PROTOS = 24

    def __init__(self):
        d = people_dir()
        self.path = d / "people.json"
        self.proto_path = d / "prototypes.npz"
        self._lock = threading.RLock()
        self.people: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.protos: dict[str, np.ndarray] = {}
        if self.proto_path.exists():
            z = np.load(self.proto_path)
            self.protos = {k: z[k] for k in z.files}
        self.counter = max([int(p.get("number", 0)) for p in self.people.values()] + [0])
        self.db = sqlite3.connect(str(d / "sightings.db"), check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS sightings (id INTEGER PRIMARY KEY AUTOINCREMENT, person TEXT,
            camera TEXT, location TEXT, room TEXT, t_start REAL, t_end REAL, path TEXT, thumb TEXT, similarity REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS s_person ON sightings(person, t_start)")
        self.db.execute("CREATE INDEX IF NOT EXISTS s_time ON sightings(t_start)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS pairs (a TEXT, b TEXT, same INTEGER, source TEXT, ts REAL)""")
        self.db.commit()
        self._dirty = False
        self._last_save = 0.0
        self._migrate_crops()

    # ---------------------------------------------------------------- photos: "to check" (pending) / confirmed
    @staticmethod
    def crop_dir(pid, status="pending") -> Path:
        d = people_dir() / "crops" / pid / status
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _migrate_crops(self):
        """Older versions stored photos directly under crops/<pid>/ - they were never checked, so: to check."""
        root = people_dir() / "crops"
        for d in root.iterdir() if root.exists() else []:
            loose = list(d.glob("*.jpg")) if d.is_dir() else []
            if loose:
                dst = self.crop_dir(d.name, "pending")
                for f in loose:
                    f.replace(dst / f.name)

    def photos(self, pid) -> list[tuple[Path, str]]:
        out = []
        for status in ("confirmed", "pending"):
            d = people_dir() / "crops" / pid / status
            if d.exists():
                out += [(f, status) for f in sorted(d.glob("*.jpg"), reverse=True)]
        return out

    def photo_counts(self, pid) -> tuple[int, int]:
        base = people_dir() / "crops" / pid
        c = len(list((base / "confirmed").glob("*.jpg"))) if (base / "confirmed").exists() else 0
        q = len(list((base / "pending").glob("*.jpg"))) if (base / "pending").exists() else 0
        return c, q

    @staticmethod
    def owner(path: Path) -> str:
        return Path(path).parent.parent.name

    def move_photos(self, paths, target_pid, status="confirmed") -> set:
        """Operator moves / confirms photos. Returns the person ids whose photos changed."""
        changed = set()
        with self._lock:
            dst = self.crop_dir(target_pid, status)
            for f in map(Path, paths):
                if f.exists():
                    changed.add(self.owner(f))
                    f.replace(dst / f.name)
            changed.add(target_pid)
        return changed

    def delete_photos(self, paths) -> set:
        changed = set()
        with self._lock:
            for f in map(Path, paths):
                if f.exists():
                    changed.add(self.owner(f))
                    f.unlink()
        return changed

    # ---------------------------------------------------------------- persistence
    def save(self, force=False):
        with self._lock:
            if not (self._dirty or force) or (not force and time.time() - self._last_save < 10):
                return
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.people, indent=1), encoding="utf-8")
            tmp.replace(self.path)
            np.savez(self.proto_path.with_suffix(".tmp.npz"), **self.protos)
            Path(str(self.proto_path.with_suffix(".tmp.npz"))).replace(self.proto_path)
            self._dirty, self._last_save = False, time.time()

    # ---------------------------------------------------------------- queries
    def label(self, pid) -> str:
        p = self.people.get(pid)
        if not p:
            return ""
        return p.get("name") or f"Person {p['number']}"

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(p, id=k) for k, p in self.people.items()]

    def match(self, emb_space, proj) -> list[tuple[str, float]]:
        """[(pid, similarity)] best first. emb_space: projected query embedding; proj: projection function."""
        with self._lock:
            scores = []
            for pid, P in self.protos.items():
                if len(P):
                    scores.append((pid, float(np.max(proj(P) @ emb_space))))
        return sorted(scores, key=lambda s: -s[1])

    # ---------------------------------------------------------------- updates
    def new_person(self, raw, crop) -> str:
        with self._lock:
            self.counter += 1
            pid = uuid.uuid4().hex[:10]
            self.people[pid] = {"number": self.counter, "name": "", "created": time.time(), "last_seen": time.time(),
                                "seen": 0, "thumb": "", "last_camera": ""}
            self.protos[pid] = _norm(raw)[None, :]
            self._thumb(pid, crop)
            self._dirty = True
            return pid

    def observe(self, pid, raws, crop, camera):
        """Add evidence to a person. People whose photos the operator has confirmed are recognised only from
        those photos - automatic matches never change them (no drift from wrong matches)."""
        with self._lock:
            if pid not in self.people:
                return
            P = self.protos.get(pid, np.zeros((0, 768), np.float32))
            for r in ([] if self.people[pid].get("verified") else _norm(raws)):
                if len(P) and float(np.max(P @ r)) > 0.93:
                    continue  # already represented
                P = np.vstack([P, r[None, :]])
                if len(P) > self.MAX_PROTOS:  # drop the most redundant prototype
                    S = P @ P.T
                    np.fill_diagonal(S, -1)
                    P = np.delete(P, int(np.argmax(S.max(1))), axis=0)
            self.protos[pid] = P
            p = self.people[pid]
            p["last_seen"], p["seen"], p["last_camera"] = time.time(), p.get("seen", 0) + 1, camera
            if crop is not None and (not p.get("thumb") or p["seen"] % 25 == 0):
                self._thumb(pid, crop)
            self._dirty = True

    def _thumb(self, pid, crop):
        if crop is None or crop.size == 0:
            return
        h, w = crop.shape[:2]
        s = 200 / max(h, w)
        path = people_dir() / "thumbs" / f"{pid}.jpg"
        cv2.imwrite(str(path), cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s)))))
        self.people[pid]["thumb"] = str(path)

    def save_crop(self, pid, crop, max_pending=80):
        """Automatically sorted photo -> the person's 'to check' pile (never used for training until confirmed)."""
        if crop is None or crop.size == 0:
            return
        d = self.crop_dir(pid, "pending")
        files = sorted(d.glob("*.jpg"))
        for f in files[:max(0, len(files) - max_pending + 1)]:
            f.unlink(missing_ok=True)
        cv2.imwrite(str(d / f"{int(time.time() * 1000)}.jpg"), cv2.resize(crop, (128, 256)))

    def rename(self, pid, name):
        with self._lock:
            if pid in self.people:
                self.people[pid]["name"] = name.strip()
                self.people[pid]["named_at"] = time.time()
                self._dirty = True
        self.save(force=True)

    def merge(self, keep, drop):
        """Operator says two entries are the same person."""
        with self._lock:
            if keep not in self.people or drop not in self.people or keep == drop:
                return
            self.protos[keep] = np.vstack([self.protos.get(keep, np.zeros((0, 768))), self.protos.get(drop, np.zeros((0, 768)))])
            while len(self.protos[keep]) > self.MAX_PROTOS:
                S = self.protos[keep] @ self.protos[keep].T
                np.fill_diagonal(S, -1)
                self.protos[keep] = np.delete(self.protos[keep], int(np.argmax(S.max(1))), axis=0)
            self.people[keep]["seen"] = self.people[keep].get("seen", 0) + self.people[drop].get("seen", 0)
            if not self.people[keep].get("name"):
                self.people[keep]["name"] = self.people[drop].get("name", "")
            for status in ("pending", "confirmed"):
                src = people_dir() / "crops" / drop / status
                if src.exists():
                    dst = self.crop_dir(keep, status)
                    for f in src.glob("*.jpg"):
                        f.replace(dst / f.name)
            import shutil
            shutil.rmtree(people_dir() / "crops" / drop, ignore_errors=True)
            self.people[keep]["verified"] = self.people[keep].get("verified") or self.people[drop].get("verified")
            self.people.pop(drop, None)
            self.protos.pop(drop, None)
            self.db.execute("UPDATE sightings SET person=? WHERE person=?", (keep, drop))
            self.db.execute("INSERT INTO pairs VALUES (?,?,?,?,?)", (keep, drop, 1, "operator", time.time()))
            self.db.commit()
            self._dirty = True
        self.save(force=True)

    def delete(self, pid):
        with self._lock:
            self.people.pop(pid, None)
            self.protos.pop(pid, None)
            self.db.execute("DELETE FROM sightings WHERE person=?", (pid,))
            self.db.commit()
            import shutil
            shutil.rmtree(people_dir() / "crops" / pid, ignore_errors=True)
            (people_dir() / "thumbs" / f"{pid}.jpg").unlink(missing_ok=True)
            self._dirty = True
        self.save(force=True)

    def purge(self, sighting_days=30, unknown_days=7):
        """Privacy / housekeeping: old sightings; unknown people seen only briefly and not for a week."""
        cutoff = time.time() - sighting_days * 86400
        with self._lock:
            self.db.execute("DELETE FROM sightings WHERE t_end < ?", (cutoff,))
            self.db.commit()
            old = [k for k, p in self.people.items() if not p.get("name") and p.get("seen", 0) < 5
                   and time.time() - p.get("last_seen", 0) > unknown_days * 86400]
        for k in old:
            self.delete(k)

    # ---------------------------------------------------------------- sightings (movement map)
    def add_sighting(self, pid, camera, location, room, t0, t1, path, thumb, sim):
        with self._lock:
            self.db.execute("INSERT INTO sightings (person, camera, location, room, t_start, t_end, path, thumb, "
                            "similarity) VALUES (?,?,?,?,?,?,?,?,?)",
                            (pid, camera, location, room, t0, t1, json.dumps(path), thumb, sim))
            self.db.commit()

    def sightings(self, pid=None, since=0, until=None) -> list[dict]:
        q = "SELECT person, camera, location, room, t_start, t_end, path, thumb FROM sightings WHERE t_start >= ?"
        args = [since]
        if until:
            q += " AND t_start <= ?"
            args.append(until)
        if pid:
            q += " AND person = ?"
            args.append(pid)
        q += " ORDER BY t_start"
        with self._lock:
            rows = self.db.execute(q, args).fetchall()
        return [{"person": r[0], "camera": r[1], "location": r[2], "room": r[3], "t_start": r[4], "t_end": r[5],
                 "path": json.loads(r[6] or "[]"), "thumb": r[7]} for r in rows]

    def add_pair(self, a, b, same, source):
        with self._lock:
            self.db.execute("INSERT INTO pairs VALUES (?,?,?,?,?)", (a, b, int(same), source, time.time()))
            self.db.commit()


# ============================================================================ per-camera identity resolution
class _TrackState:
    __slots__ = ("raws", "last_sample", "pid", "sim", "locked", "t_start", "last_seen", "path", "best_crop",
                 "best_area", "asked_ai", "asked_at", "last_resolve", "camera")

    def __init__(self, camera, now):
        self.camera = camera
        self.raws, self.path = [], []
        self.last_sample = self.last_resolve = 0.0
        self.pid, self.sim, self.locked, self.asked_ai, self.asked_at = None, 0.0, False, False, 0.0
        self.t_start = self.last_seen = now
        self.best_crop, self.best_area = None, 0


class IdentityEngine:
    MATCH = 0.70  # assign an existing person above this similarity (tracklet average vs prototypes)
    SWITCH = 0.45  # a new crop this unlike its own track means the tracker swapped two people
    CONFIDENT = 0.76  # stop re-checking
    NEW = 0.52  # below this, a new person is created
    MIN_SAMPLES = 3

    def __init__(self, cfg, device, verifier_getter=None):
        self.cfg = cfg
        self.model = ReIDModel(device)
        self.gallery = Gallery()
        self._states: dict[tuple, _TrackState] = {}
        self._queue: list[tuple] = []  # (key, crop)
        self._verifier = verifier_getter or (lambda: None)
        self._head_path = people_dir() / "site_head.npz"
        if self._head_path.exists():
            z = np.load(self._head_path)
            self.model.head = (z["w1"], z["b1"], z["w2"])
        self.head_info = json.loads((people_dir() / "head.json").read_text()) if (people_dir() / "head.json").exists() else {}
        self.visible: dict[str, dict] = {}  # camera -> {pid: ts}
        self._training = False

    # ---------------------------------------------------------------- per frame
    def observe(self, camera, frame, people, now):
        """Queue good crops for the people visible on this camera frame."""
        H, W = frame.shape[:2]
        boxes = [p["box"] for p in people]
        seen_now = {}
        for p in people:
            t = p["track"]
            key = (camera, t.id)
            st = self._states.get(key)
            if st is None:
                st = self._states[key] = _TrackState(camera, now)
            st.last_seen = now
            x1, y1, x2, y2 = p["box"]
            fx, fy = ((x1 + x2) / 2) / W, y2 / H
            if not st.path or now - st.path[-1][0] >= 1.0:
                st.path.append([round(now, 1), round(fx, 4), round(fy, 4)])
            if st.pid:
                seen_now[st.pid] = now
            p["identity"] = st.pid
            p["identity_label"] = self.gallery.label(st.pid) if st.pid else ""
            # sample a crop: big enough, confident, not cut off, not overlapping someone else
            h = y2 - y1
            if st.locked or now - st.last_sample < 0.6 or h < 100 or p.get("conf", 1) < 0.55 or p.get("fallen"):
                continue
            if x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > H - 3:
                continue
            from .analytics import box_iou
            if any(b is not p["box"] and box_iou(b, p["box"]) > 0.15 for b in boxes):
                continue
            crop = frame[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
            if crop.size:
                st.last_sample = now
                self._queue.append((key, crop.copy()))
                area = crop.shape[0] * crop.shape[1]
                if area > st.best_area:
                    st.best_crop, st.best_area = crop.copy(), area
        self.visible[camera] = {**{k: v for k, v in self.visible.get(camera, {}).items() if now - v < 3}, **seen_now}

    def flush(self, now):
        """Embed all queued crops in one GPU batch and resolve identities."""
        if self._queue:
            keys, crops = zip(*self._queue[:64])
            self._queue = self._queue[64:]
            raws = self.model.backbone(list(crops))
            for key, raw, crop in zip(keys, raws, crops):
                st = self._states.get(key)
                if st is None:
                    continue
                if len(st.raws) >= 3 and float(_norm(np.mean(st.raws, axis=0)) @ raw) < self.SWITCH:
                    # the tracker swapped two people (they crossed): close this identity, start a fresh one
                    self._end(key, st)
                    fresh = _TrackState(st.camera, now)
                    fresh.path = st.path[-1:]
                    st = self._states[key] = fresh
                    st.best_crop, st.best_area = crop, crop.shape[0] * crop.shape[1]
                st.raws = (st.raws + [raw])[-12:]
            for key in set(keys):
                st = self._states.get(key)
                if st is not None and len(st.raws) >= self.MIN_SAMPLES and now - st.last_resolve > 1.5:
                    self._resolve(key, st, now)
        # finished tracks -> sightings
        for key in [k for k, s in self._states.items() if now - s.last_seen > 4]:
            self._end(key, self._states.pop(key))
        self.gallery.save()

    STRONG = 0.80  # the same person on two cameras at once is only accepted above this (or in the same room)

    def _conflicts(self, camera, key, now):
        """(pids on another track of this camera, pids visible right now on cameras of a different room).
        One person cannot be in two places at once."""
        room = C.camera_cfg(self.cfg, camera).get("room")
        same_cam, elsewhere = set(), set()
        for k, s in self._states.items():
            if k == key or not s.pid or now - s.last_seen > 2:
                continue
            if s.camera == camera:
                same_cam.add(s.pid)
            elif not room or C.camera_cfg(self.cfg, s.camera).get("room") != room:
                elsewhere.add(s.pid)
        return same_cam, elsewhere

    def _resolve(self, key, st, now):
        st.last_resolve = now
        q = self.model.project(np.mean(st.raws, axis=0)[None, :])[0]
        same_cam, elsewhere = self._conflicts(st.camera, key, now)
        cands = [(pid, s) for pid, s in self.gallery.match(q, self.model.project)
                 if pid not in same_cam and (pid not in elsewhere or s >= self.STRONG)]
        best, sim = cands[0] if cands else (None, 0.0)
        if best and sim >= self.MATCH:
            self._assign(st, best, sim)
        elif best and sim >= self.NEW and len(st.raws) >= 5 and not st.asked_ai and st.pid is None:
            self._ask_ai(st, best, sim)  # ambiguous: the local AI decides, and the answer becomes training data
            st.asked_at = now
        elif st.pid is None and len(st.raws) >= 5 and (
                not best or sim < self.NEW or (st.asked_ai and now - st.asked_at > 20) or len(st.raws) >= 12):
            # clearly new, or the local AI is busy / slow: do not leave the person without an identity
            pid = self.gallery.new_person(np.mean(st.raws, axis=0), st.best_crop)
            self._assign(st, pid, 1.0, new=True)
        if st.pid and st.sim >= self.CONFIDENT and len(st.raws) >= 8:
            st.locked = True

    def _assign(self, st, pid, sim, new=False):
        st.pid, st.sim = pid, sim
        if not new:
            self.gallery.observe(pid, st.raws[-4:], st.best_crop, st.camera)
        if sim >= 0.72 or new:
            self.gallery.save_crop(pid, st.best_crop)

    def rebuild_person(self, pid):
        """Recompute a person's recognition data from the photos the operator confirmed."""
        g = self.gallery
        if pid not in g.people:
            return
        files = sorted((people_dir() / "crops" / pid / "confirmed").glob("*.jpg"))
        imgs = [im for im in (cv2.imread(str(f)) for f in files) if im is not None]
        with g._lock:
            if not imgs:
                g.people[pid]["verified"] = False
                g._dirty = True
                return
            E = self.model.backbone(imgs)
            keep = [0]  # farthest-point selection: diverse poses / cameras
            while len(keep) < min(Gallery.MAX_PROTOS, len(E)):
                d = 1 - (E @ E[keep].T).max(1)
                d[keep] = -1
                keep.append(int(np.argmax(d)))
            g.protos[pid] = E[keep]
            g.people[pid]["verified"] = len(imgs) >= 3
            g._thumb(pid, imgs[0])
            g._dirty = True
        g.save(force=True)

    def apply_photo_changes(self, changed):
        for pid in changed:
            if pid in self.gallery.people:
                self.rebuild_person(pid)
        for key, st in self._states.items():  # live tracks re-check against the corrected gallery
            st.locked, st.last_resolve = False, 0.0

    def _ask_ai(self, st, pid, sim):
        st.asked_ai = True
        v = self._verifier()
        ref = self.gallery.people.get(pid, {}).get("thumb")
        if v is None or not v.online or st.best_crop is None or not ref or not Path(ref).exists():
            return
        if v.pending() > 4:  # safety checks (fire, man down, PPE) come first; try again later
            st.asked_ai = False
            return
        a, b = st.best_crop.copy(), cv2.imread(ref)

        def done(res):
            if not res.get("ok"):
                return
            self.gallery.add_pair(pid, "track", res["same"], "ai")
            if res["same"] and res["confidence"] >= 0.6:
                self._assign(st, pid, max(sim, self.MATCH))
            elif not res["same"] and st.pid is None:
                npid = self.gallery.new_person(np.mean(st.raws, axis=0), st.best_crop)
                self._assign(st, npid, 1.0, new=True)

        v.submit_same_person(a, b, {"camera": st.camera}, done)

    def _end(self, key, st):
        if not st.pid or st.last_seen - st.t_start < 2:
            return
        cc = C.camera_cfg(self.cfg, st.camera)
        thumb = self.gallery.people.get(st.pid, {}).get("thumb", "")
        self.gallery.add_sighting(st.pid, st.camera, cc.get("place_path", ""), cc.get("room_name", ""), st.t_start,
                                  st.last_seen, st.path[-120:], thumb, st.sim)

    def label_for(self, camera, track_id) -> str:
        st = self._states.get((camera, track_id))
        return self.gallery.label(st.pid) if st and st.pid else ""

    def pid_for(self, camera, track_id):
        st = self._states.get((camera, track_id))
        return st.pid if st else None

    def unique_people(self, cams_people: dict[str, list]) -> int:
        """cams_people: camera -> list of pids (None = not identified yet). Cameras in the same room count a
        recognised person once; unidentified people in a room count as the max over its cameras."""
        rooms: dict[str, list[str]] = {}
        total = 0
        for cam, pids in cams_people.items():
            room = C.camera_cfg(self.cfg, cam).get("room")
            if room:
                rooms.setdefault(room, []).append(cam)
            else:
                total += len(pids)
        for room, cams in rooms.items():
            known = set()
            unknown = 0
            for cam in cams:
                pids = cams_people[cam]
                known |= {p for p in pids if p}
                unknown = max(unknown, sum(1 for p in pids if not p))
            total += len(known) + unknown
        return total

    # ---------------------------------------------------------------- training on the site's people
    def training_data(self):
        crops, labels = [], []
        for pid, p in self.gallery.people.items():
            if not p.get("name"):
                continue
            files = sorted((people_dir() / "crops" / pid / "confirmed").glob("*.jpg"))  # operator-confirmed only
            if len(files) >= 6:
                for f in files:
                    img = cv2.imread(str(f))
                    if img is not None:
                        crops.append(img)
                        labels.append(pid)
        return crops, labels

    def train_head(self, epochs=300, status=print) -> dict:
        """Learn a projection that separates the site's named people better than the generic body model.
        Deployed only if held-out identification accuracy improves."""
        if self._training:
            return {"ok": False, "message": "already training"}
        self._training = True
        try:
            crops, labels = self.training_data()
            ids = sorted(set(labels))
            if len(ids) < 2:
                return {"ok": False, "message": "confirm at least 6 photos for each of 2 or more named people first "
                                               "(People page > photos > Confirm)"}
            status(f"embedding {len(crops)} crops of {len(ids)} named people ...")
            raw = np.concatenate([self.model.backbone(crops[i:i + 32]) for i in range(0, len(crops), 32)])
            y = np.array([ids.index(l) for l in labels])
            rng = np.random.default_rng(0)
            test = np.zeros(len(y), bool)
            for k in range(len(ids)):  # hold out 25% of each person's crops
                idx = np.where(y == k)[0]
                test[rng.choice(idx, max(1, len(idx) // 4), replace=False)] = True

            def accuracy(emb):  # nearest-centroid identification on held-out crops
                cents = _norm(np.stack([emb[~test & (y == k)].mean(0) for k in range(len(ids))]))
                return float(np.mean((emb[test] @ cents.T).argmax(1) == y[test]))

            base = accuracy(_norm(raw))
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            X = torch.tensor(raw[~test], device=dev)
            Y = torch.tensor(y[~test], device=dev)
            w1 = torch.nn.Linear(768, 256).to(dev)
            w2 = torch.nn.Linear(256, 128, bias=False).to(dev)
            opt = torch.optim.Adam(list(w1.parameters()) + list(w2.parameters()), lr=1e-3, weight_decay=1e-4)
            for ep in range(epochs):  # supervised contrastive loss
                z = torch.nn.functional.normalize(w2(torch.relu(w1(X))), dim=1)
                sim = z @ z.T / 0.1
                mask = (Y[:, None] == Y[None, :]).float()
                eye = torch.eye(len(Y), device=dev)
                logp = sim - torch.logsumexp(sim - eye * 1e9, dim=1, keepdim=True)
                loss = -((mask - eye) * logp).sum(1) / (mask - eye).sum(1).clamp(min=1)
                loss = loss.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            head = (w1.weight.detach().cpu().numpy().T, w1.bias.detach().cpu().numpy(),
                    w2.weight.detach().cpu().numpy().T)
            old = self.model.head
            self.model.head = head
            new = accuracy(self.model.project(raw))
            keep = new >= base and new >= self.head_info.get("accuracy", 0) - 0.01
            if keep:
                np.savez(self._head_path, w1=head[0], b1=head[1], w2=head[2])
                self.head_info = {"accuracy": new, "base": base, "people": len(ids), "crops": len(crops),
                                  "time": time.strftime("%Y-%m-%d %H:%M"), "version": self.head_info.get("version", 0) + 1}
                (people_dir() / "head.json").write_text(json.dumps(self.head_info), encoding="utf-8")
            else:
                self.model.head = old
            msg = (f"generic body model {base:.0%} -> site-trained {new:.0%} on held-out crops of {len(ids)} people; "
                   + ("deployed" if keep else "kept the previous model"))
            status(msg)
            return {"ok": True, "base": base, "new": new, "deployed": keep, "message": msg}
        finally:
            self._training = False
