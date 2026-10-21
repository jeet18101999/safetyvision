"""Ignore memory: things the local AI or the operator said are NOT a person / NOT fire at a given spot of a camera.
Takes effect immediately (before any retraining) and survives restarts. Managed from the Learning page."""
import json
import threading
import time
import uuid
from pathlib import Path

import cv2

from .analytics import box_iou


def _overlap(a, b) -> float:
    """Intersection over the smaller box: a flicker candidate that jitters around a known spot still matches."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def _same_spot(kind, a, b, thr) -> bool:
    return _overlap(a, b) >= 0.6 if kind == "fire" else box_iou(a, b) >= thr


class IgnoreStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "ignore_thumbs").mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ignore.json"
        self._lock = threading.Lock()
        self.items: list[dict] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _norm(box, shape):
        h, w = shape[:2]
        return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]

    def add(self, kind, camera, box, shape, source, frame=None, note="") -> dict:
        nb = self._norm(box, shape)
        with self._lock:
            for it in self.items:  # refresh / grow an existing entry instead of duplicating
                if it["kind"] == kind and it["camera"] == camera and _same_spot(kind, it["box"], nb, 0.6):
                    it["ts"], it["hits"] = time.time(), it.get("hits", 1) + 1
                    if kind == "fire":
                        ub = [min(it["box"][0], nb[0]), min(it["box"][1], nb[1]),
                              max(it["box"][2], nb[2]), max(it["box"][3], nb[3])]
                        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
                        if area(ub) <= 2.0 * max(area(it["box"]), area(nb)):
                            it["box"] = ub
                    self._save()
                    return it
            it = {"id": uuid.uuid4().hex[:10], "kind": kind, "camera": camera, "box": nb, "ts": time.time(),
                  "source": source, "note": note, "hits": 1}
            if frame is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                if crop.size:
                    s = 140 / max(crop.shape[:2])
                    cv2.imwrite(str(self.thumb(it)), cv2.resize(crop, (max(1, int(crop.shape[1] * s)),
                                                                     max(1, int(crop.shape[0] * s)))))
            self.items.append(it)
            self._save()
            return it

    def thumb(self, it) -> Path:
        return self.root / "ignore_thumbs" / f"{it['id']}.jpg"

    def match(self, kind, camera, box, shape, thr=0.5) -> bool:
        nb = self._norm(box, shape)
        with self._lock:
            return any(it["kind"] == kind and it["camera"] == camera and _same_spot(kind, it["box"], nb, thr)
                       for it in self.items)

    def compact(self):
        """Merge duplicate fire spots created by older versions (exact-IoU matching)."""
        with self._lock:
            merged: list[dict] = []
            for it in sorted(self.items, key=lambda x: x["ts"]):
                host = next((m for m in merged if it["kind"] == "fire" and m["kind"] == "fire"
                             and m["camera"] == it["camera"] and _overlap(m["box"], it["box"]) >= 0.5), None)
                if host is None:
                    merged.append(it)
                    continue
                host["hits"] = host.get("hits", 1) + it.get("hits", 1)
                host["ts"] = max(host["ts"], it["ts"])
                self.thumb(it).unlink(missing_ok=True)
            removed = len(self.items) - len(merged)
            self.items = merged
            self._save()
            return removed

    def list(self, kind=None) -> list[dict]:
        with self._lock:
            return [dict(it) for it in self.items if kind is None or it["kind"] == kind]

    def remove(self, item_id):
        with self._lock:
            for it in list(self.items):
                if it["id"] == item_id:
                    self.thumb(it).unlink(missing_ok=True)
                    self.items.remove(it)
            self._save()


class NotFireMemory:
    """'Not fire' look-alikes learned from rejected fire candidates (local AI or operator): a new candidate whose
    colours match a rejected one from the same camera is skipped before it reaches the local AI. Matching is by
    colour histogram of the flagged region, so it also works when the object moves (a red jacket, a lamp, sun glare)
    - unlike the spot-based IgnoreStore. Calibrated on site data: real/simulated fire scores <= 0.41, repeats of
    rejected objects mostly >= 0.6."""
    THRESHOLD = 0.6

    def __init__(self, max_per_camera=200):
        self.max_per_camera = max_per_camera
        self._lock = threading.Lock()
        self.items: dict[str, list] = {}  # camera -> [(sample_id, hist)]
        self.skipped = 0

    @staticmethod
    def feature(frame, box):
        import numpy as np
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, min(w - 2, x1)), max(0, min(h - 2, y1))
        c = frame[y1:max(y1 + 2, min(h, y2)), x1:max(x1 + 2, min(w, x2))]
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [18, 8, 8], [0, 180, 0, 256, 0, 256]).ravel()
        return (hist / (hist.sum() + 1e-6)).astype(np.float32)

    def add(self, camera, frame, box, sid=""):
        f = self.feature(frame, box)
        with self._lock:
            lst = self.items.setdefault(camera, [])
            lst.append((sid, f))
            del lst[:-self.max_per_camera]

    def remove(self, sid):
        with self._lock:
            for cam in self.items:
                self.items[cam] = [x for x in self.items[cam] if x[0] != sid]

    def load_from_dataset(self, ds):
        """Rebuild from the fire dataset's 'not fire' samples (images are stored with normalised boxes)."""
        with self._lock:
            self.items = {}
        for s in ds.negatives():
            img = cv2.imread(str(ds.image_path(s)))
            if img is None or not s.get("camera"):
                continue
            h, w = img.shape[:2]
            b = s["box"]
            self.add(s["camera"], img, [b[0] * w, b[1] * h, b[2] * w, b[3] * h], s["id"])

    def match(self, camera, frame, box) -> float:
        """Best similarity (0..1) to a rejected look-alike of this camera; >= THRESHOLD means 'known not fire'."""
        import numpy as np
        with self._lock:
            lst = list(self.items.get(camera, []))
        if not lst:
            return 0.0
        f = self.feature(frame, box)
        return float(max(np.minimum(f, g).sum() for _, g in lst))

    def count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self.items.values())
