"""Active-learning datasets.

FireDataset  - YOLO detection dataset. Local-AI confirmed fire/smoke -> box label, rejected -> hard negative.
ClsDataset   - image-folder classification dataset (person verifier, man-down verifier, one per PPE item).

Operator corrections are stored with a weight: extra copies go into the training split so a correction
counts more than an automatic label, and they count towards triggering the next training run sooner.
"""
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import yaml

CLASSES = ["fire", "smoke"]


def _new_id():
    return time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


class _Manifest:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.manifest = self.root / "samples.jsonl"
        self.state_path = self.root / "state.json"
        self.state = {"new_samples": 0, "last_trained": None}
        if self.state_path.exists():
            self.state.update(json.loads(self.state_path.read_text(encoding="utf-8")))
        self._samples = []
        if self.manifest.exists():
            for line in self.manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self._samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def _save_manifest(self):
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(s) + "\n" for s in self._samples), encoding="utf-8")
        tmp.replace(self.manifest)

    def _append(self, rec):
        self._samples.append(rec)
        with self.manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _save_state(self):
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _bump(self, n):
        self.state["new_samples"] += n
        self._save_state()

    def get(self, sid):
        with self._lock:
            return next((s for s in self._samples if s["id"] == sid), None)

    def recent(self, n=60) -> list[dict]:
        with self._lock:
            return list(reversed(self._samples[-n:]))

    def new_samples(self) -> int:
        return int(self.state["new_samples"])

    def mark_trained(self, samples_at_start: int):
        with self._lock:
            self.state["new_samples"] = max(0, self.state["new_samples"] - samples_at_start)
            self.state["last_trained"] = time.time()
            self._save_state()


class FireDataset(_Manifest):
    kind = "detect"
    classes = CLASSES

    def __init__(self, root: Path, val_fraction=0.2, max_negative_ratio=1.5, anchor_dir: Path | None = None):
        super().__init__(root)
        self.val_fraction = val_fraction
        self.max_negative_ratio = max_negative_ratio
        # reference fire images (simulated fires on site footage, shipped in assets/fire_anchor): trained alongside
        # the site's 'not fire' samples so the model learns what is NOT fire without forgetting what fire looks like
        self.anchor_dir = Path(anchor_dir).resolve() if anchor_dir else None
        self.on_change = None  # called after a sample is relabelled or deleted (the engine refreshes look-alikes)
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (self.root / "thumbs").mkdir(parents=True, exist_ok=True)
        self.data_yaml = self.root / "data.yaml"
        self.data_yaml.write_text(yaml.safe_dump({
            "path": str(self.root.resolve()), "train": "images/train", "val": "images/val",
            "names": {i: n for i, n in enumerate(CLASSES)}}, sort_keys=False), encoding="utf-8")

    @property
    def train_data(self):
        return str(self.data_yaml)

    def counts(self) -> dict:
        with self._lock:
            c = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0}
            for s in self._samples:
                c[f"{s['split']}_{'pos' if s['positive'] else 'neg'}"] += 1
            c["new_samples"] = self.state["new_samples"]
            c["total"] = len(self._samples)
            c["by_class"] = {"fire/smoke": c["train_pos"] + c["val_pos"], "not fire": c["train_neg"] + c["val_neg"]}
            return c

    def thumb_path(self, sid) -> Path:
        return self.root / "thumbs" / f"{sid}.jpg"

    def image_path(self, s) -> Path:
        return self.root / "images" / s["split"] / f"{s['id']}.jpg"

    def negatives(self) -> list[dict]:
        with self._lock:
            return [dict(s) for s in self._samples if not s["positive"]]

    def anchor_count(self) -> dict:
        out = {"train": 0, "val": 0}
        if self.anchor_dir and self.anchor_dir.is_dir():
            for split in out:
                out[split] = sum(1 for _ in (self.anchor_dir / "images" / split).glob("*.jpg"))
        return out

    def prepare_training(self) -> dict:
        """Write data.yaml for the next run: site samples + reference fire images + extra copies of the site's
        'not fire' images (so a few dozen of them are not drowned by hundreds of reference images).
        Returns {neg_list: file with all 'not fire' image paths (false-alarm check), ...counts}."""
        anchors = self.anchor_count()
        train, val = [str(self.root / "images" / "train")], [str(self.root / "images" / "val")]
        if anchors["train"]:
            train.append(str(self.anchor_dir / "images" / "train"))
        if anchors["val"]:
            val.append(str(self.anchor_dir / "images" / "val"))
        boost = self.root / "boost"
        shutil.rmtree(boost, ignore_errors=True)
        negs = [s for s in self.negatives() if s["split"] == "train"]
        copies = 0
        if negs and anchors["train"]:
            want = int(0.35 * anchors["train"])  # 'not fire' ~ a quarter of the training images
            have = sum(1 + s.get("dups", 0) for s in negs)
            reps = min(10, max(0, -(-(want - have) // max(1, len(negs)))))
            if reps:
                (boost / "images" / "train").mkdir(parents=True, exist_ok=True)
                (boost / "labels" / "train").mkdir(parents=True, exist_ok=True)
                for s in negs:
                    src = self.image_path(s)
                    for k in range(reps):
                        shutil.copy2(src, boost / "images" / "train" / f"{s['id']}_b{k}.jpg")
                        (boost / "labels" / "train" / f"{s['id']}_b{k}.txt").write_text("", encoding="utf-8")
                        copies += 1
                train.append(str(boost / "images" / "train"))
        (self.root / "train_run.yaml").write_text(yaml.safe_dump({  # own file: never overwritten by __init__
            "path": str(self.root.resolve()), "train": train, "val": val,
            "names": {i: n for i, n in enumerate(CLASSES)}}, sort_keys=False), encoding="utf-8")
        neg_list = self.root / "negatives.txt"
        neg_list.write_text("".join(str(self.image_path(s)) + "\n" for s in self.negatives()), encoding="utf-8")
        return {"neg_list": str(neg_list), "anchors": anchors, "neg_copies": copies,
                "data": str(self.root / "train_run.yaml")}

    def sample_label(self, s) -> str:
        return s["label"] if s["positive"] else "not_fire"

    def add_sample(self, frame, box, positive: bool, label: str, meta: dict, weight: int = 1) -> dict | None:
        with self._lock:
            pos = sum(1 for s in self._samples if s["positive"])
            neg = len(self._samples) - pos
            if not positive and weight == 1 and neg >= self.max_negative_ratio * pos + 30:
                return None  # enough automatic negatives; keep dataset balanced
            same = [s for s in self._samples if s["positive"] == positive]
            n_val = sum(1 for s in same if s["split"] == "val")
            split = "val" if weight == 1 and n_val < self.val_fraction * (len(same) + 1) else "train"
            sid = _new_id()
            h, w = frame.shape[:2]
            s = min(1.0, 1280 / max(h, w))
            img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
            x1, y1, x2, y2 = [float(v) for v in box]
            nbox = [max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h)]
            cls = CLASSES.index(label) if label in CLASSES else 0
            rec = {"id": sid, "split": split, "positive": positive, "label": label, "cls": cls, "box": nbox,
                   "ts": time.time(), "dups": max(0, weight - 1) if split == "train" else 0,
                   "manual": weight > 1, **meta}
            for name in self._names(rec):
                cv2.imwrite(str(self.root / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self._write_labels(rec)
            from .verifier import crop_box
            thumb = crop_box(frame, box, pad=0.5)
            th, tw = thumb.shape[:2]
            ts = 160 / max(th, tw)
            cv2.imwrite(str(self.thumb_path(sid)), cv2.resize(thumb, (max(1, int(tw * ts)), max(1, int(th * ts)))))
            self._append(rec)
            self._bump(weight)
            return rec

    @staticmethod
    def _names(rec):
        return [rec["id"]] + [f"{rec['id']}_dup{k}" for k in range(rec.get("dups", 0))]

    def _write_labels(self, rec):
        for name in self._names(rec):
            p = self.root / "labels" / rec["split"] / f"{name}.txt"
            if not rec["positive"]:
                p.write_text("", encoding="utf-8")
                continue
            x1, y1, x2, y2 = rec["box"]
            p.write_text(f"{rec['cls']} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n",
                         encoding="utf-8")

    def relabel(self, sid, label: str, weight: int = 3):
        """label: 'fire', 'smoke' or 'not_fire'. Operator correction (overrides the AI)."""
        positive = label != "not_fire"
        with self._lock:
            s = self.get(sid)
            if s is None:
                return
            s["positive"], s["label"] = positive, label if positive else s["label"]
            s["cls"] = CLASSES.index(s["label"]) if s["label"] in CLASSES else 0
            s["manual"] = True
            if s["split"] == "train" and s.get("dups", 0) < weight - 1:
                src = self.root / "images" / "train" / f"{sid}.jpg"
                for k in range(s.get("dups", 0), weight - 1):
                    shutil.copy2(src, self.root / "images" / "train" / f"{sid}_dup{k}.jpg")
                s["dups"] = weight - 1
            self._write_labels(s)
            self._save_manifest()
            self._bump(weight)
        self._notify()

    def _notify(self):
        if self.on_change:
            try:
                self.on_change()
            except Exception:  # noqa: BLE001
                pass

    def delete(self, sid):
        with self._lock:
            s = self.get(sid)
            if s is None:
                return
            for name in self._names(s):
                (self.root / "images" / s["split"] / f"{name}.jpg").unlink(missing_ok=True)
                (self.root / "labels" / s["split"] / f"{name}.txt").unlink(missing_ok=True)
            self.thumb_path(sid).unlink(missing_ok=True)
            self._samples.remove(s)
            self._save_manifest()
        self._notify()

    def ready_for_training(self) -> tuple[bool, str]:
        c = self.counts()
        anchors = self.anchor_count()
        if anchors["train"] >= 20 and anchors["val"] >= 5:
            if c["train_pos"] + c["train_neg"] + c["val_pos"] + c["val_neg"] < 10:
                return False, "needs at least 10 checked detections (fire or 'not fire')"
            return True, "ok"
        if c["train_pos"] < 4:
            return False, f"needs at least 4 confirmed fire samples (has {c['train_pos']})"
        if c["val_pos"] < 1:
            return False, "needs at least 1 confirmed fire sample in the validation split"
        return True, "ok"


class ClsDataset(_Manifest):
    """Image-folder dataset: root/{train,val}/{class}/{id}.jpg"""
    kind = "classify"

    def __init__(self, root: Path, classes: list[str], val_fraction=0.2, max_per_class=3000):
        super().__init__(root)
        self.classes = list(classes)
        self.val_fraction = val_fraction
        self.max_per_class = max_per_class
        for split in ("train", "val"):
            for c in self.classes:
                (self.root / split / c).mkdir(parents=True, exist_ok=True)

    @property
    def train_data(self):
        return str(self.root)

    def counts(self) -> dict:
        with self._lock:
            by = {f"{sp}_{c}": 0 for sp in ("train", "val") for c in self.classes}
            for s in self._samples:
                by[f"{s['split']}_{s['label']}"] = by.get(f"{s['split']}_{s['label']}", 0) + 1
            return {"by_split": by, "by_class": {c: by[f"train_{c}"] + by[f"val_{c}"] for c in self.classes},
                    "new_samples": self.state["new_samples"], "total": len(self._samples)}

    def image_path(self, s) -> Path:
        return self.root / s["split"] / s["label"] / f"{s['id']}.jpg"

    thumb_path_for = image_path

    def sample_label(self, s) -> str:
        return s["label"]

    def add(self, crop, label: str, meta: dict, weight: int = 1) -> dict | None:
        if crop is None or crop.size == 0 or label not in self.classes:
            return None
        with self._lock:
            same = [s for s in self._samples if s["label"] == label]
            if weight == 1 and len(same) >= self.max_per_class:
                return None
            n_val = sum(1 for s in same if s["split"] == "val")
            split = "val" if weight == 1 and n_val < self.val_fraction * (len(same) + 1) else "train"
            h, w = crop.shape[:2]
            sc = min(1.0, 320 / max(h, w))
            img = cv2.resize(crop, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else crop
            rec = {"id": _new_id(), "split": split, "label": label, "ts": time.time(),
                   "dups": max(0, weight - 1) if split == "train" else 0, "manual": weight > 1, **meta}
            d = self.root / split / label
            cv2.imwrite(str(d / f"{rec['id']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            for k in range(rec["dups"]):
                cv2.imwrite(str(d / f"{rec['id']}_dup{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self._append(rec)
            self._bump(weight)
            return rec

    def _files(self, s):
        d = self.root / s["split"] / s["label"]
        return [d / f"{s['id']}.jpg"] + [d / f"{s['id']}_dup{k}.jpg" for k in range(s.get("dups", 0))]

    def relabel(self, sid, label: str, weight: int = 3):
        with self._lock:
            s = self.get(sid)
            if s is None or label not in self.classes:
                return
            src = self._files(s)[0]
            for f in self._files(s)[1:]:
                f.unlink(missing_ok=True)
            s["label"], s["manual"] = label, True
            s["dups"] = weight - 1 if s["split"] == "train" else 0
            dst = self.root / s["split"] / label / f"{sid}.jpg"
            if src.exists() and src != dst:
                shutil.move(str(src), dst)
            for k in range(s["dups"]):
                shutil.copy2(dst, dst.with_name(f"{sid}_dup{k}.jpg"))
            self._save_manifest()
            self._bump(weight)

    def delete(self, sid):
        with self._lock:
            s = self.get(sid)
            if s is None:
                return
            for f in self._files(s):
                f.unlink(missing_ok=True)
            self._samples.remove(s)
            self._save_manifest()

    def ready_for_training(self) -> tuple[bool, str]:
        by = self.counts()["by_split"]
        for c in self.classes:
            if by[f"train_{c}"] < 5:
                return False, f"needs at least 5 '{c}' training samples (has {by[f'train_{c}']})"
            if by[f"val_{c}"] < 1:
                return False, f"needs at least 1 '{c}' validation sample"
        return True, "ok"
