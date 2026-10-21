"""YOLO-based detectors (person pose, PPE, fire/smoke) plus a colour/flicker fire heuristic."""
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def select_device(pref: str = "auto") -> str:
    import torch
    if pref in (None, "", "auto"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return pref


HF_MIRROR = "https://huggingface.co/Ultralytics/{repo}/resolve/main/{name}"


def ensure_weights(weights: Path) -> None:
    """Official YOLO weights are auto-downloaded by ultralytics from GitHub; some networks block GitHub's asset
    host, so try Ultralytics' official Hugging Face mirror first."""
    weights = Path(weights)
    repos = {"yolo11": "YOLO11", "yolo26": "YOLO26", "yolov8": "YOLOv8"}
    repo = next((r for k, r in repos.items() if weights.name.startswith(k)), None)
    if weights.exists() or repo is None:
        return
    import requests
    weights.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(HF_MIRROR.format(repo=repo, name=weights.name), timeout=60, stream=True)
        r.raise_for_status()
        tmp = weights.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(weights)
    except Exception as e:
        log.warning("Mirror download of %s failed (%s); falling back to ultralytics", weights.name, e)


class YoloModel:
    def __init__(self, weights: Path, device: str, imgsz: int):
        from ultralytics import YOLO
        weights = Path(weights)
        ensure_weights(weights)
        self.path = weights
        self.model = YOLO(str(weights))  # official model names are auto-downloaded into models/
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self.device = device
        self.imgsz = imgsz
        # FP16 on GPU; ultralytics >= 8.4 renamed 'half' to 'quantize'
        from ultralytics.cfg import DEFAULT_CFG_DICT
        fp16 = device.startswith("cuda")
        self._precision = ({"quantize": 16 if fp16 else None} if "quantize" in DEFAULT_CFG_DICT else {"half": fp16})
        self._lock = threading.Lock()

    def predict(self, frames, conf, imgsz=None):
        """frames: one image or a list (batched on the GPU). Returns a list of results for a list input."""
        with self._lock:
            res = self.model.predict(frames, conf=conf, imgsz=imgsz or self.imgsz, device=self.device, verbose=False,
                                     **self._precision)
        return res if isinstance(frames, list) else res[0]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class PoseDetector:
    def __init__(self, weights, device, imgsz, batch=8):
        self.m = YoloModel(weights, device, imgsz)
        self.batch = batch

    def detect_batch(self, frames, conf) -> list[list[dict]]:
        out = []
        for chunk in _chunks(frames, self.batch):
            for r in self.m.predict(list(chunk), conf):
                out.append(self._parse(r))
        return out

    def detect(self, frame, conf) -> list[dict]:
        return self.detect_batch([frame], conf)[0]

    def _parse(self, r) -> list[dict]:
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
        for i in range(len(boxes)):
            if self.m.names.get(cls[i], "person") != "person":
                continue
            out.append({"box": boxes[i].tolist(), "conf": float(confs[i]),
                        "kpts": kpts[i] if kpts is not None else None})
        return out


class ClsModel:
    """A learned image classifier (person verifier, man-down verifier, PPE item). Hot-reloadable."""

    def __init__(self, weights, device):
        self.path = Path(weights)
        self.m = YoloModel(self.path, device, 224)
        self.version = _version_name(self.path)

    def predict(self, crops) -> list[tuple[str, float]]:
        crops = [c for c in crops]
        if not crops:
            return []
        out = []
        for chunk in _chunks(crops, 32):
            for r in self.m.predict(list(chunk), conf=0.0, imgsz=224):
                i = int(r.probs.top1)
                out.append((self.m.names[i], float(r.probs.top1conf)))
        return out


class HeuristicFire:
    """Finds flame-coloured regions that also flicker between frames. High recall, many false positives:
    every candidate is sent to the local AI for verification, and the verified results train the real model."""

    def __init__(self, max_side=480):
        self.max_side = max_side
        self.prev = {}

    def detect(self, key, frame) -> list[dict]:
        h, w = frame.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        b, g, r = [c.astype(np.int16) for c in cv2.split(small)]
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        color = (r > 180) & (r >= g) & (g > b) & (sat > 70) & (val > 170) & ((hue <= 35) | (hue >= 170))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev = self.prev.get(key)
        self.prev[key] = gray
        if prev is None or prev.shape != gray.shape:
            return []
        motion = cv2.dilate((cv2.absdiff(gray, prev) > 12).astype(np.uint8), np.ones((7, 7), np.uint8))
        mask = (color & (motion > 0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.0008 * mask.shape[0] * mask.shape[1]
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            fill = float(color[y:y + bh, x:x + bw].mean())
            pad = 0.15
            x1, y1 = max(0, x - pad * bw), max(0, y - pad * bh)
            x2, y2 = min(mask.shape[1], x + bw * (1 + pad)), min(mask.shape[0], y + bh * (1 + pad))
            out.append({"box": [x1 / s, y1 / s, x2 / s, y2 / s], "conf": round(min(0.99, 0.2 + fill), 3),
                        "label": "fire", "source": "heuristic", "area": area})
        out.sort(key=lambda d: d["area"], reverse=True)
        return out[:3]


class FireDetector:
    def __init__(self, weights, device, imgsz, mode="hybrid", batch=4):
        self.device, self.imgsz, self.mode, self.batch = device, imgsz, mode, batch
        self.model = None
        self.version = "none (heuristic only)"
        self._lock = threading.Lock()
        self.heuristic = HeuristicFire()
        self.heuristic_stride = 3  # colour/flicker check on every 3rd analysed frame per camera (~1/s)
        self._count = {}
        self.load(weights)

    def load(self, weights) -> bool:
        weights = Path(weights)
        if not weights.exists():
            return False
        try:
            m = YoloModel(weights, self.device, self.imgsz)
        except Exception as e:  # corrupt / incompatible weights: keep the previous model
            log.error("Could not load fire model %s: %s", weights, e)
            return False
        with self._lock:
            self.model = m
            self.version = weights.name if weights.name != "current.pt" else _version_name(weights)
        return True

    def detect_batch(self, keys, frames, conf) -> list[list[dict]]:
        with self._lock:
            model = self.model
        outs = [[] for _ in frames]
        if model is not None and self.mode in ("hybrid", "model"):
            results = []
            for chunk in _chunks(list(frames), self.batch):
                results.extend(model.predict(list(chunk), conf))
            for out, r in zip(outs, results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    name = model.names.get(cls[i], "fire").lower()
                    if "smoke" in name:
                        label = "smoke"
                    elif "fire" in name or "flame" in name:
                        label = "fire"
                    else:
                        continue
                    out.append({"box": boxes[i].tolist(), "conf": float(confs[i]), "label": label, "source": "model"})
        if self.mode in ("heuristic", "hybrid") or model is None:
            from .analytics import box_iou
            for key, frame, out in zip(keys, frames, outs):
                n = self._count[key] = self._count.get(key, -1) + 1
                if n % self.heuristic_stride:
                    continue
                for c in self.heuristic.detect(key, frame):
                    if not any(box_iou(c["box"], o["box"]) > 0.2 for o in out):
                        out.append(c)
        for out in outs:
            out.sort(key=lambda d: (d["source"] == "model", d["conf"]), reverse=True)
        return outs


def _version_name(current: Path) -> str:
    marker = current.with_suffix(".version")
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else "custom"
