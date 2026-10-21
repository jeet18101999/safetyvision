"""Local AI verification through Ollama (vision model, e.g. qwen2.5vl). Nothing leaves this PC."""
import base64
import itertools
import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field

import cv2
import requests

log = logging.getLogger(__name__)

FIRE_PROMPT = """You are the fire-safety verification system for an industrial CCTV installation.
Image 1 is a close-up crop of a region that an automatic detector flagged as possible FIRE or SMOKE.
Image 2 is the full camera frame; the flagged region is marked with a red rectangle.

Decide whether there is REAL fire (visible flames) and/or REAL smoke in the flagged region.
These are NOT fire: orange/red/yellow clothing, vests or helmets, painted objects, signs, lamps, LED or
sodium lights, headlights/tail lights, sunlight, sunsets, reflections, screens/monitors, welding arcs or
sparks from normal hot work, steam from vents, fog, dust, and camera glare.
Only answer fire=true if you can actually see flames. Only answer smoke=true for real smoke plumes.
Give confidence between 0 and 1 for your overall answer and a short reason."""

FIRE_SCHEMA = {
    "type": "object",
    "properties": {
        "fire": {"type": "boolean"},
        "smoke": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["fire", "smoke", "confidence", "reason"],
}

PPE_DESCRIPTIONS = {
    "helmet": "a safety helmet / hard hat worn on the head",
    "vest": "a high-visibility / reflective safety vest or jacket",
    "mask": "a face mask or respirator",
    "gloves": "work gloves on the hands",
    "goggles": "safety goggles or glasses",
    "boots": "safety shoes or boots",
}


@dataclass(order=True)
class _Job:
    priority: int
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False)
    callback: object = field(compare=False)


def _b64(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def crop_box(frame, box, pad=0.35, min_side=64):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = max(bw * pad, (min_side - bw) / 2), max(bh * pad, (min_side - bh) / 2)
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return frame[y1:y2, x1:x2].copy()


def list_models(url, timeout=5) -> list[str]:
    r = requests.get(url.rstrip("/") + "/api/tags", timeout=timeout)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


class LocalAIVerifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = cfg["ollama_url"].rstrip("/")
        self.model = cfg["model"]
        self.timeout = float(cfg.get("timeout", 120))
        self._q: queue.PriorityQueue = queue.PriorityQueue(maxsize=64)
        self._seq = itertools.count()
        self._stop = threading.Event()
        self._thread = None
        self.online = None
        self.status = "starting"
        self.busy = False
        self.avg_latency = 0.0
        self.stats = {"fire_confirmed": 0, "fire_rejected": 0, "ppe_checks": 0, "person_checks": 0,
                      "posture_checks": 0, "errors": 0}

    # ---- lifecycle -------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="local-ai", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def pending(self) -> int:
        return self._q.qsize() + (1 if self.busy else 0)

    def check_health(self) -> tuple[bool, str]:
        try:
            models = list_models(self.url)
        except Exception as e:
            return False, f"Ollama not reachable at {self.url} ({e.__class__.__name__})"
        if not any(m == self.model or m.split(":")[0] == self.model for m in models):
            return False, f"Model '{self.model}' not installed. Run: ollama pull {self.model}"
        return True, f"{self.model} ready"

    # ---- submit ----------------------------------------------------
    def _put(self, job):
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            return False

    def submit_fire(self, frame, box, meta: dict, callback) -> bool:
        """callback(result) where result = {fire, smoke, confidence, reason, ok} or {ok: False, error}"""
        return self._put(_Job(0, next(self._seq), "fire", {"frame": frame, "box": box, "meta": meta}, callback))

    def submit_ppe(self, crop, required: list[str], meta: dict, callback) -> bool:
        """callback(result) where result = {ok, visible, missing: [...]}"""
        return self._put(_Job(1, next(self._seq), "ppe", {"crop": crop, "required": required, "meta": meta}, callback))

    def submit_person(self, frame, box, meta: dict, callback) -> bool:
        """Is the static object in the box a real human? callback({ok, person, confidence, reason})"""
        return self._put(_Job(2, next(self._seq), "person", {"frame": frame, "box": box, "meta": meta}, callback))

    def submit_posture(self, frame, box, meta: dict, callback) -> bool:
        """Is this person lying / collapsed on the ground? callback({ok, down, confidence, reason})"""
        return self._put(_Job(0, next(self._seq), "posture", {"frame": frame, "box": box, "meta": meta}, callback))

    def submit_same_person(self, crop_a, crop_b, meta: dict, callback) -> bool:
        """Are two body crops the same person? callback({ok, same, confidence, reason})"""
        return self._put(_Job(3, next(self._seq), "same_person", {"a": crop_a, "b": crop_b, "meta": meta}, callback))

    # ---- worker ----------------------------------------------------
    def _run(self):
        next_health = 0.0
        while not self._stop.is_set():
            if time.time() >= next_health:
                self.online, self.status = self.check_health()
                next_health = time.time() + 30
            try:
                job = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            self.busy = True
            t0 = time.time()
            try:
                if not self.online:
                    raise RuntimeError(self.status)
                result = {"fire": self._verify_fire, "ppe": self._check_ppe, "person": self._check_person,
                          "posture": self._check_posture, "same_person": self._check_same_person}[job.kind](job.payload)
                result["ok"] = True
                dt = time.time() - t0
                self.avg_latency = dt if not self.avg_latency else 0.8 * self.avg_latency + 0.2 * dt
            except Exception as e:
                self.stats["errors"] += 1
                log.warning("Local AI %s check failed: %s", job.kind, e)
                result = {"ok": False, "error": str(e)}
                if not isinstance(e, RuntimeError):
                    next_health = 0  # re-check connectivity soon
            finally:
                self.busy = False
            result["latency"] = round(time.time() - t0, 2)
            try:
                job.callback(result)
            except Exception:
                log.exception("verifier callback failed")

    def _chat(self, prompt, images, schema) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "stream": False,
            "format": schema,
            "keep_alive": "30m",
            "options": {"temperature": 0},
        }
        r = requests.post(self.url + "/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        text = r.json()["message"]["content"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise ValueError(f"Unparseable answer: {text[:200]}")
            return json.loads(m.group(0))

    def _verify_fire(self, p) -> dict:
        frame, box = p["frame"], p["box"]
        crop = crop_box(frame, box)
        full = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), max(2, frame.shape[1] // 300))
        ans = self._chat(FIRE_PROMPT, [_b64(crop, 640), _b64(full, 896)], FIRE_SCHEMA)
        res = {"fire": bool(ans.get("fire")), "smoke": bool(ans.get("smoke")),
               "confidence": float(ans.get("confidence") or 0.0), "reason": str(ans.get("reason", ""))[:300]}
        if res["confidence"] > 1:
            res["confidence"] /= 100.0
        positive = (res["fire"] or res["smoke"]) and res["confidence"] >= float(self.cfg.get("min_confidence", 0.6))
        res["positive"] = positive
        self.stats["fire_confirmed" if positive else "fire_rejected"] += 1
        return res

    def _check_ppe(self, p) -> dict:
        required = p["required"]
        items = "\n".join(f"- {r}: {PPE_DESCRIPTIONS.get(r, r)}" for r in required)
        prompt = ("This is a CCTV crop of one worker at an industrial site. For each item below answer true if the "
                  "person is clearly wearing it, false if not. Set visible=false if the person is too small, blurred, "
                  "or occluded to judge.\n" + items)
        props = {r: {"type": "boolean"} for r in required}
        props["visible"] = {"type": "boolean"}
        schema = {"type": "object", "properties": props, "required": list(props)}
        ans = self._chat(prompt, [_b64(p["crop"], 512)], schema)
        self.stats["ppe_checks"] += 1
        visible = bool(ans.get("visible", True))
        items = {r: bool(ans.get(r, False)) for r in required}
        missing = [r for r in required if not items[r]] if visible else []
        return {"visible": visible, "missing": missing, "items": items}

    def _two_images(self, frame, box):
        crop = crop_box(frame, box, pad=0.25)
        full = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), max(2, frame.shape[1] // 300))
        return [_b64(crop, 512), _b64(full, 896)]

    def _check_same_person(self, p) -> dict:
        prompt = ("Two CCTV crops of people at a workplace. Are they the SAME person? Compare body build and height, "
                  "clothing (colours, patterns, uniform details), footwear, hair, bags and accessories. People in the "
                  "same uniform are NOT the same person unless build, hair and details also match. Answer same=true "
                  "only if you are fairly sure.")
        schema = {"type": "object", "properties": {"same": {"type": "boolean"}, "confidence": {"type": "number"},
                                                   "reason": {"type": "string"}},
                  "required": ["same", "confidence", "reason"]}
        ans = self._chat(prompt, [_b64(p["a"], 384), _b64(p["b"], 384)], schema)
        self.stats["person_checks"] += 1
        conf = float(ans.get("confidence") or 0)
        return {"same": bool(ans.get("same")), "confidence": conf / 100 if conf > 1 else conf,
                "reason": str(ans.get("reason", ""))[:200]}

    def _check_person(self, p) -> dict:
        prompt = ("CCTV check. Image 1 is a close-up of an object that a person detector found and that has not moved "
                  "for a long time. Image 2 is the full camera frame with the object in a red rectangle. "
                  "Is it a REAL HUMAN being (standing, sitting, working, sleeping or lying)? Answer person=false for "
                  "mannequins, statues, posters/pictures of people, clothes or jackets hanging, PPE hanging on a hook, "
                  "chairs, machines, boxes, shadows and reflections.")
        schema = {"type": "object", "properties": {"person": {"type": "boolean"}, "confidence": {"type": "number"},
                                                   "reason": {"type": "string"}},
                  "required": ["person", "confidence", "reason"]}
        ans = self._chat(prompt, self._two_images(p["frame"], p["box"]), schema)
        self.stats["person_checks"] += 1
        conf = float(ans.get("confidence") or 0)
        return {"person": bool(ans.get("person")), "confidence": conf / 100 if conf > 1 else conf,
                "reason": str(ans.get("reason", ""))[:300]}

    def _check_posture(self, p) -> dict:
        prompt = ("Industrial safety CCTV. Image 1 is a close-up of a person, image 2 is the full frame with the person "
                  "in a red rectangle. Is this person DOWN - fallen, collapsed or lying on the floor/ground and possibly "
                  "needing help? Answer down=false if they are standing, walking, sitting on a chair, crouching, bending "
                  "over to work, kneeling to work, or lying on a bed/sofa. If it is not a real person, answer down=false "
                  "and person=false.")
        schema = {"type": "object", "properties": {"down": {"type": "boolean"}, "person": {"type": "boolean"},
                                                   "confidence": {"type": "number"}, "reason": {"type": "string"}},
                  "required": ["down", "person", "confidence", "reason"]}
        ans = self._chat(prompt, self._two_images(p["frame"], p["box"]), schema)
        self.stats["posture_checks"] += 1
        conf = float(ans.get("confidence") or 0)
        return {"down": bool(ans.get("down")), "person": bool(ans.get("person", True)),
                "confidence": conf / 100 if conf > 1 else conf, "reason": str(ans.get("reason", ""))[:300]}
