"""Fire simulation: render realistic fire / smoke onto real footage from your own cameras, train the fire
detector on it, and measure the result.

    python -m safetyvision.simulate_fire            (uses the clips recorded by the load test)

Steps
  1. scenarios   - procedural flames (Doom-fire style heat simulation, black-body colours, glow, floor light) and
                   smoke plumes composited onto real camera clips, with exact bounding boxes
  2. local AI    - does the local vision model confirm the simulated fires? (the pipeline's verification step)
  3. dataset     - positives from simulated fires on training cameras, negatives from real footage without fire
                   (plus the real 'not fire' samples the local AI already rejected on your site)
  4. training    - YOLO fire/smoke detector, same training job the app uses
  5. evaluation  - held-out cameras never seen in training: detection rate, precision, time-to-detect;
                   real no-fire footage: false candidates per camera-hour, trained model vs colour heuristic
  6. report      - data/simulation/report.html with example images

Simulated fire is not real fire: treat the numbers as a pipeline check and a head start. The model keeps
improving from real confirmed detections and your reviews once it runs in the app.
"""
import argparse
import base64
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as C

RNG = np.random.default_rng(7)
random.seed(7)


def say(msg):
    print(f"[simulate] {msg}", flush=True)


# ============================================================================ rendering
def _palette():
    """Heat 0..1 -> BGR black-body-like colour."""
    stops = [(0.0, (0, 0, 0)), (0.18, (10, 20, 120)), (0.38, (0, 60, 220)), (0.58, (10, 150, 255)),
             (0.78, (60, 220, 255)), (1.0, (210, 250, 255))]
    lut = np.zeros((256, 3), np.float32)
    for k in range(256):
        h = k / 255
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if a <= h <= b:
                t = (h - a) / (b - a)
                lut[k] = np.array(ca) * (1 - t) + np.array(cb) * t
                break
    return lut


PAL = _palette()


class Flame:
    """Low-resolution heat field propagated upwards with random cooling and sideways drift."""

    def __init__(self, w=36, h=64, width_frac=0.55, cooling=0.035, wind=0.0):
        self.w, self.h = w, h
        self.heat = np.zeros((h, w), np.float32)
        x = np.linspace(-1, 1, w)
        self.profile = np.exp(-(x / width_frac) ** 2 * 2.2)
        self.cooling, self.wind, self.t = cooling, wind, 0

    def step(self):
        h, w = self.h, self.w
        self.t += 1
        src = np.clip(self.profile * 1.15, 0, 1) * RNG.uniform(0.8, 1.0, w) * (0.92 + 0.08 * np.sin(self.t * 0.7))
        self.heat[-1] = src
        self.heat[-2] = np.maximum(self.heat[-2], src * RNG.uniform(0.8, 1.0, w))
        cols = np.arange(w)
        for y in range(h - 2, -1, -1):
            r = RNG.integers(0, 3, w)
            sway = self.wind + np.sin(self.t * 0.21 + y * 0.15) * 0.5  # occasional sideways push, not every row
            drift = int(np.sign(sway)) if RNG.random() < abs(sway) * 0.12 else 0
            dst = np.clip(cols - r + 1 + drift, 0, w - 1)
            edge = 1.0 + 1.2 * (1 - self.profile)  # cools faster at the sides -> tongue-shaped flame
            vals = self.heat[y + 1] - ((r & 1) * self.cooling + RNG.uniform(0, self.cooling * 0.6, w)) * edge
            row = self.heat[y] * 0.9  # cells nobody writes to keep (most of) last frame's heat, like the Doom fire
            row[dst] = np.maximum(vals, 0)
            self.heat[y] = row
        return self.heat


class Smoke:
    """Rising, spreading plume from scrolling multi-octave noise."""

    def __init__(self, w=64, h=96):
        self.w, self.h = w, h
        self.field = self._noise()
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        spread = 0.3 + 0.6 * (1 - yy / h)  # widens with height
        self.mask = np.clip(1 - np.abs((xx - w / 2) / (w / 2)) / spread, 0, 1) * np.clip((yy / h) * 1.6, 0, 1)
        self.mask *= np.clip((h - yy) / (h * 0.15), 0, 1) ** 0.3

    def _noise(self):
        f = np.zeros((self.h, self.w), np.float32)
        for s, a in ((4, 0.5), (8, 0.3), (16, 0.2)):
            f += a * cv2.resize(RNG.random((max(2, self.h // s), max(2, self.w // s))).astype(np.float32),
                                (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        return f

    def step(self):
        self.field = np.roll(self.field, -2, axis=0)
        self.field[-2:] = self._noise()[-2:]
        self.field = cv2.GaussianBlur(self.field, (0, 0), 0.8) * 0.97 + self._noise() * 0.03
        return np.clip((self.field - 0.22) * 2.6, 0, 1) ** 0.7 * self.mask


def composite(frame, flame_heat, fx, fy, fw, fh, smoke=None, smoke_box=None, glow=True):
    """Blend flame (heat field) with bottom-centre at (fx, fy) and size fw x fh; smoke plume above it."""
    H, W = frame.shape[:2]
    out = frame.astype(np.float32)
    boxes = []
    if smoke is not None:
        sx, sy, sw, sh, density, tone = smoke_box
        a = cv2.resize(smoke, (sw, sh), interpolation=cv2.INTER_CUBIC) * density
        x0, y0 = int(sx - sw / 2), int(sy - sh)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + sw), min(H, y0 + sh)
        if xb > xa and yb > ya:
            al = a[ya - y0:yb - y0, xa - x0:xb - x0, None]
            before = out[ya:yb, xa:xb].copy()
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - al) + tone * al
            # label only smoke that is actually visible (light smoke on a light floor can be invisible)
            m = ((np.abs(out[ya:yb, xa:xb] - before).mean(axis=2) > 18) & (al[..., 0] > 0.15)).astype(np.uint8)
            if m.sum() > 50:
                x, y, w, h = cv2.boundingRect(m)
                boxes.append((1, [xa + x, ya + y, xa + x + w, ya + y + h]))
    if flame_heat is not None:
        heat = cv2.resize(flame_heat, (fw, fh), interpolation=cv2.INTER_CUBIC)
        heat = cv2.GaussianBlur(np.clip(heat, 0, 1), (0, 0), max(0.8, fw / 60))
        col = PAL[np.clip(heat * 255, 0, 255).astype(np.uint8)]
        alpha = np.clip((heat - 0.07) / 0.3, 0, 1)[..., None] ** 0.8
        x0, y0 = int(fx - fw / 2), int(fy - fh)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + fw), min(H, y0 + fh)
        if xb > xa and yb > ya:
            sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
            region = out[ya:yb, xa:xb]
            out[ya:yb, xa:xb] = region * (1 - alpha[sl]) + col[sl] * alpha[sl]
            if glow:  # light cast on the surroundings (flickers with the flame intensity)
                inten = float(heat.mean()) * 2.2
                gx0, gy0 = max(0, int(fx - fw * 2)), max(0, int(fy - fh * 1.3))
                gx1, gy1 = min(W, int(fx + fw * 2)), min(H, int(fy + fh * 0.5))
                yy, xx = np.mgrid[gy0:gy1, gx0:gx1].astype(np.float32)
                d = np.sqrt(((xx - fx) / (fw * 1.6)) ** 2 + ((yy - fy + fh * 0.3) / (fh * 1.1)) ** 2)
                g = (np.clip(1 - d, 0, 1) ** 2 * inten * 0.55)[..., None]
                out[gy0:gy1, gx0:gx1] += g * np.array([20, 110, 255], np.float32)
            m = (alpha[sl][..., 0] > 0.35).astype(np.uint8)
            if m.sum() > 20:
                x, y, w, h = cv2.boundingRect(m)
                boxes.append((0, [xa + x, ya + y, xa + x + w, ya + y + h]))
    return np.clip(out, 0, 255).astype(np.uint8), boxes


# ============================================================================ scenarios
def read_clip(path, max_frames=240, step=1):
    cap = cv2.VideoCapture(str(path))
    frames = []
    k = 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            frames.append(f)
        k += 1
    cap.release()
    return frames


def make_scenario(frames, kind, size, rng):
    """Returns (frames_with_fire, boxes_per_frame, meta)."""
    H, W = frames[0].shape[:2]
    fh = int(H * {"small": rng.uniform(0.07, 0.11), "medium": rng.uniform(0.12, 0.2),
                  "large": rng.uniform(0.22, 0.34)}[size])
    fw = int(fh * rng.uniform(0.55, 0.85))
    fx = int(rng.uniform(0.15, 0.85) * W)
    fy = int(rng.uniform(0.45, 0.92) * H)  # base on the floor area
    flame = Flame(width_frac=rng.uniform(0.35, 0.6), cooling=rng.uniform(0.006, 0.010), wind=rng.uniform(-0.5, 0.5)) \
        if kind in ("fire", "fire+smoke") else None
    smoke = Smoke() if kind in ("smoke", "fire+smoke") else None
    tone = rng.uniform(125, 195) if rng.random() < 0.7 else rng.uniform(35, 60)  # light grey or black smoke
    ramp = int(len(frames) * 0.15)  # fire grows during the first part of the clip
    out_frames, out_boxes = [], []
    for _ in range(20):  # let the simulation settle
        flame and flame.step()
        smoke and smoke.step()
    for k, fr in enumerate(frames):
        g = min(1.0, 0.35 + 0.65 * k / max(1, ramp))
        heat = flame.step() if flame else None
        sm = smoke.step() if smoke else None
        sbox = (fx, fy - int(fh * (0.35 if flame else 0.0) * g), int(fw * 3.4 * g), int(fh * 3.0 * g),
                rng.uniform(0.75, 0.95) if kind == "smoke" else 0.7, tone) if smoke else None
        img, boxes = composite(fr, heat, fx, fy, max(8, int(fw * g)), max(12, int(fh * g)), sm, sbox)
        out_frames.append(img)
        out_boxes.append(boxes)
    return out_frames, out_boxes, {"kind": kind, "size": size, "x": fx, "y": fy, "h": fh}


# ============================================================================ helpers
def yolo_line(cls, box, W, H):
    x1, y1, x2, y2 = box
    return f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def iou(a, b):
    from .analytics import box_iou
    return box_iou(a, b)


def b64img(img, w=420):
    s = w / img.shape[1]
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, int(img.shape[0] * s))), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(C.DATA_DIR / "loadtests" / "clips"))
    ap.add_argument("--out", default=str(C.DATA_DIR / "simulation"))
    ap.add_argument("--test-cams", default="CH3,CH8,CH12,CH15")
    ap.add_argument("--scenarios-per-cam", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--extra-negatives", default="", help="other fire_dataset folders whose 'not fire' samples to reuse")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--neg-step", type=int, default=4, help="every Nth real frame becomes a 'no fire' example")
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--no-learning-loop", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    prev_model = None
    if (out / "fire_sim.pt").exists() and not a.no_hard_negatives:  # previous run's model mines hard negatives
        prev_model = out.parent / "fire_sim_previous.pt"
        shutil.copy2(out / "fire_sim.pt", prev_model)
    shutil.rmtree(out, ignore_errors=True)
    ds = out / "dataset"
    for sp in ("train", "val", "test"):
        (ds / "images" / sp).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / sp).mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(parents=True)
    clips = sorted(Path(a.clips).glob("*.mp4"))
    if not clips:
        say("no clips found - run the load test once (it records them) or pass --clips")
        sys.exit(1)
    test_cams = {c.strip() for c in a.test_cams.split(",")}
    rng = np.random.default_rng(11)
    say(f"{len(clips)} camera clips, held-out test cameras: {sorted(test_cams)}")

    # ---------------------------------------------------------------- 1. scenarios + dataset
    scenarios = []
    counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}

    def save(split, name, img, lines):
        cv2.imwrite(str(ds / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (ds / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split][0 if lines else 1] += 1

    kinds = ["fire", "fire", "fire+smoke", "smoke", "fire", "fire+smoke"]
    sizes = ["small", "medium", "large"]
    for ci, clip in enumerate(clips):
        cam = clip.stem
        frames = read_clip(clip, 240, step=2)  # ~12 fps
        if len(frames) < 30:
            continue
        H, W = frames[0].shape[:2]
        is_test = cam in test_cams
        # real negatives from this camera (no fire in the footage) - only from the first half of the clip:
        # the second half is kept unseen for measuring false alarms
        half = len(frames) // 2
        if not is_test:
            for k in range(0, half, a.neg_step):
                save("val" if k % (a.neg_step * 5) == 0 else "train", f"neg_{cam}_{k:03d}", frames[k], [])
        n_sc = a.scenarios_per_cam + (0 if is_test else 1)  # one extra small fire per training camera
        for si in range(n_sc):
            kind = kinds[(ci * a.scenarios_per_cam + si) % len(kinds)] if si < a.scenarios_per_cam else "fire"
            size = sizes[(ci + si) % 3] if si < a.scenarios_per_cam else "small"
            fr, bx, meta = make_scenario(frames[:120], kind, size, rng)
            meta.update(camera=cam, split="test" if is_test else "train", id=f"{cam}_s{si}")
            scenarios.append({"meta": meta, "frames": fr, "boxes": bx})
            for k in range(6, len(fr), 5 if size != "small" else 3):
                lines = [yolo_line(c, b, W, H) for c, b in bx[k]]
                split = "test" if is_test else ("val" if (k // 5) % 7 == 0 else "train")
                save(split, f"sim_{meta['id']}_{k:03d}", fr[k], lines)
            if si == 0:
                cv2.imwrite(str(out / "examples" / f"{meta['id']}.jpg"), fr[len(fr) // 2])
        if prev_model and not is_test:  # hard negatives: real frames where the previous model saw fire
            from ultralytics import YOLO
            pm = YOLO(str(prev_model))
            hn = 0
            for k, r in enumerate(pm.predict(frames[:half], conf=0.2, imgsz=640, verbose=False)):
                if r.boxes is not None and len(r.boxes) and k % 2 == 0:
                    save("train", f"hardneg_{cam}_{k:03d}", frames[k], [])
                    hn += 1
            if hn:
                say(f"{cam}: {hn} hard negatives (previous model's false alarms)")
        say(f"{cam}: {n_sc} scenarios ({'test' if is_test else 'train'})")

    # real 'not fire' samples the local AI rejected on site
    for extra in [p for p in a.extra_negatives.split(";") if p]:
        n = 0
        for img in Path(extra).glob("images/*/*.jpg"):
            lab = Path(extra) / "labels" / img.parent.name / (img.stem + ".txt")
            if lab.exists() and lab.read_text().strip() == "":
                shutil.copy2(img, ds / "images" / "train" / f"ainot_{img.name}")
                (ds / "labels" / "train" / f"ainot_{img.stem}.txt").write_text("", encoding="utf-8")
                counts["train"][1] += 1
                n += 1
        say(f"added {n} real AI-rejected 'not fire' images from {extra}")
    say(f"dataset: train {counts['train'][0]} fire / {counts['train'][1]} no-fire, val {counts['val'][0]}/{counts['val'][1]}, "
        f"test {counts['test'][0]}/{counts['test'][1]}")
    for split in ("val", "test"):
        (ds / f"{split}.yaml").write_text(
            f"path: {ds.as_posix()}\ntrain: images/train\nval: images/{split}\nnames:\n  0: fire\n  1: smoke\n", encoding="utf-8")

    # ---------------------------------------------------------------- 2. local AI verdicts on simulated fires
    ai_rows = []
    if not a.no_ai:
        from .verifier import LocalAIVerifier
        cfg = C.load_config()
        v = LocalAIVerifier(cfg["verifier"])
        ok, msg = v.check_health()
        v.online = ok
        say(f"local AI: {msg}")
        if ok:
            for sc in [s for s in scenarios if s["meta"]["split"] == "test"]:
                k = len(sc["frames"]) * 2 // 3
                boxes = sc["boxes"][k]
                if not boxes:
                    continue
                box = min((b for _, b in boxes), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))  # flame box first
                t0 = time.time()
                try:
                    r = v._verify_fire({"frame": sc["frames"][k], "box": box})
                except Exception as e:
                    r = {"positive": False, "fire": False, "smoke": False, "confidence": 0, "reason": f"error: {e}"}
                ai_rows.append({**sc["meta"], "confirmed": r["positive"], "fire": r["fire"], "smoke": r["smoke"],
                                "confidence": r["confidence"], "reason": r["reason"], "secs": round(time.time() - t0, 1),
                                "img": b64img(sc["frames"][k])})
                say(f"AI on {sc['meta']['id']} ({sc['meta']['kind']}, {sc['meta']['size']}): "
                    f"{'CONFIRMED' if r['positive'] else 'rejected'} {r['confidence']:.2f} - {r['reason'][:80]}")

        try:  # free the GPU for training (the app reloads the model on its next check)
            import requests
            requests.post(cfg["verifier"]["ollama_url"].rstrip("/") + "/api/generate",
                          json={"model": cfg["verifier"]["model"], "keep_alive": 0}, timeout=30)
            time.sleep(3)
        except Exception:
            pass

    # ---------------------------------------------------------------- 3. training
    import os
    from ultralytics import YOLO

    def pick_device():
        if a.device != "auto":
            return a.device
        import torch
        return "0" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**20 > 2500 else "cpu"

    def train(weights, epochs, name):
        device = pick_device()
        say(f"training fire detector ({epochs} epochs on {'GPU' if device != 'cpu' else 'CPU'}) ...")
        runs = out / "runs"
        runs.mkdir(exist_ok=True)
        t0 = time.time()
        res_json = runs / f"{name}.json"
        cmd = [sys.executable, "-m", "safetyvision.train_job", "--task", "detect", "--data", str(ds / "val.yaml"),
               "--weights", str(weights), "--epochs", str(epochs), "--batch", "16", "--imgsz", "640",
               "--device", device, "--project", str(runs), "--name", name, "--out", str(res_json)]
        pr = subprocess.run(cmd, cwd=runs, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONPATH": str(C.ROOT)})
        (runs / f"{name}.log").write_text(pr.stdout + pr.stderr, encoding="utf-8")
        if pr.returncode != 0 or not res_json.exists():
            say(f"training failed - see runs/{name}.log")
            sys.exit(1)
        mins = (time.time() - t0) / 60
        say(f"trained in {mins:.1f} min")
        return Path(json.loads(res_json.read_text())["best"]), mins, device

    def real_frames(clip):
        frames = read_clip(clip, 240, step=5)  # ~3 fps, like live analysis
        return frames[: len(frames) // 2], frames[len(frames) // 2:]  # (first half, unseen second half)

    def evaluate(model_path, device, label):
        m = YOLO(str(model_path))
        val = m.val(data=str(ds / "test.yaml"), split="val", imgsz=640, device=device, workers=0, plots=False,
                    verbose=False, project=str(out / "runs"), name=f"eval_{label}")
        res = {"map50": float(val.box.map50), "precision": float(val.box.mp), "recall": float(val.box.mr)}
        scen_rows, examples = [], []
        for sc in [s for s in scenarios if s["meta"]["split"] == "test"]:
            hits, first, streak = 0, None, 0
            frames = sc["frames"][::2]
            preds = m.predict(frames, conf=0.25, imgsz=640, device=device, verbose=False)
            for k, r in enumerate(preds):
                gt = [b for _, b in sc["boxes"][k * 2]]
                pb = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
                hit = any(iou(p_, g) > 0.2 for p_ in pb for g in gt)
                hits += hit
                streak = streak + 1 if hit else 0
                if streak >= 2 and first is None:
                    first = (k * 2) / 12.0
            kk = len(preds) * 2 // 3
            img = frames[kk].copy()
            if preds[kk].boxes is not None:
                for bb, cc, cl in zip(preds[kk].boxes.xyxy.cpu().numpy(), preds[kk].boxes.conf.cpu().numpy(),
                                      preds[kk].boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, bb)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 235), 2)
                    cv2.putText(img, f"{'fire' if int(cl) == 0 else 'smoke'} {cc:.2f}", (x1, max(14, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 235), 2)
            examples.append((sc["meta"], b64img(img)))
            scen_rows.append({**sc["meta"], "rate": round(hits / max(1, len(frames)), 2), "time_to_detect": first})
        # false candidates on the unseen second half of every camera's real footage (no fire)
        from .detectors import HeuristicFire
        heur = HeuristicFire()
        fp = {"heuristic": 0, "model@0.25": 0, "model@0.50": 0}
        n, fp_examples, fp_cams = 0, [], {}
        for clip in clips:
            _, frames = real_frames(clip)
            preds = m.predict(frames, conf=0.25, imgsz=640, device=device, verbose=False) if frames else []
            for k, (f, r) in enumerate(zip(frames, preds)):
                n += 1
                fp["heuristic"] += bool(heur.detect(clip.stem, f))
                confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
                fp["model@0.25"] += bool(len(confs))
                fp["model@0.50"] += bool(any(c >= 0.5 for c in confs))
                if len(confs):
                    fp_cams[clip.stem] = fp_cams.get(clip.stem, 0) + 1
                if len(confs) and len(fp_examples) < 4 and k % 3 == 0:
                    img = f.copy()
                    for bb in r.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, bb)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), 2)
                    fp_examples.append((clip.stem, b64img(img)))
        hours = n / 3.0 / 3600
        res["fp_per_cam_hour"] = {k: round(v / hours) for k, v in fp.items()}
        res["fp_cameras"] = fp_cams
        det = sum(1 for r in scen_rows if r["time_to_detect"] is not None)
        say(f"[{label}] held-out: mAP50 {res['map50']:.3f}, precision {res['precision']:.3f}, recall {res['recall']:.3f}, "
            f"scenarios detected {det}/{len(scen_rows)}; false candidates/cam-hour {res['fp_per_cam_hour']} {fp_cams}")
        return res, scen_rows, examples, fp_examples

    best, train_min, device = train(C.resolve("models/yolo11n.pt"), a.epochs, "fire_sim")
    model_path = out / "fire_sim.pt"
    shutil.copy2(best, model_path)
    r1, scen_rows, examples, fp_examples = evaluate(model_path, device, "trained")
    rounds = [{"stage": "Trained on simulated fire", **{k: v for k, v in r1.items() if k != "fp_cameras"},
               "detected": sum(1 for r in scen_rows if r["time_to_detect"] is not None), "scenarios": len(scen_rows)}]

    # ---------------------------------------------------------------- 5. learning loop (what the app does live)
    loop = None
    if not a.no_ai and not a.no_learning_loop and r1["fp_cameras"]:
        from .verifier import LocalAIVerifier
        cfg = C.load_config()
        v = LocalAIVerifier(cfg["verifier"])
        v.online = v.check_health()[0]
        m = YOLO(str(model_path))
        checked = rejected = added = 0
        verdicts = []
        for clip in clips:  # the app sees the model's false candidates, asks the local AI, stores the verdicts
            first_half, _ = real_frames(clip)
            preds = m.predict(first_half, conf=0.25, imgsz=640, device=device, verbose=False) if first_half else []
            cand = [(k, f, r) for k, (f, r) in enumerate(zip(first_half, preds)) if r.boxes is not None and len(r.boxes)]
            if not cand:
                continue
            k0, f0, r0 = cand[0]
            box = r0.boxes.xyxy.cpu().numpy()[int(r0.boxes.conf.cpu().numpy().argmax())].tolist()
            ok = False
            if v.online:
                try:
                    ans = v._verify_fire({"frame": f0, "box": box})
                    checked += 1
                    ok = not ans["positive"]
                    verdicts.append({"camera": clip.stem, "rejected": ok, "reason": ans["reason"]})
                    say(f"learning loop: local AI on {clip.stem} candidate -> {'REJECTED' if ok else 'confirmed'}: "
                        f"{ans['reason'][:90]}")
                except Exception as e:
                    say(f"learning loop: AI error {e}")
            if ok:  # every rejected candidate frame becomes a 'not fire' training sample
                rejected += 1
                for k, f, _ in cand:
                    save("train", f"airej_{clip.stem}_{k:03d}", f, [])
                    added += 1
        try:
            import requests
            requests.post(cfg["verifier"]["ollama_url"].rstrip("/") + "/api/generate",
                          json={"model": cfg["verifier"]["model"], "keep_alive": 0}, timeout=30)
            time.sleep(3)
        except Exception:
            pass
        if added:
            say(f"learning loop: {rejected}/{checked} candidate spots rejected by the local AI, {added} 'not fire' "
                "samples added - retraining from the current model")
            best2, mins2, device = train(model_path, max(10, a.epochs // 3), "fire_sim_loop")
            loop_path = out / "fire_sim_loop.pt"
            shutil.copy2(best2, loop_path)
            r2, scen2, ex2, fpx2 = evaluate(loop_path, device, "after learning loop")
            rounds.append({"stage": "After one learning-loop retrain", **{k: v for k, v in r2.items() if k != "fp_cameras"},
                           "detected": sum(1 for r in scen2 if r["time_to_detect"] is not None), "scenarios": len(scen2)})
            loop = {"checked": checked, "rejected": rejected, "added": added, "verdicts": verdicts, "minutes": round(mins2, 1)}
            # keep the better model (fewer false alarms without losing detections) as the final one
            if r2["fp_per_cam_hour"]["model@0.25"] <= r1["fp_per_cam_hour"]["model@0.25"] and \
                    rounds[-1]["detected"] >= rounds[0]["detected"] - 1:
                shutil.copy2(loop_path, model_path)
                scen_rows, examples, fp_examples = scen2, ex2, fpx2
                rounds[-1]["final"] = True
            else:
                rounds[0]["final"] = True
    if not any(r.get("final") for r in rounds):
        rounds[0]["final"] = True

    final = next(r for r in rounds if r.get("final"))
    summary = {"clips": len(clips), "test_cams": sorted(test_cams), "dataset": counts, "epochs": a.epochs,
               "train_minutes": round(train_min, 1), "device": device, "hard_negatives_from_previous": bool(prev_model),
               "test": {"map50": final["map50"], "precision": final["precision"], "recall": final["recall"]},
               "scenarios": scen_rows, "ai": [{k: v for k, v in r.items() if k != "img"} for r in ai_rows],
               "false_candidates_per_cam_hour": final["fp_per_cam_hour"], "rounds": rounds, "learning_loop": loop,
               "model": str(model_path)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_report(summary, scen_rows, ai_rows, examples, fp_examples, out)
    say(f"REPORT {report}")


def write_report(s, scen_rows, ai_rows, examples, fp_examples, out: Path) -> Path:
    det = [r for r in scen_rows]
    detected = sum(1 for r in det if r["time_to_detect"] is not None)
    ttd = [r["time_to_detect"] for r in det if r["time_to_detect"] is not None]
    ai_ok = sum(1 for r in ai_rows if r["confirmed"])
    rows = "".join(f"<tr><td>{r['id']}</td><td>{r['kind']}</td><td>{r['size']}</td><td>{r['rate']:.0%}</td>"
                   f"<td>{'%.1f s' % r['time_to_detect'] if r['time_to_detect'] is not None else '<span style=color:#f85149>missed</span>'}</td></tr>"
                   for r in det)
    ai_html = "".join(f"<div class=card><img src='data:image/jpeg;base64,{r['img']}'><br><b style='color:"
                      f"{'#3fb950' if r['confirmed'] else '#f85149'}'>{'CONFIRMED' if r['confirmed'] else 'REJECTED'}"
                      f" ({r['confidence']:.2f})</b> {r['id']} - {r['kind']}, {r['size']}<br><small>{r['reason']}</small></div>"
                      for r in ai_rows)
    ex_html = "".join(f"<div class=card><img src='data:image/jpeg;base64,{b}'><br>{m['id']} - {m['kind']}, {m['size']}</div>"
                      for m, b in examples)
    fp_html = "".join(f"<div class=card><img src='data:image/jpeg;base64,{b}'><br>{c}</div>" for c, b in fp_examples)
    fp = s["false_candidates_per_cam_hour"]
    rounds_html = "".join(
        f"<tr><td>{r['stage']}{' <b>(final)</b>' if r.get('final') else ''}</td><td>{r['map50']:.2f}</td>"
        f"<td>{r['precision']:.2f}</td><td>{r['recall']:.2f}</td><td>{r['detected']}/{r['scenarios']}</td>"
        f"<td>{r['fp_per_cam_hour']['model@0.25']}</td><td>{r['fp_per_cam_hour']['model@0.50']}</td></tr>"
        for r in s.get("rounds", []))
    lp = s.get("learning_loop")
    loop_html = (f"Learning loop: the model's false candidates on the first half of each clip were checked by the local "
                 f"AI ({lp['rejected']} of {lp['checked']} spots rejected); {lp['added']} rejected frames became 'not fire' "
                 f"samples and the model was retrained in {lp['minutes']} min. False candidates are then measured on "
                 f"the unseen second half of the footage. This is the same loop the app runs continuously.") if lp else ""
    html = f"""<html><head><meta charset="utf-8"><style>
body{{font-family:Segoe UI;background:#0d1117;color:#e6edf3;margin:24px}} td,th{{padding:6px 10px;border:1px solid #30363d}}
table{{border-collapse:collapse}} .card{{display:inline-block;width:430px;margin:6px;vertical-align:top;color:#c9d1d9}}
.big{{font-size:28px;font-weight:700}} .kpi{{display:inline-block;background:#161b22;border:1px solid #30363d;
border-radius:8px;padding:10px 18px;margin:4px}} small{{color:#8b949e}}</style></head><body>
<h2>Simulated fire - trained fire detector</h2>
<p><small>{s['clips']} real camera clips, {len(scen_rows)} held-out scenarios on cameras never used in training
({', '.join(s['test_cams'])}). Trained {s['epochs']} epochs in {s['train_minutes']} min on {'GPU' if s['device'] != 'cpu' else 'CPU'}.
Training data: {s['dataset']['train'][0]} simulated-fire images and {s['dataset']['train'][1]} real images without fire.</small></p>
<div class=kpi>mAP50 (held-out)<div class=big>{s['test']['map50']:.2f}</div></div>
<div class=kpi>Precision<div class=big>{s['test']['precision']:.2f}</div></div>
<div class=kpi>Recall<div class=big>{s['test']['recall']:.2f}</div></div>
<div class=kpi>Scenarios detected<div class=big>{detected}/{len(det)}</div></div>
<div class=kpi>Median time to detect<div class=big>{(statistics.median(ttd) if ttd else 0):.1f}s</div></div>
<div class=kpi>Local AI confirmed<div class=big>{ai_ok}/{len(ai_rows)}</div></div>
<h3>How the model improved</h3>
<table><tr><th>Stage</th><th>mAP50</th><th>Precision</th><th>Recall</th><th>Held-out scenarios detected</th>
<th>False candidates / camera-hour (conf 0.25)</th><th>(conf 0.50)</th></tr>{rounds_html}</table>
<p><small>{loop_html}</small></p>
<h3>False fire candidates on your real footage (no fire), per camera-hour</h3>
<table><tr><th>Colour/flicker heuristic (what produces "fire? 0.78" today)</th><th>Trained model, conf 0.25</th>
<th>Trained model, conf 0.50</th></tr><tr><td>{fp['heuristic']}</td><td>{fp['model@0.25']}</td><td>{fp['model@0.50']}</td></tr></table>
<p><small>Every candidate costs a local-AI check. Fewer candidates = less AI load and fewer chances of a wrong alert.</small></p>
<h3>Held-out scenarios</h3><table><tr><th>Scenario</th><th>Type</th><th>Size</th><th>Frames detected</th><th>Time to detect</th></tr>{rows}</table>
<h3>Trained model on held-out cameras</h3>{ex_html}
<h3>Local AI verification of the simulated fires</h3>{ai_html or '<small>skipped</small>'}
<h3>Remaining false candidates (trained model, real footage)</h3>{fp_html or '<small>none</small>'}
<p><small>Simulated fire is not real fire. Use these numbers to check the pipeline; the model keeps learning from real
confirmed detections and your reviews once it runs in the app.</small></p></body></html>"""
    path = out / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
