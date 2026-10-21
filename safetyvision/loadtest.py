"""Load test: how many cameras can this PC analyse at the configured frame rate?

1. Records a short clip from every configured camera (reused on later runs).
2. Replays the clips as N virtual cameras (N = 16, 24, 32 ...) through the real engine - same models, same
   per-camera feature settings (a virtual copy of CH11 uses CH11's settings), optionally with the local AI.
3. Measures per-camera analysis rate, GPU / CPU / RAM, local-AI latency and queue, and writes a report.

    python -m safetyvision.loadtest --counts 16,24,32,48 --seconds 60
Monitoring must be stopped while it runs (it needs the GPU to itself).
"""
import argparse
import copy
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000")
import cv2  # noqa: E402

from . import config as C  # noqa: E402


def say(msg):
    print(f"[loadtest] {msg}", flush=True)


# ---------------------------------------------------------------- 1. clips
def record_clips(cfg, out_dir: Path, seconds=20, rerecord=False, parallel=2) -> list[dict]:
    from .xvr import _open_capture, fix_aspect, source_list
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [s for s in source_list(cfg) if C.camera_cfg(cfg, s["name"])["enabled"]]
    clips = []
    sem = threading.Semaphore(parallel)
    lock = threading.Lock()

    def rec(src):
        path = out_dir / f"{src['name']}.mp4"
        if path.exists() and path.stat().st_size > 100_000 and not rerecord:
            with lock:
                clips.append({"name": src["name"], "path": str(path)})
            return
        with sem:
            cap = _open_capture(src["url"])
            if not cap.isOpened():
                say(f"{src['name']}: could not open, skipped")
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 15
            fps = fps if 1 < fps <= 60 else 15
            writer, n, t0 = None, 0, time.time()
            while time.time() - t0 < seconds + 5 and n < seconds * fps:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = fix_aspect(frame)
                if writer is None:
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                n += 1
            cap.release()
            if writer:
                writer.release()
            say(f"{src['name']}: recorded {n} frames at {fps:.0f} fps")
            if n > fps * 3:
                with lock:
                    clips.append({"name": src["name"], "path": str(path)})

    threads = [threading.Thread(target=rec, args=(s,)) for s in sources]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    clips.sort(key=lambda c: c["name"])
    return clips


# ---------------------------------------------------------------- 2. one step
class _Sampler:
    def __init__(self):
        self.gpu = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_total = pynvml.nvmlDeviceGetMemoryInfo(self.gpu).total / 2**20
        except Exception:
            self.gpu_total = 0
        import psutil
        self.ps = psutil
        self.proc = psutil.Process()
        psutil.cpu_percent(None)

    def sample(self):
        out = {"cpu": self.ps.cpu_percent(None), "ram_mb": self.proc.memory_info().rss / 2**20}
        if self.gpu is not None:
            u = self.nv.nvmlDeviceGetUtilizationRates(self.gpu)
            m = self.nv.nvmlDeviceGetMemoryInfo(self.gpu)
            out.update(gpu=u.gpu, gpu_mem_mb=m.used / 2**20)
        return out


def run_step(cfg, clips, n, seconds, use_ai, warmup=20) -> dict:
    from PySide6.QtCore import QCoreApplication

    from .engine import Engine
    app = QCoreApplication.instance() or QCoreApplication([])
    c = copy.deepcopy(cfg)
    c["recorders"] = []
    c["extra_sources"] = []
    cams = {}
    for k in range(n):
        clip = clips[k % len(clips)]
        name = f"V{k + 1:02d}-{clip['name']}"
        c["extra_sources"].append({"name": name, "url": clip["path"], "recorder": "load test"})
        cams[name] = {**C.camera_cfg(cfg, clip["name"]), "enabled": True}
    c["cameras"] = cams
    c["verifier"]["enabled"] = use_ai
    c["learning"]["enabled"] = False  # measure detection, not dataset writes
    c["alerts"]["sound"] = False
    e = Engine(c)
    e.start()
    t0 = time.time()
    while not e.running and time.time() - t0 < 180:
        app.processEvents()
        time.sleep(0.2)
    say(f"{n} cameras: warming up {warmup}s")
    end = time.time() + warmup
    while time.time() < end:
        app.processEvents()
        time.sleep(0.2)
    sampler = _Sampler()
    samples, cam_fps, ai_q = [], [], []
    ai_done0 = sum(v for k, v in (e.verifier.stats if e.verifier else {}).items() if k != "errors")
    end = time.time() + seconds
    say(f"{n} cameras: measuring {seconds}s")
    while time.time() < end:
        t = time.time()
        while time.time() - t < 1.0:
            app.processEvents()
            time.sleep(0.05)
        s = sampler.sample()
        s["loop_fps"] = e.loop_fps
        samples.append(s)
        cam_fps.append([e.proc_fps.get(i, 0.0) for i in range(len(e.streams))])
        if e.verifier:
            ai_q.append(e.verifier.pending())
    st = e.stats()
    v = e.verifier
    ai_done = sum(val for k, val in (v.stats if v else {}).items() if k != "errors") - ai_done0
    ai = {"latency_s": round(v.avg_latency, 2) if v else None, "queue_avg": round(statistics.mean(ai_q), 1) if ai_q else None,
          "queue_max": max(ai_q) if ai_q else None, "checks_per_min": round(ai_done * 60 / seconds, 1) if v else None,
          "errors": v.stats.get("errors", 0) if v else None, "online": v.online if v else None}
    decode = [round(s.fps, 1) for s in e.streams]
    e.shutdown()
    per_cam = [statistics.mean(col) for col in zip(*cam_fps[len(cam_fps) // 3:])] if cam_fps else []
    target = float(cfg["detection"]["fps_per_camera"])
    res = {
        "cameras": n, "target_fps": target,
        "cam_fps_avg": round(statistics.mean(per_cam), 2) if per_cam else 0,
        "cam_fps_min": round(min(per_cam), 2) if per_cam else 0,
        "total_fps": round(statistics.mean(x["loop_fps"] for x in samples[len(samples) // 3:]), 1),
        "online": st["online"],
        "decode_fps_avg": round(statistics.mean(decode), 1) if decode else 0,
        "cpu_avg": round(statistics.mean(x["cpu"] for x in samples), 0),
        "ram_mb_max": round(max(x["ram_mb"] for x in samples)),
        "gpu_avg": round(statistics.mean(x.get("gpu", 0) for x in samples), 0),
        "gpu_mem_mb_max": round(max(x.get("gpu_mem_mb", 0) for x in samples)),
        "gpu_mem_total_mb": round(sampler.gpu_total), "ai": ai,
    }
    ok_rate = res["cam_fps_min"] >= 0.8 * target
    ok_ai = (not use_ai) or (ai["queue_max"] is not None and ai["queue_max"] < 12 and ai["online"])
    ok_mem = not sampler.gpu_total or res["gpu_mem_mb_max"] < 0.97 * sampler.gpu_total
    res["pass"] = bool(ok_rate and ok_ai and ok_mem)
    res["why"] = ", ".join(w for w, bad in (("cameras below 80% of target rate", not ok_rate),
                                             ("local AI can't keep up", not ok_ai),
                                             ("GPU memory full", not ok_mem)) if bad) or "ok"
    say(f"{n} cameras: {res['cam_fps_avg']} fps/cam (min {res['cam_fps_min']}), GPU {res['gpu_avg']}% "
        f"{res['gpu_mem_mb_max']} MB, CPU {res['cpu_avg']}%, AI q max {ai['queue_max']} -> "
        f"{'PASS' if res['pass'] else 'FAIL'} ({res['why']})")
    return res


# ---------------------------------------------------------------- 3. report
def write_report(results, meta, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    (out_dir / f"loadtest_{stamp}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2),
                                                    encoding="utf-8")
    passing = [r["cameras"] for r in results if r["pass"]]
    verdict = (f"This PC handles <b>{max(passing)} cameras</b> at {meta['target_fps']} analysed frames/s each"
               + (" with the local AI running." if meta["use_ai"] else " (local AI not included).")) if passing else \
        "None of the tested camera counts met the target - lower 'Analysis FPS per camera' or the camera count."
    rows = "".join(
        f"<tr><td>{r['cameras']}</td><td>{r['cam_fps_avg']} / {r['cam_fps_min']}</td><td>{r['total_fps']}</td>"
        f"<td>{r['gpu_avg']}%</td><td>{r['gpu_mem_mb_max']} / {r['gpu_mem_total_mb']} MB</td><td>{r['cpu_avg']}%</td>"
        f"<td>{r['ram_mb_max']} MB</td><td>{r['ai']['latency_s'] if r['ai']['latency_s'] is not None else '-'} s / "
        f"q{r['ai']['queue_max'] if r['ai']['queue_max'] is not None else '-'}</td>"
        f"<td style='color:{'#3fb950' if r['pass'] else '#f85149'}'><b>{'PASS' if r['pass'] else 'FAIL'}</b> "
        f"{'' if r['pass'] else r['why']}</td></tr>" for r in results)
    html = f"""<html><body style="font-family:Segoe UI;background:#0d1117;color:#e6edf3">
<h2>SafetyVision load test - {time.strftime('%d %b %Y %H:%M')}</h2>
<p>{verdict}</p>
<p style="color:#8b949e">Target {meta['target_fps']} frames/s per camera, {meta['seconds']} s per step, clips from
{meta['clips']} real cameras replayed as virtual cameras with their own feature settings. Device: {meta['device']}.
Local AI: {meta['ai_model'] if meta['use_ai'] else 'off'}.</p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;border-color:#30363d">
<tr style="background:#161b22"><th>Cameras</th><th>fps / camera (avg / min)</th><th>Total fps</th><th>GPU</th>
<th>GPU memory</th><th>CPU</th><th>RAM</th><th>Local AI latency / max queue</th><th>Result</th></tr>{rows}</table>
<p style="color:#8b949e">PASS = every camera reaches at least 80% of the target rate, the local AI queue stays short and
GPU memory is not full.</p></body></html>"""
    path = out_dir / f"loadtest_{stamp}.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="16,24,32,48")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--clip-seconds", type=int, default=20)
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--rerecord", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--config", default="", help="use another config.yaml (e.g. the installed app's)")
    a = ap.parse_args()
    cfg = C.load_config(Path(a.config)) if a.config else C.load_config()
    base = C.DATA_DIR / "loadtests"
    real_data = C.DATA_DIR
    C.DATA_DIR = base / "scratch"  # the engine's events / datasets during the test stay out of real data
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if (real_data / "ignore.json").exists():  # use the site's learned 'not fire' / 'not a person' spots
        import shutil
        shutil.copy2(real_data / "ignore.json", C.DATA_DIR / "ignore.json")
    say("recording clips from the recorders ...")
    clips = record_clips(cfg, base / "clips", a.clip_seconds, a.rerecord)
    if not clips:
        say("no clips - check the recorder settings")
        sys.exit(1)
    results = []
    for n in [int(x) for x in a.counts.split(",") if x.strip()]:
        r = run_step(cfg, clips, n, a.seconds, not a.no_ai)
        results.append(r)
        if a.stop_on_fail and not r["pass"]:
            break
    from .detectors import select_device
    meta = {"target_fps": cfg["detection"]["fps_per_camera"], "seconds": a.seconds, "clips": len(clips),
            "use_ai": not a.no_ai, "ai_model": cfg["verifier"]["model"], "device": select_device(cfg["detection"]["device"])}
    path = write_report(results, meta, real_data / "loadtests")
    say(f"REPORT {path}")


if __name__ == "__main__":
    main()
