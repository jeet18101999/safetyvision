"""Learning hub: one dataset + one self-improving model per task.

  fire          YOLO detector (fire / smoke)
  person        classifier: person / not_person        (static look-alikes: mannequins, posters, coats)
  posture       classifier: down / normal              (man-down verification)
  ppe_<item>    classifier: wearing / missing          (one per PPE item: helmet, vest, gloves, ...)

Labels come from the local AI and from operator corrections. When a task has enough new samples it is
fine-tuned in a background process (one task at a time) and deployed only if it scores at least as well
as the current model on the same validation set.
"""
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config as C
from .dataset import ClsDataset, FireDataset

log = logging.getLogger(__name__)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def task_classes(task) -> list[str]:
    if task == "person":
        return ["not_person", "person"]
    if task == "posture":
        return ["down", "normal"]
    if task.startswith("ppe_"):
        return ["missing", "wearing"]
    return ["fire", "smoke"]


def task_title(task) -> str:
    if task == "fire":
        return "Fire & smoke detector"
    if task == "person":
        return "Person verifier"
    if task == "posture":
        return "Man-down verifier"
    return "PPE: " + C.PPE_LABELS.get(task[4:], task[4:])


def all_tasks() -> list[str]:
    return ["fire", "person", "posture"] + [f"ppe_{i}" for i in C.PPE_ITEMS]


def model_dir(task) -> Path:
    return C.MODELS_DIR / task


def current_model(task, cfg=None) -> Path:
    if task == "fire" and cfg is not None:
        return C.resolve(cfg["detection"]["fire_model"])
    return model_dir(task) / "current.pt"


def load_history(task="fire") -> list[dict]:
    p = model_dir(task) / "history.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def model_version(task, cfg=None) -> str | None:
    cur = current_model(task, cfg)
    if not cur.exists():
        return None
    v = cur.with_suffix(".version")
    return v.read_text(encoding="utf-8").strip() if v.exists() else "custom"


class LearningHub:
    def __init__(self, cfg: dict, get_device, on_status, on_model_updated):
        self.cfg = cfg
        self._get_device = get_device
        self.on_status = on_status  # (task, text)
        self.on_model_updated = on_model_updated  # (task, path)
        lc = cfg["learning"]
        self.fire = FireDataset(C.DATA_DIR / "fire_dataset", lc["val_fraction"], lc["max_negative_ratio"],
                                C.ROOT / "assets" / "fire_anchor")
        self._cls: dict[str, ClsDataset] = {}
        self._queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._queued: set[str] = set()
        self.running_task = None
        self.progress = ""
        self._proc = None
        self._stop = False
        threading.Thread(target=self._worker, name="trainer", daemon=True).start()

    # ---------------------------------------------------------------- datasets
    def dataset(self, task):
        if task == "fire":
            return self.fire
        if task not in self._cls:
            self._cls[task] = ClsDataset(C.DATA_DIR / "cls" / task, task_classes(task),
                                         self.cfg["learning"]["val_fraction"])
        return self._cls[task]

    def add(self, task, crop, label, meta=None, weight=1):
        """Add a classification sample (not fire) and maybe trigger training."""
        if not self.cfg["learning"]["enabled"]:
            return None
        rec = self.dataset(task).add(crop, label, meta or {}, weight)
        if rec:
            self.maybe_train(task)
        return rec

    def history(self, task):
        return load_history(task)

    # ---------------------------------------------------------------- training queue
    def threshold(self, task) -> int:
        lc = self.cfg["learning"]
        return int(lc["min_new_samples"] if task == "fire" else lc["min_new_samples_cls"])

    def maybe_train(self, task):
        lc = self.cfg["learning"]
        if not (lc.get("enabled") and lc.get("auto_train")):
            return
        ds = self.dataset(task)
        if ds.new_samples() >= self.threshold(task) and ds.ready_for_training()[0]:
            self.start(task, "auto")

    def start(self, task, reason="manual") -> tuple[bool, str]:
        if task in self._queued or task == self.running_task:
            return False, f"{task_title(task)}: training already queued"
        ok, msg = self.dataset(task).ready_for_training()
        if not ok:
            return False, f"{task_title(task)} {msg}"
        self._queued.add(task)
        self._queue.put((task, reason))
        return True, f"{task_title(task)}: training queued"

    def stop(self):
        self._stop = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _status(self, task, text):
        self.progress = f"{task_title(task)}: {text}"
        try:
            self.on_status(task, text)
        except Exception:
            pass

    def _worker(self):
        while not self._stop:
            task, reason = self._queue.get()
            self._queued.discard(task)
            self.running_task = task
            try:
                self._run(task, reason)
            except Exception as e:
                log.exception("training %s failed", task)
                self._status(task, f"Training error: {e}")
            finally:
                self.running_task = None
                self._proc = None

    def _run(self, task, reason):
        lc = self.cfg["learning"]
        ds = self.dataset(task)
        detect = task == "fire"
        samples_at_start = ds.new_samples()
        mdir = model_dir(task)
        runs_dir = mdir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (mdir / "versions").mkdir(parents=True, exist_ok=True)
        current = current_model(task, self.cfg)
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        out_json = runs_dir / f"{run_name}.json"
        base = C.resolve(lc["base_model"] if detect else lc["cls_base_model"])
        if not base.exists():
            from .detectors import ensure_weights
            ensure_weights(base)
        start_weights = current if current.exists() else base
        device = self._get_device()
        if device.startswith("cuda") and not detect:
            device = "cpu"  # small 224px classifiers train in minutes on CPU and never compete with live detection
        elif device.startswith("cuda"):
            try:  # the local AI + live detection share the GPU; never push it into (very slow) shared memory
                import torch
                free_mb = torch.cuda.mem_get_info()[0] / 2**20
                if free_mb < float(lc.get("train_min_free_gpu_mb", 2500)):
                    device = "cpu"
                    self._status(task, f"GPU busy ({free_mb:.0f} MB free) - training on CPU instead")
            except Exception:
                pass
        extra, prep = [], {}
        if detect:
            prep = ds.prepare_training()
            extra = ["--neg-list", prep["neg_list"], "--conf", str(self.cfg["detection"].get("fire_conf", 0.25))]
            self._status(task, f"training set: {ds.counts()['by_class']['not fire']} 'not fire' samples "
                               f"(+{prep['neg_copies']} extra copies), {prep['anchors']['train']} reference fire images")
        cmd = [sys.executable, "-m", "safetyvision.train_job", "--task", "detect" if detect else "classify",
               "--data", prep["data"] if detect else ds.train_data, "--weights", str(start_weights),
               "--current", str(current) if current.exists() else "",
               "--epochs", str((lc["epochs"] if not device.startswith("cpu") else min(lc["epochs"], int(lc.get("cpu_fire_epochs", 8))))
                               if detect else lc["cls_epochs"]),
               "--batch", str(lc["batch"] if detect else 32), "--imgsz", str(lc["imgsz"] if detect else 224),
               "--device", "0" if device.startswith("cuda") else "cpu",
               "--project", str(runs_dir), "--name", run_name, "--out", str(out_json)] + extra
        log_path = runs_dir / f"{run_name}.log"
        self._status(task, f"training started ({reason}) from {start_weights.name}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(C.ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
        exe = sys.executable
        if exe.lower().endswith("pythonw.exe"):  # training needs a console-capable interpreter for its output
            exe = str(Path(exe).with_name("python.exe"))
        cmd[0] = exe
        watchdog_s = float(lc.get("train_watchdog_minutes", 15)) * 60
        with open(log_path, "w", encoding="utf-8") as logf:
            self._proc = subprocess.Popen(cmd, cwd=runs_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace", env=env, creationflags=flags)
            last_output = [time.time()]
            proc = self._proc
            stalled = [False]

            def watchdog():  # a run that prints nothing for too long is stuck (e.g. GPU memory exhausted)
                while proc.poll() is None:
                    if time.time() - last_output[0] > watchdog_s:
                        stalled[0] = True
                        proc.kill()
                        return
                    time.sleep(5)

            threading.Thread(target=watchdog, daemon=True).start()
            for line in self._proc.stdout:  # text mode turns tqdm '\r' updates into lines
                last_output[0] = time.time()
                line = ANSI.sub("", line).strip()
                if not line:
                    continue
                logf.write(line + "\n")
                m = re.match(r"(\d+)/(\d+)\s", line)
                if m and "100%" in line and "Class" not in line:
                    self._status(task, f"epoch {m.group(1)}/{m.group(2)} done")
                elif line.startswith("[SafetyVision]"):
                    self._status(task, line[len("[SafetyVision]"):].strip())
            code = self._proc.wait()
        if stalled[0]:
            self._status(task, f"training failed: no progress for {watchdog_s / 60:.0f} min, stopped. Will retry later.")
            return
        if code != 0 or not out_json.exists():
            self._status(task, f"training failed (exit {code}). See {log_path}")
            return
        res = json.loads(out_json.read_text(encoding="utf-8"))
        new, old, metric = res["new"], res["old"], res["metric"]
        min_score = float(lc["min_map50_to_deploy"] if detect else lc["min_cls_acc_to_deploy"])
        fa_new, fa_old = new.get("false_alarms"), (old or {}).get("false_alarms")
        if new["score"] < min_score:
            accept, why = False, f"{metric} {new['score']:.3f} is below the deploy minimum {min_score:.2f}"
        elif detect and fa_new is not None and fa_old is not None:
            # learning 'not fire': fewer false alarms on the site's rejected images, while still finding the
            # reference fires (a small mAP dip is accepted for clearly fewer false alarms, never a big one)
            tol = float(lc.get("fire_map_tolerance", 0.05))
            if new["score"] < old["score"] - tol:
                accept, why = False, f"{metric} {new['score']:.3f} dropped too much (current {old['score']:.3f})"
            elif fa_new > fa_old:
                accept, why = False, f"more false alarms on 'not fire' images ({fa_new} vs {fa_old})"
            elif fa_new == fa_old and new["score"] < old["score"]:
                accept, why = False, f"no fewer false alarms ({fa_new}) and {metric} {new['score']:.3f} < {old['score']:.3f}"
            else:
                accept, why = True, ""
        elif old is not None and lc.get("require_improvement", True) and new["score"] < old["score"]:
            accept, why = False, f"{metric} {new['score']:.3f} < current {old['score']:.3f}"
        else:
            accept, why = True, ""
        hist = load_history(task)
        version = f"{task}_v{len(hist) + 1:03d}"
        entry = {"version": version, "time": time.strftime("%Y-%m-%d %H:%M"), "reason": reason,
                 "samples": ds.counts()["total"], "metric": metric, "new": new, "old": old, "accepted": accept}
        if accept:
            dst = mdir / "versions" / f"{version}.pt"
            shutil.copy2(res["best"], dst)
            current.parent.mkdir(parents=True, exist_ok=True)
            tmp = current.with_suffix(".tmp")
            shutil.copy2(dst, tmp)
            tmp.replace(current)
            current.with_suffix(".version").write_text(version, encoding="utf-8")
        (mdir / "history.json").write_text(json.dumps(hist + [entry], indent=2), encoding="utf-8")
        ds.mark_trained(samples_at_start)
        if accept:
            self._status(task, f"deployed {version}: {metric} {new['score']:.3f}"
                         + (f" (was {old['score']:.3f})" if old else "")
                         + (f", false alarms on 'not fire' images {fa_old} -> {fa_new} of {new.get('neg_images')}"
                            if fa_new is not None and fa_old is not None else ""))
            self.on_model_updated(task, str(current))
        else:
            self._status(task, f"kept current model: {why}. More samples will improve the next run.")
