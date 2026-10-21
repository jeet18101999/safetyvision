"""Engine: camera streams -> batched detection -> per-camera analytics -> local-AI / learned verification
-> alerts -> learning (from the local AI and from operator review)."""
import logging
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from . import config as C
from .analytics import CameraAnalytics, box_iou, feet_point, head_point, in_polygon, is_static
from .events import EventStore, day_start
from .memory import IgnoreStore, NotFireMemory
from .trainer import LearningHub, all_tasks, current_model, task_title
from .verifier import LocalAIVerifier, crop_box
from .xvr import CameraStream, source_list

log = logging.getLogger(__name__)

BLUE = (250, 160, 50)
GREEN, ORANGE, RED, YELLOW, GREY, WHITE, CYAN = ((60, 200, 60), (0, 140, 255), (40, 40, 235), (0, 215, 255),
                                                 (150, 150, 150), (255, 255, 255), (230, 200, 40))
EVENT_LABELS = {"FIRE": "Fire", "SMOKE": "Smoke", "FIRE_UNVERIFIED": "Fire (unverified)", "MAN_DOWN": "Man down",
                "PPE": "PPE violation", "INTRUSION": "Restricted zone", "OVERCROWD": "Crowd limit",
                "CAMERA_OFFLINE": "Camera offline", "SYSTEM": "System"}
CRITICAL = {"FIRE", "SMOKE", "FIRE_UNVERIFIED", "MAN_DOWN"}
FIRE_TYPES = {"FIRE", "SMOKE", "FIRE_UNVERIFIED"}


def needs_person(cc) -> bool:
    return any(cc.get(k) for k in ("people", "man_down", "ppe", "zone", "crowd"))


class Engine(QObject):
    event_raised = Signal(dict)
    message = Signal(str)
    training_status = Signal(str, str)  # task, text
    model_updated = Signal(str)  # task
    started = Signal()
    stopped = Signal()
    fatal = Signal(str)  # detection cannot continue in this process (e.g. broken CUDA context) -> app restarts

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.streams: list[CameraStream] = []
        self.running = False
        self.loading = False
        self.device = None
        self.pose = self.fire = None
        self.cls = {}  # task -> ClsModel
        self._model_cache = {}
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.events = EventStore(C.DATA_DIR / "events.db", C.DATA_DIR / "snapshots")
        self.ignore = IgnoreStore(C.DATA_DIR)
        merged = self.ignore.compact()
        if merged:
            log.info("merged %d duplicate ignore entries", merged)
        self.hub = LearningHub(cfg, self.get_device, self.training_status.emit, self._on_model_updated)
        self.notfire = NotFireMemory()  # learned 'not fire' look-alikes (colour match per camera)
        self._notfire_audit: dict[int, int] = {}
        self.hub.fire.on_change = self._reload_notfire
        self._reload_notfire()
        self.verifier: LocalAIVerifier | None = None
        self.identity = None  # people.IdentityEngine (body re-identification)
        self._lock = threading.Lock()
        self._thread = None
        self._gen = 0
        self.ai_enabled = True  # False = view-only: streams keep running, no detection / alerts
        self._visible: set[int] = set()
        self.view_streams: dict[int, CameraStream] = {}  # sub streams for small grid tiles
        self._view_used: dict[int, float] = {}
        self._sub_active: set[int] = set()
        self._display_fps = float(cfg["ui"].get("display_fps", 25))
        self._reset_state()

    def _reset_state(self):
        self.analytics: list[CameraAnalytics] = []
        self.overlays = {}
        self.cam_stats = {}
        self._fire_confirmed = {}
        self._verify_pending = {}
        self._verify_next = {}
        self._rejected = {}  # cam -> [[box, until, count]]
        self._fire_inflight = 0
        self._last_alert = {}
        self._flash_until = {}
        self._offline_since = {}
        self._offline_alerted = set()
        self.proc_fps = {}
        self.loop_fps = 0.0

    def get_device(self) -> str:
        if self.device is None:
            from .detectors import select_device
            self.device = select_device(self.cfg["detection"].get("device", "auto"))
        return self.device

    def cam_cfg(self, i) -> dict:
        return C.camera_cfg(self.cfg, self.streams[i].name)

    def cam_label(self, i) -> str:
        return self.cam_cfg(i).get("label") or self.streams[i].name

    def cam_location(self, i) -> str:
        return self.cam_cfg(i).get("place_path") or ""

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self.running or self.loading:
            return
        self.loading = True
        self._reset_state()
        self.streams = []
        for i, src in enumerate(source_list(self.cfg)):
            s = CameraStream(i, src["name"], src["url"])
            s.recorder = src["recorder"]
            s.sub_url = src.get("sub_url")
            self.streams.append(s)
        self.analytics = [CameraAnalytics(self.cfg["detection"]) for _ in self.streams]
        for i, s in enumerate(self.streams):
            if self.cam_cfg(i)["enabled"]:
                s.start()
            else:
                s.enabled = False
        self._gen += 1
        self._thread = threading.Thread(target=self._boot_and_loop, args=(self._gen,), name="inference", daemon=True)
        self._thread.start()

    def stop(self):
        was = self.running or self.loading
        self.running = False
        self.loading = False
        for s in self.streams:
            s.stop()
        for vs in self.view_streams.values():
            vs.stop()
        self.view_streams.clear()
        self._view_used.clear()
        self._sub_active = set()
        if self.verifier:
            self.verifier.stop()
            self.verifier = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if was:
            self.stopped.emit()

    def shutdown(self):
        self.stop()
        self.hub.stop()

    def apply_config(self, cfg: dict):
        """Full restart (connection / model settings changed)."""
        was_running = self.running or self.loading
        self.stop()
        self.update_config(cfg)
        if was_running:
            self.start()

    def update_config(self, cfg: dict):
        """Live update (per-camera features, thresholds). Streams are started/stopped to match 'enabled'."""
        self.cfg = cfg
        self.hub.cfg = cfg
        if self.identity:
            self.identity.cfg = cfg
        for a in self.analytics:
            a.cfg = cfg["detection"]
        if self.verifier:
            self.verifier.cfg = cfg["verifier"]
        for i, s in enumerate(list(self.streams)):
            want = self.cam_cfg(i)["enabled"]
            if want and not s.enabled:
                ns = CameraStream(i, s.name, s.source)
                ns.recorder = getattr(s, "recorder", "")
                ns.sub_url = getattr(s, "sub_url", None)
                ns.start()
                self.streams[i] = ns
                self.analytics[i] = CameraAnalytics(cfg["detection"])
            elif not want and s.enabled:
                s.stop()
                s.enabled = False
                vs = self.view_streams.pop(i, None)
                if vs:
                    vs.stop()
                with self._lock:
                    self.overlays.pop(i, None)
                    self.cam_stats.pop(i, None)

    def _load_model(self, kind, factory, path):
        key = (kind, str(path), self.device)
        if key not in self._model_cache:
            self.message.emit(f"Loading {kind} model {Path(path).name} ...")
            self._model_cache[key] = factory()
        return self._model_cache[key]

    def _boot_and_loop(self, gen):
        try:
            from .detectors import ClsModel, FireDetector, PoseDetector, ensure_weights
            dc = self.cfg["detection"]
            self.get_device()
            self.message.emit(f"Device: {self.device}")
            pp = C.resolve(dc["person_model"])
            self.pose = self._load_model("person", lambda: PoseDetector(pp, self.device, dc["imgsz"], int(dc.get("batch", 4))), pp)
            self.fire = FireDetector(C.resolve(dc["fire_model"]), self.device, dc["imgsz"], dc.get("fire_mode", "hybrid"),
                                     int(dc.get("batch", 4)))
            for task in all_tasks():
                if task != "fire" and current_model(task).exists():
                    try:
                        self.cls[task] = ClsModel(current_model(task), self.device)
                    except Exception as e:
                        log.warning("could not load %s model: %s", task, e)
            if self.cfg["verifier"]["enabled"]:
                self.verifier = LocalAIVerifier(self.cfg["verifier"])
                self.verifier.start()
            if dc.get("identify_people", True) and self.identity is None:
                try:
                    from .people import IdentityEngine
                    self.message.emit("Loading person recognition model ...")
                    self.identity = IdentityEngine(self.cfg, self.device, lambda: self.verifier)
                except Exception as e:
                    log.exception("person recognition unavailable")
                    self.message.emit(f"Person recognition unavailable: {e}")
            for base in (self.cfg["learning"]["base_model"], self.cfg["learning"]["cls_base_model"]):
                if not C.resolve(base).exists():
                    threading.Thread(target=ensure_weights, args=(C.resolve(base),), daemon=True).start()
        except Exception as e:
            log.exception("startup failed")
            self.message.emit(f"Startup failed: {e}")
            self.loading = False
            self.stopped.emit()
            return
        if not self.loading or gen != self._gen:  # stopped while loading
            return
        self.loading = False
        self.running = True
        self.started.emit()
        self.message.emit(f"Monitoring {sum(1 for s in self.streams if s.enabled)} camera(s)")
        log.info("monitoring %d camera(s) on %s | fire model %s | learned models %s | local AI %s",
                 sum(1 for s in self.streams if s.enabled), self.device, self.fire.version if self.fire else "-",
                 sorted(self.cls) or "none", self.cfg["verifier"]["model"] if self.verifier else "off")
        self._loop(gen)

    def _on_model_updated(self, task, path):
        try:
            if task == "fire":
                if self.fire is not None:
                    self.fire.load(path)
            else:
                from .detectors import ClsModel
                self.cls[task] = ClsModel(path, self.get_device())
        except Exception:
            log.exception("hot reload of %s failed", task)
        self.model_updated.emit(task)

    # ------------------------------------------------------------------ main loop
    def _loop(self, gen):
        n = len(self.streams)
        last_proc, last_fid = [0.0] * n, [-1] * n
        counters = [[0, time.time()] for _ in range(n)]
        next_occ, next_health, next_purge = time.time() + 60, time.time() + 5, time.time() + 30
        loop_n, loop_t0 = 0, time.time()
        failures = 0
        while self.running and gen == self._gen:
            period = 1.0 / max(0.2, float(self.cfg["detection"]["fps_per_camera"]))
            now = time.time()
            due = []
            if not self.ai_enabled:  # view-only: keep the loop alive for health checks, skip analysis
                time.sleep(0.2)
                if now >= next_health:
                    next_health = now + 5
                    self._check_cameras(now)
                continue
            for i, s in enumerate(self.streams):
                if not s.enabled or now - last_proc[i] < period:
                    continue
                frame, fid, _ = s.read()
                if frame is None or fid == last_fid[i]:
                    continue
                last_fid[i], last_proc[i] = fid, now
                due.append((i, frame))
            if due:
                try:
                    self._process_batch(due)
                    failures = 0
                except Exception as ex:
                    log.exception("processing failed")
                    failures += 1
                    msg = f"{type(ex).__name__}: {ex}"
                    if "CUDA" in msg or "AcceleratorError" in msg or failures >= 20:
                        # a broken GPU context never recovers inside this process: hand over to a fresh one
                        self.running = False
                        self.fatal.emit(msg[:300])
                        return
                for i, _ in due:
                    c = counters[i]
                    c[0] += 1
                    if now - c[1] >= 3:
                        self.proc_fps[i] = c[0] / (now - c[1])
                        c[0], c[1] = 0, now
                loop_n += len(due)
                if now - loop_t0 >= 3:
                    self.loop_fps, loop_n, loop_t0 = loop_n / (now - loop_t0), 0, now
            else:
                time.sleep(0.01)
            if now >= next_health:
                next_health = now + 5
                self._check_cameras(now)
                self._check_gpu()
            if now >= next_occ:
                if int(now // 60) % 15 == 0:  # heartbeat in the log every ~15 min
                    st = self.stats()
                    log.info("heartbeat: %d/%d cameras online, %.1f frames/s analysed, %d people, AI queue %s",
                             st["online"], st["enabled"], st["analysis_fps"], st["total_people"],
                             st["verifier"]["pending"])
                next_occ = now + 60
                with self._lock:
                    counts = {self.streams[i].name: st["people"] for i, st in self.cam_stats.items()
                              if st.get("people") is not None and self.streams[i].status == "online"}
                if counts:
                    self.events.log_occupancy(counts)
            if now >= next_purge:
                next_purge = now + 86400
                threading.Thread(target=self.events.purge, args=(self.cfg["alerts"].get("retention_days", 60),),
                                 daemon=True).start()
                if self.identity is not None:
                    threading.Thread(target=self.identity.gallery.purge,
                                     args=(self.cfg["detection"].get("identity_retention_days", 30),), daemon=True).start()

    def _process_batch(self, due):
        now = time.time()
        dc = self.cfg["detection"]
        ccs = {i: self.cam_cfg(i) for i, _ in due}
        people_res, fire_res = {}, {}
        pidx = [k for k, (i, _) in enumerate(due) if dc.get("person_detection", True) and needs_person(ccs[i])]
        if pidx and self.pose:
            res = self.pose.detect_batch([due[k][1] for k in pidx], dc["person_conf"])
            people_res = {due[k][0]: r for k, r in zip(pidx, res)}
        fidx = [k for k, (i, _) in enumerate(due) if ccs[i]["fire"]]
        if fidx and self.fire:
            res = self.fire.detect_batch([due[k][0] for k in fidx], [due[k][1] for k in fidx], dc["fire_conf"])
            fire_res = {due[k][0]: r for k, r in zip(fidx, res)}
        for i, frame in due:
            self._process(i, frame, ccs[i], people_res.get(i), fire_res.get(i), now)
        if self.identity is not None and self.cfg["detection"].get("identify_people", True):
            try:
                self.identity.flush(now)
            except Exception:
                log.exception("identity flush failed")

    def _process(self, i, frame, cc, people, fire_cands, now):
        name = self.streams[i].name
        an = self.analytics[i]
        valid = []
        if people is not None:
            an.update_people(people, now, frame.shape)
            for p in people:
                t = p["track"]
                if t.person_state == "ignored" and t.max_travel > 0.5:
                    t.person_state = None  # it walked away: it is a person after all
                if t.person_state == "ignored":
                    p["ignored"] = True
                elif (t.person_state != "verified" and t.max_travel < 0.3
                      and self.ignore.match("person", name, p["box"], frame.shape)):
                    p["ignored"] = True  # learned: something at this spot is not a person
                    t.person_state = "ignored"
            self._verify_static_people(i, frame, people, now)
            valid = [p for p in people if not p.get("ignored")]
            if self.identity is not None and self.cfg["detection"].get("identify_people", True):
                try:
                    self.identity.observe(name, frame, valid, now)
                except Exception:
                    log.exception("identity observe failed")
            self._auto_person_samples(i, frame, valid)
            if cc["man_down"]:
                for p in valid:
                    if p["track"].down and p["track"].down_state is None:
                        self._handle_man_down(i, frame, p)
            if cc["ppe"] and cc["ppe_items"]:
                self._ppe(i, frame, valid, [x for x in cc["ppe_items"] if x in C.PPE_ITEMS], now)
            if cc["zone"] and len(cc["zone_points"]) >= 3:
                for p in valid:
                    t = p["track"]
                    t.in_zone = in_polygon(feet_point(p), cc["zone_points"], frame.shape)
                    if t.in_zone and not t.zone_alerted:
                        t.zone_alerted = True
                        self._raise(i, "INTRUSION", f"{self._who(i, t)} entered the restricted zone", frame,
                                    [(p["box"], RED, "ZONE")], extra={"box": p["box"], "track": t.id},
                                    key_suffix=str(t.id))
                    elif not t.in_zone:
                        t.zone_alerted = False
            if cc["crowd"] and cc["max_people"] > 0 and len(valid) > cc["max_people"]:
                self._raise(i, "OVERCROWD", f"{len(valid)} people (limit {cc['max_people']})", frame,
                            [(p["box"], ORANGE, "") for p in valid], cooldown=120)

        if fire_cands is not None:
            self._handle_fire_candidates(i, frame, fire_cands, now)

        # overlay (drawn at display time so box / head-pointer toggles apply instantly)
        pov = []
        for p in people or []:
            t = p["track"]
            who = p.get("identity_label") or f"#{t.id}"
            if p.get("ignored"):
                pov.append((p["box"], head_point(p), GREY, "not a person", 1))
                continue
            if cc["man_down"] and t.down_state == "confirmed":
                col, lab, th = RED, f"{who} MAN DOWN", 3
            elif cc["man_down"] and (t.down_state == "pending" or (p["fallen"] and t.down_state != "rejected")):
                col, lab, th = YELLOW, f"{who} down?", 2
            elif cc["zone"] and t.in_zone:
                col, lab, th = RED, f"{who} IN ZONE", 2
            elif cc["ppe"] and t.ppe_missing:
                col, lab, th = ORANGE, f"{who} NO {'/'.join(C.PPE_LABELS[m].upper() for m in t.ppe_missing)}", 2
            elif t.person_state == "pending":
                col, lab, th = CYAN, f"{who} checking", 1
            else:
                named = bool(p.get("identity") and self.identity and self.identity.gallery.people.get(p["identity"], {}).get("name"))
                col, lab, th = (BLUE if named else GREEN), who, 2
            pov.append((p["box"], head_point(p), col, lab, th))
        fov = []
        if fire_cands:
            for c in fire_cands[:3]:
                if not self._fire_suppressed(i, name, c, frame.shape, now):
                    fov.append((c["box"], YELLOW, f"{c['label']}? {c['conf']:.2f}", 1))
        for box, label, until in self._fire_confirmed.get(i, []):
            if until > now:
                fov.append((box, RED, label.upper(), 3))
        with self._lock:
            self.overlays[i] = {"ts": now, "shape": frame.shape[:2], "people": pov, "fire": fov,
                                "zone": cc["zone_points"] if cc["zone"] else None}
            self.cam_stats[i] = {"people": len(valid) if cc["people"] else None,
                                 "pids": [p.get("identity") for p in valid] if cc["people"] else [],
                                 "down": sum(1 for p in valid if p["track"].down_state == "confirmed"),
                                 "ppe": sum(1 for p in valid if p["track"].ppe_missing)}

    # ------------------------------------------------------------------ person verification (static look-alikes)
    def _who(self, i, t) -> str:
        if self.identity is not None:
            label = self.identity.label_for(self.streams[i].name, t.id)
            if label:
                return label
        return f"Person #{t.id}"

    def _trust(self):
        return float(self.cfg["learning"]["cls_trust_conf"])

    def _classify(self, task, crop):
        m = self.cls.get(task)
        if m is None or crop is None or crop.size == 0:
            return None, 0.0
        try:
            return m.predict([crop])[0]
        except Exception:
            log.exception("%s classifier failed", task)
            return None, 0.0

    def _verify_static_people(self, i, frame, people, now):
        dc = self.cfg["detection"]
        if not dc.get("person_verify", True):
            return
        name = self.streams[i].name
        for p in people:
            t = p["track"]
            if t.person_state in ("verified", "ignored", "pending") or now < t.person_retry:
                continue
            if t.max_travel > 0.3:
                t.person_state = "verified"  # it walked around: a real person
                continue
            if not is_static(t, now, dc["static_seconds"], dc["static_motion"]):
                continue
            crop = crop_box(frame, p["box"], pad=0.15)
            label, conf = self._classify("person", crop)
            if label and conf >= self._trust():
                self._set_person(i, frame, p["box"], t, label == "person", "model")
                continue
            v = self.verifier
            if not (v and v.online):
                t.person_retry = now + 60
                continue
            t.person_state = "pending"
            snap, box = frame.copy(), list(p["box"])

            def done(res, t=t, snap=snap, box=box, crop=crop):
                if not res.get("ok"):
                    t.person_state, t.person_retry = None, time.time() + 120
                    return
                self.hub.add("person", crop, "person" if res["person"] else "not_person",
                             {"camera": name, "source": "ai", "ai_conf": res["confidence"], "ai_reason": res["reason"]})
                self._set_person(i, snap, box, t, res["person"], "ai", res["reason"])

            if not v.submit_person(snap, box, {"camera": name}, done):
                t.person_state = None

    def _set_person(self, i, frame, box, t, is_person, source, note=""):
        t.person_state = "verified" if is_person else "ignored"
        if not is_person:
            self.ignore.add("person", self.streams[i].name, box, frame.shape, source, frame, note)
            self.message.emit(f"{self.cam_label(i)}: static object is not a person ({source}) - it will be ignored")

    def _auto_person_samples(self, i, frame, people):
        """Walking tracks are certainly people: occasionally keep one as a free 'person' example. Kept in balance
        with 'not a person' examples so the person verifier does not learn that everything is a person."""
        if not self.cfg["learning"]["enabled"] or not people:
            return
        by = self.hub.dataset("person").counts()["by_class"]
        cap = min(int(self.cfg["learning"].get("max_auto_person_samples", 150)), 3 * by.get("not_person", 0) + 40)
        if by.get("person", 0) >= cap:
            return
        for p in people:
            t = p["track"]
            if t.sampled_person or t.max_travel < 0.8 or p["box"][3] - p["box"][1] < 80 or random.random() > 0.1:
                continue
            t.sampled_person = True
            self.hub.add("person", crop_box(frame, p["box"], pad=0.15), "person",
                         {"camera": self.streams[i].name, "source": "motion"})

    def _ignore_live_tracks(self, camera, box, shape):
        for i, s in enumerate(self.streams):
            if s.name != camera or i >= len(self.analytics):
                continue
            h, w = shape[:2]
            for t in self.analytics[i].tracker.tracks.values():
                if box_iou(t.box, box) > 0.4:
                    t.person_state = "ignored"

    # ------------------------------------------------------------------ man down
    def _handle_man_down(self, i, frame, p):
        t = p["track"]
        t.down_state = "pending"
        name = self.streams[i].name
        snap, box = frame.copy(), list(p["box"])
        crop = crop_box(frame, box, pad=0.2)
        label, conf = self._classify("posture", crop)
        if label and conf >= self._trust():
            if label == "down":
                t.down_state = "confirmed"
                self._raise_man_down(i, snap, box, t, True, f"learned model {conf:.0%}")
            else:
                t.down_state = "rejected"
            return
        v = self.verifier
        if not (self.cfg["detection"].get("man_down_verify", True) and v and v.online):
            t.down_state = "confirmed"
            self._raise_man_down(i, snap, box, t, None, "")
            return

        def done(res):
            if not res.get("ok"):
                t.down_state = "confirmed"
                self._raise_man_down(i, snap, box, t, None, "local AI unavailable")
                return
            meta = {"camera": name, "source": "ai", "ai_conf": res["confidence"], "ai_reason": res["reason"]}
            if not res["person"]:
                t.down_state = "rejected"
                self.hub.add("person", crop, "not_person", meta)
                self._set_person(i, snap, box, t, False, "ai", res["reason"])
                return
            rec = self.hub.add("posture", crop, "down" if res["down"] else "normal", meta)
            if res["down"]:
                t.down_state = "confirmed"
                self._raise_man_down(i, snap, box, t, True, res["reason"],
                                     {"posture": rec["id"]} if rec else {})
            else:
                t.down_state = "rejected"

        if not v.submit_posture(snap, box, {"camera": name}, done):
            t.down_state = "confirmed"
            self._raise_man_down(i, snap, box, t, None, "")

    def _raise_man_down(self, i, frame, box, t, verified, reason, samples=None):
        msg = f"{self._who(i, t)} is down" + (f" - {reason}" if reason else "")
        self._raise(i, "MAN_DOWN", msg, frame, [(box, RED, "MAN DOWN")], verified=verified,
                    extra={"box": box, "track": t.id, "samples": samples or {}}, key_suffix=str(t.id))

    # ------------------------------------------------------------------ PPE
    def _ppe(self, i, frame, people, items, now):
        if not items:
            return
        dc = self.cfg["detection"]
        name = self.streams[i].name
        an = self.analytics[i]
        for p in people:
            t = p["track"]
            if t.ppe_pending or now < t.ppe_next_check or not an.ppe_eligible(p):
                continue
            t.ppe_next_check = now + float(dc.get("ppe_recheck_seconds", 20))
            crop = crop_box(frame, p["box"], pad=0.1)
            status, unsure = {}, []
            for item in items:
                label, conf = self._classify(f"ppe_{item}", crop)
                if label and conf >= self._trust():
                    status[item] = label == "wearing"
                else:
                    unsure.append(item)
            if not unsure:
                self._ppe_result(i, frame, list(p["box"]), t, status, items)
                continue
            v = self.verifier
            if not (v and v.online) or v.pending() > 4:  # fire / man-down checks have priority
                t.ppe_next_check = now + 5
                continue
            t.ppe_pending = True
            snap, box = frame.copy(), list(p["box"])

            def done(res, t=t, snap=snap, box=box, crop=crop, status=status, unsure=unsure):
                t.ppe_pending = False
                if not res.get("ok") or not res.get("visible", True):
                    return
                st = dict(status)
                st.update(res["items"])
                samples = {}
                for item in unsure:
                    if t.ppe_status.get(item) != st[item]:  # first look or the answer changed: worth learning
                        rec = self.hub.add(f"ppe_{item}", crop, "wearing" if st[item] else "missing",
                                           {"camera": name, "source": "ai"})
                        if rec:
                            samples[f"ppe_{item}"] = rec["id"]
                self._ppe_result(i, snap, box, t, st, items, samples)

            if not v.submit_ppe(crop, unsure, {"camera": name}, done):
                t.ppe_pending = False

    def _ppe_result(self, i, frame, box, t, status, items, samples=None):
        t.ppe_status.update(status)
        missing = sorted(k for k in items if t.ppe_status.get(k) is False)
        t.ppe_missing = missing
        if not missing:
            t.ppe_alerted = frozenset()
            return
        if set(missing) - t.ppe_alerted:
            t.ppe_alerted = t.ppe_alerted | frozenset(missing)
            labels = ", ".join(C.PPE_LABELS[m] for m in missing)
            self._raise(i, "PPE", f"{self._who(i, t)} without {labels}", frame,
                        [(box, ORANGE, "NO " + labels.upper())], verified=True,
                        extra={"box": box, "track": t.id, "missing": missing, "samples": samples or {}},
                        key_suffix=str(t.id), cooldown=60)

    # ------------------------------------------------------------------ fire verification + learning
    def _handle_fire_candidates(self, i, frame, cands, now):
        if not cands:
            return
        name = self.streams[i].name
        vc = self.cfg["verifier"]
        rej = [r for r in self._rejected.get(i, []) if r[1] > now]
        self._rejected[i] = rej
        cands = [c for c in cands if not self._fire_suppressed(i, name, c, frame, now)]
        if not cands:
            return
        c = cands[0]
        v = self.verifier
        if v is None or not v.online:
            if vc.get("alert_if_unavailable", True) and (c["source"] == "model" or c["conf"] >= 0.6):
                self._raise(i, "FIRE_UNVERIFIED", f"Possible {c['label']} ({c['conf']:.2f}) - local AI unavailable",
                            frame, [(c["box"], RED, c["label"].upper() + "?")],
                            extra={"box": c["box"], "label": c["label"]})
            return
        if self._verify_pending.get(i) or now < self._verify_next.get(i, 0) or self._fire_inflight >= 3:
            return  # at most 3 fire checks queued, so PPE / man-down / person checks also get the local AI
        self._verify_pending[i] = True
        self._fire_inflight += 1
        snap = frame.copy()
        meta = {"camera": name, "det_conf": round(c["conf"], 3), "det_source": c["source"], "det_label": c["label"]}
        if not v.submit_fire(snap, c["box"], meta, lambda res: self._on_fire_verified(i, snap, c, meta, res)):
            self._verify_pending[i] = False
            self._fire_inflight -= 1

    def _reload_notfire(self):
        threading.Thread(target=lambda: self.notfire.load_from_dataset(self.hub.fire), name="notfire",
                         daemon=True).start()

    def _fire_suppressed(self, i, name, c, frame, now) -> bool:
        """Regions the local AI recently rejected, learned 'not fire' spots, or things that look like a rejected
        'not fire' sample of this camera. A confident detection from the trained fire model is never suppressed -
        it is always verified. Every 10th look-alike is still sent to the local AI as a safety audit."""
        if c["source"] == "model" and c["conf"] >= 0.6:
            return False
        if any(box_iou(c["box"], r[0]) > 0.5 for r in self._rejected.get(i, []) if r[1] > now):
            return True
        if self.ignore.match("fire", name, c["box"], frame.shape):
            return True
        if self.cfg["learning"].get("notfire_lookalikes", True) and \
                self.notfire.match(name, frame, c["box"]) >= NotFireMemory.THRESHOLD:
            n = self._notfire_audit[i] = self._notfire_audit.get(i, 0) + 1
            if n % 10 == 0:
                c["audit"] = True
                return False
            self.notfire.skipped += 1
            return True
        return False

    def _on_fire_verified(self, i, frame, cand, meta, res):
        now = time.time()
        self._verify_pending[i] = False
        self._fire_inflight = max(0, self._fire_inflight - 1)
        self._verify_next[i] = now + float(self.cfg["verifier"].get("cooldown_seconds", 10))
        if not res.get("ok"):
            if self.cfg["verifier"].get("alert_if_unavailable", True) and (cand["source"] == "model" or cand["conf"] >= 0.6):
                self._raise(i, "FIRE_UNVERIFIED", f"Possible {cand['label']} ({cand['conf']:.2f}) - verification failed",
                            frame, [(cand["box"], RED, cand["label"].upper() + "?")],
                            extra={"box": cand["box"], "label": cand["label"]})
            return
        positive = res["positive"]
        label = "fire" if res["fire"] else ("smoke" if res["smoke"] else cand["label"])
        info = {**meta, "ai_conf": round(res["confidence"], 2), "ai_reason": res["reason"], "latency": res["latency"]}
        rec = None
        if self.cfg["learning"]["enabled"]:
            rec = self.hub.fire.add_sample(frame, cand["box"], positive, label, info)
        if not positive:
            self.notfire.add(self.streams[i].name, frame, cand["box"], rec["id"] if rec else "")
        elif cand.get("audit"):
            log.warning("fire on %s looked like a learned 'not fire' sample but the local AI confirmed it",
                        self.streams[i].name)
        if positive:
            self._fire_confirmed.setdefault(i, []).append((cand["box"], label, now + 10))
            self._fire_confirmed[i] = [x for x in self._fire_confirmed[i] if x[2] > now]
            self._raise(i, label.upper(), f"{label.title()} confirmed by local AI ({res['confidence']:.0%}): {res['reason']}",
                        frame, [(cand["box"], RED, label.upper())], verified=True,
                        extra={**info, "box": cand["box"], "label": label, "samples": {"fire": rec["id"]} if rec else {}})
        else:
            rej = self._rejected.setdefault(i, [])
            prev = next((r for r in rej if box_iou(r[0], cand["box"]) > 0.4), None)
            if prev:
                prev[1], prev[2] = now + 1800, prev[2] + 1
            else:
                rej.append([cand["box"], now + 1800, 1])
            if (prev and prev[2] >= 2) and cand["source"] == "heuristic":
                # rejected repeatedly at the same spot: remember permanently (managed in AI Learning > Ignored)
                self.ignore.add("fire", self.streams[i].name, cand["box"], frame.shape, "ai", frame, res["reason"])
        self.hub.maybe_train("fire")

    # ------------------------------------------------------------------ cameras
    def set_person_detection(self, on: bool):
        self.cfg["detection"]["person_detection"] = on
        if not on:
            with self._lock:
                for st in self.cam_stats.values():
                    st.update(people=None, pids=[], down=0, ppe=0)
                for ov in self.overlays.values():
                    ov["people"] = []
        log.info("person detection %s", "on" if on else "off")

    def _check_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                self.gpu_free_mb, self.gpu_total_mb = free / 2**20, total / 2**20
        except Exception:
            pass

    def _check_cameras(self, now):
        limit = float(self.cfg["alerts"].get("camera_offline_seconds", 60))
        for i, s in enumerate(self.streams):
            if not s.enabled:
                continue
            if s.status == "online":
                self._offline_since.pop(i, None)
                self._offline_alerted.discard(i)
                continue
            since = self._offline_since.setdefault(i, now)
            if now - since >= limit and i not in self._offline_alerted:
                self._offline_alerted.add(i)
                self._raise(i, "CAMERA_OFFLINE", f"No video for {int(now - since)} s", None, [], cooldown=0)

    # ------------------------------------------------------------------ alerts
    def _raise(self, i, etype, message, frame, boxes, verified=None, extra=None, key_suffix="", cooldown=None):
        now = time.time()
        key = (i, etype, key_suffix)
        cd = float(self.cfg["alerts"]["cooldown_seconds"]) if cooldown is None else cooldown
        if now - self._last_alert.get(key, 0) < cd:
            return
        self._last_alert[key] = now
        snap = None
        if frame is not None:
            snap = frame.copy()
            for box, color, label in boxes:
                _draw_box(snap, box, color, label, 3)
        ev = self.events.add(self.streams[i].name, etype, message, snap, verified, extra, raw_frame=frame)
        ev["cam_index"] = i
        ev["cam_label"] = self.cam_label(i)
        ev["location"] = self.cam_location(i)
        self._flash_until[i] = now + 6
        self.event_raised.emit(ev)

    # ------------------------------------------------------------------ operator review -> learning
    def apply_feedback(self, event_id, verdict) -> str:
        """verdict: 'correct' | 'false' | 'not_person'. Stores the review and teaches the matching model(s).
        Effects are immediate (ignore memory, relabelled samples) and feed the next training run."""
        ev = self.events.get(event_id)
        if not ev:
            return "Event not found"
        self.events.set_feedback(event_id, verdict)
        if not self.cfg["learning"]["enabled"]:
            return "Review saved"
        weight = int(self.cfg["learning"].get("correction_weight", 3))
        x, etype, cam = ev["extra"], ev["type"], ev["camera"]
        box, samples = x.get("box"), x.get("samples") or {}
        raw = cv2.imread(ev["raw"]) if ev.get("raw") and Path(ev["raw"]).exists() else None
        meta = {"camera": cam, "source": "operator", "event": event_id}
        notes = []

        def learn(task, label, pad=0.15):
            ds = self.hub.dataset(task)
            sid = samples.get(task)
            if sid and ds.get(sid):
                ds.relabel(sid, label, weight)
            elif raw is not None and box:
                ds.add(crop_box(raw, box, pad=pad), label, meta, weight)
            else:
                return
            self.hub.maybe_train(task)
            notes.append(f"{task_title(task)} learned '{label}'")

        if verdict == "not_person":
            learn("person", "not_person")
            if raw is not None and box:
                self.ignore.add("person", cam, box, raw.shape, "operator", raw)
                self._ignore_live_tracks(cam, box, raw.shape)
                notes.append("this spot is now ignored")
        elif etype in FIRE_TYPES and box:
            label = x.get("label", "fire")
            sid = samples.get("fire")
            if verdict == "correct":
                if sid and self.hub.fire.get(sid):
                    self.hub.fire.relabel(sid, label, weight)
                elif raw is not None:
                    self.hub.fire.add_sample(raw, box, True, label, meta, weight)
                notes.append(f"fire detector learned '{label}'")
            else:
                if sid and self.hub.fire.get(sid):
                    self.hub.fire.relabel(sid, "not_fire", weight)
                elif raw is not None:
                    self.hub.fire.add_sample(raw, box, False, label, meta, weight)
                if raw is not None:
                    self.ignore.add("fire", cam, box, raw.shape, "operator", raw)
                    self.notfire.add(cam, raw, box)
                notes.append("fire detector learned 'not fire'; this spot is now ignored")
            self.hub.maybe_train("fire")
        elif etype == "MAN_DOWN":
            learn("posture", "down" if verdict == "correct" else "normal", pad=0.2)
        elif etype == "PPE":
            for item in x.get("missing", []):
                learn(f"ppe_{item}", "missing" if verdict == "correct" else "wearing", pad=0.1)
        elif etype in ("INTRUSION",) and verdict == "correct":
            learn("person", "person")
        return "Review saved" + (": " + "; ".join(notes) if notes else "")

    # ------------------------------------------------------------------ UI helpers
    def stats(self) -> dict:
        with self._lock:
            cams = []
            for i, s in enumerate(self.streams):
                st = self.cam_stats.get(i, {}) if s.status == "online" else {}
                cams.append({"name": s.name, "label": self.cam_label(i), "recorder": getattr(s, "recorder", ""),
                             "view": "sub" if i in self._sub_active else "main",
                             "location": self.cam_location(i), "place_path": self.cam_location(i),
                             "place_short": self.cam_cfg(i).get("place_short") or "",
                             "status": s.status if s.enabled else "disabled",
                             "people": st.get("people"), "down": st.get("down", 0), "ppe": st.get("ppe", 0),
                             "fps": round(self.proc_fps.get(i, 0.0), 1)})
        v = self.verifier
        recs = {}
        for c in cams:
            r = recs.setdefault(c["recorder"], {"name": c["recorder"], "cameras": 0, "online": 0, "enabled": 0,
                                                "people": 0})
            r["cameras"] += 1
            r["enabled"] += c["status"] != "disabled"
            r["online"] += c["status"] == "online"
            r["people"] += c["people"] or 0
        return {
            "cameras": cams,
            "recorders": list(recs.values()),
            "total_people": self._unique_people(cams),
            "detections": sum(c["people"] or 0 for c in cams),
            "online": sum(1 for c in cams if c["status"] == "online"),
            "enabled": sum(1 for c in cams if c["status"] != "disabled"),
            "verifier": {"online": v.online if v else None, "status": v.status if v else "disabled",
                         "pending": v.pending() if v else 0, "latency": round(v.avg_latency, 1) if v else 0,
                         **(v.stats if v else {})},
            "fire_model": self.fire.version if self.fire else "-",
            "learned_models": sorted(self.cls),
            "notfire": {"known": self.notfire.count(), "skipped": self.notfire.skipped},
            "training": self.hub.progress, "training_task": self.hub.running_task,
            "running": self.running, "loading": self.loading, "device": self.device or "-",
            "analysis_fps": round(self.loop_fps, 1),
            "ai_enabled": self.ai_enabled,
            "person_detection": self.cfg["detection"].get("person_detection", True),
            "gpu_free_mb": round(getattr(self, "gpu_free_mb", -1)),
        }

    # ------------------------------------------------------------------ view-only mode / display rates
    def set_ai(self, on: bool):
        self.ai_enabled = on
        if not on:
            with self._lock:
                self.overlays.clear()
                for st in self.cam_stats.values():
                    st.update(people=None, down=0, ppe=0)
            self.proc_fps.clear()
            self.loop_fps = 0.0
        self._apply_rates()
        log.info("AI analysis %s", "resumed" if on else "paused (view-only mode)")

    def set_view(self, visible, display_fps, sub_for=()):
        """Cameras on screen get their full frame rate, the others only what analysis needs. Cameras shown as
        small grid tiles (sub_for) are displayed from the recorder's light sub stream; analysis always uses the
        main stream."""
        visible = set(visible)
        now = time.time()
        if not hasattr(self, "_sub_block"):
            self._sub_block, self._sub_started, self._sub_warned = {}, {}, set()
        from .xvr import _host_port
        troubled = {_host_port(s.source)[0] for s in self.streams if s.enabled and s.status != "online"}
        for i in set(sub_for) & visible:
            if i >= len(self.streams) or not self.streams[i].enabled or not getattr(self.streams[i], "sub_url", None):
                continue
            if self._sub_block.get(i, 0) > now or _host_port(self.streams[i].source)[0] in troubled:
                continue  # sub stream failed / too slow recently, or the recorder's main streams need the connections
            vs = self.view_streams.get(i)
            if vs is None:
                vs = CameraStream(i, self.streams[i].name + " (sub)", self.streams[i].sub_url)
                vs.start()
                self.view_streams[i] = vs
                self._sub_started[i] = now
            vs.retrieve_fps = float(display_fps)
            self._view_used[i] = now
        for i, vs in list(self.view_streams.items()):
            age = now - self._sub_started.get(i, now)
            main = self.streams[i] if i < len(self.streams) else None
            reason = None
            if _host_port(vs.source)[0] in troubled:
                reason, block = "recorder busy", 60
            elif vs.status != "online" and age > 12:
                reason, block = "did not connect", 300
            elif vs.status == "online" and age > 10 and main and main.fps > 0 and vs.fps < 0.75 * main.fps:
                reason, block = f"only {vs.fps:.1f} fps (main {main.fps:.1f})", 600
                rec = getattr(main, "recorder", "")
                if rec not in self._sub_warned:
                    self._sub_warned.add(rec)
                    self.message.emit(
                        f"{rec}: sub stream runs at {vs.fps:.0f} fps, slower than the main stream ({main.fps:.0f} fps), "
                        f"so the grid keeps the main stream. For a smoother grid set the Sub Stream frame rate to "
                        f"15-25 (and resolution D1 / 704x576) in the recorder: Main Menu > Camera > Encode.")
            if reason:
                log.info("grid sub stream %s off: %s", vs.name, reason)
                vs.stop()
                self.view_streams.pop(i, None)
                self._view_used.pop(i, None)
                self._sub_block[i] = now + block
        for i in [k for k, t in self._view_used.items() if now - t > 20]:  # tile gone / enlarged: close it
            self._view_used.pop(i, None)
            vs = self.view_streams.pop(i, None)
            if vs:
                vs.stop()
        active = {i for i in set(sub_for) & visible if i in self.view_streams and self.view_streams[i].status == "online"
                  and self.view_streams[i].read()[0] is not None}
        if visible != self._visible or display_fps != self._display_fps or active != self._sub_active:
            self._visible, self._display_fps, self._sub_active = visible, float(display_fps), active
            self._apply_rates()

    def _apply_rates(self):
        analysis = float(self.cfg["detection"]["fps_per_camera"]) + 1 if self.ai_enabled else 0.0
        for i, s in enumerate(self.streams):
            view = self._display_fps if i in self._visible and i not in self._sub_active else 0.0
            s.retrieve_fps = max(view, analysis, 0.5)

    def _view_frame(self, i):
        """(frame to display, frame key, main-stream shape). Sub stream when active and fresh, else main."""
        main, mfid, _ = self.streams[i].read()
        if i in self._sub_active:
            vs = self.view_streams.get(i)
            if vs is not None:
                f, fid, ts = vs.read()
                if f is not None and time.time() - ts < 2:
                    return f, ("s", fid), (main.shape[:2] if main is not None else f.shape[:2])
        return main, ("m", mfid), (main.shape[:2] if main is not None else None)

    def _unique_people(self, cams) -> int:
        if self.identity is None:
            return sum(c["people"] or 0 for c in cams)
        with self._lock:
            per_cam = {self.streams[i].name: list(st.get("pids", [])) for i, st in self.cam_stats.items()
                       if st.get("people") is not None and i < len(self.streams) and self.streams[i].status == "online"}
        return self.identity.unique_people(per_cam)

    def is_flashing(self, i) -> bool:
        return self._flash_until.get(i, 0) > time.time()

    def snapshot(self, i):
        if i < len(self.streams):
            return self.streams[i].read()[0]
        return None

    def render_new(self, i, tw, th, last_fid, show_boxes=True, show_heads=True):
        """(image, frame_id) or (None, last_fid) when the camera has no newer frame - saves redrawing."""
        if i >= len(self.streams) or not self.streams[i].enabled:
            return None, last_fid
        _, key, _ = self._view_frame(i)
        if key == last_fid:
            return None, last_fid
        return self.render(i, tw, th, show_boxes, show_heads), key

    def render(self, i, tw, th, show_boxes=True, show_heads=True) -> np.ndarray | None:
        if i >= len(self.streams) or tw < 8 or th < 8 or not self.streams[i].enabled:
            return None
        frame, _, shape = self._view_frame(i)
        if frame is None:
            return None
        h, w = shape  # draw in the main stream's geometry (the sub stream is stretched to the same aspect)
        s = min(tw / w, th / h)
        img = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)
        with self._lock:
            ov = self.overlays.get(i)
        if not ov or time.time() - ov["ts"] > 3 or ov["shape"] != (h, w):
            return img
        if ov["zone"] and len(ov["zone"]) >= 3:
            poly = np.array([[x * img.shape[1], y * img.shape[0]] for x, y in ov["zone"]], np.int32)
            layer = img.copy()
            cv2.fillPoly(layer, [poly], (40, 40, 200))
            cv2.addWeighted(layer, 0.18, img, 0.82, 0, img)
            cv2.polylines(img, [poly], True, (60, 60, 255), 2, cv2.LINE_AA)
        scale = max(0.5, min(1.5, img.shape[1] / 900))
        for box, head, color, label, thick in ov["people"]:
            sb = [v * s for v in box]
            if show_boxes:
                _draw_box(img, sb, color, "" if show_heads else label, thick if color != GREY else 1)
            if show_heads:
                _draw_head(img, (head[0] * s, head[1] * s), color, label, scale)
        for box, color, label, thick in ov["fire"]:
            _draw_box(img, [v * s for v in box], color, label, thick)
        return img


def _text(img, text, org, color, fs):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    dark = sum(color) > 380 or color in (YELLOW, GREEN, ORANGE, CYAN)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0) if dark else WHITE, 1, cv2.LINE_AA)


def _draw_box(img, box, color, label, thick=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if label:
        _text(img, label, (x1, y1), color, max(0.4, min(0.8, img.shape[1] / 1400)))


def _draw_head(img, pt, color, label, scale=1.0):
    """Head pointer: a marker ring on the head and a downward triangle above it, with the label on top."""
    x, y = int(pt[0]), int(pt[1])
    s = max(5, int(8 * scale))
    cv2.circle(img, (x, y), s, color, 2, cv2.LINE_AA)
    tip = y - int(s * 1.6)
    tri = np.array([[x, tip], [x - s, tip - int(s * 1.7)], [x + s, tip - int(s * 1.7)]], np.int32)
    cv2.fillPoly(img, [tri], color, cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
    if label:
        fs = max(0.38, min(0.7, 0.45 * scale))
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        _text(img, label, (x - tw / 2, tip - int(s * 1.7) - 3), color, fs)
