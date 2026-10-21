"""Configuration loading/saving with defaults."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

PPE_ITEMS = ["helmet", "vest", "gloves", "goggles", "mask", "boots"]
PPE_LABELS = {"helmet": "Helmet", "vest": "Safety vest", "gloves": "Gloves", "goggles": "Goggles",
              "mask": "Mask", "boots": "Safety shoes"}

# Per-camera features (key, label). PPE items are selected separately per camera.
FEATURES = [("people", "Man count"), ("man_down", "Man down"), ("fire", "Fire / smoke"), ("ppe", "PPE"),
            ("zone", "Restricted zone"), ("crowd", "Crowd limit")]

CAMERA_DEFAULTS = {
    "enabled": True,
    "label": "",
    "place": "",  # id of the site / building / floor / room the camera is in (see places.py)
    "map_node": "",  # place whose floor plan map_pos refers to
    "map_pos": None,  # [x, y] on that floor plan (0..1)
    "people": True,
    "man_down": True,
    "fire": True,
    "ppe": False,
    "ppe_items": ["helmet", "vest"],
    "zone": False,
    "zone_points": [],  # normalised [[x, y], ...]
    "crowd": False,
    "max_people": 10,
}

RECORDER_DEFAULTS = {
    "name": "XVR 1",
    "prefix": "",  # camera names are prefix + "CH<n>"; must be unique per recorder (e.g. "DVR2-")
    "enabled": True,
    # dahua (also CP Plus / Amcrest / most "XVR" units), hikvision, custom
    "brand": "dahua",
    "host": "192.168.1.108",
    "port": 554,
    "username": "admin",
    "password": "",
    "channels": "1-4",
    "stream": "main",  # sub = low bandwidth, main = full resolution (better for PPE / man down)
    "custom_template": "rtsp://{auth}{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}",
    "stream_urls": {},  # ONVIF devices: channel -> RTSP address reported by the device
    "model": "",
    "mac": "",
}

DEFAULTS = {
    # any number of XVR / NVR / DVR units
    "recorders": [],
    # sites > buildings > floors > rooms, each optionally with a floor-plan image (see places.py)
    "places": [],
    "locations": [],  # old flat list (migrated into places)
    "discovery_ranges": "",
    # Extra sources for testing: [{"name": "Test video", "url": "C:/videos/fire.mp4"}, {"name": "Webcam", "url": "0"}]
    "extra_sources": [],
    "camera_defaults": CAMERA_DEFAULTS,
    "cameras": {},  # per-camera overrides keyed by camera name ("CH1", ...)
    "detection": {
        "device": "auto",
        "imgsz": 640,
        "batch": 4,  # frames per GPU batch (lower if GPU memory is tight)
        "fps_per_camera": 3.0,
        "person_detection": True,  # master switch: people counting, man down, PPE, zones, name tags
        "person_model": "models/yolo11s-pose.pt",
        "person_conf": 0.45,
        "man_down_seconds": 5.0,
        "man_down_angle": 55.0,
        "man_down_verify": True,  # ask the local AI / learned model before raising man-down
        "person_verify": True,  # verify people that never move (mannequins, posters, coats...)
        "static_seconds": 45.0,
        "static_motion": 0.06,  # max movement (fraction of body height) to count as "not moving"
        "ppe_min_person_height": 90,
        "ppe_recheck_seconds": 20.0,
        # hybrid = trained model + colour/flicker heuristic (best recall while the model is still learning)
        "fire_mode": "hybrid",
        "identify_people": True,  # body re-identification: name tags, movement map, unique people count
        "identity_retention_days": 30,  # sightings (who went where) are kept this long
        "fire_model": "models/fire/current.pt",
        "fire_conf": 0.25,
    },
    "verifier": {
        "enabled": True,
        "ollama_url": "http://127.0.0.1:11434",
        "model": "qwen2.5vl:7b",
        "timeout": 120,
        "min_confidence": 0.6,
        "cooldown_seconds": 10.0,
        # If the local AI is down, still raise (unverified) alerts so nothing is missed.
        "alert_if_unavailable": True,
    },
    "learning": {
        "enabled": True,
        "auto_train": True,
        "min_new_samples": 40,  # fire detector
        "min_new_samples_cls": 30,  # person / man-down / PPE classifiers
        "epochs": 20,
        "cls_epochs": 15,
        "batch": 8,
        "imgsz": 640,
        "base_model": "models/yolo11n.pt",
        "cls_base_model": "models/yolo26n-cls.pt",
        "val_fraction": 0.2,
        "require_improvement": True,
        "min_map50_to_deploy": 0.3,
        "min_cls_acc_to_deploy": 0.85,
        "cls_trust_conf": 0.85,  # a learned classifier answers alone above this confidence, else the local AI decides
        "correction_weight": 3,
        "max_auto_person_samples": 150,  # 'walking = person' samples kept (balanced against 'not a person')
        "train_watchdog_minutes": 15,  # a training run with no progress for this long is stopped
        "train_min_free_gpu_mb": 2500,  # train on CPU when the GPU is busy (keeps live detection fast)  # your corrections count this many times in training
        "max_negative_ratio": 1.5,
        "notfire_lookalikes": True,  # skip fire candidates that look like a rejected 'not fire' sample (same camera)
        "fire_map_tolerance": 0.05,
        "cpu_fire_epochs": 8,  # fire fine-tuning on CPU (GPU busy) is ~8 min/epoch: fewer epochs from the current model  # fire model may lose this much mAP50 on reference fires if false alarms drop
    },
    "alerts": {
        "cooldown_seconds": 30.0,
        "sound": True,
        "tray_notifications": True,
        "camera_offline_seconds": 60,
        "retention_days": 60,
    },
    "ui": {
        "autostart": False,
        "display_fps": 25,  # live view: up to this many frames/s per visible camera (the camera's own rate is the limit)
        "show_boxes": True,
        "grid_sub_stream": True,  # small grid tiles show the recorder's sub stream (smoother); full screen = main
        "tile_order": [],  # camera order on the Live page (drag tiles to change)
        "show_heads": True,
        "grid_columns": 0,  # 0 = automatic
        "close_to_tray": True,
        "start_with_windows": False,
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path = CONFIG_PATH) -> dict:
    data = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _merge(DEFAULTS, data)
    _migrate(cfg, data)
    if not path.exists():
        save_config(cfg, path)
    return cfg


def _migrate(cfg, raw):
    """Carry settings from older versions: single 'xvr' -> 'recorders' list, global switches -> per camera."""
    raw = raw or {}
    if "recorders" not in raw and raw.get("xvr"):
        cfg["recorders"] = [{**RECORDER_DEFAULTS, **raw["xvr"], "name": "XVR 1", "prefix": ""}]
    cfg.pop("xvr", None)
    cfg["recorders"] = [{**RECORDER_DEFAULTS, **r} for r in cfg.get("recorders") or []]
    d = raw.get("detection", {})
    cd = cfg["camera_defaults"]
    if "cameras" not in (raw or {}):
        if "man_down_enabled" in d:
            cd["man_down"] = bool(d["man_down_enabled"])
        if "fire_enabled" in d:
            cd["fire"] = bool(d["fire_enabled"])
        if "ppe_required" in d:
            cd["ppe_items"] = [i for i in d["ppe_required"] if i in PPE_ITEMS]
    for k in ("location", "room"):
        cd.pop(k, None)
    from . import places
    places.migrate(cfg)
    for k in ("man_down_enabled", "fire_enabled", "ppe_enabled", "ppe_mode", "ppe_model", "ppe_conf",
              "ppe_required", "ppe_class_aliases"):
        cfg["detection"].pop(k, None)


def save_config(cfg: dict, path: Path = CONFIG_PATH) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def camera_cfg(cfg: dict, name: str) -> dict:
    c = copy.deepcopy(cfg.get("camera_defaults") or CAMERA_DEFAULTS)
    c.update((cfg.get("cameras") or {}).get(name) or {})
    from . import places
    c.update(places.describe(cfg, c.get("place")))  # derived: location, room, room_name, place_path ...
    return c


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def parse_channels(text) -> list[int]:
    """'1-4,6,9-10' -> [1, 2, 3, 4, 6, 9, 10]"""
    if isinstance(text, int):
        return [text]
    if isinstance(text, list):
        return [int(x) for x in text]
    out = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))
