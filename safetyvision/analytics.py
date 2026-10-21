"""Per-camera analytics: person tracking, motion/static analysis, head pointer, man-down timing, zones."""
import math
from collections import deque

import cv2
import numpy as np

KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR, KP_LS, KP_RS, KP_LH, KP_RH = 0, 1, 2, 3, 4, 5, 6, 11, 12


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    def __init__(self, tid, box, now):
        self.id = tid
        self.box = box
        self.first_seen = now
        self.last_seen = now
        self.history = deque(maxlen=600)  # (t, cx, cy, h)
        self.max_travel = 0.0  # furthest the centre has moved from its first position, in body heights
        # man down
        self.fallen_since = None
        self.upright_since = None
        self.down = False  # rule says down for long enough
        self.down_state = None  # None | pending | confirmed | rejected
        # person verification (static look-alikes)
        self.person_state = None  # None | pending | verified | ignored
        self.person_retry = 0.0
        self.sampled_person = False
        # PPE
        self.ppe_status: dict[str, bool] = {}
        self.ppe_missing: list[str] = []
        self.ppe_alerted: frozenset = frozenset()
        self.ppe_next_check = 0.0
        self.ppe_pending = False
        self.ppe_sampled = False
        # zone
        self.in_zone = False
        self.zone_alerted = False


class IoUTracker:
    """Lightweight IoU tracker with a centroid fallback (keeps the ID when a person falls and the box changes shape)."""

    def __init__(self, iou_thr=0.25, max_age=3.0):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, boxes, now) -> list[Track]:
        tracks = list(self.tracks.values())
        pairs = []
        for ti, t in enumerate(tracks):
            tcx, tcy = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            diag = math.hypot(t.box[2] - t.box[0], t.box[3] - t.box[1]) or 1
            for bi, b in enumerate(boxes):
                iou = box_iou(t.box, b)
                if iou >= self.iou_thr:
                    pairs.append((iou, ti, bi))
                else:
                    d = math.hypot((b[0] + b[2]) / 2 - tcx, (b[1] + b[3]) / 2 - tcy) / diag
                    if d < 0.5:
                        pairs.append((0.01 * (1 - d), ti, bi))
        pairs.sort(reverse=True)
        out = [None] * len(boxes)
        used = set()
        for _, ti, bi in pairs:
            if ti in used or out[bi] is not None:
                continue
            used.add(ti)
            t = tracks[ti]
            t.box, t.last_seen = boxes[bi], now
            out[bi] = t
        for bi, b in enumerate(boxes):
            if out[bi] is None:
                t = Track(self._next, b, now)
                self._next += 1
                self.tracks[t.id] = t
                out[bi] = t
        for t, b in zip(out, boxes):
            cx, cy, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(1.0, b[3] - b[1])
            t.history.append((now, cx, cy, h))
            _, fx, fy, fh = t.history[0]
            t.max_travel = max(t.max_travel, math.hypot(cx - fx, cy - fy) / max(h, fh))
        for tid in [k for k, t in self.tracks.items() if now - t.last_seen > self.max_age]:
            del self.tracks[tid]
        return out


def is_static(t: Track, now, seconds, motion_frac) -> bool:
    """True when the track has existed for `seconds` and its centre never moved more than motion_frac body heights."""
    if now - t.first_seen < seconds or len(t.history) < 5:
        return False
    pts = [(x, y, h) for (ts, x, y, h) in t.history if now - ts <= seconds]
    if len(pts) < 5:
        return False
    xs, ys, hs = zip(*pts)
    h = float(np.median(hs))
    return (max(xs) - min(xs)) / h <= motion_frac and (max(ys) - min(ys)) / h <= motion_frac


def head_point(p) -> tuple[float, float]:
    """Head position from face keypoints (nose/eyes/ears), falling back to the top of the box."""
    k = p.get("kpts")
    x1, y1, x2, y2 = p["box"]
    if k is not None:
        k = np.asarray(k)
        pts = [k[j, :2] for j in (KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR) if k[j, 2] > 0.35]
        if pts:
            m = np.mean(pts, axis=0)
            return float(m[0]), float(m[1])
    return (x1 + x2) / 2, y1 + 0.08 * (y2 - y1)


def feet_point(p) -> tuple[float, float]:
    x1, y1, x2, y2 = p["box"]
    return (x1 + x2) / 2, y2


def torso_angle(kpts, min_conf=0.35):
    """Angle of the torso from vertical in degrees (0 = standing, 90 = lying, 180 = upside down)."""
    if kpts is None:
        return None
    k = np.asarray(kpts)
    if k.shape[0] < 13 or min(k[KP_LS, 2], k[KP_RS, 2], k[KP_LH, 2], k[KP_RH, 2]) < min_conf:
        return None
    sx, sy = (k[KP_LS, 0] + k[KP_RS, 0]) / 2, (k[KP_LS, 1] + k[KP_RS, 1]) / 2
    hx, hy = (k[KP_LH, 0] + k[KP_RH, 0]) / 2, (k[KP_LH, 1] + k[KP_RH, 1]) / 2
    dx, dy = sx - hx, sy - hy
    if abs(dx) + abs(dy) < 1:
        return None
    return math.degrees(math.atan2(abs(dx), -dy))


def is_fallen(box, kpts, angle_thr=55.0) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    aspect = w / max(h, 1.0)
    ang = torso_angle(kpts)
    if ang is not None:
        return ang >= angle_thr and aspect >= 0.9
    return aspect >= 1.5


def touches_border(box, shape, margin=4) -> bool:
    h, w = shape[:2]
    return box[0] <= margin or box[1] <= margin or box[2] >= w - margin or box[3] >= h - margin


def in_polygon(pt, norm_points, shape) -> bool:
    if len(norm_points) < 3:
        return False
    h, w = shape[:2]
    poly = np.array([[x * w, y * h] for x, y in norm_points], np.float32)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


class CameraAnalytics:
    def __init__(self, det_cfg: dict):
        self.cfg = det_cfg
        self.tracker = IoUTracker()

    def update_people(self, people: list[dict], now: float, shape=None) -> None:
        """Assign tracks and update man-down timers (p['track'], p['fallen'], track.down).
        People cut off by the image border are never 'fallen': a partly visible body looks wide."""
        tracks = self.tracker.update([p["box"] for p in people], now)
        hold = float(self.cfg.get("man_down_seconds", 5.0))
        thr = float(self.cfg.get("man_down_angle", 55.0))
        for p, t in zip(people, tracks):
            p["track"] = t
            fallen = is_fallen(p["box"], p.get("kpts"), thr) and not (shape is not None and touches_border(p["box"], shape))
            p["fallen"] = fallen
            if fallen:
                t.upright_since = None
                if t.fallen_since is None:
                    t.fallen_since = now
                t.down = now - t.fallen_since >= hold
            else:
                if t.upright_since is None:
                    t.upright_since = now
                if now - t.upright_since > 1.5:  # tolerate single-frame pose glitches; person got up
                    t.fallen_since, t.down, t.down_state = None, False, None

    def ppe_eligible(self, p: dict) -> bool:
        b = p["box"]
        return (b[3] - b[1]) >= self.cfg.get("ppe_min_person_height", 90) and not p.get("fallen")
