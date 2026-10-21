"""XVR / NVR connection: RTSP URL building and threaded camera readers."""
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote

# Must be set before OpenCV opens any FFMPEG stream: TCP is far more reliable than UDP for DVR/XVR streams.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000")
import cv2  # noqa: E402

from .config import parse_channels  # noqa: E402

BRANDS = {
    "dahua": ("Dahua / CP Plus / Amcrest XVR",
              "rtsp://{auth}{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"),
    "hikvision": ("Hikvision / HiLook / HiWatch",
                  "rtsp://{auth}{host}:{port}/Streaming/Channels/{channel}0{hik_stream}"),
    "onvif": ("ONVIF / other brand (stream address from the device)", None),
    "custom": ("Custom RTSP template", None),
}


def _with_auth(uri: str, user: str, pw: str) -> str:
    """rtsp://host/path -> rtsp://user:pass@host/path"""
    if not user or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    rest = rest.split("@", 1)[-1]
    return f"{scheme}://{quote(user, safe='')}:{quote(pw, safe='')}@{rest}"


def build_url(xvr: dict, channel: int, hide_password=False) -> str:
    brand = xvr.get("brand", "dahua")
    if brand == "onvif":
        uri = (xvr.get("stream_urls") or {}).get(str(channel), "")
        pw = "****" if hide_password and xvr.get("password") else xvr.get("password", "")
        return _with_auth(uri, xvr.get("username", ""), pw)
    template = BRANDS.get(brand, BRANDS["dahua"])[1] or xvr.get("custom_template")
    user = xvr.get("username", "")
    pw = "****" if hide_password and xvr.get("password") else xvr.get("password", "")
    auth = f"{quote(user, safe='')}:{quote(pw, safe='')}@" if user else ""
    sub = xvr.get("stream", "sub") == "sub"
    return template.format(auth=auth, user=quote(user, safe=""), password=quote(pw, safe=""),
                           host=xvr.get("host", ""), port=xvr.get("port", 554), channel=channel,
                           subtype=1 if sub else 0, hik_stream=2 if sub else 1)


def source_list(cfg: dict) -> list[dict]:
    """Every camera from every enabled recorder plus extra sources: [{name, url, recorder}, ...]"""
    out, seen = [], set()
    for rec in cfg.get("recorders") or []:
        if not rec.get("enabled", True) or not rec.get("host"):
            continue
        for ch in parse_channels(rec.get("channels", "")):
            name = f"{rec.get('prefix', '')}CH{ch}"
            if name in seen:  # two recorders with the same prefix: keep names unique
                name = f"{rec.get('name', 'REC')}-CH{ch}"
            seen.add(name)
            sub_url = None
            if rec.get("stream", "main") == "main" and (rec.get("brand") in ("dahua", "hikvision") or (
                    rec.get("brand") == "custom" and "{subtype}" in rec.get("custom_template", ""))):
                sub_url = build_url(dict(rec, stream="sub"), ch)  # light stream for small grid tiles
            out.append({"name": name, "url": build_url(rec, ch), "recorder": rec.get("name", ""), "sub_url": sub_url})
    for extra in cfg.get("extra_sources") or []:
        if extra.get("url") not in (None, ""):
            out.append({"name": extra.get("name") or str(extra["url"]), "url": str(extra["url"]),
                        "recorder": extra.get("recorder", "Other sources")})
    return out


def build_sources(cfg: dict) -> list[tuple[str, str]]:
    """[(name, url), ...] - see source_list()."""
    return [(s["name"], s["url"]) for s in source_list(cfg)]


# XVRs refuse bursts of simultaneous RTSP handshakes, so connect a couple of channels at a time.
_SLOTS: dict[str, threading.Semaphore] = {}
_SLOTS_LOCK = threading.Lock()


def _connect_slot(host: str) -> threading.Semaphore:
    """Two simultaneous connection attempts per recorder: an unreachable recorder never blocks the others."""
    with _SLOTS_LOCK:
        return _SLOTS.setdefault(host or "local", threading.Semaphore(2))


def _host_port(source: str):
    from urllib.parse import urlparse
    if "://" not in source:
        return None, None
    u = urlparse(source)
    return u.hostname, u.port or (554 if u.scheme == "rtsp" else 80)


def reachable(source: str, timeout=3.0) -> bool:
    host, port = _host_port(source)
    if not host:
        return True  # file / webcam
    import socket
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def fix_aspect(frame):
    """Dahua 'Lite' encodes (960x1080, 640x720...) are 16:9 pictures squeezed horizontally; restore the true shape
    so people are not distorted (matters for man-down and PPE)."""
    h, w = frame.shape[:2]
    if h >= 480 and 0.8 <= w / h <= 0.95:
        th = min(h, 720)  # 1280x720 is plenty for analytics and keeps decoding/resizing cheap
        return cv2.resize(frame, (round(th * 16 / 9), th), interpolation=cv2.INTER_LINEAR)
    return frame


def _open_capture(source: str):
    if source.strip().isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def grab_frame(source: str, timeout=15.0):
    """Grab one (aspect-corrected) frame, e.g. for drawing a zone while monitoring is stopped."""
    out = [None]

    def work():
        cap = _open_capture(source)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                out[0] = fix_aspect(frame)
        cap.release()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    return out[0]


def probe_source(source: str, timeout=12.0) -> tuple[bool, str]:
    """Try to grab one frame. Used by the 'Test connection' button."""
    result = [False, "Timed out"]

    def work():
        cap = _open_capture(source)
        if not cap.isOpened():
            result[:] = [False, "Could not open stream (check IP, port, username/password, channel)"]
            return
        ok, frame = cap.read()
        cap.release()
        result[:] = [True, f"OK - {frame.shape[1]}x{frame.shape[0]}"] if ok else [False, "Connected but no video frames"]

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    return result[0], result[1]


class CameraStream:
    """Reads a stream in a background thread and always keeps only the newest frame (no lag build-up)."""

    def __init__(self, index: int, name: str, source: str):
        self.index = index
        self.name = name
        self.source = source
        self.status = "connecting"
        self.enabled = True
        self.retrieve_fps = self.DEFAULT_RETRIEVE_FPS
        self.fps = 0.0
        self._is_file = Path(source).is_file()
        self._lock = threading.Lock()
        self._frame = None
        self._conv = None
        self._conv_id = -1
        self._frame_id = 0
        self._ts = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.name}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # frames converted per second (the rest are only demuxed/decoded with cap.grab). The engine raises it to the
    # camera's full rate while the camera is visible on screen and lowers it when only analysis needs frames.
    DEFAULT_RETRIEVE_FPS = 10.0

    def read(self):
        """Newest frame, aspect-corrected lazily (only frames someone actually looks at are resized)."""
        with self._lock:
            raw, fid, ts = self._frame, self._frame_id, self._ts
            if raw is None:
                return None, fid, ts
            if self._conv_id != fid:
                self._conv, self._conv_id = fix_aspect(raw), fid
            return self._conv, fid, ts

    def _run(self):
        backoff = 5.0
        while not self._stop.is_set():
            self.status = "connecting"
            ok, first = False, None
            if not reachable(self.source):  # fail fast (recorder switched off, cable unplugged, wrong IP)
                self.status = "offline"
                self._stop.wait(backoff)
                backoff = min(60.0, backoff * 1.5)
                continue
            with _connect_slot(_host_port(self.source)[0]):
                cap = _open_capture(self.source)
                if cap.isOpened():  # hold the slot until the first frame arrives (handshake finished)
                    ok, first = cap.read()
                    if not ok:
                        cap.release()
            if cap.isOpened() and ok and first is not None:
                with self._lock:
                    self._frame, self._frame_id, self._ts = first, self._frame_id + 1, time.time()
            if not cap.isOpened():
                self.status = "offline"
                cap.release()
                self._stop.wait(backoff)
                backoff = min(60.0, backoff * 1.5)
                continue
            backoff = 5.0
            self.status = "online"
            file_period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25) if self._is_file else 0
            fails, n, t0, last_ret = 0, 0, time.time(), 0.0
            while not self._stop.is_set():
                start = time.time()
                if not cap.grab():  # keep up with the stream; colour conversion only for frames we keep
                    ok, frame = False, None
                elif start - last_ret >= 1.0 / max(0.5, self.retrieve_fps) - 0.004:
                    ok, frame = cap.retrieve()
                    last_ret = start
                else:
                    fails = 0
                    n += 1
                    if file_period:
                        time.sleep(max(0.0, file_period - (time.time() - start)))
                    continue
                if not ok or frame is None:
                    if self._is_file and fails < 3:
                        fails += 1
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop test videos
                        continue
                    fails += 1
                    if fails > 25:
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._ts = time.time()
                n += 1
                if time.time() - t0 >= 2:
                    self.fps = n / (time.time() - t0)
                    n, t0 = 0, time.time()
                if file_period:
                    time.sleep(max(0.0, file_period - (time.time() - start)))
            cap.release()
            if not self._stop.is_set():
                self.status = "reconnecting"
                self._stop.wait(2)
        self.status = "stopped"
