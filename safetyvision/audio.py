"""Camera audio: listen to one camera at a time (G.711 / AAC from the recorder's RTSP stream), with volume and
level meter. Decoding runs in a background thread (PyAV / FFmpeg); playback through Qt Multimedia."""
import logging
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

log = logging.getLogger(__name__)
RATE = 16000


class AudioPlayer(QObject):
    chunk = Signal(bytes)
    level = Signal(float)  # 0..1 (peak of the last chunk)
    state = Signal(str, str)  # camera, "playing" | "no audio" | "error: ..." | "stopped"

    def __init__(self):
        super().__init__()
        fmt = QAudioFormat()
        fmt.setSampleRate(RATE)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self.sink = QAudioSink(QMediaDevices.defaultAudioOutput(), fmt)
        self.sink.setBufferSize(RATE * 2 // 2)  # ~0.5 s
        self.io = None
        self.camera = None
        self._stop = threading.Event()
        self._thread = None
        self.volume = 1.0
        self.chunk.connect(self._write)

    def set_volume(self, v: float):
        self.volume = max(0.0, min(1.0, v))
        self.sink.setVolume(self.volume)

    def play(self, camera: str, url: str):
        self.stop()
        self.camera = camera
        self._stop = threading.Event()
        self.io = self.sink.start()
        self.sink.setVolume(self.volume)
        self._thread = threading.Thread(target=self._run, args=(camera, url, self._stop), name="audio", daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread = None
        if self.io is not None:
            self.sink.stop()
            self.io = None
        if self.camera:
            self.state.emit(self.camera, "stopped")
        self.camera = None

    def _write(self, data: bytes):
        if self.io is None:
            return
        if self.sink.bytesFree() < len(data):  # behind (network hiccup): drop instead of building up delay
            return
        self.io.write(data)

    def _run(self, camera, url, stop):
        import av
        while not stop.is_set():
            try:
                c = av.open(url, options={"rtsp_transport": "tcp", "timeout": "5000000"}, timeout=8)
            except Exception as e:
                self.state.emit(camera, f"error: {type(e).__name__}")
                stop.wait(5)
                continue
            try:
                if not c.streams.audio:
                    self.state.emit(camera, "no audio")
                    return
                self.state.emit(camera, "playing")
                res = av.AudioResampler(format="s16", layout="mono", rate=RATE)
                for frame in c.decode(c.streams.audio[0]):
                    if stop.is_set():
                        break
                    for out in res.resample(frame):
                        pcm = out.to_ndarray().astype(np.int16).ravel()
                        if pcm.size:
                            self.level.emit(float(np.abs(pcm).max()) / 32768.0)
                            self.chunk.emit(pcm.tobytes())
            except Exception as e:
                if not stop.is_set():
                    log.warning("audio %s: %s", camera, e)
                    self.state.emit(camera, f"error: {type(e).__name__}")
                    time.sleep(2)
            finally:
                c.close()
