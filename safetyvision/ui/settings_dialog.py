"""Settings dialog: XVR connection, detection, local AI and learning options."""
import copy
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QListWidget,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget)

from ..config import parse_channels
from ..verifier import list_models
from ..xvr import BRANDS, build_url, probe_source


class _Bridge(QObject):
    done = Signal(str, str)


def _spin(lo, hi, val, step=1, dec=None):
    if dec is None:
        w = QSpinBox()
    else:
        w = QDoubleSpinBox()
        w.setDecimals(dec)
        w.setSingleStep(step)
    w.setRange(lo, hi)
    w.setValue(val)
    return w


class SettingsDialog(QDialog):
    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(720, 640)
        self.cfg = copy.deepcopy(cfg)
        self._bridge = _Bridge()
        self._bridge.done.connect(self._on_async_done)
        tabs = QTabWidget()
        tabs.addTab(self._xvr_tab(), "Recorders")
        tabs.addTab(self._detect_tab(), "Detection")
        tabs.addTab(self._ai_tab(), "Local AI")
        tabs.addTab(self._learn_tab(), "Learning")
        tabs.addTab(self._app_tab(), "Alerts && app")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(buttons)

    # ---------------------------------------------------------------- tabs
    def _xvr_tab(self):
        """Recorders: any number of XVR / NVR / DVR units."""
        self.recs = [dict(r) for r in self.cfg.get("recorders") or []]
        self._cur = -1
        w = QWidget()
        outer = QHBoxLayout(w)
        left = QVBoxLayout()
        self.rec_list = QListWidget()
        self.rec_list.setMaximumWidth(210)
        self.rec_list.currentRowChanged.connect(self._select_rec)
        left.addWidget(QLabel("Recorders"))
        left.addWidget(self.rec_list, 1)
        row = QHBoxLayout()
        find = QPushButton("Find devices on network...")
        find.setObjectName("primary")
        find.clicked.connect(self._find_devices)
        left.addWidget(find)
        for text, fn in (("Add", self._add_rec), ("Copy", self._dup_rec), ("Remove", self._del_rec)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        left.addLayout(row)
        outer.addLayout(left)

        form_host = QWidget()
        f = QFormLayout(form_host)
        self.rec_enabled = QCheckBox("Connect to this recorder")
        self.rec_name = QLineEdit()
        self.rec_prefix = QLineEdit()
        self.rec_prefix.setPlaceholderText("e.g. DVR2-  (camera names become DVR2-CH1 ...)")
        self.brand = QComboBox()
        for key, (label, _) in BRANDS.items():
            self.brand.addItem(label, key)
        self.host = QLineEdit()
        self.port = _spin(1, 65535, 554)
        self.user = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.channels = QLineEdit()
        self.channels.setPlaceholderText("e.g. 1-16 or 1,2,5")
        self.stream = QComboBox()
        self.stream.addItem("Sub stream (low bandwidth)", "sub")
        self.stream.addItem("Main stream (full resolution, recommended)", "main")
        self.template = QLineEdit()
        self.template.setToolTip("Placeholders: {auth} {user} {password} {host} {port} {channel} {subtype} {hik_stream}")
        self.url_preview = QLabel()
        self.url_preview.setWordWrap(True)
        self.url_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.url_preview.setStyleSheet("color:#8b949e")
        test = QPushButton("Test connection (first channel)")
        test.clicked.connect(self._test_xvr)
        self.test_result = QLabel("")
        self.test_result.setWordWrap(True)
        f.addRow(self.rec_enabled)
        f.addRow("Name", self.rec_name)
        f.addRow("Camera name prefix", self.rec_prefix)
        f.addRow("Brand", self.brand)
        f.addRow("IP address", self.host)
        f.addRow("RTSP port", self.port)
        f.addRow("Username", self.user)
        f.addRow("Password", self.password)
        f.addRow("Channels", self.channels)
        f.addRow("Stream", self.stream)
        f.addRow("Custom template", self.template)
        f.addRow("URL preview", self.url_preview)
        f.addRow(test, self.test_result)
        self.extra = QPlainTextEdit("\n".join(f"{e.get('name', '')} | {e.get('url', '')}" for e in self.cfg.get("extra_sources") or []))
        self.extra.setPlaceholderText("Optional extra sources, one per line:  Name | rtsp://... or C:/video.mp4 or 0 (webcam)")
        self.extra.setMaximumHeight(70)
        f.addRow("Extra sources", self.extra)
        self.rec_form = form_host
        outer.addWidget(form_host, 1)
        for wdg in (self.rec_name, self.rec_prefix, self.host, self.user, self.password, self.channels, self.template):
            wdg.textChanged.connect(self._rec_changed)
        self.brand.currentIndexChanged.connect(self._rec_changed)
        self.stream.currentIndexChanged.connect(self._rec_changed)
        self.port.valueChanged.connect(self._rec_changed)
        self.rec_enabled.toggled.connect(self._rec_changed)
        self._refresh_rec_list()
        if not self.recs:
            self._add_rec()
        self.rec_list.setCurrentRow(0)
        return w

    def _find_devices(self):
        from .discovery_dialog import DiscoveryDialog
        dlg = DiscoveryDialog(self.cfg, self.recs, self)
        if dlg.exec() and dlg.added:
            self.recs.extend(dlg.added)
            self._refresh_rec_list()
            self.rec_list.setCurrentRow(len(self.recs) - 1)
            self.test_result.setText(f"Added {len(dlg.added)} device(s). Save to connect.")

    def _refresh_rec_list(self):
        cur = self.rec_list.currentRow()
        self.rec_list.blockSignals(True)
        self.rec_list.clear()
        for r in self.recs:
            try:
                n = len(parse_channels(r.get("channels", "")))
            except ValueError:
                n = 0
            self.rec_list.addItem(f"{r.get('name') or r.get('host')}  ({n} ch)" + ("" if r.get("enabled", True) else "  - off"))
        self.rec_list.blockSignals(False)
        if 0 <= cur < len(self.recs):
            self.rec_list.setCurrentRow(cur)

    def _select_rec(self, row):
        self._cur = -1  # don't write back while loading the form
        self.rec_form.setEnabled(0 <= row < len(self.recs))
        if not (0 <= row < len(self.recs)):
            return
        r = self.recs[row]
        self.rec_enabled.setChecked(r.get("enabled", True))
        self.rec_name.setText(r.get("name", ""))
        self.rec_prefix.setText(r.get("prefix", ""))
        self.brand.setCurrentIndex(max(0, self.brand.findData(r.get("brand", "dahua"))))
        self.host.setText(r.get("host", ""))
        self.port.setValue(int(r.get("port", 554)))
        self.user.setText(r.get("username", ""))
        self.password.setText(r.get("password", ""))
        self.channels.setText(str(r.get("channels", "")))
        self.stream.setCurrentIndex(0 if r.get("stream") == "sub" else 1)
        self.template.setText(r.get("custom_template", ""))
        self.test_result.setText("")
        self._cur = row
        self._update_preview()

    def _rec_changed(self, *_):
        if self._cur < 0:
            return
        self.recs[self._cur].update(self._xvr_values())
        self._update_preview()
        self._refresh_rec_list()

    def _add_rec(self, *_):
        from ..config import RECORDER_DEFAULTS
        n = len(self.recs) + 1
        self.recs.append({**RECORDER_DEFAULTS, "name": f"Recorder {n}", "prefix": "" if n == 1 else f"DVR{n}-",
                          "host": "", "password": ""})
        self._refresh_rec_list()
        self.rec_list.setCurrentRow(len(self.recs) - 1)

    def _dup_rec(self, *_):
        if 0 <= self._cur < len(self.recs):
            n = len(self.recs) + 1
            self.recs.append({**self.recs[self._cur], "name": f"Recorder {n}", "prefix": f"DVR{n}-", "host": ""})
            self._refresh_rec_list()
            self.rec_list.setCurrentRow(len(self.recs) - 1)

    def _del_rec(self, *_):
        if 0 <= self._cur < len(self.recs):
            del self.recs[self._cur]
            self._cur = -1
            self._refresh_rec_list()
            self.rec_list.setCurrentRow(min(len(self.recs) - 1, 0) if self.recs else -1)
            if not self.recs:
                self.rec_form.setEnabled(False)

    def _detect_tab(self):
        d = self.cfg["detection"]
        w = QWidget()
        f = QFormLayout(w)
        note = QLabel("Choose which detections run on which camera on the <b>Cameras</b> page.")
        note.setObjectName("muted")
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda:0", "cpu"])
        self.device.setCurrentText(d["device"])
        self.fps = _spin(0.2, 30, float(d["fps_per_camera"]), 0.5, 1)
        self.imgsz = QComboBox()
        self.imgsz.addItems(["480", "640", "800", "960", "1280"])
        self.imgsz.setCurrentText(str(d["imgsz"]))
        self.gpu_batch = _spin(1, 16, int(d.get("batch", 4)))
        self.person_on = QCheckBox("Person detection (counting, man down, PPE, zones, name tags)")
        self.person_on.setChecked(d.get("person_detection", True))
        self.person_model = self._path_row(d["person_model"])
        self.person_conf = _spin(0.05, 0.95, d["person_conf"], 0.05, 2)
        self.person_verify = QCheckBox("Verify people who never move with the local AI, and learn to ignore look-alikes")
        self.person_verify.setChecked(d["person_verify"])
        self.static_secs = _spin(5, 600, d["static_seconds"], 5, 0)
        self.man_down_verify = QCheckBox("Confirm man-down with the local AI / learned model before alerting")
        self.man_down_verify.setChecked(d["man_down_verify"])
        self.man_down_secs = _spin(1, 120, d["man_down_seconds"], 1, 1)
        self.man_down_angle = _spin(30, 89, d["man_down_angle"], 1, 0)
        self.ppe_min_h = _spin(30, 1000, int(d["ppe_min_person_height"]))
        self.ppe_recheck = _spin(5, 600, d["ppe_recheck_seconds"], 5, 0)
        self.fire_mode = QComboBox()
        self.fire_mode.addItem("Hybrid: trained model + colour/flicker heuristic (recommended)", "hybrid")
        self.fire_mode.addItem("Trained model only", "model")
        self.fire_mode.addItem("Heuristic only", "heuristic")
        self.fire_mode.setCurrentIndex(max(0, self.fire_mode.findData(d["fire_mode"])))
        self.fire_conf = _spin(0.05, 0.95, d["fire_conf"], 0.05, 2)
        self.identify = QCheckBox("Recognise people by body shape / clothing (name tags, movement map, unique count)")
        self.identify.setChecked(d.get("identify_people", True))
        self.id_days = _spin(1, 365, int(d.get("identity_retention_days", 30)))
        f.addRow(note)
        f.addRow("Device", self.device)
        f.addRow("Analysis FPS per camera", self.fps)
        f.addRow("Inference size", self.imgsz)
        f.addRow("GPU batch (frames)", self.gpu_batch)
        f.addRow(self.person_on)
        f.addRow("Person/pose model", self.person_model)
        f.addRow("Person confidence", self.person_conf)
        f.addRow(self.person_verify)
        f.addRow("Not moving for (s) before check", self.static_secs)
        f.addRow(self.man_down_verify)
        f.addRow("Man down after (s)", self.man_down_secs)
        f.addRow("Torso angle threshold (deg)", self.man_down_angle)
        f.addRow("PPE: min person height (px)", self.ppe_min_h)
        f.addRow("PPE: re-check each person every (s)", self.ppe_recheck)
        f.addRow("Fire detection mode", self.fire_mode)
        f.addRow("Fire candidate confidence", self.fire_conf)
        f.addRow(self.identify)
        f.addRow("Keep movement history (days)", self.id_days)
        return w

    def _ai_tab(self):
        v = self.cfg["verifier"]
        w = QWidget()
        f = QFormLayout(w)
        self.ai_enabled = QCheckBox("Use the local AI (Ollama) to verify fire, man-down, PPE and static people")
        self.ai_enabled.setChecked(v["enabled"])
        self.ai_url = QLineEdit(v["ollama_url"])
        self.ai_model = QComboBox()
        self.ai_model.setEditable(True)
        self.ai_model.setCurrentText(v["model"])
        refresh = QPushButton("Refresh models / test")
        refresh.clicked.connect(self._test_ai)
        self.ai_result = QLabel("")
        self.ai_result.setWordWrap(True)
        self.ai_min_conf = _spin(0.1, 1.0, v["min_confidence"], 0.05, 2)
        self.ai_cooldown = _spin(1, 600, v["cooldown_seconds"], 1, 0)
        self.ai_timeout = _spin(10, 600, int(v["timeout"]))
        self.ai_fallback = QCheckBox("Still alert (unverified) if the local AI is unavailable")
        self.ai_fallback.setChecked(v["alert_if_unavailable"])
        note = QLabel("Use a vision model, e.g. <b>qwen2.5vl:7b</b> (best) or <b>qwen2.5vl:3b</b> (faster, less VRAM). "
                      "Install with <code>ollama pull qwen2.5vl:7b</code>.")
        note.setWordWrap(True)
        f.addRow(self.ai_enabled)
        f.addRow("Ollama URL", self.ai_url)
        f.addRow("Vision model", self.ai_model)
        f.addRow(refresh, self.ai_result)
        f.addRow("Min AI confidence", self.ai_min_conf)
        f.addRow("Per-camera cooldown (s)", self.ai_cooldown)
        f.addRow("Request timeout (s)", self.ai_timeout)
        f.addRow(self.ai_fallback)
        f.addRow(note)
        return w

    def _learn_tab(self):
        lc = self.cfg["learning"]
        w = QWidget()
        f = QFormLayout(w)
        self.learn = QCheckBox("Learn from every local-AI verdict and every review you make")
        self.learn.setChecked(lc["enabled"])
        self.auto_train = QCheckBox("Automatically retrain models when enough new samples exist")
        self.auto_train.setChecked(lc["auto_train"])
        self.min_new = _spin(5, 10000, int(lc["min_new_samples"]))
        self.min_new_cls = _spin(5, 10000, int(lc["min_new_samples_cls"]))
        self.epochs = _spin(1, 300, int(lc["epochs"]))
        self.cls_epochs = _spin(1, 300, int(lc["cls_epochs"]))
        self.trust = _spin(0.5, 0.99, lc["cls_trust_conf"], 0.01, 2)
        self.weight = _spin(1, 10, int(lc["correction_weight"]))
        self.require_impr = QCheckBox("Deploy new weights only if accuracy is not worse than the current model")
        self.require_impr.setChecked(lc["require_improvement"])
        f.addRow(self.learn)
        f.addRow(self.auto_train)
        f.addRow("Fire model: retrain after N new samples", self.min_new)
        f.addRow("Other models: retrain after N new samples", self.min_new_cls)
        f.addRow("Fire model epochs", self.epochs)
        f.addRow("Classifier epochs", self.cls_epochs)
        f.addRow("Trust a learned model above confidence", self.trust)
        f.addRow("Your corrections count x", self.weight)
        f.addRow(self.require_impr)
        return w

    def _app_tab(self):
        a, u = self.cfg["alerts"], self.cfg["ui"]
        w = QWidget()
        f = QFormLayout(w)
        self.alert_cd = _spin(1, 3600, a["cooldown_seconds"], 5, 0)
        self.offline_s = _spin(10, 3600, int(a["camera_offline_seconds"]))
        self.retention = _spin(1, 3650, int(a["retention_days"]))
        self.sound = QCheckBox("Play sound on alerts")
        self.sound.setChecked(a["sound"])
        self.tray_notif = QCheckBox("Show desktop notifications for alerts")
        self.tray_notif.setChecked(a["tray_notifications"])
        self.autostart = QCheckBox("Start monitoring when the app opens")
        self.autostart.setChecked(u["autostart"])
        self.close_tray = QCheckBox("Closing the window keeps monitoring in the system tray")
        self.close_tray.setChecked(u["close_to_tray"])
        self.disp_fps = _spin(5, 60, int(u.get("display_fps", 25)))
        self.grid_sub = QCheckBox("Camera grid uses the recorders' sub stream (smoother; full quality when opened)")
        self.grid_sub.setChecked(u.get("grid_sub_stream", True))
        self.win_start = QCheckBox("Start SafetyVision when Windows starts (monitoring starts automatically)")
        self.win_start.setChecked(u["start_with_windows"])
        f.addRow("Alert cooldown per camera (s)", self.alert_cd)
        f.addRow("Camera offline alert after (s)", self.offline_s)
        f.addRow("Keep events and snapshots (days)", self.retention)
        f.addRow("Live view frame rate (max per camera)", self.disp_fps)
        f.addRow(self.grid_sub)
        f.addRow(self.sound)
        f.addRow(self.tray_notif)
        f.addRow(self.autostart)
        f.addRow(self.close_tray)
        f.addRow(self.win_start)
        un = QPushButton("Uninstall SafetyVision ...")
        un.clicked.connect(self._uninstall)
        f.addRow(QLabel(""))
        f.addRow(un)
        return w

    def _uninstall(self):
        import subprocess
        from ..config import ROOT
        exe = next((p for p in (ROOT / "Uninstall SafetyVision.exe", ROOT / "installer" / "SafetyVision-Setup.exe")
                    if p.exists()), None)
        if exe is None:
            self.test_result.setText("No uninstaller found (this copy was not installed with the setup program).")
            return
        subprocess.Popen([str(exe), "--uninstall"], close_fds=True)
        self.reject()

    # ---------------------------------------------------------------- helpers
    def _path_row(self, value):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(value)
        b = QPushButton("...")
        b.setFixedWidth(34)
        b.clicked.connect(lambda: (p := QFileDialog.getOpenFileName(self, "Select model", "", "YOLO weights (*.pt)")[0])
                          and edit.setText(p))
        h.addWidget(edit)
        h.addWidget(b)
        w.edit = edit
        return w

    def _xvr_values(self):
        return {"enabled": self.rec_enabled.isChecked(), "name": self.rec_name.text().strip(),
                "prefix": self.rec_prefix.text().strip(),
                "brand": self.brand.currentData(), "host": self.host.text().strip(), "port": self.port.value(),
                "username": self.user.text().strip(), "password": self.password.text(),
                "channels": self.channels.text().strip(), "stream": self.stream.currentData(),
                "custom_template": self.template.text().strip()}

    def _update_preview(self):
        x = self._xvr_values()
        try:
            chs = parse_channels(x["channels"]) or [1]
            self.url_preview.setText(f"{len(chs)} channel(s) - e.g. {build_url(x, chs[0], hide_password=True)}")
        except Exception as e:
            self.url_preview.setText(f"Invalid: {e}")

    def _test_xvr(self):
        x = self._xvr_values()
        try:
            ch = (parse_channels(x["channels"]) or [1])[0]
        except ValueError:
            self.test_result.setText("Invalid channel list")
            return
        url = build_url(x, ch)
        self.test_result.setText("Testing...")

        def work():
            ok, msg = probe_source(url)
            self._bridge.done.emit("xvr", ("Connected: " if ok else "Failed: ") + msg)

        threading.Thread(target=work, daemon=True).start()

    def _test_ai(self):
        url = self.ai_url.text().strip()
        self.ai_result.setText("Checking...")

        def work():
            try:
                models = list_models(url)
                self._bridge.done.emit("ai_models", "\n".join(models))
            except Exception as e:
                self._bridge.done.emit("ai", f"Not reachable: {e.__class__.__name__}. Is Ollama running?")

        threading.Thread(target=work, daemon=True).start()

    def _on_async_done(self, kind, text):
        if kind == "xvr":
            self.test_result.setText(text)
        elif kind == "ai_models":
            cur = self.ai_model.currentText()
            models = [m for m in text.split("\n") if m]
            self.ai_model.clear()
            self.ai_model.addItems(models)
            self.ai_model.setCurrentText(cur)
            vision = [m for m in models if any(k in m for k in ("vl", "llava", "vision", "gemma3", "minicpm-v", "moondream"))]
            ok = cur in models
            self.ai_result.setText(("Connected. " if ok else f"Connected, but '{cur}' is not installed. ")
                                   + (f"Vision models: {', '.join(vision)}" if vision else "No vision models found."))
        else:
            self.ai_result.setText(text)

    def _save(self):
        prefixes = [r.get("prefix", "") for r in self.recs if r.get("enabled", True)]
        if len(prefixes) != len(set(prefixes)):
            self.test_result.setText("Each recorder needs a different camera name prefix (e.g. '' and 'DVR2-').")
            return
        for r in self.recs:
            try:
                parse_channels(r.get("channels", ""))
            except ValueError:
                self.test_result.setText(f"Invalid channel list for {r.get('name')}")
                return
        c = self.cfg
        c["recorders"] = self.recs
        c.pop("xvr", None)
        extras = []
        for line in self.extra.toPlainText().splitlines():
            if not line.strip():
                continue
            name, _, url = line.partition("|") if "|" in line else ("", "", line)
            extras.append({"name": name.strip() or url.strip(), "url": url.strip()})
        c["extra_sources"] = extras
        c["detection"].update({
            "device": self.device.currentText(), "fps_per_camera": self.fps.value(),
            "imgsz": int(self.imgsz.currentText()), "batch": self.gpu_batch.value(),
            "person_detection": self.person_on.isChecked(),
            "person_model": self.person_model.edit.text().strip(), "person_conf": self.person_conf.value(),
            "person_verify": self.person_verify.isChecked(), "static_seconds": self.static_secs.value(),
            "man_down_verify": self.man_down_verify.isChecked(), "man_down_seconds": self.man_down_secs.value(),
            "man_down_angle": self.man_down_angle.value(), "ppe_min_person_height": self.ppe_min_h.value(),
            "ppe_recheck_seconds": self.ppe_recheck.value(), "fire_mode": self.fire_mode.currentData(),
            "fire_conf": self.fire_conf.value(), "identify_people": self.identify.isChecked(),
            "identity_retention_days": self.id_days.value()})
        c["verifier"].update({"enabled": self.ai_enabled.isChecked(), "ollama_url": self.ai_url.text().strip(),
                              "model": self.ai_model.currentText().strip(), "min_confidence": self.ai_min_conf.value(),
                              "cooldown_seconds": self.ai_cooldown.value(), "timeout": self.ai_timeout.value(),
                              "alert_if_unavailable": self.ai_fallback.isChecked()})
        c["learning"].update({"enabled": self.learn.isChecked(), "auto_train": self.auto_train.isChecked(),
                              "min_new_samples": self.min_new.value(), "min_new_samples_cls": self.min_new_cls.value(),
                              "epochs": self.epochs.value(), "cls_epochs": self.cls_epochs.value(),
                              "cls_trust_conf": self.trust.value(), "correction_weight": self.weight.value(),
                              "require_improvement": self.require_impr.isChecked()})
        c["alerts"].update({"cooldown_seconds": self.alert_cd.value(), "sound": self.sound.isChecked(),
                            "tray_notifications": self.tray_notif.isChecked(),
                            "camera_offline_seconds": self.offline_s.value(), "retention_days": self.retention.value()})
        c["ui"].update({"display_fps": self.disp_fps.value(), "grid_sub_stream": self.grid_sub.isChecked(),
                        "autostart": self.autostart.isChecked(), "close_to_tray": self.close_tray.isChecked(),
                        "start_with_windows": self.win_start.isChecked()})
        self.accept()
