"""Main window: sidebar navigation, pages, system tray, notifications."""
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPushButton, QStackedWidget, QSystemTrayIcon, QVBoxLayout, QWidget)

from .. import config as C
from ..engine import CRITICAL, EVENT_LABELS, Engine
from .icons import app_icon, glyph_icon
from .cameras_page import CamerasPage
from .pages import DashboardPage, EventsPage, LearningPage, LivePage
from .settings_dialog import SettingsDialog
from .people_page import PeoplePage
from .sitemap import SiteMapPage

NAV = [("dashboard", "Dashboard"), ("live", "Live"), ("events", "Events"), ("people", "People"),
       ("cameras", "Cameras"), ("sitemap", "Site map"), ("learning", "AI Learning")]
P_DASH, P_LIVE, P_EVENTS, P_PEOPLE, P_CAMERAS, P_MAP, P_LEARN = range(7)


class _Bridge(QObject):
    review_done = Signal(object, str)


def set_start_with_windows(enabled: bool):
    if sys.platform != "win32":
        return
    import winreg
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                         winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE)
    try:
        if enabled:
            exe = Path(sys.executable)
            pyw = exe.with_name("pythonw.exe") if exe.name.lower() == "python.exe" else exe
            winreg.SetValueEx(key, "SafetyVision", 0, winreg.REG_SZ, f'"{pyw}" "{C.ROOT / "main.py"}" --tray')
        else:
            try:
                winreg.DeleteValue(key, "SafetyVision")
            except FileNotFoundError:
                pass
    finally:
        winreg.CloseKey(key)


class MainWindow(QMainWindow):
    def __init__(self, start_in_tray=False, resume=False):
        super().__init__()
        self.setWindowTitle("SafetyVision")
        self.setWindowIcon(app_icon())
        self.resize(1600, 940)
        self.cfg = C.load_config()
        self.engine = Engine(self.cfg)
        from ..audio import AudioPlayer
        self.audio = AudioPlayer()
        self._notified_people = set()
        self._bridge = _Bridge()
        self._bridge.review_done.connect(self._on_review_done)
        self._quitting = False
        self._tray_hint_shown = False

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        self.dashboard = DashboardPage(self)
        self.live = LivePage(self)
        self.events_page = EventsPage(self)
        self.cameras = CamerasPage(self)
        self.learning = LearningPage(self)
        self.sitemap = SiteMapPage(self)
        self.people = PeoplePage(self)
        for p in (self.dashboard, self.live, self.events_page, self.people, self.cameras, self.sitemap, self.learning):
            self.stack.addWidget(p)
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.status_msg = QLabel("Stopped")
        self.status_ai = QLabel("")
        self.status_dev = QLabel("")
        sb = self.statusBar()
        sb.addWidget(self.status_msg, 1)
        sb.addPermanentWidget(self.status_ai)
        sb.addPermanentWidget(self.status_dev)
        self._build_tray()

        e = self.engine
        e.event_raised.connect(self._on_event)
        e.message.connect(self.status_msg.setText)
        e.training_status.connect(self._on_training)
        e.model_updated.connect(lambda task: self.toast(f"New model deployed: {task}"))
        e.started.connect(self._on_started)
        e.stopped.connect(self._on_stopped)
        e.fatal.connect(self._on_fatal)

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self._refresh_frames)
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._refresh_stats)
        self.stats_timer.start(1000)

        self.cameras.load()
        self.events_page.refresh_cameras()
        self.nav.setCurrentRow(1)
        if "--cpu" in sys.argv:
            self.cfg["detection"]["device"] = "cpu"  # temporary for this run (not saved)
        if start_in_tray or resume or self.cfg["ui"].get("autostart"):
            QTimer.singleShot(300, self.start)
        elif not any(r.get("host") and r.get("password") for r in self.cfg.get("recorders") or []) \
                and not self.cfg.get("extra_sources"):
            QTimer.singleShot(500, self._first_run_hint)
        self._refresh_stats()

    # ------------------------------------------------------------------ layout
    def _build_sidebar(self):
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(220)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(10, 14, 10, 12)
        brand = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(40, 40))
        logo.setStyleSheet("background:transparent")
        names = QVBoxLayout()
        names.setSpacing(0)
        t = QLabel("SafetyVision")
        t.setObjectName("brand")
        s = QLabel("AI safety monitoring")
        s.setObjectName("brandSub")
        names.addWidget(t)
        names.addWidget(s)
        brand.addWidget(logo)
        brand.addLayout(names)
        brand.addStretch()
        lay.addLayout(brand)
        lay.addSpacing(14)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setIconSize(QSize(20, 20))
        for key, label in NAV:
            it = QListWidgetItem(glyph_icon(key, "#c9d1d9"), f"  {label}")
            self.nav.addItem(it)
        self.nav.currentRowChanged.connect(self._on_nav)
        lay.addWidget(self.nav, 1)
        self.review_badge = QLabel("")
        self.review_badge.setObjectName("muted")
        self.review_badge.setWordWrap(True)
        lay.addWidget(self.review_badge)
        self.btn_settings = QPushButton("  Settings")
        self.btn_settings.setIcon(glyph_icon("settings"))
        self.btn_settings.clicked.connect(self.open_settings)
        lay.addWidget(self.btn_settings)
        self.btn_run = QPushButton("  Start monitoring")
        self.btn_run.setObjectName("primary")
        self.btn_run.setIcon(glyph_icon("play", "#ffffff"))
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self._toggle_run)
        lay.addWidget(self.btn_run)
        return side

    def _build_tray(self):
        self.tray = QSystemTrayIcon(app_icon(), self)
        self.tray.setToolTip("SafetyVision")
        menu = QMenu()
        show = QAction("Open SafetyVision", self)
        show.triggered.connect(self._show_window)
        self.tray_run = QAction("Start monitoring", self)
        self.tray_run.triggered.connect(self._toggle_run)
        quit_ = QAction("Quit", self)
        quit_.triggered.connect(self.quit)
        menu.addAction(show)
        menu.addAction(self.tray_run)
        menu.addSeparator()
        menu.addAction(quit_)
        self.tray.setContextMenu(menu)
        self._tray_menu = menu
        self.tray.activated.connect(lambda r: self._show_window() if r in (
            QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick) else None)
        self.tray.messageClicked.connect(lambda: (self._show_window(), self.nav.setCurrentRow(2)))
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _on_nav(self, row):
        self.stack.setCurrentIndex(row)
        if row == P_EVENTS:
            self.events_page.load()
        elif row == P_CAMERAS:
            self.cameras.load()
        elif row == P_MAP:
            self.sitemap.load()
        elif row == P_PEOPLE:
            self.people.load()
        elif row == P_LEARN:
            self.learning.refresh()

    # ------------------------------------------------------------------ helpers used by pages
    def save_config(self):
        C.save_config(self.cfg)

    def toast(self, text):
        self.status_msg.setText(text)

    def show_event(self, event_id):
        self.nav.setCurrentRow(2)
        self.events_page.f_review.setCurrentIndex(0)
        self.events_page.load(select_id=event_id)

    def show_cameras_page(self):
        self.nav.setCurrentRow(P_CAMERAS)

    def focus_camera(self, idx):
        self.nav.setCurrentRow(1)
        self.live.focus(idx)

    def cameras_changed(self):
        self.events_page.refresh_cameras()
        self.live.refresh_filters()
        for t in self.live.tiles:
            t.title.setText(self.engine.cam_label(t.index))

    def review(self, event_id, verdict, result_label=None):
        """Operator review -> learning (runs off the UI thread: image IO + dataset writes)."""
        def work():
            try:
                msg = self.engine.apply_feedback(event_id, verdict)
            except Exception as e:
                msg = f"Could not apply review: {e}"
            self._bridge.review_done.emit(result_label, msg)

        threading.Thread(target=work, daemon=True).start()

    def _on_review_done(self, label, msg):
        if label is not None:
            label.setText(msg)
        self.toast(msg)
        self.learning.log_line(msg)

    # ------------------------------------------------------------------ run control
    def _first_run_hint(self):
        QMessageBox.information(self, "Welcome to SafetyVision",
                                "Let's find your cameras. SafetyVision will search the network for DVR / NVR / "
                                "XVR recorders and IP cameras - enter their login and add them with one click. "
                                "Then name the cameras, put them in their sites / floors / rooms and pick detections on the Cameras page, "
                                "and press Start monitoring. (Recorders can also be added by hand in Settings.)")
        self.nav.setCurrentRow(P_CAMERAS)
        self.cameras._discover()

    def _toggle_run(self):
        if self.engine.running or self.engine.loading:
            self.stop()
        else:
            self.start()

    def start(self):
        self.btn_run.setEnabled(False)
        self.btn_run.setText("  Starting...")
        self.status_msg.setText("Starting...")
        self.engine.start()
        self.live.build_tiles()

    def stop(self):
        self.engine.stop()

    def _on_started(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("  Stop monitoring")
        self.btn_run.setObjectName("danger")
        self.btn_run.setIcon(glyph_icon("stop", "#ffffff"))
        self._repolish(self.btn_run)
        self.tray_run.setText("Stop monitoring")
        self.frame_timer.start(max(15, int(1000 / max(1, self.cfg["ui"].get("display_fps", 25)) / 2)))

    def _on_stopped(self):
        self.frame_timer.stop()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("  Start monitoring")
        self.btn_run.setObjectName("primary")
        self.btn_run.setIcon(glyph_icon("play", "#ffffff"))
        self._repolish(self.btn_run)
        self.tray_run.setText("Start monitoring")
        self.status_msg.setText("Stopped")

    @staticmethod
    def _repolish(w):
        w.style().unpolish(w)
        w.style().polish(w)

    def open_settings(self):
        old = self.cfg
        dlg = SettingsDialog(self.cfg, self)
        dlg.setWindowIcon(app_icon())
        if not dlg.exec():
            return
        self.cfg = dlg.cfg
        self.save_config()
        restart_keys = ("recorders", "extra_sources")
        det_restart = ("device", "imgsz", "person_model", "fire_mode", "batch")
        need_restart = any(old.get(k) != self.cfg.get(k) for k in restart_keys) or \
            any(old["detection"].get(k) != self.cfg["detection"].get(k) for k in det_restart) or \
            old["verifier"] != self.cfg["verifier"]
        if need_restart:
            self.engine.apply_config(self.cfg)
            if self.engine.loading or self.engine.running:
                self.live.build_tiles()
        else:
            self.engine.update_config(self.cfg)
        try:
            set_start_with_windows(self.cfg["ui"].get("start_with_windows", False))
        except Exception as e:
            self.toast(f"Could not update Windows startup: {e}")
        self.cameras.load()
        self.events_page.refresh_cameras()
        self.engine.set_person_detection(self.cfg["detection"].get("person_detection", True))
        self.live.sync_people_toggle()
        self.toast("Settings saved" + (" - monitoring restarted" if need_restart else ""))

    # ------------------------------------------------------------------ refresh
    def _refresh_frames(self):
        if self.stack.currentIndex() == P_LIVE and self.isVisible() and not self.isMinimized():
            self.live.refresh_frames()
        else:
            self.engine.set_view(set(), self.cfg["ui"].get("display_fps", 25))  # nothing on screen

    def ai_state_changed(self, on):
        self.tray.setToolTip("SafetyVision" + ("" if on else " - AI PAUSED (view only)"))
        self.toast("AI analysis resumed" if on else "AI analysis PAUSED - streaming only, no alerts")

    def _refresh_stats(self):
        s = self.engine.stats()
        page = self.stack.currentIndex()
        if page == 0:
            self.dashboard.refresh(s)
        elif page == 1:
            self.live.refresh(s)
        elif page == P_MAP:
            self.sitemap.refresh()
        elif page == P_PEOPLE and int(time.time()) % 10 == 0:
            self.people.load()
        elif page == P_LEARN and int(time.time()) % 3 == 0:
            self.learning.refresh()
        v = s["verifier"]
        self.status_ai.setText(f"Local AI: {v['status']}  |  queue {v['pending']}  |  {v['latency']}s   ")
        low = 0 <= s.get("gpu_free_mb", -1) < 400
        self.status_ai.setStyleSheet("color:#ff7b72" if low else "")
        if low:
            self.status_ai.setText(f"⚠ GPU memory almost full ({s['gpu_free_mb']} MB free) - close other programs "
                                   f"or choose qwen2.5vl:3b in Settings > Local AI   ")
        ai = f"analysing {s['analysis_fps']} frames/s" if s["ai_enabled"] else "AI PAUSED (view only)"
        if not s.get("person_detection", True):
            ai += "  |  people OFF"
        self.status_dev.setText(f"Device {s['device']}  |  {ai}  |  people on site {s['total_people']}  ")
        if int(time.time()) % 5 == 0:
            self._people_notifications()
            n = self.engine.events.unreviewed_count(time.time() - 7 * 86400)
            self.review_badge.setText(f"{n} alert(s) waiting for review in Events" if n else "")
            self.tray.setToolTip(f"SafetyVision - {s['online']}/{s['enabled']} cameras, {s['total_people']} people")

    def _people_notifications(self):
        ident = self.engine.identity
        if not ident:
            return
        n = self.people.to_name_count()
        self.nav.item(P_PEOPLE).setText(f"  People" + (f"   ({n} to name)" if n else ""))
        for pid, p in list(ident.gallery.people.items()):
            if not p.get("name") and p.get("seen", 0) >= 10 and pid not in self._notified_people:
                self._notified_people.add(pid)
                cam = self.engine.cam_label(next((i for i, s in enumerate(self.engine.streams)
                                                  if s.name == p.get("last_camera")), 0)) if self.engine.streams else ""
                self.tray.showMessage("New person seen", f"Person {p['number']} on {cam} - open People to add a name.",
                                      QSystemTrayIcon.MessageIcon.Information, 5000)
                break

    def _on_training(self, task, text):
        self.learning.log_line(f"{task}: {text}")
        if text.startswith(("deployed", "kept", "training failed")):
            self.learning.refresh()

    def _on_event(self, ev):
        self.live.add_event(ev)
        self.dashboard.add_event(ev)
        if self.stack.currentIndex() == 2:
            self.events_page.load()
        a = self.cfg["alerts"]
        if a.get("sound"):
            self._beep(ev["type"])
        if a.get("tray_notifications") and self.tray.isVisible() and ev["type"] != "CAMERA_OFFLINE":
            icon = (QSystemTrayIcon.MessageIcon.Critical if ev["type"] in CRITICAL
                    else QSystemTrayIcon.MessageIcon.Warning)
            where = ev.get("cam_label", ev["camera"]) + (f" ({ev['location']})" if ev.get("location") else "")
            self.tray.showMessage(f"{EVENT_LABELS.get(ev['type'], ev['type'])} - {where}",
                                  ev["message"][:200], icon, 8000)
        if ev["type"] in CRITICAL and self.isVisible():
            self.activateWindow()

    def _beep(self, etype):
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND if etype in CRITICAL else winsound.MB_ICONEXCLAMATION)
        except Exception:
            QApplication.beep()

    def _on_fatal(self, msg):
        """Detection broke (GPU error). Restart the whole app so monitoring continues; fall back to CPU if it
        keeps happening."""
        import json
        import logging
        from PySide6.QtCore import QProcess
        logging.getLogger(__name__).error("detection stopped: %s - restarting", msg)
        state_p = C.DATA_DIR / "restarts.json"
        now = time.time()
        try:
            hist = [t for t in json.loads(state_p.read_text()) if now - t < 900]
        except Exception:
            hist = []
        hist.append(now)
        state_p.write_text(json.dumps(hist))
        args = [str(C.ROOT / "main.py"), "--restart"]
        if len(hist) >= 3:
            args.append("--cpu")  # third GPU failure in 15 min: keep monitoring on the CPU
        try:
            self.engine.events.add("SYSTEM", "SYSTEM", f"Detection restarted after an error: {msg[:120]}")
        except Exception:
            pass
        self.tray.showMessage("SafetyVision is restarting", "Detection hit a GPU error and is restarting automatically.",
                              QSystemTrayIcon.MessageIcon.Warning, 6000)
        exe = sys.executable
        QProcess.startDetached(exe, args, str(C.ROOT))
        self.quit()

    # ------------------------------------------------------------------ window / tray
    def _show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, e):
        if not self._quitting and self.cfg["ui"].get("close_to_tray", True) and self.tray.isVisible():
            e.ignore()
            self.hide()
            if not self._tray_hint_shown:
                self._tray_hint_shown = True
                self.tray.showMessage("SafetyVision is still monitoring",
                                      "Right-click the tray icon to open or quit.",
                                      QSystemTrayIcon.MessageIcon.Information, 4000)
            return
        self._shutdown()
        super().closeEvent(e)

    def quit(self):
        self._quitting = True
        self._shutdown()
        QApplication.quit()

    def _shutdown(self):
        self.audio.stop()
        if self.engine.identity:
            self.engine.identity.gallery.save(force=True)
        self.frame_timer.stop()
        self.stats_timer.stop()
        self.engine.shutdown()
        self.tray.hide()
