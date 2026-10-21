"""Application pages: Dashboard, Live, Events (review), Learning. (Cameras: cameras_page.py)"""
import math
import os
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import config as C
from .. import places as P
from ..engine import CRITICAL, EVENT_LABELS
from ..events import day_start
from ..trainer import all_tasks, load_history, model_version, task_classes, task_title
from ..xvr import build_sources
from .icons import glyph_icon
from .widgets import (EVENT_COLORS, STATUS_COLORS, BarList, CameraTile, LineChart, StatCard, TileRenderer,
                      fill_place_combo)


def _item(text, color=None, align=None):
    it = QTableWidgetItem("" if text is None else str(text))
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if color:
        it.setForeground(QColor(color))
    if align:
        it.setTextAlignment(align)
    return it


def _title(text, sub=""):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 6)
    lay.setSpacing(0)
    t = QLabel(text)
    t.setObjectName("pageTitle")
    lay.addWidget(t)
    if sub:
        s = QLabel(sub)
        s.setObjectName("muted")
        s.setWordWrap(True)
        lay.addWidget(s)
    return w


def _btn(text, icon=None, kind=None, tip=""):
    b = QPushButton(text)
    if icon:
        b.setIcon(glyph_icon(icon, "#ffffff" if kind in ("primary", "danger") else "#c9d1d9"))
    if kind:
        b.setObjectName(kind)
    if tip:
        b.setToolTip(tip)
    return b


def open_path(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True) if not p.suffix else None
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))


def event_item(ev, cfg=None) -> QListWidgetItem:
    t = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
    cam = ev.get("cam_label") or (C.camera_cfg(cfg, ev["camera"]).get("label") if cfg else None) or ev["camera"]
    it = QListWidgetItem(f"{t}  {cam}  -  {EVENT_LABELS.get(ev['type'], ev['type']).upper()}\n{ev['message']}")
    it.setForeground(QColor(EVENT_COLORS.get(ev["type"], "#e6edf3")))
    it.setData(Qt.ItemDataRole.UserRole, ev)
    if ev.get("snapshot") and os.path.exists(ev["snapshot"]):
        it.setIcon(QIcon(QPixmap(ev["snapshot"]).scaled(160, 90, Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation)))
    return it


# ============================================================================ Dashboard
class DashboardPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        head = QHBoxLayout()
        head.addWidget(_title("Dashboard", "Site overview - live counts, alerts and how the AI is learning."), 1)
        lt = _btn("Load test...", "grid", tip="Measure how many cameras this PC can analyse")
        lt.clicked.connect(self._load_test)
        head.addWidget(lt)
        lay.addLayout(head)
        cards = QGridLayout()
        cards.setSpacing(10)
        self.c_people = StatCard("People on site", "0", "#58a6ff")
        self.c_cams = StatCard("Cameras online", "0/0", "#3fb950")
        self.c_alerts = StatCard("Alerts today", "0", "#f85149")
        self.c_review = StatCard("Waiting for your review", "0", "#e3b341", "mark them in Events to teach the AI")
        self.c_ai = StatCard("Local AI", "-", "#8b949e")
        self.c_models = StatCard("Learned models active", "0", "#a371f7")
        for k, c in enumerate((self.c_people, self.c_cams, self.c_alerts, self.c_review, self.c_ai, self.c_models)):
            cards.addWidget(c, 0, k)
        lay.addLayout(cards)
        mid = QHBoxLayout()
        p1 = QFrame()
        p1.setObjectName("panel")
        l1 = QVBoxLayout(p1)
        self.chart = LineChart("People on site - last 24 hours")
        l1.addWidget(self.chart)
        p2 = QFrame()
        p2.setObjectName("panel")
        l2 = QVBoxLayout(p2)
        self.bars = BarList("Alerts today by type")
        l2.addWidget(self.bars)
        l2.addStretch()
        p3 = QFrame()
        p3.setObjectName("panel")
        l3 = QVBoxLayout(p3)
        self.loc_bars = BarList("People by place (now)")
        l3.addWidget(self.loc_bars)
        l3.addStretch()
        mid.addWidget(p1, 3)
        mid.addWidget(p2, 2)
        mid.addWidget(p3, 2)
        lay.addLayout(mid, 2)
        bottom = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Recent alerts (double-click to review)"))
        self.recent = QListWidget()
        self.recent.setIconSize(QSize(120, 68))
        self.recent.itemDoubleClicked.connect(lambda it: win.show_event(it.data(Qt.ItemDataRole.UserRole)["id"]))
        left.addWidget(self.recent)
        right = QVBoxLayout()
        right.addWidget(QLabel("Camera health"))
        self.health = QTableWidget(0, 6)
        self.health.setHorizontalHeaderLabels(["Recorder", "Camera", "Place", "Status", "People", "Analysis fps"])
        self.health.verticalHeader().setVisible(False)
        self.health.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.health.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.health.cellDoubleClicked.connect(lambda r, _c: win.focus_camera(r))
        right.addWidget(self.health)
        bottom.addLayout(left, 3)
        bottom.addLayout(right, 2)
        lay.addLayout(bottom, 3)
        self._last_chart = 0

    def _load_test(self):
        from .tools import LoadTestDialog
        LoadTestDialog(self.win).exec()

    def refresh(self, s):
        ev = self.win.engine.events
        cams = s["cameras"]
        dup = s.get("detections", s["total_people"]) - s["total_people"]
        self.c_people.set(s["total_people"], sub=(f"unique - {s.get('detections')} detections on cameras" if dup > 0 else "unique people"))
        recs = s.get("recorders", [])
        self.c_cams.set(f"{s['online']}/{s['enabled']}",
                        "#3fb950" if s["enabled"] and s["online"] == s["enabled"] else "#d29922",
                        "  |  ".join(f"{r['name']}: {r['online']}/{r['enabled']}" for r in recs)[:70]
                        if len(recs) > 1 else (f"{len(cams) - s['enabled']} disabled" if len(cams) > s["enabled"] else ""))
        today = day_start()
        by = ev.counts_by_type(today)
        self.c_alerts.set(sum(v for k, v in by.items() if k != "CAMERA_OFFLINE"))
        self.c_review.set(ev.unreviewed_count(time.time() - 7 * 86400), sub="last 7 days - teach the AI in Events")
        v = s["verifier"]
        if v["online"] is None:
            self.c_ai.set("off" if v["status"] == "disabled" else "...", "#8b949e", "")
        else:
            self.c_ai.set("online" if v["online"] else "offline", "#3fb950" if v["online"] else "#f85149",
                          f"{v['latency']}s per check, queue {v['pending']}")
        n_models = len(s["learned_models"]) + (1 if model_version("fire", self.win.cfg) else 0)
        self.c_models.set(f"{n_models}/{len(all_tasks())}", sub=(s["training"] or "")[:60])
        self.health.setRowCount(len(cams))
        for r, c in enumerate(cams):
            self.health.setItem(r, 0, _item(c.get("recorder", "")))
            self.health.setItem(r, 1, _item(c["label"]))
            self.health.setItem(r, 2, _item(c.get("place_path") or "-"))
            self.health.setItem(r, 3, _item(c["status"], STATUS_COLORS.get(c["status"])))
            self.health.setItem(r, 4, _item("-" if c["people"] is None else c["people"]))
            self.health.setItem(r, 5, _item(c["fps"] if c["status"] == "online" else "-"))
        by_loc = {}
        for c in cams:
            if c["people"] is not None:
                k = c.get("place_short") or "Not assigned"
                by_loc[k] = by_loc.get(k, 0) + c["people"]
        self.loc_bars.set_rows([(k, n, "#58a6ff") for k, n in sorted(by_loc.items(), key=lambda kv: -kv[1])])
        if time.time() - self._last_chart > 30:
            self._last_chart = time.time()
            now = time.time()
            self.chart.set_data(ev.occupancy_series(now - 86400, 900), (now - 86400, now))
            self.bars.set_rows([(EVENT_LABELS.get(k, k), n, EVENT_COLORS.get(k, "#8b949e"))
                                for k, n in sorted(by.items(), key=lambda kv: -kv[1])])
            self.recent.clear()
            for e in ev.recent(30):
                self.recent.addItem(event_item(e, self.win.cfg))

    def add_event(self, ev):
        self.recent.insertItem(0, event_item(ev))
        self._last_chart = 0


# ============================================================================ Live
class LivePage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.tiles: list[CameraTile] = []
        self.focused = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        bar = QHBoxLayout()
        bar.addWidget(_title("Live"))
        bar.addStretch()
        self.cols = QComboBox()
        self.cols.addItem("Auto layout", 0)
        for n in (1, 2, 3, 4, 5, 6):
            self.cols.addItem(f"{n} per row", n)
        self.cols.setCurrentIndex(max(0, self.cols.findData(win.cfg["ui"].get("grid_columns", 0))))
        self.cols.currentIndexChanged.connect(self._layout_changed)
        self.show_boxes = QCheckBox("Boxes")
        self.show_boxes.setChecked(win.cfg["ui"].get("show_boxes", True))
        self.show_heads = QCheckBox("Head pointers")
        self.show_heads.setChecked(win.cfg["ui"].get("show_heads", True))
        for cb in (self.show_boxes, self.show_heads):
            cb.toggled.connect(self._toggles_changed)
        self.btn_all = _btn("All cameras", "grid")
        self.btn_all.clicked.connect(lambda: self.focus(None))
        self.rec_filter = QComboBox()
        self.rec_filter.setToolTip("Show the cameras of one recorder")
        self.rec_filter.currentIndexChanged.connect(lambda _: self.build_tiles())
        self.loc_filter = QComboBox()
        self.loc_filter.setToolTip("Show the cameras of one site / building / floor / room")
        self.loc_filter.currentIndexChanged.connect(lambda _: self.build_tiles())
        self.ai_btn = QPushButton("  AI analysis: ON")
        self.ai_btn.setCheckable(True)
        self.ai_btn.setChecked(True)
        self.ai_btn.setToolTip("Turn off to only stream the cameras (full frame rate, no detection and no alerts)")
        self.ai_btn.toggled.connect(self._toggle_ai)
        on = win.cfg["detection"].get("person_detection", True)
        self.people_btn = QPushButton("  People: ON" if on else "  People: OFF")
        self.people_btn.setCheckable(True)
        self.people_btn.setChecked(on)
        self.people_btn.setIcon(glyph_icon("people"))
        self.people_btn.setToolTip("Person detection: counting, man down, PPE, restricted zones and name tags. "
                                   "Fire detection keeps running when this is off.")
        self.people_btn.toggled.connect(self._toggle_people)
        from PySide6.QtWidgets import QProgressBar, QSlider
        self.listen_label = QLabel("")
        self.listen_label.setObjectName("muted")
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(90)
        self.volume.setToolTip("Camera audio volume")
        self.volume.valueChanged.connect(lambda v: self.win.audio.set_volume(v / 100))
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setFixedSize(70, 10)
        self._loud_at, self._listen_since = 0.0, 0.0
        self.win.audio.level.connect(self._on_level)
        self.win.audio.state.connect(self._audio_state)
        for w in (self.listen_label, self.level, self.volume):
            bar.addWidget(w)
        for w in (self.ai_btn, self.people_btn, self.loc_filter, self.rec_filter, self.show_boxes, self.show_heads,
                  self.cols, self.btn_all):
            bar.addWidget(w)
        lay.addLayout(bar)
        self.ai_banner = QLabel("AI analysis is paused - cameras are only streamed. No fire, man-down, PPE or zone "
                                "alerts until you switch it back on.")
        self.ai_banner.setStyleSheet("background:#5a1d1d; color:#ffd7d5; padding:6px 10px; border-radius:6px")
        self.ai_banner.hide()
        lay.addWidget(self.ai_banner)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(8)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QLabel("Press  ▶ Start monitoring  in the sidebar.\n\n"
                                  "Configure the XVR in Settings and choose detections per camera on the Cameras page.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color:#7d8590; font-size:12pt")
        self.grid.addWidget(self.placeholder, 0, 0)
        self.scroll.setWidget(self.grid_host)
        self.scroll.viewport().installEventFilter(self)
        self._cols = 1
        self._fitting = False
        split.addWidget(self.scroll)
        side = QWidget()
        side.setMinimumWidth(300)
        side.setMaximumWidth(420)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 0, 0, 0)
        self.people_card = StatCard("People on site", "0", "#58a6ff")
        sl.addWidget(self.people_card)
        sl.addWidget(QLabel("Live alerts - select one and mark it"))
        self.feed = QListWidget()
        self.feed.setIconSize(QSize(112, 63))
        self.feed.itemDoubleClicked.connect(lambda it: win.show_event(it.data(Qt.ItemDataRole.UserRole)["id"]))
        sl.addWidget(self.feed, 1)
        row = QHBoxLayout()
        ok = _btn("Correct", "check", "primary", "The alert was right (C)")
        bad = _btn("False", "cancel", "danger", "False alarm - the AI learns from this (F)")
        notp = _btn("Not a person", "person_off", tip="The 'person' was an object - it will be ignored (N)")
        ok.clicked.connect(lambda: self._review("correct"))
        bad.clicked.connect(lambda: self._review("false"))
        notp.clicked.connect(lambda: self._review("not_person"))
        for b in (ok, bad, notp):
            row.addWidget(b)
        sl.addLayout(row)
        split.addWidget(side)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)

    def _review(self, verdict):
        items = self.feed.selectedItems()
        if not items:
            return
        ev = items[0].data(Qt.ItemDataRole.UserRole)
        self.win.review(ev["id"], verdict)
        items[0].setText(items[0].text().split("\n")[0] + f"  [{verdict.replace('_', ' ')}]\n" +
                         items[0].text().split("\n", 1)[-1])
        items[0].setForeground(QColor("#6e7681"))

    def _toggles_changed(self):
        self.win.cfg["ui"]["show_boxes"] = self.show_boxes.isChecked()
        self.win.cfg["ui"]["show_heads"] = self.show_heads.isChecked()
        self.win.save_config()

    def _layout_changed(self):
        self.win.cfg["ui"]["grid_columns"] = self.cols.currentData()
        self.win.save_config()
        self.build_tiles()

    def build_tiles(self):
        if getattr(self, "renderer", None):
            self.renderer.targets = {}
        self.tiles = []
        self.focused = None
        self._new_grid()
        streams = self.win.engine.streams
        recs = sorted({getattr(s, "recorder", "") for s in streams})
        cur = self.rec_filter.currentData()
        self.rec_filter.blockSignals(True)
        self.rec_filter.clear()
        self.rec_filter.addItem("All recorders", None)
        for r in recs:
            self.rec_filter.addItem(r or "Cameras", r)
        self.rec_filter.setCurrentIndex(max(0, self.rec_filter.findData(cur)))
        self.rec_filter.setVisible(len(recs) > 1)
        self.rec_filter.blockSignals(False)
        self.loc_filter.setVisible(fill_place_combo(self.loc_filter, self.win.cfg, "All places"))
        want = self.rec_filter.currentData()
        want_loc = self.loc_filter.currentData()
        shown = [(i, s) for i, s in enumerate(streams)
                 if (want is None or getattr(s, "recorder", "") == want)
                 and P.camera_in(self.win.cfg, self.win.engine.cam_cfg(i).get("place"), want_loc)]
        self.placeholder.setVisible(not shown)
        if not shown:
            return
        order = self._order()
        shown.sort(key=lambda x: order.index(x[1].name) if x[1].name in order else len(order) + x[0])
        n = len(shown)
        cols = max(1, min(self.cols.currentData() or math.ceil(math.sqrt(n)), n))
        self._cols = cols
        for k, (i, s) in enumerate(shown):
            tile = CameraTile(i, self.win.engine.cam_label(i))
            tile.double_clicked.connect(self.focus)
            tile.listen.connect(self._listen)
            tile.moved.connect(self._swap)
            tile.menu_requested.connect(self._tile_menu)
            tile.set_listening(self.win.audio.camera == s.name)
            self.grid.addWidget(tile, k // cols, k % cols)
            self.tiles.append(tile)
        self._fit_tiles()

    # ---------------------------------------------------------------- layout: tiles always fit the window
    def _new_grid(self):
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()  # a fresh layout: no stale column widths from a previous layout
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(8)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.placeholder = QLabel("No cameras to show - check the filters, or start monitoring.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color:#7d8590; font-size:12pt")
        self.grid.addWidget(self.placeholder, 0, 0)
        self.scroll.setWidget(self.grid_host)

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is self.scroll.viewport() and ev.type() == QEvent.Type.Resize:
            self._fit_tiles()
        return False

    def _fit_tiles(self):
        if self._fitting or not self.tiles:
            return
        self._fitting = True
        try:
            vp = self.scroll.viewport()
            vw, vh = self.scroll.width() - 2, self.scroll.height() - 2
            sp = self.grid.spacing()
            if self.focused is not None:
                for tile in self.tiles:
                    if tile.index == self.focused:
                        tile.setFixedSize(max(160, vw), max(120, vh))
                return
            cols = self._cols
            rows = math.ceil(len(self.tiles) / cols)
            head = 34
            tw = (vw - sp * (cols - 1)) / cols
            th = (tw - 8) * 9 / 16 + head
            if rows * th + sp * (rows - 1) > vh:  # a vertical scrollbar will appear: leave room for it
                tw = (vw - self.scroll.style().pixelMetric(self.scroll.style().PixelMetric.PM_ScrollBarExtent)
                      - sp * (cols - 1)) / cols
                th = (tw - 8) * 9 / 16 + head
            for tile in self.tiles:
                tile.setFixedSize(max(120, int(tw)), max(90, int(th)))
        finally:
            self._fitting = False

    # ---------------------------------------------------------------- camera order (drag & drop / move to position)
    def _order(self) -> list[str]:
        saved = list(self.win.cfg["ui"].get("tile_order") or [])
        names = [s.name for s in self.win.engine.streams]
        return [n for n in saved if n in names] + [n for n in names if n not in saved]

    def _swap(self, src, dst):
        e = self.win.engine
        order = self._order()
        a, b = e.streams[src].name, e.streams[dst].name
        ia, ib = order.index(a), order.index(b)
        order[ia], order[ib] = order[ib], order[ia]
        self.win.cfg["ui"]["tile_order"] = order
        self.win.save_config()
        self.build_tiles()
        self.win.toast(f"Swapped {e.cam_label(src)} and {e.cam_label(dst)}")

    def _tile_menu(self, idx, pos):
        from PySide6.QtWidgets import QInputDialog, QMenu
        e = self.win.engine
        m = QMenu(self)
        m.addAction("Open full screen" if self.focused != idx else "Back to all cameras", lambda: self.focus(idx))
        m.addAction("Listen" if self.win.audio.camera != e.streams[idx].name else "Stop listening",
                    lambda: self._listen(idx))

        def move_to():
            cur = next(k for k, tile in enumerate(self.tiles) if tile.index == idx) + 1
            pos_, ok = QInputDialog.getInt(self, "Move camera", f"Move {e.cam_label(idx)} to position "
                                           f"(1-{len(self.tiles)}). The camera there takes its old place:",
                                           cur, 1, len(self.tiles))
            if ok and pos_ != cur:
                self._swap(idx, self.tiles[pos_ - 1].index)

        m.addAction("Move to position ...", move_to)
        m.addSeparator()
        m.addAction("Reset camera order", self._reset_order)
        m.exec(pos)

    def _reset_order(self):
        self.win.cfg["ui"]["tile_order"] = []
        self.win.save_config()
        self.build_tiles()

    def refresh_filters(self):
        if self.win.engine.streams:
            self.build_tiles()

    def focus(self, idx):
        if idx is not None and not any(t.index == idx for t in self.tiles):
            self.loc_filter.setCurrentIndex(0)  # camera hidden by a filter: show everything first
            self.rec_filter.setCurrentIndex(0)
        if idx is None or self.focused == idx:
            self.focused = None
            for t in self.tiles:
                t.setVisible(True)
        else:
            self.focused = idx
            for t in self.tiles:
                t.setVisible(t.index == idx)
        self._fit_tiles()

    def _listen(self, idx):
        e = self.win.engine
        s = e.streams[idx]
        if self.win.audio.camera == s.name:
            self.win.audio.stop()
        else:
            self.win.audio.play(s.name, s.source)
        for t in self.tiles:
            t.set_listening(self.win.audio.camera == e.streams[t.index].name)

    def _on_level(self, v):
        self.level.setValue(int(min(1.0, v * 3) * 100))
        now = time.time()
        if v > 0.004:
            if self._loud_at == 0 and self.listen_label.text().endswith("microphone?)"):
                self.listen_label.setText(self.listen_label.text().split("  (")[0])
            self._loud_at = now
        elif self._listen_since and now - max(self._loud_at, self._listen_since) > 5                 and not self.listen_label.text().endswith("microphone?)"):
            self.listen_label.setText(self.listen_label.text() + "  (no sound - camera may have no microphone?)")

    def _audio_state(self, cam, state):
        label = next((self.win.engine.cam_label(i) for i, s in enumerate(self.win.engine.streams) if s.name == cam), cam)
        if state == "stopped":
            self.listen_label.setText("")
            self.level.setValue(0)
        elif state == "playing":
            self._loud_at, self._listen_since = 0.0, time.time()
            self.listen_label.setText(f"\U0001F50A {label}")
        else:
            self.listen_label.setText(f"\U0001F507 {label}: {state}")

    def _toggle_people(self, on):
        self.people_btn.setText("  People: ON" if on else "  People: OFF")
        self.win.engine.set_person_detection(on)
        self.win.save_config()
        self.win.toast("Person detection on" if on else
                       "Person detection OFF - no people counting, man-down, PPE or zone alerts (fire still on)")

    def sync_people_toggle(self):
        on = self.win.cfg["detection"].get("person_detection", True)
        if self.people_btn.isChecked() != on:
            self.people_btn.blockSignals(True)
            self.people_btn.setChecked(on)
            self.people_btn.setText("  People: ON" if on else "  People: OFF")
            self.people_btn.blockSignals(False)

    def _toggle_ai(self, on):
        if not on and QMessageBox.question(
                self, "Pause AI analysis?", "Cameras keep streaming at full frame rate, but there will be NO fire, "
                "man-down, PPE or zone alerts until you switch AI analysis back on.\n\nPause AI analysis?") \
                != QMessageBox.StandardButton.Yes:
            self.ai_btn.blockSignals(True)
            self.ai_btn.setChecked(True)
            self.ai_btn.blockSignals(False)
            return
        self.win.engine.set_ai(on)
        self.ai_btn.setText("  AI analysis: ON" if on else "  AI analysis: OFF (view only)")
        self.ai_banner.setVisible(not on)
        self.win.ai_state_changed(on)

    def visible_indexes(self):
        vp = self.scroll.viewport().rect()
        out = set()
        for t in self.tiles:
            if t.isVisible():
                top_left = t.mapTo(self.scroll.viewport(), t.rect().topLeft())
                if vp.intersects(t.rect().translated(top_left)):
                    out.add(t.index)
        return out

    def refresh_frames(self):
        e = self.win.engine
        if getattr(self, "renderer", None) is None or self.renderer.engine is not e:
            self.renderer = TileRenderer(e)
        fps = self.win.cfg["ui"].get("display_fps", 25)
        visible = self.visible_indexes() if self.isVisible() else set()
        use_sub = self.win.cfg["ui"].get("grid_sub_stream", True) and self.focused is None
        small = {tile.index for tile in self.tiles if tile.index in visible and tile.view.width() <= 600}
        e.set_view(visible, fps, small if use_sub else set())
        sb, sh = self.show_boxes.isChecked(), self.show_heads.isChecked()
        self.renderer.period = 1.0 / max(1, fps)
        self.renderer.targets = {t.index: (t.view.width(), t.view.height(), sb, sh) for t in self.tiles
                                 if t.index in visible and t.view.width() > 8}
        for t in self.tiles:
            if t.index not in visible:
                continue
            img = self.renderer.take(t.index)
            if img is not None:
                t.set_frame(img)
            t.set_alert(e.is_flashing(t.index))

    def refresh(self, s):
        self.people_card.set(s["total_people"])
        for t in self.tiles:
            if t.index < len(s["cameras"]):
                c = s["cameras"][t.index]
                t.set_info(c, C.camera_cfg(self.win.cfg, c["name"]))

    def add_event(self, ev):
        self.feed.insertItem(0, event_item(ev))
        while self.feed.count() > 200:
            self.feed.takeItem(self.feed.count() - 1)


# ============================================================================ Events
class EventsPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._events = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.addWidget(_title("Events", "Review every alert: Correct or False alarm. Each review is used to "
                                       "retrain the matching model, and false alarms at the same spot are suppressed "
                                       "immediately.  Shortcuts: C = correct, F = false alarm, N = not a person."))
        top = QHBoxLayout()
        self.f_type = QComboBox()
        self.f_type.addItem("All types", None)
        for k, v in EVENT_LABELS.items():
            self.f_type.addItem(v, k)
        self.f_cam = QComboBox()
        self.f_review = QComboBox()
        for label, data in (("All", None), ("Needs review", "unreviewed"), ("Marked correct", "correct"),
                            ("Marked false", "false"), ("Not a person", "not_person")):
            self.f_review.addItem(label, data)
        self.f_range = QComboBox()
        for label, days in (("Today", 0), ("Last 7 days", 7), ("Last 30 days", 30), ("All time", None)):
            self.f_range.addItem(label, days)
        self.f_range.setCurrentIndex(1)
        self.f_loc = QComboBox()
        for cb in (self.f_type, self.f_loc, self.f_cam, self.f_review, self.f_range):
            cb.currentIndexChanged.connect(self.load)
            top.addWidget(cb)
        top.addStretch()
        exp = _btn("Export CSV", "export")
        exp.clicked.connect(self._export)
        folder = _btn("Snapshots", "folder")
        folder.clicked.connect(lambda: open_path(C.DATA_DIR / "snapshots"))
        top.addWidget(exp)
        top.addWidget(folder)
        lay.addLayout(top)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Time", "Camera", "Type", "AI", "Review", "Details"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._show)
        split.addWidget(self.table)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        self.preview = QLabel("Select an event")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(420, 260)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.setStyleSheet("background:#010409; border:1px solid #21262d; border-radius:6px")
        rl.addWidget(self.preview, 3)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rl.addWidget(self.detail)
        row = QHBoxLayout()
        self.b_ok = _btn("Correct", "check", "primary", "The alert was right (C)")
        self.b_false = _btn("False alarm", "cancel", "danger", "Wrong alert - the model learns from it (F)")
        self.b_np = _btn("Not a person", "person_off", tip="The detected 'person' is an object; ignore it from now on (N)")
        self.b_ok.clicked.connect(lambda: self._review("correct"))
        self.b_false.clicked.connect(lambda: self._review("false"))
        self.b_np.clicked.connect(lambda: self._review("not_person"))
        for b in (self.b_ok, self.b_false, self.b_np):
            row.addWidget(b)
        rl.addLayout(row)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setStyleSheet("color:#3fb950")
        rl.addWidget(self.result)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)
        for key, verdict in (("C", "correct"), ("F", "false"), ("N", "not_person")):
            QShortcut(QKeySequence(key), self, activated=lambda v=verdict: self._review(v))

    def refresh_cameras(self):
        self.f_loc.setVisible(fill_place_combo(self.f_loc, self.win.cfg, "All places"))
        cur = self.f_cam.currentData()
        self.f_cam.blockSignals(True)
        self.f_cam.clear()
        self.f_cam.addItem("All cameras", None)
        for name, _ in build_sources(self.win.cfg):
            self.f_cam.addItem(C.camera_cfg(self.win.cfg, name).get("label") or name, name)
        self.f_cam.setCurrentIndex(max(0, self.f_cam.findData(cur)))
        self.f_cam.blockSignals(False)

    def _since(self):
        d = self.f_range.currentData()
        return None if d is None else (day_start() if d == 0 else time.time() - d * 86400)

    def load(self, select_id=None):
        keep = select_id or self._selected_id()
        self._events = self.win.engine.events.recent(2000, self.f_type.currentData(), self.f_cam.currentData(),
                                                     self.f_review.currentData(), self._since())
        loc = self.f_loc.currentData()
        if loc:
            self._events = [e for e in self._events
                            if P.camera_in(self.win.cfg, C.camera_cfg(self.win.cfg, e["camera"]).get("place"), loc)]
        self.table.setRowCount(len(self._events))
        labels = {}
        for n, _ in build_sources(self.win.cfg):
            cc = C.camera_cfg(self.win.cfg, n)
            labels[n] = (cc.get("label") or n) + (f"  ({cc['place_short']})" if cc.get("place_short") else "")
        for r, e in enumerate(self._events):
            self._fill_row(r, e, labels)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        if keep:
            for r, e in enumerate(self._events):
                if e["id"] == keep:
                    self.table.selectRow(r)
                    break

    def _fill_row(self, r, e, labels=None):
        color = EVENT_COLORS.get(e["type"])
        ai = {True: "verified", False: "rejected", None: "-"}[e["verified"]]
        fb = {"correct": ("correct", "#3fb950"), "false": ("false alarm", "#f85149"),
              "not_person": ("not a person", "#d29922")}.get(e["feedback"], ("needs review", "#e3b341"))
        if e["type"] in ("CAMERA_OFFLINE", "SYSTEM"):
            fb = ("-", None)
        self.table.setItem(r, 0, _item(time.strftime("%d %b %H:%M:%S", time.localtime(e["ts"]))))
        self.table.setItem(r, 1, _item((labels or {}).get(e["camera"], e["camera"])))
        self.table.setItem(r, 2, _item(EVENT_LABELS.get(e["type"], e["type"]), color))
        self.table.setItem(r, 3, _item(ai))
        self.table.setItem(r, 4, _item(*fb))
        self.table.setItem(r, 5, _item(e["message"]))

    def _selected_id(self):
        rows = self.table.selectionModel().selectedRows()
        return self._events[rows[0].row()]["id"] if rows and rows[0].row() < len(self._events) else None

    def _show(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        e = self._events[rows[0].row()]
        if e["snapshot"] and os.path.exists(e["snapshot"]):
            self.preview.setPixmap(QPixmap(e["snapshot"]).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                                 Qt.TransformationMode.SmoothTransformation))
        else:
            self.preview.clear()
            self.preview.setText("No snapshot")
        x = e["extra"]
        extra = ""
        if x.get("ai_reason"):
            extra = f"<br><b>Local AI:</b> {x['ai_reason']}"
        self.detail.setText(f"<b>{EVENT_LABELS.get(e['type'], e['type'])}</b> - {e['message']}{extra}")
        reviewable = e["type"] != "CAMERA_OFFLINE"
        for b in (self.b_ok, self.b_false):
            b.setEnabled(reviewable)
        self.b_np.setEnabled(e["type"] in ("MAN_DOWN", "PPE", "INTRUSION", "OVERCROWD"))
        self.result.setText("")

    def _review(self, verdict):
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self.isVisible():
            return
        r = rows[0].row()
        e = self._events[r]
        if e["type"] == "CAMERA_OFFLINE" or (verdict == "not_person" and not self.b_np.isEnabled()):
            return
        e["feedback"] = verdict
        self._fill_row(r, e)
        self.win.review(e["id"], verdict, self.result)
        for k in range(r + 1, len(self._events)):  # jump to the next event that still needs review
            if self._events[k]["feedback"] is None and self._events[k]["type"] != "CAMERA_OFFLINE":
                self.table.selectRow(k)
                break

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export events", f"safetyvision_events_{time.strftime('%Y%m%d')}.csv",
                                              "CSV (*.csv)")
        if path:
            n = self.win.engine.events.export_csv(path, self._since() or 0)
            self.win.toast(f"Exported {n} events to {path}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.table.selectionModel() and self.table.selectionModel().selectedRows():
            self._show()


# ============================================================================ Learning
class _Bridge(QObject):
    done = Signal(str)


class LearningPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.task = "fire"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        head = QHBoxLayout()
        sim = _btn("Fire simulation...", "train", tip="Train the fire detector on simulated fires on your own cameras")
        sim.clicked.connect(self._fire_sim)
        lay.addLayout(head)
        head_title_holder = head
        head_title_holder.addStretch()
        head_title_holder.addWidget(sim)
        lay.addWidget(_title("AI Learning", "Every model learns continuously from the local AI's verdicts and from "
                                            "your reviews. A new version is deployed only when it scores at least as "
                                            "well as the current one on the same validation images."))
        self.models = QTableWidget(0, 6)
        self.models.setHorizontalHeaderLabels(["Model", "Active version", "Samples", "New / next training",
                                               "Last result", ""])
        self.models.verticalHeader().setVisible(False)
        self.models.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.models.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.models.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.models.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.models.itemSelectionChanged.connect(self._select_task)
        self.models.setMaximumHeight(330)
        lay.addWidget(self.models)
        tabs = QTabWidget()
        lay.addWidget(tabs, 1)

        # samples
        sw = QWidget()
        sl = QVBoxLayout(sw)
        self.sample_title = QLabel("")
        sl.addWidget(self.sample_title)
        self.samples = QListWidget()
        self.samples.setViewMode(QListWidget.ViewMode.IconMode)
        self.samples.setIconSize(QSize(130, 110))
        self.samples.setGridSize(QSize(158, 168))
        self.samples.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.samples.setMovement(QListWidget.Movement.Static)
        self.samples.setWordWrap(True)
        self.samples.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.samples.itemSelectionChanged.connect(self._show_sample)
        sl.addWidget(self.samples, 1)
        self.sample_info = QLabel("")
        self.sample_info.setWordWrap(True)
        sl.addWidget(self.sample_info)
        self.relabel_row = QHBoxLayout()
        sl.addLayout(self.relabel_row)
        tabs.addTab(sw, "Samples (correct the AI here)")

        # history
        self.hist = QTableWidget(0, 6)
        self.hist.setHorizontalHeaderLabels(["Version", "Time", "Samples", "Score new", "Score before", "Deployed"])
        self.hist.verticalHeader().setVisible(False)
        self.hist.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tabs.addTab(self.hist, "Training history")

        # ignored objects
        iw = QWidget()
        il = QVBoxLayout(iw)
        il.addWidget(QLabel("Spots the AI or you marked as 'not a person' / 'not fire'. Detections there are ignored "
                            "while they stay still. Remove an entry if something real is there now."))
        self.ignored = QListWidget()
        self.ignored.setViewMode(QListWidget.ViewMode.IconMode)
        self.ignored.setIconSize(QSize(110, 110))
        self.ignored.setGridSize(QSize(150, 160))
        self.ignored.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.ignored.setMovement(QListWidget.Movement.Static)
        self.ignored.setWordWrap(True)
        self.ignored.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        il.addWidget(self.ignored, 1)
        rm = _btn("Remove selected", "delete")
        rm.clicked.connect(self._remove_ignored)
        il.addWidget(rm, 0, Qt.AlignmentFlag.AlignLeft)
        tabs.addTab(iw, "Ignored objects")

        # log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        tabs.addTab(self.log, "Training log")
        self._sig = None

    # ---------------------------------------------------------------- models table
    def refresh(self):
        hub = self.win.engine.hub
        tasks = all_tasks()
        sig = []
        self.models.setRowCount(len(tasks))
        for r, task in enumerate(tasks):
            ds = hub.dataset(task)
            c = ds.counts()
            ver = model_version(task, self.win.cfg)
            hist = load_history(task)
            last = hist[-1] if hist else None
            running = hub.running_task == task
            self.models.setItem(r, 0, _item(task_title(task)))
            self.models.setItem(r, 1, _item(ver or ("learning..." if c["total"] else "not trained yet"),
                                            "#3fb950" if ver else "#8b949e"))
            data = _item(",  ".join(f"{k}: {v}" for k, v in c["by_class"].items()))
            if task == "fire":
                nf = self.win.engine.notfire
                anchors = ds.anchor_count()["train"]
                data.setToolTip(f"Trained together with {anchors} reference fire images, so learning 'not fire' does "
                                f"not make it forget fire.\n{nf.count()} 'not fire' look-alikes remembered per camera; "
                                f"{nf.skipped} candidates skipped since start without asking the local AI.")
                if nf.skipped:
                    data.setText(data.text() + f"   ({nf.skipped} look-alikes skipped)")
            self.models.setItem(r, 2, data)
            self.models.setItem(r, 3, _item(f"{c['new_samples']} / {hub.threshold(task)}"
                                            + ("   training now..." if running else "")))
            if last:
                res = f"{last['time']}  {last.get('metric', 'mAP50')} {last['new'].get('score', last['new'].get('map50', 0)):.2f}"
                if last["new"].get("false_alarms") is not None and (last.get("old") or {}).get("false_alarms") is not None:
                    res += f"  false alarms {last['old']['false_alarms']}->{last['new']['false_alarms']}"
                res += "  deployed" if last["accepted"] else "  kept previous"
                self.models.setItem(r, 4, _item(res, "#3fb950" if last["accepted"] else "#d29922"))
            else:
                self.models.setItem(r, 4, _item("-"))
            b = QPushButton("Training..." if running else "Train now")
            b.setEnabled(not running)
            b.clicked.connect(lambda _=False, t=task: self._train(t))
            self.models.setCellWidget(r, 5, b)
            sig.append((c["total"], c["new_samples"], ver, running, len(hist)))
        if self.models.currentRow() < 0:
            self.models.selectRow(0)
        if sig != self._sig:
            self._sig = sig
            self._load_task()

    def _fire_sim(self):
        from .tools import FireSimulationDialog
        FireSimulationDialog(self.win).exec()
        self._sig = None
        self.refresh()

    def _train(self, task):
        ok, msg = self.win.engine.hub.start(task, "manual")
        self.log_line(msg)
        if not ok:
            QMessageBox.information(self, "Training", msg)
        self.refresh()

    def _select_task(self):
        r = self.models.currentRow()
        tasks = all_tasks()
        if 0 <= r < len(tasks) and tasks[r] != self.task:
            self.task = tasks[r]
            self._load_task()

    # ---------------------------------------------------------------- selected task
    def _load_task(self):
        task = self.task
        ds = self.win.engine.hub.dataset(task)
        classes = ["fire", "smoke", "not_fire"] if task == "fire" else task_classes(task)
        self.sample_title.setText(f"<b>{task_title(task)}</b> - newest samples. Select one or more and relabel if the "
                                  f"AI was wrong; your label counts x{self.win.cfg['learning']['correction_weight']}.")
        while self.relabel_row.count():
            w = self.relabel_row.takeAt(0).widget()
            if w:
                w.deleteLater()
        for c in classes:
            b = _btn(f"Mark as '{c.replace('_', ' ')}'", "check",
                     "danger" if c in ("fire", "smoke", "down", "missing", "not_person") else "primary")
            b.clicked.connect(lambda _=False, lab=c: self._relabel(lab))
            self.relabel_row.addWidget(b)
        d = _btn("Delete", "delete")
        d.clicked.connect(self._delete)
        self.relabel_row.addWidget(d)
        self.relabel_row.addStretch()
        self.samples.clear()
        for s in ds.recent(150):
            label = ds.sample_label(s)
            who = "you" if s.get("manual") else {"ai": "local AI", "motion": "motion", "operator": "you"}.get(
                s.get("source"), "local AI")
            it = QListWidgetItem(f"{label.replace('_', ' ').upper()}\n{who} - {s.get('camera', '')} "
                                 f"{time.strftime('%d %b %H:%M', time.localtime(s['ts']))}")
            bad = label in ("fire", "smoke", "down", "missing", "not_person")
            it.setForeground(QColor("#ff7b72" if bad else "#56d364"))
            tp = ds.thumb_path(s["id"]) if task == "fire" else ds.image_path(s)
            if Path(tp).exists():
                it.setIcon(QIcon(str(tp)))
            it.setData(Qt.ItemDataRole.UserRole, s)
            self.samples.addItem(it)
        hist = load_history(task)
        self.hist.setRowCount(len(hist))
        for r, h in enumerate(reversed(hist)):
            score = lambda d: d.get("score", d.get("map50", 0))  # noqa: E731
            self.hist.setItem(r, 0, _item(h["version"]))
            self.hist.setItem(r, 1, _item(h["time"]))
            self.hist.setItem(r, 2, _item(h["samples"]))
            self.hist.setItem(r, 3, _item(f"{score(h['new']):.3f}"))
            self.hist.setItem(r, 4, _item(f"{score(h['old']):.3f}" if h.get("old") else "-"))
            self.hist.setItem(r, 5, _item("yes" if h["accepted"] else "no (kept previous)",
                                          "#3fb950" if h["accepted"] else "#d29922"))
        self._load_ignored()

    def _show_sample(self):
        items = self.samples.selectedItems()
        if items:
            s = items[0].data(Qt.ItemDataRole.UserRole)
            self.sample_info.setText(f"{len(items)} selected.  <b>AI reason:</b> {s.get('ai_reason', '-')}")

    def _relabel(self, label):
        ds = self.win.engine.hub.dataset(self.task)
        w = int(self.win.cfg["learning"]["correction_weight"])
        items = self.samples.selectedItems()
        for it in items:
            ds.relabel(it.data(Qt.ItemDataRole.UserRole)["id"], label, w)
        if items:
            self.win.engine.hub.maybe_train(self.task)
            self.log_line(f"{task_title(self.task)}: {len(items)} sample(s) relabelled as '{label}'")
            self._sig = None
            self.refresh()

    def _delete(self):
        ds = self.win.engine.hub.dataset(self.task)
        for it in self.samples.selectedItems():
            ds.delete(it.data(Qt.ItemDataRole.UserRole)["id"])
        self._sig = None
        self.refresh()

    def _load_ignored(self):
        ig = self.win.engine.ignore
        self.ignored.clear()
        for it in sorted(ig.list(), key=lambda x: -x["ts"]):
            cam = C.camera_cfg(self.win.cfg, it["camera"]).get("label") or it["camera"]
            li = QListWidgetItem(f"not {'a person' if it['kind'] == 'person' else 'fire'}\n{cam} - by {it['source']}")
            if ig.thumb(it).exists():
                li.setIcon(QIcon(str(ig.thumb(it))))
            li.setToolTip(it.get("note", ""))
            li.setData(Qt.ItemDataRole.UserRole, it["id"])
            self.ignored.addItem(li)

    def _remove_ignored(self):
        for li in self.ignored.selectedItems():
            self.win.engine.ignore.remove(li.data(Qt.ItemDataRole.UserRole))
        self._load_ignored()

    def log_line(self, text):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")
