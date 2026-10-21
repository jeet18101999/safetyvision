"""People: recognised people (by body), naming, merging, timeline and route map."""
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import config as C
from .. import places as P
from ..events import day_start
from .icons import glyph_icon
from .widgets import bgr_to_pixmap

ROUTE_COLORS = ["#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#ff7b72", "#e3b341"]


def _fmt(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))


class RouteCanvas(QWidget):
    """Floor plan of one place with the person's route through the cameras (numbered, in time order)."""

    def __init__(self):
        super().__init__()
        self.pm = None
        self.stops: list[dict] = []  # {x, y, label, t}
        self.cams: list[dict] = []  # other camera markers {x, y, label}
        self.title = ""
        self.setMinimumHeight(320)

    def set_data(self, map_path, stops, cams, title):
        self.pm = QPixmap(str(map_path)) if map_path and Path(map_path).exists() else None
        self.stops, self.cams, self.title = stops, cams, title
        self.update()

    def _rect(self):
        r = QRectF(self.rect()).adjusted(8, 28, -8, -8)
        if not self.pm or self.pm.isNull():
            return r
        s = min(r.width() / self.pm.width(), r.height() / self.pm.height())
        w, h = self.pm.width() * s, self.pm.height() * s
        return QRectF(r.x() + (r.width() - w) / 2, r.y() + (r.height() - h) / 2, w, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#010409"))
        p.setPen(QColor("#8b949e"))
        p.drawText(QRectF(10, 4, self.width() - 20, 20), Qt.AlignmentFlag.AlignLeft, self.title)
        r = self._rect()
        if self.pm and not self.pm.isNull():
            p.drawPixmap(r.toRect(), self.pm)
            p.fillRect(r, QColor(0, 0, 0, 70))
        else:
            p.setPen(QPen(QColor("#21262d"), 1))
            p.drawRect(r)
        if not self.stops:
            p.setPen(QColor("#6e7681"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "No route to draw on this plan.\nAdd floor plans and place the cameras on the Site map to see routes here.")
            return
        pt = lambda s: QPointF(r.x() + s["x"] * r.width(), r.y() + s["y"] * r.height())  # noqa: E731
        for c in self.cams:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#30363d"))
            p.drawEllipse(pt(c), 7, 7)
        path = QPainterPath(pt(self.stops[0]))
        for s in self.stops[1:]:
            path.lineTo(pt(s))
        p.setPen(QPen(QColor("#58a6ff"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        f = QFont(self.font())
        f.setPointSizeF(8)
        f.setBold(True)
        p.setFont(f)
        placed = {}
        for k, s in enumerate(self.stops):
            c = pt(s)
            off = placed.get((round(c.x()), round(c.y())), 0)
            placed[(round(c.x()), round(c.y()))] = off + 1
            p.setPen(QPen(QColor("#0d1117"), 2))
            p.setBrush(QColor("#3fb950" if k == 0 else "#f85149" if k == len(self.stops) - 1 else "#58a6ff"))
            p.drawEllipse(c, 11, 11)
            p.setPen(QColor("#0d1117"))
            p.drawText(QRectF(c.x() - 11, c.y() - 11, 22, 22), Qt.AlignmentFlag.AlignCenter, str(k + 1))
            txt = f"{s['label']}  {s['t']}"
            tw = p.fontMetrics().horizontalAdvance(txt) + 8
            box = QRectF(c.x() + 13, c.y() - 9 + off * 18, tw, 17)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(13, 17, 23, 225))
            p.drawRoundedRect(box, 3, 3)
            p.setPen(QColor("#e6edf3"))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, txt)


class _Bridge(QObject):
    trained = Signal(str)


class PeoplePage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.pid = None
        self._rows = []
        self._b = _Bridge()
        self._b.trained.connect(self._on_trained)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        head = QHBoxLayout()
        tl = QVBoxLayout()
        t = QLabel("People")
        t.setObjectName("pageTitle")
        tl.addWidget(t)
        self.sub = QLabel("")
        self.sub.setObjectName("muted")
        self.sub.setWordWrap(True)
        tl.addWidget(self.sub)
        head.addLayout(tl, 1)
        self.train_btn = QPushButton("  Train recognition")
        self.train_btn.setIcon(glyph_icon("train"))
        self.train_btn.setToolTip("Learn the body shape / appearance of your named people (better name tags)")
        self.train_btn.clicked.connect(self._train)
        head.addWidget(self.train_btn)
        lay.addLayout(head)
        bar = QHBoxLayout()
        self.filter = QComboBox()
        for label, key in (("To name", "unnamed"), ("Photos to check", "review"), ("Named", "named"),
                           ("All people", "all")):
            self.filter.addItem(label, key)
        self.filter.currentIndexChanged.connect(self.load)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search name ...")
        self.search.textChanged.connect(self.load)
        self.day = QComboBox()
        for label, d in (("Today", 0), ("Yesterday", 1), ("Last 7 days", 7), ("Last 30 days", 30)):
            self.day.addItem(label, d)
        self.day.currentIndexChanged.connect(self._load_person)
        for w in (QLabel("Show:"), self.filter, self.search, QLabel("Movements:"), self.day):
            bar.addWidget(w)
        bar.addStretch()
        lay.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid.setIconSize(QSize(96, 160))
        self.grid.setGridSize(QSize(128, 214))
        self.grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid.setMovement(QListWidget.Movement.Static)
        self.grid.setWordWrap(True)
        self.grid.itemSelectionChanged.connect(self._select)
        split.addWidget(self.grid)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        top = QHBoxLayout()
        self.photo = QLabel()
        self.photo.setFixedSize(110, 200)
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo.setStyleSheet("background:#010409; border:1px solid #21262d; border-radius:6px")
        top.addWidget(self.photo)
        info = QVBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("Type this person's name and press Enter")
        self.name.returnPressed.connect(self._save_name)
        f = self.name.font()
        f.setPointSize(13)
        self.name.setFont(f)
        info.addWidget(self.name)
        self.facts = QLabel("")
        self.facts.setObjectName("muted")
        self.facts.setWordWrap(True)
        info.addWidget(self.facts)
        btns = QHBoxLayout()
        save = QPushButton("  Save name")
        save.setObjectName("primary")
        save.setIcon(glyph_icon("check", "#ffffff"))
        save.clicked.connect(self._save_name)
        merge = QPushButton("  Same person as ...")
        merge.setIcon(glyph_icon("people"))
        merge.setToolTip("This entry and another one are the same person: merge them (the AI learns from it)")
        merge.clicked.connect(self._merge)
        delete = QPushButton("  Delete")
        delete.setIcon(glyph_icon("delete"))
        delete.clicked.connect(self._delete)
        for b in (save, merge, delete):
            btns.addWidget(b)
        btns.addStretch()
        info.addLayout(btns)
        top.addLayout(info, 1)
        rl.addLayout(top)
        self.tabs = QTabWidget()
        # ---- photos: the operator decides what belongs to this person (only confirmed photos are used)
        pw = QWidget()
        pl = QVBoxLayout(pw)
        self.photo_hint = QLabel("")
        self.photo_hint.setWordWrap(True)
        pl.addWidget(self.photo_hint)
        self.photos = QListWidget()
        self.photos.setViewMode(QListWidget.ViewMode.IconMode)
        self.photos.setIconSize(QSize(64, 128))
        self.photos.setGridSize(QSize(86, 170))
        self.photos.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.photos.setMovement(QListWidget.Movement.Static)
        self.photos.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        pl.addWidget(self.photos, 1)
        prow = QHBoxLayout()
        b_conf = QPushButton("  Confirm - this is the person")
        b_conf.setObjectName("primary")
        b_conf.setIcon(glyph_icon("check", "#ffffff"))
        b_conf.setToolTip("Selected photos really show this person (Enter)")
        b_conf.clicked.connect(self._confirm_photos)
        b_move = QPushButton("  Move to ...")
        b_move.setIcon(glyph_icon("people"))
        b_move.setToolTip("Selected photos show someone else - move them to the right person (or a new person)")
        b_move.clicked.connect(self._move_photos)
        b_del = QPushButton("  Delete")
        b_del.setObjectName("danger")
        b_del.setIcon(glyph_icon("delete", "#ffffff"))
        b_del.setToolTip("Bad photo (two people, blurred, cut off) - remove it (Del)")
        b_del.clicked.connect(self._delete_photos)
        b_sel = QPushButton("Select all 'to check'")
        b_sel.clicked.connect(self._select_pending)
        for b in (b_conf, b_move, b_del, b_sel):
            prow.addWidget(b)
        prow.addStretch()
        pl.addLayout(prow)
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("Delete"), self.photos, activated=self._delete_photos)
        QShortcut(QKeySequence("Return"), self.photos, activated=self._confirm_photos)
        self.tabs.addTab(pw, "Photos")
        self.route_tab = QWidget()
        rtl = QVBoxLayout(self.route_tab)
        self.route_text = QLabel("")
        self.route_text.setWordWrap(True)
        rtl.addWidget(self.route_text)
        self.route_loc = QComboBox()
        self.route_loc.currentIndexChanged.connect(self._draw_route)
        rtl.addWidget(self.route_loc)
        self.route = RouteCanvas()
        rtl.addWidget(self.route, 1)
        self.tabs.addTab(self.route_tab, "Route map")
        tw = QWidget()
        twl = QVBoxLayout(tw)
        self.timeline = QTableWidget(0, 6)
        self.timeline.setHorizontalHeaderLabels(["From", "To", "Time", "Camera", "Room", "Place"])
        self.timeline.verticalHeader().setVisible(False)
        self.timeline.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.timeline.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.timeline.itemSelectionChanged.connect(self._show_path)
        twl.addWidget(self.timeline, 2)
        self.path_view = QLabel("Select a row to see where the person walked in that camera")
        self.path_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.path_view.setMinimumHeight(260)
        self.path_view.setStyleSheet("background:#010409; border:1px solid #21262d; color:#6e7681")
        twl.addWidget(self.path_view, 3)
        self.tabs.addTab(tw, "Timeline && paths")
        rl.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        lay.addWidget(split, 1)
        self._sightings = []

    # ---------------------------------------------------------------- helpers
    @property
    def ident(self):
        return self.win.engine.identity

    def to_name_count(self) -> int:
        if not self.ident:
            return 0
        return sum(1 for p in self.ident.gallery.people.values() if not p.get("name") and p.get("seen", 0) >= 10)

    def _label(self, cam):
        cc = C.camera_cfg(self.win.cfg, cam)
        return cc.get("label") or cam

    # ---------------------------------------------------------------- list
    def load(self):
        if not self.ident:
            self.sub.setText("Person recognition starts with monitoring (turn it on in Settings > Detection).")
            self.grid.clear()
            return
        info = self.ident.head_info
        model = (f"Site-trained recognition v{info['version']}: {info['accuracy']:.0%} on held-out images "
                 f"(generic {info['base']:.0%})") if info else "Using the generic body-recognition model - name people, then Train recognition."
        n = self.to_name_count()
        self.sub.setText(f"People are recognised by body shape and clothing (no face recognition). "
                         f"{n} new people are waiting for a name. {model}")
        mode, q = self.filter.currentData(), self.search.text().strip().lower()
        people = sorted(self.ident.gallery.list(), key=lambda p: -p.get("last_seen", 0))
        rows = []
        for p in people:
            named = bool(p.get("name"))
            if mode == "unnamed" and (named or p.get("seen", 0) < 3):
                continue
            if mode == "named" and not named:
                continue
            if mode == "review" and not self.ident.gallery.photo_counts(p["id"])[1]:
                continue
            if q and q not in (p.get("name") or f"person {p['number']}").lower():
                continue
            rows.append(p)
        cur = self.pid
        self.grid.clear()
        self._rows = rows
        for p in rows:
            label = p.get("name") or f"Person {p['number']}"
            nc, nq = self.ident.gallery.photo_counts(p["id"])
            it = QListWidgetItem(f"{label}\n{self._label(p.get('last_camera', ''))} {_fmt(p.get('last_seen', 0))}"
                                 + (f"\n{nq} photo(s) to check" if nq else ("\n\u2713 checked" if nc else "")))
            it.setForeground(QColor("#79c0ff" if p.get("name") else "#e3b341"))
            if p.get("thumb") and os.path.exists(p["thumb"]):
                it.setIcon(QIcon(p["thumb"]))
            it.setData(Qt.ItemDataRole.UserRole, p["id"])
            self.grid.addItem(it)
            if p["id"] == cur:
                it.setSelected(True)

    def _select(self):
        items = self.grid.selectedItems()
        if items:
            self.pid = items[0].data(Qt.ItemDataRole.UserRole)
            self._load_person()

    # ---------------------------------------------------------------- person detail
    def _load_person(self):
        if not self.ident or not self.pid or self.pid not in self.ident.gallery.people:
            return
        g = self.ident.gallery
        p = g.people[self.pid]
        self.name.setText(p.get("name", ""))
        self.name.setPlaceholderText(f"Person {p['number']} - type a name and press Enter")
        if p.get("thumb") and os.path.exists(p["thumb"]):
            self.photo.setPixmap(QPixmap(p["thumb"]).scaled(108, 198, Qt.AspectRatioMode.KeepAspectRatio,
                                                            Qt.TransformationMode.SmoothTransformation))
        d = self.day.currentData()
        since = day_start() - d * 86400 if d else day_start()
        until = day_start() if d == 1 else None
        self._sightings = g.sightings(self.pid, since, until)
        total = sum(s["t_end"] - s["t_start"] for s in self._sightings)
        cams = sorted({self._label(s["camera"]) for s in self._sightings})
        self.facts.setText(f"First seen {time.strftime('%d %b %H:%M', time.localtime(p['created']))} - last seen "
                           f"{self._label(p.get('last_camera', ''))} at {_fmt(p.get('last_seen', 0))}. "
                           f"{len(self._sightings)} sightings ({total / 60:.0f} min on camera) on: {', '.join(cams) or '-'}")
        self._load_photos()
        self.timeline.setRowCount(len(self._sightings))
        for r, s in enumerate(self._sightings):
            vals = [_fmt(s["t_start"]), _fmt(s["t_end"]), f"{(s['t_end'] - s['t_start']) / 60:.1f} min",
                    self._label(s["camera"]), s.get("room") or "-", s.get("location") or "-"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.timeline.setItem(r, c, it)
        # route summary + floor plans the route crosses
        steps = []
        for s in self._sightings:
            lab = self._label(s["camera"])
            if not steps or steps[-1][0] != lab:
                steps.append((lab, _fmt(s["t_start"])))
        self.route_text.setText(" → ".join(f"<b>{a}</b> {b}" for a, b in steps[:40]) or "No movements in this period.")
        plans = []  # floor plans the person's cameras are placed on, in order of first visit
        for s in self._sightings:
            mn = C.camera_cfg(self.win.cfg, s["camera"]).get("map_node")
            if mn and mn not in plans and P.get(self.win.cfg, mn):
                plans.append(mn)
        cur = self.route_loc.currentData()
        self.route_loc.blockSignals(True)
        self.route_loc.clear()
        for mn in plans:
            self.route_loc.addItem(P.path_text(self.win.cfg, mn), mn)
        self.route_loc.setCurrentIndex(max(0, self.route_loc.findData(cur)))
        self.route_loc.blockSignals(False)
        self._draw_route()

    # ---------------------------------------------------------------- photos (operator-curated training data)
    def _load_photos(self):
        g = self.ident.gallery
        p = g.people.get(self.pid, {})
        self.photos.clear()
        items = g.photos(self.pid)
        nc = sum(1 for _, s in items if s == "confirmed")
        nq = len(items) - nc
        for f, status in items:
            it = QListWidgetItem("\u2713 confirmed" if status == "confirmed" else "to check")
            it.setIcon(QIcon(str(f)))
            it.setForeground(QColor("#3fb950" if status == "confirmed" else "#e3b341"))
            it.setData(Qt.ItemDataRole.UserRole, str(f))
            it.setToolTip(time.strftime("%d %b %H:%M:%S", time.localtime(int(f.stem) / 1000)) if f.stem.isdigit() else "")
            self.photos.addItem(it)
        who = p.get("name") or f"Person {p.get('number', '')}"
        state = ("recognised only from your confirmed photos" if p.get("verified")
                 else "confirm at least 3 photos so recognition stops learning by itself")
        self.photo_hint.setText(f"<b>{nc} confirmed</b>, <b style='color:#e3b341'>{nq} to check</b> - {who} is {state}. "
                                "Photos are only <i>sorted</i> automatically; only photos you confirm are used for "
                                "recognition and training. Move photos of someone else to the right person, delete bad "
                                "ones (two people, blurred, cut off).")
        self.tabs.setTabText(0, f"Photos ({nq} to check)" if nq else "Photos")

    def _selected_photos(self):
        return [i.data(Qt.ItemDataRole.UserRole) for i in self.photos.selectedItems()]

    def _select_pending(self):
        for k in range(self.photos.count()):
            it = self.photos.item(k)
            it.setSelected("confirmed" not in it.text())

    def _after_photo_change(self, changed, msg):
        self.ident.apply_photo_changes(changed)
        self.win.toast(msg)
        self._load_photos()
        self.load()

    def _confirm_photos(self):
        paths = self._selected_photos()
        if self.ident and self.pid and paths:
            changed = self.ident.gallery.move_photos(paths, self.pid, "confirmed")
            self._after_photo_change(changed, f"{len(paths)} photo(s) confirmed")

    def _delete_photos(self):
        paths = self._selected_photos()
        if self.ident and paths:
            changed = self.ident.gallery.delete_photos(paths)
            self._after_photo_change(changed, f"{len(paths)} photo(s) deleted")

    def _move_photos(self):
        paths = self._selected_photos()
        if not self.ident or not paths:
            return
        g = self.ident.gallery
        others = sorted((p for p in g.list() if p["id"] != self.pid),
                        key=lambda p: (not p.get("name"), (p.get("name") or "").lower(), -p.get("last_seen", 0)))
        labels = ["+ New person ..."] + [f"{p.get('name') or 'Person ' + str(p['number'])}   "
                                         f"(last seen {self._label(p.get('last_camera', ''))} {_fmt(p.get('last_seen', 0))})"
                                         for p in others]
        choice, ok = QInputDialog.getItem(self, "Move photos", f"The {len(paths)} selected photo(s) show:", labels, 0, True)
        if not ok or not choice:
            return
        if choice == labels[0]:
            name, ok = QInputDialog.getText(self, "New person", "Name (optional):")
            if not ok:
                return
            img = cv2.imread(paths[0])
            target = g.new_person(self.ident.model.backbone([img])[0], img)
            if name.strip():
                g.rename(target, name.strip())
        else:
            target = others[labels.index(choice) - 1]["id"]
        changed = g.move_photos(paths, target, "confirmed")
        self._after_photo_change(changed, f"{len(paths)} photo(s) moved to {g.label(target)}")

    def _draw_route(self, *_):
        loc = self.route_loc.currentData()
        cfg = self.win.cfg
        locd = P.get(cfg, loc)
        stops, cams = [], []
        for name, cc in (cfg.get("cameras") or {}).items():
            if loc and cc.get("map_node") == loc and cc.get("map_pos"):
                cams.append({"x": cc["map_pos"][0], "y": cc["map_pos"][1], "label": cc.get("label") or name})
        for s in self._sightings:
            cc = C.camera_cfg(cfg, s["camera"])
            if loc and cc.get("map_node") == loc and cc.get("map_pos"):
                if stops and stops[-1]["cam"] == s["camera"]:
                    continue
                stops.append({"x": cc["map_pos"][0], "y": cc["map_pos"][1], "label": cc.get("label") or s["camera"],
                              "t": _fmt(s["t_start"])[:5], "cam": s["camera"]})
        self.route.set_data(C.resolve(locd["map"]) if locd and locd.get("map") else None, stops, cams,
                            f"{P.path_text(cfg, loc) if locd else 'No floor plan'} - {len(stops)} stops")

    def _show_path(self):
        rows = self.timeline.selectionModel().selectedRows()
        if not rows:
            return
        s = self._sightings[rows[0].row()]
        e = self.win.engine
        idx = next((i for i, st in enumerate(e.streams) if st.name == s["camera"]), None)
        frame = e.snapshot(idx) if idx is not None else None
        if frame is None:
            self.path_view.setText("No current image from this camera")
            return
        img = frame.copy()
        H, W = img.shape[:2]
        pts = [(int(x * W), int(y * H)) for _, x, y in s["path"]]
        for k in range(1, len(pts)):  # colour goes from green (start) to red (end)
            a = k / max(1, len(pts) - 1)
            cv2.line(img, pts[k - 1], pts[k], (int(60 * (1 - a)), int(200 * (1 - a) + 60 * a), int(60 + 180 * a)), 4, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[0], 10, (60, 200, 60), -1)
            cv2.circle(img, pts[-1], 10, (40, 40, 235), -1)
        cv2.putText(img, f"{self._label(s['camera'])}  {_fmt(s['t_start'])} - {_fmt(s['t_end'])}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        pm = bgr_to_pixmap(img)
        self.path_view.setPixmap(pm.scaled(self.path_view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))

    # ---------------------------------------------------------------- actions
    def _save_name(self):
        if not self.ident or not self.pid:
            return
        name = self.name.text().strip()
        clash = next((k for k, p in self.ident.gallery.people.items() if k != self.pid and p.get("name") == name), None)
        if name and clash and QMessageBox.question(
                self, "Same name", f"'{name}' already exists. Is this the same person? (Yes = merge them)") \
                == QMessageBox.StandardButton.Yes:
            self.ident.gallery.merge(clash, self.pid)
            self.pid = clash
        else:
            self.ident.gallery.rename(self.pid, name)
        self.win.toast(f"Saved: {name or 'unnamed'} - the name tag now follows this person on every camera")
        self.load()

    def _merge(self):
        if not self.ident or not self.pid:
            return
        others = [p for p in self.ident.gallery.list() if p["id"] != self.pid]
        labels = [f"{p.get('name') or 'Person ' + str(p['number'])}  (last seen {_fmt(p.get('last_seen', 0))})" for p in others]
        choice, ok = QInputDialog.getItem(self, "Same person", "This is the same person as:", labels, 0, False)
        if ok and choice:
            other = others[labels.index(choice)]
            keep, drop = (other["id"], self.pid) if other.get("name") else (self.pid, other["id"])
            self.ident.gallery.merge(keep, drop)
            self.pid = keep
            self.load()
            self._load_person()

    def _delete(self):
        if self.ident and self.pid and QMessageBox.question(
                self, "Delete person", "Delete this person and all their sightings?") == QMessageBox.StandardButton.Yes:
            self.ident.gallery.delete(self.pid)
            self.pid = None
            self.load()

    def _train(self):
        if not self.ident:
            return
        self.train_btn.setEnabled(False)
        self.train_btn.setText("  Training ...")
        threading.Thread(target=lambda: self._b.trained.emit(
            self.ident.train_head(status=lambda s: None)["message"]), daemon=True).start()

    def _on_trained(self, msg):
        self.train_btn.setEnabled(True)
        self.train_btn.setText("  Train recognition")
        QMessageBox.information(self, "Person recognition", msg)
        self.load()
