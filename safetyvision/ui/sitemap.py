"""Site map: floor plans of sites / buildings / floors / rooms with the cameras placed on them (live status)."""
import shutil
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem,
                               QMenu, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from .. import config as C
from .. import places as P
from ..xvr import source_list
from .icons import glyph_icon

MAPS_DIR = lambda: C.DATA_DIR / "maps"  # noqa: E731


def store_floor_plan(node, src) -> None:
    """Copy a floor-plan image into data/maps and attach it to a place node."""
    MAPS_DIR().mkdir(parents=True, exist_ok=True)
    dst = MAPS_DIR() / f"{node['id']}-{uuid.uuid4().hex[:6]}{Path(src).suffix.lower()}"
    shutil.copy2(src, dst)
    node["map"] = str(dst.relative_to(C.ROOT)) if dst.is_relative_to(C.ROOT) else str(dst)


class MapCanvas(QWidget):
    moved = Signal(str, float, float)  # camera, x, y (normalised)
    clicked = Signal(float, float)
    removed = Signal(str)
    open_camera = Signal(str)

    def __init__(self):
        super().__init__()
        self.pm: QPixmap | None = None
        self.markers: list[dict] = []  # {name, label, x, y, status, people, alert}
        self._drag = None
        self.setMouseTracking(True)
        self.setMinimumSize(600, 420)

    def set_map(self, path):
        self.pm = QPixmap(str(path)) if path and Path(path).exists() else None
        self.update()

    def _rect(self) -> QRectF:
        if not self.pm or self.pm.isNull():
            return QRectF(self.rect()).adjusted(10, 10, -10, -10)
        s = min(self.width() / self.pm.width(), self.height() / self.pm.height())
        w, h = self.pm.width() * s, self.pm.height() * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _to_norm(self, pos):
        r = self._rect()
        return (min(1, max(0, (pos.x() - r.x()) / r.width())), min(1, max(0, (pos.y() - r.y()) / r.height())))

    def _at(self, pos):
        r = self._rect()
        for m in reversed(self.markers):
            p = QPointF(r.x() + m["x"] * r.width(), r.y() + m["y"] * r.height())
            if (p - pos).manhattanLength() < 22:
                return m
        return None

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#010409"))
        r = self._rect()
        if self.pm and not self.pm.isNull():
            p.drawPixmap(r.toRect(), self.pm)
        else:
            p.setPen(QPen(QColor("#161b22"), 1))
            for k in range(0, int(r.width()), 40):
                p.drawLine(QPointF(r.x() + k, r.y()), QPointF(r.x() + k, r.bottom()))
            for k in range(0, int(r.height()), 40):
                p.drawLine(QPointF(r.x(), r.y() + k), QPointF(r.right(), r.y() + k))
            p.setPen(QColor("#6e7681"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "No floor plan yet - click 'Floor plan...' to add a picture of this area\n"
                       "(a drawing, a photo of the evacuation plan, or a screenshot from Google Maps)")
        blink = int(time.time() * 2) % 2 == 0
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        f.setBold(True)
        p.setFont(f)
        for m in self.markers:
            c = QPointF(r.x() + m["x"] * r.width(), r.y() + m["y"] * r.height())
            col = QColor({"online": "#3fb950", "disabled": "#6e7681"}.get(m["status"], "#d29922"))
            if m["status"] == "offline":
                col = QColor("#f85149")
            if m["alert"]:
                col = QColor("#ff2d2d") if blink else QColor("#ffb3b3")
                p.setPen(Qt.PenStyle.NoPen)
                glow = QColor(255, 45, 45, 70)
                p.setBrush(glow)
                p.drawEllipse(c, 26, 26)
            p.setPen(QPen(QColor("#0d1117"), 2))
            p.setBrush(col)
            p.drawEllipse(c, 12, 12)
            p.setPen(QPen(QColor("#0d1117"), 2))  # small camera lens
            p.setBrush(QColor("#0d1117"))
            p.drawEllipse(c, 4, 4)
            text = m["label"] + (f"  {m['people']}\U0001F464" if m["people"] else "")
            tw = p.fontMetrics().horizontalAdvance(text) + 10
            box = QRectF(c.x() - tw / 2, c.y() + 15, tw, 18)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(13, 17, 23, 220))
            p.drawRoundedRect(box, 4, 4)
            p.setPen(QColor("#e6edf3"))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, e):
        m = self._at(e.position())
        if e.button() == Qt.MouseButton.RightButton and m:
            menu = QMenu(self)
            menu.addAction("Open live view", lambda: self.open_camera.emit(m["name"]))
            menu.addAction("Remove from map", lambda: self.removed.emit(m["name"]))
            menu.exec(e.globalPosition().toPoint())
            return
        if e.button() == Qt.MouseButton.LeftButton:
            if m:
                self._drag = m
            else:
                x, y = self._to_norm(e.position())
                self.clicked.emit(x, y)

    def mouseMoveEvent(self, e):
        m = self._at(e.position())
        if self._drag:
            self._drag["x"], self._drag["y"] = self._to_norm(e.position())
            self.update()
        self.setCursor(Qt.CursorShape.OpenHandCursor if m or self._drag else Qt.CursorShape.CrossCursor)
        self.setToolTip(f"{m['label']} ({m['name']}) - {m['status']}, {m['people'] or 0} people" if m else "")

    def mouseReleaseEvent(self, _):
        if self._drag:
            self.moved.emit(self._drag["name"], self._drag["x"], self._drag["y"])
            self._drag = None

    def mouseDoubleClickEvent(self, e):
        m = self._at(e.position())
        if m:
            self.open_camera.emit(m["name"])


class SiteMapPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        t = QLabel("Site map")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        sub = QLabel("Pick a site, building, floor or room, add its floor plan and place each camera on it. Markers "
                     "show live status and people count, and flash red on an alert. Create and arrange places on the "
                     "Cameras page.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Place:"))
        self.loc = QComboBox()
        self.loc.setMinimumWidth(260)
        self.loc.currentIndexChanged.connect(self._load_location)
        bar.addWidget(self.loc)
        for text, icon, fn in (("Floor plan...", "folder", self._set_map), ("Remove plan", "delete", self._clear_map),
                               ("Manage places...", "cameras", self._manage)):
            b = QPushButton("  " + text)
            b.setIcon(glyph_icon(icon))
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)
        body = QHBoxLayout()
        self.canvas = MapCanvas()
        self.canvas.clicked.connect(self._place)
        self.canvas.moved.connect(self._moved)
        self.canvas.removed.connect(self._remove)
        self.canvas.open_camera.connect(self._open)
        body.addWidget(self.canvas, 1)
        side = QVBoxLayout()
        side.addWidget(QLabel("Cameras - select one, then click on the plan"))
        self.cams = QListWidget()
        self.cams.setMinimumWidth(280)
        self.cams.setMaximumWidth(360)
        side.addWidget(self.cams, 1)
        hint = QLabel("White = in this place, green = on this plan, grey = elsewhere (placing it moves it here). "
                      "Drag a marker to move it, right-click to remove it, double-click to open the live view.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        side.addWidget(hint)
        body.addLayout(side)
        lay.addLayout(body, 1)

    # ---------------------------------------------------------------- data helpers
    @property
    def cfg(self):
        return self.win.cfg

    def _cur(self):
        return P.get(self.cfg, self.loc.currentData())

    def _cam_set(self, name, **kv):
        self.cfg.setdefault("cameras", {}).setdefault(name, {}).update(kv)

    def _commit(self):
        self.win.save_config()
        self.win.engine.update_config(self.cfg)
        self.win.cameras_changed()

    # ---------------------------------------------------------------- page
    def load(self):
        cur = self.loc.currentData()
        self.loc.blockSignals(True)
        self.loc.clear()
        for nid, label, typ in P.options(self.cfg):
            n = P.get(self.cfg, nid)
            self.loc.addItem(label + ("" if n.get("map") else "   (no plan)"), nid)
        if cur is None:  # first time: the first place that has a plan
            cur = next((n["id"] for _, n in P.walk(self.cfg) if n.get("map")), None)
        self.loc.setCurrentIndex(max(0, self.loc.findData(cur)))
        self.loc.blockSignals(False)
        self._load_location()

    def _load_location(self, *_):
        node = self._cur()
        self.canvas.set_map(C.resolve(node["map"]) if node and node.get("map") else None)
        self.cams.clear()
        for s in source_list(self.cfg):
            cc = C.camera_cfg(self.cfg, s["name"])
            here = node is not None and P.camera_in(self.cfg, cc.get("place"), node["id"])
            placed = node is not None and cc.get("map_node") == node["id"] and cc.get("map_pos")
            label = cc.get("label") or s["name"]
            where = cc.get("place_short") or "not assigned"
            it = QListWidgetItem(f"{'[on map] ' if placed else ''}{label}   ({s['name']}, {where})")
            it.setToolTip(cc.get("place_path") or "Not assigned to a place")
            it.setData(Qt.ItemDataRole.UserRole, s["name"])
            it.setForeground(QColor("#3fb950" if placed else "#e6edf3" if here else "#8b949e"))
            self.cams.addItem(it)
        self.refresh()

    def refresh(self):
        node = self._cur()
        stats = {c["name"]: c for c in self.win.engine.stats()["cameras"]}
        idx = {s.name: i for i, s in enumerate(self.win.engine.streams)}
        markers = []
        if node:
            for s in source_list(self.cfg):
                cc = C.camera_cfg(self.cfg, s["name"])
                if cc.get("map_node") != node["id"] or not cc.get("map_pos"):
                    continue
                st = stats.get(s["name"], {})
                i = idx.get(s["name"])
                markers.append({"name": s["name"], "label": cc.get("label") or s["name"],
                                "x": cc["map_pos"][0], "y": cc["map_pos"][1],
                                "status": st.get("status", "stopped" if cc["enabled"] else "disabled"),
                                "people": st.get("people"), "alert": i is not None and self.win.engine.is_flashing(i)})
        self.canvas.markers = markers
        self.canvas.update()

    # ---------------------------------------------------------------- actions
    def _manage(self):
        self.win.show_cameras_page()

    def _set_map(self):
        node = self._cur()
        if not node:
            QMessageBox.information(self, "Floor plan", "Create a site / building / floor / room on the Cameras page first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Floor plan for {node['name']}", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        store_floor_plan(node, path)
        self._commit()
        self.load()

    def _clear_map(self):
        node = self._cur()
        if not node or not node.get("map"):
            return
        if QMessageBox.question(self, "Remove plan", f"Remove the floor plan of '{node['name']}'? Camera markers on "
                                                     "it are removed too.") != QMessageBox.StandardButton.Yes:
            return
        node["map"] = ""
        for cc in (self.cfg.get("cameras") or {}).values():
            if cc.get("map_node") == node["id"]:
                cc["map_node"], cc["map_pos"] = "", None
        self._commit()
        self.load()

    def _place(self, x, y):
        node = self._cur()
        items = self.cams.selectedItems()
        if not node:
            QMessageBox.information(self, "Site map", "Create places on the Cameras page first.")
            return
        if not node.get("map"):
            QMessageBox.information(self, "Site map", "Add a floor plan for this place first ('Floor plan...').")
            return
        if not items:
            return
        name = items[0].data(Qt.ItemDataRole.UserRole)
        kv = {"map_node": node["id"], "map_pos": [round(x, 4), round(y, 4)]}
        if not P.camera_in(self.cfg, C.camera_cfg(self.cfg, name).get("place"), node["id"]):
            kv["place"] = node["id"]  # a camera on this plan is in this place
        self._cam_set(name, **kv)
        self._commit()
        row = self.cams.currentRow()
        self._load_location()
        if row + 1 < self.cams.count():  # next camera ready to place
            self.cams.setCurrentRow(row + 1)

    def _moved(self, name, x, y):
        self._cam_set(name, map_pos=[round(x, 4), round(y, 4)])
        self._commit()

    def _remove(self, name):
        self._cam_set(name, map_pos=None, map_node="")
        self._commit()
        self._load_location()

    def _open(self, name):
        idx = next((i for i, s in enumerate(self.win.engine.streams) if s.name == name), None)
        if idx is not None:
            self.win.focus_camera(idx)
