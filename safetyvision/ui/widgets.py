"""Reusable widgets and the application style sheet."""
import time

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

STATUS_COLORS = {"online": "#3fb950", "connecting": "#d29922", "reconnecting": "#d29922", "offline": "#f85149",
                 "stopped": "#8b949e", "disabled": "#6e7681"}
EVENT_COLORS = {"FIRE": "#f85149", "SMOKE": "#a371f7", "FIRE_UNVERIFIED": "#d29922", "MAN_DOWN": "#ff4d6d",
                "PPE": "#ff8c1a", "INTRUSION": "#e3b341", "OVERCROWD": "#58a6ff", "CAMERA_OFFLINE": "#8b949e", "SYSTEM": "#8b949e"}
FEATURE_CHIPS = {"people": ("COUNT", "#58a6ff"), "man_down": ("DOWN", "#ff4d6d"), "fire": ("FIRE", "#f0883e"),
                 "ppe": ("PPE", "#e3b341"), "zone": ("ZONE", "#f85149"), "crowd": ("CROWD", "#a371f7")}


def bgr_to_pixmap(img) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def chip(text, color) -> str:
    return (f"<span style='background:{color}22; color:{color}; border-radius:3px;'>&nbsp;{text}&nbsp;</span>")


class TileRenderer:
    """Background threads that resize + annotate camera frames into ready-to-show QImages (OpenCV releases the
    GIL, so this runs in parallel with the UI and the analysis thread)."""

    def __init__(self, engine, workers=2):
        import threading
        self.engine = engine
        self.targets: dict[int, tuple] = {}  # tile index -> (width, height, boxes, heads)
        self.ready: dict[int, QImage] = {}
        self._last: dict[int, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.period = 1 / 25
        self._threads = [threading.Thread(target=self._run, args=(k, workers), daemon=True) for k in range(workers)]
        for th in self._threads:
            th.start()

    def stop(self):
        self._stop.set()

    def take(self, idx):
        with self._lock:
            return self.ready.pop(idx, None)

    def _run(self, k, n):
        while not self._stop.is_set():
            t0 = time.time()
            for idx, (w, h, sb, sh) in list(self.targets.items()):
                if idx % n != k:
                    continue
                try:
                    img, fid = self.engine.render_new(idx, w, h, self._last.get(idx, -1), sb, sh)
                except Exception:
                    continue
                if img is None:
                    continue
                self._last[idx] = fid
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format.Format_RGB888).copy()
                with self._lock:
                    self.ready[idx] = q
            self._stop.wait(max(0.004, self.period - (time.time() - t0)))


class CameraTile(QFrame):
    double_clicked = Signal(int)
    listen = Signal(int)
    moved = Signal(int, int)  # dragged camera index, dropped-on camera index
    menu_requested = Signal(int, object)  # camera index, global position

    def __init__(self, index: int, name: str):
        super().__init__()
        self.index = index
        self.setObjectName("tile")
        self.setAcceptDrops(True)
        self._press = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = QLabel(name)
        self.title.setObjectName("tileTitle")
        self.chips = QLabel("")
        self.chips.setObjectName("tileChips")
        self.badge = QLabel("")
        self.badge.setObjectName("tileBadge")
        from PySide6.QtWidgets import QToolButton
        self.speaker = QToolButton()
        self.speaker.setText("\U0001F508")
        self.speaker.setToolTip("Listen to this camera")
        self.speaker.setAutoRaise(True)
        self.speaker.setCheckable(True)
        self.speaker.clicked.connect(lambda: self.listen.emit(self.index))
        for lbl in (self.title, self.chips, self.badge):  # text is clipped instead of forcing the tile wider
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            lbl.setMinimumWidth(0)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self.speaker)
        head.addWidget(self.title, 3)
        head.addWidget(self.chips, 2)
        head.addWidget(self.badge, 4)
        lay.addLayout(head)
        self.view = QLabel("Connecting...")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.view.setMinimumSize(64, 36)
        self.view.setObjectName("tileView")
        lay.addWidget(self.view, 1)
        self._alert = False
        self._has_frame = False
        self.last_fid = -1
        self._shown = 0
        self._t0 = time.time()
        self.view_fps = 0.0

    def set_frame(self, img):
        if img is not None:
            self.view.setPixmap(QPixmap.fromImage(img) if isinstance(img, QImage) else bgr_to_pixmap(img))
            self._has_frame = True
            self._shown += 1
            dt = time.time() - self._t0
            if dt >= 2:
                self.view_fps, self._shown, self._t0 = self._shown / dt, 0, time.time()

    def set_info(self, cam: dict, features: dict):
        loc = cam.get("place_short") or cam.get("location")
        self.title.setText(cam["label"] + (f"  <span style='color:#7d8590;font-weight:400'>{loc}</span>" if loc else ""))
        self.chips.setText(" ".join(chip(*FEATURE_CHIPS[k]) for k in FEATURE_CHIPS if features.get(k)))
        color = STATUS_COLORS.get(cam["status"], "#8b949e")
        parts = [f"<span style='color:{color}'>&#9679;</span> {cam['status']}"]
        if cam["people"] is not None:
            parts.append(f"&#128100; <b>{cam['people']}</b>")
        if cam["down"]:
            parts.append("<span style='color:#ff4d6d'><b>MAN DOWN</b></span>")
        if cam["ppe"]:
            parts.append(f"<span style='color:#ff8c1a'><b>PPE {cam['ppe']}</b></span>")
        if cam["status"] == "online":
            q = {"sub": "SD", "main": "HD"}.get(cam.get("view", ""), "")
            parts.append(f"{q} {self.view_fps:.0f} fps".strip() + (f" · AI {cam['fps']:.1f}" if cam["fps"] else ""))
        self.badge.setText("  ".join(parts))
        if cam["status"] != "online" and not self._has_frame or cam["status"] in ("offline", "disabled"):
            self.view.clear()
            self._has_frame = False
            self.view.setText({"disabled": "Camera disabled (Cameras page)"}.get(cam["status"], cam["status"].title()))

    def set_listening(self, on: bool):
        self.speaker.setChecked(on)
        self.speaker.setText("\U0001F50A" if on else "\U0001F508")

    def set_alert(self, on: bool):
        if on != self._alert:
            self._alert = on
            self.setProperty("alert", on)
            self.style().unpolish(self)
            self.style().polish(self)

    def mouseDoubleClickEvent(self, e):
        self.double_clicked.emit(self.index)
        super().mouseDoubleClickEvent(e)

    # ---- drag a tile onto another tile to swap their places
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        from PySide6.QtCore import QMimeData
        from PySide6.QtGui import QDrag
        from PySide6.QtWidgets import QApplication
        if self._press is not None and e.buttons() & Qt.MouseButton.LeftButton and \
                (e.position().toPoint() - self._press).manhattanLength() >= QApplication.startDragDistance():
            self._press = None
            drag = QDrag(self)
            md = QMimeData()
            md.setText(f"svtile:{self.index}")
            drag.setMimeData(md)
            pm = self.view.grab()
            if not pm.isNull():
                drag.setPixmap(pm.scaledToWidth(min(200, max(60, pm.width()))))
            drag.exec(Qt.DropAction.MoveAction)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press = None
        super().mouseReleaseEvent(e)

    def _set_drop(self, on):
        self.setProperty("drop", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        if e.mimeData().text().startswith("svtile:") and e.mimeData().text() != f"svtile:{self.index}":
            e.acceptProposedAction()
            self._set_drop(True)

    def dragLeaveEvent(self, e):
        self._set_drop(False)

    def dropEvent(self, e):
        self._set_drop(False)
        try:
            src = int(e.mimeData().text().split(":", 1)[1])
        except ValueError:
            return
        e.acceptProposedAction()
        if src != self.index:
            self.moved.emit(src, self.index)

    def contextMenuEvent(self, e):
        self.menu_requested.emit(self.index, e.globalPos())


class StatCard(QFrame):
    def __init__(self, title: str, value: str = "-", accent="#58a6ff", sub=""):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        self.value = QLabel(value)
        self.value.setObjectName("cardValue")
        self.value.setStyleSheet(f"color:{accent}")
        self.sub = QLabel(sub)
        self.sub.setObjectName("cardSub")
        lay.addWidget(t)
        lay.addWidget(self.value)
        lay.addWidget(self.sub)

    def set(self, v, color=None, sub=None):
        self.value.setText(str(v))
        if color:
            self.value.setStyleSheet(f"color:{color}")
        if sub is not None:
            self.sub.setText(sub)


class LineChart(QWidget):
    """Minimal area/line chart: people on site over time."""

    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.points: list[tuple[float, float]] = []
        self.x_range = (time.time() - 86400, time.time())
        self.setMinimumHeight(180)

    def set_data(self, points, x_range):
        self.points, self.x_range = points, x_range
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(44, 30, -14, -26)
        p.setPen(QColor("#8b949e"))
        f = QFont(self.font())
        f.setPointSizeF(9)
        p.setFont(f)
        p.drawText(QRectF(8, 4, self.width() - 16, 20), Qt.AlignmentFlag.AlignLeft, self.title)
        x0, x1 = self.x_range
        ymax = max([v for _, v in self.points] + [4])
        ymax = int(ymax * 1.15) + 1
        grid = QPen(QColor("#21262d"))
        for k in range(5):
            y = r.bottom() - r.height() * k / 4
            p.setPen(grid)
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor("#6e7681"))
            p.drawText(QRectF(0, y - 8, 38, 16), Qt.AlignmentFlag.AlignRight, str(round(ymax * k / 4)))
        for k in range(7):
            ts = x0 + (x1 - x0) * k / 6
            x = r.left() + r.width() * k / 6
            p.drawText(QRectF(x - 30, r.bottom() + 4, 60, 16), Qt.AlignmentFlag.AlignCenter,
                       time.strftime("%H:%M", time.localtime(ts)))
        if len(self.points) < 1:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "No data yet - occupancy is logged every minute while monitoring")
            return

        def pt(ts, v):
            return QPointF(r.left() + r.width() * (ts - x0) / max(1, x1 - x0), r.bottom() - r.height() * v / ymax)

        line = QPainterPath()
        area = QPainterPath()
        first = pt(*self.points[0])
        line.moveTo(first)
        area.moveTo(QPointF(first.x(), r.bottom()))
        area.lineTo(first)
        for ts, v in self.points[1:]:
            line.lineTo(pt(ts, v))
            area.lineTo(pt(ts, v))
        last = pt(*self.points[-1])
        area.lineTo(QPointF(last.x(), r.bottom()))
        area.closeSubpath()
        c = QColor("#2f81f7")
        fill = QColor(c)
        fill.setAlpha(50)
        p.fillPath(area, fill)
        p.setPen(QPen(c, 2))
        p.drawPath(line)
        p.setBrush(c)
        p.drawEllipse(last, 3.5, 3.5)


class BarList(QWidget):
    """Horizontal bars: events today by type."""

    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.rows: list[tuple[str, int, str]] = []
        self.setMinimumHeight(180)

    def set_rows(self, rows):
        self.rows = rows
        self.setMinimumHeight(40 + 26 * max(1, len(rows)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont(self.font())
        f.setPointSizeF(9)
        p.setFont(f)
        p.setPen(QColor("#8b949e"))
        p.drawText(QRectF(8, 4, self.width() - 16, 20), Qt.AlignmentFlag.AlignLeft, self.title)
        if not self.rows:
            p.drawText(QRectF(self.rect()).adjusted(0, 24, 0, 0), Qt.AlignmentFlag.AlignCenter, "No alerts today")
            return
        mx = max(n for _, n, _ in self.rows) or 1
        y = 32
        lw = 118
        for label, n, color in self.rows:
            p.setPen(QColor("#c9d1d9"))
            p.drawText(QRectF(8, y, lw - 8, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
            bw = (self.width() - lw - 50) * n / mx
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(lw, y + 3, max(3, bw), 14), 3, 3)
            p.setPen(QColor("#c9d1d9"))
            p.drawText(QRectF(lw + bw + 6, y, 40, 20), Qt.AlignmentFlag.AlignVCenter, str(n))
            y += 26


APP_QSS = """
* { font-family:'Segoe UI Variable Text','Segoe UI'; font-size:10pt; }
QWidget { background:#0d1117; color:#e6edf3; }
QWidget#sidebar { background:#010409; border-right:1px solid #21262d; }
QLabel#brand { font-size:13pt; font-weight:700; color:#e6edf3; background:transparent; }
QLabel#brandSub { color:#7d8590; font-size:8.5pt; background:transparent; }
QListWidget#nav { background:transparent; border:none; outline:0; }
QListWidget#nav::item { padding:10px 12px; margin:2px 8px; border-radius:8px; color:#c9d1d9; }
QListWidget#nav::item:hover { background:#161b22; }
QListWidget#nav::item:selected { background:#1f6feb33; color:#58a6ff; }
QLabel#pageTitle { font-size:16pt; font-weight:700; background:transparent; }
QLabel#muted { color:#7d8590; background:transparent; }
QPushButton { background:#21262d; border:1px solid #30363d; border-radius:6px; padding:6px 14px; }
QPushButton:hover { background:#30363d; }
QPushButton:disabled { color:#6e7681; }
QPushButton#primary { background:#238636; border-color:#2ea043; color:white; font-weight:600; }
QPushButton#primary:hover { background:#2ea043; }
QPushButton#danger { background:#b62324; border-color:#f85149; color:white; font-weight:600; }
QPushButton#danger:hover { background:#da3633; }
QPushButton#ghost { background:transparent; border:1px solid #30363d; }
QPushButton:checked { background:#1f6feb44; border-color:#1f6feb; color:#58a6ff; }
QFrame#tile { background:#010409; border:2px solid #21262d; border-radius:8px; }
QFrame#tile[alert="true"] { border:2px solid #f85149; background:#2d0b0b; }
QFrame#tile[drop="true"] { border:2px dashed #58a6ff; background:#0b1d3d; }
QLabel#tileTitle { font-weight:600; background:transparent; }
QLabel#tileChips { font-size:7.5pt; font-weight:700; background:transparent; }
QLabel#tileBadge { color:#8b949e; font-size:8.5pt; background:transparent; }
QLabel#tileView { background:#000; color:#6e7681; border-radius:4px; }
QFrame#card, QFrame#panel { background:#161b22; border:1px solid #21262d; border-radius:10px; }
QLabel#cardTitle { color:#8b949e; font-size:9pt; background:transparent; }
QLabel#cardValue { font-size:22pt; font-weight:700; background:transparent; }
QLabel#cardSub { color:#7d8590; font-size:8.5pt; background:transparent; }
QTableWidget, QListWidget, QPlainTextEdit, QTreeWidget { background:#010409; border:1px solid #21262d; border-radius:6px;
               gridline-color:#161b22; alternate-background-color:#0d1117; selection-background-color:#1f6feb55; }
QListWidget::item:selected, QTableWidget::item:selected, QTreeWidget::item:selected { color:#ffffff; background:#1f6feb66; }
QHeaderView::section { background:#161b22; border:none; border-right:1px solid #21262d; padding:6px; color:#8b949e; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#010409; border:1px solid #30363d; border-radius:6px; padding:5px; }
QComboBox QAbstractItemView { background:#161b22; selection-background-color:#1f6feb55; }
QGroupBox { border:1px solid #21262d; border-radius:10px; margin-top:14px; padding-top:10px; background:#0d1117; }
QGroupBox::title { subcontrol-origin: margin; left:12px; padding:0 4px; color:#8b949e; }
QProgressBar { border:1px solid #30363d; border-radius:5px; text-align:center; background:#010409; height:16px; }
QProgressBar::chunk { background:#238636; border-radius:4px; }
QStatusBar { background:#010409; border-top:1px solid #21262d; color:#8b949e; }
QScrollBar:vertical { background:transparent; width:10px; }
QScrollBar::handle:vertical { background:#30363d; border-radius:5px; min-height:24px; }
QScrollBar:horizontal { background:transparent; height:10px; }
QScrollBar::handle:horizontal { background:#30363d; border-radius:5px; min-width:24px; }
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
QCheckBox { background:transparent; spacing:6px; }
QTableWidget::indicator, QCheckBox::indicator { width:16px; height:16px; border:1px solid #6e7681; border-radius:4px;
               background:#0d1117; }
QTableWidget::indicator:checked, QCheckBox::indicator:checked { background:#238636; border-color:#3fb950; }
QTableWidget::indicator:hover, QCheckBox::indicator:hover { border-color:#58a6ff; }
QTabWidget::pane { border:1px solid #21262d; border-radius:6px; }
QTabBar::tab { background:#161b22; border:1px solid #21262d; padding:7px 16px; margin-right:2px;
               border-top-left-radius:6px; border-top-right-radius:6px; }
QTabBar::tab:selected { background:#0d1117; color:#58a6ff; }
QToolTip { background:#161b22; color:#e6edf3; border:1px solid #30363d; }
QSplitter::handle { background:#21262d; }
"""


PLACE_ICONS = {"site": "\U0001F3ED", "building": "\U0001F3E2", "floor": "\U0001F4F6", "room": "\U0001F6AA"}


def fill_place_combo(combo, cfg, first_text="All places", keep=True):
    """Fill a QComboBox with the site / building / floor / room tree (item data = place id, first item = None)."""
    from .. import places
    cur = combo.currentData() if keep else None
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(first_text, None)
    for nid, label, typ in places.options(cfg):
        combo.addItem(f"{label}", nid)
        combo.setItemData(combo.count() - 1, f"{places.TYPE_LABEL[typ]}: {places.path_text(cfg, nid)}",
                          Qt.ItemDataRole.ToolTipRole)
    combo.setCurrentIndex(max(0, combo.findData(cur)))
    combo.blockSignals(False)
    return combo.count() > 1
