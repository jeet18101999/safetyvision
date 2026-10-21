"""Draw a restricted-zone polygon on a camera image."""
import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .widgets import bgr_to_pixmap


class _Canvas(QWidget):
    def __init__(self, frame, points):
        super().__init__()
        self.pm = bgr_to_pixmap(frame) if frame is not None else None
        self.points = [list(p) for p in points]
        self.setMinimumSize(800, 450)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _rect(self):
        if self.pm is None:
            return self.rect()
        pw, ph = self.pm.width(), self.pm.height()
        s = min(self.width() / pw, self.height() / ph)
        w, h = pw * s, ph * s
        return (self.width() - w) / 2, (self.height() - h) / 2, w, h

    def mousePressEvent(self, e):
        if self.pm is None:
            return
        x0, y0, w, h = self._rect()
        if e.button() == Qt.MouseButton.LeftButton:
            nx, ny = (e.position().x() - x0) / w, (e.position().y() - y0) / h
            if 0 <= nx <= 1 and 0 <= ny <= 1:
                self.points.append([round(nx, 4), round(ny, 4)])
        elif e.button() == Qt.MouseButton.RightButton and self.points:
            self.points.pop()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#000"))
        if self.pm is None:
            p.setPen(QColor("#8b949e"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image from this camera")
            return
        x0, y0, w, h = self._rect()
        p.drawPixmap(int(x0), int(y0), int(w), int(h), self.pm)
        pts = [QPointF(x0 + x * w, y0 + y * h) for x, y in self.points]
        if len(pts) >= 3:
            fill = QColor("#f85149")
            fill.setAlpha(60)
            p.setBrush(fill)
        p.setPen(QPen(QColor("#ff6b6b"), 2))
        if len(pts) >= 2:
            p.drawPolygon(QPolygonF(pts)) if len(pts) >= 3 else p.drawPolyline(QPolygonF(pts))
        p.setBrush(QColor("#ffffff"))
        for q in pts:
            p.drawEllipse(q, 4, 4)


class ZoneEditor(QDialog):
    def __init__(self, camera_label, frame, points, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Restricted zone - {camera_label}")
        self.resize(1100, 700)
        self.canvas = _Canvas(frame, points)
        info = QLabel("Left-click to add corners of the restricted area, right-click to undo the last corner. "
                      "A person whose feet are inside the area raises an alert.")
        info.setWordWrap(True)
        info.setObjectName("muted")
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: (self.canvas.points.clear(), self.canvas.update()))
        save = QPushButton("Save zone")
        save.setObjectName("primary")
        save.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(clear)
        row.addStretch()
        row.addWidget(cancel)
        row.addWidget(save)
        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(row)

    @property
    def points(self):
        return self.canvas.points if len(self.canvas.points) >= 3 else []
