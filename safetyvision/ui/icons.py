"""App icon (drawn in code, exported to .ico/.png) and UI glyph icons (Windows Segoe Fluent / MDL2 font)."""
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QIcon, QImage, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap, QRadialGradient)

GLYPHS = {
    "dashboard": "", "live": "", "events": "", "cameras": "", "learning": "",
    "settings": "", "play": "", "stop": "", "check": "", "cancel": "",
    "person_off": "", "export": "", "folder": "", "delete": "", "refresh": "",
    "zone": "", "grid": "", "people": "", "robot": "", "train": "",
    "shield": "", "fullscreen": "", "info": "", "sitemap": "", "search": "",
}


@lru_cache(maxsize=1)
def _icon_family() -> str:
    fams = set(QFontDatabase.families())
    for f in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if f in fams:
            return f
    return ""


@lru_cache(maxsize=256)
def glyph_icon(name: str, color: str = "#c9d1d9", size: int = 32) -> QIcon:
    fam = _icon_family()
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor(color))
    if fam:
        f = QFont(fam)
        f.setPixelSize(int(size * 0.72))
        p.setFont(f)
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, GLYPHS.get(name, "?"))
    else:
        p.drawEllipse(QRectF(size * 0.25, size * 0.25, size * 0.5, size * 0.5))
    p.end()
    return QIcon(pm)


def paint_app_icon(size: int) -> QImage:
    """Shield + camera lens + flame: 'AI safety camera'."""
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 256.0

    # rounded tile background
    bg = QLinearGradient(0, 0, size, size)
    bg.setColorAt(0, QColor("#16233a"))
    bg.setColorAt(1, QColor("#0a0f1a"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(bg))
    p.drawRoundedRect(QRectF(4 * s, 4 * s, 248 * s, 248 * s), 52 * s, 52 * s)

    # shield
    sh = QPainterPath()
    sh.moveTo(128 * s, 26 * s)
    sh.cubicTo(160 * s, 44 * s, 190 * s, 50 * s, 214 * s, 50 * s)
    sh.cubicTo(214 * s, 140 * s, 190 * s, 196 * s, 128 * s, 232 * s)
    sh.cubicTo(66 * s, 196 * s, 42 * s, 140 * s, 42 * s, 50 * s)
    sh.cubicTo(66 * s, 50 * s, 96 * s, 44 * s, 128 * s, 26 * s)
    sg = QLinearGradient(0, 26 * s, 0, 232 * s)
    sg.setColorAt(0, QColor("#2f81f7"))
    sg.setColorAt(1, QColor("#1146a8"))
    p.setBrush(QBrush(sg))
    p.setPen(QPen(QColor("#9ecbff"), max(1.0, 5 * s)))
    p.drawPath(sh)

    # camera lens ring
    c = QPointF(128 * s, 124 * s)
    p.setPen(QPen(QColor("#ffffff"), max(1.0, 9 * s)))
    p.setBrush(QColor("#0b1d3d"))
    p.drawEllipse(c, 56 * s, 56 * s)

    # flame inside the lens
    fl = QPainterPath()
    fl.moveTo(128 * s, 80 * s)
    fl.cubicTo(150 * s, 104 * s, 162 * s, 122 * s, 158 * s, 144 * s)
    fl.cubicTo(154 * s, 166 * s, 138 * s, 174 * s, 128 * s, 174 * s)
    fl.cubicTo(112 * s, 174 * s, 98 * s, 164 * s, 97 * s, 144 * s)
    fl.cubicTo(96 * s, 128 * s, 106 * s, 118 * s, 112 * s, 106 * s)
    fl.cubicTo(114 * s, 118 * s, 118 * s, 124 * s, 124 * s, 126 * s)
    fl.cubicTo(122 * s, 110 * s, 124 * s, 94 * s, 128 * s, 80 * s)
    fg = QRadialGradient(QPointF(128 * s, 160 * s), 70 * s)
    fg.setColorAt(0, QColor("#ffe66d"))
    fg.setColorAt(0.45, QColor("#ffa62b"))
    fg.setColorAt(1, QColor("#f0431d"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(fg))
    p.drawPath(fl)

    # small "AI" status dot
    p.setBrush(QColor("#3fb950"))
    p.setPen(QPen(QColor("#0a0f1a"), max(1.0, 4 * s)))
    p.drawEllipse(QPointF(186 * s, 176 * s), 18 * s, 18 * s)
    p.end()
    return img


def app_icon() -> QIcon:
    ic = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(QPixmap.fromImage(paint_app_icon(sz)))
    return ic


def export_icon_files(out_dir):
    """Writes icon.png and a multi-size icon.ico (used by the installer and shortcuts)."""
    from pathlib import Path

    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    big = paint_app_icon(256)
    big.save(str(out / "icon.png"))
    Image.open(out / "icon.png").save(out / "icon.ico",
                                      sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return out / "icon.ico"
