"""Find devices on the network and add them as recorders / cameras."""
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .. import config as C
from ..discovery import detect_channels, discover
from .icons import app_icon, glyph_icon


class _Bridge(QObject):
    progress = Signal(str, float)
    found = Signal(list)
    detected = Signal(int, dict)


COLS = ["", "IP address", "Type", "Brand / model", "Name", "Found by", "Ports", "Status", "Channels"]


class DiscoveryDialog(QDialog):
    """Returns the recorders to add in .added (list of recorder dicts)."""

    def __init__(self, cfg: dict, existing: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find devices on the network")
        self.setWindowIcon(app_icon())
        self.resize(1180, 680)
        self.cfg = cfg
        self.existing = existing
        self.devices: list[dict] = []
        self.detect: dict[int, dict] = {}
        self.added: list[dict] = []
        self._b = _Bridge()
        self._b.progress.connect(self._on_progress)
        self._b.found.connect(self._on_found)
        self._b.detected.connect(self._on_detected)
        self._stop = threading.Event()

        lay = QVBoxLayout(self)
        t = QLabel("Find DVR / NVR / XVR recorders and IP cameras")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        info = QLabel("Searches your networks with ONVIF, Dahua and Hikvision discovery and scans for camera ports. "
                      "Devices behind a router are only found by the scan - add their address range below "
                      "(e.g. 192.168.1.0/24 or 10.0.0.1-10.0.0.80).")
        info.setWordWrap(True)
        info.setObjectName("muted")
        lay.addWidget(info)
        row = QHBoxLayout()
        row.addWidget(QLabel("Extra address ranges:"))
        self.ranges = QLineEdit(cfg.get("discovery_ranges", ""))
        self.ranges.setPlaceholderText("optional, e.g. 192.168.1.0/24, 10.0.0.10-10.0.0.60")
        row.addWidget(self.ranges, 1)
        self.scan_btn = QPushButton("  Scan network")
        self.scan_btn.setObjectName("primary")
        self.scan_btn.setIcon(glyph_icon("refresh", "#ffffff"))
        self.scan_btn.clicked.connect(self.scan)
        row.addWidget(self.scan_btn)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.status = QLabel("")
        self.status.setObjectName("muted")
        lay.addWidget(self.bar)
        lay.addWidget(self.status)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)

        cred = QHBoxLayout()
        cred.addWidget(QLabel("Login for ticked devices:"))
        self.user = QLineEdit("admin")
        self.user.setMaximumWidth(140)
        self.pw = QLineEdit()
        self.pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.pw.setPlaceholderText("password")
        self.pw.setMaximumWidth(180)
        self.stream = QComboBox()
        self.stream.addItem("Main stream", "main")
        self.stream.addItem("Sub stream", "sub")
        cred.addWidget(self.user)
        cred.addWidget(self.pw)
        cred.addWidget(self.stream)
        self.test_btn = QPushButton("  Test login && detect channels")
        self.test_btn.setIcon(glyph_icon("check"))
        self.test_btn.clicked.connect(self.test_selected)
        cred.addWidget(self.test_btn)
        cred.addStretch()
        self.add_btn = QPushButton("  Add ticked devices")
        self.add_btn.setObjectName("primary")
        self.add_btn.setIcon(glyph_icon("check", "#ffffff"))
        self.add_btn.clicked.connect(self.add_selected)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        cred.addWidget(self.add_btn)
        cred.addWidget(close)
        lay.addLayout(cred)
        self.scan()

    # ---------------------------------------------------------------- scan
    def scan(self):
        self.scan_btn.setEnabled(False)
        self.table.setRowCount(0)
        self.detect.clear()
        known = [r.get("host") for r in self.existing if r.get("host")]
        extra = self.ranges.text().strip()

        def work():
            try:
                res = discover(extra, known, progress=lambda s, f: self._b.progress.emit(s, f), stop=self._stop)
            except Exception as e:
                self._b.progress.emit(f"Scan failed: {e}", 1.0)
                res = []
            self._b.found.emit(res)

        threading.Thread(target=work, daemon=True).start()

    def _on_progress(self, text, frac):
        self.status.setText(text)
        self.bar.setValue(int(frac * 1000))

    def _known(self, ip):
        return next((r for r in self.existing if r.get("host") == ip), None)

    def _on_found(self, devices):
        self.scan_btn.setEnabled(True)
        self.devices = devices
        self.table.setRowCount(len(devices))
        for r, d in enumerate(devices):
            known = self._known(d["ip"])
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            likely = not known and d.get("kind") != "Unknown"
            chk.setCheckState(Qt.CheckState.Checked if likely else Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, chk)
            brand = {"dahua": "Dahua / CP Plus", "hikvision": "Hikvision", "uniview": "Uniview",
                     "xiongmai": "Xiongmai (generic DVR)"}.get(d.get("brand", ""), d.get("brand", "").title() or "-")
            model = d.get("model") or d.get("banner") or ""
            vals = [d["ip"], d.get("kind", ""), f"{brand}  {model}".strip(), d.get("name", ""),
                    ", ".join(d.get("methods", [])), ", ".join(map(str, d.get("ports", []))),
                    f"already added ({known['name']})" if known else "new",
                    str(d["channels"]) if d.get("channels") else ""]
            for c, v in enumerate(vals, start=1):
                it = QTableWidgetItem(v)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if c == 7:
                    it.setForeground(QColor("#8b949e" if known else "#3fb950"))
                self.table.setItem(r, c, it)
        if not devices:
            self.status.setText("No devices found. If the recorder is on another subnet, add its range above "
                                "(e.g. 192.168.1.0/24) and scan again.")

    # ---------------------------------------------------------------- detect / add
    def _ticked(self):
        return [r for r in range(self.table.rowCount())
                if self.table.item(r, 0) and self.table.item(r, 0).checkState() == Qt.CheckState.Checked]

    def test_selected(self):
        rows = self._ticked()
        if not rows:
            self.status.setText("Tick at least one device.")
            return
        user, pw, stream = self.user.text().strip(), self.pw.text(), self.stream.currentData()
        for r in rows:
            self.table.item(r, 8).setText("testing ...")

        def work(r, dev):
            try:
                res = detect_channels(dev, user, pw, stream)
            except Exception as e:
                res = {"ok": False, "message": str(e)}
            self._b.detected.emit(r, res)

        for r in rows:
            threading.Thread(target=work, args=(r, self.devices[r]), daemon=True).start()

    def _on_detected(self, r, res):
        self.detect[r] = res
        it = self.table.item(r, 8)
        if res.get("ok"):
            it.setText(f"{res['channels']} channel(s) - {res['message']}")
            it.setForeground(QColor("#3fb950"))
        else:
            it.setText(res.get("message", "failed"))
            it.setForeground(QColor("#f85149"))

    def _unique_prefix(self, base):
        used = {r.get("prefix", "") for r in self.existing + self.added}
        if "" not in used:
            return ""
        k = 2
        while f"{base}{k}-" in used:
            k += 1
        return f"{base}{k}-"

    def add_selected(self):
        rows = [r for r in self._ticked() if not self._known(self.devices[r]["ip"])]
        missing = [r for r in rows if not self.detect.get(r, {}).get("ok")]
        if missing:
            self.status.setText("Run 'Test login & detect channels' first - every ticked device needs a working login.")
            return
        for r in rows:
            d, res = self.devices[r], self.detect[r]
            brand = res["brand"] if res["brand"] in ("dahua", "hikvision", "onvif") else "dahua"
            is_cam = d.get("kind") == "IP camera" or res["channels"] == 1
            rec = {**C.RECORDER_DEFAULTS,
                   "name": d.get("name") or f"{'Camera' if is_cam else 'Recorder'} {d['ip']}",
                   "prefix": self._unique_prefix("CAM" if is_cam else "DVR"),
                   "brand": brand, "host": d["ip"], "port": 554, "username": self.user.text().strip(),
                   "password": self.pw.text(), "channels": f"1-{res['channels']}" if res["channels"] > 1 else "1",
                   "stream": self.stream.currentData(), "model": res.get("model") or d.get("model", ""),
                   "mac": d.get("mac", ""), "serial": res.get("serial", "")}
            if brand == "onvif":
                rec["stream_urls"] = {str(k + 1): s["uri"] for k, s in enumerate(res.get("onvif_streams", []))}
            self.added.append(rec)
        if self.ranges.text().strip():
            self.cfg["discovery_ranges"] = self.ranges.text().strip()
        self.accept()

    def reject(self):
        self._stop.set()
        super().reject()
