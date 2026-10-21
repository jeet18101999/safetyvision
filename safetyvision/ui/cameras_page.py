"""Cameras page: sites > buildings > floors > rooms on the left, per-camera settings in the middle and a live
preview of the selected camera on the right."""
import threading
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QMenu, QMessageBox, QPushButton, QSizePolicy, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from .. import config as C
from .. import places as P
from ..xvr import build_sources, grab_frame, source_list
from .pages import _btn, _item, _title
from .widgets import PLACE_ICONS, bgr_to_pixmap, fill_place_combo
from .zone_editor import ZoneEditor

FEATURE_COLS = [("people", "Man count"), ("man_down", "Man down"), ("fire", "Fire / smoke"), ("ppe", "PPE check")]
ALL = "__all__"
NONE = "__none__"


class _Grabber(QObject):
    got = Signal(str, object)


class CamerasPage(QWidget):
    """Per-camera name, place (site / building / floor / room) and detection features."""
    C_ON, C_CAM, C_NAME, C_PLACE = 0, 1, 2, 3
    C_FEAT = 4

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.names: list[str] = []
        self.recorder_of: dict[str, str] = {}
        self._zones = {}
        self._preview_name = None
        self._grabbing = set()
        self._grabbed: dict[str, tuple] = {}  # name -> (frame, time) when not monitoring
        self._grabber = _Grabber()
        self._grabber.got.connect(self._on_grabbed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        head = QHBoxLayout()
        head.addWidget(_title("Cameras", "Group cameras by site, building, floor and room, name them and pick their "
                                         "detections. Click a camera to see it. Click a column header to toggle it "
                                         "for all shown cameras."), 1)
        disc = _btn("Discover devices...", "refresh", "primary", "Find DVR / NVR / IP cameras on the network")
        disc.clicked.connect(self._discover)
        head.addWidget(disc)
        lay.addLayout(head)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_tree_panel())
        split.addWidget(self._build_table_panel())
        split.addWidget(self._build_preview_panel())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([250, 900, 360])
        lay.addWidget(split, 1)

        row = QHBoxLayout()
        self.hint = QLabel("")
        self.hint.setObjectName("muted")
        row.addWidget(self.hint, 1)
        revert = _btn("Revert", "refresh")
        revert.clicked.connect(self.load)
        save = _btn("Save && apply", "check", "primary")
        save.clicked.connect(self.save)
        row.addWidget(revert)
        row.addWidget(save)
        lay.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_preview)
        self.timer.start(500)

    # ================================================================ places tree
    def _build_tree_panel(self):
        w = QFrame()
        w.setObjectName("panel")
        v = QVBoxLayout(w)
        t = QLabel("<b>Sites, buildings, floors, rooms</b>")
        v.addWidget(t)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.currentItemChanged.connect(lambda *_: self._apply_filter())
        self.tree.itemDoubleClicked.connect(lambda it, _: self._rename() if it.data(0, Qt.ItemDataRole.UserRole)
                                            not in (ALL, NONE) else None)
        v.addWidget(self.tree, 1)
        g = QHBoxLayout()
        self.add_btn = _btn("Add", "cameras", "primary", "Add a site, or a building / floor / room inside the selected place")
        self.add_btn.clicked.connect(lambda: self._add_menu(self.add_btn))
        more = _btn("More", "settings")
        more.clicked.connect(lambda: self._tree_menu(None, more))
        g.addWidget(self.add_btn)
        g.addWidget(more)
        v.addLayout(g)
        self.assign_btn = _btn("Put selected cameras here", "check",
                               tip="Assign the selected camera rows to the place selected in this tree")
        self.assign_btn.clicked.connect(self._assign_selected)
        v.addWidget(self.assign_btn)
        hint = QLabel("A site can hold buildings, floors or rooms; buildings hold floors or rooms. Cameras in the "
                      "same room count each person once. Floor plans are added here or on the Site map.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        v.addWidget(hint)
        return w

    def _cur_place(self):
        it = self.tree.currentItem()
        nid = it.data(0, Qt.ItemDataRole.UserRole) if it else None
        return None if nid in (ALL, NONE, None) else nid

    def _fill_tree(self, select=None):
        cfg = self.win.cfg
        keep = select if select is not None else (self.tree.currentItem().data(0, Qt.ItemDataRole.UserRole)
                                                  if self.tree.currentItem() else ALL)
        counts = {}
        for name in self.names:
            pid = self._row_place(self.names.index(name)) if self.table.rowCount() else C.camera_cfg(cfg, name).get("place")
            for n in P.chain(cfg, pid):
                counts[n["id"]] = counts.get(n["id"], 0) + 1
        unassigned = sum(1 for r in range(self.table.rowCount()) if not self._row_place(r))
        self.tree.blockSignals(True)
        self.tree.clear()
        root = QTreeWidgetItem([f"All cameras ({len(self.names)})"])
        root.setData(0, Qt.ItemDataRole.UserRole, ALL)
        self.tree.addTopLevelItem(root)
        items = {}
        for d, n in P.walk(cfg):
            parent = items.get(n.get("parent"))
            text = f"{PLACE_ICONS.get(n['type'], '')} {n['name']}  ({counts.get(n['id'], 0)})" + \
                   ("  \U0001F5FA" if n.get("map") else "")
            it = QTreeWidgetItem([text])
            it.setToolTip(0, f"{P.TYPE_LABEL[n['type']]}: {P.path_text(cfg, n['id'])}")
            it.setData(0, Qt.ItemDataRole.UserRole, n["id"])
            (parent or root).addChild(it)
            items[n["id"]] = it
        un = QTreeWidgetItem([f"Not assigned ({unassigned})"])
        un.setData(0, Qt.ItemDataRole.UserRole, NONE)
        root.addChild(un)
        self.tree.expandAll()
        target = items.get(keep) or (un if keep == NONE else root)
        self.tree.setCurrentItem(target)
        self.tree.blockSignals(False)
        self._apply_filter()

    def _add_menu(self, anchor):
        cur = self._cur_place()
        ptype = P.get(self.win.cfg, cur)["type"] if cur else None
        m = QMenu(self)
        m.addAction("New site ...", lambda: self._add("site", None))
        if cur:
            for t in P.CHILD_TYPES[ptype]:
                m.addAction(f"New {t} in '{P.get(self.win.cfg, cur)['name']}' ...", lambda t=t: self._add(t, cur))
        m.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def _tree_menu(self, pos, anchor=None):
        if pos is not None:
            it = self.tree.itemAt(pos)
            if it:
                self.tree.setCurrentItem(it)
        cur = self._cur_place()
        m = QMenu(self)
        m.addAction("New site ...", lambda: self._add("site", None))
        if cur:
            n = P.get(self.win.cfg, cur)
            for t in P.CHILD_TYPES[n["type"]]:
                m.addAction(f"New {t} in '{n['name']}' ...", lambda t=t: self._add(t, cur))
            m.addSeparator()
            m.addAction("Rename ...", self._rename)
            m.addAction("Floor plan image ..." if not n.get("map") else "Change floor plan image ...", self._set_map)
            if n.get("map"):
                m.addAction("Remove floor plan", lambda: self._clear_map(cur))
            m.addAction("Put selected cameras here", self._assign_selected)
            m.addSeparator()
            m.addAction("Delete ...", self._delete)
        if anchor is not None:
            m.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        else:
            m.exec(self.tree.viewport().mapToGlobal(pos))

    def _add(self, ntype, parent):
        examples = {"site": "Main plant, Warehouse 2", "building": "Block A, Admin building",
                    "floor": "Ground floor, Floor 1", "room": "Store room, Assembly hall, Gate"}
        name, ok = QInputDialog.getText(self, f"New {ntype}", f"Name of the {ntype} (e.g. {examples[ntype]}):")
        name = name.strip()
        if not ok or not name:
            return
        if any(n["name"].lower() == name.lower() for n in P.children(self.win.cfg, parent)):
            QMessageBox.information(self, "Places", f"'{name}' already exists here.")
            return
        n = P.add(self.win.cfg, name, ntype, parent)
        self.win.save_config()
        self._places_changed(select=n["id"])

    def _rename(self):
        cur = self._cur_place()
        n = P.get(self.win.cfg, cur)
        if not n:
            return
        name, ok = QInputDialog.getText(self, f"Rename {n['type']}", "New name:", text=n["name"])
        if ok and name.strip():
            n["name"] = name.strip()
            self.win.save_config()
            self._places_changed()

    def _delete(self):
        cur = self._cur_place()
        n = P.get(self.win.cfg, cur)
        if not n:
            return
        inside = len(P.subtree_ids(self.win.cfg, cur)) - 1
        msg = f"Delete {n['type']} '{n['name']}'" + (f" and the {inside} place(s) inside it" if inside else "") + \
              "?\nIts cameras keep working and move up to the place above (or become unassigned)."
        if QMessageBox.question(self, "Delete place", msg) != QMessageBox.StandardButton.Yes:
            return
        gone = P.subtree_ids(self.win.cfg, cur)
        parent = P.delete(self.win.cfg, cur)
        for r in range(self.table.rowCount()):  # unsaved choices in the table follow too
            if self._row_place(r) in gone:
                self._set_row_place(r, parent)
        self.win.save_config()
        self._places_changed(select=parent or ALL)

    def _set_map(self):
        from .sitemap import store_floor_plan
        n = P.get(self.win.cfg, self._cur_place())
        if not n:
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Floor plan for {n['name']}", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        store_floor_plan(n, path)
        self.win.save_config()
        self._places_changed()
        self.win.toast("Floor plan added - place the cameras on it on the Site map page")

    def _clear_map(self, nid):
        n = P.get(self.win.cfg, nid)
        n["map"] = ""
        for cc in (self.win.cfg.get("cameras") or {}).values():
            if cc.get("map_node") == nid:
                cc["map_node"], cc["map_pos"] = "", None
        self.win.save_config()
        self._places_changed()

    def _places_changed(self, select=None):
        for r in range(self.table.rowCount()):
            fill_place_combo(self.table.cellWidget(r, self.C_PLACE), self.win.cfg, "- not assigned -")
        self._fill_tree(select)
        self.win.cameras_changed()

    def _assign_selected(self):
        cur = self._cur_place()
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        if not cur:
            self.win.toast("Select a site / building / floor / room in the tree first")
            return
        if not rows:
            self.win.toast("Select camera rows first (click / Ctrl+click / Shift+click rows), then press again")
            return
        for r in rows:
            self._set_row_place(r, cur)
        self._fill_tree()
        self.win.toast(f"{len(rows)} camera(s) put in {P.path_text(self.win.cfg, cur)} - press Save & apply")

    def _apply_filter(self):
        it = self.tree.currentItem()
        sel = it.data(0, Qt.ItemDataRole.UserRole) if it else ALL
        for r in range(self.table.rowCount()):
            pid = self._row_place(r)
            show = sel == ALL or (sel == NONE and not pid) or (sel not in (ALL, NONE) and P.camera_in(self.win.cfg, pid, sel))
            self.table.setRowHidden(r, not show)
        self.assign_btn.setEnabled(sel not in (ALL, NONE, None))

    # ================================================================ camera table
    def _build_table_panel(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        bulk = QHBoxLayout()
        bulk.addWidget(QLabel("Recorder:"))
        self.rec_pick = QComboBox()
        bulk.addWidget(self.rec_pick)
        for text, on in (("Enable all its cameras", True), ("Disable all its cameras", False)):
            b = _btn(text)
            b.clicked.connect(lambda _=False, val=on: self._bulk_enable(val))
            bulk.addWidget(b)
        bulk.addStretch()
        v.addLayout(bulk)
        self.cols = (["On", "Camera", "Name", "Place"] + [lbl for _, lbl in FEATURE_COLS]
                     + [C.PPE_LABELS[i] for i in C.PPE_ITEMS] + ["Zone", "Zone area", "Crowd limit"])
        self.C_PPE = self.C_FEAT + len(FEATURE_COLS)
        self.C_ZONE = self.C_PPE + len(C.PPE_ITEMS)
        self.table = QTableWidget(0, len(self.cols))
        self.table.setHorizontalHeaderLabels(self.cols)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(self.C_NAME, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(self.C_PLACE, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(self.C_NAME, 170)
        self.table.setColumnWidth(self.C_PLACE, 190)
        hh.sectionClicked.connect(self._toggle_column)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.currentCellChanged.connect(lambda r, *_: self._select_row(r))
        v.addWidget(self.table, 1)
        return w

    def _check(self, on, enabled=True):
        it = QTableWidgetItem()
        it.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled if enabled else Qt.ItemFlag.NoItemFlags)
        it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return it

    def _place_combo(self, current):
        cb = QComboBox()
        fill_place_combo(cb, self.win.cfg, "- not assigned -", keep=False)
        cb.setCurrentIndex(max(0, cb.findData(current or None)))
        cb.setMinimumWidth(150)
        cb.currentIndexChanged.connect(lambda _: (self._fill_tree(), self._update_preview_text()))
        return cb

    def _row_place(self, r):
        cb = self.table.cellWidget(r, self.C_PLACE)
        return cb.currentData() if cb else None

    def _set_row_place(self, r, pid):
        cb = self.table.cellWidget(r, self.C_PLACE)
        cb.blockSignals(True)
        cb.setCurrentIndex(max(0, cb.findData(pid or None)))
        cb.blockSignals(False)

    def load(self):
        cfg = self.win.cfg
        srcs = source_list(cfg)
        self.names = [s["name"] for s in srcs]
        self.recorder_of = {s["name"]: s["recorder"] for s in srcs}
        self.rec_pick.clear()
        for r in dict.fromkeys(s["recorder"] for s in srcs):
            self.rec_pick.addItem(r or "Cameras", r)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.names))
        self._zones = {}
        for r, name in enumerate(self.names):
            cc = C.camera_cfg(cfg, name)
            self.table.setItem(r, self.C_ON, self._check(cc["enabled"]))
            cam = _item(name)
            cam.setToolTip(f"Recorder: {self.recorder_of.get(name, '')}")
            self.table.setItem(r, self.C_CAM, cam)
            nm = QTableWidgetItem(cc.get("label") or "")
            nm.setToolTip("Friendly name shown everywhere, e.g. 'Gate - entry'")
            self.table.setItem(r, self.C_NAME, nm)
            self.table.setCellWidget(r, self.C_PLACE, self._place_combo(cc.get("place")))
            for k, (key, _) in enumerate(FEATURE_COLS):
                self.table.setItem(r, self.C_FEAT + k, self._check(cc[key]))
            for k, item in enumerate(C.PPE_ITEMS):
                self.table.setItem(r, self.C_PPE + k, self._check(item in cc["ppe_items"]))
            self._zones[name] = list(cc.get("zone_points") or [])
            self.table.setItem(r, self.C_ZONE, self._check(cc["zone"]))
            b = QPushButton("Draw..." if not self._zones[name] else f"Edit ({len(self._zones[name])} pts)")
            b.clicked.connect(lambda _=False, n=name: self._draw_zone(n))
            self.table.setCellWidget(r, self.C_ZONE + 1, b)
            sp = QSpinBox()
            sp.setRange(0, 999)
            sp.setSpecialValueText("off")
            sp.setValue(cc["max_people"] if cc["crowd"] else 0)
            self.table.setCellWidget(r, self.C_ZONE + 2, sp)
        self.table.blockSignals(False)
        self._fill_tree()
        self._update_hint()
        if self.names:
            r = self.names.index(self._preview_name) if self._preview_name in self.names else 0
            self.table.setCurrentCell(r, self.C_NAME)
            self._select_row(r)

    def _bulk_enable(self, on):
        rec = self.rec_pick.currentData()
        for r, name in enumerate(self.names):
            if self.recorder_of.get(name) == rec:
                self.table.item(r, self.C_ON).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._update_hint()

    def _update_hint(self):
        on = sum(1 for r in range(self.table.rowCount())
                 if self.table.item(r, self.C_ON).checkState() == Qt.CheckState.Checked)
        nrec = len(set(self.recorder_of.values()))
        self.hint.setText(f"{on} of {len(self.names)} cameras enabled on {nrec} recorder(s). "
                          "Changes take effect after Save & apply.")

    def _toggle_column(self, col):
        if col in (self.C_CAM, self.C_NAME, self.C_PLACE) or col >= self.C_ZONE + 1:
            return
        rows = [r for r in range(self.table.rowCount()) if not self.table.isRowHidden(r)]
        items = [self.table.item(r, col) for r in rows]
        on = not all(i.checkState() == Qt.CheckState.Checked for i in items if i)
        for i in items:
            if i:
                i.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._update_hint()

    def _frame_for(self, name, wait=False):
        e = self.win.engine
        idx = next((i for i, s in enumerate(e.streams) if s.name == name and s.enabled), None)
        frame = e.snapshot(idx) if idx is not None else None
        if frame is None and wait:
            url = dict(build_sources(self.win.cfg)).get(name)
            frame = grab_frame(url) if url else None
        return frame

    def _draw_zone(self, name):
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            frame = self._frame_for(name, wait=True)
        finally:
            QApplication.restoreOverrideCursor()
        label = C.camera_cfg(self.win.cfg, name).get("label") or name
        dlg = ZoneEditor(label, frame, self._zones.get(name, []), self)
        if dlg.exec():
            self._zones[name] = dlg.points
            r = self.names.index(name)
            self.table.cellWidget(r, self.C_ZONE + 1).setText(f"Edit ({len(dlg.points)} pts)" if dlg.points else "Draw...")
            self.table.item(r, self.C_ZONE).setCheckState(Qt.CheckState.Checked if dlg.points else Qt.CheckState.Unchecked)

    # ================================================================ preview
    def _build_preview_panel(self):
        w = QFrame()
        w.setObjectName("panel")
        v = QVBoxLayout(w)
        self.pv_title = QLabel("<b>Preview</b>")
        self.pv_title.setWordWrap(True)
        v.addWidget(self.pv_title)
        self.pv_img = QLabel("Click a camera to see it")
        self.pv_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pv_img.setMinimumSize(240, 135)
        self.pv_img.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.pv_img.setStyleSheet("background:#05070a; border-radius:6px; color:#7d8590")
        v.addWidget(self.pv_img, 3)
        self.pv_info = QLabel("")
        self.pv_info.setWordWrap(True)
        self.pv_info.setObjectName("muted")
        self.pv_info.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.pv_info)
        b = QHBoxLayout()
        self.pv_refresh = _btn("Refresh", "refresh", tip="Take a new picture from the camera")
        self.pv_refresh.clicked.connect(lambda: self._request_grab(self._preview_name, force=True))
        self.pv_live = _btn("Open in Live", "live", tip="Show this camera full screen on the Live page")
        self.pv_live.clicked.connect(self._open_live)
        b.addWidget(self.pv_refresh)
        b.addWidget(self.pv_live)
        v.addLayout(b)
        v.addStretch(1)
        return w

    def _select_row(self, r):
        if r is None or r < 0 or r >= len(self.names):
            return
        name = self.names[r]
        if name != self._preview_name:
            self._preview_name = name
            got = self._grabbed.get(name)
            if got and got[0] is not None:
                self._show(got[0])
            else:
                self.pv_img.setText("Connecting to the camera ...")
        self._update_preview_text()
        self._refresh_preview()

    def _update_preview_text(self):
        name = self._preview_name
        if name not in self.names:
            return
        r = self.names.index(name)
        label = self.table.item(r, self.C_NAME).text().strip() if self.table.item(r, self.C_NAME) else ""
        pid = self._row_place(r)
        self.pv_title.setText(f"<b>{label or name}</b>" + (f"  <span style='color:#7d8590'>{name}</span>" if label else ""))
        place = P.path_text(self.win.cfg, pid) if pid else "<i>not assigned to a place</i>"
        e = self.win.engine
        idx = next((i for i, s in enumerate(e.streams) if s.name == name), None)
        status = e.streams[idx].status if idx is not None and e.streams[idx].enabled else "not streaming"
        self.pv_info.setText(f"Recorder: {self.recorder_of.get(name) or '-'}<br>Place: {place}<br>Status: {status}")

    def _refresh_preview(self):
        name = self._preview_name
        if not name or not self.isVisible():
            return
        frame = self._frame_for(name)
        if frame is not None:
            self._show(frame)
            return
        got = self._grabbed.get(name)
        if got is None or time.time() - got[1] > 30:
            self._request_grab(name)

    def _request_grab(self, name, force=False):
        if not name or name in self._grabbing:
            return
        if not force and self._frame_for(name) is not None:
            return
        url = dict(build_sources(self.win.cfg)).get(name)
        if not url:
            return
        self._grabbing.add(name)
        if name == self._preview_name and name not in self._grabbed:
            self.pv_img.setText("Connecting to the camera ...")

        def work():
            try:
                frame = grab_frame(url)
            except Exception:  # noqa: BLE001
                frame = None
            self._grabber.got.emit(name, frame)

        threading.Thread(target=work, daemon=True).start()

    def _on_grabbed(self, name, frame):
        self._grabbing.discard(name)
        if frame is not None:
            self._grabbed[name] = (frame, time.time())
        else:
            self._grabbed.pop(name, None)
        if name == self._preview_name:
            if frame is not None:
                self._show(frame)
            else:
                self._grabbed[name] = (None, time.time())  # don't retry for 30 s
                self.pv_img.setText("No picture - the camera is offline or unreachable")

    def _show(self, frame):
        if frame is None:
            return
        pm = bgr_to_pixmap(frame)
        self.pv_img.setPixmap(pm.scaled(self.pv_img.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))

    def _open_live(self):
        e = self.win.engine
        idx = next((i for i, s in enumerate(e.streams) if s.name == self._preview_name and s.enabled), None)
        if idx is None:
            self.win.toast("This camera is not streaming - enable it and start monitoring")
            return
        self.win.focus_camera(idx)

    # ================================================================ save
    def save(self):
        cfg = self.win.cfg
        cams = cfg.setdefault("cameras", {})
        for r, name in enumerate(self.names):
            chk = lambda c: self.table.item(r, c).checkState() == Qt.CheckState.Checked  # noqa: E731
            limit = self.table.cellWidget(r, self.C_ZONE + 2).value()
            place = self._row_place(r) or ""
            prev = cams.get(name, {})
            map_node = prev.get("map_node", "")
            keep_pos = bool(map_node and place) and P.camera_in(cfg, place, map_node)  # plan still above the camera
            cams[name] = {**prev,  # keep settings managed elsewhere (e.g. map position)
                          "enabled": chk(self.C_ON), "label": self.table.item(r, self.C_NAME).text().strip(),
                          "place": place,
                          "map_node": map_node if keep_pos else "",
                          "map_pos": prev.get("map_pos") if keep_pos else None,
                          **{key: chk(self.C_FEAT + k) for k, (key, _) in enumerate(FEATURE_COLS)},
                          "ppe_items": [i for k, i in enumerate(C.PPE_ITEMS) if chk(self.C_PPE + k)],
                          "zone": chk(self.C_ZONE) and len(self._zones.get(name, [])) >= 3,
                          "zone_points": self._zones.get(name, []),
                          "crowd": limit > 0, "max_people": limit or cfg["camera_defaults"]["max_people"]}
            for k in ("location", "room"):
                cams[name].pop(k, None)
        self.win.save_config()
        self.win.engine.update_config(cfg)
        self.win.cameras_changed()
        self._update_hint()
        self.load()
        self.win.toast("Camera settings applied")

    def _discover(self):
        from .discovery_dialog import DiscoveryDialog
        dlg = DiscoveryDialog(self.win.cfg, self.win.cfg.get("recorders") or [], self)
        if dlg.exec() and dlg.added:
            self.win.cfg.setdefault("recorders", []).extend(dlg.added)
            self.win.save_config()
            self.win.engine.apply_config(self.win.cfg)
            if self.win.engine.running or self.win.engine.loading:
                self.win.live.build_tiles()
            self.load()
            self.win.cameras_changed()
            self.win.toast(f"Added {len(dlg.added)} device(s) - name them and put them in their places")
