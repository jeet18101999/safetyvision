"""Tool dialogs: load test and fire simulation (run as background processes, report shown in the app)."""
import re
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QUrl
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QTextBrowser, QVBoxLayout)

from .. import config as C
from .icons import app_icon, glyph_icon


def _python():
    exe = Path(sys.executable)
    return str(exe.with_name("python.exe")) if exe.name.lower() == "pythonw.exe" else str(exe)


class ToolDialog(QDialog):
    """Runs `python -m safetyvision.<module> <args>` and shows progress, then the HTML report."""

    def __init__(self, title, intro, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowIcon(app_icon())
        self.resize(1100, 780)
        self.lay = QVBoxLayout(self)
        t = QLabel(title)
        t.setObjectName("pageTitle")
        self.lay.addWidget(t)
        i = QLabel(intro)
        i.setObjectName("muted")
        i.setWordWrap(True)
        self.lay.addWidget(i)
        self.options = QHBoxLayout()
        self.lay.addLayout(self.options)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.hide()
        self.lay.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.lay.addWidget(self.log)
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(True)
        self.lay.addWidget(self.report, 1)
        self.buttons = QHBoxLayout()
        self.buttons.addStretch()
        self.run_btn = QPushButton("  Run")
        self.run_btn.setObjectName("primary")
        self.run_btn.setIcon(glyph_icon("play", "#ffffff"))
        self.run_btn.clicked.connect(self.start)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        self.buttons.addWidget(self.run_btn)
        self.buttons.addWidget(close)
        self.lay.addLayout(self.buttons)
        self.proc = None
        self.report_path = None

    def module_args(self) -> tuple[str, list[str]]:
        raise NotImplementedError

    def before_start(self) -> bool:
        return True

    def after_finish(self, ok):
        pass

    def start(self):
        if not self.before_start():
            return
        mod, args = self.module_args()
        self.log.clear()
        self.report.clear()
        self.bar.show()
        self.run_btn.setEnabled(False)
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", str(C.ROOT))
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(C.ROOT))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.start(_python(), ["-m", mod, *args])

    def _read(self):
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            if line.startswith("[") and "]" in line[:15]:
                self.log.appendPlainText(line)
                m = re.search(r"REPORT (.+\.html)", line)
                if m:
                    self.report_path = Path(m.group(1).strip())

    def _done(self, code, _status):
        self.bar.hide()
        self.run_btn.setEnabled(True)
        if self.report_path and self.report_path.exists():
            self.report.setSearchPaths([str(self.report_path.parent)])
            self.report.setHtml(self.report_path.read_text(encoding="utf-8"))
        elif code != 0:
            self.log.appendPlainText(f"Finished with errors (exit {code}).")
        self.after_finish(code == 0)

    def closeEvent(self, e):
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            if QMessageBox.question(self, "Stop?", "The test is still running. Stop it?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.proc.kill()
        super().closeEvent(e)


class LoadTestDialog(ToolDialog):
    def __init__(self, win):
        super().__init__("Load test", "Records a short clip from every camera, then replays them as many virtual cameras "
                                      "through the real detection pipeline (same models, same per-camera settings) and "
                                      "measures the analysis rate, GPU, CPU and local-AI delay. Monitoring pauses while "
                                      "the test runs.", win)
        self.win = win
        self.counts = QLineEdit("16,24,32,48")
        self.secs = QSpinBox()
        self.secs.setRange(20, 600)
        self.secs.setValue(60)
        self.ai = QCheckBox("Include the local AI")
        self.ai.setChecked(True)
        self.rerecord = QCheckBox("Record fresh clips")
        for w in (QLabel("Camera counts"), self.counts, QLabel("Seconds per step"), self.secs, self.ai, self.rerecord):
            self.options.addWidget(w)
        self.options.addStretch()
        self._was_running = False
        latest = sorted((C.DATA_DIR / "loadtests").glob("loadtest_*.html"))
        if latest:
            self.report.setHtml(latest[-1].read_text(encoding="utf-8"))

    def before_start(self):
        self._was_running = self.win.engine.running or self.win.engine.loading
        if self._was_running:
            self.win.stop()
        return True

    def module_args(self):
        args = ["--counts", self.counts.text().replace(" ", ""), "--seconds", str(self.secs.value())]
        if not self.ai.isChecked():
            args.append("--no-ai")
        if self.rerecord.isChecked():
            args.append("--rerecord")
        return "safetyvision.loadtest", args

    def after_finish(self, ok):
        if self._was_running:
            self.win.start()


class FireSimulationDialog(ToolDialog):
    def __init__(self, win):
        super().__init__("Fire simulation", "Renders realistic simulated fires and smoke on footage from your own cameras, "
                                            "checks them with the local AI, trains a fire detector on them and tests it on "
                                            "cameras it never saw, plus on real footage without fire. Uses the clips "
                                            "recorded by the load test. Simulated fire is not real fire - use the result "
                                            "as a head start; the model keeps learning from real detections.", win)
        self.win = win
        self.epochs = QSpinBox()
        self.epochs.setRange(5, 300)
        self.epochs.setValue(40)
        self.test_cams = QLineEdit("CH3,CH8,CH12,CH15")
        for w in (QLabel("Epochs"), self.epochs, QLabel("Held-out test cameras"), self.test_cams):
            self.options.addWidget(w)
        self.options.addStretch()
        self.use_btn = QPushButton("  Use this model in the app")
        self.use_btn.setIcon(glyph_icon("check", "#ffffff"))
        self.use_btn.setObjectName("primary")
        self.use_btn.clicked.connect(self._deploy)
        self.buttons.insertWidget(1, self.use_btn)
        rep = C.DATA_DIR / "simulation" / "report.html"
        if rep.exists():
            self.report.setHtml(rep.read_text(encoding="utf-8"))
        self.use_btn.setEnabled((C.DATA_DIR / "simulation" / "fire_sim.pt").exists())

    def module_args(self):
        negs = [str(C.DATA_DIR / "fire_dataset")]
        return "safetyvision.simulate_fire", ["--epochs", str(self.epochs.value()), "--test-cams",
                                              self.test_cams.text().replace(" ", ""), "--extra-negatives", ";".join(negs)]

    def after_finish(self, ok):
        self.use_btn.setEnabled((C.DATA_DIR / "simulation" / "fire_sim.pt").exists())

    def _deploy(self):
        src = C.DATA_DIR / "simulation" / "fire_sim.pt"
        if QMessageBox.question(
                self, "Use simulated model",
                "Deploy the simulation-trained fire detector?\n\nIt runs together with the colour/flicker check and every "
                "detection is still verified by the local AI. From now on it keeps retraining on real confirmed "
                "detections and your reviews.") != QMessageBox.StandardButton.Yes:
            return
        cur = C.resolve(self.win.cfg["detection"]["fire_model"])
        cur.parent.mkdir(parents=True, exist_ok=True)
        (cur.parent / "versions").mkdir(exist_ok=True)
        shutil.copy2(src, cur.parent / "versions" / "fire_sim_v1.pt")
        shutil.copy2(src, cur)
        cur.with_suffix(".version").write_text("fire_sim_v1", encoding="utf-8")
        self.win.engine._on_model_updated("fire", str(cur))
        self.win.toast("Simulation-trained fire model deployed (fire_sim_v1)")
        QMessageBox.information(self, "Deployed", "fire_sim_v1 is now the active fire model.")
