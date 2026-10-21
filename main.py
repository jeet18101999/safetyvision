"""SafetyVision - XVR safety monitoring (fire/smoke, PPE, man count, man down, zones) with local-AI verification
and continuous learning."""
import logging
import sys
from logging.handlers import RotatingFileHandler

from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMessageBox

from safetyvision.config import DATA_DIR

INSTANCE_KEY = "SafetyVision-single-instance"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  RotatingFileHandler(DATA_DIR / "safetyvision.log", maxBytes=5_000_000,
                                                      backupCount=3, encoding="utf-8")])
    try:  # taskbar groups the app under its own icon instead of python's
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SafetyVision.App")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("SafetyVision")
    app.setQuitOnLastWindowClosed(False)

    # one instance only: a second copy would open every camera stream twice
    restart = "--restart" in sys.argv
    sock = QLocalSocket()
    if restart:  # wait for the previous instance to finish shutting down
        import time
        for _ in range(40):
            sock.connectToServer(INSTANCE_KEY)
            if not sock.waitForConnected(200):
                break
            sock.disconnectFromServer()
            time.sleep(0.5)
    sock.connectToServer(INSTANCE_KEY)
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.flush()
        sock.waitForBytesWritten(300)
        return 0
    QLocalServer.removeServer(INSTANCE_KEY)
    server = QLocalServer()
    server.listen(INSTANCE_KEY)

    from safetyvision.ui.icons import app_icon
    from safetyvision.ui.main_window import MainWindow
    from safetyvision.ui.widgets import APP_QSS
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    app.setWindowIcon(app_icon())
    tray = "--tray" in sys.argv
    try:
        w = MainWindow(start_in_tray=tray, resume=restart)
    except Exception as e:
        logging.exception("startup failed")
        QMessageBox.critical(None, "SafetyVision", f"Could not start:\n{e}")
        return 1
    server.newConnection.connect(lambda: (server.nextPendingConnection(), w._show_window()))
    if not tray:
        w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
