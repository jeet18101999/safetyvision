"""Builds dist/SafetyVision-Setup.exe (one-file installer with the app files embedded).

    .venv\\Scripts\\python.exe installer\\build_installer.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "installer"
PAYLOAD = BUILD / "payload"
APP_FILES = ["main.py", "requirements.txt", "README.md", "safetyvision", "assets"]


def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pyinstaller"])
    if not (ROOT / "assets" / "icon.ico").exists():
        from PySide6.QtWidgets import QApplication
        QApplication([])
        sys.path.insert(0, str(ROOT))
        from safetyvision.ui.icons import export_icon_files
        export_icon_files(ROOT / "assets")
    shutil.rmtree(BUILD, ignore_errors=True)
    PAYLOAD.mkdir(parents=True)
    for name in APP_FILES:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, PAYLOAD / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, PAYLOAD / name)
    icon = ROOT / "assets" / "icon.ico"
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed", "--name", "SafetyVision-Setup",
        "--icon", str(icon), "--add-data", f"{PAYLOAD}{';' if sys.platform == 'win32' else ':'}payload",
        "--distpath", str(ROOT / "dist"), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD),
        str(ROOT / "installer" / "setup_app.py")])
    print("Built", ROOT / "dist" / "SafetyVision-Setup.exe")


if __name__ == "__main__":
    main()
