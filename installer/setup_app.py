"""SafetyVision one-click installer (built into SafetyVision-Setup.exe with PyInstaller - see build_installer.py).

Installs per-user (no admin) into %LOCALAPPDATA%\\Programs\\SafetyVision:
  1. app files          2. Python 3.11 (reuses an existing one)     3. PyTorch (CUDA if an NVIDIA GPU is present)
  4. dependencies       5. detection models                           6. Ollama + local vision model
  7. shortcuts, uninstaller entry, optional start with Windows
Re-running it upgrades the app and keeps config.yaml, data/ and trained models.
"""
import argparse
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

APP = "SafetyVision"
VERSION = "2.5.1"
DEFAULT_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Programs" / APP
PY_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
OLLAMA_URL = "https://ollama.com/download/OllamaSetup.exe"
TORCH_CUDA = "https://download.pytorch.org/whl/cu126"
TORCH_CPU = "https://download.pytorch.org/whl/cpu"
HF = "https://huggingface.co/Ultralytics/{repo}/resolve/main/{name}"
MODELS = [("YOLO11", "yolo11s-pose.pt"), ("YOLO11", "yolo11n.pt"), ("YOLO26", "yolo26n-cls.pt")]
EXTRA_MODELS = [("https://huggingface.co/opencv/person_reid_youtureid/resolve/main/person_reid_youtu_2021nov.onnx",
                 "reid/person_reid_youtu_2021nov.onnx")]  # body re-identification (OpenCV Zoo, Apache-2.0)
KEEP = {"config.yaml", "data", "models", ".venv", "python"}
UNINSTALLER = f"Uninstall {APP}.exe"
# Everything the installer creates (the uninstaller removes exactly these - never anything else in the folder)
APP_ITEMS = [".venv", "python", "assets", "installer", "safetyvision", "main.py", "README.md", "requirements.txt",
             "models", UNINSTALLER]
DATA_ITEMS = ["config.yaml", "data"]
MANIFEST = "install.json"  # inside installer/


def is_app_dir(p) -> bool:
    p = Path(p)
    return (p / "main.py").is_file() and (p / "safetyvision").is_dir()


def normalize_target(p) -> Path:
    """Never spread the app over a drive root or a folder that already holds other things."""
    p = Path(p)
    if is_app_dir(p):
        return p  # upgrade an existing installation in place
    if p.parent == p or (p.exists() and any(p.iterdir())):
        return p / APP
    return p
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def payload_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "payload"
    built = Path(__file__).resolve().parent.parent / "build" / "installer" / "payload"
    if not built.exists():
        raise SystemExit("Run installer/build_installer.py first (creates the clean app payload).")
    return built


class Installer:
    def __init__(self, target: Path, local_ai=True, desktop=True, autostart=False, log=print, progress=lambda p: None):
        self.target = normalize_target(target)
        self.local_ai, self.desktop, self.autostart = local_ai, desktop, autostart
        self.log, self.progress = log, progress

    # ---------------------------------------------------------------- helpers
    def run(self, cmd, check=True, cwd=None):
        self.log("  > " + " ".join(str(c) for c in cmd)[:180])
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
                             text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        last = ""
        for line in p.stdout:
            line = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line)  # terminal control codes (ollama progress)
            line = line.encode("ascii", "ignore").decode().strip()  # drop spinner / progress glyphs
            if line and line != last and not line.startswith(("\u2501", "|")):
                last = line
                if any(k in line for k in ("Successfully", "Downloading", "ERROR", "error", "pulling", "success",
                                           "Installing collected")):
                    self.log("    " + line[:160])
        code = p.wait()
        if check and code != 0:
            raise RuntimeError(f"command failed ({code}): {cmd[0]}")
        return code

    def download(self, url, dest: Path):
        self.log(f"  downloading {url.rsplit('/', 1)[-1]} ...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": f"{APP}-installer"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done, t0 = 0, time.time()
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and time.time() - t0 > 2:
                    t0 = time.time()
                    self.log(f"    {done / 2**20:.0f} / {total / 2**20:.0f} MB")
        tmp.replace(dest)

    # ---------------------------------------------------------------- steps
    def install(self):
        steps = [("Copying application files", self.copy_files), ("Finding Python 3.11", self.python),
                 ("Creating environment", self.venv), ("Installing PyTorch", self.torch),
                 ("Installing dependencies", self.deps), ("Downloading detection models", self.models)]
        if self.local_ai:
            steps.append(("Installing local AI (Ollama + vision model)", self.ollama))
        steps.append(("Creating shortcuts", self.shortcuts))
        steps.append(("Registering the uninstaller", self.write_manifest))
        for k, (title, fn) in enumerate(steps):
            self.log(f"[{k + 1}/{len(steps)}] {title}")
            self.progress(k / len(steps))
            fn()
        self.progress(1.0)
        self.log("Installation complete.")

    def copy_files(self):
        src = payload_dir()
        self.target.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.name in KEEP or item.name == "__pycache__":
                continue
            dst = self.target / item.name
            if item.is_dir():
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(item, dst, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(item, dst)
        (self.target / "installer").mkdir(exist_ok=True)
        if getattr(sys, "frozen", False):  # keep the setup program for repair, and as the uninstaller
            for dst in (self.target / "installer" / f"{APP}-Setup.exe", self.target / UNINSTALLER):
                if Path(sys.executable).resolve() != dst.resolve():
                    shutil.copy2(sys.executable, dst)
        self.log(f"  installing into {self.target}")

    def _find_python(self):
        cands = [self.target / "python" / "python.exe"]
        try:
            out = subprocess.run(["py", "-3.11", "-c", "import sys;print(sys.executable)"], capture_output=True,
                                 text=True, creationflags=NO_WINDOW)
            if out.returncode == 0:
                cands.append(Path(out.stdout.strip()))
        except FileNotFoundError:
            pass
        la = Path(os.environ.get("LOCALAPPDATA", ""))
        cands += [la / "Programs" / "Python" / "Python311" / "python.exe", Path(r"C:\Python311\python.exe")]
        return next((c for c in cands if c and c.exists()), None)

    def python(self):
        self.py = self._find_python()
        if self.py:
            self.log(f"  using {self.py}")
            return
        exe = Path(tempfile.gettempdir()) / "python-3.11.9-amd64.exe"
        self.download(PY_URL, exe)
        self.run([exe, "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0", "Include_test=0",
                  "Shortcuts=0", f"TargetDir={self.target / 'python'}"])
        self.py = self.target / "python" / "python.exe"

    @property
    def vpy(self):
        return self.target / ".venv" / "Scripts" / "python.exe"

    def venv(self):
        if not self.vpy.exists():
            self.run([self.py, "-m", "venv", self.target / ".venv"])
        self.run([self.vpy, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    @staticmethod
    def gpu_mb() -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, creationflags=NO_WINDOW)
            return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else 0
        except Exception:
            return 0

    def torch(self):
        gpu = self.gpu_mb()
        self.log(f"  NVIDIA GPU: {'%d MB' % gpu if gpu else 'not found - installing CPU version'}")
        self.run([self.vpy, "-m", "pip", "install", "torch", "torchvision", "--index-url",
                  TORCH_CUDA if gpu else TORCH_CPU])

    def deps(self):
        self.run([self.vpy, "-m", "pip", "install", "-r", self.target / "requirements.txt"])

    def models(self):
        mdir = self.target / "models"
        mdir.mkdir(exist_ok=True)
        for repo, name in MODELS:
            dst = mdir / name
            if dst.exists():
                continue
            try:
                self.download(HF.format(repo=repo, name=name), dst)
            except Exception as e:
                self.log(f"  mirror failed ({e}); the app will download {name} on first start")
        for url, rel in EXTRA_MODELS:
            dst = mdir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    self.download(url, dst)
                except Exception as e:
                    self.log(f"  {rel}: {e}; the app will download it on first start")

    def ollama(self):
        exe = shutil.which("ollama") or next((str(p) for p in (
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",) if p.exists()), None)
        if not exe:
            setup = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
            self.download(OLLAMA_URL, setup)
            self.run([setup, "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"])
            exe = str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")
        # make sure the server is up
        for _ in range(30):
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
                break
            except Exception:
                if _ == 0:
                    subprocess.Popen([exe, "serve"], creationflags=NO_WINDOW)
                time.sleep(2)
        model = "qwen2.5vl:7b" if self.gpu_mb() >= 7500 else "qwen2.5vl:3b"
        self.log(f"  vision model: {model} (this download is several GB, please wait)")
        self.run([exe, "pull", model])
        cfg = self.target / "config.yaml"
        if not cfg.exists():
            cfg.write_text(f"verifier:\n  model: {model}\n", encoding="utf-8")

    def shortcuts(self):
        pyw = self.target / ".venv" / "Scripts" / "pythonw.exe"
        icon = self.target / "assets" / "icon.ico"
        main = self.target / "main.py"
        appdata = Path(os.environ["APPDATA"])
        menu = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / APP
        menu.mkdir(parents=True, exist_ok=True)
        links = [(menu / f"{APP}.lnk", pyw, f'"{main}"')]
        uninstall = self.target / UNINSTALLER
        if not uninstall.exists():
            uninstall = self.target / "installer" / f"{APP}-Setup.exe"
        if uninstall.exists():
            links.append((menu / f"Uninstall {APP}.lnk", uninstall, "--uninstall"))
        if self.desktop:
            links.append((Path(os.path.join(os.environ["USERPROFILE"], "Desktop")) / f"{APP}.lnk", pyw, f'"{main}"'))
        for lnk, target, args in links:
            ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
                  f"$s.TargetPath='{target}';$s.Arguments='{args}';$s.WorkingDirectory='{self.target}';"
                  f"$s.IconLocation='{icon},0';$s.Description='{APP} - AI safety monitoring';$s.Save()")
            self.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps])
        import winreg
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP}")
        for name, val in (("DisplayName", APP), ("DisplayVersion", VERSION), ("Publisher", APP),
                          ("DisplayIcon", str(icon)), ("InstallLocation", str(self.target)),
                          ("UninstallString", f'"{uninstall}" --uninstall')):
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, val)
        winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
        winreg.CloseKey(key)
        if self.autostart:
            run = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                                 winreg.KEY_SET_VALUE)
            winreg.SetValueEx(run, APP, 0, winreg.REG_SZ, f'"{pyw}" "{main}" --tray')
            winreg.CloseKey(run)

    def write_manifest(self):
        import json
        items = [n for n in APP_ITEMS if (self.target / n).exists()]
        (self.target / "installer").mkdir(exist_ok=True)
        (self.target / "installer" / MANIFEST).write_text(json.dumps(
            {"app": APP, "version": VERSION, "installed": time.strftime("%Y-%m-%d %H:%M"), "items": items,
             "data": DATA_ITEMS}, indent=2), encoding="utf-8")

    def launch(self):
        subprocess.Popen([str(self.target / ".venv" / "Scripts" / "pythonw.exe"), str(self.target / "main.py")],
                         cwd=self.target, creationflags=NO_WINDOW)


def registered_dir():
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP}")
        loc = Path(winreg.QueryValueEx(key, "InstallLocation")[0])
        winreg.CloseKey(key)
        return loc
    except OSError:
        return None


def find_install(explicit=None) -> Path | None:
    """The installation to remove: --dir, else the folder this uninstaller sits in, else the registered one."""
    cands = [Path(explicit)] if explicit else []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        cands += [exe_dir, exe_dir.parent]  # Uninstall SafetyVision.exe / installer\SafetyVision-Setup.exe
    cands += [registered_dir(), DEFAULT_DIR]
    return next((c for c in cands if c and is_app_dir(c)), None)


def _ps(cmd) -> str:
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
                          capture_output=True, text=True, creationflags=NO_WINDOW).stdout


def stop_running(target: Path, log=print):
    """Stop only processes running THIS installation's main.py / Python environment."""
    t = str(target).replace("'", "''").rstrip("\\")
    out = _ps("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -like "
              f"'*{t}\\main.py*' -or $_.CommandLine -like '*{t}\\.venv\\*') }} | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
    n = len([x for x in out.split() if x.strip().isdigit()])
    if n:
        log(f"  stopped {n} running SafetyVision process(es)")
        time.sleep(2)


def _shortcut_points_to(lnk: Path, target: Path) -> bool:
    out = _ps(f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}'); $s.TargetPath + ' ' + $s.Arguments")
    return str(target).lower().rstrip("\\") in out.lower()


def uninstall(target: Path, delete_data=False, log=print):
    import json
    import winreg
    target = Path(target)
    if not is_app_dir(target):
        raise RuntimeError(f"{target} is not a {APP} installation - nothing removed")
    man = target / "installer" / MANIFEST
    info = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
    items = [n for n in info.get("items", APP_ITEMS) if n in APP_ITEMS]  # never trust more than the known list
    if delete_data:
        items += [n for n in DATA_ITEMS]
    log(f"Removing {APP} from {target}")
    stop_running(target, log)
    # shortcuts that belong to this installation
    menu = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / APP
    desktop = Path(os.environ["USERPROFILE"]) / "Desktop" / f"{APP}.lnk"
    for lnk in list(menu.glob("*.lnk")) + [desktop]:
        if lnk.exists() and _shortcut_points_to(lnk, target):
            lnk.unlink(missing_ok=True)
            log(f"  removed shortcut {lnk.name}")
    if menu.exists() and not any(menu.iterdir()):
        menu.rmdir()
    # registry: uninstall entry and Windows start-up only if they point here
    if registered_dir() and registered_dir().resolve() == target.resolve():
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP}")
            log("  removed from Windows 'Apps & features'")
        except OSError:
            pass
    try:
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                           winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE)
        val = winreg.QueryValueEx(k, APP)[0]
        if str(target).lower() in val.lower():
            winreg.DeleteValue(k, APP)
            log("  removed from Windows start-up")
        winreg.CloseKey(k)
    except OSError:
        pass
    # files: only the listed items, only direct children of the installation folder
    me = Path(sys.executable).resolve() if getattr(sys, "frozen", False) else None
    for name in items:
        p = target / name
        if not p.exists() or p.parent.resolve() != target.resolve():
            continue
        if me and (p.resolve() == me or me.is_relative_to(p.resolve())):
            continue  # running copy (should not happen - we relaunch from TEMP)
        log(f"  deleting {name}")
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    kept = [n for n in DATA_ITEMS if (target / n).exists()]
    if target.parent != target:
        try:
            target.rmdir()  # only succeeds when empty
            log(f"  removed folder {target}")
        except OSError:
            pass
    log(f"{APP} has been removed." + (f" Your data was kept in {target} ({', '.join(kept)})." if kept else ""))
    return kept


def relaunch_from_temp(args) -> bool:
    """A running .exe cannot delete itself: copy the uninstaller to TEMP and run it from there."""
    if not getattr(sys, "frozen", False) or args.from_temp:
        return False
    target = find_install(args.dir if args.dir != str(DEFAULT_DIR) else None)
    me = Path(sys.executable).resolve()
    if not target or not me.is_relative_to(target.resolve()):
        return False
    tmp = Path(tempfile.gettempdir()) / f"{APP}-uninstall-{os.getpid()}.exe"
    shutil.copy2(me, tmp)
    extra = ["--silent"] if args.silent else []
    extra += ["--delete-data"] if args.delete_data else []
    subprocess.Popen([str(tmp), "--uninstall", "--from-temp", "--dir", str(target), *extra], close_fds=True)
    return True


def delete_self_later():
    if getattr(sys, "frozen", False) and Path(sys.executable).parent.resolve() == Path(tempfile.gettempdir()).resolve():
        subprocess.Popen(f'cmd /c ping 127.0.0.1 -n 4 >nul & del "{sys.executable}"', shell=True,
                         creationflags=NO_WINDOW | 0x00000008)  # DETACHED_PROCESS


def uninstall_gui(args):
    import tkinter as tk
    from tkinter import messagebox, ttk
    target = find_install(args.dir if args.dir != str(DEFAULT_DIR) else None)
    root = tk.Tk()
    root.title(f"Uninstall {APP}")
    root.geometry("640x430")
    root.configure(bg="#0d1117")
    ico = payload_dir() / "assets" / "icon.ico"
    if ico.exists():
        root.iconbitmap(str(ico))
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background="#0d1117", foreground="#e6edf3")
    style.configure("TCheckbutton", background="#0d1117", foreground="#e6edf3")
    style.map("TCheckbutton", background=[("active", "#0d1117")])
    style.configure("Danger.TButton", background="#b62324", foreground="white", padding=8, borderwidth=0)
    style.map("Danger.TButton", background=[("active", "#da3633"), ("disabled", "#30363d")])
    style.configure("TButton", background="#21262d", foreground="#e6edf3", padding=6, borderwidth=0)
    tk.Label(root, text=f"Uninstall {APP}", font=("Segoe UI", 17, "bold"), fg="#e6edf3", bg="#0d1117").pack(
        anchor="w", padx=20, pady=(18, 4))
    if not target:
        tk.Label(root, text=f"No {APP} installation was found.", fg="#ff7b72", bg="#0d1117").pack(anchor="w", padx=20)
        ttk.Button(root, text="Close", command=root.destroy).pack(pady=20)
        root.mainloop()
        delete_self_later()
        return
    tk.Label(root, text=f"Installed in: {target}", fg="#8b949e", bg="#0d1117", font=("Segoe UI", 10)).pack(
        anchor="w", padx=20)
    tk.Label(root, text="Removes the program, its Python environment, shortcuts and the start-up entry. "
                        "Only SafetyVision's own files are touched.", wraplength=590, justify="left",
             fg="#c9d1d9", bg="#0d1117").pack(anchor="w", padx=20, pady=(10, 4))
    data_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(root, text="Also delete my data: settings (config.yaml) and the data folder - events, snapshots, "
                               "people photos, AI training data", variable=data_var).pack(anchor="w", padx=20, pady=4)
    tk.Label(root, text="Keep it unticked to reinstall later and continue where you left off. The local AI (Ollama) "
                        "is shared with other programs and is not removed.", wraplength=590, justify="left",
             fg="#8b949e", bg="#0d1117").pack(anchor="w", padx=20)
    logbox = tk.Text(root, height=8, bg="#010409", fg="#8b949e", relief="flat", font=("Consolas", 9))
    logbox.pack(fill="both", expand=True, padx=20, pady=10)
    row = tk.Frame(root, bg="#0d1117")
    row.pack(fill="x", padx=20, pady=(0, 16))
    q: "queue.Queue" = queue.Queue()

    def work():
        try:
            uninstall(target, data_var.get(), log=lambda s: q.put(("log", s)))
            q.put(("done", None))
        except Exception as e:
            q.put(("done", str(e)))

    def go():
        if data_var.get() and not messagebox.askyesno(APP, "Delete all SafetyVision data too? This cannot be undone."):
            return
        btn.configure(state="disabled", text="Removing ...")
        threading.Thread(target=work, daemon=True).start()

    def pump():
        while not q.empty():
            kind, val = q.get()
            if kind == "log":
                logbox.insert("end", val + "\n")
                logbox.see("end")
            else:
                btn.configure(text="Close", state="normal", command=root.destroy, style="TButton")
                cancel.pack_forget()
                (messagebox.showerror(APP, val) if val else messagebox.showinfo(APP, f"{APP} has been removed."))
        root.after(150, pump)

    btn = ttk.Button(row, text="Uninstall", style="Danger.TButton", command=go)
    cancel = ttk.Button(row, text="Cancel", command=root.destroy)
    cancel.pack(side="right")
    btn.pack(side="right", padx=8)
    pump()
    root.mainloop()
    delete_self_later()


# ============================================================================ GUI
def gui(args):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title(f"{APP} Setup")
    root.geometry("720x560")
    root.configure(bg="#0d1117")
    ico = payload_dir() / "assets" / "icon.ico"
    if ico.exists():
        root.iconbitmap(str(ico))
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background="#0d1117", foreground="#e6edf3", fieldbackground="#161b22")
    style.configure("TCheckbutton", background="#0d1117", foreground="#e6edf3")
    style.map("TCheckbutton", background=[("active", "#0d1117")])
    style.configure("Accent.TButton", background="#238636", foreground="white", padding=8, borderwidth=0)
    style.map("Accent.TButton", background=[("active", "#2ea043"), ("disabled", "#30363d")])
    style.configure("TButton", background="#21262d", foreground="#e6edf3", padding=6, borderwidth=0)
    style.configure("Horizontal.TProgressbar", troughcolor="#161b22", background="#238636")


    head = tk.Frame(root, bg="#0d1117")
    head.pack(fill="x", padx=20, pady=(18, 6))
    png = payload_dir() / "assets" / "icon.png"
    if png.exists():
        img = tk.PhotoImage(file=str(png)).subsample(4, 4)
        lbl = tk.Label(head, image=img, bg="#0d1117")
        lbl.image = img
        lbl.pack(side="left")
    tf = tk.Frame(head, bg="#0d1117")
    tf.pack(side="left", padx=12)
    tk.Label(tf, text=f"{APP} {VERSION}", font=("Segoe UI", 18, "bold"), fg="#e6edf3", bg="#0d1117").pack(anchor="w")
    tk.Label(tf, text="AI safety monitoring for XVR / NVR cameras - fire, PPE, man count, man down",
             font=("Segoe UI", 10), fg="#8b949e", bg="#0d1117").pack(anchor="w")

    opts = tk.Frame(root, bg="#0d1117")
    opts.pack(fill="x", padx=20, pady=6)
    tk.Label(opts, text="Install folder", fg="#8b949e", bg="#0d1117").grid(row=0, column=0, sticky="w")
    dir_var = tk.StringVar(value=str(DEFAULT_DIR))
    ent = tk.Entry(opts, textvariable=dir_var, bg="#161b22", fg="#e6edf3", insertbackground="#e6edf3", relief="flat",
                   width=70)
    ent.grid(row=1, column=0, sticky="we", ipady=4)
    ttk.Button(opts, text="Browse...", command=lambda: dir_var.set(
        filedialog.askdirectory(initialdir=dir_var.get()) or dir_var.get())).grid(row=1, column=1, padx=6)
    ai_var, desk_var, auto_var, launch_var = (tk.BooleanVar(value=v) for v in (True, True, False, True))
    ttk.Checkbutton(opts, text="Install the local AI (Ollama + vision model, about 6 GB download)",
                    variable=ai_var).grid(row=2, column=0, sticky="w", pady=(10, 0))
    ttk.Checkbutton(opts, text="Create a desktop shortcut", variable=desk_var).grid(row=3, column=0, sticky="w")
    ttk.Checkbutton(opts, text="Start SafetyVision when Windows starts", variable=auto_var).grid(row=4, column=0,
                                                                                                sticky="w")
    ttk.Checkbutton(opts, text="Launch SafetyVision when finished", variable=launch_var).grid(row=5, column=0,
                                                                                             sticky="w")
    pb = ttk.Progressbar(root, mode="determinate", maximum=1.0)
    pb.pack(fill="x", padx=20, pady=(10, 4))
    logbox = tk.Text(root, height=14, bg="#010409", fg="#8b949e", relief="flat", font=("Consolas", 9))
    logbox.pack(fill="both", expand=True, padx=20, pady=4)
    btn = ttk.Button(root, text="Install", style="Accent.TButton")
    btn.pack(pady=(4, 16))
    q: "queue.Queue" = queue.Queue()

    def pump():
        while not q.empty():
            kind, val = q.get()
            if kind == "log":
                logbox.insert("end", val + "\n")
                logbox.see("end")
            elif kind == "p":
                pb["value"] = val
            elif kind == "done":
                btn.configure(text="Close", state="normal", command=root.destroy)
                if val is None:
                    messagebox.showinfo(APP, f"{APP} is installed.\nStart it from the desktop or Start menu.")
                else:
                    messagebox.showerror(APP, f"Installation failed:\n{val}\n\nSee the log for details. "
                                              "You can run the setup again - finished steps are skipped.")
        root.after(150, pump)

    def work():
        q.put(("log", f"Install folder: {normalize_target(dir_var.get())}"))
        inst = Installer(Path(dir_var.get()), ai_var.get(), desk_var.get(), auto_var.get(),
                         log=lambda s: q.put(("log", s)), progress=lambda p: q.put(("p", p)))
        try:
            inst.install()
            if launch_var.get():
                inst.launch()
            q.put(("done", None))
        except Exception as e:
            q.put(("log", f"ERROR: {e}"))
            q.put(("done", str(e)))

    def start():
        btn.configure(state="disabled", text="Installing...")
        threading.Thread(target=work, daemon=True).start()

    btn.configure(command=start)
    pump()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--silent", action="store_true", help="install without UI")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--no-local-ai", action="store_true")
    ap.add_argument("--no-shortcuts", action="store_true")
    ap.add_argument("--delete-data", action="store_true", help="with --uninstall: also delete settings and data")
    ap.add_argument("--from-temp", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if getattr(sys, "frozen", False) and Path(sys.executable).name.lower().startswith("uninstall"):
        args.uninstall = True
    if args.uninstall:
        if relaunch_from_temp(args):
            return
        if args.silent:
            target = find_install(args.dir if args.dir != str(DEFAULT_DIR) else None)
            if target:
                uninstall(target, args.delete_data)
            delete_self_later()
            return
        uninstall_gui(args)
        return
    if args.silent:
        inst = Installer(Path(args.dir), not args.no_local_ai, not args.no_shortcuts)
        if args.no_shortcuts:
            inst.shortcuts = lambda: None
        inst.install()
        return
    gui(args)


if __name__ == "__main__":
    main()
