# Диктовка – русский голосовой ввод в любое окно (Kimi, браузер, мессенджеры).
# F8 – начать запись, F8 ещё раз – распознать и вставить текст туда, где стоит курсор.
# Распознаёт локально: Whisper (faster-whisper) на видеокарте NVIDIA или на процессоре.

import os, sys, json, time, threading, queue, ctypes, traceback

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # без пугающего «unauthenticated requests»
import warnings; warnings.filterwarnings("ignore")
LOG = os.path.join(BASE, "diktovka.log")
CFG_PATH = os.path.join(BASE, "config.json")

DEFAULT_CFG = {
    "hotkey": "f8",              # клавиша вкл/выкл записи
    "language": "ru",
    "model": "large-v3-turbo",   # на слабом ноутбуке без видеокарты можно "small"
    "device": "auto",            # auto / cuda / cpu
    "beep": True,                # звук при старте и остановке записи
    "auto_send": False,          # True – после вставки нажать Enter (сразу отправить)
    "prompt": "Привет. Это голосовое сообщение на русском языке, с точками и запятыми.",
}


def log(*a):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
    except Exception:
        pass


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH, encoding="utf-8-sig") as f:
                cfg.update(json.load(f))
        except Exception as e:
            log("config.json не прочитан:", e)
    else:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CFG, f, ensure_ascii=False, indent=2)
    return cfg


def add_cuda_dlls():
    # Библиотеки CUDA ставятся pip-пакетами nvidia-* – подсказываем Windows, где их искать
    try:
        import site
        roots = site.getsitepackages() + [site.getusersitepackages()]
    except Exception:
        roots = []
    roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    for r in roots:
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            d = os.path.join(r, "nvidia", sub, "bin")
            if os.path.isdir(d):
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass


def load_model(cfg):
    add_cuda_dlls()
    import ctranslate2
    from faster_whisper import WhisperModel
    dev = cfg["device"]
    if dev == "auto":
        try:
            dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            dev = "cpu"
    if dev == "cuda":
        try:
            m = WhisperModel(cfg["model"], device="cuda", compute_type="float16")
            warm(m, cfg)
            return m, "видеокарта"
        except Exception as e:
            log("CUDA не завелась, перехожу на процессор:", e)
    threads = max(2, (os.cpu_count() or 4) - 1)
    m = WhisperModel(cfg["model"], device="cpu", compute_type="int8", cpu_threads=threads)
    return m, "процессор"


def warm(model, cfg):
    import numpy as np
    list(model.transcribe(np.zeros(16000, dtype="float32"), language=cfg["language"])[0])


def transcribe(model, audio, cfg):
    segs, _ = model.transcribe(
        audio, language=cfg["language"], beam_size=5, vad_filter=True,
        initial_prompt=cfg.get("prompt") or None, condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segs).strip()


# ---------- окошко-индикатор (не забирает фокус у Kimi) ----------

class Overlay:
    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)
        self.root.configure(bg="#2B292E")
        self.label = tk.Label(self.root, text="", fg="white", bg="#2B292E",
                              font=("Segoe UI", 12, "bold"), padx=18, pady=9)
        self.label.pack()
        self.q = queue.Queue()
        self.hide_at = None
        self.root.update_idletasks()
        self._no_activate()
        self.root.after(50, self._poll)

    def _no_activate(self):
        try:
            u = ctypes.windll.user32
            hwnd = u.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            GWL_EXSTYLE = -20
            style = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
            # NOACTIVATE | TOOLWINDOW | TRANSPARENT (клики насквозь) | TOPMOST
            u.SetWindowLongW(hwnd, GWL_EXSTYLE, style | 0x08000000 | 0x80 | 0x20 | 0x08)
        except Exception as e:
            log("overlay style:", e)

    def show(self, text, color, seconds=None):
        self.q.put((text, color, seconds))

    def hide(self):
        self.q.put(None)

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item is None:
                    self.root.attributes("-alpha", 0.0)
                    self.hide_at = None
                    continue
                text, color, seconds = item
                self.label.configure(text=text, bg=color)
                self.root.configure(bg=color)
                self.root.update_idletasks()
                w, h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
                sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
                self.root.geometry(f"+{(sw - w) // 2}+{sh - h - 90}")
                self.root.attributes("-alpha", 0.93)
                self.hide_at = time.time() + seconds if seconds else None
        except queue.Empty:
            pass
        if self.hide_at and time.time() > self.hide_at:
            self.root.attributes("-alpha", 0.0)
            self.hide_at = None
        self.root.after(50, self._poll)


# ---------- основная логика ----------

class App:
    def __init__(self, cfg, overlay):
        self.cfg = cfg
        self.ov = overlay
        self.model = None
        self.recording = False
        self.busy = False
        self.chunks = []
        self.stream = None
        self.target = None
        self.lock = threading.Lock()
        self.key = cfg["hotkey"].upper()

    def beep(self, f, ms):
        if self.cfg.get("beep"):
            import winsound
            threading.Thread(target=winsound.Beep, args=(f, ms), daemon=True).start()

    def toggle(self):
        with self.lock:
            if self.busy:
                return
            if self.model is None:
                self.ov.show("Модель ещё загружается, секунду…", "#8A6D1F", 2)
                return
            if not self.recording:
                self.start()
            else:
                self.recording = False
                self.busy = True
                threading.Thread(target=self.finish, daemon=True).start()

    def start(self):
        import sounddevice as sd
        self.target = ctypes.windll.user32.GetForegroundWindow()
        self.chunks = []

        def cb(indata, frames, t, status):
            self.chunks.append(indata[:, 0].copy())

        try:
            self.stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=cb)
            self.stream.start()
        except Exception as e:
            log("микрофон:", e)
            self.ov.show("Не вижу микрофон – проверь его в Windows", "#B3261E", 4)
            return
        self.recording = True
        log("запись началась")
        self.beep(880, 90)
        self.ov.show(f"●  Говорите…   {self.key} – готово", "#C62828")

    def finish(self):
        import numpy as np
        try:
            self.stream.stop(); self.stream.close()
            self.beep(660, 90)
            audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype="float32")
            log(f"запись остановлена, {len(audio)/16000:.1f} с")
            if len(audio) < 16000 * 0.3:
                self.ov.show("Пусто – ничего не записалось", "#555555", 2)
                return
            self.ov.show("Распознаю…", "#567FE8")
            t0 = time.time()
            text = transcribe(self.model, audio, self.cfg)
            log(f"{len(audio)/16000:.1f} с речи → {time.time()-t0:.1f} с: {text}")
            if not text:
                self.ov.show("Не расслышал, попробуй ещё раз", "#555555", 2)
                return
            self.paste(text)
            self.ov.hide()
        except Exception:
            log(traceback.format_exc())
            self.ov.show("Ошибка – подробности в diktovka.log", "#B3261E", 4)
        finally:
            self.busy = False

    def paste(self, text):
        import pyperclip
        u = ctypes.windll.user32
        try:
            old = pyperclip.paste()
        except Exception:
            old = None
        pyperclip.copy(text)
        if self.target and u.GetForegroundWindow() != self.target:
            u.SetForegroundWindow(self.target)
            time.sleep(0.15)
        time.sleep(0.05)
        press([0x11, 0x56])          # Ctrl+V
        if self.cfg.get("auto_send"):
            time.sleep(0.15)
            press([0x0D])            # Enter
        time.sleep(0.4)
        if old is not None:
            try:
                pyperclip.copy(old)
            except Exception:
                pass

    def load(self):
        try:
            self.ov.show("Диктовка: загружаю модель…", "#555555")
            t0 = time.time()
            self.model, where = load_model(self.cfg)
            log(f"модель {self.cfg['model']} загружена на {where} за {time.time()-t0:.1f} с")
            self.ov.show(f"Диктовка готова – жми {self.key}  ({where})", "#2E7D32", 4)
        except Exception:
            log(traceback.format_exc())
            self.ov.show("Модель не загрузилась – смотри diktovka.log", "#B3261E", 8)


def press(vks):
    # нажать и отпустить сочетание клавиш (виртуальные коды Windows)
    kb = ctypes.windll.user32.keybd_event
    for vk in vks:
        kb(vk, 0, 0, 0)
    for vk in reversed(vks):
        kb(vk, 0, 2, 0)  # KEYEVENTF_KEYUP


MODS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
NAMED = {"space": 0x20, "pause": 0x13, "insert": 0x2D, "home": 0x24, "end": 0x23,
         "pageup": 0x21, "pagedown": 0x22, "scrolllock": 0x91, "capslock": 0x14,
         "`": 0xC0, "tab": 0x09, "enter": 0x0D}


def parse_hotkey(s):
    mods, vk = 0, None
    for part in s.lower().replace(" ", "").split("+"):
        if part in MODS:
            mods |= MODS[part]
        elif part in NAMED:
            vk = NAMED[part]
        elif len(part) >= 2 and part[0] == "f" and part[1:].isdigit():
            vk = 0x6F + int(part[1:])          # F1 = 0x70
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
    if vk is None:
        raise ValueError(f"Не понял клавишу «{s}» в config.json")
    return mods, vk


def hotkey_loop(app):
    # Системная горячая клавиша Windows: в отличие от перехвата клавиатуры,
    # не отключается после сна, блокировки экрана или подвисания.
    from ctypes import wintypes
    u = ctypes.windll.user32
    try:
        mods, vk = parse_hotkey(app.cfg["hotkey"])
    except Exception as e:
        log(e); app.ov.show(str(e), "#B3261E", 8); return
    if not u.RegisterHotKey(None, 1, mods | 0x4000, vk):   # 0x4000 – без автоповтора
        log("RegisterHotKey не удалось, клавиша занята:", app.cfg["hotkey"])
        app.ov.show(f"{app.key} занята другой программой – смени клавишу в config.json", "#B3261E", 10)
        return
    msg = wintypes.MSG()
    while u.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        if msg.message == 0x0312:  # WM_HOTKEY
            app.toggle()


def tray(app):
    try:
        import pystray
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((2, 2, 62, 62), fill="#567FE8")
        d.rounded_rectangle((24, 12, 40, 40), 8, fill="white")
        d.arc((16, 22, 48, 48), 0, 180, fill="white", width=4)
        d.line((32, 48, 32, 54), fill="white", width=4)

        def open_cfg(icon, item):
            os.startfile(CFG_PATH)

        def quit_(icon, item):
            icon.stop()
            os._exit(0)

        icon = pystray.Icon("diktovka", img, f"Диктовка ({app.key})", menu=pystray.Menu(
            pystray.MenuItem(f"Запись: {app.key}", lambda i, it: app.toggle(), default=True),
            pystray.MenuItem("Настройки (config.json)", open_cfg),
            pystray.MenuItem("Выход", quit_),
        ))
        icon.run_detached()
    except Exception as e:
        log("трей:", e)


def single_instance():
    ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\Diktovka_AIM")
    return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def main():
    cfg = load_cfg()

    if "--download" in sys.argv:          # вызывается установщиком: скачать модель и проверить
        m, where = load_model(cfg)
        print(f"OK: модель {cfg['model']} работает, считает {where}")
        return
    if "--test" in sys.argv:              # проверка на файле: --test file.wav
        import numpy as np, subprocess
        wav = sys.argv[sys.argv.index("--test") + 1]
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", wav, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                             capture_output=True).stdout
        audio = np.frombuffer(raw, dtype="float32")
        m, where = load_model(cfg)
        t0 = time.time()
        text = transcribe(m, audio, cfg)
        print(f"{where}, {len(audio)/16000:.1f} с речи распознано за {time.time()-t0:.1f} с: {text}")
        return

    if not single_instance():
        ctypes.windll.user32.MessageBoxW(None, "Диктовка уже запущена – значок микрофона возле часов.", "Диктовка", 0x40)
        return

    ov = Overlay()
    app = App(cfg, ov)
    threading.Thread(target=hotkey_loop, args=(app,), daemon=True).start()
    threading.Thread(target=app.load, daemon=True).start()
    tray(app)
    log("запущена, клавиша", cfg["hotkey"])
    ov.root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
