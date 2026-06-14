# -*- coding: utf-8 -*-
"""
AutoSRT - Yapay Zeka Destekli Altyazı Motoru
=============================================
faster-whisper (CUDA/float16) tabanlı, mikro-dilimli SRT üretici masaüstü uygulaması.

Mimari notlar:
- Whisper işlemi ASLA ana GUI thread'inde çalışmaz -> threading.Thread ile arka plana alınır.
- Thread <-> GUI iletişimi thread-safe bir queue.Queue üzerinden yapılır; arayüz bu kuyruğu
  after() ile periyodik boşaltır (Tkinter thread-safe değildir).
- Çekirdek transcribe mantığı (DLL bulma, format_timestamp, mikro-dilimleme) BİREBİR korundu;
  varsayılan değerlerde (karakter sınırı=45, min süre=0, sadece-cümle kapalı) çıktı orijinalle aynıdır.
- Konsol (CMD) penceresi açılmaz: pythonw ile çalıştırılır; python.exe ile başlatılırsa
  kendi sahibi olduğu konsolu otomatik gizler.
- İşlem bitince Windows bildirimi gösterilir (tıklayınca çıktı klasörünü açar).
"""

import os
import sys
import site
import math
import re
import time
import queue
import threading
import gc


# --- KONSOL (CMD) PENCERESINI GIZLE ---------------------------------------
def _hide_own_console():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            return
        arr = (ctypes.c_uint32 * 4)()
        count = kernel32.GetConsoleProcessList(arr, 4)
        if count <= 1:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


_hide_own_console()


# --- GPU DLL OTONOM ENTEGRASYONU (BIREBIR KORUNDU) ------------------------
for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
    for lib in ["cublas", "cudnn"]:
        bin_path = os.path.join(site_dir, "nvidia", lib, "bin")
        if os.path.exists(bin_path):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(bin_path)
# --------------------------------------------------------------------------

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD


# --- ZAMAN DAMGASI FORMATLAMA (BIREBIR KORUNDU) ---
def format_timestamp(seconds: float):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _clock(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --- MODEL KONFIGURASYONU (BIREBIR KORUNDU) ---
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MODEL_SIZES = ["large-v3", "large-v2", "medium", "small", "base", "tiny"]

VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".wmv", ".m4v",
    ".mpg", ".mpeg", ".ts", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg",
)

# --- PALET: her renk (açık, koyu) çifti; ctk appearance moduyla otomatik değişir ---
BG          = ("#f4f5f7", "#0a0a0d")
SIDEBAR     = ("#ffffff", "#0f0f14")
SURFACE     = ("#ffffff", "#121218")
SURFACE2    = ("#e7eaf0", "#17171f")
HOVER       = ("#dadfe7", "#1d1d27")
BORDER      = ("#c4ccd8", "#242430")
BORDER_SOFT = ("#ccd4df", "#1b1b24")
TEXT        = ("#16181d", "#f3f4f6")
MUTED       = ("#5b6472", "#8a93a3")
FAINT       = ("#8b94a3", "#45454f")

GREEN = "#16a34a"
RED   = "#f43f5e"
WARN  = "#d97706"
# Günlük paneli: açık modda yumuşak gri, koyu modda gömülü terminal koyusu
TERM_BG    = ("#e9edf3", "#0c0c10")
TERM_TXT   = ("#15803d", "#9af2bd")
TERM_PANEL = ("#dfe4ec", "#13131a")   # panel başlık/temizle düğmesi zemini

FONT_UI   = "Segoe UI"
FONT_MONO = "Consolas"

ACCENTS = {
    "indigo":  ("#6366f1", "#4f46e5"),
    "emerald": ("#10b981", "#059669"),
    "sky":     ("#0ea5e9", "#0284c7"),
    "rose":    ("#fb7185", "#f43f5e"),
}

# --- DİL METİNLERİ (TR / EN) ---
L = {
    "tr": {
        "subtitle": "Yapay Zeka Destekli Altyazı Motoru",
        "model": "MODEL", "language": "DİL", "theme": "TEMA", "appearance": "GÖRÜNÜM",
        "sub_group": "ALTYAZI AYARLARI",
        "char_limit": "Karakter sınırı", "min_dur": "Min. süre", "sec": "sn",
        "sentence_only": "Cümle sonunda böl",
        "drop_title": "Videoyu buraya sürükleyip bırakın",
        "or": "veya", "browse": "Video Seç",
        "start": "Altyazı Oluştur", "need_file": "Önce bir video seçin",
        "stop": "Durdur", "stopping": "Durduruluyor…",
        "open_folder": "Klasörü Aç", "log": "Günlük", "clear": "Temizle",
        "st_ready": "Hazır", "st_loading": "Model yükleniyor",
        "st_analyzing": "Ses analiz ediliyor", "st_generating": "Altyazılar oluşturuluyor",
        "st_done": "Tamamlandı", "st_stopped": "Durduruldu", "st_error": "Hata",
        "ready_msg": "AutoSRT hazır. Bir video seçin veya sürükleyin.",
        "starting": "İşlem başlatılıyor… (Arayüz kilitlenmez, arka planda çalışır)",
        "loading_model": "Model yükleniyor: {} ({}/{})",
        "first_use": "(İlk kullanımda model indirilebilir, lütfen bekleyin…)",
        "model_ready": "Model hazır.",
        "settings_line": "Ayarlar → karakter sınırı: {}, min süre: {} sn, cümle sonu: {}",
        "on": "açık", "off": "kapalı",
        "detected": "Algılanan dil: {} | Süre: {}",
        "done_blocks": "TAMAMLANDI! {} altyazı bloğu oluşturuldu.",
        "saved": "Kaydedildi: {}",
        "cancelled": "İşlem kullanıcı tarafından durduruldu.",
        "err_no_file": "HATA: Önce bir video seçin veya sürükleyin.",
        "cpu_fb": "UYARI: CUDA cihazı bulunamadı → CPU/int8 moduna geçildi.",
        "stats": "Geçen {} • Kalan ~{} • {} blok • Dil: {}",
        "unsupported": "UYARI: Desteklenmeyen uzantı olabilir → {}",
        "not_found": "HATA: Dosya bulunamadı → {}",
        "open_fail": "Klasör açılamadı: {}",
        "dev_gpu": "GPU hazır", "dev_cpu": "CPU modu", "dev_check": "Donanım denetleniyor…",
        "selected": "Seçilen video: {}",
        "notify_title": "Altyazı hazır ✓", "notify_msg": "{} oluşturuldu — açmak için tıklayın",
    },
    "en": {
        "subtitle": "AI-Powered Subtitle Engine",
        "model": "MODEL", "language": "LANGUAGE", "theme": "THEME", "appearance": "APPEARANCE",
        "sub_group": "SUBTITLE SETTINGS",
        "char_limit": "Character limit", "min_dur": "Min. duration", "sec": "s",
        "sentence_only": "Split at sentence end",
        "drop_title": "Drag & drop your video here",
        "or": "or", "browse": "Choose Video",
        "start": "Generate Subtitles", "need_file": "Select a video first",
        "stop": "Stop", "stopping": "Stopping…",
        "open_folder": "Open Folder", "log": "Log", "clear": "Clear",
        "st_ready": "Ready", "st_loading": "Loading model",
        "st_analyzing": "Analyzing audio", "st_generating": "Generating subtitles",
        "st_done": "Completed", "st_stopped": "Stopped", "st_error": "Error",
        "ready_msg": "AutoSRT is ready. Choose or drop a video.",
        "starting": "Starting… (UI stays responsive, runs in background)",
        "loading_model": "Loading model: {} ({}/{})",
        "first_use": "(Model may download on first use, please wait…)",
        "model_ready": "Model ready.",
        "settings_line": "Settings → char limit: {}, min duration: {} s, sentence-end: {}",
        "on": "on", "off": "off",
        "detected": "Detected language: {} | Duration: {}",
        "done_blocks": "DONE! {} subtitle blocks created.",
        "saved": "Saved: {}",
        "cancelled": "Stopped by the user.",
        "err_no_file": "ERROR: Choose or drop a video first.",
        "cpu_fb": "WARNING: No CUDA device found → switched to CPU/int8.",
        "stats": "Elapsed {} • ETA ~{} • {} blocks • Lang: {}",
        "unsupported": "WARNING: Possibly unsupported extension → {}",
        "not_found": "ERROR: File not found → {}",
        "open_fail": "Could not open folder: {}",
        "dev_gpu": "GPU ready", "dev_cpu": "CPU mode", "dev_check": "Checking hardware…",
        "selected": "Selected video: {}",
        "notify_title": "Subtitles ready ✓", "notify_msg": "{} created — click to open",
    },
}


class _Cancelled(Exception):
    pass


class _LogStream:
    """sys.stdout/stderr yerine geçer; çıktıyı log kuyruğuna aktarır."""
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("raw", s))
        return len(s) if s else 0

    def flush(self):
        pass

    def isatty(self):
        return False


class AutoSRTApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)

        self.video_path = None
        self.is_running = False
        self.model = None
        self._model_key = None
        self.log_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.last_srt = None
        self.lang = "tr"
        self.accent, self.accent_hover = ACCENTS["indigo"]
        self._pb_mode = None
        self._status_key = "st_ready"
        self._status_color = GREEN
        self._last_stats = None

        sys.stdout = _LogStream(self.log_queue)
        sys.stderr = _LogStream(self.log_queue)

        ctk.set_appearance_mode("dark")
        self.title("AutoSRT")
        self.geometry("1100x800")
        self.minsize(960, 600)
        self.configure(fg_color=BG)
        self._set_window_icon()
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()
        self._apply_accent(*ACCENTS["indigo"])
        self._log_key("ready_msg")
        self._update_primary()

        self.after(80, self._poll_log_queue)
        threading.Thread(target=self._detect_device, daemon=True).start()

    def _set_window_icon(self):
        """Pencere + gorev cubugu ikonunu icon.ico'dan ayarlar."""
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            ico = os.path.join(base, "icon.ico")
            if os.path.exists(ico):
                # Gorev cubugunda dogru gruplama + ikon icin AppUserModelID
                try:
                    import ctypes
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AutoSRT")
                except Exception:
                    pass
                self.iconbitmap(ico)
                # CTk bazen ikonu geciktirerek ezer; bir kez daha uygula.
                self.after(300, lambda: self.iconbitmap(ico))
        except Exception:
            pass

    def t(self, key, *args):
        s = L[self.lang].get(key, key)
        return s.format(*args) if args else s

    # ===================================================================== KENAR ÇUBUĞU
    def _build_sidebar(self):
        # Kaydırılabilir kenar çubuğu: her ekran yüksekliği/DPI'da tüm kontroller erişilir
        sb = ctk.CTkScrollableFrame(self, width=258, corner_radius=0, fg_color=SIDEBAR,
                                    scrollbar_button_color=BORDER,
                                    scrollbar_button_hover_color=MUTED)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_columnconfigure(0, weight=1)
        r = 0

        # Marka
        brand = ctk.CTkFrame(sb, fg_color="transparent")
        brand.grid(row=r, column=0, sticky="ew", padx=20, pady=(20, 2)); r += 1
        self.logo_dot = ctk.CTkLabel(brand, text="◆", font=ctk.CTkFont(FONT_UI, 22))
        self.logo_dot.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(brand, text="AutoSRT",
                     font=ctk.CTkFont(FONT_UI, 22, weight="bold"),
                     text_color=TEXT).pack(side="left")
        self.brand_rule = ctk.CTkFrame(sb, height=2, corner_radius=2, fg_color=self.accent)
        self.brand_rule.grid(row=r, column=0, sticky="ew", padx=20, pady=(2, 16)); r += 1

        # MODEL
        self.cap_model = self._caption(sb, r, "model"); r += 1
        self.model_var = ctk.StringVar(value=MODEL_SIZE)
        self.model_menu = ctk.CTkOptionMenu(
            sb, values=MODEL_SIZES, variable=self.model_var,
            height=36, corner_radius=10, fg_color=SURFACE2, button_color=self.accent,
            button_hover_color=self.accent_hover, dropdown_fg_color=SURFACE2,
            text_color=TEXT, dropdown_text_color=TEXT,
            font=ctk.CTkFont(FONT_UI, 13), dropdown_font=ctk.CTkFont(FONT_UI, 13))
        self.model_menu.grid(row=r, column=0, sticky="ew", padx=20, pady=(2, 14)); r += 1

        # DİL
        self.cap_lang = self._caption(sb, r, "language"); r += 1
        lang_row = ctk.CTkFrame(sb, fg_color="transparent")
        lang_row.grid(row=r, column=0, sticky="ew", padx=20, pady=(2, 14)); r += 1
        lang_row.grid_columnconfigure((0, 1), weight=1)
        self.lang_btns = {}
        for i, code in enumerate(("tr", "en")):
            b = ctk.CTkButton(lang_row, text=code.upper(), height=34, corner_radius=10,
                              font=ctk.CTkFont(FONT_UI, 13, weight="bold"),
                              command=lambda c=code: self._set_lang(c))
            b.grid(row=0, column=i, sticky="ew", padx=(0, 8) if i == 0 else 0)
            self.lang_btns[code] = b

        # ALTYAZI AYARLARI
        self.cap_sub = self._caption(sb, r, "sub_group"); r += 1
        self.char_cap, self.char_val, self.char_slider = self._slider_block(
            sb, r, "char_limit", 20, 90, 70, 45, lambda v: str(int(float(v)))); r += 1
        self.min_cap, self.min_val, self.min_slider = self._slider_block(
            sb, r, "min_dur", 0.0, 2.0, 20, 0.0,
            lambda v: f"{float(v):.1f} {self.t('sec')}"); r += 1
        self.sentence_switch = ctk.CTkSwitch(
            sb, text=self.t("sentence_only"), font=ctk.CTkFont(FONT_UI, 12),
            text_color=TEXT)
        self.sentence_switch.grid(row=r, column=0, sticky="w", padx=20, pady=(4, 14)); r += 1

        # GÖRÜNÜM (açık/koyu)
        self.cap_appear = self._caption(sb, r, "appearance"); r += 1
        self.appear_seg = ctk.CTkSegmentedButton(
            sb, values=["☾", "☀"], command=self._set_mode, height=32,
            font=ctk.CTkFont(FONT_UI, 15))
        self.appear_seg.set("☾")
        self.appear_seg.grid(row=r, column=0, sticky="ew", padx=20, pady=(2, 14)); r += 1

        # TEMA (vurgu rengi)
        self.cap_theme = self._caption(sb, r, "theme"); r += 1
        sw = ctk.CTkFrame(sb, fg_color="transparent")
        sw.grid(row=r, column=0, sticky="w", padx=20, pady=(2, 14)); r += 1
        self.swatches = {}
        for name, (col, _) in ACCENTS.items():
            b = ctk.CTkButton(sw, text="", width=26, height=26, corner_radius=13,
                              fg_color=col, hover_color=col, border_width=0,
                              command=lambda n=name: self._set_accent(n))
            b.pack(side="left", padx=(0, 10))
            self.swatches[name] = b

        ctk.CTkFrame(sb, height=1, fg_color=BORDER).grid(
            row=r, column=0, sticky="ew", padx=20, pady=(2, 12)); r += 1
        dev = ctk.CTkFrame(sb, fg_color="transparent")
        dev.grid(row=r, column=0, sticky="ew", padx=20, pady=(0, 16)); r += 1
        self.dev_dot = ctk.CTkLabel(dev, text="●", text_color=MUTED,
                                    font=ctk.CTkFont(FONT_UI, 12))
        self.dev_dot.pack(side="left", padx=(0, 8))
        self.dev_lbl = ctk.CTkLabel(dev, text=self.t("dev_check"), text_color=MUTED,
                                    font=ctk.CTkFont(FONT_UI, 12))
        self.dev_lbl.pack(side="left")
        ctk.CTkLabel(dev, text="v1.3", text_color=FAINT,
                     font=ctk.CTkFont(FONT_UI, 11)).pack(side="right")

    def _caption(self, parent, row, key):
        lbl = ctk.CTkLabel(parent, text=self.t(key),
                           font=ctk.CTkFont(FONT_UI, 11, weight="bold"), text_color=MUTED)
        lbl.grid(row=row, column=0, sticky="w", padx=20, pady=(0, 6))
        return lbl

    def _slider_block(self, parent, row, cap_key, frm, to, steps, default, value_fmt):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 8))
        f.grid_columnconfigure(0, weight=1)
        cap = ctk.CTkLabel(f, text=self.t(cap_key), font=ctk.CTkFont(FONT_UI, 12),
                           text_color=MUTED)
        cap.grid(row=0, column=0, sticky="w")
        val = ctk.CTkLabel(f, text=value_fmt(default),
                           font=ctk.CTkFont(FONT_UI, 12, weight="bold"))
        val.grid(row=0, column=1, sticky="e")
        sl = ctk.CTkSlider(f, from_=frm, to=to, number_of_steps=steps, height=16)
        sl.set(default)
        sl.configure(command=lambda v, vl=val, fn=value_fmt: vl.configure(text=fn(v)))
        sl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        return cap, val, sl

    # ===================================================================== ANA ALAN
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=36, pady=32)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(6, weight=1)

        head = ctk.CTkFrame(main, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 22))
        ctk.CTkLabel(head, text="AutoSRT",
                     font=ctk.CTkFont(FONT_UI, 30, weight="bold"),
                     text_color=TEXT).pack(anchor="w")
        self.subtitle_lbl = ctk.CTkLabel(head, text=self.t("subtitle"),
                                         font=ctk.CTkFont(FONT_UI, 14), text_color=MUTED)
        self.subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Sürükle-bırak kartı
        self.drop_frame = ctk.CTkFrame(main, height=200, corner_radius=18,
                                       border_width=2, border_color=BORDER_SOFT,
                                       fg_color=SURFACE)
        self.drop_frame.grid(row=1, column=0, sticky="ew")
        self.drop_frame.grid_propagate(False)
        self.drop_frame.grid_columnconfigure(0, weight=1)
        self.drop_frame.grid_rowconfigure((0, 4), weight=1)

        self.badge = ctk.CTkFrame(self.drop_frame, width=64, height=64, corner_radius=32,
                                  fg_color=SURFACE2)
        self.badge.grid(row=1, column=0)
        self.badge.grid_propagate(False)
        self.drop_icon = ctk.CTkLabel(self.badge, text="↓",
                                      font=ctk.CTkFont(FONT_UI, 28, weight="bold"))
        self.drop_icon.place(relx=0.5, rely=0.5, anchor="center")

        self.drop_title = ctk.CTkLabel(self.drop_frame, text=self.t("drop_title"),
                                       font=ctk.CTkFont(FONT_UI, 17, weight="bold"),
                                       text_color=TEXT)
        self.drop_title.grid(row=2, column=0, pady=(12, 2))
        self.or_lbl = ctk.CTkLabel(self.drop_frame, text=f"—  {self.t('or')}  —",
                                   font=ctk.CTkFont(FONT_UI, 12), text_color=FAINT)
        self.or_lbl.grid(row=3, column=0, pady=2)
        self.browse_btn = ctk.CTkButton(self.drop_frame, text=self.t("browse"),
                                        width=160, height=38, corner_radius=10,
                                        font=ctk.CTkFont(FONT_UI, 13, weight="bold"),
                                        command=self.browse_file)
        self.browse_btn.grid(row=4, column=0, pady=(0, 6))

        for w in (self.drop_frame, self.badge, self.drop_icon, self.drop_title):
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<Drop>>", self.on_drop)
            w.dnd_bind("<<DropEnter>>", self._on_drag_enter)
            w.dnd_bind("<<DropLeave>>", self._on_drag_leave)

        # Dosya çipi
        self.chip = ctk.CTkFrame(main, fg_color=SURFACE2, corner_radius=12,
                                 border_width=1, border_color=BORDER)
        self.chip.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.chip.grid_columnconfigure(1, weight=1)
        self.chip_icon = ctk.CTkLabel(self.chip, text="🎞",
                                      font=ctk.CTkFont(FONT_UI, 16))
        self.chip_icon.grid(row=0, column=0, rowspan=2, padx=(16, 12), pady=12)
        self.chip_name = ctk.CTkLabel(self.chip, text="", anchor="w",
                                      font=ctk.CTkFont(FONT_UI, 14, weight="bold"),
                                      text_color=TEXT)
        self.chip_name.grid(row=0, column=1, sticky="w", pady=(12, 0))
        self.chip_path = ctk.CTkLabel(self.chip, text="", anchor="w",
                                      font=ctk.CTkFont(FONT_UI, 11), text_color=MUTED)
        self.chip_path.grid(row=1, column=1, sticky="w", pady=(0, 12))
        ctk.CTkButton(self.chip, text="✕", width=34, height=34, corner_radius=8,
                      fg_color="transparent", hover_color=HOVER, text_color=MUTED,
                      font=ctk.CTkFont(FONT_UI, 14), command=self._clear_video
                      ).grid(row=0, column=2, rowspan=2, padx=(8, 12))
        self.chip.grid_remove()

        # Eylem butonları
        actions = ctk.CTkFrame(main, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        actions.grid_columnconfigure(0, weight=1)
        self.primary_btn = ctk.CTkButton(actions, text=self.t("start"), height=50,
                                         corner_radius=12,
                                         font=ctk.CTkFont(FONT_UI, 16, weight="bold"),
                                         command=self._on_primary)
        self.primary_btn.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.open_btn = ctk.CTkButton(actions, text=self.t("open_folder"), width=150,
                                      height=50, corner_radius=12, fg_color=SURFACE2,
                                      hover_color=HOVER, border_width=1, border_color=BORDER,
                                      text_color=TEXT, text_color_disabled=MUTED,
                                      font=ctk.CTkFont(FONT_UI, 13),
                                      state="disabled", command=self.open_output_folder)
        self.open_btn.grid(row=0, column=1)

        # İlerleme
        prog = ctk.CTkFrame(main, fg_color="transparent")
        prog.grid(row=4, column=0, sticky="ew", pady=(20, 0))
        prog.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(prog, height=8, corner_radius=4,
                                           fg_color=SURFACE2)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 14))
        self.progress.set(0)
        self.percent_lbl = ctk.CTkLabel(prog, text="0%", width=52,
                                        font=ctk.CTkFont(FONT_UI, 14, weight="bold"))
        self.percent_lbl.grid(row=0, column=1)

        # Durum + istatistik
        srow = ctk.CTkFrame(main, fg_color="transparent")
        srow.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        srow.grid_columnconfigure(1, weight=1)
        self.status_dot = ctk.CTkLabel(srow, text="●", text_color=GREEN,
                                       font=ctk.CTkFont(FONT_UI, 13))
        self.status_dot.grid(row=0, column=0, padx=(0, 8))
        self.status_lbl = ctk.CTkLabel(srow, text=self.t("st_ready"),
                                       font=ctk.CTkFont(FONT_UI, 13, weight="bold"),
                                       text_color=TEXT)
        self.status_lbl.grid(row=0, column=1, sticky="w")
        self.stats_lbl = ctk.CTkLabel(srow, text="", font=ctk.CTkFont(FONT_UI, 12),
                                      text_color=MUTED, anchor="e")
        self.stats_lbl.grid(row=0, column=2, sticky="e")

        # Günlük paneli
        logc = ctk.CTkFrame(main, corner_radius=16, fg_color=TERM_BG,
                            border_width=1, border_color=BORDER_SOFT)
        logc.grid(row=6, column=0, sticky="nsew", pady=(18, 0))
        logc.grid_columnconfigure(0, weight=1)
        logc.grid_rowconfigure(1, weight=1)
        lhead = ctk.CTkFrame(logc, fg_color="transparent")
        lhead.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 0))
        lhead.grid_columnconfigure(0, weight=1)
        self.log_cap = ctk.CTkLabel(lhead, text=self.t("log"),
                                    font=ctk.CTkFont(FONT_UI, 12, weight="bold"),
                                    text_color=MUTED)
        self.log_cap.grid(row=0, column=0, sticky="w")
        self.clear_btn = ctk.CTkButton(lhead, text=self.t("clear"), width=72, height=28,
                                       corner_radius=8, fg_color=TERM_PANEL, hover_color=HOVER,
                                       text_color=TEXT, border_width=1, border_color=BORDER,
                                       font=ctk.CTkFont(FONT_UI, 11), command=self._clear_log)
        self.clear_btn.grid(row=0, column=1, sticky="e")
        self.log_box = ctk.CTkTextbox(logc, corner_radius=10, fg_color=TERM_BG,
                                      text_color=TERM_TXT,
                                      font=ctk.CTkFont(FONT_MONO, 13), wrap="word")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=14)
        self.log_box.configure(state="disabled")

    # ===================================================================== TEMA / DİL / MOD
    def _set_accent(self, name):
        self._apply_accent(*ACCENTS[name])

    def _apply_accent(self, accent, hover):
        self.accent, self.accent_hover = accent, hover
        self.logo_dot.configure(text_color=accent)
        self.brand_rule.configure(fg_color=accent)
        self.drop_icon.configure(text_color=accent)
        self.browse_btn.configure(fg_color=accent, hover_color=hover)
        self.model_menu.configure(button_color=accent, button_hover_color=hover)
        self.progress.configure(progress_color=accent)
        self.percent_lbl.configure(text_color=accent)
        self.char_val.configure(text_color=accent)
        self.min_val.configure(text_color=accent)
        self.char_slider.configure(button_color=accent, button_hover_color=hover,
                                   progress_color=accent)
        self.min_slider.configure(button_color=accent, button_hover_color=hover,
                                  progress_color=accent)
        self.sentence_switch.configure(progress_color=accent)
        self.appear_seg.configure(selected_color=accent, selected_hover_color=hover)
        for name, (col, _) in ACCENTS.items():
            sel = (col == accent)
            self.swatches[name].configure(border_width=2 if sel else 0, border_color=TEXT)
        self._refresh_lang_buttons()
        self._update_primary()

    def _set_mode(self, val):
        ctk.set_appearance_mode("dark" if val == "☾" else "light")

    def _set_lang(self, code):
        self.lang = code
        self._retext()

    def _refresh_lang_buttons(self):
        for code, b in self.lang_btns.items():
            if code == self.lang:
                b.configure(fg_color=self.accent, hover_color=self.accent_hover,
                            text_color="#ffffff", border_width=0)
            else:
                b.configure(fg_color="transparent", hover_color=HOVER,
                            text_color=MUTED, border_width=1, border_color=BORDER)

    def _retext(self):
        self.cap_model.configure(text=self.t("model"))
        self.cap_lang.configure(text=self.t("language"))
        self.cap_sub.configure(text=self.t("sub_group"))
        self.char_cap.configure(text=self.t("char_limit"))
        self.char_val.configure(text=str(int(self.char_slider.get())))
        self.min_cap.configure(text=self.t("min_dur"))
        self.min_val.configure(text=f"{self.min_slider.get():.1f} {self.t('sec')}")
        self.sentence_switch.configure(text=self.t("sentence_only"))
        self.cap_appear.configure(text=self.t("appearance"))
        self.cap_theme.configure(text=self.t("theme"))
        self.subtitle_lbl.configure(text=self.t("subtitle"))
        self.drop_title.configure(text=self.t("drop_title"))
        self.or_lbl.configure(text=f"—  {self.t('or')}  —")
        self.browse_btn.configure(text=self.t("browse"))
        self.open_btn.configure(text=self.t("open_folder"))
        self.log_cap.configure(text=self.t("log"))
        self.clear_btn.configure(text=self.t("clear"))
        self._refresh_lang_buttons()
        self._render_status()
        if self._last_stats:
            self._set_stats(self._last_stats)
        self._update_primary()

    # ===================================================================== DOSYA / DnD
    def browse_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title=self.t("browse"),
            filetypes=[("Video & Audio", " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)),
                       ("All files", "*.*")])
        if path:
            self._set_video(path)

    def on_drop(self, event):
        self._on_drag_leave(event)
        paths = self.tk.splitlist(event.data)
        if paths:
            self._set_video(paths[0])

    def _on_drag_enter(self, event):
        self.drop_frame.configure(border_color=self.accent, fg_color=SURFACE2)
        self.badge.configure(fg_color=HOVER)
        return event.action

    def _on_drag_leave(self, event):
        self.drop_frame.configure(border_color=BORDER_SOFT, fg_color=SURFACE)
        self.badge.configure(fg_color=SURFACE2)
        return event.action

    def _set_video(self, path):
        path = os.path.normpath(path.strip())
        if not os.path.isfile(path):
            self._log(self.t("not_found", path))
            return
        if not path.lower().endswith(VIDEO_EXTENSIONS):
            self._log(self.t("unsupported", path))
        self.video_path = path
        self.drop_icon.configure(text="✓")
        self.chip_name.configure(text=os.path.basename(path))
        self.chip_path.configure(text=os.path.dirname(path))
        self.chip.grid()
        self._log(self.t("selected", path))
        self.open_btn.configure(state="disabled")
        self.last_srt = None
        self._update_primary()

    def _clear_video(self):
        if self.is_running:
            return
        self.video_path = None
        self.chip.grid_remove()
        self.drop_icon.configure(text="↓")
        self._update_primary()

    # ===================================================================== BAŞLAT / DURDUR
    def _on_primary(self):
        if self.is_running:
            self.cancel_event.set()
            self.primary_btn.configure(text=self.t("stopping"), state="disabled")
            return
        if not self.video_path:
            self._log(self.t("err_no_file"))
            return

        srt_path = os.path.splitext(self.video_path)[0] + ".srt"
        max_chars = int(self.char_slider.get())
        min_dur = round(self.min_slider.get(), 1)
        sentence_only = bool(self.sentence_switch.get())

        self.cancel_event.clear()
        self._set_busy(True)
        self.progress.set(0)
        self.percent_lbl.configure(text="0%")
        self._log("─" * 56)
        self._log(self.t("starting"))
        self._log(self.t("settings_line", max_chars, f"{min_dur:.1f}",
                         self.t("on") if sentence_only else self.t("off")))
        threading.Thread(target=self._transcription_worker,
                         args=(self.video_path, srt_path, self.model_var.get(),
                               max_chars, min_dur, sentence_only),
                         daemon=True).start()

    def open_output_folder(self):
        if self.last_srt and os.path.exists(self.last_srt):
            try:
                os.startfile(os.path.dirname(self.last_srt))
            except Exception as e:
                self._log(self.t("open_fail", e))

    def _detect_device(self):
        try:
            import ctranslate2
            ok = ctranslate2.get_cuda_device_count() >= 1
        except Exception:
            ok = False
        self.q(("device", ok))

    def _notify(self, srt_path):
        """İşlem bitince Windows bildirimi; tıklayınca çıktı klasörünü açar."""
        def run():
            try:
                from winotify import Notification
                folder = os.path.dirname(srt_path)
                uri = "file:///" + folder.replace("\\", "/")
                toast = Notification(app_id="AutoSRT",
                                     title=self.t("notify_title"),
                                     msg=self.t("notify_msg", os.path.basename(srt_path)),
                                     duration="short", launch=uri)
                toast.add_actions(label=self.t("open_folder"), launch=uri)
                toast.show()
            except Exception as e:
                self.log_queue.put(("raw", f"\n[notify] {e}\n"))
        threading.Thread(target=run, daemon=True).start()

    # ============================================= ARKA PLAN TRANSCRIBE (THREAD)
    def _transcription_worker(self, video_path, srt_path, model_size,
                              max_chars, min_dur, sentence_only):
        try:
            from faster_whisper import WhisperModel

            device, compute = DEVICE, COMPUTE_TYPE
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() < 1:
                    device, compute = "cpu", "int8"
                    self._log(self.t("cpu_fb"))
            except Exception:
                pass

            self.q(("status", "st_loading", WARN))
            self.q(("progress", None))
            key = (model_size, device, compute)
            if self.model is None or self._model_key != key:
                # VRAM tahliye protokolu: yeni modeli yuklemeden ONCE eskisini
                # bellekten bosalt. Aksi halde dusuk VRAM'li GPU'larda (6 GB)
                # eski + yeni model bir an icin ayni anda bellekte olur -> OOM.
                if self.model is not None:
                    del self.model
                    self.model = None
                    self._model_key = None
                    gc.collect()
                self._log(self.t("loading_model", model_size, device, compute))
                self._log(self.t("first_use"))
                self.model = WhisperModel(model_size, device=device, compute_type=compute)
                self._model_key = key
                self._log(self.t("model_ready"))
            if self.cancel_event.is_set():
                raise _Cancelled()

            self.q(("status", "st_analyzing", self.accent))
            segments, info = self.model.transcribe(
                video_path,
                beam_size=10,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=100,
                ),
                word_timestamps=True,
                condition_on_previous_text=False,
            )

            duration = info.duration or 0
            lang = f"{info.language} ({info.language_probability:.0%})"
            self._log(self.t("detected", lang, format_timestamp(duration)))
            self._log("─" * 56)
            self.q(("status", "st_generating", self.accent))
            self.q(("progress", 0.0))

            t0 = time.time()
            # Hangi noktalama bölme tetikler? (sentence_only -> sadece cümle sonu)
            punct = ('.', '?', '!') if sentence_only else ('.', ',', '?', '!', ':', ';')

            blocks = []  # [start, end, text]

            def _tick(end_t):
                if duration:
                    frac = min(max(end_t / duration, 0.0), 1.0)
                    el = time.time() - t0
                    eta = (el / frac - el) if frac > 0.01 else 0
                    self.q(("progress", frac))
                    self.q(("stats", {"el": el, "eta": eta,
                                      "blocks": len(blocks), "lang": lang}))

            # --- MİKRO-DİLİMLEME DÖNGÜSÜ (yapı BİREBİR; sınır/punkt parametreli) ---
            for segment in segments:
                if self.cancel_event.is_set():
                    raise _Cancelled()
                current_chunk_words = []
                chunk_start = None

                for word in segment.words:
                    if chunk_start is None:
                        chunk_start = word.start

                    current_chunk_words.append(word)
                    text_so_far = "".join([w.word for w in current_chunk_words]).strip()
                    ends_with_punct = word.word.strip().endswith(punct)

                    if len(text_so_far) >= max_chars or ends_with_punct:
                        chunk_end = word.end
                        blocks.append([chunk_start, chunk_end, text_so_far])
                        self._log(f"[{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}] {text_so_far}")
                        current_chunk_words = []
                        chunk_start = None

                if current_chunk_words:
                    text_so_far = "".join([w.word for w in current_chunk_words]).strip()
                    chunk_end = current_chunk_words[-1].end
                    blocks.append([chunk_start, chunk_end, text_so_far])
                    self._log(f"[{format_timestamp(chunk_start)} - {format_timestamp(chunk_end)}] {text_so_far}")

                _tick(segment.end)
            # --- DÖNGÜ SONU ---

            # Okunabilirlik: çok kısa blokları en az min_dur kadar ekranda tut
            # (sonraki bloğun başlangıcına taşmadan; yalnızca uzatır, kısaltmaz).
            if min_dur > 0:
                for i, b in enumerate(blocks):
                    if b[1] - b[0] < min_dur:
                        new_end = b[0] + min_dur
                        if i + 1 < len(blocks):
                            new_end = min(new_end, blocks[i + 1][0] - 0.05)
                        if new_end > b[1]:
                            b[1] = new_end

            with open(srt_path, "w", encoding="utf-8") as srt_file:
                for idx, (s, e, txt) in enumerate(blocks, 1):
                    srt_file.write(f"{idx}\n")
                    srt_file.write(f"{format_timestamp(s)} --> {format_timestamp(e)}\n")
                    srt_file.write(f"{txt}\n\n")

            self.q(("progress", 1.0))
            self._log("─" * 56)
            self._log("✓ " + self.t("done_blocks", len(blocks)))
            self._log(self.t("saved", srt_path))
            self.q(("done", srt_path))

        except _Cancelled:
            self._log("■ " + self.t("cancelled"))
            self.q(("cancelled", None))
        except Exception as e:
            self._log(f"✕ {type(e).__name__}: {e}")
            self.q(("error", str(e)))

    # ===================================================================== KUYRUK
    def _log(self, message):
        self.log_queue.put(("log", message))

    def _log_key(self, key, *a):
        self._log(self.t(key, *a))

    def q(self, item):
        self.log_queue.put(item)

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "raw":
                    self._write_raw(item[1])
                elif kind == "status":
                    self._set_status(item[1], item[2])
                elif kind == "progress":
                    self._set_progress(item[1])
                elif kind == "stats":
                    self._set_stats(item[1])
                elif kind == "device":
                    self._set_device(item[1])
                elif kind == "done":
                    self.last_srt = item[1]
                    self.open_btn.configure(state="normal")
                    self._set_status("st_done", GREEN)
                    self.percent_lbl.configure(text="100%")
                    self._set_busy(False)
                    self._notify(item[1])
                elif kind == "cancelled":
                    self._set_status("st_stopped", WARN)
                    self._set_busy(False)
                elif kind == "error":
                    self._set_status("st_error", RED)
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    # ----------------------------------------------------- durum / ilerleme
    def _set_status(self, key, color):
        self._status_key, self._status_color = key, color
        self._render_status()

    def _render_status(self):
        self.status_dot.configure(text_color=self._status_color)
        self.status_lbl.configure(text=self.t(self._status_key))

    def _set_device(self, ok):
        self.dev_dot.configure(text_color=GREEN if ok else WARN)
        self.dev_lbl.configure(text=self.t("dev_gpu" if ok else "dev_cpu"))

    def _set_progress(self, frac):
        if frac is None:
            if self._pb_mode != "ind":
                self.progress.configure(mode="indeterminate")
                self.progress.start()
                self._pb_mode = "ind"
            self.percent_lbl.configure(text="•••")
        else:
            if self._pb_mode != "det":
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self._pb_mode = "det"
            self.progress.set(frac)
            self.percent_lbl.configure(text=f"{int(frac * 100)}%")

    def _set_stats(self, d):
        self._last_stats = d
        self.stats_lbl.configure(text=self.t("stats", _clock(d["el"]), _clock(d["eta"]),
                                             d["blocks"], d["lang"]))

    # ----------------------------------------------------- günlük yazımı
    def _append_log(self, message):
        box = self.log_box
        box.configure(state="normal")
        if box.get("end-2c", "end-1c") not in ("\n", ""):
            box.insert("end", "\n")
        box.insert("end", message + "\n")
        box.see("end")
        box.configure(state="disabled")

    def _write_raw(self, s):
        box = self.log_box
        box.configure(state="normal")
        s = s.replace("\r\n", "\n")
        for part in re.split(r"([\r\n])", s):
            if part == "\n":
                box.insert("end", "\n")
            elif part == "\r":
                box.delete(box.index("end-1c linestart"), "end-1c")
            elif part:
                box.insert("end", part)
        box.see("end")
        box.configure(state="disabled")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ----------------------------------------------------- durum kilidi
    def _set_busy(self, busy):
        self.is_running = busy
        if busy:
            self.browse_btn.configure(state="disabled")
            self.model_menu.configure(state="disabled")
            self.primary_btn.configure(text=self.t("stop"), state="normal",
                                       fg_color=RED, hover_color="#e11d48",
                                       text_color="#ffffff", border_width=0)
        else:
            self.cancel_event.clear()
            self.browse_btn.configure(state="normal")
            self.model_menu.configure(state="normal")
            self.progress.stop()
            self._pb_mode = "det"
            self._update_primary()

    def _update_primary(self):
        if getattr(self, "is_running", False):
            return
        if self.video_path:
            self.primary_btn.configure(text=self.t("start"), state="normal",
                                       fg_color=self.accent, hover_color=self.accent_hover,
                                       text_color="#ffffff", border_width=0)
        else:
            self.primary_btn.configure(text=self.t("need_file"), state="disabled",
                                       fg_color=SURFACE2, hover_color=HOVER,
                                       text_color_disabled=MUTED,
                                       border_width=1, border_color=BORDER)


if __name__ == "__main__":
    app = AutoSRTApp()
    app.mainloop()
