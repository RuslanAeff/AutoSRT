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
import json
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


def format_timestamp_vtt(seconds: float):
    # WebVTT milisaniye ayıracı olarak nokta kullanır
    return format_timestamp(seconds).replace(",", ".")


def _clock(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --- ÇIKTI BİÇİMİ YAZICILARI ---
def write_srt(blocks, path):
    with open(path, "w", encoding="utf-8") as f:
        for idx, (s, e, txt) in enumerate(blocks, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(s)} --> {format_timestamp(e)}\n")
            f.write(f"{txt}\n\n")


def write_vtt(blocks, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, (s, e, txt) in enumerate(blocks, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp_vtt(s)} --> {format_timestamp_vtt(e)}\n")
            f.write(f"{txt}\n\n")


def write_txt(blocks, path):
    with open(path, "w", encoding="utf-8") as f:
        for (s, e, txt) in blocks:
            f.write(txt.replace("\n", " ").strip() + "\n")


def write_json(blocks, path):
    data = [{"index": i, "start": round(s, 3), "end": round(e, 3), "text": txt}
            for i, (s, e, txt) in enumerate(blocks, 1)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# biçim anahtarı -> (uzantı, yazıcı)
WRITERS = {
    "srt":  ("srt",  write_srt),
    "vtt":  ("vtt",  write_vtt),
    "txt":  ("txt",  write_txt),
    "json": ("json", write_json),
}

_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def _ts_to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def blocks_to_srt_text(blocks):
    out = []
    for idx, (s, e, txt) in enumerate(blocks, 1):
        out.append(f"{idx}\n{format_timestamp(s)} --> {format_timestamp(e)}\n{txt}\n")
    return "\n".join(out)


def parse_srt(text):
    """Düzenleyicideki SRT metnini bloklara geri çevirir."""
    blocks = []
    for entry in re.split(r"\n\s*\n", text.strip()):
        lines = entry.strip().splitlines()
        if not lines:
            continue
        ts_idx, m = None, None
        for i, ln in enumerate(lines):
            m = _TS_RE.search(ln)
            if m:
                ts_idx = i
                break
        if ts_idx is None:
            continue
        start = _ts_to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ts_to_sec(m.group(5), m.group(6), m.group(7), m.group(8))
        txt = "\n".join(lines[ts_idx + 1:]).strip()
        if txt:
            blocks.append([start, end, txt])
    return blocks


# --- MODEL KONFIGURASYONU (BIREBIR KORUNDU) ---
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MODEL_SIZES = ["large-v3", "large-v2", "medium", "small", "base", "tiny"]

# Kaynak dil seçenekleri (yerel ad, whisper kodu); "Otomatik" ayrıca eklenir
NATIVE_LANGS = [
    ("Türkçe", "tr"), ("English", "en"), ("Deutsch", "de"), ("Español", "es"),
    ("Français", "fr"), ("Italiano", "it"), ("Русский", "ru"), ("Português", "pt"),
    ("العربية", "ar"), ("中文", "zh"), ("日本語", "ja"), ("한국어", "ko"),
    ("Nederlands", "nl"), ("Polski", "pl"), ("Українська", "uk"),
]

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
        "model": "MODEL", "language": "ARAYÜZ DİLİ", "theme": "TEMA", "appearance": "GÖRÜNÜM",
        "src_lang": "KAYNAK DİL", "auto": "Otomatik (algıla)",
        "out_format": "ÇIKTI BİÇİMİ", "out_folder": "ÇIKTI KLASÖRÜ",
        "out_same": "Kaynak dosyanın yanı", "choose_folder": "Klasör Seç",
        "term_hint": "TERİM İPUCU", "term_ph": "İsimler, terimler (virgülle)…",
        "sub_group": "ALTYAZI AYARLARI",
        "char_limit": "Karakter sınırı", "min_dur": "Min. süre", "sec": "sn",
        "sentence_only": "Cümle sonunda böl",
        "edit_before": "Kaydetmeden önce düzenle",
        "drop_title": "Videoları buraya sürükleyip bırakın",
        "or": "veya", "browse": "Dosya Seç", "add_files": "Dosya Ekle",
        "queue": "SIRA", "files_n": "{} dosya", "clear_all": "Tümünü temizle",
        "start": "Altyazı Oluştur", "start_n": "Altyazı Oluştur ({})",
        "need_file": "Önce dosya ekleyin",
        "stop": "Durdur", "stopping": "Durduruluyor…",
        "open_folder": "Klasörü Aç", "log": "Günlük", "clear": "Temizle",
        "st_ready": "Hazır", "st_loading": "Model yükleniyor",
        "st_analyzing": "Ses analiz ediliyor", "st_generating": "Altyazılar oluşturuluyor",
        "st_editing": "Düzenleme bekleniyor", "st_done": "Tamamlandı",
        "st_stopped": "Durduruldu", "st_error": "Hata",
        "ready_msg": "AutoSRT hazır. Dosya ekleyin veya sürükleyin.",
        "starting": "İşlem başlatılıyor… (Arayüz kilitlenmez, arka planda çalışır)",
        "loading_model": "Model yükleniyor: {} ({}/{})",
        "first_use": "(İlk kullanımda model indirilebilir, lütfen bekleyin…)",
        "model_ready": "Model hazır.",
        "settings_line": "Ayarlar → karakter: {}, min süre: {} sn, cümle sonu: {}, biçim: {}",
        "on": "açık", "off": "kapalı",
        "detected": "Algılanan dil: {} | Süre: {}",
        "file_progress": "Dosya {}/{}",
        "done_blocks": "{} altyazı bloğu oluşturuldu.",
        "batch_done": "TAMAMLANDI! {} dosya işlendi.",
        "saved": "Kaydedildi: {}",
        "cancelled": "İşlem kullanıcı tarafından durduruldu.",
        "err_no_file": "HATA: Önce bir dosya ekleyin veya sürükleyin.",
        "no_format": "HATA: En az bir çıktı biçimi seçin.",
        "cpu_fb": "UYARI: CUDA cihazı bulunamadı → CPU/int8 moduna geçildi.",
        "stats": "Geçen {} • Kalan ~{} • {} blok • Dil: {}",
        "unsupported": "UYARI: Desteklenmeyen uzantı olabilir → {}",
        "not_found": "HATA: Dosya bulunamadı → {}",
        "open_fail": "Klasör açılamadı: {}",
        "dev_gpu": "GPU hazır", "dev_cpu": "CPU modu", "dev_check": "Donanım denetleniyor…",
        "added_n": "{} dosya sıraya eklendi.",
        "notify_title": "Altyazı hazır ✓", "notify_msg": "{} oluşturuldu — açmak için tıklayın",
        "editor_title": "Altyazıyı düzenle — {}",
        "editor_hint": "Metni ve zamanlamayı düzenleyebilirsiniz (SRT biçimi). Kaydedince seçili biçimlere yazılır.",
        "save": "Kaydet", "cancel": "İptal",
    },
    "en": {
        "subtitle": "AI-Powered Subtitle Engine",
        "model": "MODEL", "language": "INTERFACE", "theme": "THEME", "appearance": "APPEARANCE",
        "src_lang": "SOURCE LANGUAGE", "auto": "Auto (detect)",
        "out_format": "OUTPUT FORMAT", "out_folder": "OUTPUT FOLDER",
        "out_same": "Next to source file", "choose_folder": "Choose Folder",
        "term_hint": "VOCABULARY HINT", "term_ph": "Names, terms (comma-separated)…",
        "sub_group": "SUBTITLE SETTINGS",
        "char_limit": "Character limit", "min_dur": "Min. duration", "sec": "s",
        "sentence_only": "Split at sentence end",
        "edit_before": "Edit before saving",
        "drop_title": "Drag & drop your videos here",
        "or": "or", "browse": "Choose Files", "add_files": "Add Files",
        "queue": "QUEUE", "files_n": "{} files", "clear_all": "Clear all",
        "start": "Generate Subtitles", "start_n": "Generate Subtitles ({})",
        "need_file": "Add a file first",
        "stop": "Stop", "stopping": "Stopping…",
        "open_folder": "Open Folder", "log": "Log", "clear": "Clear",
        "st_ready": "Ready", "st_loading": "Loading model",
        "st_analyzing": "Analyzing audio", "st_generating": "Generating subtitles",
        "st_editing": "Waiting for edit", "st_done": "Completed",
        "st_stopped": "Stopped", "st_error": "Error",
        "ready_msg": "AutoSRT is ready. Add or drop files.",
        "starting": "Starting… (UI stays responsive, runs in background)",
        "loading_model": "Loading model: {} ({}/{})",
        "first_use": "(Model may download on first use, please wait…)",
        "model_ready": "Model ready.",
        "settings_line": "Settings → chars: {}, min dur: {} s, sentence-end: {}, format: {}",
        "on": "on", "off": "off",
        "detected": "Detected language: {} | Duration: {}",
        "file_progress": "File {}/{}",
        "done_blocks": "{} subtitle blocks created.",
        "batch_done": "DONE! {} files processed.",
        "saved": "Saved: {}",
        "cancelled": "Stopped by the user.",
        "err_no_file": "ERROR: Add or drop a file first.",
        "no_format": "ERROR: Select at least one output format.",
        "cpu_fb": "WARNING: No CUDA device found → switched to CPU/int8.",
        "stats": "Elapsed {} • ETA ~{} • {} blocks • Lang: {}",
        "unsupported": "WARNING: Possibly unsupported extension → {}",
        "not_found": "ERROR: File not found → {}",
        "open_fail": "Could not open folder: {}",
        "dev_gpu": "GPU ready", "dev_cpu": "CPU mode", "dev_check": "Checking hardware…",
        "added_n": "{} file(s) added to the queue.",
        "notify_title": "Subtitles ready ✓", "notify_msg": "{} created — click to open",
        "editor_title": "Edit subtitles — {}",
        "editor_hint": "Edit text and timing (SRT format). On save, written to selected formats.",
        "save": "Save", "cancel": "Cancel",
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

        self.files = []                 # sıradaki dosya yolları
        self.is_running = False
        self.model = None
        self._model_key = None
        self.log_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.last_srt = None
        self.out_dir = None             # None -> kaynağın yanı
        self._cur_src_code = None       # None -> otomatik algıla
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
        self.geometry("1100x820")
        self.minsize(980, 640)
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
            self._ico_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "icon.ico")
            if os.path.exists(self._ico_path):
                try:
                    import ctypes
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AutoSRT")
                except Exception:
                    pass
                self.iconbitmap(self._ico_path)
                # CTk bazen ikonu geciktirerek ezer; bir kez daha uygula.
                self.after(300, lambda: self.iconbitmap(self._ico_path))
        except Exception:
            self._ico_path = None

    def _set_toplevel_icon(self, win):
        try:
            if getattr(self, "_ico_path", None) and os.path.exists(self._ico_path):
                win.iconbitmap(self._ico_path)
        except Exception:
            pass

    def t(self, key, *args):
        s = L[self.lang].get(key, key)
        return s.format(*args) if args else s

    # ===================================================================== KENAR ÇUBUĞU
    def _build_sidebar(self):
        # Kaydırılabilir kenar çubuğu: her ekran yüksekliği/DPI'da tüm kontroller erişilir
        sb = ctk.CTkScrollableFrame(self, width=262, corner_radius=0, fg_color=SIDEBAR,
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

        # KAYNAK DİL
        self.cap_src = self._caption(sb, r, "src_lang"); r += 1
        self.src_var = ctk.StringVar(value=self.t("auto"))
        self.src_menu = ctk.CTkOptionMenu(
            sb, values=[self.t("auto")], variable=self.src_var,
            command=self._on_src_select,
            height=36, corner_radius=10, fg_color=SURFACE2, button_color=self.accent,
            button_hover_color=self.accent_hover, dropdown_fg_color=SURFACE2,
            text_color=TEXT, dropdown_text_color=TEXT,
            font=ctk.CTkFont(FONT_UI, 13), dropdown_font=ctk.CTkFont(FONT_UI, 13))
        self.src_menu.grid(row=r, column=0, sticky="ew", padx=20, pady=(2, 14)); r += 1
        self._rebuild_src_menu()

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
        self.sentence_switch.grid(row=r, column=0, sticky="w", padx=20, pady=(4, 8)); r += 1
        self.preview_switch = ctk.CTkSwitch(
            sb, text=self.t("edit_before"), font=ctk.CTkFont(FONT_UI, 12),
            text_color=TEXT)
        self.preview_switch.grid(row=r, column=0, sticky="w", padx=20, pady=(4, 14)); r += 1

        # ÇIKTI BİÇİMİ
        self.cap_fmt = self._caption(sb, r, "out_format"); r += 1
        fmtf = ctk.CTkFrame(sb, fg_color="transparent")
        fmtf.grid(row=r, column=0, sticky="ew", padx=20, pady=(2, 14)); r += 1
        fmtf.grid_columnconfigure((0, 1), weight=1)
        self.fmt_vars = {}
        self.fmt_boxes = {}
        for i, fmt in enumerate(("srt", "vtt", "txt", "json")):
            var = ctk.BooleanVar(value=(fmt == "srt"))
            cb = ctk.CTkCheckBox(fmtf, text=fmt.upper(), variable=var,
                                 font=ctk.CTkFont(FONT_UI, 12), text_color=TEXT,
                                 checkbox_width=20, checkbox_height=20, corner_radius=5)
            cb.grid(row=i // 2, column=i % 2, sticky="w", pady=4)
            self.fmt_vars[fmt] = var
            self.fmt_boxes[fmt] = cb

        # ÇIKTI KLASÖRÜ
        self.cap_out = self._caption(sb, r, "out_folder"); r += 1
        outf = ctk.CTkFrame(sb, fg_color="transparent")
        outf.grid(row=r, column=0, sticky="ew", padx=20, pady=(2, 4)); r += 1
        outf.grid_columnconfigure(0, weight=1)
        self.out_btn = ctk.CTkButton(outf, text=self.t("choose_folder"), height=32,
                                     corner_radius=10, fg_color=SURFACE2, hover_color=HOVER,
                                     text_color=TEXT, border_width=1, border_color=BORDER,
                                     font=ctk.CTkFont(FONT_UI, 12), command=self._choose_out_dir)
        self.out_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.out_clear = ctk.CTkButton(outf, text="✕", width=32, height=32, corner_radius=8,
                                       fg_color="transparent", hover_color=HOVER,
                                       text_color=MUTED, font=ctk.CTkFont(FONT_UI, 13),
                                       command=self._clear_out_dir)
        self.out_clear.grid(row=0, column=1)
        self.out_lbl = ctk.CTkLabel(sb, text=self.t("out_same"), text_color=MUTED,
                                    font=ctk.CTkFont(FONT_UI, 11), anchor="w",
                                    wraplength=210, justify="left")
        self.out_lbl.grid(row=r, column=0, sticky="ew", padx=20, pady=(0, 14)); r += 1

        # TERİM İPUCU
        self.cap_term = self._caption(sb, r, "term_hint"); r += 1
        self.term_entry = ctk.CTkEntry(sb, placeholder_text=self.t("term_ph"),
                                       height=36, corner_radius=10, fg_color=SURFACE2,
                                       border_color=BORDER, text_color=TEXT,
                                       font=ctk.CTkFont(FONT_UI, 12))
        self.term_entry.grid(row=r, column=0, sticky="ew", padx=20, pady=(2, 14)); r += 1

        # ARAYÜZ DİLİ
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
        ctk.CTkLabel(dev, text="v1.4", text_color=FAINT,
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

    def _rebuild_src_menu(self):
        """Kaynak dil menüsünü kurar; 'Otomatik' etiketi arayüz diline göre güncellenir."""
        auto = self.t("auto")
        values = [auto] + [n for n, _ in NATIVE_LANGS]
        self._src_codes = {auto: None}
        for n, c in NATIVE_LANGS:
            self._src_codes[n] = c
        self.src_menu.configure(values=values)
        disp = auto
        for d, c in self._src_codes.items():
            if c == self._cur_src_code:
                disp = d
                break
        self.src_var.set(disp)

    def _on_src_select(self, value):
        self._cur_src_code = self._src_codes.get(value)

    # ===================================================================== ANA ALAN
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=36, pady=32)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(6, weight=1)

        head = ctk.CTkFrame(main, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        ctk.CTkLabel(head, text="AutoSRT",
                     font=ctk.CTkFont(FONT_UI, 30, weight="bold"),
                     text_color=TEXT).pack(anchor="w")
        self.subtitle_lbl = ctk.CTkLabel(head, text=self.t("subtitle"),
                                         font=ctk.CTkFont(FONT_UI, 14), text_color=MUTED)
        self.subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Sürükle-bırak kartı
        self.drop_frame = ctk.CTkFrame(main, height=176, corner_radius=18,
                                       border_width=2, border_color=BORDER_SOFT,
                                       fg_color=SURFACE)
        self.drop_frame.grid(row=1, column=0, sticky="ew")
        self.drop_frame.grid_propagate(False)
        self.drop_frame.grid_columnconfigure(0, weight=1)
        self.drop_frame.grid_rowconfigure((0, 4), weight=1)

        self.badge = ctk.CTkFrame(self.drop_frame, width=56, height=56, corner_radius=28,
                                  fg_color=SURFACE2)
        self.badge.grid(row=1, column=0)
        self.badge.grid_propagate(False)
        self.drop_icon = ctk.CTkLabel(self.badge, text="↓",
                                      font=ctk.CTkFont(FONT_UI, 26, weight="bold"))
        self.drop_icon.place(relx=0.5, rely=0.5, anchor="center")

        self.drop_title = ctk.CTkLabel(self.drop_frame, text=self.t("drop_title"),
                                       font=ctk.CTkFont(FONT_UI, 17, weight="bold"),
                                       text_color=TEXT)
        self.drop_title.grid(row=2, column=0, pady=(10, 2))
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

        # Dosya sırası (kuyruk)
        self.files_card = ctk.CTkFrame(main, fg_color=SURFACE2, corner_radius=12,
                                       border_width=1, border_color=BORDER)
        self.files_card.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.files_card.grid_columnconfigure(0, weight=1)
        fhead = ctk.CTkFrame(self.files_card, fg_color="transparent")
        fhead.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        fhead.grid_columnconfigure(0, weight=1)
        self.files_count = ctk.CTkLabel(fhead, text="", anchor="w",
                                        font=ctk.CTkFont(FONT_UI, 11, weight="bold"),
                                        text_color=MUTED)
        self.files_count.grid(row=0, column=0, sticky="w")
        self.clear_all_btn = ctk.CTkButton(fhead, text=self.t("clear_all"), width=110,
                                           height=26, corner_radius=8, fg_color="transparent",
                                           hover_color=HOVER, text_color=MUTED,
                                           border_width=1, border_color=BORDER,
                                           font=ctk.CTkFont(FONT_UI, 11),
                                           command=self._clear_files)
        self.clear_all_btn.grid(row=0, column=1, sticky="e")
        self.files_list = ctk.CTkScrollableFrame(self.files_card, height=104,
                                                 fg_color="transparent",
                                                 scrollbar_button_color=BORDER,
                                                 scrollbar_button_hover_color=MUTED)
        self.files_list.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 10))
        self.files_list.grid_columnconfigure(0, weight=1)
        self.files_card.grid_remove()

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
        self.src_menu.configure(button_color=accent, button_hover_color=hover)
        self.progress.configure(progress_color=accent)
        self.percent_lbl.configure(text_color=accent)
        self.char_val.configure(text_color=accent)
        self.min_val.configure(text_color=accent)
        self.char_slider.configure(button_color=accent, button_hover_color=hover,
                                   progress_color=accent)
        self.min_slider.configure(button_color=accent, button_hover_color=hover,
                                  progress_color=accent)
        self.sentence_switch.configure(progress_color=accent)
        self.preview_switch.configure(progress_color=accent)
        for cb in self.fmt_boxes.values():
            cb.configure(fg_color=accent, hover_color=hover)
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
        self.cap_src.configure(text=self.t("src_lang"))
        self._rebuild_src_menu()
        self.cap_lang.configure(text=self.t("language"))
        self.cap_sub.configure(text=self.t("sub_group"))
        self.char_cap.configure(text=self.t("char_limit"))
        self.char_val.configure(text=str(int(self.char_slider.get())))
        self.min_cap.configure(text=self.t("min_dur"))
        self.min_val.configure(text=f"{self.min_slider.get():.1f} {self.t('sec')}")
        self.sentence_switch.configure(text=self.t("sentence_only"))
        self.preview_switch.configure(text=self.t("edit_before"))
        self.cap_fmt.configure(text=self.t("out_format"))
        self.cap_out.configure(text=self.t("out_folder"))
        self.out_btn.configure(text=self.t("choose_folder"))
        self._render_out_label()
        self.cap_term.configure(text=self.t("term_hint"))
        self.term_entry.configure(placeholder_text=self.t("term_ph"))
        self.cap_appear.configure(text=self.t("appearance"))
        self.cap_theme.configure(text=self.t("theme"))
        self.subtitle_lbl.configure(text=self.t("subtitle"))
        self.drop_title.configure(text=self.t("drop_title"))
        self.or_lbl.configure(text=f"—  {self.t('or')}  —")
        self.browse_btn.configure(text=self.t("browse"))
        self.open_btn.configure(text=self.t("open_folder"))
        self.clear_all_btn.configure(text=self.t("clear_all"))
        self.log_cap.configure(text=self.t("log"))
        self.clear_btn.configure(text=self.t("clear"))
        self._render_files()
        self._refresh_lang_buttons()
        self._render_status()
        if self._last_stats:
            self._set_stats(self._last_stats)
        self._update_primary()

    # ===================================================================== DOSYA / DnD
    def browse_file(self):
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title=self.t("browse"),
            filetypes=[("Video & Audio", " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)),
                       ("All files", "*.*")])
        if paths:
            self._add_files(paths)

    def on_drop(self, event):
        self._on_drag_leave(event)
        if self.is_running:
            return
        paths = self.tk.splitlist(event.data)
        if paths:
            self._add_files(paths)

    def _on_drag_enter(self, event):
        self.drop_frame.configure(border_color=self.accent, fg_color=SURFACE2)
        self.badge.configure(fg_color=HOVER)
        return event.action

    def _on_drag_leave(self, event):
        self.drop_frame.configure(border_color=BORDER_SOFT, fg_color=SURFACE)
        self.badge.configure(fg_color=SURFACE2)
        return event.action

    def _choose_out_dir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title=self.t("out_folder"))
        if d:
            self.out_dir = os.path.normpath(d)
            self._render_out_label()

    def _clear_out_dir(self):
        self.out_dir = None
        self._render_out_label()

    def _render_out_label(self):
        if self.out_dir:
            p = self.out_dir
            if len(p) > 34:
                p = "…" + p[-33:]
            self.out_lbl.configure(text=p, text_color=TEXT)
        else:
            self.out_lbl.configure(text=self.t("out_same"), text_color=MUTED)

    def _add_files(self, paths):
        if self.is_running:
            return
        added = 0
        for raw in paths:
            path = os.path.normpath(str(raw).strip())
            if not os.path.isfile(path):
                self._log(self.t("not_found", path))
                continue
            if path in self.files:
                continue
            if not path.lower().endswith(VIDEO_EXTENSIONS):
                self._log(self.t("unsupported", path))
            self.files.append(path)
            added += 1
        if added:
            self.drop_icon.configure(text="✓")
            self.last_srt = None
            self.open_btn.configure(state="disabled")
            self._log(self.t("added_n", added))
        self._render_files()
        self._update_primary()

    def _remove_file(self, path):
        if self.is_running:
            return
        if path in self.files:
            self.files.remove(path)
        if not self.files:
            self.drop_icon.configure(text="↓")
        self._render_files()
        self._update_primary()

    def _clear_files(self):
        if self.is_running:
            return
        self.files = []
        self.drop_icon.configure(text="↓")
        self._render_files()
        self._update_primary()

    def _render_files(self):
        for w in self.files_list.winfo_children():
            w.destroy()
        if not self.files:
            self.files_card.grid_remove()
            return
        self.files_card.grid()
        self.files_count.configure(text=self.t("files_n", len(self.files)))
        for i, path in enumerate(self.files):
            row = ctk.CTkFrame(self.files_list, fg_color=SURFACE, corner_radius=8)
            row.grid(row=i, column=0, sticky="ew", padx=2, pady=3)
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text="🎞", font=ctk.CTkFont(FONT_UI, 14)).grid(
                row=0, column=0, padx=(10, 8), pady=7)
            ctk.CTkLabel(row, text=os.path.basename(path), anchor="w",
                         font=ctk.CTkFont(FONT_UI, 12), text_color=TEXT).grid(
                row=0, column=1, sticky="w")
            ctk.CTkButton(row, text="✕", width=28, height=28, corner_radius=6,
                          fg_color="transparent", hover_color=HOVER, text_color=MUTED,
                          font=ctk.CTkFont(FONT_UI, 12),
                          command=lambda p=path: self._remove_file(p)).grid(
                row=0, column=2, padx=(4, 8))

    # ===================================================================== BAŞLAT / DURDUR
    def _on_primary(self):
        if self.is_running:
            self.cancel_event.set()
            self.primary_btn.configure(text=self.t("stopping"), state="disabled")
            return
        if not self.files:
            self._log(self.t("err_no_file"))
            return

        formats = [f for f in ("srt", "vtt", "txt", "json") if self.fmt_vars[f].get()]
        if not formats:
            self._log(self.t("no_format"))
            return

        opts = {
            "model": self.model_var.get(),
            "max_chars": int(self.char_slider.get()),
            "min_dur": round(self.min_slider.get(), 1),
            "sentence_only": bool(self.sentence_switch.get()),
            "src_lang": self._cur_src_code,
            "initial_prompt": (self.term_entry.get().strip() or None),
            "formats": formats,
            "out_dir": self.out_dir,
            "preview": bool(self.preview_switch.get()),
        }

        self.cancel_event.clear()
        self._set_busy(True)
        self.progress.set(0)
        self.percent_lbl.configure(text="0%")
        self._log("─" * 56)
        self._log(self.t("starting"))
        self._log(self.t("settings_line", opts["max_chars"], f"{opts['min_dur']:.1f}",
                         self.t("on") if opts["sentence_only"] else self.t("off"),
                         ", ".join(f.upper() for f in formats)))
        threading.Thread(target=self._batch_worker,
                         args=(list(self.files), opts), daemon=True).start()

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

    def _notify(self, out_path):
        """İşlem bitince Windows bildirimi; tıklayınca çıktı klasörünü açar."""
        def run():
            try:
                from winotify import Notification
                folder = os.path.dirname(out_path)
                uri = "file:///" + folder.replace("\\", "/")
                toast = Notification(app_id="AutoSRT",
                                     title=self.t("notify_title"),
                                     msg=self.t("notify_msg", os.path.basename(out_path)),
                                     duration="short", launch=uri)
                toast.add_actions(label=self.t("open_folder"), launch=uri)
                toast.show()
            except Exception as e:
                self.log_queue.put(("raw", f"\n[notify] {e}\n"))
        threading.Thread(target=run, daemon=True).start()

    # ============================================= ARKA PLAN TRANSCRIBE (THREAD)
    def _batch_worker(self, files, opts):
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
            key = (opts["model"], device, compute)
            if self.model is None or self._model_key != key:
                # VRAM tahliye protokolu: yeni modeli yuklemeden ONCE eskisini
                # bellekten bosalt. Aksi halde dusuk VRAM'li GPU'larda (6 GB)
                # eski + yeni model bir an icin ayni anda bellekte olur -> OOM.
                if self.model is not None:
                    del self.model
                    self.model = None
                    self._model_key = None
                    gc.collect()
                self._log(self.t("loading_model", opts["model"], device, compute))
                self._log(self.t("first_use"))
                self.model = WhisperModel(opts["model"], device=device, compute_type=compute)
                self._model_key = key
                self._log(self.t("model_ready"))
            if self.cancel_event.is_set():
                raise _Cancelled()

            total = len(files)
            saved_any = None
            for fi, path in enumerate(files):
                if self.cancel_event.is_set():
                    raise _Cancelled()
                self._log("─" * 56)
                self._log(self.t("file_progress", fi + 1, total)
                          + f"  •  {os.path.basename(path)}")

                blocks = self._transcribe_one(path, opts, fi, total)

                # Önizleme/düzenleme (toggle açıksa): worker burada GUI'yi bekler
                if opts["preview"]:
                    self.q(("status", "st_editing", WARN))
                    ev = threading.Event()
                    result = {"blocks": None, "cancel": False}
                    self.q(("preview", {"name": os.path.basename(path),
                                        "blocks": blocks, "event": ev, "result": result}))
                    ev.wait()
                    if result["cancel"]:
                        raise _Cancelled()
                    if result["blocks"] is not None:
                        blocks = result["blocks"]

                # Kaydet (seçili biçimler -> hedef klasör)
                out_dir = opts["out_dir"] or os.path.dirname(path)
                os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                for fmt in opts["formats"]:
                    ext, writer = WRITERS[fmt]
                    out_path = os.path.join(out_dir, base + "." + ext)
                    writer(blocks, out_path)
                    self._log(self.t("saved", out_path))
                    saved_any = out_path
                self._log("✓ " + self.t("done_blocks", len(blocks)))

            self.q(("progress", 1.0))
            self._log("─" * 56)
            self._log("✓ " + self.t("batch_done", total))
            self.q(("done", saved_any))

        except _Cancelled:
            self._log("■ " + self.t("cancelled"))
            self.q(("cancelled", None))
        except Exception as e:
            self._log(f"✕ {type(e).__name__}: {e}")
            self.q(("error", str(e)))

    def _transcribe_one(self, video_path, opts, fi, total):
        self.q(("status", "st_analyzing", self.accent))
        segments, info = self.model.transcribe(
            video_path,
            beam_size=10,
            language=opts["src_lang"],
            initial_prompt=opts["initial_prompt"],
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
        self.q(("status", "st_generating", self.accent))

        t0 = time.time()
        max_chars = opts["max_chars"]
        sentence_only = opts["sentence_only"]
        # Hangi noktalama bölme tetikler? (sentence_only -> sadece cümle sonu)
        punct = ('.', '?', '!') if sentence_only else ('.', ',', '?', '!', ':', ';')

        blocks = []  # [start, end, text]

        def _tick(end_t):
            if duration:
                frac = min(max(end_t / duration, 0.0), 1.0)
                overall = (fi + frac) / total
                el = time.time() - t0
                eta = (el / frac - el) if frac > 0.01 else 0
                self.q(("progress", overall))
                self.q(("stats", {"el": el, "eta": eta, "blocks": len(blocks),
                                  "lang": lang, "fi": fi + 1, "total": total}))

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
        min_dur = opts["min_dur"]
        if min_dur > 0:
            for i, b in enumerate(blocks):
                if b[1] - b[0] < min_dur:
                    new_end = b[0] + min_dur
                    if i + 1 < len(blocks):
                        new_end = min(new_end, blocks[i + 1][0] - 0.05)
                    if new_end > b[1]:
                        b[1] = new_end

        return blocks

    # ----------------------------------------------------- önizleme/düzenleme penceresi
    def _open_editor(self, payload):
        ev = payload["event"]
        result = payload["result"]
        blocks = payload["blocks"]
        try:
            win = ctk.CTkToplevel(self)
            win.title(self.t("editor_title", payload["name"]))
            win.geometry("780x620")
            win.configure(fg_color=BG)
            win.transient(self)
            win.grid_columnconfigure(0, weight=1)
            win.grid_rowconfigure(1, weight=1)
            self.after(220, lambda: self._set_toplevel_icon(win))

            ctk.CTkLabel(win, text=self.t("editor_hint"), text_color=MUTED,
                         font=ctk.CTkFont(FONT_UI, 12), anchor="w",
                         wraplength=720, justify="left").grid(
                row=0, column=0, sticky="ew", padx=20, pady=(18, 8))

            box = ctk.CTkTextbox(win, corner_radius=12, fg_color=TERM_BG,
                                 text_color=TEXT, font=ctk.CTkFont(FONT_MONO, 13),
                                 wrap="word", border_width=1, border_color=BORDER_SOFT)
            box.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 12))
            box.insert("1.0", blocks_to_srt_text(blocks))

            bar = ctk.CTkFrame(win, fg_color="transparent")
            bar.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 18))
            bar.grid_columnconfigure(0, weight=1)

            def do_save():
                parsed = parse_srt(box.get("1.0", "end"))
                result["blocks"] = parsed if parsed else blocks
                ev.set()
                win.destroy()

            def do_cancel():
                result["cancel"] = True
                ev.set()
                win.destroy()

            ctk.CTkButton(bar, text=self.t("cancel"), width=120, height=42, corner_radius=10,
                          fg_color=SURFACE2, hover_color=HOVER, text_color=TEXT,
                          border_width=1, border_color=BORDER,
                          font=ctk.CTkFont(FONT_UI, 13), command=do_cancel).grid(
                row=0, column=1, padx=(0, 10))
            ctk.CTkButton(bar, text=self.t("save"), width=160, height=42, corner_radius=10,
                          fg_color=self.accent, hover_color=self.accent_hover,
                          text_color="#ffffff", font=ctk.CTkFont(FONT_UI, 14, weight="bold"),
                          command=do_save).grid(row=0, column=2)

            win.protocol("WM_DELETE_WINDOW", do_cancel)
            win.after(120, lambda: (win.lift(), win.focus_force(), self._grab(win)))
        except Exception as e:
            # Düzenleyici açılamazsa worker'ı serbest bırak; orijinal bloklar kullanılır.
            self.log_queue.put(("raw", f"\n[editor] {e}\n"))
            result["blocks"] = None
            ev.set()

    def _grab(self, win):
        try:
            win.grab_set()
        except Exception:
            pass

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
                elif kind == "preview":
                    self._open_editor(item[1])
                elif kind == "done":
                    self.last_srt = item[1]
                    if item[1]:
                        self.open_btn.configure(state="normal")
                    self._set_status("st_done", GREEN)
                    self.percent_lbl.configure(text="100%")
                    self._set_busy(False)
                    if item[1]:
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
        prefix = ""
        if d.get("total", 1) > 1:
            prefix = self.t("file_progress", d.get("fi", 1), d["total"]) + " • "
        self.stats_lbl.configure(text=prefix + self.t("stats", _clock(d["el"]),
                                                       _clock(d["eta"]), d["blocks"],
                                                       d["lang"]))

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
            self.clear_all_btn.configure(state="disabled")
            self.primary_btn.configure(text=self.t("stop"), state="normal",
                                       fg_color=RED, hover_color="#e11d48",
                                       text_color="#ffffff", border_width=0)
        else:
            self.cancel_event.clear()
            self.browse_btn.configure(state="normal")
            self.model_menu.configure(state="normal")
            self.clear_all_btn.configure(state="normal")
            self.progress.stop()
            self._pb_mode = "det"
            self._update_primary()

    def _update_primary(self):
        if getattr(self, "is_running", False):
            return
        n = len(self.files)
        if n > 0:
            txt = self.t("start_n", n) if n > 1 else self.t("start")
            self.primary_btn.configure(text=txt, state="normal",
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
