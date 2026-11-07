# -*- coding: utf-8 -*-
"""Parakeet TDT 0.6B v3 motoru (onnx-asr + onnxruntime-gpu).

Neden AYRI SÜREÇ?
  * VRAM garantisi: süreç bitince sürücü belleği işletim sistemine kesin olarak
    geri verir — Whisper ile Parakeet ASLA aynı anda VRAM'de olmaz.
  * DLL izolasyonu: ctranslate2 (Whisper) kendi cudnn64_9.dll stub'ını taşır;
    onnxruntime'ın cuDNN alt kütüphaneleriyle aynı süreçte karışma riski yok.
  * Dayanıklılık: onnxruntime'da yerel bir çökme GUI'yi düşürmez.

cuDNN 9 alt kütüphanelerini (cudnn_engines_*.dll) LoadLibraryEx +
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS ile arar → PATH YETMEZ, her nvidia/*/bin
klasörü os.add_dll_directory ile eklenmelidir (yoksa "Could not locate
cudnn_engines_tensor_ir64_9.dll" → CUDNN_BACKEND_API_FAILED → CPU'ya düşüş).

Model kartı (nvidia/parakeet-tdt-0.6b-v3): 25 Avrupa dili. Türkçe YOK.
"""
import json
import os
import subprocess
import sys
import threading

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"

# Model kartındaki 25 dil (ISO 639-1)
SUPPORTED_LANGS = frozenset((
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
))

WINDOW_MAX = 24.0      # Parakeet tek seferde ~20–30 sn ister; VAD aralarından bölünür
TOKEN_TAIL = 0.3       # pencerenin son kelimesi için tahmini bitiş payı (sn)
WORD_MAX = 1.5         # bir kelimenin en fazla süresi (uzun sessizlikleri yutmasın)


def supports(lang):
    return (lang or "").split("-")[0].lower() in SUPPORTED_LANGS


def tokens_to_words(tokens, timestamps, offset):
    """SentencePiece token'larını kelimelere birleştirir.
    Yeni kelime ' ' ile başlayan token'la başlar; noktalama token'ları
    önceki kelimeye yapışır. Zaman damgaları token BAŞLANGIÇLARIDIR."""
    words = []
    for tok, ts in zip(tokens or [], timestamps or []):
        if not tok:
            continue
        t = float(ts) + offset
        if tok.startswith(" ") or tok.startswith("▁") or not words:
            text = tok.replace("▁", " ").strip()
            if not text:
                continue
            words.append({"word": " " + text, "start": t, "last": t})
        else:
            words[-1]["word"] += tok
            words[-1]["last"] = t
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else w["last"] + TOKEN_TAIL
        w["end"] = round(min(max(nxt, w["last"] + 0.04), w["start"] + WORD_MAX), 3)
        w["start"] = round(w["start"], 3)
        del w["last"]
    return words


# =====================================================================
#  Alt süreç tarafı:  python -m autosrt_core.parakeet <wav> <out.json> [--cpu]
# =====================================================================
def _add_nvidia_dll_dirs():
    import glob
    import site
    dirs = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        dirs.extend(glob.glob(os.path.join(sp, "nvidia", "*", "bin")))
    for d in dirs:
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def _emit(kind, **kw):
    sys.stdout.write(json.dumps(dict(kind=kind, **kw), ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _worker_main(argv):
    wav_path, out_path = argv[0], argv[1]
    force_cpu = "--cpu" in argv
    _add_nvidia_dll_dirs()

    import numpy as np
    import onnxruntime as ort
    try:
        ort.preload_dlls()
    except Exception:
        pass
    import onnx_asr

    from .audio import read_wav, vad_chunks, speech_windows
    from . import SR

    audio = read_wav(wav_path)
    total = len(audio) / SR
    windows = speech_windows(vad_chunks(audio, 0.5), total, max_len=WINDOW_MAX)
    _emit("info", total=total, windows=len(windows))

    providers = ["CPUExecutionProvider"]
    if not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = [("CUDAExecutionProvider", {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
        }), "CPUExecutionProvider"]

    model = onnx_asr.load_model(MODEL_NAME, providers=providers).with_timestamps()

    # Isınma + gerçek sağlayıcı tespiti: onnxruntime CUDA başarısız olursa
    # sessizce CPU'ya düşer; bunu kullanıcıya dürüstçe bildirmek için kontrol.
    model.recognize(np.zeros(SR, dtype=np.float32), sample_rate=SR)
    device = "cpu"
    try:
        sess = getattr(getattr(model, "asr", None), "_encoder", None)
        if sess is not None and "CUDAExecutionProvider" in sess.get_providers():
            device = "cuda"
    except Exception:
        pass
    _emit("loaded", device=device)

    words = []
    for i, (a, b) in enumerate(windows):
        seg = audio[int(a * SR):int(b * SR)]
        if len(seg) < SR * 0.2:
            continue
        r = model.recognize(seg, sample_rate=SR)
        words.extend(tokens_to_words(r.tokens, r.timestamps, a))
        _emit("progress", frac=(i + 1) / max(1, len(windows)), words=len(words))

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"words": words, "device": device, "windows": len(windows)}, f,
                  ensure_ascii=False)
    os.replace(tmp, out_path)
    _emit("done", words=len(words))


if __name__ == "__main__":
    try:
        _worker_main(sys.argv[1:])
    except Exception as e:  # ham hata üst sürece JSON olarak gider
        _emit("error", detail="%s: %s" % (type(e).__name__, e))
        sys.exit(1)


# =====================================================================
#  Ana süreç tarafı: TranscriptionEngine uyumlu sarmalayıcı
# =====================================================================
class ParakeetError(Exception):
    def __init__(self, key, detail=""):
        super().__init__(key)
        self.key = key
        self.detail = detail


def available():
    try:
        import importlib.util
        return (importlib.util.find_spec("onnx_asr") is not None
                and importlib.util.find_spec("onnxruntime") is not None)
    except Exception:
        return False


def run_parakeet(wav_path, out_path, progress_cb=None, stop_event=None, log=None,
                 force_cpu=False):
    """Parakeet'i alt süreçte çalıştırır; kelime listesini döndürür.
    progress_cb(frac, n_words); log(kind, payload) bilgilendirme için."""
    log = log or (lambda *a: None)
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = sys.executable
    # pythonw.exe altında da stdout borusu çalışır; konsol açılmasın:
    flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
    cmd = [exe, "-X", "utf8", "-m", "autosrt_core.parakeet", wav_path, out_path]
    if force_cpu:
        cmd.append("--cpu")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=pkg_parent, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", creationflags=flags, env=env)

    # stderr'i ayrı thread'de boşalt (onnxruntime uyarıları boruyu tıkamasın)
    err_tail = []

    def _drain():
        for line in proc.stderr:
            err_tail.append(line.rstrip())
            del err_tail[:-30]
    threading.Thread(target=_drain, daemon=True).start()

    # Durdur düğmesi: süreci öldür
    killer_stop = threading.Event()

    def _watch():
        while not killer_stop.wait(0.3):
            if stop_event is not None and stop_event.is_set():
                try:
                    proc.kill()
                except Exception:
                    pass
                return
    threading.Thread(target=_watch, daemon=True).start()

    error = None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            k = msg.get("kind")
            if k == "progress" and progress_cb:
                progress_cb(msg.get("frac", 0.0), msg.get("words", 0))
            elif k == "error":
                error = msg.get("detail")
            else:
                log(k, msg)
        proc.wait()
    finally:
        killer_stop.set()

    if stop_event is not None and stop_event.is_set():
        return None
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise ParakeetError("pk_err_run", error or "\n".join(err_tail[-8:]))
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)
