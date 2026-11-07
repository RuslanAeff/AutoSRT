# -*- coding: utf-8 -*-
"""hybrid_cli.py — hibrit boru hattını arayüzsüz çalıştırır (test / toplu iş).

    python hybrid_cli.py "video.mp4"                         # Whisper + Parakeet + Gemini
    python hybrid_cli.py "video.mp4" --mode whisper_parakeet # Gemini yok
    python hybrid_cli.py "video.mp4" --lang ru --chunk-min 8
    python hybrid_cli.py "video.mp4" --keep                  # iş klasörünü silme

Ctrl+C güvenlidir: tamamlanan aşamalar kayıtlıdır, aynı komut kaldığı yerden sürer.
GUI ile aynı çekirdeği kullanır (main.py'den import edilir).
"""
import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as A                                   # noqa: E402  (DLL entegrasyonu)
from autosrt_core import judge as J                # noqa: E402
from autosrt_core import pipeline as P             # noqa: E402

LANG = "tr"


def T(key, *a):
    s = A.L[LANG].get(key, key)
    try:
        return s.format(*a) if a else s
    except Exception:
        return s + " " + " ".join(map(str, a))


class CliCallbacks(P.Callbacks):
    def __init__(self):
        self._last_stage = None

    def logt(self, key, *args):
        print("  " + T(key, *args), flush=True)

    def log(self, text):
        print("  " + text, flush=True)

    def stage(self, key, *args):
        s = T(key, *args)
        if s != self._last_stage:
            print("\n▶ " + s, flush=True)
            self._last_stage = s

    def progress(self, frac):
        pass

    def wait(self, seconds, kind):
        if seconds:
            print("\r  ⏳ " + T("wait_" + kind, seconds) + "   ", end="", flush=True)
        else:
            print("\r" + " " * 70 + "\r", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "whisper_parakeet"])
    ap.add_argument("--lang", default=None, help="kaynak dil (en, ru, fr, es, de…); boş = algıla")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--gemini-model", default=J.DEFAULT_MODEL)
    ap.add_argument("--chunk-min", type=float, default=10)
    ap.add_argument("--max-chars", type=int, default=45)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--keep", action="store_true", help="iş klasörünü silme")
    args = ap.parse_args()

    opts = {"engine": args.mode, "model": args.model, "beam_size": 10, "vad_threshold": 0.5,
            "src_lang": args.lang, "initial_prompt": None, "max_chars": args.max_chars,
            "sentence_only": False, "min_dur": 0.0, "gemini_model": args.gemini_model,
            "chunk_min_auto": args.chunk_min}

    stop = threading.Event()
    eng = A.WhisperEngine(logt=lambda k, *a: print("  " + T(k, *a)),
                          log=lambda s: None, status=lambda *a: None)
    pipe = P.HybridPipeline(eng, A.slice_words_to_blocks, CliCallbacks(), stop_event=stop,
                            secret_loader=A.load_secret,
                            app_dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        out = pipe.run(args.video, opts)
    except KeyboardInterrupt:
        stop.set()
        print("\n■ Durduruldu — ilerleme kaydedildi.")
        return 130
    except P.PipelineError as e:
        print("\n" + T(e.key))
        if e.detail:
            print("  ✕ " + e.detail)
        return 2

    if out["status"] != "done":
        print("Manuel mod: GUI sihirbazını kullanın.")
        return 3

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.video))
    base = os.path.splitext(os.path.basename(args.video))[0]
    src = A._sanitize_blocks(out["src_blocks"])
    # GUI ile aynı adlandırma: Gemini yoksa "video.srt", varsa "video.<dil>.srt" + "video.tr.srt"
    if out["tr_blocks"] is None:
        p1 = os.path.join(out_dir, "%s.srt" % base)
    else:
        p1 = os.path.join(out_dir, "%s.%s.srt" % (base, out["src_lang"]))
    A.write_srt(src, p1)
    print("\n" + T("saved", p1))
    if out["tr_blocks"] is not None:
        p2 = os.path.join(out_dir, "%s.tr.srt" % base)
        A.write_srt(A._sanitize_blocks(out["tr_blocks"]), p2)
        print(T("saved", p2))
    st = out["stats"]
    print(T("hy_done", st["lines"], st["dis"], st["low"], st["unverified"]))
    if not args.keep:
        out["job"].cleanup()
        print(T("hy_cleanup"))
    else:
        print("İş klasörü: " + out["job"].dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
