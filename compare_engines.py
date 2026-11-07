# -*- coding: utf-8 -*-
"""
compare_engines.py — Aynı ses dosyasını iki motorla da işleyip ölçer.

Amaç: "hangisi daha iyi" sorusunu tahminle değil ölçümle yanıtlamak.
İki SRT üretir ve blok/kelime sayısı, ilk-son blok zaman damgası farkı ve
işlem süresini karşılaştırmalı olarak raporlar.

Kullanım:
    python compare_engines.py "video.mp4"
    python compare_engines.py "video.mp4" --lang tr --terms "Ada,Bora" --out-dir cikti
    python compare_engines.py "video.mp4" --only whisper     # tek motor

Gemini için GEMINI_API_KEY ortam değişkeni (ya da uygulamada kayıtlı anahtar)
gerekir. GUI ile aynı motor kodunu kullanır — main.py'den import edilir.
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as A          # noqa: E402  (DLL entegrasyonu main içinde yapılır)


def _fmt(sec):
    return A.format_timestamp(sec) if sec is not None else "—"


def run_engine(name, path, args):
    """Tek motoru çalıştırır; ölçüm sözlüğü döndürür."""
    def logt(key, *a):
        try:
            s = A.L["tr"].get(key, key)
            print("    " + (s.format(*a) if a else s))
        except Exception:
            print("    " + key)

    cls = A.GeminiEngine if name == "gemini" else A.WhisperEngine
    eng = cls(logt=logt, log=lambda s: None, status=lambda k, kind: None)
    eng.configure({
        "model": args.model,
        "beam_size": 5 if args.speed == "fast" else 10,
        "vad_threshold": 0.5,
        "api_key": "",                    # anahtar env / app-data'dan gelir
    })

    print("\n=== %s ===" % name.upper())
    t0 = time.time()
    try:
        eng.prepare(threading.Event())
        words = eng.transcribe(path, args.lang, args.terms,
                               lambda f, n=0: None, threading.Event())
    except A.EngineError as e:
        print("    HATA: %s" % A.L["tr"].get(e.key, e.key))
        if e.detail:
            print("    detay: %s" % e.detail)
        return None
    except Exception as e:
        print("    HATA: %s: %s" % (type(e).__name__, e))
        return None
    elapsed = time.time() - t0

    # GUI ile AYNI ardıl işlem: ortak dilimleyici + min-süre + sanitize
    blocks = A.slice_words_to_blocks(words, args.max_chars, args.sentence_only)
    if args.min_dur > 0:
        for i, b in enumerate(blocks):
            if b[1] - b[0] < args.min_dur:
                new_end = b[0] + args.min_dur
                if i + 1 < len(blocks):
                    new_end = min(new_end, blocks[i + 1][0] - 0.05)
                if new_end > b[1]:
                    b[1] = new_end
    blocks = A._sanitize_blocks(blocks)

    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "%s.%s.srt" % (base, name))
    A.write_srt(blocks, out_path)

    return {
        "engine": name,
        "elapsed": elapsed,
        "blocks": len(blocks),
        "words": len(words),
        "chars": sum(len(b[2]) for b in blocks),
        "first": blocks[0][0] if blocks else None,
        "last": blocks[-1][1] if blocks else None,
        "srt": out_path,
    }


def main():
    p = argparse.ArgumentParser(description="AutoSRT motor karşılaştırması")
    p.add_argument("audio", help="video ya da ses dosyası")
    p.add_argument("--only", choices=("whisper", "gemini"), help="tek motoru çalıştır")
    p.add_argument("--lang", default=None, help="kaynak dil (whisper kodu: tr, en, az…)")
    p.add_argument("--terms", default=None, help="terim ipucu (virgülle ayrılmış)")
    p.add_argument("--model", default=A.MODEL_SIZE, help="Whisper model boyutu")
    p.add_argument("--speed", choices=("fast", "balanced"), default="balanced")
    p.add_argument("--max-chars", type=int, default=45, dest="max_chars")
    p.add_argument("--min-dur", type=float, default=0.0, dest="min_dur")
    p.add_argument("--sentence-only", action="store_true", dest="sentence_only")
    p.add_argument("--out-dir", default=None, dest="out_dir")
    args = p.parse_args()

    if not os.path.exists(args.audio):
        print("Dosya bulunamadı: %s" % args.audio)
        return 2

    names = [args.only] if args.only else ["whisper", "gemini"]
    if "gemini" in names and not args.only:
        if not A.GeminiEngine.available():
            print("(google-genai kurulu değil → Gemini atlanıyor)")
            names.remove("gemini")
        elif not A.gemini_api_key():
            print("(GEMINI_API_KEY yok → Gemini atlanıyor)")
            names.remove("gemini")

    print("Dosya: %s" % os.path.basename(args.audio))
    print("Ayarlar: max_chars=%d, min_dur=%.1f, sadece cümle sonu=%s"
          % (args.max_chars, args.min_dur, args.sentence_only))

    results = [r for r in (run_engine(n, args.audio, args) for n in names) if r]
    if not results:
        print("\nHiçbir motor çalıştırılamadı.")
        return 1

    print("\n" + "=" * 74)
    print("%-10s %9s %8s %8s %9s %11s %11s"
          % ("MOTOR", "SÜRE", "BLOK", "KELİME", "KARAKTER", "İLK BLOK", "SON BLOK"))
    print("-" * 74)
    for r in results:
        print("%-10s %8.1fs %8d %8d %9d %11s %11s"
              % (r["engine"], r["elapsed"], r["blocks"], r["words"], r["chars"],
                 _fmt(r["first"]), _fmt(r["last"])))
    print("=" * 74)

    if len(results) == 2:
        a, b = results
        print("\nFarklar (%s → %s):" % (a["engine"], b["engine"]))
        print("  işlem süresi : %+.1f sn  (%s %.1f× hızlı)"
              % (b["elapsed"] - a["elapsed"],
                 a["engine"] if a["elapsed"] < b["elapsed"] else b["engine"],
                 max(a["elapsed"], b["elapsed"]) / max(min(a["elapsed"], b["elapsed"]), 1e-9)))
        print("  blok sayısı  : %+d" % (b["blocks"] - a["blocks"]))
        print("  kelime sayısı: %+d" % (b["words"] - a["words"]))
        print("  karakter     : %+d" % (b["chars"] - a["chars"]))
        if a["first"] is not None and b["first"] is not None:
            print("  ilk blok farkı: %+.2f sn" % (b["first"] - a["first"]))
        if a["last"] is not None and b["last"] is not None:
            print("  son blok farkı: %+.2f sn" % (b["last"] - a["last"]))

    print("\nÜretilen SRT dosyaları:")
    for r in results:
        print("  %s" % r["srt"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
