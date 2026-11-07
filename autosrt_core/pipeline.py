# -*- coding: utf-8 -*-
"""Hibrit boru hattı: ses → Whisper → Parakeet → hizalama → Gemini → çıktı.

UI'dan bağımsızdır: Whisper motoru, dilimleyici ve geri çağrılar dışarıdan verilir.
Tüm aşamalar iş klasörüne checkpoint'lenir; tekrar çalıştırıldığında tamamlanmış
aşamalar atlanır. Başarıyla bitince iş klasörü silinir.

Modlar:
    whisper           — bu modül KULLANILMAZ (main.py'deki mevcut yol, birebir)
    whisper_parakeet  — Whisper metni + anlaşmazlık işaretleri (Gemini yok)
    hybrid            — Whisper + Parakeet + Gemini hakem (kaynak dil + Türkçe)
"""
import os
import time

from . import SR
from . import align as AL
from . import audio as AU
from . import judge as J
from . import manual as M
from . import parakeet as PK
from .jobs import Job, fingerprint, load_json, save_json

LOW_CONF = 0.6          # bunun altındaki Gemini güveni editörde işaretlenir
CLIP_PAD = 0.3          # parça sesinin satır sınırlarına eklenen pay (sn)

# Dosya içi ilerleme ağırlıkları (aşama → pay)
WEIGHTS = {"audio": 0.03, "whisper": 0.47, "parakeet": 0.10, "lines": 0.02,
           "gemini": 0.35, "output": 0.03}


class PipelineError(Exception):
    """Kullanıcıya gösterilebilir hata: i18n anahtarı + ham detay (yalnız günlüğe)."""
    def __init__(self, key, detail=""):
        super().__init__(key)
        self.key = key
        self.detail = detail


class Callbacks:
    """Varsayılan (sessiz) geri çağrılar. main.py kendi kuyruğuna bağlar."""
    def logt(self, key, *args): pass
    def log(self, text): pass
    def stage(self, key, *args): pass
    def progress(self, frac): pass
    def wait(self, seconds, kind): pass
    def stats(self, d): pass


# ------------------------------------------------------------------ yardımcılar
def apply_min_duration(blocks, min_dur):
    """main.py'deki min-süre kuralının aynısı: yalnız uzatır, sonraki bloğa taşmaz."""
    if min_dur <= 0:
        return blocks
    for i, b in enumerate(blocks):
        if b[1] - b[0] < min_dur:
            new_end = b[0] + min_dur
            if i + 1 < len(blocks):
                new_end = min(new_end, blocks[i + 1][0] - 0.05)
            if new_end > b[1]:
                b[1] = new_end
    return blocks


def plan_chunks(lines, pending_ids, chunk_sec):
    """Bekleyen satırları ≤ chunk_sec'lik parçalara böler.

    * Bölme her zaman iki SATIR arasından yapılır (kelime ortadan bölünmez) ve
      parçanın son %40'ındaki en uzun sessizlik seçilir.
    * Aradaki satırlar zaten çözülmüşse (ör. manuel modda kısmi cevap) yakın
      bekleyen satırlar yine aynı parçada toplanır; ses klibi aralığı kapsar.
    """
    pend = [ln for ln in lines if ln["id"] in pending_ids]
    chunks, cur = [], []
    for ln in pend:
        if cur and ln["end"] - cur[0]["start"] > chunk_sec:
            # cur'u en iyi sessizlikten böl
            k = len(cur)
            floor = cur[0]["start"] + 0.6 * chunk_sec
            best_gap = -1.0
            for j in range(1, len(cur)):
                if cur[j]["start"] < floor:
                    continue
                gap = cur[j]["start"] - cur[j - 1]["end"]
                if gap >= best_gap:
                    best_gap, k = gap, j
            chunks.append(cur[:k])
            cur = cur[k:]
            # kalan + yeni satır hâlâ sığmıyorsa (çok uzun tek boşluk) parçayı kapat
            if cur and ln["end"] - cur[0]["start"] > chunk_sec:
                chunks.append(cur)
                cur = []
        cur.append(ln)
    if cur:
        chunks.append(cur)
    return chunks


def clip_range(chunk, total):
    return max(0.0, chunk[0]["start"] - CLIP_PAD), min(total, chunk[-1]["end"] + CLIP_PAD)


def build_outputs(lines, results, mode, min_dur=0.0):
    """Satırlar + Gemini sonuçları → (kaynak_blokları, türkçe_blokları|None, işaretler).
    ZAMAN DAMGALARI HER ZAMAN WHISPER'DAN (lines) gelir."""
    src, tr, flags = [], [], []
    for ln in lines:
        r = results.get(ln["id"]) if results else None
        f = {"dis": bool(ln.get("disagreement")), "low": False,
             "unverified": False, "note": ""}
        if mode == "hybrid":
            if r and r.get("via") != "fallback":
                s, t = r["source_text"], r["turkish_text"]
                conf = r.get("confidence")
                f["low"] = conf is not None and conf < LOW_CONF
                f["note"] = r.get("note") or ""
            else:
                # Gemini doğrulayamadı → Whisper metni (Türkçe dosyada da)
                s = t = ln["whisper"]
                f["unverified"] = True
            tr.append([ln["start"], ln["end"], t])
        else:
            s = ln["whisper"]
        src.append([ln["start"], ln["end"], s])
        flags.append(f)
    apply_min_duration(src, min_dur)
    if mode == "hybrid":
        # Türkçe bloklar kaynakla AYNI zamanlamayı kullanır
        for a, b in zip(src, tr):
            b[0], b[1] = a[0], a[1]
        return src, tr, flags
    return src, None, flags


def lang_code(s):
    return (s or "").split()[0].split("-")[0].lower() if s else ""


# ------------------------------------------------------------------ boru hattı
class HybridPipeline:
    def __init__(self, whisper_engine, slicer, cb=None, stop_event=None,
                 manual_event=None, secret_loader=None, app_dir=None, jobs_root=None):
        self.whisper = whisper_engine
        self.slicer = slicer
        self.cb = cb or Callbacks()
        self.stop_event = stop_event
        self.manual_event = manual_event
        self.secret_loader = secret_loader
        self.app_dir = app_dir
        self.jobs_root = jobs_root

    # --- yardımcılar ---
    def _check(self):
        if self.stop_event is not None and self.stop_event.is_set():
            raise J.Cancelled()

    def _prog(self, stage, frac, mode):
        order = ["audio", "whisper", "parakeet", "lines", "gemini", "output"]
        w = dict(WEIGHTS)
        if mode != "hybrid":
            w["gemini"] = 0.0
        tot = sum(w.values())
        done = sum(w[s] for s in order[:order.index(stage)])
        self.cb.progress(min(1.0, (done + w[stage] * max(0.0, min(1.0, frac))) / tot))

    @staticmethod
    def fp_whisper(opts):
        return fingerprint({"m": opts.get("model"), "b": opts.get("beam_size"),
                            "v": opts.get("vad_threshold"), "l": opts.get("src_lang"),
                            "p": opts.get("initial_prompt")})

    # --- ana akış ---
    def run(self, video, opts):
        """Dönüş: dict(status="done"|"manual", src_blocks, tr_blocks, flags,
        src_lang, stats, job). status="manual" → kullanıcı manuel moda geçti."""
        mode = opts["engine"]
        cb = self.cb
        job = Job(video, root=self.jobs_root)
        done_before = [s for s in ("audio", "whisper", "parakeet", "lines")
                       if job.stage_info(s)]
        if done_before or job.load_results():
            cb.logt("hy_resume", len(done_before), len(job.load_results()))

        # 1) SES ----------------------------------------------------------
        cb.stage("stg_audio")
        wav = job.path("audio.wav")
        fp_audio = fingerprint({"v": 1, "sr": SR})
        if not job.is_done("audio", fp_audio):
            job.invalidate("audio")
            t = time.time()
            try:
                dur = AU.extract_wav(video, wav)
            except Exception as e:
                raise PipelineError("hy_err_audio", str(e))
            job.mark_done("audio", fp_audio, file="audio.wav", duration=dur)
            cb.logt("hy_audio", AL_fmt(dur), "%.1f" % (time.time() - t))
        else:
            cb.logt("hy_cached", "audio")
        total = job.stage_info("audio").get("duration") or 0.0
        self._prog("audio", 1.0, mode)
        self._check()

        # 2) WHISPER ------------------------------------------------------
        cb.stage("stg_whisper")
        fp_w = self.fp_whisper(opts)
        if not job.is_done("whisper", fp_w):
            job.invalidate("whisper")
            t0 = time.time()
            self.whisper.configure(opts)
            self.whisper.prepare(self.stop_event)
            self._check()

            def wprog(frac, n=0):
                self._prog("whisper", frac, mode)
                el = time.time() - t0
                cb.stats({"el": el, "eta": (el / frac - el) if frac > 0.01 else 0,
                          "blocks": n, "unit": "w",
                          "lang": self.whisper.detected_lang or "-"})
            try:
                words = self.whisper.transcribe(wav, opts.get("src_lang"),
                                                opts.get("initial_prompt"), wprog,
                                                self.stop_event)
            finally:
                # VRAM: Parakeet'ten ÖNCE Whisper modeli mutlaka boşaltılır.
                self.whisper.release()
                cb.logt("hy_released")
            self._check()
            det = self.whisper.detected_lang or ""
            save_json(job.path("whisper.json"), {
                "words": [{"word": w["word"], "start": float(w["start"]),
                           "end": float(w["end"]), "seg_end": bool(w.get("seg_end"))}
                          for w in words],
                "detected": det})
            job.mark_done("whisper", fp_w, file="whisper.json")
            cb.logt("hy_whisper_done", len(words), det or "-")
        else:
            cb.logt("hy_cached", "whisper")
        wdata = load_json(job.path("whisper.json"), {})
        wwords = wdata.get("words") or []
        src_lang = opts.get("src_lang") or lang_code(wdata.get("detected"))
        self._prog("whisper", 1.0, mode)
        self._check()

        # 3) PARAKEET -----------------------------------------------------
        cb.stage("stg_parakeet")
        pk_state = "off"
        pwords = []
        if not PK.available():
            pk_state = "skip:unavailable"
            cb.logt("pk_unavailable")
        elif not PK.supports(src_lang):
            pk_state = "skip:lang:" + (src_lang or "?")
            cb.logt("pk_skip_lang", src_lang or "?")
            cb.stage("stg_parakeet_skip", src_lang or "?")
        else:
            fp_p = fingerprint({"m": PK.MODEL_NAME, "v": 1})
            if not job.is_done("parakeet", fp_p):
                job.invalidate("parakeet")
                t0 = time.time()
                info = {}

                def plog(kind, msg):
                    if kind == "loaded":
                        info["device"] = msg.get("device")
                        cb.logt("pk_loaded", msg.get("device"))
                        if msg.get("device") != "cuda":
                            cb.logt("pk_cpu")
                try:
                    res = PK.run_parakeet(
                        wav, job.path("parakeet.json"),
                        progress_cb=lambda f, n=0: self._prog("parakeet", f, mode),
                        stop_event=self.stop_event, log=plog)
                except PK.ParakeetError as e:
                    res = None
                    cb.logt("pk_failed")
                    if e.detail:
                        cb.log("✕ " + e.detail.strip().splitlines()[-1][:300])
                self._check()
                if res is not None:
                    job.mark_done("parakeet", fp_p, file="parakeet.json",
                                  device=res.get("device"))
                    cb.logt("pk_done", len(res.get("words") or []),
                            "%.1f" % (time.time() - t0))
                    pk_state = fp_p
                else:
                    pk_state = "skip:failed"
            else:
                cb.logt("hy_cached", "parakeet")
                pk_state = fp_p
            if not pk_state.startswith("skip"):
                pwords = (load_json(job.path("parakeet.json"), {}) or {}).get("words") or []
        self._prog("parakeet", 1.0, mode)
        self._check()

        # 4) HİZALAMA / SATIRLAR -----------------------------------------
        cb.stage("stg_align")
        fp_l = fingerprint({"w": fp_w, "p": pk_state, "mc": opts.get("max_chars"),
                            "so": bool(opts.get("sentence_only"))})
        if not job.is_done("lines", fp_l):
            job.invalidate("lines")
            blocks = self.slicer(wwords, opts["max_chars"], opts["sentence_only"])
            lines, st = AL.align(wwords, pwords, blocks, src_lang or "en")
            save_json(job.path("lines.json"), {"lines": lines, "stats": st,
                                               "src_lang": src_lang})
            job.mark_done("lines", fp_l, file="lines.json")
        ldata = load_json(job.path("lines.json"), {})
        lines = ldata.get("lines") or []
        st = ldata.get("stats") or {}
        if pwords:
            cb.logt("al_done", len(lines), st.get("disagreements", 0))
        self._prog("lines", 1.0, mode)
        self._check()

        # 5) GEMINI -------------------------------------------------------
        results = {}
        if mode == "hybrid" and lines:
            results = self._gemini_stage(job, lines, total, opts, src_lang)
            if results is None:          # kullanıcı manuel moda geçti
                return {"status": "manual", "job": job, "src_lang": src_lang}
        self._prog("gemini", 1.0, mode)

        # 6) ÇIKTI --------------------------------------------------------
        cb.stage("stg_output")
        src, tr, flags = build_outputs(lines, results, mode, opts.get("min_dur", 0.0))
        stats = {"lines": len(lines), "dis": sum(f["dis"] for f in flags),
                 "low": sum(f["low"] for f in flags),
                 "unverified": sum(f["unverified"] for f in flags)}
        self._prog("output", 1.0, mode)
        return {"status": "done", "job": job, "src_blocks": src, "tr_blocks": tr,
                "flags": flags, "src_lang": src_lang or "src", "stats": stats}

    # --- Gemini aşaması ---
    def _gemini_stage(self, job, lines, total, opts, src_lang):
        cb = self.cb
        results = job.load_results()
        pending = {ln["id"] for ln in lines if ln["id"] not in results}
        if not pending:
            cb.logt("hy_cached", "gemini")
            return results
        if opts.get("manual_mode") or (self.manual_event is not None
                                       and self.manual_event.is_set()):
            cb.logt("hy_manual")
            return None

        key = J.resolve_api_key(self.app_dir, self.secret_loader)
        if not key:
            raise PipelineError("gj_err_nokey")
        try:
            judge = J.GeminiJudge(
                key, opts.get("gemini_model") or J.DEFAULT_MODEL,
                log=lambda k, *a: cb.logt(k, *a), on_wait=cb.wait,
                stop_event=self.stop_event, manual_event=self.manual_event)
        except ImportError as e:
            raise PipelineError("gj_err_nosdk", str(e))
        del key

        chunk_sec = float(opts.get("chunk_min_auto", 10)) * 60.0
        chunks = plan_chunks(lines, pending, chunk_sec)
        audio = AU.read_wav(job.path("audio.wav"))
        by_id = {ln["id"]: i for i, ln in enumerate(lines)}
        n = len(chunks)
        try:
            for ci, ch in enumerate(chunks):
                self._check()
                a, b = clip_range(ch, total)
                cb.stage("stg_gemini", ci + 1, n)
                cb.logt("gj_part", ci + 1, n, AL_fmt(a), AL_fmt(b), len(ch))
                clip = job.path("chunk.wav")
                AU.write_slice(audio, a, b, clip)
                ctx = J.build_context(lines[:by_id[ch[0]["id"]]], results)

                good, todo = {}, list(ch)
                for attempt in range(2):              # ilk deneme + 1 otomatik tekrar
                    items = J.build_items(todo, a)
                    try:
                        text = judge.judge_chunk(clip, items, ctx, src_lang)
                    except J.JudgeError as e:
                        if e.kind == "invalid" and attempt == 0:
                            cb.logt("gj_invalid", ci + 1, len(todo))
                            continue
                        if e.kind == "invalid":
                            break
                        raise
                    v = J.validate(text, [ln["id"] for ln in todo])
                    for rid, row in v.good.items():
                        row["via"] = "auto"
                        good[rid] = row
                    if v.ok:
                        break
                    todo = [ln for ln in todo if ln["id"] in v.missing]
                    if attempt == 0:
                        cb.logt("gj_invalid", ci + 1, len(todo))
                        if v.parse_error:
                            cb.log("✕ " + v.parse_error[:200])
                # iki denemeden sonra hâlâ eksik → Whisper metnine geri düş
                missing = [ln["id"] for ln in ch if ln["id"] not in good]
                for rid in missing:
                    good[rid] = {"source_text": lines[by_id[rid]]["whisper"],
                                 "turkish_text": lines[by_id[rid]]["whisper"],
                                 "confidence": None, "note": "", "via": "fallback"}
                if missing:
                    cb.logt("gj_fallback", ci + 1, len(missing))
                results = job.merge_results(good)       # checkpoint: parça bitti
                cb.logt("gj_part_ok", ci + 1, n, len(ch) - len(missing))
                self._prog("gemini", (ci + 1) / n, opts["engine"])
        except J.SwitchToManual:
            cb.logt("hy_manual")
            return None
        except J.JudgeError as e:
            raise PipelineError("gj_err_" + e.kind, e.detail)
        finally:
            try:
                os.remove(job.path("chunk.wav"))
            except OSError:
                pass
        job.mark_done("gemini", "all")
        return results


def AL_fmt(sec):
    sec = max(0.0, float(sec))
    return "%02d:%02d:%02d" % (int(sec // 3600), int(sec % 3600 // 60), int(sec % 60))


# ------------------------------------------------------------------ manuel oturum
class ManualSession:
    """Manuel sihirbazın arka ucu. Aynı iş klasörünü (checkpoint) kullanır."""

    def __init__(self, video, chunk_min=5, jobs_root=None):
        self.job = Job(video, root=jobs_root)
        ld = load_json(self.job.path("lines.json"), {}) or {}
        self.lines = ld.get("lines") or []
        self.src_lang = ld.get("src_lang") or "en"
        self.total = self.job.stage_info("audio").get("duration") or 0.0
        self.chunk_sec = float(chunk_min) * 60.0
        self._audio = None
        self.current = None      # aktif paket

    @property
    def ready(self):
        return bool(self.lines) and os.path.exists(self.job.path("audio.wav"))

    def progress(self):
        res = self.job.load_results()
        done = sum(1 for ln in self.lines if ln["id"] in res)
        return done, len(self.lines)

    def remaining_chunks(self):
        res = self.job.load_results()
        pending = {ln["id"] for ln in self.lines if ln["id"] not in res}
        return plan_chunks(self.lines, pending, self.chunk_sec)

    def next_package(self):
        """Sıradaki parçanın paketini hazırlar; hepsi bittiyse None."""
        chunks = self.remaining_chunks()
        if not chunks:
            self.current = None
            return None
        if self._audio is None:
            self._audio = AU.read_wav(self.job.path("audio.wav"))
        ch = chunks[0]
        a, b = clip_range(ch, self.total)
        res = self.job.load_results()
        idx = next(i for i, ln in enumerate(self.lines) if ln["id"] == ch[0]["id"])
        ctx = J.build_context(self.lines[:idx], res)
        done, tot = self.progress()
        part_no = 1 + sum(1 for _ in os.listdir(self.job.manual_dir) if _.endswith("_ids.json"))
        wav, prompt = M.write_package(self.job, part_no, self._audio, ch, ctx,
                                      self.src_lang, a, b)
        self.current = {"part": part_no, "remaining": len(chunks), "wav": wav,
                        "prompt": prompt, "ids": [ln["id"] for ln in ch],
                        "start": a, "end": b, "done": done, "total": tot}
        return self.current

    def submit(self, pasted_text):
        """Yapıştırılan cevabı doğrula; geçerli satırları kaydet. Validation döner."""
        if not self.current:
            raise RuntimeError("no active package")
        v = J.validate(pasted_text, self.current["ids"])
        if v.good:
            for row in v.good.values():
                row["via"] = "manual"
            self.job.merge_results(v.good)
        return v

    def all_done(self):
        return not self.remaining_chunks()
