# -*- coding: utf-8 -*-
"""Gemini hakem: ses + iki transkript → düzeltilmiş kaynak metin + Türkçe çeviri.

Gemini YALNIZCA metni düzeltir; satır id'leri ve zaman damgaları Whisper'dan
gelir ve asla değişmez (senkron kaymaz).

Hem otomatik mod (API) hem manuel mod (web sohbeti) aynı istem metnini,
aynı şemayı ve aynı doğrulamayı kullanır — bu modül ikisinin ortak kaynağıdır.
"""
import json
import os
import re
import time

DEFAULT_MODEL = "gemini-3.8-flash"     # ücretsiz katmanda; ses girdisi + yapılandırılmış çıktı
# Arayüzdeki açılır liste. Hepsi resmî fiyat sayfasında ücretsiz katmanda ve ses
# girdisi kabul ediyor (Kasım 2026). Liste dışı bir ad da elle yazılabilir.
GEMINI_MODELS = (
    "gemini-3.8-flash",        # varsayılan: en güçlü Flash
    "gemini-3.5-flash-lite",   # daha hızlı / hafif
    "gemini-3.1-flash-lite",
)
CONTEXT_LINES = 4                      # önceki parçadan bağlam satırı (yalnızca okunur)
MAX_WAIT_SEC = 300                     # tek bir geri çekilmede en uzun bekleme
MAX_TRIES = 6                          # 429/5xx için toplam deneme
FILE_ACTIVE_TIMEOUT = 300              # Files API: PROCESSING → ACTIVE bekleme sınırı

LANG_NAMES = {
    "en": "English", "ru": "Russian", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "uk": "Ukrainian", "pl": "Polish", "nl": "Dutch",
    "tr": "Turkish", "az": "Azerbaijani", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean",
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "source_text": {"type": "string"},
                    "turkish_text": {"type": "string"},
                    "confidence": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["id", "source_text", "turkish_text"],
            },
        },
    },
    "required": ["lines"],
}


def lang_name(code):
    return LANG_NAMES.get((code or "").split("-")[0].lower(), code or "the original language")


def system_instruction(src_lang):
    src = lang_name(src_lang)
    return f"""You are a professional subtitle editor and {src}→Turkish translator.

You receive an audio clip and a JSON array of subtitle lines for that clip. Each line has:
  id, start, end (seconds, relative to the START of this audio clip),
  whisper  (transcript from ASR model A),
  parakeet (transcript from ASR model B; null if model B was not run),
  disagreement (true when the two transcripts differ after normalization).

Your job:
1. Listen to the audio. Compare both transcripts for every line. Pay special attention to
   lines with "disagreement": true — decide what is actually said, using the audio as the
   final authority (names, invented words, slang, numbers).
2. Write the correct {src} text for each line in "source_text": clean, properly capitalized
   and punctuated subtitle text. Keep only what is spoken within that line's time range;
   do not move words between lines, do not merge or split lines.
3. Translate each line into natural, fluent, idiomatic Turkish in "turkish_text", suitable
   for subtitles. Keep consistency with the surrounding lines (a sentence may continue
   across several lines — translate so each line reads naturally on its own while the
   whole sentence stays correct). Keep proper names as spoken.
4. "confidence": 0.0–1.0, how sure you are of source_text. "note": optional, very short,
   only when something is genuinely uncertain.

Strict rules:
- NEVER change ids, NEVER change the number of lines, NEVER change or output timestamps.
- Return exactly one output object per input line, in the same order.
- Never leave source_text or turkish_text empty; if a line is unintelligible, keep the most
  plausible transcript and give it a low confidence.
- Lines under "context" belong to the previous part: they are for reference only.
  Do NOT correct them and do NOT include them in the output.
"""


def build_items(lines, clip_start):
    out = []
    for ln in lines:
        out.append({
            "id": ln["id"],
            "start": round(ln["start"] - clip_start, 2),
            "end": round(ln["end"] - clip_start, 2),
            "whisper": ln["whisper"],
            "parakeet": ln.get("parakeet"),
            "disagreement": bool(ln.get("disagreement")),
        })
    return out


def build_context(prev_lines, results):
    ctx = []
    for ln in prev_lines[-CONTEXT_LINES:]:
        r = results.get(ln["id"]) or {}
        ctx.append({"id": ln["id"],
                    "source_text": r.get("source_text") or ln["whisper"],
                    "turkish_text": r.get("turkish_text") or ""})
    return ctx


def user_payload(items, context):
    return json.dumps({"context": context, "lines": items}, ensure_ascii=False, indent=1)


# ------------------------------------------------------------ doğrulama
class Validation:
    def __init__(self):
        self.good = {}        # id -> {"source_text","turkish_text","confidence","note"}
        self.missing = []     # beklenen ama gelmeyen / boş gelen id'ler
        self.extra = []       # beklenmeyen id'ler
        self.empty = []       # boş metinli id'ler
        self.order_ok = True
        self.parse_error = None
        self.expected = 0

    @property
    def ok(self):
        return self.parse_error is None and not self.missing and self.expected > 0

    def summary(self):
        return {"got": len(self.good), "expected": self.expected, "missing": self.missing,
                "extra": self.extra, "empty": self.empty, "order_ok": self.order_ok,
                "parse_error": self.parse_error}


def extract_json(text):
    """Model/sohbet cevabından JSON'u çıkarır: ```json bloklarını, baştaki/sondaki
    açıklama metnini tolere eder."""
    if text is None:
        raise ValueError("empty response")
    if not isinstance(text, str):
        return text
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found")
    s = s[min(starts):]
    end = max(s.rfind("}"), s.rfind("]"))
    if end >= 0:
        s = s[:end + 1]
    return json.loads(s)


def validate(raw, expected_ids):
    """Şemaya göre doğrula: tüm id'ler var mı, sıra doğru mu, boş metin var mı,
    satır sayısı tutuyor mu. Geçerli satırlar kısmi başarıda da korunur."""
    v = Validation()
    v.expected = len(expected_ids)
    try:
        data = extract_json(raw)
    except Exception as e:
        v.parse_error = "%s: %s" % (type(e).__name__, e)
        v.missing = list(expected_ids)
        return v
    rows = data.get("lines") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        v.parse_error = "no 'lines' array"
        v.missing = list(expected_ids)
        return v

    exp_set = set(expected_ids)
    seen_order = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            rid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if rid not in exp_set:
            v.extra.append(rid)
            continue
        if rid in v.good:
            continue                      # yineleneni yok say (ilk geçerli)
        src = str(row.get("source_text") or "").strip()
        tr = str(row.get("turkish_text") or "").strip()
        if not src or not tr:
            v.empty.append(rid)
            continue
        conf = row.get("confidence")
        try:
            conf = None if conf is None else max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = None
        note = str(row.get("note") or "").strip()[:200]
        v.good[rid] = {"source_text": src, "turkish_text": tr,
                       "confidence": conf, "note": note}
        seen_order.append(rid)
    v.order_ok = seen_order == [i for i in expected_ids if i in v.good]
    v.missing = [i for i in expected_ids if i not in v.good]
    return v


# ------------------------------------------------------------ hata sınıflandırma
class JudgeError(Exception):
    """kind ∈ rate | quota_daily | server | auth | model | net | invalid | other"""
    def __init__(self, kind, detail="", retry_after=None):
        super().__init__(kind)
        self.kind = kind
        self.detail = detail
        self.retry_after = retry_after


class SwitchToManual(Exception):
    """Kullanıcı beklemek yerine 'Manuel moda geç'e bastı."""


class Cancelled(Exception):
    pass


def _retry_delay(details_text):
    m = re.search(r'"?retryDelay"?\s*[:=]\s*"?(\d+(?:\.\d+)?)s', details_text or "")
    return float(m.group(1)) if m else None


def classify(e):
    """Ham exception → JudgeError. Anahtar asla mesaja girmez (SDK da koymaz)."""
    if isinstance(e, JudgeError):
        return e
    code = getattr(e, "code", None)
    status = str(getattr(e, "status", "") or "")
    details = getattr(e, "details", None)
    try:
        dtext = json.dumps(details) if details is not None else ""
    except Exception:
        dtext = str(details)
    msg = ("%s %s %s %s" % (type(e).__name__, e, status, dtext)).lower()
    detail = ("%s %s" % (code or "", getattr(e, "message", None) or str(e)))[:500]
    # anahtarı içerebilecek her şeyi maskele (savunma amaçlı)
    detail = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "AIza***", detail)

    if code == 429 or "resource_exhausted" in msg or "rate limit" in msg or "quota" in msg:
        if "perday" in msg or "per_day" in msg or "requestsperday" in msg or "daily" in msg:
            return JudgeError("quota_daily", detail)
        return JudgeError("rate", detail, _retry_delay(dtext))
    if code in (500, 502, 503, 504) or "unavailable" in msg or "deadline" in msg or "overloaded" in msg:
        return JudgeError("server", detail)
    if ("api key" in msg or "api_key" in msg or "unauthenticated" in msg
            or code in (401, 403) or "permission_denied" in msg):
        return JudgeError("auth", detail)
    if code == 404 or "not_found" in msg or "is not found" in msg:
        return JudgeError("model", detail)
    if (isinstance(e, OSError) or "connection" in msg or "timed out" in msg or "timeout" in msg
            or "network" in msg or "name resolution" in msg or "getaddrinfo" in msg):
        return JudgeError("net", detail)
    return JudgeError("other", detail)


# ------------------------------------------------------------ API anahtarı
def resolve_api_key(app_dir=None, secret_loader=None):
    """Öncelik: (1) GEMINI_API_KEY ortam değişkeni, (2) .env (uygulama klasörü),
    (3) %APPDATA%\\AutoSRT\\secrets.json. Anahtar hiçbir yerde loglanmaz."""
    k = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if k:
        return k
    for d in filter(None, [app_dir, os.getcwd()]):
        p = os.path.join(d, ".env")
        if os.path.isfile(p):
            try:
                from dotenv import dotenv_values
                k = (dotenv_values(p).get("GEMINI_API_KEY") or "").strip()
            except Exception:
                k = _parse_env_file(p)
            if k:
                return k
    if secret_loader:
        return (secret_loader("gemini_api_key") or "").strip()
    return ""


def _parse_env_file(p):
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


# ------------------------------------------------------------ otomatik mod
class GeminiJudge:
    def __init__(self, api_key, model=DEFAULT_MODEL, log=None, on_wait=None,
                 stop_event=None, manual_event=None):
        self.model = model or DEFAULT_MODEL
        self.log = log or (lambda *a: None)
        self.on_wait = on_wait or (lambda *a: None)     # on_wait(seconds_left, kind)
        self.stop_event = stop_event
        self.manual_event = manual_event
        self._mask = api_key            # hata ayrıntısında ASLA görünmesin
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=600_000))   # ms

    # --- iptal / manuel geçiş kontrolü ---
    def _check(self):
        if self.stop_event is not None and self.stop_event.is_set():
            raise Cancelled()
        if self.manual_event is not None and self.manual_event.is_set():
            raise SwitchToManual()

    def _sleep(self, sec, kind):
        """Geri sayımlı bekleme; her saniye arayüze kalan süreyi bildirir."""
        end = time.time() + sec
        while True:
            self._check()
            left = end - time.time()
            if left <= 0:
                break
            self.on_wait(int(left + 0.999), kind)
            time.sleep(min(1.0, left))
        self.on_wait(0, kind)

    def _call(self, fn):
        """429/5xx/ağ hatasında üstel geri çekilme. Günlük kota, geçersiz anahtar
        ve bilinmeyen model beklemeye DEĞMEZ → hemen yükselt."""
        delay = 10.0
        for attempt in range(MAX_TRIES):
            self._check()
            try:
                return fn()
            except (Cancelled, SwitchToManual):
                raise
            except Exception as e:
                je = classify(e)
                if self._mask:
                    je.detail = je.detail.replace(self._mask, "***")
                if je.kind not in ("rate", "server", "net") or attempt == MAX_TRIES - 1:
                    raise je
                wait = min(MAX_WAIT_SEC, max(delay, (je.retry_after or 0) + 1))
                self.log("gj_retry", attempt + 1, MAX_TRIES, int(wait), je.kind)
                self._sleep(wait, je.kind)
                delay = min(MAX_WAIT_SEC, delay * 2)

    def _upload(self, wav_path):
        f = self._call(lambda: self.client.files.upload(
            file=wav_path, config={"mime_type": "audio/wav"}))
        t0 = time.time()
        while True:
            state = str(getattr(f, "state", "") or "")
            if state.endswith("ACTIVE"):
                return f
            if state.endswith("FAILED"):
                raise JudgeError("other", "file processing FAILED")
            if time.time() - t0 > FILE_ACTIVE_TIMEOUT:
                raise JudgeError("server", "file stuck in PROCESSING")
            self._sleep(2, "processing")
            f = self._call(lambda: self.client.files.get(name=f.name))

    def judge_chunk(self, wav_path, items, context, src_lang):
        """Tek parça: yükle → ACTIVE bekle → yapılandırılmış JSON iste.
        Ham yanıt metnini döndürür (doğrulama çağıranda). Uzak dosya HER durumda silinir."""
        types = self._types
        uploaded = None
        try:
            uploaded = self._upload(wav_path)
            cfg = types.GenerateContentConfig(
                system_instruction=system_instruction(src_lang),
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA,
                temperature=0.2,
                max_output_tokens=60000,
                # Araç kullanmıyoruz: SDK'nın "AFC" uyarısını ve gereksiz döngüyü kapat.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            contents = [uploaded, user_payload(items, context)]
            resp = self._call(lambda: self.client.models.generate_content(
                model=self.model, contents=contents, config=cfg))
            text = getattr(resp, "text", None)
            if not text:
                fr = ""
                try:
                    fr = str(resp.candidates[0].finish_reason)
                except Exception:
                    pass
                raise JudgeError("invalid", "empty response " + fr)
            return text
        finally:
            if uploaded is not None:
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    pass
