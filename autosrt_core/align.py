# -*- coding: utf-8 -*-
"""Whisper ↔ Parakeet kelime hizalaması.

Yöntem (zaman + edit-distance hibriti):
  1. Whisper kelimeleri altyazı bloklarına (satırlara) atanır — satırlar ve
     zaman damgaları HER ZAMAN Whisper'dan gelir.
  2. Zaman çizgisi, Whisper'daki konuşma aralarından ≤ WINDOW sn'lik
     pencerelere bölünür; her Parakeet kelimesi orta noktasına göre bir
     pencereye düşer. (Modeller arası zaman kayması < 0.5 sn olduğundan
     aralardan bölmek eşleşmeyi bozmaz, büyük O(n·m) maliyetini önler.)
  3. Her pencerede normalize edilmiş belirteçler üzerinde difflib ile
     edit-distance hizalaması yapılır. Eşleşen/yer değiştiren Parakeet
     kelimeleri karşılık gelen Whisper kelimesinin satırına, araya giren
     (insert) kelimeler ise zamana en yakın satıra atanır.
  4. Satır başına normalize metinler karşılaştırılır → disagreement.

Normalizasyon: küçük harf, noktalama/tire/kesme işareti farkı, Rusça ё→е ve
rakam↔yazı (num2words) farkları anlaşmazlık SAYILMAZ.
"""
import difflib
import re
import unicodedata

WINDOW = 30.0        # hizalama penceresinin en fazla uzunluğu (sn)
WINDOW_GAP = 0.25    # pencere sınırı için gereken en küçük Whisper arası (sn)
ORPHAN_MAX = 2.0     # hiçbir satıra eşlenmeyen kelime en fazla bu kadar uzaktaki satıra eklenir

_NUM_RE = re.compile(r"^\d+([.,]\d+)?(st|nd|rd|th|er|e|ème|º|°|-?[a-zа-я]{1,3})?$", re.I)
_ORD_SUFFIX = {"st", "nd", "rd", "th", "er", "e", "ème", "º", "°"}


def _strip_punct(s):
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P")
                   and not unicodedata.category(ch).startswith("S"))


def _num_to_words(tok, lang):
    try:
        from num2words import num2words
    except Exception:
        return None
    m = re.match(r"^(\d+(?:[.,]\d+)?)(.*)$", tok)
    if not m:
        return None
    num, suf = m.group(1), m.group(2).lower().lstrip("-")
    try:
        if suf in _ORD_SUFFIX:
            return num2words(int(num), lang=lang, to="ordinal")
        val = float(num.replace(",", ".")) if ("." in num or "," in num) else int(num)
        return num2words(val, lang=lang)
    except Exception:
        return None


def normalize_tokens(text, lang="en"):
    """Metni karşılaştırma belirteçlerine çevirir."""
    lang = (lang or "en").split("-")[0].lower()
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("ё", "е").replace("’", "'").replace("‘", "'")
    # kısaltma kesme işaretleri: don't → dont (iki model farklı bölebilir)
    text = text.replace("'", "")
    out = []
    for raw in re.split(r"[\s\-–—/]+", text):
        if not raw:
            continue
        core = raw.strip(".,!?;:\"«»()[]{}…")
        if core and _NUM_RE.match(core):
            words = _num_to_words(core, lang)
            if words:
                out.extend(t for t in re.split(r"[\s\-,]+", _strip_punct_keep_space(words)) if t)
                continue
        tok = _strip_punct(raw)
        if tok:
            out.append(tok)
    return out


def _strip_punct_keep_space(s):
    return "".join(ch if ch.isspace() or ch == "-" else ch for ch in _strip_punct(s.lower()))


def assign_words_to_blocks(words, blocks):
    """slice_words_to_blocks'un ürettiği blokların hangi kelimelerden oluştuğunu
    bulur. Bloklar ardışık kelime gruplarıdır; her bloğun metni kelimelerinin
    "".join(...).strip()'ine eşittir. Dönüş: her kelime için blok indeksi."""
    owner = [0] * len(words)
    wi = 0
    for bi, (_s, _e, text) in enumerate(blocks):
        acc = []
        start_wi = wi
        while wi < len(words):
            acc.append(words[wi]["word"])
            owner[wi] = bi
            wi += 1
            if "".join(acc).strip() == text:
                break
        else:
            # Beklenmedik: metin eşleşmedi. Kalanları son bloğa ver, güvenli çık.
            for k in range(start_wi, len(words)):
                owner[k] = bi
            return owner
    for k in range(wi, len(words)):
        owner[k] = len(blocks) - 1 if blocks else 0
    return owner


def _windows(wwords):
    """Whisper kelime dizisini konuşma aralarından ≤ WINDOW sn pencerelere böl.
    Dönüş: pencere sınır zamanları listesi [t0, t1, ..., +inf]."""
    if not wwords:
        return [float("-inf"), float("inf")]
    cuts = [float("-inf")]
    win_start = wwords[0]["start"]
    best = None  # (gap, cut_time)
    for i in range(1, len(wwords)):
        gap = wwords[i]["start"] - wwords[i - 1]["end"]
        mid = (wwords[i]["start"] + wwords[i - 1]["end"]) / 2.0
        if gap >= WINDOW_GAP and (best is None or gap >= best[0] * 0.8):
            best = (gap, mid)
        if wwords[i]["end"] - win_start > WINDOW and best is not None:
            cuts.append(best[1])
            win_start = best[1]
            best = None
    cuts.append(float("inf"))
    return cuts


def align(wwords, pwords, blocks, lang="en"):
    """Satırları üretir.

    wwords / pwords: [{"word", "start", "end"}, ...] (ortak sözleşme)
    blocks: slice_words_to_blocks çıktısı [[start, end, text], ...]
    Dönüş: (lines, stats)
      lines: [{"id", "start", "end", "whisper", "parakeet", "disagreement"}]
    """
    owner = assign_words_to_blocks(wwords, blocks)
    n_blocks = len(blocks)
    p_assigned = [[] for _ in range(n_blocks)]   # her satır için Parakeet kelime indeksleri
    orphans = 0

    if pwords and n_blocks:
        cuts = _windows(wwords)
        # Whisper ve Parakeet kelimelerini pencerelere dağıt
        def bucket(words, use_mid):
            out = [[] for _ in range(len(cuts) - 1)]
            k = 0
            for i, w in enumerate(words):
                t = (w["start"] + w["end"]) / 2.0 if use_mid else w["start"]
                while k < len(cuts) - 2 and t >= cuts[k + 1]:
                    k += 1
                out[k].append(i)
            return out

        wb = bucket(wwords, True)
        pb = bucket(pwords, True)

        for wi_list, pi_list in zip(wb, pb):
            if not pi_list:
                continue
            # normalize belirteçler + belirteç→kelime eşlemesi
            wt, wt_owner = [], []
            for wi in wi_list:
                for t in normalize_tokens(wwords[wi]["word"], lang):
                    wt.append(t)
                    wt_owner.append(wi)
            pt, pt_owner = [], []
            for pi in pi_list:
                for t in normalize_tokens(pwords[pi]["word"], lang):
                    pt.append(t)
                    pt_owner.append(pi)

            placed = {}
            if wt and pt:
                sm = difflib.SequenceMatcher(a=wt, b=pt, autojunk=False)
                for tag, a0, a1, b0, b1 in sm.get_opcodes():
                    if tag in ("equal", "replace") and a1 > a0:
                        for j in range(b0, b1):
                            # replace'te orana göre en yakın Whisper belirteci
                            a = a0 + min(a1 - a0 - 1, (j - b0) * (a1 - a0) // max(1, b1 - b0))
                            placed.setdefault(pt_owner[j], owner[wt_owner[a]])
            # insert ya da belirteçsiz kalanlar: zamana göre en yakın satır
            for pi in pi_list:
                if pi in placed:
                    continue
                bi, dist = _nearest_block(pwords[pi], blocks, wi_list, owner)
                if bi is None or dist > ORPHAN_MAX:
                    orphans += 1
                    continue
                placed[pi] = bi
            for pi, bi in placed.items():
                p_assigned[bi].append(pi)

    lines = []
    n_dis = 0
    for bi, (s, e, text) in enumerate(blocks):
        pis = sorted(set(p_assigned[bi]))
        ptext = " ".join(pwords[i]["word"].strip() for i in pis).strip()
        dis = False
        if pwords:
            dis = normalize_tokens(text, lang) != normalize_tokens(ptext, lang)
            n_dis += dis
        lines.append({"id": bi + 1, "start": round(s, 3), "end": round(e, 3),
                      "whisper": text, "parakeet": ptext if pwords else None,
                      "disagreement": bool(dis)})
    stats = {"lines": len(lines), "disagreements": n_dis, "orphans": orphans}
    return lines, stats


def _nearest_block(pw, blocks, wi_list, owner):
    """Parakeet kelimesine zamanca en yakın satır (pencere içindeki adaylardan)."""
    cand = sorted({owner[wi] for wi in wi_list})
    if not cand:
        cand = range(len(blocks))
    mid = (pw["start"] + pw["end"]) / 2.0
    best, bd = None, None
    for bi in cand:
        s, e = blocks[bi][0], blocks[bi][1]
        d = 0.0 if s <= mid <= e else min(abs(mid - s), abs(mid - e))
        if bd is None or d < bd:
            best, bd = bi, d
    return best, (bd if bd is not None else float("inf"))
