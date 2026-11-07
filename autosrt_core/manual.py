# -*- coding: utf-8 -*-
"""Manuel mod: API yerine Gemini web sohbeti.

Her parça için iş klasörünün manual\\ alt klasörüne:
    part_XX.wav         sohbete sürüklenecek ses parçası
    part_XX_prompt.txt  panoya kopyalanacak TEK prompt (talimat + transkript JSON)

Yapıştırılan cevap otomatik moddakiyle AYNI doğrulamadan geçer. Eksik id'ler
kaydedilmez; parçalar her seferinde KALAN satırlardan planlandığı için eksikler
bir sonraki pakete kendiliğinden girer. Otomatik ve manuel mod aynı
results.json'u kullanır → istenen an geçiş yapılabilir.
"""
import json
import os

from . import judge as J

OUTPUT_FORMAT = """Return ONLY a JSON object in a single ```json code block, with this exact shape
(one entry per input line, same order, no other text before or after):

```json
{"lines": [
  {"id": 12, "source_text": "...", "turkish_text": "...", "confidence": 0.9, "note": ""}
]}
```"""


def build_prompt(items, context, src_lang):
    """Sohbete yapıştırılacak tek metin."""
    return (
        J.system_instruction(src_lang)
        + "\nThe audio clip for these lines is attached to this message.\n\n"
        + OUTPUT_FORMAT
        + "\n\nINPUT:\n```json\n"
        + J.user_payload(items, context)
        + "\n```\n"
    )


def write_package(job, part_no, audio, chunk, context, src_lang, clip_start, clip_end):
    """Paketi diske yazar; (wav_yolu, prompt_metni) döndürür."""
    from .audio import write_slice
    d = job.manual_dir
    wav = os.path.join(d, "part_%02d.wav" % part_no)
    write_slice(audio, clip_start, clip_end, wav)
    items = J.build_items(chunk, clip_start)
    prompt = build_prompt(items, context, src_lang)
    with open(os.path.join(d, "part_%02d_prompt.txt" % part_no), "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(os.path.join(d, "part_%02d_ids.json" % part_no), "w", encoding="utf-8") as f:
        json.dump([ln["id"] for ln in chunk], f)
    return wav, prompt
