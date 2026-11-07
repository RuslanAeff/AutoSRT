# -*- coding: utf-8 -*-
"""İş klasörü + checkpoint.

Her video için uygulamanın veri dizininde (video yanında DEĞİL) bir klasör:

    %APPDATA%\\AutoSRT\\jobs\\<job_id>\\
        manifest.json     aşama durumları + parmak izleri
        audio.wav         16 kHz mono (bir kez çıkarılır)
        whisper.json      kelime listesi + algılanan dil
        parakeet.json     kelime listesi (ya da atlandı bilgisi)
        lines.json        hizalanmış satırlar (id/start/end/whisper/parakeet/disagreement)
        results.json      Gemini sonuçları — SATIR id'sine göre (parçaya göre değil)
        manual\\           manuel paketler (ses parçası + prompt)

Gemini sonuçları satır id'sine göre saklanır; parçalar her seferinde KALAN
satırlardan yeniden planlanır. Bu sayede otomatik mod (10 dk parça) ile manuel
mod (5 dk parça) aynı checkpoint'i paylaşır ve iş ortasında geçiş yapılabilir.

Her aşamanın bir parmak izi (fingerprint) vardır: o aşamanın sonucunu etkileyen
ayarlar değişirse aşama ve ona BAĞLI aşamalar geçersiz sayılır (Job.DEPENDENTS).
"""
import hashlib
import json
import os
import shutil
import time

STAGES = ("audio", "whisper", "parakeet", "lines", "gemini")


def jobs_root():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "AutoSRT", "jobs")


def job_id_for(video_path):
    """Yol + boyut + değiştirilme zamanı → kararlı kimlik. Aynı video tekrar
    açılınca aynı klasör bulunur; dosya değişmişse yeni iş başlar."""
    p = os.path.normcase(os.path.abspath(video_path))
    try:
        st = os.stat(p)
        sig = "%s|%d|%d" % (p, st.st_size, int(st.st_mtime))
    except OSError:
        sig = p
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


def fingerprint(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False)
                        .encode("utf-8")).hexdigest()[:12]


def save_json(path, data):
    """Atomik yazım: önce .tmp, sonra os.replace (çökmede yarım dosya kalmaz)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


class Job:
    def __init__(self, video_path, root=None):
        self.video = os.path.abspath(video_path)
        self.id = job_id_for(video_path)
        self.dir = os.path.join(root or jobs_root(), self.id)
        os.makedirs(self.dir, exist_ok=True)
        self._mpath = os.path.join(self.dir, "manifest.json")
        self.manifest = load_json(self._mpath, None) or {
            "video": self.video, "created": time.time(), "stages": {}}

    # --- yollar ---
    def path(self, name):
        return os.path.join(self.dir, name)

    @property
    def manual_dir(self):
        d = os.path.join(self.dir, "manual")
        os.makedirs(d, exist_ok=True)
        return d

    # --- manifest ---
    def _save(self):
        save_json(self._mpath, self.manifest)

    def is_done(self, stage, fp):
        """Aşama tamamlanmış VE parmak izi aynı VE dosyası diskte mi?"""
        st = self.manifest["stages"].get(stage)
        if not st or st.get("fp") != fp:
            return False
        f = st.get("file")
        return f is None or os.path.exists(self.path(f))

    def mark_done(self, stage, fp, file=None, **extra):
        self.manifest["stages"][stage] = dict(fp=fp, file=file, t=time.time(), **extra)
        self._save()

    def stage_info(self, stage):
        return self.manifest["stages"].get(stage) or {}

    # Bağımlılık grafiği: bir aşama değişince hangi aşamalar geçersizleşir.
    # Parakeet yalnız sese bağlıdır — Whisper ayarı değişince yeniden ÇALIŞMAZ.
    DEPENDENTS = {
        "audio": ("audio", "whisper", "parakeet", "lines", "gemini"),
        "whisper": ("whisper", "lines", "gemini"),
        "parakeet": ("parakeet", "lines", "gemini"),
        "lines": ("lines", "gemini"),
        "gemini": ("gemini",),
    }

    def invalidate(self, stage):
        """stage'i ve ona bağlı aşamaları geçersiz kıl (dosyalarıyla birlikte).
        Satırlar değişirse Gemini sonuçları (id'ye bağlı) ve manuel paketler de gider."""
        for s in self.DEPENDENTS[stage]:
            st = self.manifest["stages"].pop(s, None)
            if st and st.get("file"):
                try:
                    os.remove(self.path(st["file"]))
                except OSError:
                    pass
        if "gemini" in self.DEPENDENTS[stage]:
            try:
                os.remove(self.path("results.json"))
            except OSError:
                pass
            shutil.rmtree(os.path.join(self.dir, "manual"), ignore_errors=True)
        self._save()

    # --- Gemini sonuçları (satır id'sine göre) ---
    def load_results(self):
        return {int(k): v for k, v in (load_json(self.path("results.json"), {}) or {}).items()}

    def save_results(self, results):
        save_json(self.path("results.json"), {str(k): v for k, v in results.items()})

    def merge_results(self, new):
        res = self.load_results()
        res.update(new)
        self.save_results(res)
        return res

    # --- temizlik ---
    def cleanup(self):
        """İş başarıyla bitti: tüm geçici dosyaları sil (geriye yalnız SRT'ler kalır)."""
        shutil.rmtree(self.dir, ignore_errors=True)


def cleanup_stale(max_age_days=30, root=None):
    """Çok eski, yarım kalmış iş klasörlerini temizler (ortalık dağılmasın)."""
    root = root or jobs_root()
    if not os.path.isdir(root):
        return 0
    n, cutoff = 0, time.time() - max_age_days * 86400
    for name in os.listdir(root):
        d = os.path.join(root, name)
        try:
            if os.path.isdir(d) and os.path.getmtime(d) < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            pass
    return n
