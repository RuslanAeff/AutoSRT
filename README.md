<p align="center">
  <img src="icon.png" alt="AutoSRT" width="128" height="128">
</p>

<h1 align="center">AutoSRT</h1>

<p align="center">
  <b>Yapay Zeka Destekli Altyazı Motoru</b><br>
  <i>AI-Powered Subtitle Engine</i> — Generate precise, micro-segmented <code>.srt</code> subtitle files from any video or audio.<br>
  Hybrid engine: <b>faster-whisper</b> + <b>NVIDIA Parakeet</b> locally on your GPU, with <b>Gemini</b> as the judge → corrected source subtitles <b>and</b> a Turkish translation.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.7-6366f1" alt="Version">
  <img src="https://img.shields.io/badge/Parakeet-TDT%200.6B%20v3-76b900?logo=nvidia&logoColor=white" alt="Parakeet">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/faster--whisper-1.2%2B-green" alt="faster-whisper">
  <img src="https://img.shields.io/badge/Gemini-3.5%20Transcribe-4285F4?logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/CUDA-12-76b900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-purple" alt="License">
</p>

---

## Özellikler / Features

| | TR | EN |
|---|---|---|
| 🔀 | **Dört motor modu**: Sadece Whisper · Whisper + Parakeet · **Whisper + Parakeet + Gemini hakem** (varsayılan) · Gemini Transcribe | **Four engine modes**: Whisper only · Whisper + Parakeet · **Whisper + Parakeet + Gemini judge** (default) · Gemini Transcribe |
| ⚖️ | **Gemini hakem**: sesi dinler, iki transkripti karşılaştırır → düzeltilmiş kaynak SRT + **Türkçe SRT** | **Gemini judge**: listens, compares both transcripts → corrected source SRT + **Turkish SRT** |
| ⏱️ | Zaman damgaları **her zaman Whisper'dan** — senkron kaymaz | Timestamps **always from Whisper** — no sync drift |
| ↻ | **Kaldığı yerden devam** (checkpoint): çökme/limit sonrası tamamlanan aşamalar atlanır | **Resume** (checkpoint): finished stages are skipped after a crash/limit |
| 🧑‍💻 | **Manuel mod**: API yerine Gemini web sohbeti — sesi sürükle, promptu kopyala, cevabı yapıştır | **Manual mode**: Gemini web chat instead of the API — drag audio, copy prompt, paste answer |
| 🙈 | Şüpheli satırları renkle işaretleme — spoiler yememek için kapatılabilir | Doubtful-line highlighting — can be turned off to avoid spoilers |
| 📦 | **Toplu işleme** — birden çok dosyayı sıraya al, model bir kez yüklenir | **Batch processing** — queue multiple files, model loaded once |
| ↕️ | Kuyrukta sıra değiştirme (▲ / ▼) ve tek tek kaldırma | Reorder queue (▲ / ▼) and remove items |
| 🧾 | **Çoklu çıktı biçimi**: SRT · VTT · TXT · JSON | **Multi-format output**: SRT · VTT · TXT · JSON |
| 🌍 | **Kaynak dil seçimi** (otomatik algıla veya sabitle) | **Source language** picker (auto-detect or fixed) |
| 📁 | Özel çıktı klasörü (yoksa kaynağın yanına) | Custom output folder (defaults next to source) |
| 🗣️ | **Terim ipucu** (`initial_prompt`) — isim/jargon yazımı | **Vocabulary hint** (`initial_prompt`) — names/jargon spelling |
| ✏️ | **Yerleşik düzenleyici** — kaydetmeden önce metni/zamanı düzelt | **Built-in editor** — fix text/timing before saving |
| 🎚️ | Okunabilirlik: karakter sınırı, min. süre, cümle-sonu bölme | Readability: char limit, min duration, sentence-end split |
| 🎨 | Sade premium arayüz · açık/koyu mod · 4 vurgu rengi | Premium UI · light/dark mode · 4 accent themes |
| 🌐 | Canlı TR / EN arayüz dili | Live TR / EN interface language |
| 🔔 | İşlem bitince Windows bildirimi (tıkla → klasörü aç) | Windows toast on completion (click → open folder) |
| 🖱️ | Sürükle & bırak (çoklu dosya) | Drag & drop (multiple files) |
| 📊 | Gerçek zamanlı ilerleme + ETA + canlı istatistik | Real-time progress + ETA + live stats |
| ⏹ | Dilediğin an durdurabilme | Stop anytime mid-run |
| 🔇 | CMD/konsol penceresi açılmaz (pythonw) | No CMD/console window (pythonw) |
| 🧠 | Model seçici (large-v3 → tiny), bellekte önbellekleme + VRAM tahliyesi | Model picker (large-v3 → tiny), cached + VRAM eviction |
| ⚙️ | Çekirdek mikro-dilimleme mantığı birebir korundu | Core micro-slicing logic preserved exactly |

---

## Ekran Görüntüsü / Screenshot

> _(Uygulama ilk açıldığında — indigo tema, TR dil)_

---

## Kurulum / Installation

### Gereksinimler / Requirements
- **Windows 10/11** (birincil hedef) — macOS'ta Whisper + Gemini çalışır, Parakeet CPU'da çalışır
- **Python 3.11+** (3.14'te de test edildi ve çalıştı)
- **NVIDIA GPU** + sürücü (CUDA 12 uyumlu; 6 GB VRAM yeterli — iki model asla aynı anda yüklenmez)
- GPU yoksa Whisper CPU/int8'e, Parakeet CPU'ya otomatik geçer
- Sistemde **ffmpeg gerekmez** (ses PyAV ile çıkarılır)

### Adımlar (Windows, PowerShell) / Steps

```powershell
# 1. Repoyu klonla
git clone https://github.com/RuslanAeff/AutoSRT.git
cd AutoSRT

# 2. (Önerilir) Sanal ortam kur
python -m venv .venv
.\.venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. GPU onnxruntime (Parakeet için) — ÖNEMLİ, bu sırayla:
#    faster-whisper CPU "onnxruntime"u da kurar; ikisi aynı klasöre yazdığından
#    CPU sürümünü kaldırıp GPU sürümünü temiz kurun.
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.26.0

# 5. Gemini anahtarı (hakem modu için)
copy .env.example .env
notepad .env        # GEMINI_API_KEY=... satırını doldurun
```

> **Notlar**
> - İlk çalıştırmada Whisper modeli (`large-v3` ~3 GB) ve Parakeet modeli (~2.5 GB) otomatik indirilir.
> - `onnxruntime-gpu` **1.26.x**'te kalın: 1.27+ CUDA 13 ile derlenir ve NVIDIA sürücüsü ≥ 580 ister.
>   Sürücünüz yeniyse (≥ 580) güncel sürüm de çalışır.
> - Parakeet GPU'yu kullanamazsa (günlükte "CPU'da çalışıyor" yazar) yine çalışır — RTX 3060'ta
>   GPU ~100×, CPU ~20× gerçek zaman hızındadır.
> - `.env` `.gitignore`'dadır; anahtar hiçbir yerde loglanmaz ve GitHub'a gitmez.
> - `pip install -r requirements.txt`'i ileride yeniden çalıştırırsanız 4. adımı tekrarlayın.

---

## Çalıştırma / Usage

```bash
# Konsolsuz (önerilen)
pythonw main.py

# Veya proje klasöründeki başlatıcıya çift tıkla:
AutoSRT.vbs
```

1. Bir veya **birden çok** dosyayı sürükleyip bırakın **veya** "Dosya Seç" ile seçin → kuyruğa eklenir (▲/▼ ile sırala, ✕ ile kaldır).
2. (İsteğe bağlı) Sol panelden model, **kaynak dil**, **çıktı biçimi/klasörü**, **terim ipucu**, tema ve okunabilirlik ayarlarını belirleyin.
3. **"Altyazı Oluştur"** butonuna tıklayın (birden çok dosyada "Altyazı Oluştur (N)").
4. İlerleme çubuğu, ETA ve Günlük panelinden takip edin.
5. "Kaydetmeden önce düzenle" açıksa her dosya için düzenleyici açılır; değilse seçili biçimler doğrudan kaydedilir. Bitince Windows bildirimi gelir.

### Motor seçimi / Engine selection

Ana ekrandaki **MOTOR** satırından dört moddan biri seçilir:

| Mod | Ne yapar | Çıktı | İnternet |
|---|---|---|---|
| **Sadece Whisper** | v1.6'daki akışın birebir aynısı | `video.srt` | Hayır |
| **Whisper + Parakeet** | İki yerel model; farklı duydukları satırlar editörde işaretlenir | `video.srt` | Hayır |
| **Whisper + Parakeet + Gemini** *(varsayılan)* | Gemini sesi dinleyip hakemlik yapar, metni düzeltir ve Türkçeye çevirir | `video.<dil>.srt` + `video.tr.srt` | Evet |
| **Gemini Transcribe** | v1.6 bulut motoru (tüm dosya Gemini 3.5 Transcribe'a gider) | `video.srt` | Evet |

### Hibrit motor nasıl çalışır? / How the hybrid engine works

```
video ─► 16 kHz mono WAV (bir kez)
      ─► Whisper large-v3 (kelime zaman damgası) ─► JSON ─► modeli VRAM'den boşalt
      ─► Parakeet TDT 0.6B v3 (ayrı süreç, GPU)  ─► JSON ─► süreç biter, VRAM serbest
      ─► hizalama (zaman + edit-distance) ─► satır başına anlaşmazlık bayrağı
      ─► Gemini parça 1..N (ses + iki transkript JSON) ─► şema doğrulaması
      ─► kaynak SRT + Türkçe SRT  (zaman damgaları HER ZAMAN Whisper'dan)
```

- **Parakeet dilleri:** model kartına göre 25 Avrupa dili — İngilizce, Rusça, Fransızca,
  İspanyolca, Almanca dahil. **Türkçe yok.** Desteklenmeyen bir dil seçilirse (ya da Whisper
  öyle algılarsa) Parakeet adımı atlanır, aşama zincirinde "Parakeet atlandı" görünür ve
  Whisper + Gemini ile devam edilir.
- **Anlaşmazlık sayılmayanlar:** büyük/küçük harf, noktalama, kesme işareti, tire,
  rakam ↔ yazı (`20` = `twenty`, `num2words` ile), Rusça `ё` ↔ `е`.
- **Gemini parçaları:** varsayılan 10 dk (ayardan 5–25 dk). Bölme her zaman iki satır
  arasındaki en uzun sessizlikten yapılır; kelime ortadan bölünmez. Önceki parçanın son
  4 satırı yalnızca **bağlam** olarak gönderilir, yeniden düzeltilmez.
- **Doğrulama:** her cevapta tüm id'ler, sıra, boş metin ve satır sayısı kontrol edilir.
  Hatalı parça **bir kez** otomatik tekrar denenir; yine eksik kalan satırlarda Whisper metni
  kullanılır ve editörde "Gemini doğrulayamadı" diye işaretlenir.
- **Limitler:** 429'da (dakikalık sınır) Gemini'nin önerdiği bekleme süresi + üstel geri
  çekilme ile otomatik beklenir; geri sayım durum satırında görünür. Günlük kota, geçersiz
  anahtar ve ağ hataları için açık Türkçe mesaj verilir. **"Manuel moda geç"** düğmesi iş
  sürerken her zaman görünür.
- **Model:** varsayılan `gemini-3.8-flash` (ücretsiz katmanda; ses girdisi + yapılandırılmış
  çıktı). Kenar çubuğundaki "Gemini modeli" alanından değiştirilebilir. Güncel ücretsiz
  katman limitleri yalnızca [AI Studio](https://aistudio.google.com/) panelinde görünür.

> ⚠️ **Gizlilik:** Hakem modunda ses parçaları Gemini'ye yüklenir ve iş bitince Files
> API'den silinir. Ücretsiz katmanda Google, gönderilen içeriği ürünlerini geliştirmek
> için kullanabilir. Tamamen yerel çalışmak için "Whisper + Parakeet" modunu seçin.

### Kaldığı yerden devam / Resume (checkpoint)

Her video için `%APPDATA%\AutoSRT\jobs\<id>\` altında bir iş klasörü açılır (video
yanında değil). Ses, Whisper, Parakeet, hizalama ve **her Gemini parçasının sonucu**
diske kaydedilir. Uygulama çöker, kapanır ya da kotaya takılırsa aynı videoyu tekrar
başlatmanız yeterli: tamamlanan aşamalar atlanır. Bir ayarı değiştirirseniz yalnızca
ondan etkilenen aşamalar yeniden çalışır (ör. Whisper modeli değişirse Parakeet yeniden
çalışmaz). İş başarıyla bitince klasör silinir; geriye yalnızca altyazı dosyaları kalır.
30 günden eski yarım kalmış işler uygulama açılışında otomatik silinir.

### Manuel mod / Manual mode (Gemini web sohbeti)

API yerine gemini.google.com sohbetini kullanmak için: kenar çubuğunda **"Manuel mod"**
anahtarını açın ya da iş sürerken **"Manuel moda geç"**e basın. Sihirbaz her parça için:

1. **Ses parçası** — "Buradan sürükleyin" alanından doğrudan tarayıcıya sürükleyin
   (ya da "Klasörde göster").
2. **Prompt** — "Promptu kopyala" → aynı mesaja yapıştırıp gönderin.
3. **Cevap** — Gemini'nin cevabının tamamını yapıştırın → "Doğrula ve kaydet".

Sohbetler uzun cevapları kesebildiği için manuel parça varsayılanı **5 dk**'dır. Eksik
satır varsa hangi id'lerin eksik olduğu söylenir ve bunlar bir sonraki pakete eklenir.
Otomatik ve manuel mod **aynı checkpoint'i** kullanır — istediğiniz an "Otomatik moda dön".

### Komut satırı / CLI

```powershell
python hybrid_cli.py "bolum.mkv"                          # Whisper + Parakeet + Gemini
python hybrid_cli.py "bolum.mkv" --lang ru --chunk-min 8
python hybrid_cli.py "bolum.mkv" --mode whisper_parakeet  # tamamen yerel
```

Ctrl+C güvenlidir; aynı komut kaldığı yerden devam eder.

### Gemini kurulumu / Gemini setup

```bash
pip install google-genai
```

API anahtarını [Google AI Studio](https://aistudio.google.com/)'dan alın, sonra
**iki yoldan biriyle** verin (öncelik sırası bu şekildedir):

```bash
# 1) Ortam değişkeni (önerilen — hiçbir yere yazılmaz)
setx GEMINI_API_KEY "anahtariniz"        # Windows
export GEMINI_API_KEY="anahtariniz"      # bash
```

**2)** Ya da motoru "Gemini (bulut)" yapıp kenar çubuğundaki **GEMINI API ANAHTARI**
alanına yapıştırın. Bu anahtar `%APPDATA%\AutoSRT\secrets.json` dosyasına —
yani **proje klasörünün dışına** — kaydedilir; `settings.json`'a yazılmaz ve
depoya hiçbir zaman girmez. Anahtar yoksa motor seçilemez ve günlükte net bir
uyarı çıkar.

> ⚠️ **Gizlilik uyarısı:** Gemini modu seçili ses dosyanızı Google sunucularına
> **yükler**. İş bittiğinde (hata alınsa bile) yüklenen dosya Files API'den
> otomatik olarak silinir, geçici WAV dosyaları da temizlenir. Verinizin
> bilgisayardan çıkmasını istemiyorsanız Whisper motorunu kullanın.

> ⏱️ **Süre limiti:** Kelime zaman damgası açıkken Gemini tek istekte en fazla
> **30 dakika** ses kabul eder. AutoSRT daha uzun dosyaları otomatik olarak
> ~25 dakikalık parçalara böler, bölme noktalarını **sessizliğe** hizalar
> (kelime ortasında kesmemek için), her parçanın zaman damgalarına kendi
> ofsetini ekler ve parça sınırlarındaki çakışmaları düzeltir. Bu tamamen
> saydamdır — tek bir altyazı dosyası alırsınız.

**Ayarların karşılığı:** Kaynak dil seçimi BCP-47 koduna çevrilir (`tr` → `tr-TR`,
`az` → `az-AZ`); "Otomatik" seçiliyken dil algılama Gemini'ye bırakılır. Terim
ipucu virgülle ayrılır ve ilk 100 terim `custom_vocabulary` olarak gönderilir
(Whisper'da aynı alan `initial_prompt` olarak kullanılır).

### Motorları karşılaştırma / Comparing engines

Hangi motorun sizin içeriğinizde daha iyi olduğunu tahmin etmek yerine ölçün:

```bash
python compare_engines.py "video.mp4"
python compare_engines.py "video.mp4" --lang tr --terms "Ada,Bora"
python compare_engines.py "video.mp4" --only whisper
```

Aynı dosyayı iki motorla da işler, `video.whisper.srt` ve `video.gemini.srt`
üretir; blok sayısı, kelime sayısı, karakter sayısı, ilk/son blok zaman damgası
ve işlem süresini karşılaştırmalı tablo olarak yazar. Gemini için anahtar yoksa
o motoru atlar.

### Toplu işleme / Batch
- Birden çok dosya kuyruğa alınır ve **sırayla** işlenir; model yalnızca **bir kez** yüklenir (sonraki dosyalar hızlı).
- ▲ / ▼ ile sırayı değiştirin, ✕ ile kaldırın, "Tümünü temizle" ile sıfırlayın.

### Çıktı biçimi / Output formats
- **SRT** (standart), **VTT** (web/YouTube), **TXT** (düz transkript), **JSON** (kelime/blok zaman damgalı). Birden çoğu aynı anda seçilebilir.

### Kaynak dil & terim ipucu / Source language & vocabulary hint
- **Kaynak dil**: "Otomatik (algıla)" veya sabit dil seçin (sabitlemek hız + doğruluk kazandırır).
- **Terim ipucu**: özel isim/marka/jargon gibi terimleri virgülle yazın → `initial_prompt` olarak modele verilir, yazımları düzelir.

### Yerleşik düzenleyici / Built-in editor
- "Kaydetmeden önce düzenle" açıkken her dosyanın altyazısı düzenlenebilir SRT olarak açılır. Metni/zamanı düzeltip **Kaydet**'e basın → seçili biçimlere yazılır. Kayıttan önce doğrulama yapılır ve diske yazmadan önce zaman bütünlüğü (overlap) güvenceye alınır.

### Okunabilirlik ayarları / Readability settings
Altyazıların ekranda "şak şak" hızlı geçmesini engellemek için:
- **Karakter sınırı** (20–90, varsayılan 45): yükseltince satırlar uzar, daha az/yavaş değişir.
- **Min. süre** (0–2 sn): çok kısa bloklar, sonraki bloğa taşmadan en az bu süre ekranda kalır.
- **Cümle sonunda böl**: açıkken yalnızca `. ? !` ile böler (virgülde bölmez) → daha akıcı.

> Varsayılan değerlerde (45 / 0 / kapalı / sadece SRT) çıktı, orijinal CLI koduyla **birebir** aynıdır.

---

## Bağımlılıklar / Dependencies

```
customtkinter>=5.2.2
tkinterdnd2>=0.4.2
winotify>=1.1.0
faster-whisper>=1.0.3      # Whisper (yerel)
google-genai>=2.20.0       # Gemini hakem + Gemini Transcribe (bulut)
onnx-asr[hub]>=0.12.0      # Parakeet TDT 0.6B v3 (ONNX, NeMo gerekmez)
onnxruntime-gpu==1.26.0    # CUDA 12 derlemesi — kurulum adımı 4'e bakın
num2words>=0.5.13          # "20" == "twenty" normalizasyonu
python-dotenv>=1.0.0       # .env'den GEMINI_API_KEY
nvidia-cublas-cu12  nvidia-cudnn-cu12  nvidia-cuda-runtime-cu12
nvidia-cufft-cu12   nvidia-curand-cu12
```

`google-genai` kurulu değilse uygulama normal açılır; Gemini gerektiren modlar devre
dışı kalır. `onnx-asr` yoksa Parakeet adımı atlanır. "Sadece Whisper" modu bu
paketlerin hiçbirinden etkilenmez.

**Neden NeMo değil de onnx-asr?** NeMo, PyTorch ile birlikte birkaç GB tutuyor, Windows'u
resmî olarak desteklemiyor ve Python 3.14 için paketi yok. sherpa-onnx'te Windows GPU için
özel derleme gerekiyor. onnx-asr ise saf Python (onnxruntime + numpy), Parakeet v3'ün ONNX
dönüşümünü HuggingFace'ten kendisi indiriyor ve token zaman damgası veriyor.

---

## Teknik Mimari / Architecture

```
AutoSRT/
├── main.py            # GUI + Whisper/Gemini Transcribe motorları + dilimleme
├── autosrt_core/      # Hibrit motor çekirdeği (Tkinter'dan bağımsız)
│   ├── audio.py       #   16 kHz WAV çıkarma (PyAV), VAD pencereleri
│   ├── parakeet.py    #   Parakeet TDT v3 — ayrı süreçte, GPU (onnx-asr)
│   ├── align.py       #   normalizasyon + zaman/edit-distance hizalama
│   ├── judge.py       #   Gemini hakem: istem, şema, doğrulama, 429/kota yönetimi
│   ├── manual.py      #   manuel mod paketleri
│   ├── jobs.py        #   iş klasörü + checkpoint + bağımlılık grafiği
│   └── pipeline.py    #   aşama orkestratörü + ManualSession
├── hybrid_cli.py      # Hibrit boru hattını arayüzsüz çalıştırır
├── tests/             # python -m unittest discover -s tests
├── .env.example       # GEMINI_API_KEY şablonu (.env git'e girmez)
├── compare_engines.py # Whisper ↔ Gemini Transcribe karşılaştırma yardımcısı
├── AutoSRT.vbs        # Konsolsuz Windows başlatıcısı
├── icon.png         # Master ikon (1024×1024)
├── icon.ico         # Pencere / kısayol ikonu (çok boyutlu)
├── requirements.txt
├── LICENSE
└── README.md
```

**Thread modeli:**
- Ana GUI thread'i hiçbir zaman kilitlenmez.
- Whisper işlemi `threading.Thread` (daemon) ile arka planda çalışır; toplu işlemde tek worker dosyaları sırayla işler.
- Thread ↔ GUI iletişimi yalnız `queue.Queue` + `after(80ms)` polling ile yapılır (Tkinter thread-safety).
- Yerleşik düzenleyici de bu kanaldan açılır: worker `Event` ile bekler, GUI pencereyi açar; "Durdur" sırasında editör güvenle kapatılır (kilitlenme yok).
- `sys.stdout/stderr` günlük kuyruğuna yönlendirilir → model indirme ilerlemesi de panelde görünür.

**Motor katmanı:**
- `TranscriptionEngine.transcribe(audio_path, language, vocab_hint, progress_cb, stop_event)`
  her motor için **aynı** sözleşmeyi döndürür:
  `[{"word": str, "start": float, "end": float, "seg_end": bool}, ...]`
- `WhisperEngine` — faster-whisper; model önbelleği, VRAM tahliyesi, CPU fallback
  ve VAD ön-taraması burada.
- `GeminiEngine` — Gemini 3.5 Transcribe; `verbatim` modda kelime zaman damgası
  (`smart` mod zaman damgasıyla uyumsuzdur, kullanılmaz), 25 dk'lık parçalama,
  429/5xx için üstel geri çekilmeli 3 deneme, yüklenen dosyanın garantili silinmesi.
  Ses çıkarma **PyAV** ile yapılır (faster-whisper ile gelir) — sistemde ayrıca
  ffmpeg/ffprobe kurulu olması gerekmez.
- Dilimleme, `_sanitize_blocks` ve dosya yazıcıları iki motor için de **ortaktır**;
  motor değiştirmek altyazı biçimlendirmesini etkilemez.

**Mikro-dilimleme mantığı (varsayılan `max_chars=45`):**
- `word_timestamps=True` ile kelime düzeyinde zaman damgası alınır.
- Karakter sınırına veya (ayara göre) noktalama işaretine ulaşıldığında yeni blok açılır.
- İsteğe bağlı min-süre son-işlemiyle çok kısa bloklar uzatılır (taşma koruması ile).
- Diske yazmadan önce bloklar `_sanitize_blocks` ile sıralanır, geçersiz süreler atılır, örtüşme (overlap) kırpılır → her zaman geçerli altyazı.
- Her blok seçili biçim(ler)e yazılır; SRT/VTT zaman damgaları `HH:MM:SS,mmm` / `HH:MM:SS.mmm`.

**Bellek / VRAM:**
- Model `(model_size, device, compute)` anahtarıyla önbelleklenir. Anahtar değişince yeni model yüklenmeden önce eskisi `del + gc.collect()` ile boşaltılır (düşük VRAM'li GPU'larda OOM önlemi).

---

## GPU yoksa / CPU Fallback

Uygulama CUDA cihazı bulamazsa otomatik olarak `cpu / int8` moduna geçer — herhangi bir ayar değişikliği gerekmez.

Veya `main.py`'de sabit olarak değiştirilebilir:
```python
DEVICE       = "cpu"
COMPUTE_TYPE = "int8"
```

---

## Lisans / License

MIT License — bkz. [LICENSE](LICENSE)
