<p align="center">
  <img src="icon.png" alt="AutoSRT" width="128" height="128">
</p>

<h1 align="center">AutoSRT</h1>

<p align="center">
  <b>Yapay Zeka Destekli Altyazı Motoru</b><br>
  <i>AI-Powered Subtitle Engine</i> — Generate precise, micro-segmented <code>.srt</code> subtitle files from any video or audio.<br>
  Choose your engine: <b>faster-whisper</b> locally on your NVIDIA GPU, or <b>Google Gemini 3.5 Transcribe</b> in the cloud.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.6-6366f1" alt="Version">
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
| 🔀 | **İki motor**: Whisper (yerel, varsayılan) · Gemini 3.5 Transcribe (bulut) | **Two engines**: Whisper (local, default) · Gemini 3.5 Transcribe (cloud) |
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
- **Python 3.11+** (3.14'te de test edildi ve çalıştı)
- **NVIDIA GPU** + güncel sürücü (CUDA 12 uyumlu)
- GPU yoksa otomatik olarak CPU/int8 moduna geçer

### Adımlar / Steps

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI_ADIN/AutoSRT.git
cd AutoSRT

# 2. (Önerilir) Sanal ortam kur
python -m venv .venv
# Windows:
.\.venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

> **Not:** İlk çalıştırmada seçilen model (örn. `large-v3` ~3 GB) otomatik indirilir. İndirme ilerlemesi uygulama içindeki Günlük panelinde görünür.

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

Sol paneldeki **MOTOR** düğmesiyle iki motor arasında geçiş yapılır. Varsayılan
**Whisper (yerel)**'dir ve öyle kalır — Gemini tamamen isteğe bağlı bir eklentidir.

| | Whisper (yerel) | Gemini (bulut) |
|---|---|---|
| Nerede çalışır | Kendi GPU'nuzda | Google sunucularında |
| İnternet | Gerekmez | Gerekir |
| Ücret | Yok | API kullanımı ücretlidir |
| Gizlilik | Dosya bilgisayardan çıkmaz | **Dosya Google'a yüklenir** |
| Kurulum | Model indirilir (~3 GB) | API anahtarı |
| Sol panelde | Model seçici görünür | API anahtarı alanı görünür |

Motorlardan hangisi seçilirse seçilsin **altyazı üretimi aynıdır**: mikro-dilimleme,
karakter sınırı, min. süre, cümle-sonu bölme, düzenleyici ve tüm çıktı biçimleri
ortaktır. Motor yalnızca "kelime + zaman damgası" listesini üretir.

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
faster-whisper>=1.0.3      # Motor 1 (yerel, varsayılan)
google-genai>=2.20.0       # Motor 2 (bulut, isteğe bağlı)
nvidia-cublas-cu12
nvidia-cudnn-cu12
```

`google-genai` kurulu değilse uygulama normal açılır; yalnızca Gemini motoru
devre dışı görünür. Whisper hiçbir şekilde etkilenmez.

---

## Teknik Mimari / Architecture

```
AutoSRT/
├── main.py            # Tek dosya uygulama — GUI + motor katmanı
├── compare_engines.py # İki motoru aynı dosyada ölçüp karşılaştıran yardımcı
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
