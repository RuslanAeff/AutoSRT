# AutoSRT — Yapay Zeka Destekli Altyazı Motoru

> **AI-Powered Subtitle Engine** — Generate precise, micro-segmented `.srt` subtitle files from any video or audio using a local Whisper model on your NVIDIA GPU.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![faster-whisper](https://img.shields.io/badge/faster--whisper-1.2%2B-green)
![CUDA](https://img.shields.io/badge/CUDA-12-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## Özellikler / Features

| | TR | EN |
|---|---|---|
| 🎨 | Kenar çubuklu, sade premium arayüz | Sidebar-based premium UI |
| 🌗 | Açık / Koyu mod geçişi | Light / Dark mode toggle |
| 🌐 | Canlı TR / EN dil değiştirme | Live language toggle (TR / EN) |
| 🎨 | 4 vurgu rengi — tüm arayüz birlikte değişir | 4 accent themes applied system-wide |
| 🎚️ | Okunabilirlik ayarları: karakter sınırı, min. süre, cümle-sonu bölme | Readability controls: char limit, min duration, sentence-end split |
| 🔔 | İşlem bitince Windows bildirimi (tıkla → klasörü aç) | Windows toast on completion (click → open folder) |
| 🖱️ | Sürükle & bırak + parlama efekti | Drag & drop with glow-on-hover |
| 📊 | Gerçek zamanlı ilerleme + ETA + canlı istatistik | Real-time progress bar + ETA + live stats |
| ⏹ | Dilediğin an durdurabilme | Stop button mid-transcription |
| 🔇 | CMD/konsol penceresi açılmaz | No CMD/console window (pythonw) |
| 🧠 | Model seçici (large-v3 → tiny), önbellekleme | Model picker; model cached in memory |
| 💾 | SRT videonun yanına otomatik kaydedilir | SRT auto-saved next to the video file |
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

1. Videoyu sürükleyip bırakın **veya** "Video Seç" ile seçin.
2. (İsteğe bağlı) Sol panelden model, dil, tema ve **okunabilirlik ayarlarını** belirleyin.
3. **"Altyazı Oluştur"** butonuna tıklayın.
4. İlerleme çubuğu, ETA ve Günlük panelinden takip edin.
5. Bitince Windows bildirimi gelir; `.srt` dosyası videonun yanına otomatik kaydedilir.

### Okunabilirlik ayarları / Readability settings
Altyazıların ekranda "şak şak" hızlı geçmesini engellemek için:
- **Karakter sınırı** (20–90, varsayılan 45): yükseltince satırlar uzar, daha az/yavaş değişir.
- **Min. süre** (0–2 sn): çok kısa bloklar, sonraki bloğa taşmadan en az bu süre ekranda kalır.
- **Cümle sonunda böl**: açıkken yalnızca `. ? !` ile böler (virgülde bölmez) → daha akıcı.

> Varsayılan değerlerde (45 / 0 / kapalı) çıktı, orijinal CLI koduyla **birebir** aynıdır.

---

## Bağımlılıklar / Dependencies

```
customtkinter>=5.2.2
tkinterdnd2>=0.4.2
winotify>=1.1.0
faster-whisper>=1.0.3
nvidia-cublas-cu12
nvidia-cudnn-cu12
```

---

## Teknik Mimari / Architecture

```
AutoSRT/
├── main.py          # Tek dosya uygulama — GUI + transcribe motoru
├── AutoSRT.vbs      # Konsolsuz Windows başlatıcısı
├── requirements.txt
└── README.md
```

**Thread modeli:**
- Ana GUI thread'i hiçbir zaman kilitlenmez.
- Whisper işlemi `threading.Thread` ile arka planda çalışır.
- Thread ↔ GUI iletişimi `queue.Queue` + `after()` polling ile yapılır (Tkinter thread-safety).
- `sys.stdout/stderr` günlük kuyruğuna yönlendirilir → model indirme ilerlemesi de panelde görünür.

**Mikro-dilimleme mantığı (varsayılan `max_chars=45`):**
- `word_timestamps=True` ile kelime düzeyinde zaman damgası alınır.
- Karakter sınırına veya (ayara göre) noktalama işaretine ulaşıldığında yeni blok açılır.
- İsteğe bağlı min-süre son-işlemiyle çok kısa bloklar uzatılır (taşma koruması ile).
- Her blok `format_timestamp()` ile `HH:MM:SS,mmm` formatına dönüştürülür.

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
