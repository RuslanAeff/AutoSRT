# AGENTS.md — AutoSRT

Bu depoda çalışan her yapay zeka asistanı / geliştirici için **bozulmaması gereken
kurallar**. Bir kural bir değişiklikle çelişiyorsa kuralı değil değişikliği uyarlayın.
Her değişiklikten sonra: `python -m unittest discover -s tests`.

## 1. Pencere her zaman ekrana sığar (testle kilitli)

Kural: Hiçbir AutoSRT penceresi açıldığı monitörün **çalışma alanını** (görev çubuğu
hariç) aşmaz, ortalanmış açılır ve ana içerik pencere ne kadar küçülürse küçülsün
**kesilmez** — kaydırılır. Windows ölçeklemesi (%125, %150…) dahil.

- Pencere boyutu/konumu **yalnızca** `AutoSRTApp._place_window(win, istenen, en_küçük)`
  ile verilir; düzeltme sigortası `_keep_on_screen`. Başka hiçbir yerde
  `.geometry(...)`, `.minsize(...)`, `.maxsize(...)` çağırmayın.
  Boyut istekleri `main.py`'deki `WIN_MAIN / WIN_EDITOR / WIN_WIZARD` sabitlerindedir;
  bunlar bir **istek**tir, ekran küçükse otomatik küçülür.
- Yeni bir pencere (`CTkToplevel`) eklerken: bir `WIN_*` sabiti tanımlayıp
  `self._place_window(win, *WIN_YENI)` çağırın.
- Ana içerik `_FitScrollFrame` içinde yaşar (`_build_main`). Onu düz bir `CTkFrame` ile
  değiştirmeyin, içeriği dışına taşımayın. Yeni widget eklemek serbesttir; kap,
  içeriğin yüksekliğini kendisi izler.
- Ana pencere gizli kurulur (`withdraw`) ve `_reveal_main` ile gösterilir; bu sırayı
  bozmayın (yoksa pencere yanlış yerde belirip zıplar).
- Hesap: `autosrt_core/screenfit.py` (saf, Tkinter'sız). Kilit: `tests/test_screenfit.py`
  (tüm monitör/ölçek kombinasyonları + "sabit geometry yasak" AST denetimi + gerçek
  pencere testi). Bu test kırılıyorsa düzeltilecek olan arayüz kodudur.

## 2. "Sadece Whisper" modu birebir korunur

`_batch_worker`, `_transcribe_one`, `WhisperEngine`, `slice_words_to_blocks`,
`format_timestamp`, yazıcılar: varsayılan ayarlarda çıktı eski sürümle **aynı** kalmalı.
Hibrit modlar ayrı worker kullanır (`_hybrid_batch_worker`).

## 3. Hibrit motor

- Zaman damgaları **her zaman Whisper'dan** gelir; Gemini yalnız metni düzeltir/çevirir.
- Whisper ve Parakeet **asla aynı anda VRAM'de** olmaz (Parakeet ayrı süreçte çalışır).
- Gemini sonuçları satır id'sine göre checkpoint'lenir (`autosrt_core/jobs.py`).

## 4. Güvenlik

- API anahtarı yalnız `GEMINI_API_KEY` ortam değişkeni, `.env` (git'e girmez) ya da
  `%APPDATA%\AutoSRT\secrets.json`'da durur; `settings.json`'a, loglara, koda yazılmaz.
- `.env.example` yalnızca **boş** şablondur — içine gerçek anahtar yazmayın.

## 5. Arayüz metinleri

Her yeni metin `L["tr"]` **ve** `L["en"]` tablosuna eklenir; Türkçe karakterler doğru
Unicode olmalıdır (ç ş ı İ ğ ü ö).
