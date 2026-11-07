# -*- coding: utf-8 -*-
"""AutoSRT hibrit motor çekirdeği (Whisper + Parakeet + Gemini hakem).

Bu paket Tkinter'a ve main.py'ye BAĞIMLI DEĞİLDİR: Whisper motoru, dilimleyici
ve günlük geri çağrıları pipeline'a dışarıdan (main.py'den) verilir. Böylece
her modül tek başına test edilebilir ve döngüsel import oluşmaz.
"""

SR = 16000  # tüm boru hattının örnekleme hızı (Whisper / Parakeet / VAD)
