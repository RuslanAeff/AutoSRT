# -*- coding: utf-8 -*-
"""Ses yardımcıları.

Sistemde ffmpeg kurulu olması GEREKMEZ: çözme faster-whisper'ın decode_audio'su
ile (PyAV — ffmpeg kütüphaneleri gömülü) yapılır. Video bir kez 16 kHz mono
WAV'a çevrilir; sonraki tüm aşamalar bu WAV'ı okur.
"""
import os
import wave

import numpy as np

from . import SR


def extract_wav(src_path, dst_path):
    """Videodan/sesten 16 kHz mono PCM16 WAV üretir; süreyi (sn) döndürür.
    Önce .part dosyasına yazar, sonra atomik olarak yerine taşır — yarım
    kalmış bir WAV checkpoint'te 'tamamlandı' sanılmasın diye."""
    from faster_whisper.audio import decode_audio
    audio = decode_audio(src_path, sampling_rate=SR)
    tmp = dst_path + ".part"
    write_wav(audio, tmp)
    os.replace(tmp, dst_path)
    return len(audio) / SR


def write_wav(samples, path):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    """PCM16 mono WAV → float32 [-1, 1] dizi."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("unexpected wav format: %s" % path)
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def write_slice(audio, start, end, path):
    """[start, end) saniye aralığını ayrı WAV olarak yazar."""
    a = max(0, int(start * SR))
    b = min(len(audio), int(end * SR))
    write_wav(audio[a:b], path)


def vad_chunks(audio, threshold=0.5):
    """Silero VAD konuşma parçaları (örnek indeksli dict listesi).
    faster-whisper'ın iç modülü kullanılır; değişirse boş liste döner."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        return get_speech_timestamps(audio, VadOptions(
            threshold=threshold, min_silence_duration_ms=300, speech_pad_ms=100))
    except Exception:
        return []


def speech_windows(chunks, total_sec, max_len=24.0, merge_gap=2.0):
    """VAD parçalarını Parakeet'in tek seferde işleyebileceği (≤ max_len sn)
    pencerelere birleştirir. Pencere sınırları konuşma ARALARINA denk gelir.
    max_len'den uzun tek bir konuşma parçası zorunlu olarak sabit adımlarla
    bölünür (nadir; ~24 sn kesintisiz konuşma).
    Dönüş: [(start_sec, end_sec), ...]"""
    spans = [(c["start"] / SR, c["end"] / SR) for c in chunks]
    if not spans:
        # VAD hiçbir şey bulamadıysa tüm dosyayı sabit pencerelerle tara.
        spans = [(0.0, total_sec)] if total_sec > 0 else []

    # 1) çok uzun parçaları böl
    pieces = []
    for s, e in spans:
        while e - s > max_len:
            pieces.append((s, s + max_len))
            s += max_len
        if e > s:
            pieces.append((s, e))

    # 2) yakın parçaları max_len'i aşmadan birleştir
    out = []
    for s, e in pieces:
        if out and (s - out[-1][1]) <= merge_gap and (e - out[-1][0]) <= max_len:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out
