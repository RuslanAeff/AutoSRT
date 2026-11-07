# -*- coding: utf-8 -*-
"""Pencere ↔ ekran uyumu.

DEĞİŞMEZ KURAL (tests/test_screenfit.py ile kilitli):
    Hiçbir AutoSRT penceresi, açıldığı monitörün ÇALIŞMA ALANINI (görev çubuğu
    hariç) aşmaz ve o alanın ortasında açılır. Boyutlar "istenen" değerlerdir;
    ekran küçükse otomatik küçülür — hiçbir zaman ekranı taşırmaz.

Neden gerekli: CustomTkinter genişlik/yüksekliği Windows ölçeklemesiyle (%125,
%150…) ÇARPAR ama konumu çarpmaz. 1920×1080 + %125'te "1100x820" penceresi
1375×1025 fiziksel piksel olur, görev çubuğu hariç 1020 piksele sığmaz.

Bu modül Tkinter'a bağımlı değildir: hesap saf fonksiyondur (fit_geometry),
Windows API'leri yalnızca ölçüm için ve tamamı try/except içinde kullanılır.
"""
import math
import sys
from dataclasses import dataclass

MARGIN = 0.94          # çalışma alanının en fazla bu oranı kullanılır (nefes payı)


@dataclass(frozen=True)
class Fit:
    width: int         # mantıksal (CTk birimi — ölçekleme CTk tarafından uygulanır)
    height: int
    x: int             # fiziksel piksel (CTk konumu ölçeklemez)
    y: int
    min_width: int     # mantıksal
    min_height: int


def frame_overhead(scale):
    """Pencere çerçevesi + başlık çubuğu (fiziksel px), bilinçli olarak cömert tahmin.
    Gösterimden sonra gerçek ölçümle ayrıca doğrulanır (bkz. overflow_fix)."""
    s = scale if scale and scale > 0 else 1.0
    return int(math.ceil(16 * s)), int(math.ceil(40 * s))


def fit_geometry(work, scale, desired, minimum, margin=MARGIN):
    """work=(sol, üst, sağ, alt) fiziksel px; scale=CTk pencere ölçeği;
    desired/minimum=(genişlik, yükseklik) mantıksal. Dönüş: Fit.

    Garanti: x >= sol, y >= üst, x + dış_genişlik <= sağ, y + dış_yükseklik <= alt.
    """
    left, top, right, bottom = (int(v) for v in work)
    s = scale if scale and scale > 0 else 1.0
    fw, fh = frame_overhead(s)
    area_w = max(1, right - left)
    area_h = max(1, bottom - top)

    max_w = max(1, int(((area_w - fw) * margin) / s))
    max_h = max(1, int(((area_h - fh) * margin) / s))
    w = max(1, min(int(desired[0]), max_w))
    h = max(1, min(int(desired[1]), max_h))
    min_w = max(1, min(int(minimum[0]), w))
    min_h = max(1, min(int(minimum[1]), h))

    outer_w = int(math.ceil(w * s)) + fw
    outer_h = int(math.ceil(h * s)) + fh
    x = left + max(0, (area_w - outer_w) // 2)
    y = top + max(0, (area_h - outer_h) // 2)
    return Fit(w, h, x, y, min_w, min_h)


def overflow_fix(rect, work):
    """Gösterilmiş pencerenin GERÇEK görünür dış dikdörtgeni (fiziksel px) çalışma
    alanından taşıyorsa: (dx, dy, shrink_w, shrink_h) fiziksel px döndürür.
    Önce kaydırır; kaydırma yetmezse küçültme miktarını verir. Taşma yoksa (0,0,0,0)."""
    l, t, r, b = rect
    wl, wt, wr, wb = work
    dx = dy = shrink_w = shrink_h = 0
    width, height = r - l, b - t
    # genişlik
    if width > wr - wl:
        shrink_w = width - (wr - wl)
        width = wr - wl
    if l + dx < wl:
        dx = wl - l
    if l + dx + width > wr:
        dx = wr - width - l
    # yükseklik
    if height > wb - wt:
        shrink_h = height - (wb - wt)
        height = wb - wt
    if t + dy < wt:
        dy = wt - t
    if t + dy + height > wb:
        dy = wb - height - t
    return dx, dy, shrink_w, shrink_h


# ------------------------------------------------------------- Windows ölçümleri
def monitor_work_area(point=None):
    """Noktanın (yoksa fare imlecinin) bulunduğu monitörün çalışma alanı
    (görev çubuğu hariç), fiziksel px: (sol, üst, sağ, alt). Ölçülemezse None."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        pt = wintypes.POINT()
        if point is None:
            user32.GetCursorPos(ctypes.byref(pt))
        else:
            pt.x, pt.y = int(point[0]), int(point[1])

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        user32.MonitorFromPoint.restype = wintypes.HANDLE
        hmon = user32.MonitorFromPoint(pt, 2)          # MONITOR_DEFAULTTONEAREST
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if hmon and user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            rc = mi.rcWork
            if rc.right > rc.left and rc.bottom > rc.top:
                return rc.left, rc.top, rc.right, rc.bottom
    except Exception:
        pass
    return None


def visible_window_rect(hwnd):
    """Pencerenin GÖRÜNÜR dış dikdörtgeni (Windows 10/11'in görünmez yeniden
    boyutlandırma kenarları hariç), fiziksel px. Ölçülemezse None."""
    if sys.platform != "win32" or not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect), ctypes.sizeof(rect))
        if hr != 0:
            if not ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
                return None
        if rect.right > rect.left and rect.bottom > rect.top:
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        pass
    return None
