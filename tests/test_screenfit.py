# -*- coding: utf-8 -*-
"""Pencere ↔ ekran DEĞİŞMEZ KURALI'nın kilidi (bkz. AGENTS.md).

Kural: Hiçbir AutoSRT penceresi açıldığı monitörün çalışma alanını (görev çubuğu
hariç) aşmaz, ortalanmış açılır ve ana içerik pencere ne kadar küçülürse küçülsün
KESİLMEZ (kaydırılır). Bu dosyadaki testlerden biri kırılıyorsa düzeltilmesi
gereken arayüz kodudur, test değil.

    python -m unittest discover -s tests -v
    (GUI testini atlamak için: set AUTOSRT_SKIP_GUI=1)
"""
import ast
import math
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MAIN_PY = os.path.join(ROOT, "main.py")

from autosrt_core import screenfit as SF   # noqa: E402

# Pencere boyutunu/konumunu değiştirebilen TEK yetkili metotlar
ALLOWED_GEOMETRY_FUNCS = {"_place_window", "_keep_on_screen"}
GEOMETRY_CALLS = {"geometry", "wm_geometry", "minsize", "wm_minsize", "maxsize", "wm_maxsize"}

MONITORS = [
    (0, 0, 1920, 1020),        # 1080p + alt görev çubuğu
    (0, 0, 1366, 728),         # yaygın dizüstü
    (0, 0, 1280, 680),
    (0, 0, 1024, 728),
    (0, 48, 1920, 1080),       # üstte görev çubuğu
    (-1920, 0, 0, 1040),       # soldaki ikinci monitör (negatif koordinat)
    (1920, -200, 4480, 1240),  # sağda, kaydırılmış 1440p
    (0, 0, 1080, 1872),        # dikey monitör
    (0, 0, 3840, 2100),        # 4K
]
SCALES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]


def _main_tree():
    with open(MAIN_PY, encoding="utf-8") as f:
        return ast.parse(f.read())


def geometry_offenders(tree):
    """İzin verilen iki metot DIŞINDA pencere boyutu/konumu veren çağrılar."""
    offenders = []

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr in GEOMETRY_CALLS
                    and (child.args or child.keywords)
                    and func not in ALLOWED_GEOMETRY_FUNCS):
                offenders.append(f"main.py:{child.lineno} {child.func.attr}() in {func}")
            visit(child, f)

    visit(tree, "<module>")
    return offenders


def _window_specs():
    """main.py'deki WIN_* (istenen, en küçük) değerlerini import etmeden okur."""
    tree = _main_tree()
    specs = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", "")
            if name.startswith("WIN_"):
                specs[name] = ast.literal_eval(node.value)
    return specs


class TestFitGeometry(unittest.TestCase):
    def test_every_window_fits_every_monitor_and_scale(self):
        specs = _window_specs()
        self.assertTrue({"WIN_MAIN", "WIN_EDITOR", "WIN_WIZARD"} <= set(specs), specs)
        for name, (desired, minimum) in specs.items():
            for work in MONITORS:
                for s in SCALES:
                    f = SF.fit_geometry(work, s, desired, minimum)
                    fw, fh = SF.frame_overhead(s)
                    outer = (f.x, f.y, f.x + math.ceil(f.width * s) + fw,
                             f.y + math.ceil(f.height * s) + fh)
                    msg = f"{name} {work} x{s} -> {f} outer={outer}"
                    self.assertGreaterEqual(outer[0], work[0], msg)
                    self.assertGreaterEqual(outer[1], work[1], msg)
                    self.assertLessEqual(outer[2], work[2], msg)
                    self.assertLessEqual(outer[3], work[3], msg)
                    self.assertLessEqual(f.width, desired[0], msg)
                    self.assertLessEqual(f.height, desired[1], msg)
                    self.assertLessEqual(f.min_width, f.width, msg)
                    self.assertLessEqual(f.min_height, f.height, msg)

    def test_large_screen_gets_requested_size_centered(self):
        f = SF.fit_geometry((0, 0, 3840, 2100), 1.0, (1100, 820), (980, 540))
        self.assertEqual((f.width, f.height), (1100, 820))
        fw, fh = SF.frame_overhead(1.0)
        self.assertLessEqual(abs((f.x + (1100 + fw) / 2) - 1920), 1)
        self.assertLessEqual(abs((f.y + (820 + fh) / 2) - 1050), 1)

    def test_users_reported_case_1080p_125(self):
        # Kullanıcının ekranı: 1920×1080, %125, alt görev çubuğu → eski sürüm taşıyordu.
        f = SF.fit_geometry((0, 0, 1920, 1020), 1.25, (1100, 820), (980, 540))
        fw, fh = SF.frame_overhead(1.25)
        self.assertLessEqual(f.y + math.ceil(f.height * 1.25) + fh, 1020)
        self.assertLess(f.height, 820)

    def test_overflow_fix_shifts_then_shrinks(self):
        work = (0, 0, 1920, 1020)
        self.assertEqual(SF.overflow_fix((100, 100, 900, 900), work), (0, 0, 0, 0))
        self.assertEqual(SF.overflow_fix((100, 300, 900, 1100), work), (0, -80, 0, 0))
        self.assertEqual(SF.overflow_fix((-50, 10, 750, 500), work), (50, 0, 0, 0))
        dx, dy, sw, sh = SF.overflow_fix((0, 0, 1000, 1200), work)
        self.assertEqual((sh, dy), (180, 0))


class TestGeometryIsOnlySetInOnePlace(unittest.TestCase):
    """Sabit boyutlu .geometry('1100x820') gibi çağrılar ekranı taşırır. Pencere
    boyutu yalnızca _place_window / _keep_on_screen içinden verilebilir."""

    def test_no_geometry_calls_outside_allowed_functions(self):
        offenders = geometry_offenders(_main_tree())
        self.assertEqual(offenders, [], "Pencere boyutu yalnızca _place_window ile verilir "
                                        "(AGENTS.md):\n" + "\n".join(offenders))

    def test_guard_really_catches_a_fixed_geometry(self):
        """Koruma testinin kendisi çalışıyor mu? Eski hatalı kodu yakalamalı."""
        bad = ("class App:\n"
               "    def __init__(self):\n"
               "        self.geometry('1100x820')\n"
               "        self.minsize(980, 640)\n"
               "    def _place_window(self, win):\n"
               "        win.geometry('10x10+0+0')\n")
        found = geometry_offenders(ast.parse(bad))
        self.assertEqual(len(found), 2, found)
        self.assertTrue(all("__init__" in x for x in found), found)

    def test_main_content_lives_in_fit_scroll_frame(self):
        tree = _main_tree()
        build = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "_build_main")
        names = {getattr(c.func, "id", None) for c in ast.walk(build) if isinstance(c, ast.Call)}
        self.assertIn("_FitScrollFrame", names,
                      "Ana içerik _FitScrollFrame içinde olmalı (AGENTS.md)")


@unittest.skipUnless(sys.platform == "win32" and os.environ.get("AUTOSRT_SKIP_GUI") != "1",
                     "GUI testi yalnız Windows'ta")
class TestRealWindow(unittest.TestCase):
    """Gerçek pencere: açılınca çalışma alanının içinde mi, küçülünce kayıyor mu?"""

    def _pump(self, app, sec):
        end = time.time() + sec
        while time.time() < end:
            app.update()
            time.sleep(0.02)

    def test_window_inside_work_area_and_content_never_clipped(self):
        saved = sys.stdout, sys.stderr
        try:
            import main as A
            app = A.AutoSRTApp()
            try:
                self._pump(app, 1.2)
                rect = SF.visible_window_rect(int(app.wm_frame(), 16))
                work = app._fit_work
                self.assertIsNotNone(rect)
                self.assertTrue(rect[0] >= work[0] and rect[1] >= work[1]
                                and rect[2] <= work[2] and rect[3] <= work[3],
                                f"pencere {rect} çalışma alanı {work} dışına taşıyor")
                self.assertIsInstance(app.main_scroll, A._FitScrollFrame)

                # Pencere içerikten kısa olursa: kesilme yok, kaydırma var.
                app._place_window(app, (1000, 360), (300, 200))
                self._pump(app, 1.0)
                ms = app.main_scroll
                ms.sync()
                need = ms.inner.winfo_reqheight()
                visible = ms._canvas.winfo_height()
                region_h = float(str(ms._canvas.cget("scrollregion")).split()[3])
                self.assertGreater(need, visible)
                self.assertTrue(ms.scrolling)
                self.assertGreaterEqual(region_h, need)
            finally:
                app.destroy()
        finally:
            sys.stdout, sys.stderr = saved


if __name__ == "__main__":
    unittest.main()
