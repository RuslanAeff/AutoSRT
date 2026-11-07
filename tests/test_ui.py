# -*- coding: utf-8 -*-
"""Arayüz gezinmesi ve durum kilitleri; model/API/kişisel ayar gerektirmez."""
import ast
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch


class TestTranslations(unittest.TestCase):
    def test_both_languages_have_the_same_keys(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
        translations = next(ast.literal_eval(n.value) for n in tree.body
                            if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "L" for t in n.targets))
        self.assertEqual(set(translations["tr"]), set(translations["en"]))

    def test_palette_text_contrast_in_both_appearances(self):
        import main as A

        def luminance(color):
            rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
            return sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722)))

        def contrast(a, b):
            x, y = sorted((luminance(a), luminance(b)))
            return (y + 0.05) / (x + 0.05)

        for mode in (0, 1):
            for name, (fill, hover) in A.ACCENTS.items():
                for color in (fill, hover):
                    with self.subTest(mode=mode, accent=name, color=color[mode]):
                        self.assertGreaterEqual(contrast(color[mode], A.ON_ACCENT[mode]), 4.5)
            for surface in (A.BG, A.SIDEBAR, A.SURFACE, A.SURFACE2, A.GLASS):
                for text in (A.TEXT, A.MUTED):
                    self.assertGreaterEqual(contrast(text[mode], surface[mode]), 4.5)
            self.assertGreaterEqual(contrast(A.TEXT[mode], A.SEGMENT[mode]), 4.5)


class TestGlassSurface(unittest.TestCase):
    """İnce şerit (kuyrukta dosya varken) üzerinde dekor cam kartlar çizilmez;
    çizilirse sağdaki 'Dosya Ekle' düğmesinin üstüne biner."""

    @staticmethod
    def _bright_pixels(img):
        raw = img.convert("RGBA").tobytes()          # getdata Pillow 14'te kalkıyor
        return sum(1 for i in range(0, len(raw), 4)
                   if raw[i + 3] and min(raw[i], raw[i + 1], raw[i + 2]) >= 225)

    def test_floating_sheets_only_on_full_height_drop_zone(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import autosrt_widgets as W
        fill, edge, bg = "#343438", "#62626e", "#202022"      # koyu tema cam yüzeyi
        full = W.surface_image(1000, 134, 20, fill, edge, bg, 1, True, 1.0)
        strip = W.surface_image(1000, 60, 20, fill, edge, bg, 1, True, 1.0)
        self.assertGreater(self._bright_pixels(full), 50)      # kartlar var
        self.assertEqual(self._bright_pixels(strip), 0)        # şeritte yok
        self.assertLess(60, W.GLASS_SHEETS_MIN_H)
        # Şeritte renkli ışıltı da yok: düğmenin durduğu sağ uç düz cam rengidir,
        # yoksa düğmenin düz renkli köşeleri dikdörtgen leke gibi görünür.
        r, g, b, _ = strip.getpixel((930, 30))
        fr, fg, fb = (int(fill[i:i + 2], 16) for i in (1, 3, 5))
        self.assertLessEqual(max(abs(r - fr), abs(g - fg), abs(b - fb)), 2)


@unittest.skipUnless(sys.platform == "win32" and os.environ.get("AUTOSRT_SKIP_GUI") != "1",
                     "GUI testi yalnız Windows'ta")
class TestWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main as A
        cls.A = A
        saved = sys.stdout, sys.stderr
        with patch.object(A.AutoSRTApp, "_load_settings"), \
                patch.object(A.AutoSRTApp, "_detect_device"), patch.object(A.HJ, "cleanup_stale"):
            cls.app = A.AutoSRTApp()
        sys.stdout, sys.stderr = saved
        cls.errors = []
        cls.app.report_callback_exception = lambda *exc: cls.errors.append(str(exc[1]))
        cls.pump(0.6)

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()
        if cls.errors:
            raise AssertionError(cls.errors)

    @classmethod
    def pump(cls, seconds=0.15):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            cls.app.update()
            time.sleep(0.01)

    def tearDown(self):
        self.assertEqual(self.errors, [], "Tk callback hatası")

    def test_tabs_and_translation_preserve_settings(self):
        app = self.app
        app.char_slider.set(67)
        app.term_entry.delete(0, "end")
        app.term_entry.insert(0, "AutoSRT, Parakeet")
        for lang in ("en", "tr"):
            app._set_lang(lang)
            for key, panel in app._settings_panels.items():
                app._on_settings_tab(app.t("tab_" + key))
                self.pump()
                self.assertTrue(panel.winfo_viewable())
                self.assertEqual(sum(bool(p.winfo_viewable()) for p in app._settings_panels.values()), 1)
                self.assertEqual(app.char_slider.get(), 67)
                self.assertEqual(app.term_entry.get(), "AutoSRT, Parakeet")
        app.char_slider.set(45)

    def test_engine_cards_preserve_modes_and_busy_lock(self):
        app, A = self.app, self.A
        with patch.object(A.GeminiEngine, "available", return_value=True), \
                patch.object(A.PKE, "available", return_value=True), \
                patch.object(A, "gemini_api_key", return_value="test-only"):
            for key in A.ENGINES:
                app.engine_cards[key].invoke()
                self.assertEqual(app._engine, key)
                self.assertEqual(bool(app.model_menu.grid_info()), key != "gemini")
                self.assertEqual(bool(app.key_entry.grid_info()), key in ("hybrid", "gemini"))
            app._set_busy(True)
            for lang in ("en", "tr"):
                app._set_lang(lang)
                for card in app.engine_cards.values():
                    self.assertEqual(card.cget("state"), "disabled")
                    card.invoke()
                self.assertEqual(app._engine, "gemini")
            app._set_busy(False)
            app.engine_cards["whisper"].invoke()
            self.assertEqual(app._engine, "whisper")

    def test_theme_selection_keeps_unselected_labels_readable(self):
        app = self.app
        for mode in ("☀", "☾"):
            app._set_mode(mode)
            for accent in self.A.ACCENTS:
                app._set_accent(accent)
                app._on_settings_tab(app.t("tab_look"))
                for value, button in app.settings_tabs._buttons_dict.items():
                    expected = self.A.TEXT if value == app.settings_tabs.get() else self.A.MUTED
                    self.assertEqual(button.cget("text_color"), expected)
        app._set_accent("graphite")

    def test_palette_migration_and_saved_choices(self):
        app, A = self.app, self.A
        settings = app._collect_settings()
        settings.update(palette_version=1, accent="indigo", appearance="dark", term="")
        with patch.object(A, "load_secret", return_value=None):
            app._apply_settings(settings)
            self.assertEqual(app._accent_name, "graphite")
            self.assertEqual(A.ctk.get_appearance_mode(), "Dark")
            self.assertEqual(app._engine, settings["engine"])
            settings.update(palette_version=2, accent="indigo", appearance="light")
            app._apply_settings(settings)
            self.assertEqual(app._accent_name, "indigo")
            self.assertEqual(A.ctk.get_appearance_mode(), "Light")
            self.assertEqual(app._collect_settings()["palette_version"], 2)
        app._set_accent("graphite")

    def test_long_queue_names_keep_controls_accessible(self):
        app = self.app
        app.files = ["C:/media/" + "long_filename_" * 30 + ".mp4", "C:/media/second.wav"]
        app._render_files()
        self.pump()
        row = app.files_list.winfo_children()[0]
        label = next(w for w in row.winfo_children()
                     if isinstance(w, self.A.ctk.CTkLabel) and w.grid_info().get("column") == 1)
        self.assertIn("…", label.cget("text"))
        self.assertTrue(label.cget("text").endswith(".mp4"))
        for w in row.winfo_children():
            self.assertLessEqual(w.winfo_x() + w.winfo_width(), row.winfo_width())
        app._move_file(app.files[1], -1)
        self.assertTrue(app.files[0].endswith("second.wav"))
        app._clear_files()
        self.assertFalse(app._drop_compact)

    def test_narrow_window_scrolls_in_both_directions(self):
        app = self.app
        app._place_window(app, (520, 360), (300, 200))
        self.pump(0.4)
        ms = app.main_scroll
        ms.sync()
        region = [float(n) for n in str(ms._canvas.cget("scrollregion")).split()]
        self.assertGreater(region[2], ms._canvas.winfo_width())
        self.assertGreater(region[3], ms._canvas.winfo_height())
        self.assertTrue(ms._hsb.winfo_ismapped())
        self.assertTrue(ms.scrolling)
        app._place_window(app, *self.A.WIN_MAIN)
        self.pump(0.4)
