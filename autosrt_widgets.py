# -*- coding: utf-8 -*-
"""DPI-aware desktop surfaces and controls. No transcription dependencies."""
from functools import lru_cache
import time

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFilter, ImageTk

# Bu yükseklikten (mantıksal px) kısa cam yüzeylerde dekor (renkli ışıltı + yüzen
# kartlar) ÇİZİLMEZ; yalnız ışıklı kenar ve üst parlama kalır. İnce şeritte
# (kuyrukta dosya varken) dekor, sağdaki düğmenin altına/üstüne denk geliyordu:
# kartlar düğmeyi örtüyor, ışıltı da düğmenin düz renkli köşelerini belli ediyordu.
GLASS_SHEETS_MIN_H = 110


def mix(a, b, amount):
    aa = tuple(int(a[i:i + 2], 16) for i in (1, 3, 5))
    bb = tuple(int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x + (y - x) * amount):02x}" for x, y in zip(aa, bb))


@lru_cache(maxsize=64)
def symbol(kind, size, color):
    """Small outlined icons, drawn at 4× resolution instead of font glyphs."""
    size = max(8, int(size))
    im = Image.new("RGBA", (size * 4, size * 4))
    draw = ImageDraw.Draw(im)
    unit = size * 4 / 24

    def line(points, width=1.8):
        pts = [(x * unit, y * unit) for x, y in points]
        stroke = max(1, round(width * unit))
        draw.line(pts, fill=color, width=stroke, joint="curve")
        r = stroke / 2
        for x, y in pts:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    if kind == "chevron":
        line(((6, 9), (12, 15), (18, 9)), 2)
    elif kind == "check":
        line(((5, 12), (10, 17), (19, 7)), 2)
    else:  # arrow entering a tray, optically centred in its 24×24 view box
        line(((12, 3), (12, 14)))
        line(((7.5, 10), (12, 14.5), (16.5, 10)))
        line(((4, 15), (4, 20), (20, 20), (20, 15)))
    return im.resize((size, size), Image.Resampling.LANCZOS)


@lru_cache(maxsize=64)
def surface_image(width, height, radius, fill, border, background, stroke=1, glass=False, scale=1.0):
    """Antialiased continuous borders; optional composited frosted layers."""
    width, height = max(2, width), max(2, height)
    quality = 3
    wh = (width * quality, height * quality)
    panel = Image.new("RGBA", (width, height), fill)
    if glass and height / scale >= GLASS_SHEETS_MIN_H:
        # The left side stays quiet for controls/text. The material and the
        # translucent sheets share a blurred backdrop on the right side.
        haze = Image.new("RGBA", (width, height))
        hd = ImageDraw.Draw(haze)
        dark = sum(int(fill[i:i + 2], 16) for i in (1, 3, 5)) < 384
        hd.ellipse((width * .77, -height, width * 1.12, height * 1.5),
                   fill=(80, 154, 245, 38 if dark else 42))
        hd.ellipse((width * .90, height * .15, width * 1.25, height * 2),
                   fill=(91, 204, 176, 28 if dark else 38))
        panel = Image.alpha_composite(panel, haze.filter(ImageFilter.GaussianBlur(26 * scale)))
        if width / scale >= 700:
            for offset, angle in ((-20, -10), (14, 8)):
                cw, ch = int(110 * scale), int(72 * scale)
                x, y = width - int(176 * scale) + int(offset * scale), (height - ch) // 2
                tile = panel.crop((x, y, x + cw, y + ch)).filter(ImageFilter.GaussianBlur(9 * scale))
                tint = (245, 251, 255, 80 if dark else 130)
                tile = Image.alpha_composite(tile, Image.new("RGBA", tile.size, tint))
                td = ImageDraw.Draw(tile)
                td.rounded_rectangle((1, 1, cw - 2, ch - 2), radius=12 * scale,
                                     outline=(255, 255, 255, 190), width=max(1, round(scale)))
                for i, length in enumerate((55, 70, 43)):
                    yy = int((23 + 12 * i) * scale)
                    td.rounded_rectangle((18 * scale, yy, (18 + length) * scale, yy + 3 * scale),
                                         radius=1.5 * scale, fill=(110, 133, 159, 150))
                mask = Image.new("L", tile.size)
                ImageDraw.Draw(mask).rounded_rectangle((0, 0, cw - 1, ch - 1), radius=12 * scale, fill=255)
                tile.putalpha(mask)
                tile = tile.rotate(angle, Image.Resampling.BICUBIC, expand=True)
                panel.alpha_composite(tile, (x - (tile.width - cw) // 2, y - (tile.height - ch) // 2))

    # Compose at 3× so the radius, border and highlight have no font-shape seams.
    result = Image.new("RGBA", wh, background)
    mask = Image.new("L", wh)
    md = ImageDraw.Draw(mask)
    inset = quality
    bounds = (inset, inset, wh[0] - inset - 1, wh[1] - inset - 1)
    md.rounded_rectangle(bounds, radius=radius * quality, fill=255)
    result.paste(panel.resize(wh, Image.Resampling.BICUBIC), (0, 0), mask)
    rd = ImageDraw.Draw(result)
    rd.rounded_rectangle(bounds, radius=radius * quality, outline=border,
                         width=max(1, round(stroke * quality)))
    if glass:
        light = mix(fill, "#ffffff", .45)
        rd.line((radius * quality, inset + quality, wh[0] - radius * quality, inset + quality),
                fill=light, width=quality)
    return result.resize((width, height), Image.Resampling.LANCZOS)


class SmoothFrame(ctk.CTkFrame):
    """CTk layout with a supersampled, unbroken rounded surface."""
    def __init__(self, *args, glass=False, **kwargs):
        self._glass = glass
        self._surface_photo = None
        super().__init__(*args, **kwargs)

    def _draw(self, no_color_updates=False):
        ctk.CTkBaseClass._draw(self, no_color_updates)
        if not hasattr(self, "_canvas") or not self._canvas.winfo_exists():
            return
        scale = self._get_widget_scaling()
        bg = self._apply_appearance_mode(self._bg_color)
        fill = bg if self._fg_color == "transparent" else self._apply_appearance_mode(self._fg_color)
        img = surface_image(round(self._current_width * scale), round(self._current_height * scale),
                            self._corner_radius * scale, fill,
                            self._apply_appearance_mode(self._border_color), bg,
                            max(.8, self._border_width * scale), self._glass, scale)
        self._surface_photo = ImageTk.PhotoImage(img, master=self._canvas)
        self._canvas.configure(bg=bg)
        self._canvas.delete("surface")
        self._canvas.create_image(0, 0, image=self._surface_photo, anchor="nw", tags="surface")
        self._canvas.tag_lower("surface")


class ChoiceCard(ctk.CTkFrame):
    """One hit area for title, description and padding; cancellable color easing."""
    DURATION = .16

    def __init__(self, master, *, palette, command, title="", description="", compact=False,
                 accent, ink, **kwargs):
        self.palette, self.command = palette, command
        self.title, self.description = title, description
        self.accent, self.ink = accent, ink
        self.compact = compact
        self.selected = self.hovered = self.focused = False
        self._state = "normal"
        self._animation = None
        self._paint = None
        self._surface_photo = self._mark_photo = None
        self._title_font = ctk.CTkFont("Segoe UI", 12 if compact else 14, weight="bold")
        self._description_font = ctk.CTkFont("Segoe UI", 12)
        super().__init__(master, fg_color=palette["surface"], corner_radius=14,
                         border_width=1, border_color=palette["border"], **kwargs)
        self._canvas.configure(cursor="hand2", takefocus=1)
        self._canvas.bind("<Enter>", lambda e: self._hover(True))
        self._canvas.bind("<Leave>", lambda e: self._hover(False))
        self._canvas.bind("<Button-1>", lambda e: self.invoke())
        self._canvas.bind("<Return>", lambda e: self.invoke())
        self._canvas.bind("<space>", lambda e: self.invoke())
        self._canvas.bind("<FocusIn>", lambda e: self._focus(True))
        self._canvas.bind("<FocusOut>", lambda e: self._focus(False))

    def _target(self):
        fill = self._apply_appearance_mode(self.palette["surface"])
        accent = self._apply_appearance_mode(self.accent)
        edge = self._apply_appearance_mode(self.palette["border"])
        active_hover = self.hovered and self._state == "normal"
        amount = .13 if active_hover else (.07 if self.selected else 0)
        return (mix(fill, accent, amount), accent if self.selected or self.focused
                else mix(edge, accent, .5 if active_hover else 0))

    def _transition(self, animate=True):
        self._cancel_animation()
        target = self._target()
        if not animate or self._paint is None or target == self._paint:
            self._paint = target
            self._draw()
            return
        start, began = self._paint, time.monotonic()

        def step():
            self._animation = None
            elapsed = min(1, (time.monotonic() - began) / self.DURATION)
            eased = 1 - (1 - elapsed) ** 3
            self._paint = tuple(mix(a, b, eased) for a, b in zip(start, target))
            self._draw()
            if elapsed < 1:
                self._animation = self.after(16, step)
        step()

    def _cancel_animation(self):
        if self._animation is not None:
            self.after_cancel(self._animation)
            self._animation = None

    def _hover(self, active):
        self.hovered = active and self._state == "normal"
        self._transition()

    def _focus(self, active):
        self.focused = active
        self._transition()

    def invoke(self):
        if self._state == "normal" and self.command:
            return self.command()

    def configure(self, require_redraw=False, **kwargs):
        changed = False
        for key in ("title", "description", "selected", "accent", "ink"):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
                changed = True
        if "state" in kwargs:
            self._state = kwargs.pop("state")
            self.hovered = False
            self._canvas.configure(cursor="hand2" if self._state == "normal" else "",
                                   takefocus=int(self._state == "normal"))
            changed = True
        super().configure(require_redraw=require_redraw, **kwargs)
        if changed:
            self._transition()

    def cget(self, key):
        if key == "state":
            return self._state
        return super().cget(key)

    def _draw(self, no_color_updates=False):
        ctk.CTkBaseClass._draw(self, no_color_updates)
        if not hasattr(self, "_canvas") or not self._canvas.winfo_exists():
            return
        scale = self._get_widget_scaling()
        w, h = round(self._current_width * scale), round(self._current_height * scale)
        fill, edge = self._paint or self._target()
        bg = self._apply_appearance_mode(self._bg_color)
        self._surface_photo = ImageTk.PhotoImage(
            surface_image(w, h, self._corner_radius * scale, fill, edge, bg,
                          (1.5 if self.selected or self.focused else 1) * scale), master=self._canvas)
        self._canvas.configure(bg=bg)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, image=self._surface_photo, anchor="nw")
        text = self._apply_appearance_mode(self.palette["text"])
        muted = self._apply_appearance_mode(self.palette["muted"])
        mark_size = round((20 if self.compact else 19) * scale)
        mark = Image.new("RGBA", (mark_size * 4, mark_size * 4))
        md = ImageDraw.Draw(mark)
        selected_fill = self._apply_appearance_mode(self.accent)
        md.ellipse((3, 3, mark.width - 4, mark.height - 4),
                   fill=selected_fill if self.selected or self.compact else fill,
                   outline=selected_fill if self.selected or self.compact else edge, width=5)
        mark = mark.resize((mark_size, mark_size), Image.Resampling.LANCZOS)
        if self.selected:
            mark.alpha_composite(symbol("check", mark_size, self._apply_appearance_mode(self.ink)))
        self._mark_photo = ImageTk.PhotoImage(mark, master=self._canvas)
        mx = 22 * scale if self.compact else w - 24 * scale
        my = h / 2 if self.compact else 22 * scale
        self._canvas.create_image(mx, my, image=self._mark_photo)
        self._canvas.create_text((42 if self.compact else 16) * scale,
                                 h / 2 if self.compact else 22 * scale,
                                 text=self.title, anchor="w", fill=text if self._state == "normal" else muted,
                                 font=self._apply_font_scaling(self._title_font))
        if not self.compact:
            self._canvas.create_text(16 * scale, 40 * scale, text=self.description, anchor="nw",
                                     fill=muted, width=max(20, w - 32 * scale),
                                     font=self._apply_font_scaling(self._description_font))

    def _set_appearance_mode(self, mode_string):
        self._cancel_animation()
        self._paint = None
        super()._set_appearance_mode(mode_string)

    def destroy(self):
        self._cancel_animation()
        super().destroy()


class _ChevronMixin:
    def _draw(self, no_color_updates=False):
        super()._draw(no_color_updates)
        self._canvas.itemconfigure("dropdown_arrow", state="hidden")
        color = self._text_color_disabled if self._state == "disabled" else self._text_color
        # CTk defaults include Tk names such as gray74; Pillow expects RGB.
        rgb = self.winfo_rgb(self._apply_appearance_mode(color))
        color = "#" + "".join(f"{v // 257:02x}" for v in rgb)
        size = max(10, round(self._apply_widget_scaling(14)))
        self._chevron_photo = ImageTk.PhotoImage(
            symbol("chevron", size, color), master=self._canvas)
        self._canvas.delete("clean_chevron")
        self._canvas.create_image(self._apply_widget_scaling(self._current_width - self._current_height / 2),
                                   self._apply_widget_scaling(self._current_height / 2),
                                   image=self._chevron_photo, tags="clean_chevron")


class OptionMenu(_ChevronMixin, ctk.CTkOptionMenu):
    pass


class ComboBox(_ChevronMixin, ctk.CTkComboBox):
    pass
