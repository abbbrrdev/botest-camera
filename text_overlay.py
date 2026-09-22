import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Шрифты с поддержкой кириллицы в типичных местах на разных ОС —
# ищем первый, что реально есть на этой машине.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
    "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    "C:\\Windows\\Fonts\\segoeui.ttf",  # Windows (запасной)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux (Debian/Ubuntu)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux (запасной)
]

_font_cache = {}


def _find_font_path():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    raise RuntimeError(
        "Не найден системный шрифт с поддержкой кириллицы. "
        "Добавьте путь к шрифту в FONT_CANDIDATES в text_overlay.py."
    )


def _get_font(size: int):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_find_font_path(), size)
    return _font_cache[size]


def put_text_ru(frame, text, org, font_size=32, color_bgr=(0, 255, 0)):
    """cv2.putText не умеет кириллицу — рисуем через PIL и возвращаем обратно как BGR-кадр."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, font=_get_font(font_size), fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
