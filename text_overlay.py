import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
_font_cache = {}


def _get_font(size: int):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


def put_text_ru(frame, text, org, font_size=32, color_bgr=(0, 255, 0)):
    """cv2.putText не умеет кириллицу — рисуем через PIL и возвращаем обратно как BGR-кадр."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, font=_get_font(font_size), fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
