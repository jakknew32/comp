"""วาดข้อความภาษาไทยเป็นภาพ

ทำเป็นภาพแทนที่จะฝังเป็นข้อความใน PDF เพราะ PDF ทั่วไปไม่ทำ complex text shaping
ภาษาไทยมีสระบน/ล่างและวรรณยุกต์ที่ต้องซ้อนกับพยัญชนะ ถ้าฝังตรงๆ จะไปวางผิดตำแหน่ง
แล้วอ่านไม่ออก วิธีนี้ให้ผลถูกต้องเสมอ
"""

from __future__ import annotations

import functools
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

from ..config import FONT_BOLD, FONT_REGULAR

SUPERSAMPLE = 4


class FontUnavailableError(RuntimeError):
    """ไม่พบฟอนต์ที่ต้องใช้ในการวาดข้อความ"""


def _require_raqm() -> None:
    if not features.check("raqm"):
        raise FontUnavailableError(
            "Pillow ตัวนี้ไม่ได้ build พร้อม libraqm จึงจัดตำแหน่งสระภาษาไทยไม่ได้ "
            "ต้องติดตั้ง Pillow ใหม่ที่มากับ raqm"
        )


def check_setup() -> None:
    """ตรวจว่าพร้อมวาดข้อความไทยหรือไม่ ใช้ตอนเปิดเซิร์ฟเวอร์

    เช็กการครอบคลุม glyph ด้วย เพราะฟอนต์ที่มีภาษาไทยครบแต่ไม่มีตัวเลข
    จะผ่านการโหลดปกติ แต่พิมพ์เลขออกมาเป็นกล่องสี่เหลี่ยมโดยไม่มี error ให้เห็น
    """
    _require_raqm()
    if not Path(FONT_REGULAR).exists():
        raise FontUnavailableError(f"ไม่พบฟอนต์ {FONT_REGULAR}")

    missing = missing_glyphs(_COVERAGE_PROBE)
    if missing:
        raise FontUnavailableError(
            f"ฟอนต์ {Path(FONT_REGULAR).name} ไม่มีอักขระเหล่านี้ จะแสดงเป็นกล่องสี่เหลี่ยม: "
            + " ".join(missing)
        )


# ตัวอักษรที่ต้องมีเสมอ: ไทย (พร้อมสระบนล่างและวรรณยุกต์) อักษรละติน และตัวเลข
_COVERAGE_PROBE = "กขฃคฅงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ่้๊๋์็ๆฯ0123456789ABCXYZabcxyz.,/-()"


def missing_glyphs(text: str, probe_px: int = 48) -> list[str]:
    """คืนรายการอักขระใน text ที่ฟอนต์ไม่มีให้

    เทียบภาพเรนเดอร์ของอักขระนั้นกับภาพเรนเดอร์ของอักขระในช่วงส่วนตัวเป็นการเอกสาร
    (U+E000) ซึ่งไม่มีในฟอนต์ใดโดยปกติ ถ้าได้ภาพเหมือนกันเป๊ะแปลว่าอักขระนั้น
    ถูกวาดด้วยกล่อง .notdef นั่นคือไม่มี glyph จริง

    วิธีนี้จับกรณี tofu ได้ ต่างจากการเช็ค bbox ซึ่งจะไม่จับ เพราะกล่อง tofu
    มีพื้นที่หมึกอยู่แล้ว
    """
    font = load_font(probe_px)
    notdef = bytes(font.getmask("", mode="L"))
    notdef_size = font.getmask("", mode="L").size

    missing: list[str] = []
    for char in dict.fromkeys(text):
        if char.isspace():
            continue
        mask = font.getmask(char, mode="L")
        if mask.size == (0, 0):
            missing.append(char)
            continue
        if mask.size == notdef_size and bytes(mask) == notdef:
            missing.append(char)
    return missing



@functools.lru_cache(maxsize=64)
def load_font(px_size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """โหลดฟอนต์ที่ขนาดพิกเซลตามต้องการ (cache ไว้เพราะโหลดบ่อย)"""
    _require_raqm()
    path = Path(FONT_BOLD if bold else FONT_REGULAR)
    if not path.exists():
        raise FontUnavailableError(f"ไม่พบฟอนต์ {path}")
    return ImageFont.truetype(str(path), px_size, layout_engine=ImageFont.Layout.RAQM)


def measure_text(text: str, px_size: int, bold: bool = False) -> tuple[int, int]:
    """วัดขนาดข้อความที่จะวาด (ก่อนคูณด้วย SUPERSAMPLE)"""
    font = load_font(px_size, bold)
    left, top, right, bottom = font.getbbox(text or " ")
    return right - left, bottom - top


def _padding_for(px_size: int) -> int:
    """ขอบเว้นรอบตัวอักษรในหน่วยพิกเซลของภาพผลลัพธ์

    ใช้กันสระบน/ล่างและวรรณยุกต์ที่ล้ำกรอบของตัวอักษรออกไป
    """
    return max(1, (px_size * SUPERSAMPLE // 3) // SUPERSAMPLE)


def _final_width(text: str, px_size: int, bold: bool) -> int:
    """ความกว้างของภาพที่จะได้จริง รวมขอบเว้นสองข้าง"""
    text_width, _ = measure_text(text, px_size, bold)
    return text_width + 2 * _padding_for(px_size)


def fit_px_size(
    text: str,
    max_width: int,
    start_px: int,
    bold: bool = False,
    min_px: int = 8,
) -> int:
    """หาขนาดฟอนต์ใหญ่ที่สุดที่ภาพผลลัพธ์ยังพอดีความกว้างที่กำหนด

    ต้องนับขอบเว้นรอบตัวอักษรด้วย ไม่ใช่แค่ความกว้างตัวอักษร
    ไม่งั้นภาพจะล้นช่องที่จัดให้
    """
    if not text:
        return start_px
    size = max(min_px, start_px)
    while size > min_px:
        if _final_width(text, size, bold) <= max_width:
            return size
        size -= 1
    return min_px


def render_text(
    text: str,
    px_size: int,
    bold: bool = False,
    max_width: int | None = None,
) -> Image.Image:
    """วาดข้อความเป็นภาพโทนเทา (255 = หมึก, 0 = ใส)

    วาดที่ความละเอียดสูงกว่าจริง SUPERSAMPLE เท่าแล้วลดขนาดลง
    เพื่อให้ขอบเส้นเรียบแม้ขยายไปใช้ที่ 300-600 DPI
    """
    if max_width is not None:
        px_size = fit_px_size(text, max_width, px_size, bold)

    render_px = max(1, px_size * SUPERSAMPLE)
    font = load_font(render_px, bold)

    probe = font.getbbox(text or " ")
    text_w = max(1, probe[2] - probe[0])
    text_h = max(1, probe[3] - probe[1])

    # เว้นขอบเล็กน้อยรอบตัวอักษร สระบน/ล่างมักล้ำกรอบของ probe
    pad_x = max(4, render_px // 3)
    pad_y = max(4, render_px // 2)
    canvas_w = text_w + pad_x * 2
    canvas_h = text_h + pad_y * 2

    canvas = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(canvas)
    # ย้ายพิกัดให้กรอบตัวอักษรเริ่มที่ pad เสมอ
    draw.text((pad_x - probe[0], pad_y - probe[1]), text, fill=255, font=font)

    if SUPERSAMPLE == 1:
        return canvas

    out_w = max(1, round(canvas_w / SUPERSAMPLE))
    out_h = max(1, round(canvas_h / SUPERSAMPLE))
    return canvas.resize((out_w, out_h), Image.Resampling.LANCZOS)


def paste_text(
    page: Image.Image,
    text: str,
    center_x: int,
    top_y: int,
    px_size: int,
    bold: bool = False,
    max_width: int | None = None,
) -> tuple[int, int]:
    """วางข้อความกึ่งกลางตามแกน X ลงบนภาพหน้า คืน (ความกว้าง, ความสูง) ที่ใช้จริง"""
    if not text:
        return 0, 0
    mask = render_text(text, px_size, bold=bold, max_width=max_width)
    left = int(round(center_x - mask.width / 2))
    page.paste(0, (left, top_y), mask)
    return mask.width, mask.height
