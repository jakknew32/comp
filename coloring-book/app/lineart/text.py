"""วาดข้อความ (รวมถึงภาษาไทย) เป็นภาพ PNG โปรใสด้

ทำไมต้องเป็น PNG แทนที่จะฝังข้อความลง PDF ตรง ๆ
------------------------------------------------
ReportLab และ PyMuPDF **ไม่ทำ complex text shaping** สำหรับภาษาไทย
สระบนล่างและวรรณยุกต์จะถูกวางผิดตำแหน่ง อ่านไม่ออก
ทางแก้คือวาดข้อความเป็นภาพด้วย Pillow ซึ่งใช้ libraqm จัดตำแหน่งสระ/วรรณยุกต์
ได้ถูกต้อง แล้วประกอบลงหน้าที่เป็น raster อยู่แล้ว — จึงไม่เสียอะไรเลย

ข้อกำหนดสำคัญ
-------------
* ใช้ฟอนต์ที่ vendor มาเองใน ``fonts/`` ห้ามพึ่งฟอนต์ของระบบ
* ต้องมี libraqm — ตรวจด้วย :func:`raqm_available` และเตือนให้ชัด
* วาดที่ 4 เท่าขนาดจริงแล้วย่อลงมา เพื่อให้ขอบเรียบแม้ต้องขยายถึง 600 DPI
"""

from __future__ import annotations

import functools
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from ..config import FONT_PATH, TEXT_SCALE

#: ภาษาที่ส่งให้ libraqm เพื่อเปิดการจัดตำแหน่งสระ/วรรณยุกต์
RAQM_LANGUAGE = "th"


def raqm_available() -> bool:
    """ตรวจว่า Pillow ที่ติดตั้ง build มาพร้อม libraqm หรือไม่

    ถ้าไม่มี การวาดภาษาไทยจะผิดรูป (สระบนล่าง/วรรณยุกต์ไปทับกัน)
    โปรแกรมต้องแจ้งผู้ใช้ ไม่ใช่ปล่อยให้เห็นผลลัพธ์ผิดรูปเงียบ ๆ
    """
    try:
        return bool(features.check("raqm"))
    except Exception:  # pragma: no cover - กันพลาด
        return False


def shaping_warning() -> Optional[str]:
    """ข้อความเตือนเรื่องการจัดวางสระ ถ้า libraqm ไม่พร้อม (ไม่มีก็คืน ``None``)"""
    if raqm_available():
        return None
    return (
        "Pillow ที่ติดตั้งไม่ได้ build มาพร้อม libraqm — "
        "สระบนล่างและวรรณยุกต์ภาษาไทยอาจวางผิดตำแหน่ง "
        "ต้องติดตั้ง Pillow ใหม่โดยมี libraqm ดูวิธีใน README"
    )


@functools.lru_cache(maxsize=64)
def load_font(size_px: int) -> ImageFont.FreeTypeFont:
    """โหลดฟอนต์หน้าต่างหนา (cache ไว้ เพราะโหลดซ้ำช้า)"""
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ฟอนต์ที่ {FONT_PATH} — ต้อง vendor NotoSansThai มาไว้ใน fonts/"
        )
    return ImageFont.truetype(str(FONT_PATH), size_px)


def text_size(text: str, size_px: int) -> tuple[int, int]:
    """วัดขนาดข้อความเป็นพิกเซล ณ ขนาดฟอนต์ที่ระบุ (ไม่รวม padding)

    ใช้ Raqm เสมอถ้ามี เพื่อให้การวัดตรงกับการวาดจริง (ขนาดสระบนล่าง
    ทำให้ความสูงต่างจาก basic layout)
    """
    font = load_font(size_px)
    if not text:
        return (0, size_px)

    if raqm_available():
        left, top, right, bottom = font.getbbox(text, language=RAQM_LANGUAGE)
    else:  # pragma: no cover - เส้นทางสำรอง
        left, top, right, bottom = font.getbbox(text)

    return (int(round(right - left)), int(round(bottom - top)))


def render_text(
    text: str,
    size_px: int,
    *,
    max_width: Optional[int] = None,
    color: int = 0,
    supersample: int = TEXT_SCALE,
    pad_ratio: float = 0.12,
) -> Image.Image:
    """วาดข้อความเป็นภาพ RGBA โปรใสด้

    คืนภาพที่วาดที่ ``supersample`` เท่าขนาดจริง เพื่อให้ผู้เรียกใช้
    ย่อลงมาเองตอนประกอบหน้า — ขอบจึงเรียบแม้ขยายถึง 600 DPI

    :param max_width: ความกว้างสูงสุดของภาพผลลัพธ์ **รวม padding**
        ถ้าระบุ ข้อความที่กว้างเกินจะถูกย่อขนาดลงให้พอดี
    :param color: ค่าสี 0 = ดำ (โปรไสเลอร์ต้องการสีเข้ม ไม่ใช่สีหมึกพิมพ์)
    """
    if not text:
        return Image.new("RGBA", (1, 1), (255, 255, 255, 0))

    ss = max(1, int(supersample))
    font_size = size_px * ss
    pad_factor = 1.0 + 2.0 * pad_ratio

    # ถ้ากว้างเกินกำหนด ให้ลดขนาดฟอนต์ลงจนพอดี
    # ต้องหารด้วย pad_factor เพราะ padding จะถูกเติมภายหลัง
    if max_width:
        target = (max_width * ss) / pad_factor
        measured, _ = text_size(text, font_size)
        if measured > target > 0:
            font_size = max(4, int(font_size * (target / measured)))

    font = load_font(font_size)

    if raqm_available():
        left, top, right, bottom = font.getbbox(text, language=RAQM_LANGUAGE)
    else:  # pragma: no cover - เส้นทางสำรอง
        left, top, right, bottom = font.getbbox(text)

    # เผื่อพื้นที่สำหรับสระบนล่าง/วรรณยุกต์ที่อาจล้ำเลย bounding box
    # ของ Raqm เล็กน้อยในบางฟอนต์
    pad_x = int(round((right - left) * pad_ratio)) + 2
    pad_y = int(round((bottom - top) * pad_ratio)) + 2

    width = max(1, int(round(right - left)) + pad_x * 2)
    height = max(1, int(round(bottom - top)) + pad_y * 2)

    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # ย้ายพิกัดให้ bbox เริ่มที่ (pad_x, pad_y)
    origin_x = pad_x - left
    origin_y = pad_y - top

    kwargs = {"font": font, "fill": (color, color, color, 255)}
    if raqm_available():
        kwargs["language"] = RAQM_LANGUAGE
    draw.text((origin_x, origin_y), text, **kwargs)

    return image


def render_text_mask(
    text: str,
    size_px: int,
    *,
    max_width: Optional[int] = None,
    threshold: int = 128,
) -> np.ndarray:
    """วาดข้อความแล้วแปลงเป็น binary mask (255 = หมึก) โปรด้วย OpenCV

    ใช้สำหรับประกอบลงหน้า A4 ที่เป็นภาพขาวดำ โดยไม่ต้องกังวลเรื่อง
    alpha channel
    """
    rgba = render_text(text, size_px, max_width=max_width)
    alpha = np.asarray(rgba)[:, :, 3]
    import cv2

    _, mask = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)
    return mask


def render_text_mask_scaled(
    text: str,
    size_px: int,
    *,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    threshold: int = 128,
) -> np.ndarray:
    """วาดข้อความแล้วย่อกลับมาที่ขนาดจริง โดยเคารพความกว้าง/สูงสูงสุด

    นี่คือฟังก์ชันที่ผู้ประกอบหน้าควรใช้ เพราะรับประกันว่า
    * ขอบเรียบ (วาดที่ 4 เท่าแล้วย่อด้วย INTER_AREA)
    * ผลลัพธ์ไม่ล้นกรอบ (ย่อตามข้อจำกัดที่เล็กกว่า)
    """
    import cv2

    mask = render_text_mask(text, size_px, max_width=max_width, threshold=threshold)
    if mask.size == 0:
        return mask

    height, width = mask.shape[:2]
    scale = 1.0
    if max_width and width > max_width:
        scale = min(scale, max_width / float(width))
    if max_height and height > max_height:
        scale = min(scale, max_height / float(height))

    if scale < 1.0:
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return mask


def ink_bounding_box(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """หา bounding box ของหมึกใน mask

    คืน ``(x, y, w, h)`` หรือ ``None`` ถ้าไม่พบหมึก — ใช้ในเทสต์เพื่อ
    ยืนยันว่าสระไม่ถูกตัดขอบ
    """
    import cv2

    coords = cv2.findNonZero((mask > 127).astype(np.uint8))
    if coords is None:
        return None
    return cv2.boundingRect(coords)
