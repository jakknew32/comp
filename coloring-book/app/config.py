"""ค่าคงที่กระดาษ A4, พารามิเตอร์ และค่าเริ่มต้นของทั้งโปรแกรม.

ทุกค่าที่ผู้ใช้ปรับได้ถูกทำเป็น ``Optional[float]`` — ค่า ``None`` หมายถึง
"ให้เซิร์ฟเวอร์ตัดสินใจเอง" ซึ่งเป็นแกนของแนวคิด อัตโนมัติ + ปรับเองได้
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------
# หน้ากระดาษ
# --------------------------------------------------------------------------

DPI = 300
MM_PER_INCH = 25.4

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

A4_WIDTH_PX = 2480  # round(210 / 25.4 * 300)
A4_HEIGHT_PX = 3508  # round(297 / 25.4 * 300)

MARGIN_MM = 12.0

#: กรอบมุมมนรอบเซลล์
CORNER_RADIUS_MM = 8.0
FRAME_WIDTH_MM = 2.5

#: ความหนาเส้นการ์ตูนเป้าหมาย (ตัวอักษรให้ผู้ใช้ปรับได้ 1.0–5.0 mm)
TARGET_LINE_MM = 2.5

#: ช่องว่างระหว่างช่องในกริด
CELL_GAP_MM = 6.0

#: ช่องว่างใต้ภาพเผื่อชื่อกำกับ
CAPTION_GAP_MM = 4.0
CAPTION_HEIGHT_MM = 12.0

#: ขนาดตัวอักษรชื่อกำกับ (pt ที่ 300 DPI)
CAPTION_PT = 20.0
PAGE_NUMBER_PT = 16.0
COVER_TITLE_PT = 72.0
COVER_SUBTITLE_PT = 22.0

TEXT_SCALE = 4  # วาดข้อความที่ 4x แล้วย่อ เพื่อให้ขอบเรียบ


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    """แปลงมิลลิเมตรเป็นพิกเซลที่ DPI ที่ระบุ (ปัดเป็นจำนวนเต็ม)."""
    return int(round(mm / MM_PER_INCH * dpi))


def pt_to_px(pt: float, dpi: int = DPI) -> int:
    """แปลงพอยต์เป็นพิกเซลที่ DPI ที่ระบุ."""
    return int(round(pt / 72.0 * dpi))


# --------------------------------------------------------------------------
# เส้นทาง
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "fonts"
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "samples"
FONT_PATH = FONT_DIR / "NotoSansThai-Regular.ttf"

APP_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# โหมดภาพต่อหน้า
# --------------------------------------------------------------------------

#: โหมดเริ่มต้น = 1 ภาพเต็มหน้า
DEFAULT_PER_PAGE = 1

#: จำนวนช่องที่อนุญาตต่อหน้า -> (คอลัมน์, แถว)
PER_PAGE_GRID: dict[int, tuple[int, int]] = {
    1: (1, 1),
    2: (1, 2),
    4: (2, 2),
    6: (2, 3),
    9: (3, 3),
    12: (3, 4),
}

DEFAULT_BOOK_TITLE = "สมุดระบายสีของฉัน"


# --------------------------------------------------------------------------
# พารามิเตอร์
# --------------------------------------------------------------------------


@dataclass
class ConvertParams:
    """พารามิเตอร์ขั้นตอนแปลงภาพลายเส้น.

    ค่า ``None`` = ให้ระบบวัด/คำนวณเอง (ดู :mod:`app.lineart.convert`)
    """

    #: ความหนาเส้นการ์ตูนเป้าหมายเป็นมิลลิเมตร (None = ตามค่ากลาง)
    target_line_mm: Optional[float] = None

    #: ความไวต่อจุดรบกวน 0–1 (None = คำนวณจากพื้นที่ภาพ)
    despeckle: Optional[float] = None

    #: sigma ของ XDoG เป็นพิกเซล (None = สัดส่วนกับขนาดภาพ)
    dog_sigma: Optional[float] = None

    #: จำนวนรอบ morphological close เพื่อซ่อมเส้นขาด (None = อัตโนมัติ)
    close_rounds: Optional[int] = None

    #: ปิดช่องว่างด้วยการเติม contour (None = อัตโนมัติ)
    close_holes: Optional[bool] = None

    #: ขยายภาพที่เล็กกว่านี้เป็นเต็มกรอบ (None = อัตโนมัติ)
    upscale: Optional[bool] = None

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class LayoutParams:
    """พารามิเตอร์การจัดหน้า A4."""

    per_page: int = DEFAULT_PER_PAGE
    frame: bool = True
    cover: bool = True
    book_title: str = DEFAULT_BOOK_TITLE
    page_numbers: bool = True
    captions: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class RequestParams:
    """พารามิเตอร์ที่ client ส่งมาทั้งหมด."""

    convert: ConvertParams = field(default_factory=ConvertParams)
    layout: LayoutParams = field(default_factory=LayoutParams)

    def as_dict(self) -> dict[str, Any]:
        return {
            "convert": self.convert.as_dict(),
            "layout": self.layout.as_dict(),
        }


#: ขอบเขต slider ที่ฝั่ง UI ใช้
TARGET_LINE_MM_RANGE = (1.0, 5.0)
DESPECKLE_RANGE = (0.0, 1.0)

#: จำกัดขนาดไฟล์ที่อนุญาตต่อไฟล์
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

#: ความยาวขอบภาพต่ำกว่านี้ (px) ถือว่า "ความละเอียดต่ำ" ต้องเตือนผู้ใช้
LOW_RES_PX = 1500
