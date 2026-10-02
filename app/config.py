"""ค่าคงที่และพารามิเตอร์ทั้งหมดของโปรแกรมสร้างสมุดระบายสี A4"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "fonts"
STATIC_DIR = BASE_DIR / "static"

# Sarabun ครอบคลุมทั้งภาษาไทย อักษรละติน และตัวเลข ได้ในฟอนต์เดียว
# Noto Sans Thai ใช้ไม่ได้เพราะไม่มี glyph ตัวเลข จะออกมาเป็นกล่องสี่เหลี่ยม
FONT_REGULAR = FONT_DIR / "Sarabun-Regular.ttf"
FONT_BOLD = FONT_DIR / "Sarabun-Bold.ttf"

# --- A4 ที่ 300 DPI -------------------------------------------------------------
DPI = 300
MM_PER_INCH = 25.4

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A4_WIDTH_PX = round(A4_WIDTH_MM / MM_PER_INCH * DPI)  # 2480
A4_HEIGHT_PX = round(A4_HEIGHT_MM / MM_PER_INCH * DPI)  # 3508

PAGE_MARGIN_MM = 12.0
FRAME_RADIUS_MM = 8.0
FRAME_STROKE_MM = 2.5
CELL_GAP_MM = 6.0

CAPTION_SIZE_PT = 20.0
PAGE_NUMBER_SIZE_PT = 12.0
COVER_TITLE_SIZE_PT = 44.0
COVER_SUBTITLE_SIZE_PT = 16.0
# จำนวนภาพย่อสูงสุดบนหน้าปก มากเกินจะดูยุ่ง
COVER_MAX_THUMBS = 9
COVER_GAP_MM = 5.0

# ตัวคูณสำหรับแปลงค่า "auto" ให้เป็นตัวเลขที่ใช้ได้จริง
DETECT_BINARY_RATIO = 0.90

# ต่ำกว่านี้คือความละเอียดไม่พอสำหรับขยายเป็น A4 จะเตือนผู้ใช้
MIN_LONG_EDGE_PX = 1400

SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

GRID_OPTIONS: dict[int, tuple[int, int]] = {
    1: (1, 1),
    2: (1, 2),
    4: (2, 2),
    6: (2, 3),
    9: (3, 3),
    12: (3, 4),
}


def mm_to_px(mm: float) -> int:
    """แปลงมิลลิเมตรเป็นพิกเซลที่ DPI ปัจจุบัน"""
    return max(1, round(mm / MM_PER_INCH * DPI))


def pt_to_px(pt: float) -> int:
    """แปลงพอยต์เป็นพิกเซลที่ DPI ปัจจุบัน (1pt = 1/72 นิ้ว)"""
    return max(1, round(pt / 72.0 * DPI))


@dataclass
class LineArtParams:
    """พารามิเตอร์การแปลงภาพลายเส้น

    ค่า None หมายถึง "ให้โปรแกรมตัดสินใจเองจากภาพ" ซึ่งเป็นค่าเริ่มต้นของทุกช่อง
    ยกเว้น target_line_mm ที่มีค่าเริ่มต้นเป็นตัวเลขชัดเจน
    """

    target_line_mm: float | None = 2.5
    speckle_ratio: float | None = None
    denoise: int | None = None
    xdog_sigma: float | None = None
    xdog_tau: float | None = 0.98
    xdog_phi: float | None = 18.0
    close_iterations: int = 1
    strip_border: bool = True
    use_ai: bool = False
    max_long_edge: int = 1400

    def to_dict(self) -> dict:
        return asdict(self)

    def resolved(self) -> dict[str, float | int | bool]:
        """ค่าที่ผ่านการตัดสินใจแล้ว ใช้ตอนแสดงผลในหน้าเว็บ"""
        return asdict(self)


@dataclass
class BookParams:
    """พารามิเตอร์การจัดหน้าสมุด"""

    title: str = "สมุดระบายสีของฉัน"
    author: str = ""
    per_page: int = 1
    show_frame: bool = True
    show_caption: bool = True
    show_page_number: bool = True
    include_cover: bool = True
    caption_size_pt: float = CAPTION_SIZE_PT

    def to_dict(self) -> dict:
        return asdict(self)

    def grid(self) -> tuple[int, int]:
        return GRID_OPTIONS.get(self.per_page, GRID_OPTIONS[1])
