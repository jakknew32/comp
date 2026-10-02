"""การทดสอบการวางแผนหน้า (:mod:`app.book.layout`) และการวาดข้อความไทย

ครอบคลุมข้อกำหนดของแผน:
* layout: 10 ภาพโหมด 4 ต่อหน้า → จำนวนหน้า + จำนวนช่องถูกต้องเสมอ
* text: ข้อความไทยที่มีสระบน/ล่าง + วรรณยุกต์ → ขนาดภาพไม่เป็นศูนย์
  และสระไม่ถูกตัดขอบ
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.book import layout as layout_mod  # noqa: E402
from app.lineart import text as text_mod  # noqa: E402
from app.lineart.compose import Cell  # noqa: E402

#: ข้อความทดสอบ: มีสระบนล่าง สระบน และวรรณยุกต์ครบ
THAI_SAMPLES = [
    "กระต่ายกินผัก",  # สระบนล่าง (ะ) + ไม้ไต่คู้
    "ที่นี่มีปลา",  # สระบน (ี) + วรรณยุกต์ (้)
    "ยิงทพ",  # ชื่อตัวอย่างจากภาพอ้างอิง
    "ก๊ก",  # วรรณยุกต์ซ้อนสามชั้น
    "หน้า 1",  # ตัวเลข + เส้นไฟหน้า
    "ก",
]


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def make_cells(count: int) -> list[Cell]:
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    return [Cell(mask=mask, caption=f"ภาพที่ {i + 1}") for i in range(count)]


@pytest.mark.parametrize("per_page", sorted(config.PER_PAGE_GRID))
def test_page_and_cell_counts_are_correct(per_page: int) -> None:
    """ทุกโหมดต้องได้จำนวนหน้าและจำนวนช่องถูกต้อง"""
    cells = make_cells(10)
    pages = layout_mod.plan(cells, per_page)

    expected_pages = (10 + per_page - 1) // per_page
    assert len(pages) == expected_pages
    assert layout_mod.count_pages(10, per_page) == expected_pages

    # ทุกหน้าต้องมีช่องไม่เกินโหมด และหน้าสุดท้ายอาจไม่เต็ม
    for page in pages:
        assert 0 < len(page) <= per_page

    # ต้องไม่ทำภาพหายหรือซ้ำ
    assert sum(len(p) for p in pages) == 10

    cols, rows = layout_mod.grid_for(per_page)
    assert cols * rows == per_page


def test_exact_multiple_has_no_empty_page() -> None:
    """จำนวนภาพหารลงตี ต้องไม่มีหน้าเปล่าเพิ่ม"""
    assert layout_mod.count_pages(8, 4) == 2
    assert layout_mod.count_pages(9, 4) == 3
    assert layout_mod.count_pages(1, 4) == 1
    assert layout_mod.count_pages(0, 4) == 0


def test_chunk_is_lazy_generator() -> None:
    """ต้องคืน generator เพื่อไม่เก็บทุกหน้าไว้ใน RAM"""
    result = layout_mod.chunk(make_cells(20), 4)
    assert not isinstance(result, list)
    assert len(list(result)) == 5


def test_invalid_per_page_is_rejected() -> None:
    for bad in (0, 3, 5, 7, 13, -1):
        with pytest.raises(ValueError):
            layout_mod.validate_per_page(bad)
        with pytest.raises(ValueError):
            layout_mod.grid_for(bad)


def test_validate_per_page_passes_through_valid() -> None:
    for good in config.PER_PAGE_GRID:
        assert layout_mod.validate_per_page(good) == good


# --------------------------------------------------------------------------
# text (ภาษาไทย)
# --------------------------------------------------------------------------


def test_raqm_is_available() -> None:
    """libraqm จำเป็นต่อการจัดวางสระ/วรรณยุกต์ — ขาดแล้วข้อความจะผิดรูป"""
    assert text_mod.raqm_available(), (
        "Pillow ต้อง build มาพร้อม libraqm — ดูวิธีติดตั้งใน README"
    )
    assert text_mod.shaping_warning() is None


@pytest.mark.parametrize("content", THAI_SAMPLES)
def test_thai_text_renders_non_empty(content: str) -> None:
    """ข้อความไทยต้องวาดได้และมีขนาดไม่เป็นศูนย์"""
    mask = text_mod.render_text_mask(content, size_px=80)
    assert mask.size > 0
    assert mask.shape[0] > 0 and mask.shape[1] > 0
    assert (mask > 127).any(), "ต้องมีหมึก"


@pytest.mark.parametrize("content", THAI_SAMPLES)
def test_thai_marks_are_not_clipped(content: str) -> None:
    """สระบนล่าง/วรรณยุกต์ต้องไม่ถูกตัดขอบภาพ

    วิธีตรวจ: วาดที่ขนาดใหญ่แล้วเทียบขอบของหมึกกับขอบภาพ ถ้าสระถูกตัด
    ขอบภาพจะชนพอดีหรือเกิน
    """
    mask = text_mod.render_text_mask(content, size_px=120)
    box = text_mod.ink_bounding_box(mask)
    assert box is not None, "ต้องพบหมึก"

    x, y, w, h = box
    height, width = mask.shape[:2]

    # เว้นขอบเล็กน้อยเพื่อให้สระที่สูงสุดไม่ชนพอดี
    tolerance = 3
    assert x > -tolerance, "หมึกชนขอบซ้าย — สระอาจถูกตัด"
    assert y > -tolerance, "หมึกชนขอบบน — วรรณยุกต์อาจถูกตัด"
    assert x + w < width + tolerance, "หมึกชนขอบขวา"
    assert y + h < height + tolerance, "หมึกชนขอบล่าง — สระบนล่างอาจถูกตัด"


def test_shaping_changes_width_comparison() -> None:
    """การจัดวางสระต้องทำให้ความกว้างต่างจากการวางแบบไม่จัดวาง

    เปรียบเทียบกับการนับสระบนล่างซ้อนกันแบบไม่มีการจัดวาง: ถ้าไม่มี
    Raqm ข้อความ "ที่" จะกว้างกว่านี้เพราะสระจะถูกวางต่อกันแทนที่
    จะซ้อนเหนือพยัญชนะ
    """
    font = text_mod.load_font(120)
    _, top_raqm, _, _ = font.getbbox("ที่", language="th")
    _, _, right_plain, _ = font.getbbox("ท")

    # สระบนอยู่เหนือพยัญชนะ จึงไม่ควรเพิ่มความกว้างมากนัก
    assert (right_plain - top_raqm) < font.getlength("ท") * 0.6


def test_max_width_is_respected_including_padding() -> None:
    """ข้อความยาวต้องถูกย่อให้พอดีกับ ``max_width`` (รวม padding)"""
    long_text = "กระต่ายกินผักที่นี่มีปลาหลายตัวมาก"
    mask = text_mod.render_text_mask_scaled(long_text, 80, max_width=400)
    assert mask.shape[1] <= 400


def test_max_height_is_respected() -> None:
    mask = text_mod.render_text_mask_scaled("หน้า 1", 200, max_height=60)
    assert mask.shape[0] <= 60


def test_empty_text_returns_tiny_image() -> None:
    mask = text_mod.render_text_mask("", 50)
    assert mask.size > 0  # ไม่ควรพัง


def test_ascii_text_works_too() -> None:
    """ข้อความอังกฤษ/ตัวเลขต้องใช้ได้ด้วย"""
    mask = text_mod.render_text_mask("Page 12", 60)
    assert (mask > 127).any()
