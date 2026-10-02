"""ทดสอบการวาดข้อความไทย

เรื่องนี้เป็นจุดเสี่ยงที่สุดของโปรแกรม เพราะถ้าฟอนต์หรือการจัดวางสระผิด
ผู้ใช้จะเห็นข้อความที่อ่านไม่ออกโดยที่โปรแกรมไม่ได้แจ้ง error
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.lineart import text

# ประโยคที่มีครบทั้งสระบน สระล่าง และวรรณยุกต์
THAI_SAMPLES = [
    "กระต่ายกินผัก",
    "ที่นี่มีปลา",
    "ยีระแหน",
    "ถ่ายจากกระดาษ",
    "สมุดระบายสีของฉัน",
]


def test_font_setup_is_valid() -> None:
    """ฟอนต์ต้องครอบคลุมทั้งไทย ละติน และตัวเลข

    เคยพลาดเพราะใช้ Noto Sans Thai ซึ่งไม่มีตัวเลข
    ทำให้เลขหน้าออกมาเป็นกล่องสี่เหลี่ยมโดยไม่มี error
    """
    text.check_setup()


def test_no_missing_glyphs() -> None:
    missing = text.missing_glyphs(text._COVERAGE_PROBE)
    assert missing == [], f"ฟอนต์ขาดอักขระ: {missing}"


@pytest.mark.parametrize("sample", THAI_SAMPLES)
def test_thai_text_renders_with_ink(sample: str) -> None:
    mask = text.render_text(sample, 48, bold=True)
    assert mask.width > 0 and mask.height > 0
    assert np.asarray(mask).max() > 0, "วาดแล้วไม่มีหมึกเลย"


@pytest.mark.parametrize("sample", THAI_SAMPLES)
def test_thai_marks_are_not_clipped(sample: str) -> None:
    """สระบน/ล่างและวรรณยุกต์ต้องไม่ถูกตัดขอบ

    ถ้าถูกตัดจะเห็นเป็นจุดที่ขาดหายตรงหัวหรือท้ายคำ
    """
    mask = text.render_text(sample, 64, bold=True)
    array = np.asarray(mask)
    rows = np.where(array.max(axis=1) > 128)[0]
    assert rows.size > 0
    # ต้องมีขอบเว้นอย่างน้อยเล็กน้อยทั้งด้านบนและล่าง
    assert rows.min() >= 2, "ข้อความถูกตัดด้านบน (สระหรือวรรณยุกต์หาย)"
    assert rows.max() <= array.shape[0] - 3, "ข้อความถูกตัดด้านล่าง (สระล่างหาย)"

    cols = np.where(array.max(axis=0) > 128)[0]
    assert cols.min() >= 1
    assert cols.max() <= array.shape[1] - 2


def test_digits_render_as_digits_not_boxes() -> None:
    """"หน้า 12" ต้องแสดงเลขจริง ไม่ใช่กล่องสี่เหลี่ยม"""
    assert "12" not in text.missing_glyphs("12")
    assert text.render_text("หน้า 12", 48).width > 0


def test_fit_px_size_shrinks_to_fit() -> None:
    size = text.fit_px_size("กระต่ายกินผักอยู่ในสวน", max_width=120, start_px=80)
    width, _ = text.measure_text("กระต่ายกินผักอยู่ในสวน", size)
    assert width <= 120
    assert size < 80


def test_render_respects_max_width() -> None:
    mask = text.render_text("กระต่ายกินผักอยู่ในสวน", 80, max_width=150)
    assert mask.width <= 150


def test_paste_text_centres_and_reports_size() -> None:
    page = Image.new("L", (600, 200), 255)
    width, height = text.paste_text(
        page, "ยีระแหน", center_x=300, top_y=20, px_size=40
    )
    assert width > 0 and height > 0
    array = np.asarray(page)
    cols = np.where(array.min(axis=0) < 128)[0]
    # ข้อความต้องอยู่กลางแนวนอน คือซ้ายและขวาห่างกันพอสมควร
    left_gap = cols.min()
    right_gap = 600 - cols.max()
    assert abs(left_gap - right_gap) < 6


def test_empty_text_is_noop() -> None:
    page = Image.new("L", (100, 100), 255)
    assert text.paste_text(page, "", center_x=50, top_y=10, px_size=20) == (0, 0)
    assert np.asarray(page).min() == 255
