"""ทดสอบการจัดหน้าสมุดและการประกอบ PDF"""

from __future__ import annotations

import re

import numpy as np
import pytest
from PIL import Image

from app.config import (
    A4_HEIGHT_PX,
    A4_WIDTH_PX,
    MM_PER_INCH,
    BookParams,
    LineArtParams,
)
from app.book import cover, layout, pdf
from app.lineart import compose


def make_mask(width: int = 200, height: int = 260) -> np.ndarray:
    """mask ลายเส้นง่ายๆ มีกรอบและเส้นกลาง"""
    mask = np.zeros((height, width), np.uint8)
    mask[10:11, 10 : width - 10] = 255
    mask[height - 11 : height - 10, 10 : width - 10] = 255
    mask[10 : height - 10, 10:11] = 255
    mask[10 : height - 10, width - 11 : width - 10] = 255
    mask[height // 2 - 1 : height // 2 + 1, 20 : width - 20] = 255
    return mask


def ink_count(page: Image.Image) -> int:
    """นับจำนวนพิกเซลหมึกบนหน้า

    หน้าที่ได้เป็นภาพโทนเดียว (mode "1") ซึ่งเมื่อแปลงเป็น numpy
    จะกลายเป็นค่า boolean ที่ True คือสีขาว
    ถ้านับด้วย sum() ตรงๆ จะได้จำนวนพิกเซลขาว ไม่ใช่หมึก จึงต้องกลับด้าน
    """
    return int((np.asarray(page.convert("L")) < 128).sum())


# --- แผนการแบ่งหน้า ---------------------------------------------------------


@pytest.mark.parametrize(
    "count,per_page,with_cover,expected",
    [
        (10, 1, True, 11),
        (10, 1, False, 10),
        (10, 4, True, 4),
        (10, 4, False, 3),
        (3, 4, True, 2),
        (0, 1, True, 1),
        (12, 12, True, 2),
    ],
)
def test_plan_pages_page_count(count, per_page, with_cover, expected) -> None:
    book = BookParams(per_page=per_page, include_cover=with_cover)
    pages = layout.plan_pages(count, book)
    assert len(pages) == expected
    assert pages[0].is_cover == with_cover


def test_plan_pages_covers_every_image_exactly_once() -> None:
    book = BookParams(per_page=4, include_cover=True)
    pages = layout.plan_pages(17, book)
    seen = [item for page in pages for item in page.items]
    assert sorted(seen) == list(range(17))


def test_cells_per_page_matches_grid() -> None:
    for per_page, (rows, cols) in [(1, (1, 1)), (4, (2, 2)), (6, (2, 3)), (9, (3, 3))]:
        book = BookParams(per_page=per_page)
        assert layout.cells_per_page(book) == rows * cols


# --- ช่องบนหน้า -------------------------------------------------------------


@pytest.mark.parametrize("per_page", [1, 2, 4, 6, 9, 12])
def test_content_boxes_fit_inside_frame(per_page: int) -> None:
    book = BookParams(per_page=per_page, show_frame=True)
    boxes = compose.content_boxes(book)
    assert len(boxes) == layout.cells_per_page(book)
    frame = compose.frame_box()
    for box in boxes:
        assert box.x >= frame.x
        assert box.y >= frame.y
        assert box.right <= frame.right
        assert box.bottom <= frame.bottom
        assert box.w > 0 and box.h > 0


def test_content_boxes_do_not_overlap() -> None:
    boxes = compose.content_boxes(BookParams(per_page=6))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            overlap_x = min(a.right, b.right) - max(a.x, b.x)
            overlap_y = min(a.bottom, b.bottom) - max(a.y, b.y)
            assert not (overlap_x > 0 and overlap_y > 0), "ช่องภาพทับกัน"


def test_frame_toggle_removes_inset() -> None:
    with_frame = compose.content_boxes(BookParams(per_page=1, show_frame=True))
    without = compose.content_boxes(BookParams(per_page=1, show_frame=False))
    assert without[0].w > with_frame[0].w


# --- การประกอบหน้า ---------------------------------------------------------


def test_content_page_is_a4_bilevel() -> None:
    page = compose.render_content_page(
        [make_mask()], ["ทดสอบ"], BookParams(per_page=1), LineArtParams(), 1
    )
    assert page.size == (A4_WIDTH_PX, A4_HEIGHT_PX)
    assert page.mode == "1"


def test_page_number_appears_only_when_requested() -> None:
    with_num = compose.render_content_page(
        [make_mask()], ["ทดสอบ"], BookParams(show_page_number=True), LineArtParams(), 7
    )
    without = compose.render_content_page(
        [make_mask()], ["ทดสอบ"], BookParams(show_page_number=False), LineArtParams(), 7
    )
    assert ink_count(with_num) > ink_count(without)


def test_caption_appears_only_when_requested() -> None:
    with_cap = compose.render_content_page(
        [make_mask()], ["ชื่อกำกับ"], BookParams(show_caption=True), LineArtParams(), 0
    )
    without = compose.render_content_page(
        [make_mask()], ["ชื่อกำกับ"], BookParams(show_caption=False), LineArtParams(), 0
    )
    assert ink_count(with_cap) > ink_count(without)


def test_thicker_target_produces_more_ink() -> None:
    """ค่าความหนาเส้นที่มากขึ้นต้องทำให้หมึกบนหน้าเพิ่มขึ้นจริง"""
    thin = compose.render_content_page(
        [make_mask()], ["x"], BookParams(), LineArtParams(target_line_mm=1.0), 0
    )
    thick = compose.render_content_page(
        [make_mask()], ["x"], BookParams(), LineArtParams(target_line_mm=5.0), 0
    )
    assert ink_count(thick) > ink_count(thin)


def test_empty_slots_leave_blank_space() -> None:
    """ช่องที่ไม่มีภาพต้องเว้นว่าง ไม่ใช่วาดของปลอม"""
    book = BookParams(per_page=4)
    boxes = compose.content_boxes(book)
    one = np.zeros((1, 1), np.uint8)
    page = compose.render_content_page(
        [make_mask(), one, one, one], ["มี", "", "", ""], book, LineArtParams(), 1
    )
    array = np.asarray(page.convert("L"))
    box = boxes[1]
    region = array[box.y : box.bottom, box.x : box.right]
    assert region.max() == 255, "ช่องที่ 2 ควรว่างเปล่า"


# --- หน้าปก -----------------------------------------------------------------


def test_cover_is_a4_bilevel() -> None:
    page = cover.build_cover(
        [make_mask()], ["ทดสอบ"], BookParams(), LineArtParams(), total_pages=1
    )
    assert page.size == (A4_WIDTH_PX, A4_HEIGHT_PX)
    assert page.mode == "1"


def test_cover_handles_more_images_than_slots() -> None:
    masks = [make_mask() for _ in range(10)]
    page = cover.build_cover(
        masks, ["ทดสอบ"] * 10, BookParams(), LineArtParams(), total_pages=10
    )
    assert page.mode == "1"


# --- PDF --------------------------------------------------------------------


def test_pdf_page_count_and_size() -> None:
    masks = [make_mask() for _ in range(5)]
    result = pdf.build_book(
        masks, ["ทดสอบ"] * 5, BookParams(per_page=1, include_cover=True), LineArtParams()
    )
    assert result.page_count == 6  # ปก 1 + เนื้อหา 5
    assert result.pdf_bytes.startswith(b"%PDF")


def test_pdf_uses_bilevel_compression() -> None:
    result = pdf.build_book(
        [make_mask()], ["ทดสอบ"], BookParams(), LineArtParams()
    )
    # CCITT G4 เป็นการบีบอัดภาพขาวดำแบบไม่สูญเสียรายละเอียด
    # เหมาะกับภาพลายเส้นมากกว่า JPEG ที่ทำให้เส้นขรุขระ
    assert b"CCITTFaxDecode" in result.pdf_bytes


def test_pdf_media_box_is_a4_in_points() -> None:
    result = pdf.build_book([make_mask()], ["ทดสอบ"], BookParams(), LineArtParams())
    match = re.search(rb"/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]", result.pdf_bytes)
    assert match, "ไม่พบ MediaBox ใน PDF"
    width = float(match.group(1))
    height = float(match.group(2))
    # A4 = 210 x 297 มิลลิเมตร เท่ากับ 595.28 x 841.89 พอยต์
    assert width == pytest.approx(210.0 / MM_PER_INCH * 72, abs=1.0)
    assert height == pytest.approx(297.0 / MM_PER_INCH * 72, abs=1.0)


def test_pdf_without_images_still_returns_cover() -> None:
    result = pdf.build_book([], [], BookParams(), LineArtParams())
    assert result.page_count == 1
    assert result.warnings


@pytest.mark.parametrize("per_page", [1, 2, 4, 6, 9, 12])
def test_every_grid_mode_builds(per_page: int) -> None:
    book = BookParams(per_page=per_page, include_cover=True)
    masks = [make_mask() for _ in range(7)]
    result = pdf.build_book(masks, ["ทดสอบ"] * 7, book, LineArtParams())
    expected_pages = 1 + -(-7 // layout.cells_per_page(book))
    assert result.page_count == expected_pages
