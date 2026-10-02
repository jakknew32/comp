"""การทดสอบการรวม PDF (:mod:`app.book.pdf`) และการประกอบหน้า

ครอบคลุมข้อกำหนดของแผน:
* PDF ที่ได้มีจำนวนหน้า = ปก + เพดาน(ภาพ/ช่อง) และขนาดหน้าเป็น A4
* หน้า A4 เป็นภาพขาวดำที่ถูกต้อง ไม่ล้นขอบกระดาษ
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.book import cover as cover_mod  # noqa: E402
from app.book import layout as layout_mod  # noqa: E402
from app.book import pdf as pdf_mod  # noqa: E402
from app.lineart.compose import (  # noqa: E402
    Cell,
    compose_page,
    new_page,
    rounded_rect_mask,
    stamp,
)


def dummy_cell(size: int = 200, caption: str = "ทดสอบ") -> Cell:
    """ภาพลายเส้นจำลอง: วงกลมหนา (255 = หมึก)"""
    import cv2

    mask = np.full((size, size), 255, np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.35), 0, 12)
    cv2.circle(mask, (int(size * 0.4), int(size * 0.4)), 14, 0, -1)
    return Cell(mask=mask, caption=caption)


# --------------------------------------------------------------------------
# pack 1-bit
# --------------------------------------------------------------------------


def test_pack_1bit_pads_rows_to_byte_boundary() -> None:
    """แต่ละแถวต้องถูกเติมจนหารลงตัว"""
    for width in (7, 8, 9, 100, 2480):
        mask = np.zeros((3, width), np.uint8)
        mask[:, 0] = 255  # หมึกที่พิกเซลแรกของทุกแถว
        packed = pdf_mod.pack_1bit(mask)
        row_bytes = (width + 7) // 8
        assert len(packed) == 3 * row_bytes
        # MSB first → บิตแรกเป็น 1 ทุกแถว
        assert all(packed[i * row_bytes] & 0x80 for i in range(3))


def test_pack_1bit_rejects_colour_input() -> None:
    with pytest.raises(pdf_mod.PdfError):
        pdf_mod.pack_1bit(np.zeros((10, 10, 3), np.uint8))


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def test_pdf_page_count_matches_input() -> None:
    pages = [new_page() for _ in range(5)]
    data = pdf_mod.build_pdf(pages)

    assert data.startswith(b"%PDF-1.4")
    assert data.rstrip().endswith(b"%%EOF")
    assert pdf_mod.page_count(data) == 5


def test_pdf_media_box_is_a4() -> None:
    """ขนาดหน้าต้องเป็น A4 (210 × 297 มม. = 595 × 842 pt)"""
    data = pdf_mod.build_pdf([new_page()])
    expected = (
        f"/MediaBox [0 0 {pdf_mod.A4_WIDTH_PT:.2f} {pdf_mod.A4_HEIGHT_PT:.2f}]"
    ).encode("latin-1")

    assert expected in data
    assert pdf_mod.A4_WIDTH_PT == pytest.approx(595.28, abs=0.1)
    assert pdf_mod.A4_HEIGHT_PT == pytest.approx(841.89, abs=0.1)


def test_pdf_requires_at_least_one_page() -> None:
    with pytest.raises(pdf_mod.PdfError):
        pdf_mod.build_pdf([])


def test_pdf_image_dimensions_match_page() -> None:
    page = new_page()
    data = pdf_mod.build_pdf([page])
    expected = (
        f"/Width {config.A4_WIDTH_PX} /Height {config.A4_HEIGHT_PX}"
    ).encode("latin-1")
    assert expected in data


def test_pdf_xref_offsets_are_valid() -> None:
    """ตำแหน่งใน xref ต้องชี้ไปที่ object จริง"""
    data = pdf_mod.build_pdf([new_page(), new_page()])

    marker = b"startxref\n"
    index = data.rfind(marker)
    assert index != -1
    start = int(data[index + len(marker) :].split(b"%%EOF")[0].strip())

    assert data[start : start + 4] == b"xref"

    # ตรวจว่า offset แรก ๆ ชี้ไปที่ "<n> 0 obj"
    xref_body = data[start:].split(b"trailer")[0].decode("latin-1")
    lines = xref_body.strip().splitlines()
    count = int(lines[0].split()[1])
    assert count >= 3

    for line in lines[2:4]:
        offset = int(line.split()[0])
        assert b" 0 obj" in data[offset : offset + 20]


def test_pdf_stream_is_valid_zlib() -> None:
    """stream ของภาพต้องถอดได้ด้วย zlib (ไม่เสีย)"""
    mask = np.full((64, 64), 255, np.uint8)
    mask[20:30, 20:30] = 0
    packed = pdf_mod.pack_1bit(mask)
    assert zlib.decompress(zlib.compress(packed, 9)) == packed


def test_pdf_escapes_title() -> None:
    """ชื่อสมุดที่มีวงเล็บวงเหลี่ยมต้องไม่ทำให้ PDF เสีย"""
    data = pdf_mod.build_pdf([new_page()], title="หนังสือ (ฉบับพิเศษ)")
    assert b"/Title (" in data
    assert data.rstrip().endswith(b"%%EOF")


# --------------------------------------------------------------------------
# compose
# --------------------------------------------------------------------------


def test_new_page_is_a4_white() -> None:
    page = new_page()
    assert page.shape == (config.A4_HEIGHT_PX, config.A4_WIDTH_PX)
    assert page.dtype == np.uint8
    assert (page == 255).all()


def test_stamp_converts_ink_convention() -> None:
    """``stamp`` ต้องแปลง 255=หมึก ให้เป็น 0=หมึก ตามข้อตกลงของหน้า"""
    page = np.full((20, 20), 255, np.uint8)
    ink = np.zeros((5, 5), np.uint8)
    ink[1:3, 1:3] = 255

    stamp(page, 5, 5, ink)
    assert (page[6:8, 6:8] == 0).all(), "หมึกต้องกลายเป็น 0"
    assert (page[0, 0] == 255), "พื้นที่นอกหมึกต้องเป็น 255"


def test_rounded_rect_is_a_ring_not_a_block() -> None:
    """กรอบมุมมนต้องเป็นวงแหวน ภายในต้องโล่ง"""
    mask = rounded_rect_mask(200, 200, radius=20, border_width=10)

    assert mask[100, 100] == 0, "ภายในกรอบต้องโล่ง"
    assert mask[0, 100] == 255, "ขอบบนต้องมีเส้น"
    assert mask[100, 0] == 255, "ขอบซ้ายต้องมีเส้น"
    assert mask[0, 0] == 0, "มุมที่โค้งต้องไม่มีเส้น"
    assert mask[199, 199] == 0, "มุมขวาล่างต้องไม่มีเส้น"

    ring = int((mask > 0).sum())
    assert 0 < ring < mask.size * 0.5


def test_rounded_rect_border_width_is_uniform() -> None:
    """ความหนาเส้นกรอบต้องสม่ำเสมอทุกด้าน (รวมถึงมุม)"""
    width, border = 300, 12
    mask = rounded_rect_mask(width, width, radius=30, border_width=border)

    top = int((mask[0] > 0).sum())
    left = int((mask[:, 0] > 0).sum())
    assert abs(top - left) <= 2, "ความยาวเส้นด้านบนและซ้ายต่างกันมาก"

    # วัดความหนาตามแนวนอนกลางกรอบ
    row = int(np.nonzero(mask[150] > 0)[0][0])
    assert row <= 1


def test_compose_page_places_image_and_caption() -> None:
    page = compose_page(
        [dummy_cell(caption="ยิงทพ")], cols=1, rows=1, page_number=1
    )
    assert page.shape == (config.A4_HEIGHT_PX, config.A4_WIDTH_PX)
    # ต้องมีหมึก (ภาพ + กรอบ + ข้อความ)
    assert (page < 128).sum() > 0


def test_compose_page_number_can_be_disabled() -> None:
    with_number = compose_page([dummy_cell()], cols=1, rows=1, page_number=7)
    without = compose_page([dummy_cell()], cols=1, rows=1, page_number=None)
    assert not np.array_equal(with_number, without)
    assert (without < 128).sum() < (with_number < 128).sum()


def test_compose_ink_stays_inside_page() -> None:
    """หมึกต้องไม่ล้นออกขอบกระดาษ"""
    page = compose_page(
        [dummy_cell(caption="ชื่อกำกับยาวมาก ๆ ที่ควรถูกย่อให้พอดีกรอบ")],
        cols=2, rows=2, page_number=3,
    )
    assert (page[0, :] < 128).sum() == 0, "หมึกรั่วขอบบน"
    assert (page[-1, :] < 128).sum() == 0, "หมึกรั่วขอบล่าง"
    assert (page[:, 0] < 128).sum() == 0, "หมึกรั่วขอบซ้าย"
    assert (page[:, -1] < 128).sum() == 0, "หมึกรั่วขอบขวา"


def test_empty_cells_leave_blank_frame() -> None:
    """ช่องที่ไม่มีภาพต้องเหลือแต่กรอบ ไม่ใช่หน้าว่างเปล่า"""
    page = compose_page([], cols=2, rows=2)
    assert (page < 128).sum() > 0, "ต้องยังมีกรอบ"


# --------------------------------------------------------------------------
# หน้าปกและทั้งเล่ม
# --------------------------------------------------------------------------


def test_cover_has_no_page_number() -> None:
    """หน้าปกต้องไม่มีเลขหน้า"""
    cover = cover_mod.make_cover([dummy_cell()], book_title="สมุดทดสอบ", page_count=3)
    assert cover.shape == (config.A4_HEIGHT_PX, config.A4_WIDTH_PX)
    assert (cover < 128).sum() > 0


def test_cover_without_images_still_renders() -> None:
    cover = cover_mod.make_cover([], book_title="สมุดว่าง")
    assert (cover < 128).sum() > 0


def test_full_book_page_count() -> None:
    """จำนวนหน้าทั้งเล่ม = ปก + เพดาน(ภาพ/ช่อง)"""
    cells = [dummy_cell(caption=f"ภาพที่ {i + 1}") for i in range(10)]

    def pages(per_page: int, with_cover: bool = True):
        if with_cover:
            yield cover_mod.make_cover(cells, page_count=10)
        for number, group in enumerate(layout_mod.chunk(cells, per_page), 1):
            cols, rows = layout_mod.grid_for(per_page)
            yield compose_page(group, cols=cols, rows=rows, page_number=number)

    for per_page in (1, 4):
        data = pdf_mod.build_pdf(pages(per_page))
        expected = 1 + layout_mod.count_pages(10, per_page)
        assert pdf_mod.page_count(data) == expected


def test_book_pdf_is_compact() -> None:
    """หน้า A4 ที่เป็นภาพขาวดำต้องบีบอัดได้เล็ก"""
    pages = [compose_page([dummy_cell()], cols=1, rows=1, page_number=i) for i in range(3)]
    data = pdf_mod.build_pdf(pages)
    # ดิบ 3 หน้า = 3 × 2480×3508 ≈ 26 MB
    assert len(data) < 1_500_000, "PDF ใหญ่ผิดปกติสำหรับภาพลายเส้น"
