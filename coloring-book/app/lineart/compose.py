"""ประกอบหน้ากระดาษ A4: กรอบมุมมน, grid, ชื่อกำกับ, เลขหน้า

หน้า A4 ทั้งหน้าถูกสร้างเป็นภาพขาวดำที่ 300 DPI (2480 × 3508 px) แล้ว
นำไปห่อเป็น PDF ทีหลัง วิธีนี้คุมมุมมน/grid/ข้อความไทยได้เป๊ะโดยไม่ต้อง
เขียนโค้ด PDF เลย
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..config import (
    A4_HEIGHT_PX,
    A4_WIDTH_PX,
    CAPTION_GAP_MM,
    CAPTION_HEIGHT_MM,
    CAPTION_PT,
    CELL_GAP_MM,
    CORNER_RADIUS_MM,
    COVER_SUBTITLE_PT,
    COVER_TITLE_PT,
    DEFAULT_BOOK_TITLE,
    DPI,
    FRAME_WIDTH_MM,
    MARGIN_MM,
    PAGE_NUMBER_PT,
    mm_to_px,
    pt_to_px,
)
from . import text as text_mod
from .convert import fit_into


@dataclass
class Cell:
    """ช่องหนึ่งช่องบนหน้า พร้อมภาพลายเส้นที่จะวาง"""

    mask: np.ndarray
    caption: str = ""


@dataclass
class PageGeometry:
    """ตำแหน่งของช่องทั้งหมดบนหน้า คำนวณครั้งเดียวแล้วใช้ซ้ำ"""

    x: int
    y: int
    w: int
    h: int
    image_h: int
    caption_h: int
    corner_radius: int
    frame_width: int


def new_page() -> np.ndarray:
    """สร้างหน้ากระดาษ A4 สีขาว (255 = ไม่มีหมึก)"""
    return np.full((A4_HEIGHT_PX, A4_WIDTH_PX), 255, dtype=np.uint8)


def stamp(page: np.ndarray, y: int, x: int, ink_mask: np.ndarray) -> None:
    """ประทับหมึกลงหน้ากระดาษ

    ปิดช่องรอบข้อตกลงเรื่องสีที่พลาดบ่อยที่สุดของโปรเจกต์นี้:
    ภาพลายเส้นและ mask ของข้อความใช้ **255 = หมึก** แต่หน้ากระดาษใช้
    **0 = หมึก** (เพราะเป็นภาพขาวดำที่จะส่งเข้า PDF) ถ้าวางตรง ๆ
    ภาพจะกลับสีทั้งหมด — ฟังก์ชันนี้คือจุดเดียวที่ต้องแปลง

    :param ink_mask: mask ที่ 255 = หมึก (ชิดซ้ายบนคือพิกเซล (0, 0))
    """
    if ink_mask is None or ink_mask.size == 0:
        return
    height, width = ink_mask.shape[:2]
    region = page[y : y + height, x : x + width]
    region[ink_mask > 127] = 0


def _grid_geometry(
    cols: int,
    rows: int,
    x0: int,
    y0: int,
    avail_w: int,
    avail_h: int,
    with_captions: bool,
) -> list[PageGeometry]:
    """คำนวณช่องทั้งหมดในพื้นที่ที่กำหนด

    ช่องกว้างเท่ากันทุกช่อง ความสูงเท่ากันทุกช่อง เหลือพิกเซลที่ไม่ลงตัว
    ช่องให้ชิดขวา/ล่าง
    """
    gap = mm_to_px(CELL_GAP_MM)

    cell_w = (avail_w - gap * (cols - 1)) // cols
    cell_h = (avail_h - gap * (rows - 1)) // rows

    caption_h = mm_to_px(CAPTION_HEIGHT_MM) if with_captions else 0
    caption_gap = mm_to_px(CAPTION_GAP_MM) if with_captions else 0
    image_h = cell_h - caption_h - caption_gap

    # กรอบไม่ล้ำขอบกระดาษ: ย่อกรอบให้หนึ่งในห้าของความหนา
    frame_width = mm_to_px(FRAME_WIDTH_MM)
    frame_width = max(2, min(frame_width, min(cell_w, cell_h) // 5))
    corner_radius = mm_to_px(CORNER_RADIUS_MM)
    corner_radius = max(1, min(corner_radius, min(cell_w, image_h) // 2 - 1))

    cells: list[PageGeometry] = []
    for row in range(rows):
        for col in range(cols):
            cells.append(
                PageGeometry(
                    x=x0 + col * (cell_w + gap),
                    y=y0 + row * (cell_h + gap),
                    w=cell_w,
                    h=cell_h,
                    image_h=image_h,
                    caption_h=caption_h,
                    corner_radius=corner_radius,
                    frame_width=frame_width,
                )
            )
    return cells


def compute_geometry(
    cols: int,
    rows: int,
    *,
    width_px: int = A4_WIDTH_PX,
    height_px: int = A4_HEIGHT_PX,
    with_captions: bool = True,
) -> list[PageGeometry]:
    """คำนวณตำแหน่งและขนาดช่องทุกช่องบนหน้า A4

    คืนค่าเรียงตามแถว (row-major) จำนวน ``cols * rows`` ช่อง
    """
    margin = mm_to_px(MARGIN_MM)
    return _grid_geometry(
        cols,
        rows,
        margin,
        margin,
        width_px - margin * 2,
        height_px - margin * 2,
        with_captions,
    )


def _fill_rounded_rect(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
    value: int,
) -> None:
    """เติมสี่เหลี่ยมมุมมนลง canvas ที่ตำแหน่ง (x, y) ขนาด w×h

    ประกอบจากสี่เหลี่ยมตรงกลาง + วงกลมมนสี่มุม ทำให้มุมโค้งสม่ำเสมอ
    """
    r = int(max(0, min(radius, min(w, h) // 2)))
    if r == 0:
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), value, -1)
        return

    cv2.rectangle(canvas, (x + r, y), (x + w - 1 - r, y + h - 1), value, -1)
    cv2.rectangle(canvas, (x, y + r), (x + w - 1, y + h - 1 - r), value, -1)
    for cx, cy in (
        (x + r, y + r),
        (x + w - 1 - r, y + r),
        (x + r, y + h - 1 - r),
        (x + w - 1 - r, y + h - 1 - r),
    ):
        cv2.circle(canvas, (cx, cy), r, value, -1)


def rounded_rect_mask(
    width: int, height: int, radius: int, border_width: int
) -> np.ndarray:
    """สร้างกรอบมุมมนเป็น mask (255 = เส้นกรอบ, 0 = พื้นที่ภายใน)

    เขียนเป็น "สี่เหลี่ยมมุมมนใหญ่ ลบสี่เหลี่ยมมุมมนเล็กที่ย่อเข้าไป
    ``border_width``" ทำให้ได้ความหนาเส้นคงที่ทุกด้านรวมถึงมุม
    (ถ้าใช้เส้น-ต่อ-เส้น ความหนาจะไม่เท่ากันตรงมุม)
    """
    width = max(1, width)
    height = max(1, height)
    border_width = int(max(1, min(border_width, min(width, height) // 2 - 1)))

    mask = np.zeros((height, width), dtype=np.uint8)
    _fill_rounded_rect(mask, 0, 0, width, height, radius, 255)

    inner_w = width - border_width * 2
    inner_h = height - border_width * 2
    if inner_w > 2 and inner_h > 2:
        _fill_rounded_rect(
            mask,
            border_width,
            border_width,
            inner_w,
            inner_h,
            max(1, radius - border_width),
            0,
        )
    return mask


def draw_cell(
    page: np.ndarray,
    geom: PageGeometry,
    mask: Optional[np.ndarray],
    *,
    caption: str = "",
    draw_frame: bool = True,
    draw_caption: bool = True,
    reserve_right: int = 0,
) -> None:
    """วางภาพลายเส้นหนึ่งใบลงช่อง พร้อมกรอบและชื่อกำกับ

    :param reserve_right: ความกว้างที่ต้องเว้นไว้ด้านขวาของแถบชื่อกำกับ
        (สำหรับเลขหน้า) เพื่อไม่ให้ข้อความทับกัน
    """
    # --- กรอบมุมมนรอบทั้งเซลล์ (รวมพื้นที่ชื่อกำกับ) ---
    if draw_frame:
        frame = rounded_rect_mask(
            geom.w, geom.h, geom.corner_radius, geom.frame_width
        )
        region = page[geom.y : geom.y + geom.h, geom.x : geom.x + geom.w]
        region[frame > 0] = 0

    # --- ภาพ (ย่อแบบ contain ไม่ยืดสัดส่วน) ---
    if mask is not None and mask.size > 0:
        inner_pad = geom.frame_width + 2
        box_w = max(1, geom.w - inner_pad * 2)
        box_h = max(1, geom.image_h - inner_pad)
        placed, _, _ = fit_into(mask, box_w, box_h)
        # วางชิดขวาล่างของช่องภาพ (มองเห็นชัดกว่ากลางแนวตั้งสำหรับภาพวาด)
        page[
            geom.y + inner_pad : geom.y + inner_pad + box_h,
            geom.x + inner_pad : geom.x + inner_pad + box_w,
        ] = placed

    # --- ชื่อกำกับใต้ภาพ กึ่งกลางในพื้นที่ที่เหลือจากเลขหน้า ---
    if draw_caption and caption and geom.caption_h > 0:
        size_px = pt_to_px(CAPTION_PT, DPI)
        band_x = geom.x + geom.frame_width + 2
        band_w = max(1, geom.w - geom.frame_width * 2 - 4 - reserve_right)
        caption_img = text_mod.render_text_mask_scaled(
            caption,
            size_px,
            max_width=band_w,
            max_height=geom.caption_h - 4,
        )
        if caption_img.size > 0:
            ch, cw = caption_img.shape[:2]
            cx = band_x + (band_w - cw) // 2
            cy = geom.y + geom.image_h + (geom.caption_h - ch) // 2
            stamp(page, cy, cx, caption_img)


def page_number_mask(number: int) -> np.ndarray:
    """สร้าง mask ของเลขหน้า "หน้า N" (255 = หมึก)

    เลขหน้าถูกวางไว้ *ภายใน* กรอบมุมมน ไม่ใช่ขอบกระดาษ เพื่อไม่ให้หมึกล้น
    """
    label = f"หน้า {number}"
    return text_mod.render_text_mask_scaled(label, pt_to_px(PAGE_NUMBER_PT, DPI))


def compose_page(
    cells: Sequence[Cell],
    *,
    cols: int,
    rows: int,
    page_number: Optional[int] = None,
    frame: bool = True,
    captions: bool = True,
) -> np.ndarray:
    """ประกอบหน้า A4 หนึ่งหน้าจากช่องต่าง ๆ

    :param cells: ช่องที่จะวาง (เรียง row-major) ถ้ามีน้อยกว่า
        ``cols * rows`` ช่องที่เหลือจะเว้นว่าง
    :param page_number: ถ้าเป็น ``None`` จะไม่วาดเลขหน้า (ใช้กับหน้าปก)
    """
    page = new_page()
    geometry = compute_geometry(cols, rows, with_captions=captions)

    # เลขหน้าวาดที่ช่องขวาล่างสุดของหน้า และช่องนั้นต้องเว้นที่ให้
    # ไม่เช่นนั้นชื่อกำกับจะไปชนกับเลขหน้า
    last_index = len(geometry) - 1
    num_mask: Optional[np.ndarray] = None
    if page_number is not None:
        num_mask = page_number_mask(page_number)

    for index, geom in enumerate(geometry):
        cell = cells[index] if index < len(cells) else None
        reserve = 0
        if index == last_index and num_mask is not None:
            reserve = num_mask.shape[1] + 12

        draw_cell(
            page,
            geom,
            cell.mask if cell is not None else None,
            caption=(cell.caption if cell is not None else "") or "",
            draw_frame=frame,
            draw_caption=captions,
            reserve_right=reserve,
        )

    if num_mask is not None:
        geom = geometry[last_index]
        mh, mw = num_mask.shape[:2]
        x = geom.x + geom.w - mw - geom.frame_width - 6
        y = geom.y + geom.h - mh - geom.frame_width - 6
        stamp(page, max(0, y), max(0, x), num_mask)

    return page


def compose_cover(
    cells: Sequence[Cell],
    *,
    book_title: str = DEFAULT_BOOK_TITLE,
    page_count: int = 0,
    author: str = "",
    created: Optional[_dt.date] = None,
    frame: bool = True,
    grid_cols: int = 2,
    grid_rows: int = 3,
) -> np.ndarray:
    """สร้างหน้าปก

    ประกอบด้วย
    * ชื่อสมุดขนาดใหญ่กึ่งกลางด้านบน
    * ตารางย่อภาพ 6 ภาพแรก (2×3) อยู่ในกรอบมุมมน
    * บรรทัดรอง: "ชื่อเล่ม · จำนวน N หน้า · วันที่สร้าง"
    * **ไม่มีเลขหน้า**
    """
    page = new_page()
    created = created or _dt.date.today()

    margin = mm_to_px(MARGIN_MM)
    content_w = A4_WIDTH_PX - margin * 2

    # --- ชื่อสมุด ---
    title_mask = text_mod.render_text_mask_scaled(
        book_title,
        pt_to_px(COVER_TITLE_PT, DPI),
        max_width=content_w,
        max_height=mm_to_px(30),
    )
    if title_mask.size > 0:
        th, tw = title_mask.shape[:2]
        tx = (A4_WIDTH_PX - tw) // 2
        ty = margin + mm_to_px(6)
        stamp(page, ty, tx, title_mask)
        cursor_y = ty + th + mm_to_px(8)
    else:  # pragma: no cover - ป้องกันขอบเขต
        cursor_y = margin + mm_to_px(20)

    # --- ตารางย่อภาพ ---
    thumb_cells = list(cells)[: grid_cols * grid_rows]

    # เผื่อที่ให้บรรทัดรองด้านล่าง
    subtitle_h = mm_to_px(20)
    avail_h = A4_HEIGHT_PX - cursor_y - margin - subtitle_h
    if avail_h < mm_to_px(30):  # pragma: no cover - ชื่อสมุดยาวผิดปกติ
        avail_h = mm_to_px(30)

    # ลดจำนวนแถวลงจนกว้างพอที่จะดูเป็นภาพ ไม่ใช่เส้นจิ๋ว
    rows = grid_rows
    while rows > 1:
        probe = _grid_geometry(
            grid_cols, rows, 0, 0, content_w, avail_h, with_captions=False
        )
        if min(c.image_h for c in probe) >= mm_to_px(20):
            break
        rows -= 1

    thumbs = _grid_geometry(
        grid_cols, rows, margin, cursor_y, content_w, avail_h, with_captions=False
    )

    for index, geom in enumerate(thumbs):
        cell = thumb_cells[index] if index < len(thumb_cells) else None
        draw_cell(
            page,
            geom,
            cell.mask if cell is not None else None,
            caption="",
            draw_frame=frame,
            draw_caption=False,
        )

    # --- บรรทัดรอง ---
    parts = []
    if author:
        parts.append(author)
    if page_count:
        parts.append(f"จำนวน {page_count} หน้า")
    parts.append(f"วันที่สร้าง {created.strftime('%d/%m/%Y')}")
    subtitle = " · ".join(parts)

    sub_mask = text_mod.render_text_mask_scaled(
        subtitle,
        pt_to_px(COVER_SUBTITLE_PT, DPI),
        max_width=content_w,
        max_height=mm_to_px(14),
    )
    if sub_mask.size > 0:
        sh, sw = sub_mask.shape[:2]
        sx = (A4_WIDTH_PX - sw) // 2
        sy = A4_HEIGHT_PX - margin - mm_to_px(10) - sh
        stamp(page, sy, sx, sub_mask)

    return page
