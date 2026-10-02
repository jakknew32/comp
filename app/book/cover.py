"""สร้างหน้าปกของสมุดระบายสี

หน้าปกใช้ตารางย่อภาพจริงจากเล่ม ไม่ใช่ภาพตัวอย่างแยก
เพื่อให้ผู้ใช้เห็นว่าหน้าตาของสมุดหน้าตาที่จะได้จริง
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..config import (
    A4_HEIGHT_PX,
    A4_WIDTH_PX,
    COVER_GAP_MM,
    COVER_MAX_THUMBS,
    COVER_SUBTITLE_SIZE_PT,
    COVER_TITLE_SIZE_PT,
    FRAME_RADIUS_MM,
    FRAME_STROKE_MM,
    PAGE_MARGIN_MM,
    BookParams,
    LineArtParams,
    mm_to_px,
    pt_to_px,
)
from ..lineart import compose, text


def _inner_box() -> compose.Box:
    margin = mm_to_px(PAGE_MARGIN_MM)
    frame = compose.Box(margin, margin, A4_WIDTH_PX - 2 * margin, A4_HEIGHT_PX - 2 * margin)
    return frame.inset(mm_to_px(FRAME_STROKE_MM) // 2 + mm_to_px(7))


def _subtitle_line(book: BookParams, total_pages: int) -> str:
    from datetime import date

    parts = []
    if book.author:
        parts.append(book.author)
    parts.append(f"ระบายสีได้ {total_pages} หน้า")
    parts.append(date.today().strftime("%d/%m/%Y"))
    return " · ".join(parts)


def _best_grid_shape(area: compose.Box, gap: int, count: int) -> tuple[int, int]:
    """เลือกรูปร่างตารางที่ทำให้ภาพแต่ละใบใหญ่ที่สุด

    ถ้าใช้ตารางตายตัว เช่น 2×3 เสมอ จะเหลือช่องว่างทิ้งเมื่อมีภาพไม่ครบ
    เช่น 4 ภาพในตาราง 2×3 จะเหลือ 2 ช่องเปล่า หน้าปกดูไม่เป็นระเบียบ
    จึงลองทุกรูปร่างที่เป็นไปได้แล้วเลือกอันที่ช่องใหญ่ที่สุด
    """
    best_shape = (1, 1)
    best_cell = -1

    for rows in range(1, 4):
        for cols in range(1, 4):
            if rows * cols < count:
                continue
            cell = min(
                (area.w - gap * (cols - 1)) // cols,
                (area.h - gap * (rows - 1)) // rows,
            )
            if cell > best_cell:
                best_shape = (rows, cols)
                best_cell = cell

    return best_shape


def _cover_grid_boxes(
    area: compose.Box, gap: int, count: int
) -> list[compose.Box]:
    """วางช่องตารางโดยใช้ช่องทรงสี่เหลี่ยมจัตุรัส แล้วกึ่งกลางทั้งตาราง

    ถ้ายืดช่องให้เต็มพื้นที่ที่เหลือ ช่องจะสูงเกินภาพมาก
    ภาพจะตั้งอยู่กลางช่องที่สูงเปล่า ๆ ดูไม่เป็นระเบียบ
    ช่องทรงสี่เหลี่ยมจัตุรัสทำให้ตารางเรียบร้อยกว่า
    """
    if count <= 0:
        return []

    rows, cols = _best_grid_shape(area, gap, count)

    cell = min(
        (area.w - gap * (cols - 1)) // cols,
        (area.h - gap * (rows - 1)) // rows,
    )
    cell = max(1, cell)

    grid_w = cell * cols + gap * (cols - 1)
    grid_h = cell * rows + gap * (rows - 1)

    # จัดให้ตารางอยู่กลางทั้งแนวตั้งและแนวนอนของพื้นที่ที่เหลือ
    origin_x = area.x + (area.w - grid_w) // 2
    origin_y = area.y + (area.h - grid_h) // 2

    boxes: list[compose.Box] = []
    for row in range(rows):
        for col in range(cols):
            boxes.append(
                compose.Box(
                    origin_x + col * (cell + gap),
                    origin_y + row * (cell + gap),
                    cell,
                    cell,
                )
            )
    return boxes


def build_cover(
    masks: list[np.ndarray],
    captions: list[str],
    book: BookParams,
    lineart: LineArtParams,
    total_pages: int,
) -> tuple[Image.Image, list[str]]:
    """ประกอบหน้าปก 1 หน้า คืน (ภาพ, ข้อความเตือน)"""
    page = compose.new_page()

    margin = mm_to_px(PAGE_MARGIN_MM)
    frame = compose.Box(margin, margin, A4_WIDTH_PX - 2 * margin, A4_HEIGHT_PX - 2 * margin)
    compose.draw_rounded_frame(
        page, frame, mm_to_px(FRAME_STROKE_MM), mm_to_px(FRAME_RADIUS_MM)
    )

    inner = _inner_box()
    gap = mm_to_px(COVER_GAP_MM)

    # วัดความสูงจริงของหัวข้อก่อน เพราะสระบน/ล่างทำให้ตัวอักษรสูงกว่าขนาดฟอนต์
    # ถ้าคำนวณจากขนาดฟอนต์ล้วน บรรทัดรองจะไปซ้อนกับหัวข้อ
    title_mask = text.render_text(
        book.title or "สมุดระบายสี",
        pt_to_px(COVER_TITLE_SIZE_PT),
        bold=True,
        max_width=inner.w,
    )
    subtitle_mask = text.render_text(
        _subtitle_line(book, total_pages),
        pt_to_px(COVER_SUBTITLE_SIZE_PT),
        bold=False,
        max_width=inner.w,
    )

    title_top = inner.y + mm_to_px(4)
    subtitle_top = title_top + title_mask.height + gap

    grid_area = compose.Box(
        inner.x,
        subtitle_top + subtitle_mask.height + gap * 2,
        inner.w,
        inner.bottom - (subtitle_top + subtitle_mask.height + gap * 2),
    )

    # หน้าปกไม่ควรมีภาพเยอะจนดูยุ่ง เก็บไว้ไม่เกิน 9 ภาพ
    count = min(len(masks), COVER_MAX_THUMBS)
    boxes = _cover_grid_boxes(grid_area, gap, count)
    notes: list[str] = []

    # เส้นบนหน้าปกบางกว่าเนื้อหา เพื่อไม่ให้แย่งความสนใจจากชื่อสมุด
    target_line_px = mm_to_px((lineart.target_line_mm or 2.5) * 0.7)

    for index, box in enumerate(boxes):
        if index < count:
            art, note = compose.place_artwork(masks[index], box, target_line_px)
            if note and note not in notes:
                notes.append(note)
        else:
            art = Image.new("L", (box.w, box.h), compose.WHITE)
        page.paste(art, (box.x, box.y))

    center_x = inner.x + inner.w // 2
    page.paste(0, (center_x - title_mask.width // 2, title_top), title_mask)
    page.paste(0, (center_x - subtitle_mask.width // 2, subtitle_top), subtitle_mask)

    return compose.to_bilevel(page), notes


def blank_cover(book: BookParams) -> Image.Image:
    """หน้าปกว่าง ใช้เมื่อไม่มีภาพให้วาง"""
    page = compose.new_page()
    margin = mm_to_px(PAGE_MARGIN_MM)
    frame = compose.Box(margin, margin, A4_WIDTH_PX - 2 * margin, A4_HEIGHT_PX - 2 * margin)
    compose.draw_rounded_frame(
        page, frame, mm_to_px(FRAME_STROKE_MM), mm_to_px(FRAME_RADIUS_MM)
    )
    inner = _inner_box()
    title_mask = text.render_text(
        book.title or "สมุดระบายสี",
        pt_to_px(COVER_TITLE_SIZE_PT),
        bold=True,
        max_width=inner.w,
    )
    center_x = inner.x + inner.w // 2
    page.paste(
        0,
        (center_x - title_mask.width // 2, inner.y + mm_to_px(60)),
        title_mask,
    )
    return compose.to_bilevel(page)
