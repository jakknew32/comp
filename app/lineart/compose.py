"""ประกอบภาพลายเส้นให้เป็นหน้ากระดาษ A4

ผลลัพธ์เป็นภาพโทนเดียว (mode "1") เพื่อให้บีบอัดเป็น PDF ได้เล็กและคมชัด
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import (
    A4_HEIGHT_PX,
    A4_WIDTH_PX,
    CELL_GAP_MM,
    FRAME_RADIUS_MM,
    FRAME_STROKE_MM,
    GRID_OPTIONS,
    PAGE_MARGIN_MM,
    BookParams,
    LineArtParams,
    mm_to_px,
    pt_to_px,
)
from . import measure, text

WHITE = 255
BLACK = 0


@dataclass
class Box:
    """กรอบสี่เจ้าในหน่วยพิกเซล"""

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def inset(self, amount: int) -> "Box":
        return Box(self.x + amount, self.y + amount, self.w - 2 * amount, self.h - 2 * amount)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.right, self.bottom)


def new_page() -> Image.Image:
    return Image.new("L", (A4_WIDTH_PX, A4_HEIGHT_PX), WHITE)


def draw_rounded_frame(page: Image.Image, box: Box, stroke_px: int, radius_px: int) -> None:
    """วาดกรอบมุมมน

    ใช้ภาพแยกต่างหากแล้ววางทับ เพราะ ImageDraw ลงภาพโทนเดียวตรงๆ
    จะได้ขอบหยัก ๆ ต้องผ่านการ threshold ก่อน
    """
    layer = Image.new("L", page.size, WHITE)
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(
        box.as_tuple(),
        radius=radius_px,
        outline=BLACK,
        width=stroke_px,
    )
    page.paste(0, (0, 0), layer.point(lambda v: 255 if v < 128 else 0))


def frame_box() -> Box:
    margin = mm_to_px(PAGE_MARGIN_MM)
    return Box(margin, margin, A4_WIDTH_PX - 2 * margin, A4_HEIGHT_PX - 2 * margin)


def content_boxes(book: BookParams) -> list[Box]:
    """คำนวณช่องสำหรับภาพแต่ละใบ พร้อมเว้นที่ให้ชื่อกำกับ"""
    rows, cols = GRID_OPTIONS.get(book.per_page, GRID_OPTIONS[1])
    gap = mm_to_px(CELL_GAP_MM)

    inner = frame_box()
    if book.show_frame:
        inner = inner.inset(mm_to_px(FRAME_STROKE_MM) // 2 + mm_to_px(2))

    caption_h = 0
    if book.show_caption:
        caption_h = pt_to_px(book.caption_size_pt) + gap

    cell_w = (inner.w - gap * (cols - 1)) // cols
    cell_h = (inner.h - gap * (rows - 1)) // rows

    boxes: list[Box] = []
    for row in range(rows):
        for col in range(cols):
            x = inner.x + col * (cell_w + gap)
            y = inner.y + row * (cell_h + gap)
            art_h = max(1, cell_h - caption_h)
            boxes.append(Box(x, y, cell_w, art_h))
    return boxes


def place_artwork(
    mask: np.ndarray,
    box: Box,
    target_line_px: int,
) -> tuple[Image.Image, str | None]:
    """ย่อ/ขยายภาพลายเส้นให้พอดีช่อง แล้วปรับความหนาเส้นให้ได้ตามเป้าหมาย

    คืน (ภาพ, ข้อความเตือน) โดยข้อความเตือนจะมีเมื่อปรับความหนาไม่สำเร็จ
    เช่น ถ้าเส้นต้นฉบับหนามากจนย่อแล้วรายละเอียดหาย
    ผู้ใช้ต้องได้รู้ว่าค่าที่สั่งไม่ได้ถูกนำไปใช้จริง
    """
    if mask.size == 0 or box.w <= 0 or box.h <= 0:
        return Image.new("L", (max(1, box.w), max(1, box.h)), WHITE), None

    height, width = mask.shape[:2]
    if height == 0 or width == 0:
        return Image.new("L", (max(1, box.w), max(1, box.h)), WHITE), None

    scale = min(box.w / width, box.h / height)
    new_w = max(1, min(box.w, round(width * scale)))
    new_h = max(1, min(box.h, round(height * scale)))

    if scale < 1.0:
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # ภาพลายเส้นเป็นภาพโทนเดียว การขยายด้วยอินเตอร์โพเลชันแบบ smooth
        # จะเกิด ringing รอบเส้น แล้วพอ threshold ทั้งหน้าจะกลายเป็นรอยหยัก
        # ใช้ nearest แล้วค่อยหนาเส้นทีหลังจะได้ขอบคมที่สุด
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized = np.where(resized >= 128, 255, 0).astype(np.uint8)

    # ปรับความหนาเส้นให้ตรงเป้าหมายที่หน้ากระดาษจริง ทั้งหนาขึ้นและบางลง
    # ทำหลัง resize เสมอ เพราะถ้าทำก่อนแล้วค่อยขยาย เส้นจะกลายเป็นหย่นราว
    adjusted = measure.rescale_stroke_width(resized, target_line_px)
    resized = np.where(adjusted.mask >= 128, 255, 0).astype(np.uint8)

    # เขียนกลับเป็นภาพขาวหลัง หมึกดำ
    rgb = np.full((new_h, new_w), WHITE, dtype=np.uint8)
    rgb[resized > 0] = BLACK
    art = Image.fromarray(rgb, mode="L")

    canvas = Image.new("L", (box.w, box.h), WHITE)
    canvas.paste(art, ((box.w - new_w) // 2, (box.h - new_h) // 2))
    return canvas, adjusted.note


def render_content_page(
    masks: list[np.ndarray],
    captions: list[str],
    book: BookParams,
    lineart: LineArtParams,
    page_number: int,
) -> tuple[Image.Image, list[str]]:
    """ประกอบหน้าเนื้อหาหนึ่งหน้า คืน (ภาพ, ข้อความเตือน)

    masks ต้องมีจำนวนเท่ากับจำนวนช่องที่ว่างอยู่ในหน้านั้น
    """
    page = new_page()
    boxes = content_boxes(book)
    notes: list[str] = []

    if book.show_frame:
        draw_rounded_frame(
            page,
            frame_box(),
            mm_to_px(FRAME_STROKE_MM),
            mm_to_px(FRAME_RADIUS_MM),
        )

    target_line_px = mm_to_px(lineart.target_line_mm or 2.5)

    for index, box in enumerate(boxes):
        if index >= len(masks):
            break
        art, note = place_artwork(masks[index], box, target_line_px)
        page.paste(art, (box.x, box.y))
        if note and note not in notes:
            notes.append(note)

        if book.show_caption:
            caption = captions[index] if index < len(captions) else ""
            text.paste_text(
                page,
                caption,
                center_x=box.x + box.w // 2,
                top_y=box.bottom + mm_to_px(2),
                px_size=pt_to_px(book.caption_size_pt),
                bold=False,
                max_width=box.w,
            )

    if book.show_page_number and page_number > 0:
        label = f"หน้า {page_number}"
        # วางไว้ภายในกรอบเสมอ ไม่งั้นจะทับเส้นกรอบและถูกตัดตอนพิมพ์
        inset = mm_to_px(FRAME_STROKE_MM) + mm_to_px(2)
        area = frame_box().inset(inset)
        mask_img = text.render_text(
            label,
            pt_to_px(12),
            bold=False,
            max_width=max(1, area.w),
        )
        page.paste(0, (area.right - mask_img.width, area.bottom - mask_img.height), mask_img)

    return to_bilevel(page), notes


def to_bilevel(page: Image.Image) -> Image.Image:
    """แปลงเป็นภาพขาวดำล้วนโดยไม่ใช้ dither

    ภาพลายเส้นต้องคมชัด dithering จะทำให้พื้นขาวกลายเป็นจุดเสีย
    """
    return page.convert("1", dither=Image.Dither.NONE)
