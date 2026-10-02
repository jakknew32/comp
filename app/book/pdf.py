"""รวมหน้าทั้งหมดเป็นไฟล์ PDF เดียว

ใช้ตัวเขียน PDF ของ Pillow ซึ่งเข้ารหัสภาพโทนเดียวด้วย CCITT Group 4
เหมาะกับภาพลายเส้นมาก เพราะไม่สูญเสียรายละเอียดและไฟล์เล็ก
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..config import DPI, BookParams, LineArtParams
from ..lineart import compose
from . import cover, layout


@dataclass
class BookResult:
    pdf_bytes: bytes
    page_count: int
    warnings: list[str] = field(default_factory=list)


def build_book(
    masks: list[np.ndarray],
    captions: list[str],
    book: BookParams,
    lineart: LineArtParams,
) -> BookResult:
    """ประกอบสมุดทั้งเล่มแล้วคืนเป็น bytes ของ PDF

    masks คือภาพลายเส้นที่ผ่านการประมวลผลแล้ว ยังไม่ถูกย่อหรือหนาเส้นเพิ่ม
    ขั้นตอนเหล่านั้นทำใน compose ตอนวางลงหน้า เพราะต้องรู้ขนาดช่องก่อน
    """
    warnings: list[str] = []

    if not masks:
        warnings.append("ไม่มีภาพที่ประมวลผลได้ จึงสร้างหน้าปกเปล่า")
        pages = [cover.blank_cover(book)]
        return _to_pdf(pages, warnings)

    content_pages = layout.plan_pages(len(masks), book)
    images: list[Image.Image] = []

    for page in content_pages:
        if page.is_cover:
            total = max(0, content_pages[-1].index)
            images.append(
                cover.build_cover(masks, captions, book, lineart, total_pages=total)
            )
            continue

        page_masks, page_captions = layout.masks_for_page(
            page, masks, captions, book, lineart
        )
        # เลขหน้านับเฉพาะหน้าเนื้อหา ไม่นับหน้าปก
        content_number = page.index - (1 if content_pages and content_pages[0].is_cover else 0)
        images.append(
            compose.render_content_page(
                page_masks,
                page_captions,
                book,
                lineart,
                page_number=content_number,
            )
        )

    return _to_pdf(images, warnings)


def _to_pdf(images: list[Image.Image], warnings: list[str]) -> BookResult:
    if not images:
        raise ValueError("ไม่มีหน้าให้เขียน PDF")

    buffer = io.BytesIO()
    first, rest = images[0], images[1:]
    first.save(
        buffer,
        format="PDF",
        save_all=True,
        # ต้องเป็น list เสมอ ถ้าส่ง None ตัวเขียน PDF ของ Pillow จะพัง
        append_images=rest,
        resolution=float(DPI),
    )
    return BookResult(pdf_bytes=buffer.getvalue(), page_count=len(images), warnings=warnings)


def render_preview(
    mask: np.ndarray,
    caption: str,
    book: BookParams,
    lineart: LineArtParams,
) -> Image.Image:
    """สร้างภาพตัวอย่างหน้าเดียวสำหรับหน้าเว็บ

    ใช้โหมดเต็มหน้าเสมอ ไม่ว่าที่ผู้ใช้เลือกโหมดกี่ภาพต่อหน้า
    เพราะเป้าหมายของตัวอย่างคือดูผลของการปรับความหนาเส้น
    """
    preview_book = BookParams(
        title=book.title,
        per_page=1,
        show_frame=book.show_frame,
        show_caption=book.show_caption,
        show_page_number=book.show_page_number,
        include_cover=False,
        caption_size_pt=book.caption_size_pt,
    )
    return compose.render_content_page(
        [mask], [caption], preview_book, lineart, page_number=0
    )
