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
    """ประกอบสมุดทั้งเล่มแล้วคืนเป็น bytes ของ PDF"""
    images, warnings = build_pages(masks, captions, book, lineart)
    return _to_pdf(images, warnings)


def build_pages(
    masks: list[np.ndarray],
    captions: list[str],
    book: BookParams,
    lineart: LineArtParams,
) -> tuple[list[Image.Image], list[str]]:
    """ประกอบสมุดทั้งเล่มแล้วคืนเป็นภาพของแต่ละหน้า"""
    warnings: list[str] = []
    images = list(iter_pages(masks, captions, book, lineart, warnings))
    return images, warnings


def iter_pages(
    masks: list[np.ndarray],
    captions: list[str],
    book: BookParams,
    lineart: LineArtParams,
    warnings: list[str],
):
    """สร้างทีละหน้าแล้วส่งออกทันที ผู้เรียกแปลงและทิ้งภาพได้ก่อนสร้างหน้าถัดไป

    วิธีนี้ไม่ต้องถือภาพ A4 300 DPI ทุกหน้าไว้ในหน่วยความจำพร้อมกัน
    ซึ่งสำคัญบนเซิร์ฟเวอร์ที่หน่วยความจำจำกัด

    masks คือภาพลายเส้นที่ผ่านการประมวลผลแล้ว ยังไม่ถูกย่อหรือหนาเส้นเพิ่ม
    ขั้นตอนเหล่านั้นทำใน compose ตอนวางลงหน้า เพราะต้องรู้ขนาดช่องก่อน
    """
    if not masks:
        warnings.append("ไม่มีภาพที่ประมวลผลได้ จึงสร้างหน้าปกเปล่า")
        yield cover.blank_cover(book)
        return

    content_pages = layout.plan_pages(len(masks), book)

    for page in content_pages:
        if page.is_cover:
            total = max(0, content_pages[-1].index)
            cover_image, cover_notes = cover.build_cover(
                masks, captions, book, lineart, total_pages=total
            )
            for note in cover_notes:
                if note not in warnings:
                    warnings.append(note)
            yield cover_image
            continue

        page_masks, page_captions = layout.masks_for_page(
            page, masks, captions, book, lineart
        )
        # เลขหน้านับเฉพาะหน้าเนื้อหา ไม่นับหน้าปก
        content_number = page.index - (
            1 if content_pages and content_pages[0].is_cover else 0
        )
        image, page_notes = compose.render_content_page(
            page_masks,
            page_captions,
            book,
            lineart,
            page_number=content_number,
        )
        for note in page_notes:
            if note not in warnings:
                warnings.append(note)
        yield image


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
) -> tuple[Image.Image, list[str]]:
    """สร้างภาพตัวอย่างหน้าเดียวสำหรับหน้าเว็บ คืน (ภาพ, ข้อความเตือน)

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
