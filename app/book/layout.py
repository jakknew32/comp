"""แบ่งภาพทั้งหมดลงหน้า ตามจำนวนภาพต่อหน้าที่ผู้ใช้เลือก"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BookParams, GRID_OPTIONS
from ..lineart import compose


@dataclass
class PagePlan:
    """หนึ่งหน้าของสมุด"""

    index: int
    items: list[int]
    is_cover: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.items


def cells_per_page(book: BookParams) -> int:
    """จำนวนช่องภาพที่ใส่ได้หนึ่งหน้า"""
    rows, cols = GRID_OPTIONS.get(book.per_page, GRID_OPTIONS[1])
    return rows * cols


def plan_pages(count: int, book: BookParams, include_cover: bool | None = None) -> list[PagePlan]:
    """คืนแผนการแบ่งหน้า

    หน้าสุดท้ายอาจมีช่องว่างเหลือ ซึ่งปล่อยให้ว่างไปเฉยๆ ไม่ต้องเติมภาพปลอม
    """
    per_page = cells_per_page(book)
    want_cover = book.include_cover if include_cover is None else include_cover

    pages: list[PagePlan] = []
    page_index = 0
    if want_cover:
        pages.append(PagePlan(index=page_index, items=[], is_cover=True))
        page_index += 1

    for start in range(0, count, per_page):
        chunk = list(range(start, min(start + per_page, count)))
        pages.append(PagePlan(index=page_index, items=chunk))
        page_index += 1

    return pages


def masks_for_page(
    page: PagePlan,
    masks: list[np.ndarray],
    captions: list[str],
    book: BookParams,
    lineart,
) -> tuple[list[np.ndarray], list[str]]:
    """ดึง mask และชื่อกำกับที่ตรงกับหน้านั้น

    หน้าว่าง (เช่น ช่องที่เหลือของหน้าสุดท้าย) จะได้ค่าว่าง
    และ compose จะข้ามการวาดให้อัตโนมัติ
    """
    if page.is_cover:
        return [], []

    boxes = compose.content_boxes(book)
    page_masks: list[np.ndarray] = []
    page_captions: list[str] = []

    for slot in range(len(boxes)):
        if slot < len(page.items):
            source = page.items[slot]
            page_masks.append(masks[source])
            page_captions.append(captions[source] if source < len(captions) else "")
        else:
            page_masks.append(np.zeros((1, 1), dtype=np.uint8))
            page_captions.append("")

    return page_masks, page_captions
