"""วางแผนการแบ่งภาพลงหน้า

ภาพแต่ละใบถูกสุ่มลง "ช่อง" ตามโหมดภาพต่อหน้าที่ผู้ใช้เลือก
ช่องที่เหลือในหน้าสุดท้ายจะเว้นว่าง (ไม่ยืดภาพให้เต็มช่อง)
"""

from __future__ import annotations

from typing import Iterator, Sequence

from ..config import PER_PAGE_GRID
from ..lineart.compose import Cell


def grid_for(per_page: int) -> tuple[int, int]:
    """คืน ``(คอลัมน์, แถว)`` ของโหมดภาพต่อหน้า

    :raises ValueError: ถ้าโหมดไม่รองรับ
    """
    if per_page not in PER_PAGE_GRID:
        supported = ", ".join(str(k) for k in sorted(PER_PAGE_GRID))
        raise ValueError(f"ภาพต่อหน้าต้องเป็นหนึ่งใน: {supported} (ได้รับ {per_page})")
    return PER_PAGE_GRID[per_page]


def validate_per_page(per_page: int) -> int:
    """ตรวจและคืนค่าโหมดที่ถูกต้อง พร้อมข้อความแนะนำโหมดที่ใกล้ที่สุด"""
    if per_page in PER_PAGE_GRID:
        return per_page
    nearest = min(PER_PAGE_GRID, key=lambda k: (abs(k - per_page), k))
    raise ValueError(
        f"ภาพต่อหน้า {per_page} ไม่รองรับ (ต้องเป็น 1, 2, 4, 6, 9 หรือ 12) "
        f"ลองใช้ {nearest}"
    )


def count_pages(item_count: int, per_page: int) -> int:
    """จำนวนหน้าเนื้อหาที่ต้องใช้ (ไม่รวมหน้าปก)"""
    per_page = validate_per_page(per_page)
    if item_count <= 0:
        return 0
    return (item_count + per_page - 1) // per_page


def chunk(cells: Sequence[Cell], per_page: int) -> Iterator[list[Cell]]:
    """แบ่งเซลล์เป็นกลุ่ม กลุ่มละหนึ่งหน้า

    คืน generator เพื่อไม่ต้องเก็บทุกหน้าไว้ในหน่วยความจำพร้อมกัน
    (ความเสี่ยงข้อ 4 — หน้า A4 300 DPI กินพื้นที่มาก)
    """
    per_page = validate_per_page(per_page)
    for start in range(0, len(cells), per_page):
        yield list(cells[start : start + per_page])


def plan(cells: Sequence[Cell], per_page: int) -> list[list[Cell]]:
    """คืนรายการของช่องแยกตามหน้า (สะดวกสำหรับเทสต์และการนับหน้า)"""
    return list(chunk(cells, per_page))
