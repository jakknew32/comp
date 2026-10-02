"""สร้างหน้าปกของสมุดระบายสี

หน้าปกใช้องค์ประกอบเดียวกับหน้าเนื้อหา — กรอบมุมมน, ภาพย่อ, ข้อความไทย
ที่วาดด้วย Pillow+Raqm — จึงไม่มีความเสี่ยงเรื่อง shaping เหมือนกัน
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional, Sequence

import numpy as np

from ..config import DEFAULT_BOOK_TITLE
from ..lineart.compose import Cell, compose_cover


def make_cover(
    cells: Sequence[Cell],
    *,
    book_title: str = DEFAULT_BOOK_TITLE,
    page_count: int = 0,
    author: str = "",
    created: Optional[_dt.date] = None,
    frame: bool = True,
) -> np.ndarray:
    """สร้างหน้าปก

    * ชื่อสมุดขนาดใหญ่กึ่งกลางด้านบน
    * ตารางย่อภาพ 6 ภาพแรก (2×3) อยู่ในกรอบมุมมน
    * บรรทัดรอง: "ชื่อเล่ม · จำนวน N หน้า · วันที่สร้าง"
    * **ไม่มีเลขหน้า** (หน้าปกไม่นับเป็นเนื้อหา)
    """
    return compose_cover(
        cells,
        book_title=book_title.strip() or DEFAULT_BOOK_TITLE,
        page_count=page_count,
        author=author.strip(),
        created=created,
        frame=frame,
    )
