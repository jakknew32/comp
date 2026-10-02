"""ตรวจว่าภาพที่อัปโหลดเป็นภาพลายเส้นจริงหรือภาพถ่าย

โปรแกรมนี้รองรับ **เฉพาะภาพลายเส้น** ภาพถ่ายจริงจะผ่านการกรอง
แล้วได้ "เส้นตามภาพ" ซึ่งไม่ใช่การ์ตูนน่ารัก ตัวตรวจนี้มีหน้าที่บอก
หน้าเว็บว่าจะต้องเตือนผู้ใช้หรือไม่
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

#: พิกเซลที่ถือว่า "เกือบขาว" หรือ "เกือบดำ"
NEAR_WHITE = 233  # >8% ไม่นับว่าเกือบขาว
NEAR_BLACK = 23  # <8%

#: เกณฑ์: ถ้า binary ratio สูงกว่านี้ถือว่าเป็น line art
BINARY_RATIO_THRESHOLD = 0.90

#: จำนวนสีที่ยังถือว่า "ไม่มีสี"
MAX_MEAN_SATURATION = 40.0

#: เกณฑ์ความละเอียดต่ำกว่านี้ (px) ต้องเตือนว่าจะเบลอเมื่อขยายเป็น 300 DPI
LOW_RES_PX = 1500


@dataclass
class DetectionResult:
    """ผลการตรวจประเภทภาพ ใช้แสดงสถานะรายภาพใน UI"""

    is_line_art: bool
    is_photo: bool
    binary_ratio: float
    mean_saturation: float
    distinct_colors: int
    long_edge_px: int
    low_resolution: bool
    ink_ratio: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze(bgr: np.ndarray) -> DetectionResult:
    """วิเคราย์ภาพแล้วตัดสินว่าเป็น line art หรือภาพถ่าย"""
    if bgr is None or bgr.size == 0:
        raise ValueError("ภาพว่างเปล่า")

    # รองรับทั้งภาพสี (BGR) และภาพเทา
    if bgr.ndim == 2:
        gray = bgr
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    long_edge = max(h, w)

    # สัดส่วนพิกเซลที่ "เกือบขาว/เกือบดำ" — ลายเส้นเกือบทั้งภาพจะเป็นสองค่านี้
    extreme = (gray <= NEAR_BLACK) | (gray >= NEAR_WHITE)
    binary_ratio = float(extreme.mean())

    # สี: วัด saturation เฉลี่ย (ภาพเทา/ขาวดำ -> 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())

    # จำนวนสีที่พบมาก (quantize เพื่อไม่ให้ noise ทำให้นับเกินจริง)
    small = cv2.resize(bgr, (min(w, 128), min(h, 128)), interpolation=cv2.INTER_AREA)
    quantized = (small // 32).astype(np.uint8)
    colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
    total = int(counts.sum())
    # นับเฉพาะสีที่พบมากพอจะเป็น "สีของภาพ" (อย่างน้อย 0.5% ของภาพ)
    significant = int((counts >= total * 0.005).sum())

    # สัดส่วนหมึก (พิกเซลที่มืดจริง) ใช้เป็นข้อมูลประกอบ
    ink_ratio = float((gray < 128).mean())

    is_line_art = (
        binary_ratio > BINARY_RATIO_THRESHOLD
        and mean_saturation < MAX_MEAN_SATURATION
    )

    if is_line_art:
        reason = "พบภาพลายเส้น (พิกเซลส่วนใหญ่เป็นขาวหรือดำ สีน้อย)"
    else:
        if mean_saturation >= MAX_MEAN_SATURATION:
            reason = f"ภาพมีสีค่อนข้างมาก (ความอิ่มสีเฉลี่ย {mean_saturation:.0f})"
        else:
            reason = (
                f"พิกเซลเกือบขาว/ดำมีเพียง {binary_ratio * 100:.1f}% "
                f"(ต้องมากกว่า {BINARY_RATIO_THRESHOLD * 100:.0f}%)"
            )
        reason += " — อาจเป็นภาพถ่าย โปรแกรมจะได้เส้นตามภาพ ไม่ใช่ภาพลายเส้น"

    return DetectionResult(
        is_line_art=is_line_art,
        is_photo=not is_line_art,
        binary_ratio=binary_ratio,
        mean_saturation=mean_saturation,
        distinct_colors=significant,
        long_edge_px=int(long_edge),
        low_resolution=bool(long_edge < LOW_RES_PX),
        ink_ratio=ink_ratio,
        reason=reason,
    )
