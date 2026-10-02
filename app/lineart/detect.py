"""ตรวจว่าภาพที่อัปโหลดเป็นภาพลายเส้นหรือภาพถ่ายจริง

ใช้เพื่อเตือนผู้ใช้ ไม่ใช่เพื่อปฏิเสธ — ทุกภาพยังผ่านการประมวลผลต่อ
เพราะโปรแกรมนี้ไม่มี AI ที่จะเปลี่ยนภาพถ่ายให้เป็นการ์ตูนน่ารักได้
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ..config import DETECT_BINARY_RATIO

# เกณฑ์ความสว่างที่ถือว่า "เกือบขาว" หรือ "เกือบดำ" คือพิกเซลที่ระบายสีทันทีไม่ได้
NEAR_WHITE = 235
NEAR_BLACK = 20

# ภาพลายเส้นที่ถ่ายจากกระดาษจะมีเงา ทำให้มีพิกเซลสีเทาปน
# จึงต้องใช้เกณฑ์ผ่อนปรนกว่าภาพดิจิทัล
BINARY_RATIO_CLEAN = DETECT_BINARY_RATIO
BINARY_RATIO_PHOTOCOPY = 0.72


@dataclass
class DetectionResult:
    is_line_art: bool
    binary_ratio: float
    mean_saturation: float
    distinct_colors: int
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def analyze(image: np.ndarray) -> DetectionResult:
    """วิเคราะห์ภาพแล้วตัดสินว่าเป็น line art หรือไม่"""
    gray = _to_gray(image)

    # วิเคราะห์บนภาพย่อ เพราะต้องการแค่สัดส่วน ไม่ต้องการความละเอียดเต็ม
    scale = 512.0 / max(gray.shape[:2])
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray
    small = cv2.GaussianBlur(small, (3, 3), 0)

    binary_ratio = float(
        np.mean((small >= NEAR_WHITE) | (small <= NEAR_BLACK))
    )

    if image.ndim == 3 and image.shape[2] >= 3:
        hsv = cv2.cvtColor(
            image if image.shape[2] == 3 else image[:, :, :3], cv2.COLOR_BGR2HSV
        )
        sat_small = cv2.resize(
            hsv[:, :, 1], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA
        )
        mean_saturation = float(np.mean(sat_small)) / 255.0
    else:
        mean_saturation = 0.0

    # จำนวนสีที่พบมาก ภาพลายเส้นจะมีไม่กี่สี (ขาว/ดำ/เทาไล่ระดับ)
    quantized = (small // 32).astype(np.uint8)
    packed = np.packbits(quantized, axis=None)
    distinct_colors = int(np.count_nonzero(np.bincount(packed, minlength=packed.size)))

    is_line_art = binary_ratio >= BINARY_RATIO_CLEAN and mean_saturation < 0.12
    photocopy = binary_ratio >= BINARY_RATIO_PHOTOCOPY and mean_saturation < 0.20

    if is_line_art:
        confidence = min(1.0, (binary_ratio - BINARY_RATIO_CLEAN) / 0.08 + 0.75)
        reason = "ภาพลายเส้นพร้อมใช้ — พื้นหลังสะอาดและเส้นชัด"
    elif photocopy:
        confidence = 0.6 + (binary_ratio - BINARY_RATIO_PHOTOCOPY) * 2.0
        reason = "น่าจะเป็นภาพลายเส้นที่ถ่ายจากกระดาษ (มีเงา/สีคลาด) — จะจัดการให้"
    else:
        confidence = min(0.95, max(0.05, binary_ratio))
        reason = (
            "ภาพนี้ดูเป็นภาพถ่ายจริง ไม่ใช่ภาพลายเส้น "
            "ผลลัพธ์จะเป็นเส้นตามภาพต้นฉบับ ไม่ใช่การ์ตูนน่ารัก"
        )

    return DetectionResult(
        is_line_art=bool(is_line_art or photocopy),
        binary_ratio=round(binary_ratio, 4),
        mean_saturation=round(mean_saturation, 4),
        distinct_colors=distinct_colors,
        confidence=round(float(min(1.0, max(0.0, confidence))), 3),
        reason=reason,
    )
