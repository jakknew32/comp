"""วัดความหนาเส้นจริงด้วย distance transform

นี่คือหัวใจของ "จัดการมาเอง" — โปรแกรมไม่เดาค่าความหนาเส้น แต่วัดจาก
ภาพจริง แล้วคำนวณว่าต้อง dilate ด้วยเคอร์เนลวงรีใหญ่แค่ไหนถึงจะได้
ความหนาเป้าหมาย

หลักการ: สำหรับเส้นที่หนา ``w`` พิกเซล ค่า distance transform สูงสุด
ที่กลางเส้นจะประมาณ ``w / 2`` ดังนั้นความหนาเฉลี่ย = 2 × ค่าเฉลี่ยของ
distance transform ที่พิกเซลหมึก (เฉลี่ยเฉพาะระดับสูงสุดในแต่ละ skeleton
เพื่อไม่ให้ก้อนทึบขนาดใหญ่ เช่น ตาของการ์ตูน ดึงค่าเฉลี่ยขึ้นมาทั้งก้อน)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass
class StrokeMeasurement:
    """ผลการวัดความหนาเส้นของภาพหนึ่งใบ"""

    measured_px: float
    target_px: float
    kernel_px: int
    ink_ratio: float
    reliable: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distance_map(binary: np.ndarray) -> np.ndarray:
    """distance transform ที่ปลอดภัยจากภาพที่ไม่มีพื้นหลัง

    ``cv2.distanceTransform`` คืนค่า ``inf`` สำหรับพิกเซลที่ไม่มีพื้นหลัง
    เลย (ภาพที่เต็มไปด้วยหมึกทั้งหมด) จึงต้องเติมกรอบพื้นหลัง 1 พิกเซล
    รอบภาพก่อนเสมอ — ทำให้ได้ค่าที่มีความหมายเสมอ
    """
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    return dist[1:-1, 1:-1]


def _ridge_values(binary: np.ndarray) -> np.ndarray:
    """ค่า distance transform ณ จุดกึ่งกลางของเส้น (ridge)

    จุดกึ่งกลางคือพิกเซลที่ค่า distance ไม่น้อยกว่าพิกเซลรอบ ๆ ทั้ง 8 ตัว
    เหตุผลที่ต้องวัดที่จุดกึ่งกลางและไม่ใช่ค่าเฉลี่ยทั้งหมด เพราะค่าเฉลี่ย
    จะถูกลากขึ้นโดยพิกเซลขอบ (distance ≈ 1) และถูกลากลงโดยก้อนทึบ
    ขนาดใหญ่ที่มีแกนกว้าง
    """
    dist = _distance_map(binary)
    kernel = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(dist, kernel)
    ridge = (dist >= local_max - 1e-6) & (binary > 0)
    values = dist[ridge]
    if values.size == 0:  # pragma: no cover - กันพลาด
        values = dist[binary > 0]
    return values


def measure_stroke_width(mask: np.ndarray) -> float:
    """วัดความหนาเส้นเฉลี่ยเป็นพิกเซลจาก binary mask (255 = หมึก)

    สำหรับเส้นหนา ``w`` พิกเซล ค่า distance transform ที่จุดกึ่งกลาง
    จะเท่ากับ ``(w + 1) / 2`` เพราะ OpenCV วัดระยะถึง **พิกเซลพื้นหลัง
    ที่ใกล้ที่สุด** (ไม่ใช่ถึงขอบ) ดังนั้นกลับค่ากลับมาได้ด้วย

    ``width = 2 × ridge_distance − 1``

    ใช้ค่ามัธยฐานแทนค่าเฉลี่ย เพื่อให้ทนต่อก้อนทึบขนาดใหญ่
    (เช่นตาของการ์ตูน) ที่ distance สูงกว่าความหนาเส้นจริงมาก

    :returns: ความหนาเฉลี่ยเป็นพิกเซล (0.0 ถ้าไม่พบหมึก)
    """
    if mask is None or mask.size == 0:
        raise ValueError("mask ว่างเปล่า")

    binary = (mask > 127).astype(np.uint8)
    if not binary.any():
        return 0.0

    values = _ridge_values(binary)
    finite = values[np.isfinite(values)]
    if finite.size == 0:  # pragma: no cover - กันพลาด
        return 0.0

    return float(np.median(finite) * 2.0 - 1.0)


def measure_median_stroke(mask: np.ndarray) -> float:
    """วัดความหนาเส้นด้วยค่ามัธยฐานของ ridge (ชื่อเดิม เก็บไว้เพื่อเข้าใจง่าย)"""
    binary = (mask > 127).astype(np.uint8)
    if not binary.any():
        return 0.0
    values = _ridge_values(binary)
    finite = values[np.isfinite(values)]
    if finite.size == 0:  # pragma: no cover - กันพลาด
        return 0.0
    return float(np.median(finite) * 2.0 - 1.0)


def _odd(n: float) -> int:
    """แปลงเป็นเลขคี่ ๆ ไม่น้อยกว่า 1 (ขนาดเคอร์เนลวงรีต้องเป็นเลขคี่)."""
    value = int(round(n))
    if value % 2 == 0:
        value += 1
    return max(1, value)


def dilate_kernel_size(target_px: float, measured_px: float) -> int:
    """คำนวณขนาดเคอร์เนลวงรีที่ต้อง dilate เพื่อให้ได้ความหนาเป้าหมาย

    การ dilate ด้วยเคอร์เนลวงรีขนาด ``k`` เพิ่มความหนาขึ้นประมาณ
    ``k - 1`` พิกเซล (ระยะจากขอบเดิมไปยังขอบใหม่) เมื่อเส้นหนาพอ
    สมควร ดังนั้น ``k ≈ (target - measured) + 1``
    """
    needed = target_px - measured_px + 1.0
    return _odd(needed)


def plan_thickness(
    mask: np.ndarray,
    target_px: float,
    *,
    max_kernel: int = 101,
) -> StrokeMeasurement:
    """วัดแล้ววางแผนทำให้เส้นหนาเป็นเป้าหมาย

    คืนค่า :class:`StrokeMeasurement` ซึ่งมีขนาดเคอร์เนลวงรีที่ควรใช้
    ``kernel_px = 1`` หมายถึง "ไม่ต้องทำอะไร ใช้ขนาดเดิม"
    """
    measured = measure_stroke_width(mask)
    ink_ratio = float((mask > 127).mean())

    if measured <= 0.0:
        return StrokeMeasurement(
            measured_px=0.0,
            target_px=float(target_px),
            kernel_px=1,
            ink_ratio=ink_ratio,
            reliable=False,
            note="ไม่พบเส้นในภาพเลย",
        )

    if target_px <= measured + 0.5:
        return StrokeMeasurement(
            measured_px=round(measured, 2),
            target_px=float(target_px),
            kernel_px=1,
            ink_ratio=ink_ratio,
            reliable=True,
            note="เส้นหนากว่าเป้าหมายอยู่แล้ว (ไม่ย่อให้บางลง เพราะจะทำให้เส้นขาด)",
        )

    kernel = dilate_kernel_size(target_px, measured)
    if kernel > max_kernel:
        return StrokeMeasurement(
            measured_px=round(measured, 2),
            target_px=float(target_px),
            kernel_px=1,
            ink_ratio=ink_ratio,
            reliable=False,
            note=(
                f"เส้นต้นฉบับบางมาก ({measured:.1f} px) "
                f"ขยายถึง {target_px:.0f} px ไม่ได้โดยไม่ทำให้เส้นเชื่อมกัน"
            ),
        )

    return StrokeMeasurement(
        measured_px=round(measured, 2),
        target_px=float(target_px),
        kernel_px=kernel,
        ink_ratio=ink_ratio,
        reliable=True,
        note=(
            f"วัดได้ {measured:.1f} px → dilate {kernel}×{kernel} "
            f"เพื่อไปให้ถึง {target_px:.0f} px"
        ),
    )


def apply_thickness(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    """dilate เส้นด้วยเคอร์เนลวงรีขนาดที่กำหนด"""
    if kernel_px <= 1:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_px, kernel_px)
    )
    return cv2.dilate(mask, kernel, iterations=1)
