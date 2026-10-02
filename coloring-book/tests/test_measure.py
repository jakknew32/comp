"""การทดสอบโมดูลวัดความหนาเส้น (:mod:`app.lineart.measure`)

ทดสอบด้วยเส้นสังเคราะห์ที่ทราบความหนาจริง ตามข้อกำหนดของแผน:
วัดได้ภายใน ±1 พิกเซล
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lineart import measure  # noqa: E402


def make_line(thickness: int, size: int = 400) -> np.ndarray:
    """สร้างภาพที่มีเส้นแนวนอนหนาเท่ากับ ``thickness`` พิกเซล (255 = หมึก)

    ใช้ ``rectangle`` แทน ``line`` เพราะ ``cv2.line`` วาดเส้นหนาให้
    เป็นจำนวนคี่และใหญ่กว่าที่ระบุเสมอ ทำให้วัดความหนาจริงไม่ได้
    """
    image = np.zeros((size, size), dtype=np.uint8)
    top = size // 2 - thickness // 2
    cv2.rectangle(
        image, (20, top), (size - 20, top + thickness - 1), 255, thickness=-1
    )
    return image


@pytest.mark.parametrize("thickness", [2, 3, 4, 5, 8, 12, 20])
def test_measure_known_thickness(thickness: int) -> None:
    """วัดเส้นที่ทราบความหนาได้ภายใน ±1 พิกเซล"""
    measured = measure.measure_stroke_width(make_line(thickness))
    assert measured == pytest.approx(thickness, abs=1.0)


def test_measure_is_robust_to_large_solid_blob() -> None:
    """ก้อนทึบขนาดใหญ่ (เช่นตาของการ์ตูน) ต้องไม่ดึงค่าเฉลี่ยขึ้น

    ภาพมีเส้นหนา 4 พิกเซลปนกับก้อนทึบกลมใหญ่มาก ค่าที่วัดได้ต้องยัง
    ใกล้เคียงความหนาเส้น ไม่ใช่รัศมีของก้อนทึบ
    """
    image = make_line(4)
    cv2.circle(image, (200, 200), 120, 255, thickness=-1)

    measured = measure.measure_stroke_width(image)
    assert measured == pytest.approx(4, abs=2.0)


def test_measure_empty_mask_returns_zero() -> None:
    assert measure.measure_stroke_width(np.zeros((50, 50), np.uint8)) == 0.0


def test_measure_all_ink_does_not_produce_infinity() -> None:
    """ภาพที่เต็มไปด้วยหมึกไม่มีพื้นหลังให้วัดระยะ

    กรณีนี้เคยทำให้ ``cv2.distanceTransform`` คืน ``inf`` ซึ่งจะทำให้
    ขั้นตอนถัดไปพัง ต้องได้ค่าที่มีความหมายแทน
    """
    measured = measure.measure_stroke_width(
        np.full((60, 60), 255, dtype=np.uint8)
    )
    assert np.isfinite(measured)


def test_dilate_kernel_size_grows_with_gap() -> None:
    """ขนาดเคอร์เนลวงรีต้องใหญ่ขึ้นเมื่อเส้นบางกว่าเป้าหมาย"""
    thin = measure.dilate_kernel_size(target_px=30, measured_px=3)
    thick = measure.dilate_kernel_size(target_px=30, measured_px=20)
    assert thin > thick
    assert thin % 2 == 1  # ขนาดเคอร์เนลวงรีต้องเป็นเลขคี่


def test_plan_thickness_reaches_target() -> None:
    """หลัง dilate ตามแผน ความหนาเส้นต้องเข้าใกล้เป้าหมาย"""
    mask = make_line(4)
    plan = measure.plan_thickness(mask, target_px=20)

    assert plan.reliable is True
    assert plan.kernel_px > 1

    dilated = measure.apply_thickness(mask, plan.kernel_px)
    measured = measure.measure_stroke_width(dilated)
    assert measured == pytest.approx(20, abs=4.0)


def test_plan_thickness_does_not_thin_existing_strokes() -> None:
    """ถ้าเส้นหนากว่าเป้าหมายอยู่แล้ว ต้องไม่ย่อให้บางลง"""
    mask = make_line(30)
    plan = measure.plan_thickness(mask, target_px=10)

    assert plan.kernel_px == 1
    result = measure.apply_thickness(mask, plan.kernel_px)
    assert np.array_equal(result, mask)


def test_plan_thickness_reports_no_ink() -> None:
    plan = measure.plan_thickness(np.zeros((40, 40), np.uint8), target_px=30)
    assert plan.reliable is False
    assert plan.measured_px == 0.0
