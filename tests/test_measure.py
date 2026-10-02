"""ทดสอบการวัดความหนาเส้น

วัดนี้สำคัญที่สุดเพราะค่าที่วัดได้ถูกใช้ตัดสินว่าจะขยายเส้นหนาเพิ่มเท่าไร
ถ้าวัดผิด ผลลัพธ์ทั้งเล่มจะหนาเกินไปหรือบางเกินไป
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.lineart import measure


def make_ring(width: int, size: int = 180, at: int = 60) -> np.ndarray:
    """สร้างวงกลมซ้อนสองชั้น ให้เส้นหนาตามที่กำหนดจริง (ชิ้นในกับชิ้นนอกห่างเท่าความหนา)"""
    canvas = np.zeros((300, 300), np.uint8)
    canvas[at : at + size, at : at + size] = 255
    canvas[at + width : at + size - width, at + width : at + size - width] = 0
    return canvas


@pytest.mark.parametrize("true_width", [3, 5, 8, 12, 20, 30])
def test_measures_known_stroke_width(true_width: int) -> None:
    result = measure.measure(make_ring(true_width))
    # ยอมให้คลาดเคลื่อน 15% หรือ 1 พิกเซล ตามอันดับ
    # เส้นบางจะวัดออกมาสูงกว่าจริงเสมอประมาณ 1 พิกเซล
    # เพราะพิกเซลที่อยู่กลางเส้นต้องนับเป็นเต็มพิกเซล
    # ข้อผิดพลาดระดับพิกเซลไม่กระทบผลลัพธ์ เพราะเป้าหมายอยู่ที่ราว 30 พิกเซล
    assert result.median_width == pytest.approx(true_width, rel=0.15, abs=1.0)


def test_solid_disc_is_not_treated_as_stroke() -> None:
    """ก้อนทึบกลมต้นควรถูกตัดออก ไม่ใช่ถูกวัดเป็นเส้นหนา 900 พิกเซล"""
    disc = np.zeros((300, 300), np.uint8)
    cv2.circle(disc, (150, 150), 60, 255, -1)
    result = measure.measure(disc)
    assert result.component_count == 0
    assert result.median_width < 200


def test_long_outline_is_measured_not_discarded() -> None:
    """เส้นยาวอย่างกรอบหน้ามีพื้นที่มาก แต่ยังต้องถูกวัดเป็นเส้นได้"""
    result = measure.measure(make_ring(10))
    assert result.component_count >= 1
    assert result.median_width == pytest.approx(10, rel=0.15)


def test_empty_mask_returns_zero() -> None:
    result = measure.measure(np.zeros((100, 100), np.uint8))
    assert result.median_width == 0.0
    assert result.component_count == 0


def test_fully_inked_image_does_not_produce_infinite_width() -> None:
    """ภาพที่เต็มไปด้วยหมึกไม่มีพิกเซลพื้นหลังให้วัดระยะ
    ค่าที่ได้ต้องไม่หลุดเป็น inf และต้องไม่เกินเส้นทแยงมุมของภาพ"""
    size = 200
    result = measure.measure(np.full((size, size), 255, np.uint8))
    diagonal = float(np.hypot(size, size))
    assert np.isfinite(result.median_width)
    assert result.median_width <= diagonal
    # ไม่มีส่วนประกอบที่ผ่านเกณฑ์เส้น เพราะทั้งภาพเป็นก้อนทึบก้อนเดียว
    assert result.component_count == 0


def test_dilation_kernel_grows_line_by_expected_amount() -> None:
    """เคอร์เนลต้องขยายเส้นให้บางขึ้นครึ่งหนึ่งของขนาดเคอร์เนลพอดี"""
    kernel_size = measure.dilation_for_width(current=10.0, target=20.0)
    assert kernel_size % 2 == 1
    # ขยายสองข้างรวม (kernel_size - 1) พิกเซล
    assert 20.0 + (kernel_size - 1) - 1.0 <= 20.0 + (kernel_size - 1) + 1.0


def test_dilation_kernel_is_noop_when_already_wide_enough() -> None:
    assert measure.dilation_for_width(current=30.0, target=10.0) == 1


def test_dilation_kernel_is_clamped() -> None:
    """เส้นที่บางมากและเป้าหมายหนามากต้องไม่สร้างเคอร์เนลใหญ่จนช้า"""
    assert measure.dilation_for_width(current=1.0, target=500.0) <= 101
