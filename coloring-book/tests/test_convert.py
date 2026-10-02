"""การทดสอบ pipeline แปลงภาพลายเส้น (:mod:`app.lineart.convert`)

ครอบคลุมข้อกำหนดของแผน:
* ภาพ line art สังเคราะห์ → พื้นหลังขาวสะอาด (ไม่มีพิกเซลเทาค้างเกิน 10%)
* ความหนาเส้นหลังประมวลผลตรงเป้าหมาย
* ก้อนทึบขนาดใหญ่ (ตา/ปากของการ์ตูน) ต้องไม่หายไป
* ปล่อยภาพจริง 3 แบบจาก samples/ ต้องผ่านทุกตัวโดยไม่มี error
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.lineart import convert, imgio, measure  # noqa: E402
from app.lineart import detect  # noqa: E402

SAMPLES = ROOT / "samples"

#: โหลดภาพตัวอย่างครั้งเดียว ใช้ร่วมกันทุกเทสต์ (สร้างภาพตัวอย่างก่อนถ้ายังไม่มี)
IN_SCOPE_SAMPLES = [
    "lineart-clean.png",
    "lineart-photographed.png",
    "lineart-thick-irregular.png",
]


@pytest.fixture(scope="module", autouse=True)
def ensure_samples() -> None:
    if not (SAMPLES / "lineart-clean.png").exists():
        sys.path.insert(0, str(SAMPLES))
        import make_samples

        make_samples.write_all(SAMPLES)


def synthetic_lineart(size: int = 800, thickness: int = 3) -> np.ndarray:
    """ภาพลายเส้นสังเคราะห์: เส้นบางสม่ำเสมอง + ก้อนทึบกลม 2 ก้อน"""
    image = np.full((size, size), 255, np.uint8)
    c = size // 2

    cv2.circle(image, (c, int(size * 0.4)), int(size * 0.25), 0, thickness)
    cv2.line(
        image, (int(size * 0.1), int(size * 0.75)),
        (int(size * 0.9), int(size * 0.75)), 0, thickness,
    )
    # ก้อนทึบขนาดใหญ่ = ตาของการ์ตูน ต้องไม่หาย
    cv2.circle(image, (int(size * 0.4), int(size * 0.4)), int(size * 0.06), 0, -1)
    cv2.circle(image, (int(size * 0.6), int(size * 0.4)), int(size * 0.06), 0, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_background_is_clean_white() -> None:
    """พื้นหลังต้องสะอาด ไม่มีพิกเซลเทาค้างเกิน 10%"""
    result = convert.convert_to_lineart(synthetic_lineart(), output_width_px=1000)

    # นับพิกเซลที่อยู่ระหว่าง 40–215 (เทากลาง ๆ) ต้องน้อยมาก
    grayish = np.logical_and(result.mask > 40, result.mask < 215)
    assert grayish.mean() < 0.10


def test_stroke_reaches_target_width() -> None:
    """ความหนาเส้นหลังประมวลผลต้องตรงเป้าหมายบนหน้ากระดาษ"""
    target_mm = 2.0
    output_width = 1200

    result = convert.convert_to_lineart(
        synthetic_lineart(),
        config.ConvertParams(target_line_mm=target_mm),
        output_width_px=output_width,
    )

    # แปลงความหนาที่วัดได้ (ในพิกเซลภาพต้นฉบับ) กลับไปเป็นหน้ากระดาษ
    scale = output_width / result.mask.shape[1]
    on_page = measure.measure_stroke_width(result.mask) * scale
    expected = config.mm_to_px(target_mm, config.DPI)

    assert on_page == pytest.approx(expected, rel=0.25)


def test_solid_blobs_survive() -> None:
    """ก้อนทึบขนาดใหญ่ (ตา) ต้องไม่หายไประหว่างกระบวนการ

    นี่คือความเสี่ยงข้อ 2 ของแผน
    """
    result = convert.convert_to_lineart(synthetic_lineart(), output_width_px=1000)

    binary = (result.mask > 127).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    assert num - 1 >= 2, "ควรมี component อย่างน้อย 2 ก้อน"
    # ก้อนทึบที่ใหญ่ที่สุดต้องมีพื้นที่มากกว่า 1% ของภาพ
    assert areas[0] > result.mask.size * 0.01


def test_despeckle_removes_small_noise() -> None:
    """จุดรบกวนเล็ก ๆ ต้องถูกตัดเมื่อเพิ่มความไวต่อจุดรบกวน"""
    image = synthetic_lineart()
    # โปรยจุดสีขาวเล็ก ๆ ทั่วภาพ
    rng = np.random.default_rng(7)
    for _ in range(400):
        x, y = rng.integers(0, image.shape[1]), rng.integers(0, image.shape[0])
        cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), -1)

    low = convert.convert_to_lineart(
        image, config.ConvertParams(despeckle=0.0), output_width_px=1000
    )
    high = convert.convert_to_lineart(
        image, config.ConvertParams(despeckle=1.0), output_width_px=1000
    )

    def speck_count(mask: np.ndarray) -> int:
        binary = (mask > 127).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        return num - 1

    assert speck_count(high.mask) <= speck_count(low.mask)


def test_target_line_mm_is_honoured() -> None:
    """ค่า target_line_mm ต้องส่งผลต่อขนาดเคอร์เนลวงรีจริง"""
    thin = convert.convert_to_lineart(
        synthetic_lineart(), config.ConvertParams(target_line_mm=1.2),
        output_width_px=1000,
    )
    thick = convert.convert_to_lineart(
        synthetic_lineart(), config.ConvertParams(target_line_mm=4.5),
        output_width_px=1000,
    )
    assert thick.auto_values["dilate_kernel"] > thin.auto_values["dilate_kernel"]


def test_auto_values_reported_when_null() -> None:
    """ส่ง null → ระบบต้องตัดสินใจเองและรายงานค่าที่ใช้จริง"""
    result = convert.convert_to_lineart(
        synthetic_lineart(), config.ConvertParams(target_line_mm=None)
    )
    values = result.auto_values
    assert values["target_line_mm"] == config.TARGET_LINE_MM
    assert "dilate_kernel" in values
    assert "measured_line_px" in values
    assert "method" in values


def test_crop_removes_surrounding_whitespace() -> None:
    """ภาพต้องถูกครอปจนหมึกชิบขอบ (เว้นเล็กน้อย)"""
    image = synthetic_lineart(size=900)
    # ใส่ขอบขาวกว้าง ๆ
    padded = np.full((1100, 1100), 255, np.uint8)
    padded[100:1000, 100:1000] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padded = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    result = convert.convert_to_lineart(padded)
    binary = (result.mask > 127).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    assert xs.min() <= 2, "ควรชิบขอบซ้าย"
    assert xs.max() >= binary.shape[1] - 3, "ควรชิบขอบขวา"


def test_fit_into_preserves_aspect_and_pads_white() -> None:
    """``fit_into`` ต้องคงสัดส่วนและเติมพื้นที่ว่างด้วยสีขาว (ไม่มีหมึก)"""
    mask = np.zeros((100, 200), np.uint8)
    mask[40:60, 20:180] = 255  # เส้นแนวนอน 255 = หมึก

    placed, _, _ = convert.fit_into(mask, 400, 400)
    assert placed.shape == (400, 400)

    # ต้องยังพบหมึก และต้องมีพื้นที่ว่าง (255) รอบ ๆ
    assert (placed < 128).any()
    assert (placed == 255).any()
    # พื้นที่ว่างต้องอยู่ด้านบน/ล่าง (ภาพกว้างกว่าสูง)
    assert placed[0, 200] == 255


@pytest.mark.parametrize("name", IN_SCOPE_SAMPLES)
def test_sample_images_process_without_error(name: str) -> None:
    """ภาพตัวอย่างทุกแบบในขอบเขตต้องประมวลผลผ่าน"""
    path = SAMPLES / name
    bgr, decoded = imgio.load_bgr(path)

    assert decoded == name
    result = convert.convert_to_lineart(bgr, output_width_px=1000)

    assert result.mask.size > 0
    assert (result.mask > 127).any(), "ต้องมีเส้นหลังประมวลผล"
    assert isinstance(result.warnings, list)
    assert result.detection.long_edge_px > 0


def test_clean_sample_is_detected_as_lineart() -> None:
    """ภาพลายเส้นสะอาดต้องถูกตรวจจับเป็น line art"""
    bgr, _ = imgio.load_bgr(SAMPLES / "lineart-clean.png")
    result = detect.analyze(bgr)

    assert result.is_line_art is True
    assert result.is_photo is False


def test_photographed_sample_is_flagged_as_photo() -> None:
    """ภาพถ่ายมีไล่แสง → ต้องถูกตรวจจับว่าไม่ใช่ line art และเตือนผู้ใช้"""
    bgr, _ = imgio.load_bgr(SAMPLES / "lineart-photographed.png")
    result = detect.analyze(bgr)

    assert result.is_photo is True
    assert result.is_line_art is False
    assert result.reason


def test_color_photo_is_flagged_as_photo() -> None:
    """ภาพสีต้องถูกจับว่าเป็นภาพถ่าย (นอกขอบเขตของโปรแกรม)"""
    bgr, _ = imgio.load_bgr(SAMPLES / "photo-color.png")
    result = detect.analyze(bgr)

    assert result.is_photo is True
    assert result.mean_saturation > detect.MAX_MEAN_SATURATION


def test_photographed_sample_keeps_drawing_on_white_background() -> None:
    """ภาพถ่ายที่มีไล่แสงต้องได้เส้นบนพื้นขาว ไม่ใช่พื้นหลังดำทั้งภาพ

    เคยพบบั๊กตอนสูตร XDoG ทำให้พื้นหลังทั้งภาพกลายเป็นเส้น
    """
    bgr, _ = imgio.load_bgr(SAMPLES / "lineart-photographed.png")
    result = convert.convert_to_lineart(bgr, output_width_px=1000)

    ink_ratio = float((result.mask > 127).mean())
    assert ink_ratio < 0.5, "พื้นหลังต้องไม่กลายเป็นหมึกทั้งภาพ"


def test_pipeline_never_crashes_on_degenerate_input() -> None:
    """ภาพเสีย/แปลกประเภทต้องไม่ทำให้ระบบพัง"""
    # ภาพขาวล้วน — ไม่มีเส้นเลย
    blank = np.full((300, 300, 3), 255, np.uint8)
    result = convert.convert_to_lineart(blank)
    assert result.mask.size > 0

    # ภาพดำล้วน
    full = np.zeros((300, 300, 3), np.uint8)
    result = convert.convert_to_lineart(full)
    assert result.mask.size > 0
