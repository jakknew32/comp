"""ทดสอบ pipeline การแปลงภาพเป็นลายเส้น"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import LineArtParams
from app.lineart import convert, detect, measure
from tests import fixtures


def test_clean_lineart_is_detected() -> None:
    result = detect.analyze(fixtures.synthetic_lineart())
    assert result.is_line_art
    assert result.confidence > 0.5


def test_photo_is_flagged_as_not_lineart() -> None:
    """ภาพถ่ายต้องถูกจับได้ เพื่อให้หน้าเว็บเตือนผู้ใช้ได้ทันที"""
    result = detect.analyze(fixtures.synthetic_photo())
    assert not result.is_line_art
    assert "ภาพถ่าย" in result.reason


def test_clean_lineart_conversion_keeps_ink_ratio() -> None:
    source = fixtures.synthetic_lineart()
    result = convert.convert(source, LineArtParams())
    source_ink = float((source < 128).mean())
    result_ink = float((result.mask > 0).mean())
    assert result_ink == pytest.approx(source_ink, rel=0.25)
    assert not result.used_xdog


def test_solid_eyes_survive_conversion() -> None:
    """ตาของการ์ตูนเป็นก้อนทึบ ต้องไม่กลายเป็นวงแหวนโหว่กลาง

    เคยพลาดตอนใช้ adaptive threshold ซึ่งกัดช่องภายในก้อนหมึบทิ้ง
    ทำให้ตากลายเป็นวงแหวน ซึ่งพิมพ์ออกมาแล้วระบายสีไม่ได้
    """
    mask = convert.convert(
        fixtures.synthetic_lineart(), LineArtParams(strip_border=False)
    ).mask
    assert not mask[:5].any(), "ภาพควรถูกครอปจนไม่เหลือหมึกติดขอบ"

    # หาก้อนหมึกทึบทรงกลม ซึ่งคือตา โดยดูจากสัดส่วนพื้นที่ต่อกรอบ
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    solids = []
    for i in range(1, count):
        area = float(stats[i, cv2.CC_STAT_AREA])
        box_w = float(stats[i, cv2.CC_STAT_WIDTH])
        box_h = float(stats[i, cv2.CC_STAT_HEIGHT])
        box_area = box_w * box_h
        if box_area <= 0:
            continue
        # วงกลมที่เต็มจะมีพื้นที่หมึกประมาณ 78% ของกรอบ
        # วงแหวนจะมีพื้นที่หมึกน้อยกว่านั้นมาก
        if 0.70 < area / box_area < 0.85:
            solids.append(i)

    assert len(solids) >= 2, "ไม่พบก้อนหมึกทึบรูปกลมอย่างน้อย 2 ก้อน (คาดว่าเป็นตา)"

    # ยืนยันว่าไม่มีรูโหว่สีขาวอยู่ภายในกล่องของก้อนทึบเหล่านั้น
    inverse = (mask == 0).astype(np.uint8)
    hole_count, _, hole_stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=8
    )
    for solid in solids:
        x = stats[solid, cv2.CC_STAT_LEFT]
        y = stats[solid, cv2.CC_STAT_TOP]
        w = stats[solid, cv2.CC_STAT_WIDTH]
        for hole in range(1, hole_count):
            if hole_stats[hole, cv2.CC_STAT_AREA] < 200:
                continue
            hx = hole_stats[hole, cv2.CC_STAT_LEFT]
            hy = hole_stats[hole, cv2.CC_STAT_TOP]
            hw = hole_stats[hole, cv2.CC_STAT_WIDTH]
            # รูโหวะที่อยู่ภายในกล่องของก้อนทึบ แปลว่าก้อนนั้นถูกกัดเป็นวงแหวน
            assert not (x < hx and y < hy and hx + hw < x + w), (
                "พบรูโหว่ภายในก้อนทึบ แสดงว่าก้อนถูกกัดจนเป็นวงแหวน"
            )


def test_photocopy_shading_is_removed() -> None:
    """ภาพที่ถ่ายจากกระดาษต้องให้ผลใกล้เคียงต้นฉบับที่ไม่มีเงา"""
    clean = convert.convert(
        fixtures.synthetic_lineart(), LineArtParams(strip_border=False)
    )
    shaded = convert.convert(
        fixtures.photocopied_lineart(), LineArtParams(strip_border=False)
    )

    assert shaded.detection.is_line_art
    assert not shaded.used_xdog
    clean_ink = float((clean.mask > 0).mean())
    shaded_ink = float((shaded.mask > 0).mean())
    assert shaded_ink == pytest.approx(clean_ink, rel=0.15)


def test_photo_does_not_crash_and_returns_something() -> None:
    result = convert.convert(fixtures.synthetic_photo(), LineArtParams())
    assert result.mask.size > 0
    assert any("ภาพถ่าย" in w for w in result.warnings)


def test_existing_border_is_stripped() -> None:
    """ภาพที่มีกรอบมาติดมาต้องถูกตัดออก ไม่ให้ซ้อนกับกรอบของหน้า"""
    with_border = convert.convert(fixtures.synthetic_lineart(), LineArtParams())
    without = convert.convert(
        fixtures.synthetic_lineart(), LineArtParams(strip_border=False)
    )
    assert any("ตัดกรอบ" in w for w in with_border.warnings)
    assert not any("ตัดกรอบ" in w for w in without.warnings)
    # หลังตัดกรอบ ภาพต้องเล็กลง
    assert with_border.mask.shape[0] < without.mask.shape[0]


def test_speckles_are_removed() -> None:
    """จุดรบกวนเล็กๆ ต้องหายไป แต่เส้นจริงต้องยังอยู่"""
    result = convert.convert(fixtures.synthetic_lineart(), LineArtParams())
    count, _, stats, _ = cv2.connectedComponentsWithStats(result.mask, connectivity=8)
    tiny = [
        stats[i, cv2.CC_STAT_AREA]
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] < 20
    ]
    assert not tiny, f"ยังเหลือจุดรบกวน {len(tiny)} จุด"


def test_measured_stroke_width_is_plausible() -> None:
    result = convert.convert(fixtures.synthetic_lineart(), LineArtParams())
    assert 3.0 <= result.source_stroke.median_width <= 25.0


def test_low_resolution_produces_warning() -> None:
    tiny = fixtures.synthetic_lineart(300, 400)
    result = convert.convert(tiny, LineArtParams())
    assert any("ความละเอียด" in w for w in result.warnings)


def test_blank_image_is_reported_as_empty() -> None:
    blank = np.full((400, 400), 255, np.uint8)
    result = convert.convert(blank, LineArtParams())
    assert result.is_empty or result.source_stroke.median_width == 0.0


def test_does_not_dilate_below_current_width() -> None:
    """ค่าเป้าหมายที่บางกว่าเส้นเดิมต้องไม่ทำให้เส้นหนาขึ้น"""
    assert measure.dilation_for_width(current=20.0, target=1.0) == 1
