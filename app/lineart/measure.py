"""วัดความหนาเส้นจริงของภาพลายเส้น

ใช้ distance transform: สำหรับเส้นหนา w พิกเซล ค่าระยะสูงสุดจากขอบจะเป็น w/2
การรู้ความหนาจริงก่อนคือหัวใจของการทำให้เส้นหนาสม่ำเสมอโดยไม่ต้องเดาค่าตายตัว
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# ก้อนทึบ (เช่น ตาสีดำของการ์ตูน) มีความกว้างใกล้เคียงขนาดภาพ ต้องตัดออก
# ใช้ค่าสัดส่วน width / sqrt(area) เป็นเกณฑ์:
#   เส้นยาวบาง (เช่น กรอบหน้า) : width / sqrt(area) = sqrt(w/L) น้อยมาก
#   ก้อนกลมทึบ               : width / sqrt(area) = 2/sqrt(pi) ≈ 1.13
# จึงตัดเมื่อค่านี้สูงเกินเกณฑ์ ซึ่งจะไม่ไปรบกวนเส้นยาวที่มีพื้นที่มาก
BLOB_RATIO = 0.6

# ก้อนเล็กกว่านี้ถือเป็นจุดรบกวน ไม่ต้องเอามาวัด
MIN_COMPONENT_AREA_RATIO = 0.00002


@dataclass
class StrokeMetrics:
    """ผลการวัดความหนาเส้น หน่วยคือพิกเซลของภาพที่วัด"""

    median_width: float
    min_width: float
    max_width: float
    component_count: int
    ink_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def _as_mask(mask: np.ndarray) -> np.ndarray:
    """แปลงอะไรก็ได้ให้เป็น mask เดียว: 255 = หมึก"""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def measure(mask: np.ndarray) -> StrokeMetrics:
    """วัดความหนาเส้นที่พบมากที่สุดในภาพ

    ใช้ค่ามัธยฐานของความกว้างต่อส่วนประกอบ เพราะเส้นเส้นรอบนอก (outline)
    มีจำนวนมากกว่าลายประดับอย่างชัดเจน ค่ามัธยฐานจึงสะท้อนสไตล์ที่ต้องการ
    """
    mask = _as_mask(mask)
    height, width = mask.shape[:2]
    total_area = height * width

    ink_ratio = float(np.count_nonzero(mask)) / total_area
    if ink_ratio <= 0.0:
        return StrokeMetrics(0.0, 0.0, 0.0, 0, 0.0)

    # ใช้ mask 5x5 เพราะ mask 3x3 ทำให้เส้นบางวัดได้ค่ามากเกินจริง
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # ภาพที่เต็มไปด้วยหมึกทั้งหมดไม่มีพิกเซลพื้นหลังให้วัดระยะ
    # OpenCV คืนค่าที่ลอยฟ้ามาก ซึ่งทำให้เกิด overflow และค่าหลุดประมาณ
    # ไม่มีเส้นใบ้ใดหนากว่าครึ่งเส้นทแยงมุมของภาพอยู่แล้ว จึงตัดที่นั้นได้เลย
    distance = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
    distance = np.minimum(distance, float(np.hypot(height, width)) / 2.0)

    # พิกเซลบนสันเส้นคือจุดที่ระยะไม่น้อยกว่าพิกเซลรอบๆ
    # ความหนาเส้นจริงคือ 2 เท่าระยะที่จุดนั้น ซึ่งแม่นกว่าการใช้ค่าสูงสุดมาก
    # เพราะค่าสูงสุดจะเติมขึ้นตามหัวมุมและจุดตัดกันของเส้น
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ridge = (distance >= cv2.dilate(distance, kernel) - 1e-6) & (distance > 1.0)

    if not ridge.any():
        return StrokeMetrics(0.0, 0.0, 0.0, 0, round(ink_ratio, 5))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(4.0, MIN_COMPONENT_AREA_RATIO * total_area)

    ridge_labels = labels[ridge]
    ridge_dist = distance[ridge]
    order = np.argsort(ridge_labels, kind="stable")
    ridge_labels = ridge_labels[order]
    ridge_dist = ridge_dist[order]

    widths: list[float] = []
    boundaries = np.flatnonzero(np.diff(ridge_labels)) + 1
    groups = np.split(np.arange(len(ridge_labels)), boundaries)
    peaks = _component_peaks(distance, labels, count)

    for group in groups:
        if group.size == 0:
            continue
        label = int(ridge_labels[group[0]])
        if label == 0:
            continue
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # ตัดก้อนทึบออก เช่น ตา ปาก หรือจุดเต็ม
        # เกณฑ์นี้เทียบความกว้างเต็ม (ไม่ใช่ระยะครึ่งหนึ่ง) กับ BLOB_RATIO
        # ก้อนกลมทึบจะได้ความกว้างประมาณ 1.13 * sqrt(area) จึงถูกตัด
        # ส่วนเส้นยาวจะได้ sqrt(w/L) ซึ่งเล็กกว่าเกณฑ์มาก จึงผ่าน
        if peaks[label] * 2.0 > BLOB_RATIO * np.sqrt(area):
            continue
        widths.append(float(np.median(ridge_dist[group]) * 2.0))

    if not widths:
        # ไม่มีส่วนประกอบที่เข้าเงื่อนไข (เช่น ภาพที่เต็มไปด้วยหมึกทั้งหมด)
        # ใช้ค่ามัธยฐานของสันทั้งภาพแทน ซึ่งยังดีกว่าการเดาค่าคงที่
        fallback = _finite(float(np.median(ridge_dist) * 2.0))
        return StrokeMetrics(
            round(fallback, 2), round(fallback, 2), round(fallback, 2), 0, round(ink_ratio, 5)
        )

    arr = np.asarray(widths, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return StrokeMetrics(0.0, 0.0, 0.0, 0, round(ink_ratio, 5))
    return StrokeMetrics(
        median_width=round(_finite(float(np.median(arr))), 2),
        min_width=round(_finite(float(arr.min())), 2),
        max_width=round(_finite(float(arr.max())), 2),
        component_count=len(arr),
        ink_ratio=round(ink_ratio, 5),
    )


def _finite(value: float) -> float:
    """กันค่าที่ไม่เป็นจำนวนจำกัดไม่ให้หลุดออกไปใช้คำนวณต่อ"""
    return value if np.isfinite(value) else 0.0


def _component_peaks(
    distance: np.ndarray, labels: np.ndarray, count: int
) -> np.ndarray:
    """ระยะสูงสุดของแต่ละส่วนประกอบ คืนเป็นอาร์เรย์ความยาว count

    คำนวณด้วย np.maximum.at แทนการวนลูปเพื่อไม่ให้ช้เมื่อภาพมีจุดรบกวนเยอะ
    """
    peaks = np.zeros(count, dtype=np.float32)
    flat_labels = labels.ravel()
    np.maximum.at(peaks, flat_labels, distance.ravel().astype(np.float32))
    return peaks.astype(np.float64)


def dilation_for_width(current: float, target: float) -> int:
    """ขนาดเคอร์เนลสำหรับ dilate เพื่อเพิ่มความหนาเส้นจาก current ไปเป็น target

    dilate ด้วยเคอร์เนลวงรีที่รัศมี r จะขยายเส้นออกทั้งสองข้างรวม 2r พิกเซล
    ถ้า current >= target แล้วคืน 1 (ไม่ต้องทำอะไร) เพราะย่อเส้นด้วยวิธีนี้ไม่ได้
    การย่อเส้นต้องใช้ rescale_stroke_width แทน
    """
    grow = target - current
    if grow <= 0.0:
        return 1
    # kernel size ต้องเป็นจำนวนคี่ รัศมี r = (k-1)/2 → 2r = k-1
    size = int(round(grow)) + 1
    if size % 2 == 0:
        size += 1
    return max(1, min(size, 101))


# ถ้าย่อเส้นแล้วส่วนประกอบของภาพลดลงเกินสัดส่วนนี้ แปลว่าเส้นขาดเป็นส่วน ๆ
# เช่น ลายประดับเล็กที่บางกว่าเป้าหมายจะหายไปทั้งก้อน
THIN_MAX_COMPONENT_LOSS = 0.5
# ย่อเส้นย่อมทำให้จำนวนพิกเซลหมึกลดลงเสมอ จึงใช้เกณฑ์นี้แบบหยาบมาก
# เพื่อกันเฉพาะกรณีที่ภาพหายไปเกือบทั้งหน้า
THIN_MAX_INK_LOSS = 0.92
# จำนวนรอบการชดเชยความคลาดจากการย่อ และความคลาดที่ยอมรับได้เป็นพิกเซล
THIN_MAX_PASSES = 4
THIN_TOLERANCE = 1.0


@dataclass
class RescaleResult:
    """ผลการปรับความหนาเส้นให้ตรงเป้าหมาย"""

    mask: np.ndarray
    achieved_width: float
    thinned: bool
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def rescale_stroke_width(mask: np.ndarray, target: float) -> RescaleResult:
    """ปรับความหนาเส้นให้เท่ากับ target ทั้งหนาขึ้นและบางลง

    เดิมโปรแกรมทำได้แค่หนาขึ้น ถ้าเส้นเดิมหนากว่าเป้าหมายก็ปล่อยไว้
    ทำให้ผู้ใช้ที่มีงานเส้นหนาอยู่แล้วปรับให้บางลงไม่ได้เลย

    การย่อทำโดยเก็บเฉพาะพิกเซลที่ห่างจากพื้นหลังไม่เกิน target/2
    วิธีนี้ได้ความหนาสม่ำเสมอกว่าการกัดด้วยเคอร์เนล และไม่ทำให้เส้นเป็นเหลี่ยม
    """
    mask = _as_mask(mask)
    if target <= 0 or not mask.any():
        return RescaleResult(mask, 0.0, False)

    before = measure(mask)
    current = before.median_width
    if current <= 0:
        return RescaleResult(mask, 0.0, False)

    if current < target:
        kernel_size = dilation_for_width(current, target)
        if kernel_size <= 1:
            return RescaleResult(mask, current, False)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        grown = np.where(cv2.dilate(mask, kernel) >= 128, 255, 0).astype(np.uint8)
        return RescaleResult(grown, target, False)

    # ต้องย่อเส้น
    # การกัดด้วยวงกลมรัศมี r คือการเก็บเฉพาะพิกเซลที่ห่างจากพื้นหลังอย่างน้อย r
    # เส้นหนา w จึงเหลือกว้าง w - 2r ดังนั้น r = (current - target) / 2
    # ทำด้วย distance transform ครั้งเดียวได้ผลเหมือนกัดด้วยวงกลมพอดี
    # และได้เส้นที่หนาสม่ำเสมอกว่าการกัดด้วยเคอร์เนลสี่เหลี่ยมหรือวงรี
    radius = max(0.0, (current - target) / 2.0)
    if radius <= 0.0:
        return RescaleResult(mask, current, False)

    thinned = None
    achieved = current
    for _ in range(THIN_MAX_PASSES):
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        candidate = np.where(distance >= radius, 255, 0).astype(np.uint8)
        if not candidate.any():
            break
        thinned = candidate
        achieved = measure(candidate).median_width
        # ระยะในภาพดิจิตอลมีขั้นจำกัด ผลจึงไม่ค่อยตรงเป้าเป๊ะ
        # จึงวัดซ้ำแล้วชดเชยตามความคลาดที่พบจริง
        error = achieved - target
        if abs(error) <= THIN_TOLERANCE or radius <= 0.0:
            break
        radius = max(0.0, radius + error / 2.0)

    if thinned is None:
        return RescaleResult(mask, current, False, "ย่อเส้นจนหมด จึงคงความหนาเดิมไว้")

    after = measure(thinned)

    # ตรวจว่าไม่ได้ทำลายงานไปมากเกินไป
    ink_loss = 1.0 - (float(np.count_nonzero(thinned)) / float(np.count_nonzero(mask)))
    component_loss = 1.0
    if before.component_count > 0:
        component_loss = 1.0 - (after.component_count / before.component_count)

    if ink_loss > THIN_MAX_INK_LOSS or component_loss > THIN_MAX_COMPONENT_LOSS:
        return RescaleResult(
            mask,
            current,
            False,
            "ย่อเส้นแล้วรายละเอียดหายมากเกินไป จึงคงความหนาเดิมไว้ "
            "ลองเพิ่มความหนาเส้นหรือใช้ภาพที่มีเส้นหนาสม่ำเสมอกัน",
        )

    return RescaleResult(thinned, after.median_width, True)

