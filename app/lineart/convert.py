"""แปลงภาพให้เป็นภาพลายเส้นที่สะอาด

หลักการ: XDoG ดึงเส้นขอบออกมาแล้วปิดช่องว่างภายในตัวเส้นให้กลายเป็นเส้นทึบ
ความหนาจริงวัดจากภาพผลลัพธ์ ไม่ใช่ค่าที่ตั้งไว้ลอยๆ
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import MIN_LONG_EDGE_PX, LineArtParams
from . import detect, measure
from .imgio import ImageLoadError

# เมื่อพบกรอบโปร่งที่ใหญ่พอ จะถือว่าเป็นกรอบของภาพและตัดออกให้
BORDER_MIN_AREA_RATIO = 0.55
# สูงสุดที่ยอมให้เป็น "หมึก" ภายในกรอบได้ กรอบจริงจะมีพื้นที่โล่งๆ ข้างใน
# ถ้าสูงเกินนี้แปลว่าเป็นแผงที่ถูกระบายสีทับ ไม่ใช่กรอบ
BORDER_MAX_INTERIOR_INK = 0.35

# ขอบเขตการค้นหาขนาดเคอร์เนลที่ใช้ปิดช่องว่างภายในเส้น
MIN_CLOSE_KERNEL = 3
MAX_CLOSE_KERNEL = 15
CLOSE_KERNEL_STEP = 2
# เมื่อ ink เพิ่มขึ้นน้อยกว่านี้เทียบกับรอบก่อน แปลว่าเต็มที่แล้ว
# การปิดเพิ่มต่อจะไม่ได้ "เติม" แต่เริ่มไปรวมเส้นคนละเส้นเข้าหากัน ซึ่งไม่ต้องการ
SATURATION_RATIO = 0.05
HEAVY_INK_RATIO = 0.16
HEAVY_INK_OUTLINE_RADIUS = 4.0


@dataclass
class ConvertResult:
    """ผลลัพธ์การแปลงภาพ 1 ใบ"""

    mask: np.ndarray
    detection: detect.DetectionResult
    source_stroke: measure.StrokeMetrics
    close_kernel: int
    work_scale: float
    used_xdog: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.mask.size == 0 or not self.mask.any()


def _downscale(image: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, float]:
    """ย่อภาพให้ด้านยาวไม่เกิน max_long_edge คืนค่าภาพกับสเกลที่ใช้จริง"""
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return image, 1.0
    scale = max_long_edge / long_edge
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """ลบเงาและไล่แสงออกจากภาพ

    ภาพที่ถ่ายจากกระดาษเกือบทั้งใบจะมีแสงไม่สม่ำเสมอ ถ้าไม่ลบออกเส้นจะหายไป
    ตรงที่มืดและหนักเกินไปตรงที่สว่าง วิธีนี้คือหารด้วยภาพเบลอขนาดใหญ่
    แล้วดึงช่วงความสว่างกลับเต็ม 0-255
    """
    blur_sigma = max(8.0, min(gray.shape[:2]) / 12.0)
    background = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    # กันหารด้วยศูนย์ในบริเวณที่พื้นหลังมืดมาก
    ratio = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)

    # ดึงช่วงโดยใช้เปอร์เซ็นไทล์แทน min/max จริง เพราะพิกเซลเสียจาก
    # ฝุ่นหรือความสว่างจัดจังหวะจะทำให้ภาพเสียทั้งภาพ
    low, high = np.percentile(ratio, (0.5, 99.5))
    if high - low < 1e-3:
        # ภาพราบสนิท ไม่มีอะไรให้ปรับ คืนค่าเดิม
        return gray
    stretched = (ratio - low) / (high - low) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def xdog(gray: np.ndarray, sigma: float, tau: float, phi: float, k: float = 1.6) -> np.ndarray:
    """Extended Difference of Gaussians

    ให้เส้นที่ต่อเนื่องและมีความหนาสม่ำเสมอกว่า Canny มากสำหรับงานลายเส้น
    แต่ขอบจะออกมาเป็นสองเส้น (ขอบนอกและขอบในของตัวเส้น) ซึ่งต้องปิดทีหลัง

    ข้อสำคัญ: ค่าที่ได้มีขั้วกลับกัน  พื้นที่ราบจะได้ค่าสูง (ใกล้ 255)
    ส่วนตัวเส้นจะได้ค่าต่ำ ใช้ ink_sketch() เพื่อแปลงเป็น mask ที่ถูกต้อง
    """
    source = gray.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(source, (0, 0), sigma)
    g2 = cv2.GaussianBlur(source, (0, 0), sigma * k)
    diff = g1 - tau * g2
    eps = 0.01
    # 1 ตรงที่ diff สูงกว่า eps, ค่อยๆ ลดลงผ่าน tanh เมื่อต่ำกว่า
    shaped = np.where(diff >= eps, 1.0, 1.0 + np.tanh(phi * (diff - eps)))
    return np.clip(shaped * 255.0, 0, 255).astype(np.uint8)


def ink_sketch(response: np.ndarray) -> np.ndarray:
    """แปลงผลลัพธ์ XDoG ให้เป็น mask ที่ 255 = หมึก

    XDoG ให้ค่าสูงที่พื้นราบและค่าต่ำที่ตัวเส้น จึงต้องกลับขั้วก่อน
    """
    return cv2.threshold(response, 128, 255, cv2.THRESH_BINARY_INV)[1]


def binarize(gray: np.ndarray) -> np.ndarray:
    """ทำให้เป็นขาวดำด้วยเกณฑ์ Otsu โดยถือว่าส่วนที่สว่างที่สุดคือพื้นหลัง

    เลือก Otsu แทน adaptive threshold เพราะ adaptive จะกัดช่องภายในก้อนหมึบ
    ให้กลายเป็นหลุดโล่ง ซึ่งทำให้ตาสีดำของการ์ตูนกลายเป็นวงแหวน
    และยังดึงเงาที่เหลือออกมาเป็นเส้นเกินจริง
    ส่วนเงาไล่ระดับจัดการที่ normalize_illumination แล้ว
    """
    threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return binary


def despeckle(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    """ลบจุดและเส้นเล็กที่เป็นจุดรบกวน

    ใช้เกณฑ์พื้นที่เท่านั้น จึงไม่ไปแตะก้อนทึบขนาดใหญ่อย่างตาของการ์ตูน
    """
    height, width = mask.shape[:2]
    min_area = max(3.0, min_area_ratio * height * width)
    if min_area <= 3.0:
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    keep = np.ones(count, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            keep[label] = False
    if keep.all():
        return mask
    return np.where(keep[labels], mask, 0).astype(np.uint8)


def _ink_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / max(1, mask.size)


def auto_close_kernel(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """หาเคอร์เนลที่ใช้ปิดช่องว่างภายในเส้นให้พอดีโดยไม่รวมเส้นเข้าหากันเกิน

    ขยายเคอร์เนลทีละขั้นจนกว่าอัตราส่วนหมึกจะหยุดเพิ่ม ณ จุดนั้นคือช่องภายในเส้น
    ถูกปิดครบแล้ว การขยายต่อจะเป็นการรวมเส้นคนละเส้น ซึ่งไม่ต้องการ
    """
    best_kernel = MIN_CLOSE_KERNEL
    best_mask = mask
    previous_ratio = _ink_ratio(mask)
    previous_increase = None

    kernel_size = MIN_CLOSE_KERNEL
    while kernel_size <= MAX_CLOSE_KERNEL:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        current_ratio = _ink_ratio(closed)
        increase = (current_ratio - previous_ratio) / max(previous_ratio, 1e-9)

        if previous_increase is not None and increase < SATURATION_RATIO:
            # รอบนี้ไม่ได้เติมอะไรอีกแล้ว คงค่าเคอร์เนลจากรอบก่อนหน้า
            break

        best_kernel = kernel_size
        best_mask = closed
        previous_ratio = current_ratio
        previous_increase = increase
        kernel_size += CLOSE_KERNEL_STEP

    return best_kernel, best_mask


def outline_heavy_ink(mask: np.ndarray, radius_px: float = HEAVY_INK_OUTLINE_RADIUS) -> np.ndarray:
    """Turn large filled black regions into printable outlines.

    Photos and web images often contain solid black shadows or patterns. Keeping
    those areas as ink creates unusable coloring pages, so for ink-heavy masks we
    retain only pixels close to a white boundary.
    """
    if mask.size == 0:
        return mask
    binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
    if not binary.any():
        return binary
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outlined = np.where((binary > 0) & (distance <= radius_px), 255, 0).astype(np.uint8)
    return outlined if outlined.any() else binary


def crop_to_ink(mask: np.ndarray, padding_ratio: float = 0.03) -> np.ndarray:
    """ครอปให้เหลือแต่ส่วนที่มีหมึก พร้อมขอบเว้นเล็กน้อย"""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return mask
    x, y, width, height = cv2.boundingRect(coords)
    pad_x = max(2, round(width * padding_ratio))
    pad_y = max(2, round(height * padding_ratio))
    height_px, width_px = mask.shape[:2]
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width_px, x + width + pad_x)
    y1 = min(height_px, y + height + pad_y)
    return mask[y0:y1, x0:x1]


def strip_existing_border(
    mask: np.ndarray, stroke_px: float = 6.0
) -> tuple[np.ndarray, bool]:
    """ตัดกรอบที่มีอยู่เดิมของภาพต้นฉบับออก

    ภาพลายเส้นที่ดาวน์โหลดมามักมีกรอบมาตั้งแต่ต้นฉบับอยู่แล้ว ถ้าไม่ตัดออก
    เมื่อนำมาวางบนหน้าที่มีกรอบของโปรแกรม จะกลายเป็นกรอบซ้อนสองชั้น

    เกณฑ์การพิจารณาว่าเป็นกรอบ: เป็นวงปิดที่เกือบเต็มพื้นที่ภาพ
    และด้านในโล่ง ไม่ใช่แผงที่เต็มหมึก

    stroke_px คือความหนาเส้นที่วัดได้ ใช้กำหนดว่าจะตัดเข้าไปลึกแค่ไหน
    จึงไม่เหลือเศษมุมของกรอบเดิมติดอยู่
    """
    height, width = mask.shape[:2]
    min_area = (height * width) * BORDER_MIN_AREA_RATIO

    # เผื่อกรอบที่ไม่ได้อยู่ติดขอบสุด ให้เผื่อขอบเว้นไว้หน่อย
    margin_x = round(width * 0.02)
    margin_y = round(height * 0.02)
    inner = mask[margin_y : height - margin_y, margin_x : width - margin_x]
    if inner.size == 0:
        return mask, False

    contours, hierarchy = cv2.findContours(
        inner, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return mask, False

    inner_area = inner.shape[0] * inner.shape[1]
    best = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area >= inner_area * 0.995:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # กรอบแท้จะเกือบเต็มทั้งแนวตั้งและแนวนอนของส่วนที่เหลือ
        if w < inner.shape[1] * 0.80 or h < inner.shape[0] * 0.80:
            continue
        # ต้องเป็นกรอบโปร่ง ไม่ใช่แผงที่เต็มหมึก
        # ถ้าข้างในเต็มหมึกเกือบทั้งก้อน แปลว่านี่คือภาพที่ถูกเติมสีทับ
        # ไม่ใช่กรอบ และการตัดทิ้งจะทำให้งานหายทั้งชิ้น
        window = inner[y : y + h, x : x + w]
        if window.size == 0:
            continue
        if _ink_ratio(window) > BORDER_MAX_INTERIOR_INK:
            continue
        if area > best_area:
            best, best_area = contour, area

    if best is None:
        return mask, False

    # ตัดด้วยการเติมวงกรอบแล้วกัดออก ไม่ใช่การครอปสี่เหลี่ยม
    # เพราะมุมโค้งของกรอบจะยื่นเข้ามาในสี่เหลี่ยม การครอปตรงๆ จะทิ้งเศษมุมไว้
    # แล้วไปโผล่เป็นรอยเล็กๆ ตามมุมของกรอบหน้าใหม่
    filled = np.zeros_like(inner)
    cv2.drawContours(filled, [best], -1, 255, thickness=cv2.FILLED)

    # กัดเข้าไปลึกพอที่กินเส้นกรอบทั้งเส้น ไม่งั้นเศษขอบกรอบจะยังติดอยู่
    thickness_probe = max(4, round(stroke_px * 1.5))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thickness_probe + 1, 2 * thickness_probe + 1)
    )
    interior = cv2.erode(filled, kernel)

    if not interior.any():
        return mask, False

    # เชื่อมกลับเข้าพิกัลเดิมก่อนนำไปตัดกับภาพ
    interior_full = np.zeros_like(mask)
    interior_full[margin_y : margin_y + inner.shape[0], margin_x : margin_x + inner.shape[1]] = interior

    stripped = mask.copy()
    stripped[interior_full == 0] = 0

    if not stripped.any():
        return mask, False

    return stripped, True


def _auto_params(metrics: measure.StrokeMetrics, params: LineArtParams) -> dict:
    """เลือกค่าที่ยังเป็น None จากสถิติของภาพจริง"""
    median = max(1.0, metrics.median_width or 5.0)
    speckle = params.speckle_ratio
    if speckle is None:
        # เกณฑ์นี้ต้องกำจัดจุดรบกวนจากการสแกน/บีบอัด (มักเล็กกว่า ~100 พิกเซล)
        # แต่ต้องไม่ตัดลายประดับเล็กที่เป็นส่วนหนึ่งของงานจริงทิ้ง
        # จึงอ้างอิงความหนาเส้นเพียงเล็กน้อย และจำกัดช่วงไว้แคบ
        speckle = float(np.clip(median * 0.000012, 0.00004, 0.00015))

    denoise = params.denoise
    if denoise is None:
        denoise = int(np.clip(round(median / 3.0), 0, 7))
        if denoise % 2 == 0:
            denoise += 1

    sigma = params.xdog_sigma
    if sigma is None:
        # ต้องใหญ่พอที่จะไม่จับพื้นผิว แต่เล็กพอที่จะเก็บรายละเอียด
        sigma = float(np.clip(median / 2.5, 0.8, 3.0))

    return {
        "speckle_ratio": float(speckle),
        "denoise": int(denoise),
        "xdog_sigma": float(sigma),
    }


def convert(image: np.ndarray, params: LineArtParams | None = None) -> ConvertResult:
    """แปลงภาพนำเข้าเป็น mask ลายเส้น

    ถ้าเป็นภาพลายเส้นอยู่แล้วจะข้ามขั้น XDoG ไปใช้การทำให้เป็นขาวดำแทน
    เพราะการดึงเส้นขอบซ้ำบนภาพที่เป็นเส้นอยู่แล้วจะทำให้รายละเอียดหาย
    """
    if image is None or image.size == 0:
        raise ImageLoadError("ไม่พบข้อมูลภาพ")

    params = params or LineArtParams()
    warnings: list[str] = []

    work_image, work_scale = _downscale(image, params.max_long_edge)
    gray = _to_gray(work_image)

    detection = detect.analyze(work_image)
    if not detection.is_line_art:
        warnings.append(detection.reason)

    source_height, source_width = gray.shape[:2]
    if max(source_height, source_width) < MIN_LONG_EDGE_PX:
        warnings.append(
            "ความละเอียดภาพต่ำ เมื่อขยายเป็น A4 เส้นอาจไม่คมชัด "
            "แนะนำภาพที่ด้านยาวอย่างน้อย 1500 พิกเซล"
        )

    pre_metrics = measure.measure(_preview_mask(gray))
    resolved = _auto_params(pre_metrics, params)

    # ลบเงาก่อนเสมอทั้งสองเส้นทาง ไม่งั้นเกณฑ์ Otsu จะเลือกค่าผิด
    # เมื่อภาพถ่ายจากกระดาษที่มีแสงไม่สม่ำเสมอ
    normalized = normalize_illumination(gray)
    if resolved["denoise"] >= 3:
        normalized = cv2.medianBlur(normalized, int(resolved["denoise"]))

    if detection.is_line_art and detection.binary_ratio >= detect.BINARY_RATIO_CLEAN:
        # ภาพลายเส้นอยู่แล้ว การดึงเส้นขอบซ้ำจะทำให้รายละเอียดหาย
        mask = binarize(normalized)
        used_xdog = False
    else:
        lines = xdog(
            normalized,
            sigma=resolved["xdog_sigma"],
            tau=params.xdog_tau or 0.98,
            phi=params.xdog_phi or 18.0,
        )
        mask = ink_sketch(lines)
        used_xdog = True

    mask = despeckle(mask, resolved["speckle_ratio"])

    kernel_size, mask = auto_close_kernel(mask)
    if params.close_iterations > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        for _ in range(params.close_iterations - 1):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if _ink_ratio(mask) > HEAVY_INK_RATIO:
        mask = outline_heavy_ink(mask)
        warnings.append(
            "ภาพมีพื้นที่ดำทึบมาก จึงแปลงปื้นดำให้เหลือเฉพาะเส้นขอบเพื่อให้ระบายสีได้"
        )

    mask = despeckle(mask, resolved["speckle_ratio"] * 0.5)

    stripped_border = False
    if params.strip_border:
        stroke_px = measure.measure(mask).median_width
        mask, stripped_border = strip_existing_border(mask, stroke_px)
        if stripped_border:
            warnings.append("ตัดกรอบเดิมของภาพออกแล้ว เพื่อไม่ให้กรอบซ้อนกัน")

    mask = crop_to_ink(mask)

    source_stroke = measure.measure(mask)
    if source_stroke.component_count == 0 and not used_xdog:
        # ภาพที่ Otsu แยกไม่ออกจนไม่เหลือเส้นเลย (เช่น ภาพทึบทั้งหน้า
        # หรือภาพที่สว่างจนล้วน) ลองใหม่ด้วยการดึงเส้นขอบซึ่งไวกว่า
        lines = xdog(
            normalized,
            resolved["xdog_sigma"],
            params.xdog_tau or 0.98,
            params.xdog_phi or 18.0,
        )
        mask = crop_to_ink(despeckle(ink_sketch(lines), resolved["speckle_ratio"]))
        source_stroke = measure.measure(mask)
        used_xdog = True
        warnings.append("แยกเส้นด้วยเกณฑ์ความสว่างไม่ได้ เลยใช้การดึงเส้นขอบแทน")

    return ConvertResult(
        mask=mask,
        detection=detection,
        source_stroke=source_stroke,
        close_kernel=kernel_size,
        work_scale=work_scale,
        used_xdog=used_xdog,
        warnings=warnings,
    )


def _preview_mask(gray: np.ndarray) -> np.ndarray:
    """สร้าง mask คร่าวๆ ใช้วัดสถิติเพื่อเลือกพารามิเตอร์ก่อนประมวลผลจริง

    ตั้งใจให้ถูกต้องพอที่จะเลือกค่าได้ ไม่จำเป็นต้องสมบูรณ์ เพราะผลลัพธ์สุดท้าย
    วัดความหนาเส้นซ้ำอีกครั้งหลังประมวลผลเสร็จอยู่แล้ว
    """
    binary = binarize(normalize_illumination(gray))
    return despeckle(binary, 0.0002)
