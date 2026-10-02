"""Pipeline หลักสำหรับแปลงภาพเป็นภาพลายเส้นที่พร้อมระบายสี

ทุกขั้นมีพารามิเตอร์ที่ **คำนวณอัตโนมัติ** จากตัวภาพจริง แล้ว override
ได้จาก slider ในหน้าเว็บ ค่าที่ส่งมาเป็น ``None`` หมายถึง "ให้ระบบตัดสินใจ"

.. note::
   จุดต่อสำหรับ AI ในอนาคตอยู่ที่ :func:`convert_to_lineart` — ถ้าจะเพิ่ม
   ขั้นตอน img2img (เช่น FLUX.1-Kontext / ControlNet lineart) ให้แทรก
   provider ใหม่ต่อท้าย pipeline เดิม โดยไม่ต้องแก้โครงสร้างไฟล์อื่น
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from ..config import ConvertParams, DPI, LOW_RES_PX, mm_to_px
from . import detect as detect_mod
from . import measure as measure_mod
from .imgio import to_gray

#: ขนาดยาวสูงสุดของภาพระหว่างประมวลผล (เร็วขึ้นมาก และได้ความละเอียด
#: ที่ 300 DPI อยู่แล้ว)
MAX_PROCESS_EDGE = 1400

#: พารามิเตอร์คงที่ของ XDoG (Extended Difference of Gaussians)
XDog_TAU = 0.98
XDog_PHI = 18.0


@dataclass
class ConvertResult:
    """ผลลัพธ์การแปลงภาพหนึ่งใบ"""

    #: ภาพลายเส้น binary (255 = หมึก) ขนาดตามที่ crop แล้ว
    mask: np.ndarray
    #: ผลตรวจประเภทภาพต้นทาง
    detection: detect_mod.DetectionResult
    #: ผลวัด/วางแผนความหนาเส้น
    stroke: measure_mod.StrokeMeasurement
    #: ค่าที่ระบบตัดสินใจจริงในแต่ละขั้น (ส่งกลับไปแสดงใน UI)
    auto_values: dict[str, Any] = field(default_factory=dict)
    #: คำเตือนที่ควรแสดงผู้ใช้
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# ขั้นตอนย่อย
# --------------------------------------------------------------------------


def normalize_illumination(gray: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """ปรับแสงให้สม่ำเสมอเพื่อลบเงาและไล่แสง

    ขั้นนี้สำคัญที่สุดสำหรับภาพที่ถ่ายจากหน้ากระดาษ ผลลัพธ์คือภาพที่
    เหลือแต่โครงเส้น ไม่มีไล่แสงค้าง
    """
    gray_f = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray_f, (0, 0), sigma)
    return gray_f / (background + 1e-3)


def resize_to_max_edge(image: np.ndarray, max_edge: int = MAX_PROCESS_EDGE) -> np.ndarray:
    """ย่อภาพให้ด้านยาวไม่เกิน ``max_edge`` โดยรักษาสัดส่วน"""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return image
    scale = max_edge / float(long_edge)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def xdog(
    gray: np.ndarray,
    sigma: float,
    tau: Optional[float] = None,
    phi: Optional[float] = None,
) -> np.ndarray:
    """Extended Difference of Gaussians — ดึงเส้นจากภาพลายเส้น

    ให้เส้นต่อเนื่องและสะอาดกว่า Canny มากสำหรับงานลายเส้น เพราะใช้
    ผลต่างของสองฟิลเตอร์ Gaussian ที่คนละระดับความละเอียด

    **ทำไมต้องใช้ค่าต่างสัมบูรณ์ + Otsu แทนสูตร ``g1 - tau*g2 > phi``**
    สูตรดั้งเดิมเขียนมาสำหรับภาพที่ normalize ให้อยู่ราว ๆ ช่วง [-1, 1]
    และเส้นเป็น *เส้นสว่างบนพื้นมืด* แต่ภาพลายเส้นของเราเป็น
    *เส้นมืดบนพื้นสว่าง* และภาพถ่ายมีค่าความสว่างพื้นหลังไม่ใช่ 0
    ถ้าใช้สูตรเดิมตรง ๆ บนพื้นหลังที่สม่ำเสมอ จะได้
    ``g1 - 0.98*g2 ~= 0.02 * ความสว่างพื้นหลัง`` ซึ่งมัก **มากกว่า**
    เกณฑ์ ``phi`` ทำให้พื้นหลังทั้งภาพถูกทำเครื่องหมายเป็นเส้นทั้งหมด
    (สังเกตได้จากภาพถ่ายที่ได้พื้นหลังดำสนิท)

    วิธีที่ใช้แทน: เอาค่าต่างสัมบูรณ์ของสองระดับความละเอียด (จับได้ทั้งเส้นมืด
    และเส้นสว่าง ซึ่งจำเป็นสำหรับรายละเอียดอย่างตาของการ์ตูน) แล้วเลือก
    เกณฑ์ด้วย Otsu บนค่าตอบสนองที่ปรับสเกลให้อยู่ในช่วง 0–255
    ทำให้ไม่ต้องเดาค่าคงที่และใช้ได้กับทุกชนิดของภาพ

    :param tau: ถ้าระบุ จะใช้เกณฑ์คงที่แทน Otsu (0–255 บนสเกลที่ปรับแล้ว)
    :param phi: ไม่ใช้ในการตัดสินใจหลัก เก็บไว้เพื่อความเข้ากันได้
        กับสูตร XDoG ต้นฉบับ — ใช้เป็นค่าเกณฑ์ขั้นต่ำเมื่อระบุ ``tau``
    """
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * 1.57)

    response = np.abs(g1.astype(np.float32) - g2.astype(np.float32))

    # ปรับสเกลให้เส้นที่แรงที่สุดเต็มช่วง 0–255 เพื่อให้ Otsu ทำงานได้ดี
    # ไม่ว่าภาพจะสว่างหรือมืดแค่ไหน
    high = float(np.percentile(response, 99.5))
    if high < 1e-3:
        return np.zeros(gray.shape[:2], dtype=np.uint8)

    scaled = np.clip(response * (255.0 / high), 0, 255).astype(np.uint8)

    if tau is not None:
        threshold = int(round(tau))
    else:
        threshold = 0  # ให้ Otsu เลือกเอง

    _, edges = cv2.threshold(
        scaled, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return edges


def canny_fallback(gray: np.ndarray) -> np.ndarray:
    """ทางเลือกสำรองเมื่อ XDoG ให้ผลไม่ดี — Canny ที่คำนวณ threshold จาก median"""
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    if upper <= lower:
        lower, upper = 50, 150
    return cv2.Canny(blurred, lower, upper)


def despeckle(
    mask: np.ndarray,
    sensitivity: float,
    min_area_override: Optional[int] = None,
) -> np.ndarray:
    """ตัด component ที่เล็กเกินกำหนด เพื่อกำจัดจุดรบกวน

    ``sensitivity`` = 0 → เกณฑ์พื้นฐาน (เก็บทุกก้อนที่ใหญ่พอ)
    ``sensitivity`` = 1 → เข้มขึ้นมาก (ตัดแม้แต่ก้อนค่อนข้างใหญ่)

    จุดเสี่ยง: ต้องรักษาก้อนใหญ่ไว้ เช่นตาสีดำของการ์ตูน
    ก้อนทึบขนาดใหญ่มีพื้นที่มากกว่าเกณฑ์เสมอ จึงไม่ถูกตัด
    """
    height, width = mask.shape[:2]
    total_area = height * width

    if min_area_override is not None:
        min_area = int(min_area_override)
    else:
        # ค่าฐาน 0.0002 × พื้นที่ แล้วขยายตามความไว (ยิ่งไวยิ่งตัดแรง)
        base = 0.0002 * total_area
        min_area = base * (1.0 + sensitivity * 24.0)

    # อย่าตัดก้อนที่ใหญ่ผิดปกติเด็ดขาด: เพดานสูงสุด 2% ของพื้นที่
    min_area = min(min_area, total_area * 0.02)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return mask

    out = np.zeros_like(mask)
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = 255
    return out


def close_gaps(mask: np.ndarray, rounds: int) -> np.ndarray:
    """ซ่อมเส้นขาดด้วย morphological close (เคอร์เนลวงรี 3×3)"""
    if rounds <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=rounds)


def fill_closed_shapes(mask: np.ndarray, max_hole_ratio: float = 0.06) -> np.ndarray:
    """ปิดช่องว่างด้วยการเติม contour ภายนอก

    เส้นที่ควรเป็นวงปิด (เช่น ตากลมโต) จะถูกเติมให้เป็นก้อนทึบ
    ตามแผนต้อง "เปิดกรณีพิเศษถ้าผลลัพธ์ทำให้ก้อนทึบ (ตา/ปาก) หาย"
    ฟังก์ชันนี้จึงมี guard: ตรวจพื้นที่หมึกหลังเติม ถ้าเพิ่มขึ้นเกิน
    ``max_hole_ratio`` ให้คืนภาพเดิม (เส้นเล็ก ๆ เช่นปากหรือจมูปลายทาง
    ต้องไม่ถูกกลืนไปกับพื้นที่ใหญ่)
    """
    binary = (mask > 127).astype(np.uint8)
    original_ink = int(binary.sum())
    if original_ink == 0:
        return mask

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    added = int(filled.sum()) - original_ink
    if added <= 0:
        return mask
    if added / original_ink > max_hole_ratio:
        # เติมแล้วก้อนทึบขยายใหญ่เกินกว่าจะปลอดภัย — คืนภาพเดิม
        return mask

    return filled * 255


def restore_large_blobs(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """คืนก้อนทึบขนาดใหญ่กลับ ถ้ากระบวนการทำให้หาย

    ป้องกันความเสี่ยงข้อ 2: ตาของการ์ตูนเป็นก้อนทึบขนาดใหญ่ที่ต้องไม่หาย
    """
    orig_binary = (original > 127).astype(np.uint8)
    proc_binary = (processed > 127).astype(np.uint8)
    if orig_binary.sum() == 0:
        return processed

    # ก้อนที่อยู่ในต้นฉบับแต่ไม่เหลือในผลลัพธ์เลย
    lost = cv2.bitwise_and(orig_binary, cv2.bitwise_not(proc_binary))
    if lost.sum() == 0:
        return processed

    num, labels, stats, _ = cv2.connectedComponentsWithStats(lost, connectivity=8)
    result = proc_binary.copy()
    total = orig_binary.size
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= total * 0.0005:  # ก้อนที่หายไปใหญ่พอสมควรคืน
            result[labels == label] = 1
    return result * 255


def crop_to_ink(mask: np.ndarray, padding_ratio: float = 0.03) -> np.ndarray:
    """ครอปช่องว่างรอบหมึก เว้น 3% แล้วคืนภาพพร้อมกรอบเดิม"""
    binary = (mask > 127).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return mask

    x, y, w, h = cv2.boundingRect(coords)
    pad_x = int(round(w * padding_ratio))
    pad_y = int(round(h * padding_ratio))

    height, width = mask.shape[:2]
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)

    return mask[y0:y1, x0:x1]


def _resolve_target_px(target_line_mm: Optional[float]) -> float:
    """แปลงความหนาเป้าหมายเป็นพิกเซลที่ DPI ของหน้ากระดาษ."""
    if target_line_mm is None:
        from ..config import TARGET_LINE_MM

        target_line_mm = TARGET_LINE_MM
    return float(mm_to_px(target_line_mm, DPI))


def _resolve_sigma(source_stroke_px: float, dog_sigma: Optional[float]) -> float:
    """ค่า sigma ของ XDoG — อิงจากความหนาเส้นที่วัดได้จริง

    XDoG ตอบสนองต่อโครงสร้างที่ใหญ่กว่า sigma ประมาณหนึ่งเท่า ถ้าใช้
    sigma ใหญ่เกินความหนาเส้น เส้นบาง ๆ จะหายไปทั้งหมด ดังนั้นจึงผูก
    sigma กับความหนาเส้นที่วัดได้ ไม่ใช่กับขนาดภาพอย่างเดียว
    """
    if dog_sigma is not None:
        return float(dog_sigma)
    if source_stroke_px <= 0:
        return 1.0
    return float(min(4.0, max(0.8, source_stroke_px * 0.8)))


def _resolve_close_rounds(close_rounds: Optional[int]) -> int:
    if close_rounds is not None:
        return int(close_rounds)
    return 1


# --------------------------------------------------------------------------
# Pipeline หลัก
# --------------------------------------------------------------------------


def convert_to_lineart(
    bgr: np.ndarray,
    params: Optional[ConvertParams] = None,
    *,
    output_width_px: Optional[int] = None,
) -> ConvertResult:
    """แปลงภาพเป็นภาพลายเส้นพร้อมระบายสี

    :param bgr: ภาพต้นทาง BGR uint8
    :param params: พารามิเตอร์ที่คำนวณอัตโนมัติ (ค่า ``None`` = ให้ระบบตัดสินใจ)
    :param output_width_px: ความกว้างสุดท้ายบนหน้ากระดาษ (px) ถ้าระบุ
        ระบบจะคิดเป้าหมายความหนาเส้นให้คงเดิมหลังขยายลงหน้ากระดาษ
        ไม่เช่นนั้นภาพที่ความละเอียดต่างกันจะได้เส้นหนาต่างกันบนกระดาษ
    :returns: :class:`ConvertResult` ที่มี mask, ผลตรวจประเภทภาพ
        ผลวัดความหนาเส้น และค่าที่ระบบเลือกจริง
    """
    params = params or ConvertParams()
    warnings: list[str] = []
    auto_values: dict[str, Any] = {}

    if bgr is None or bgr.size == 0:
        raise ValueError("ภาพว่างเปล่า")

    # --- ขั้น 1: ตรวจประเภทภาพ ---
    detection = detect_mod.analyze(bgr)
    auto_values["target_line_mm"] = (
        params.target_line_mm
        if params.target_line_mm is not None
        else _target_line_mm_default()
    )
    auto_values["despeckle"] = (
        params.despeckle if params.despeckle is not None else 0.0
    )

    if detection.is_photo:
        warnings.append(
            "ภาพนี้อาจไม่ใช่ภาพลายเส้น — โปรแกรมจะได้ 'เส้นตามภาพ' "
            "ไม่ใช่การ์ตูนน่ารัก"
        )
    if detection.low_resolution:
        warnings.append(
            f"ความละเอียดต่ำ ({detection.long_edge_px} px) — "
            f"แนะนำอย่างน้อย {LOW_RES_PX} px ด้านยาวเพื่อไม่ให้เส้นเบลอเมื่อขยาย"
        )

    # --- ขั้น 2: โหลดเป็นเทา + EXIF (ผ่านมาแล้วจาก imgio) ---
    gray = to_gray(bgr)

    # --- ขั้น 3: ย่อขนาดเพื่อความเร็ว ---
    gray = resize_to_max_edge(gray)

    # --- ขั้น 4: ปรับแสงสม่ำเสมอ ---
    normalized = normalize_illumination(gray)
    norm_u8 = np.clip(normalized * 128.0, 0, 255).astype(np.uint8)

    # --- ขั้น 5: วัดความหนาเส้นต้นฉบับ แล้วเลือกวิธีดึงเส้นให้เหมาะกับภาพ ---
    source_stroke = _estimate_source_stroke(norm_u8)
    auto_values["source_stroke_px"] = round(source_stroke, 2)
    sigma = _resolve_sigma(source_stroke, params.dog_sigma)
    auto_values["dog_sigma"] = round(sigma, 2)

    if detection.is_line_art and params.dog_sigma is None:
        mask = _threshold_lineart(norm_u8, 0.6)
        auto_values["method"] = "threshold (ตรวจพบว่าเป็น line art อยู่แล้ว)"
    else:
        mask = xdog(norm_u8, sigma)
        auto_values["method"] = "XDoG"
        # XDoG อาจได้ผลไม่ดีกับภาพที่ไม่ใช่ลายเส้น → ตรวจแล้วสลับเป็น Canny
        if _ink_ratio(mask) < 0.002:
            mask = canny_fallback(norm_u8)
            auto_values["method"] = "Canny (สำรอง เพราะ XDoG ได้เส้นน้อยเกินไป)"

    original_mask = mask.copy()

    # --- ขั้น 6: กำจัดจุดรบกวน ---
    sensitivity = params.despeckle if params.despeckle is not None else 0.0
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    mask = despeckle(mask, sensitivity)
    auto_values["min_component_area"] = int(
        min(0.0002 * mask.size * (1.0 + sensitivity * 24.0), mask.size * 0.02)
    )

    # --- ขั้น 7: ซ่อมเส้นขาด ---
    rounds = _resolve_close_rounds(params.close_rounds)
    mask = close_gaps(mask, rounds)
    auto_values["close_rounds"] = rounds

    # --- ขั้น 8: ปิดช่องว่าง ---
    do_fill = params.close_holes if params.close_holes is not None else True
    if do_fill:
        mask = fill_closed_shapes(mask)
    auto_values["close_holes"] = bool(do_fill)

    # --- กันก้อนทึบขนาดใหญ่หาย (ความเสี่ยงข้อ 2) ---
    mask = restore_large_blobs(original_mask, mask)

    # --- ขั้น 9: ครอปช่องว่างก่อนหนาเส้น ---
    # ต้อง crop ก่อน เพราะต้องรู้ขนาดสุดท้ายของภาพถึงจะคำนวณสเกล
    # การขยายขึ้นหน้ากระดาษได้ถูกต้อง
    mask = crop_to_ink(mask)

    # --- ขั้น 10: หนาเส้นให้สม่ำเสมอด้วยการวัดจริง ---
    # เป้าหมายกำหนดเป็น "ความหนาบนหน้ากระดาษ" แล้วย้อนกลับมาหารด้วย
    # สเกลการขยาย ทำให้ได้เส้นหนาเท่ากันบนกระดาษไม่ว่าภาพต้นฉบับ
    # จะเล็กหรือใหญ่เท่าไร
    target_page_px = _resolve_target_px(params.target_line_mm)
    scale = 1.0
    if output_width_px and mask.shape[1] > 0:
        scale = max(0.05, output_width_px / float(mask.shape[1]))
    target_px = target_page_px / scale

    stroke = measure_mod.plan_thickness(mask, target_px)
    mask = measure_mod.apply_thickness(mask, stroke.kernel_px)
    stroke.measured_px = round(stroke.measured_px, 2)
    stroke.target_px = round(stroke.target_px, 2)
    auto_values["source_width_px"] = int(mask.shape[1])
    auto_values["page_scale"] = round(scale, 3)
    auto_values["target_line_px_on_page"] = round(target_page_px, 1)
    auto_values["measured_line_px"] = stroke.measured_px
    auto_values["dilate_kernel"] = stroke.kernel_px
    if not stroke.reliable and stroke.note:
        warnings.append(stroke.note)

    # --- ขั้น 11: ทำให้ขอบเรียบหลังขยาย ---
    mask = _anti_alias(mask)

    return ConvertResult(
        mask=mask,
        detection=detection,
        stroke=stroke,
        auto_values=auto_values,
        warnings=warnings,
    )


def _target_line_mm_default() -> float:
    from ..config import TARGET_LINE_MM

    return float(TARGET_LINE_MM)


def _ink_ratio(mask: np.ndarray) -> float:
    return float((mask > 127).mean())


def _threshold_lineart(norm_u8: np.ndarray, sigma: float) -> np.ndarray:
    """threshold สำหรับภาพที่เป็นลายเส้นอยู่แล้ว

    ภาพลายเส้นมี histogram ที่ bimodal ชัดเจน จึงใช้ Otsu หลังปรับแสง
    ได้ขอบที่คมกว่า XDoG มาก

    **ข้อควรระวัง:** ต้องไม่ blur มากกว่า ~1 พิกเซล ก่อน threshold
    เพราะเส้นในภาพลายเส้นบางเพียง 2–4 พิกเซล การ blur ด้วย sigma ใหญ่
    เช่น 5 จะลบเส้นหายไปทั้งหมด เหลือแต่ก้อนทึบใหญ่ ๆ (ตาของการ์ตูน)
    """
    smooth = cv2.GaussianBlur(norm_u8, (0, 0), min(max(sigma, 0.0), 1.0))
    _, mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _estimate_source_stroke(norm_u8: np.ndarray) -> float:
    """ประมาณความหนาเส้นของภาพต้นฉบับก่อนปรับขนาด

    ใช้ตัดสินว่าค่า sigma ของ XDoG และการเลือกวิธีดึงเส้น
    คืนค่า 0.0 ถ้าหาเส้นไม่พบ
    """
    mask = _threshold_lineart(norm_u8, 0.6)
    return measure_mod.measure_stroke_width(mask)


def _anti_alias(mask: np.ndarray, scale: int = 2) -> np.ndarray:
    """ทำให้ขอบเส้นเรียบขึ้นหลัง dilate

    dilate ด้วยเคอร์เนลวงรีแบบ binary ทำให้ขอบเป็นขั้น เมื่อนำไปย่อ
    ลง A4 จะเห็นรอยหยัก การ dilate ที่ภาพใหญ่กว่าแล้วย่อลงมาช่วยให้
    ขอบเรียบขึ้น
    """
    if scale <= 1:
        return mask
    height, width = mask.shape[:2]
    big = cv2.resize(
        mask, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    big = cv2.morphologyEx(big, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.resize(big, (width, height), interpolation=cv2.INTER_AREA)


def fit_into(mask: np.ndarray, box_w: int, box_h: int) -> tuple[np.ndarray, int, int]:
    """ย่อ/ขยายภาพลายเส้นให้พอดีกรอบแบบ ``contain`` (ไม่ยืดสัดส่วน)

    :param mask: ภาพลายเส้น (255 = หมึก)
    :returns: ``(ภาพที่วางแล้ว, offset_x, offset_y)`` โดยภาพผลลัพธ์เป็น
        ขนาดเท่ากรอบพอดี **พร้อมกลับสีตามข้อตกลงของหน้ากระดาษแล้ว**
        คือ 0 = หมึก, 255 = พื้นหลัง เพื่อให้ผู้เรียกใช้วางลงหน้าได้ตรง ๆ
        โดยไม่ต้องกลับสีเอง (การกลับสีผิดที่ทำให้ภาพออกมาเป็นลบ)
    """
    height, width = mask.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("ภาพว่างเปล่า")

    scale = min(box_w / float(width), box_h / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    off_x = (box_w - new_w) // 2
    off_y = (box_h - new_h) // 2

    # กลับสีจากข้อตกลงของภาพลายเส้น (255=หมึก) เป็นข้อตกลงของหน้า (0=หมึก)
    ink = 255 - resized

    # พื้นที่ว่างรอบภาพต้องเป็น "ไม่มีหมึก" (255) ไม่ใช่สีดำ
    canvas = np.full((box_h, box_w), 255, dtype=np.uint8)
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = ink
    return canvas, off_x, off_y
