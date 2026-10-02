"""การเชื่อมต่อ AI เพื่อช่วยแปลงภาพถ่ายเป็นภาพลายเส้น

โปรแกรมนี้ทำงานได้ครบโดยไม่ต้องมี AI เลย
ส่วนนี้เป็นขั้นตอนเสริมที่ช่วยจัดการกับภาพถ่ายจริง
ซึ่ง OpenCV ทำได้แค่ได้เส้นตามภาพต้นฉบับ ไม่ได้ภาพการ์ตูนน่ารัก

การเรียกใช้มีเงื่อนไขสำคัญ:
- ต้องมีคีย์ของผู้ให้บริการ เก็บเป็นตัวแปรแวดล้อมของเซิร์ฟเวอร์
- เสียค่าใช้จ่ายต่อหนึ่งภาพ
- เรียกผ่านอินเทอร์เน็ต ทำให้ภาพออกจากเครื่องผู้ใช้
- ถ้าเรียกไม่สำเร็จต้องกลับไปใช้การประมวลผลเดิมต่อได้เสมอ
"""

from __future__ import annotations

import numpy as np

from ..lineart import detect
from . import gemini, huggingface
from .gemini import AiError
from .settings import (
    ESTIMATED_COST_PER_IMAGE_USD,
    PROMPT_TH,
    AiProvider,
    AiSettings,
    load_settings,
)

__all__ = [
    "AiError",
    "AiProvider",
    "AiSettings",
    "load_settings",
    "PROMPT_TH",
    "ESTIMATED_COST_PER_IMAGE_USD",
    "AiOutcome",
    "enhance",
    "status",
]


class AiOutcome:
    """ผลการพยายามใช้ AI

    ต้องแยกระหว่างสถานะสำเร็จ กับล้มเหลว เพราะผู้ใช้ต้องได้รับรู้
    และต้องเห็นว่าค่าใช้จ่ายถูกใช้ไปหรือไม่
    """

    def __init__(
        self,
        used: bool,
        image: np.ndarray,
        note: str | None = None,
    ) -> None:
        self.used = used
        self.image = image
        self.note = note

    @property
    def failed(self) -> bool:
        return not self.used and self.note is not None


def status(settings: AiSettings) -> dict:
    """สรุปสถานะ AI สำหรับหน้าเว็บ

    หน้าเว็บต้องซ่อนตัวเลือกนี้เมื่อยังตั้งค่าไม่เรียบร้อย
    ไม่ให้ผู้ใช้กดแล้วล้มเหลวโดยไม่รู้ตัว
    """
    if settings.provider == AiProvider.HUGGINGFACE:
        return {
            "configured": settings.configured,
            "provider": "huggingface",
            "model": settings.hf_model if settings.configured else None,
            "estimated_cost_per_image_usd": (
                0.0 if settings.configured else None
            ),
            "message": None if settings.configured else settings.describe_missing(),
        }

    return {
        "configured": settings.configured,
        "provider": "gemini",
        "model": settings.model if settings.configured else None,
        "estimated_cost_per_image_usd": (
            ESTIMATED_COST_PER_IMAGE_USD if settings.configured else None
        ),
        "message": None if settings.configured else settings.describe_missing(),
    }


def should_use_ai(
    image: np.ndarray,
    settings: AiSettings,
    enabled: bool,
) -> bool:
    """ตัดสินว่าควรเรียก AI หรือไม่

    เรียกเฉพาะภาพที่ไม่ใช่ภาพลายเส้นเท่านั้น
    เพราะภาพลายเส้นผ่านการประมวลผลเดิมได้ดีอยู่แล้ว
    การยิง AI ในกรณีนี้เป็นการเสียเงินเปล่า
    """
    if not enabled or not settings.configured:
        return False
    return not detect.analyze(image).is_line_art


def enhance(
    image: np.ndarray,
    settings: AiSettings,
    enabled: bool,
    prompt: str = PROMPT_TH,
    provider: AiProvider | None = None,
    model: str | None = None,
) -> "AiOutcome":
    """พยายามแปลงภาพด้วย AI โดยไม่ทำให้ขั้นตอนอื่นล้มเหลว

    คืนภาพเดิมกลับไปเสมอถ้า AI ใช้ไม่ได้ พร้อมข้อความอธิบาย
    """
    if not enabled:
        return AiOutcome(False, image)

    # ใช้ provider/model จากพารามิเตอร์ ถ้าระบุ มิฉะนั้นใช้จาก settings
    effective_provider = provider if provider is not None else settings.provider
    effective_model = model if model is not None else (
        settings.hf_model if settings.provider == AiProvider.HUGGINGFACE else settings.model
    )

    # สร้าง settings ชั่วคราวสำหรับการเรียกครั้งนี้
    effective_settings = AiSettings(
        provider=AiProvider(effective_provider),
        api_key=settings.api_key,
        hf_token=settings.hf_token,
        model=settings.model,
        hf_model=settings.hf_model,
        base_url=settings.base_url,
        timeout=settings.timeout,
    )
    # Override model for the specific provider
    if effective_provider == AiProvider.HUGGINGFACE:
        effective_settings.hf_model = model or settings.hf_model
    else:
        effective_settings.model = model or settings.model

    if not enabled:
        return AiOutcome(False, image)

    if not settings.configured:
        return AiOutcome(False, image, settings.describe_missing())

    if detect.analyze(image).is_line_art:
        return AiOutcome(False, image)

    try:
        if effective_provider == AiProvider.HUGGINGFACE:
            from . import huggingface
            converted = huggingface.hf_convert_to_lineart(
                image, effective_settings, prompt=prompt, model_key=effective_settings.hf_model
            )
        else:
            from . import gemini
            converted = gemini.convert_to_lineart(image, effective_settings, prompt)
    except AiError as exc:
        return AiOutcome(False, image, f"ใช้ AI ไม่สำเร็จ: {exc}")
    except Exception as exc:  # noqa: BLE001 - ต้องกันไม่ให้ขั้นตอนอื่นพัง
        return AiOutcome(False, image, f"ใช้ AI ไม่สำเร็จ: {type(exc).__name__}")

    if converted.size == 0:
        return AiOutcome(False, image, "AI คืนภาพที่ว่างเปล่า")

    return AiOutcome(True, converted)