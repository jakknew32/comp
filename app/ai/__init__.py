"""การเชื่อมต่อ AI สองด้าน: แปลงภาพถ่ายเป็นลายเส้น และสร้างภาพจากข้อความ

โปรแกรมนี้ทำงานได้ครบโดยไม่ต้องมี AI เลย
ส่วนนี้เป็นขั้นตอนเสริมซึ่งมีเงื่อนไขเหมือนกันทุกฟังก์ชัน:
- ต้องมีคีย์ของผู้ให้บริการ เก็บเป็นตัวแปรแวดล้อมของเซิร์ฟเวอร์
- เสียค่าใช้จ่ายต่อหนึ่งภาพ
- เรียกผ่านอินเทอร์เน็ต ทำให้ภาพออกจากเครื่องผู้ใช้
- ถ้าเรียกไม่สำเร็จต้องกลับไปใช้การประมวลผลเดิมต่อได้เสมอ
"""

from __future__ import annotations

import numpy as np

from ..lineart import detect
from . import gemini, generate, huggingface
from .gemini import AiError
from .generate import (
    GenerateResult,
    STYLE_PRESETS,
    build_prompt,
    generate_image,
    list_styles,
)
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
    "GenerateResult",
    "STYLE_PRESETS",
    "build_prompt",
    "generate_image",
    "list_styles",
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
    provider = (
        "huggingface"
        if settings.provider == AiProvider.HUGGINGFACE
        else "gemini"
    )
    model = (
        settings.hf_model if provider == "huggingface" else settings.model
    )
    cost = (
        0.0 if provider == "huggingface" else ESTIMATED_COST_PER_IMAGE_USD
    )
    return {
        "configured": settings.configured,
        "provider": provider,
        "model": model if settings.configured else None,
        "estimated_cost_per_image_usd": (
            cost if settings.configured else None
        ),
        "message": None if settings.configured else settings.describe_missing(),
        "styles": list_styles(),
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
    if not settings.configured:
        return AiOutcome(False, image, settings.describe_missing())
    if detect.analyze(image).is_line_art:
        return AiOutcome(False, image)

    chosen = provider if provider is not None else settings.provider
    try:
        if chosen == AiProvider.HUGGINGFACE:
            converted = huggingface.hf_convert_to_lineart(
                image,
                settings,
                prompt=prompt,
                model_key=model or settings.hf_model,
            )
        else:
            call = AiSettings(
                provider=settings.provider,
                api_key=settings.api_key,
                hf_token=settings.hf_token,
                base_url=settings.base_url,
                timeout=settings.timeout,
                model=model or settings.model,
                hf_model=settings.hf_model,
            )
            converted = gemini.convert_to_lineart(image, call, prompt)
    except AiError as exc:
        return AiOutcome(False, image, f"ใช้ AI ไม่สำเร็จ: {exc}")
    except Exception as exc:  # noqa: BLE001 - ต้องกันไม่ให้ขั้นตอนอื่นพัง
        return AiOutcome(False, image, f"ใช้ AI ไม่สำเร็จ: {type(exc).__name__}")

    if converted.size == 0:
        return AiOutcome(False, image, "AI คืนภาพที่ว่างเปล่า")

    return AiOutcome(True, converted)
