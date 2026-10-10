"""การเชื่อมต่อ AI สองด้าน: แปลงภาพถ่ายเป็นลายเส้น และสร้างภาพจากข้อความ

โปรแกรมนี้ทำงานได้ครบโดยไม่ต้องมี AI เลย
ส่วนนี้เป็นขั้นตอนเสริมซึ่งมีเงื่อนไขเหมือนกันทุกฟังก์ชัน:
- ต้องมีคีย์ของผู้ให้บริการ เก็บเป็นตัวแปรแวดล้อมของเซิร์ฟเวอร์
- เสียค่าใช้จ่ายต่อหนึ่งภาพ
- เรียกผ่านอินเทอร์เน็ต ทำให้ภาพออกจากเครื่องผู้ใช้
- ถ้าเรียกไม่สำเร็จต้องกลับไปใช้การประมวลผลเดิมต่อได้เสมอ
"""

from __future__ import annotations

import logging

import numpy as np

from ..lineart import detect
from . import gemini, generate, huggingface  # noqa: F401 - generate ใช้ re-export ผ่านแพ็กเกจนี้
from .gemini import AiError

logger = logging.getLogger("coloring_book.ai")
from .generate import (
    GenerateResult,
    STYLE_PRESETS,
    build_prompt,
    generate_image,
    list_styles,
)
from .settings import (
    ESTIMATED_COST_PER_IMAGE_USD,
    NO_PROVIDER_READY_MESSAGE,
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
    provider = settings.provider.value
    model = settings.model
    cost = ESTIMATED_COST_PER_IMAGE_USD

    # เจ้าที่พร้อมใช้งานทั้งหมด เรียงตามลำดับที่จะถูกเรียกจริง
    # ให้ผู้ดูแลเห็นได้ว่าเว็บจะล้มไปใช้เจ้าไหนถ้าเจ้าหลักใช้ไม่ได้
    chain = generate.provider_chain(settings)

    # ต้องรายงานตามเจ้าที่จะเรียกจริงตัวแรก ไม่ใช่เจ้าที่ตั้งไว้
    # ถ้าไม่งั้นจะโชว์ว่าเสียเงินต่อภาพ ทั้งที่จริง ๆ ใช้เจ้าฟรี
    effective = chain[0] if chain else settings.provider
    provider, model, cost = _describe_provider(settings, effective)

    return {
        "configured": settings.configured,
        "provider": provider,
        "requested_provider": settings.provider.value,
        "model": model if settings.configured else None,
        "estimated_cost_per_image_usd": cost if settings.configured else None,
        "message": (
            None
            if settings.configured
            else (settings.describe_missing() or NO_PROVIDER_READY_MESSAGE)
        ),
        "endpoint": _endpoint_for(settings, effective),
        "styles": list_styles(),
        "fallback_chain": [p.value for p in chain],
        "free_providers_available": [
            p.value
            for p in (
                AiProvider.POLLINATIONS,
                AiProvider.CLOUDFLARE,
            )
            if settings.is_ready(p)
        ],
    }


def _describe_provider(
    settings: AiSettings,
    provider: AiProvider,
) -> tuple[str, str, float]:
    """ชื่อเจ้า โมเดล และค่าใช้จ่ายต่อภาพ ของเจ้านั้น

    ผู้ให้บริการฟรีไม่คิดเงิน ต้องรายงานเป็นศูนย์จริง ๆ
    ไม่ใช่ราคาของเจ้าที่ตั้งไว้แต่ไม่ได้ใช้
    """
    if provider == AiProvider.HUGGINGFACE:
        return provider.value, settings.hf_model, 0.0
    if provider == AiProvider.POLLINATIONS:
        return provider.value, settings.pollinations_model, 0.0
    if provider == AiProvider.CLOUDFLARE:
        return provider.value, settings.cf_model, 0.0
    return provider.value, settings.model, ESTIMATED_COST_PER_IMAGE_USD


def _endpoint_for(settings: AiSettings, provider: AiProvider) -> str:
    """URL ที่จะยิงจริง ใช้แสดงให้ผู้ดูแลเช็คว่าตั้งค่าถูกที่"""
    return {
        AiProvider.GEMINI: settings.base_url,
        AiProvider.HUGGINGFACE: settings.hf_base_url,
        AiProvider.POLLINATIONS: settings.pollinations_base_url,
        AiProvider.CLOUDFLARE: "https://api.cloudflare.com/client/v4",
    }.get(provider, "")


def should_use_ai(
    image: np.ndarray,
    settings: AiSettings,
    enabled: bool,
) -> bool:
    """ตัดสินว่าควรเรียก AI หรือไม่

    เรียกเฉพาะภาพที่ไม่ใช่ภาพลายเส้นเท่านั้น
    เพราะภาพลายเส้นผ่านการประมวลผลเดิมได้ดีอยู่แล้ว
    การยิง AI ในกรณีนี้เป็นการเสียเงินเปล่า

    ต้องเช็คเจ้าที่รับภาพแนบด้วย ไม่ใช่แค่ configured
    เพราะ configured จะเป็นจริงเมื่อมี Pollinations ซึ่งทำงานนี้ไม่ได้
    """
    if not enabled:
        return False
    if not _lineart_chain(settings, settings.provider):
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
    # งานนี้ต้องใช้เจ้าที่รับภาพแนบ ไม่ใช่แค่เช็ค configured
    # เพราะ Pollinations อาจทำให้ configured เป็นจริง แต่แปลงภาพไม่ได้
    if not _lineart_chain(settings, settings.provider):
        return AiOutcome(
            False,
            image,
            "การแปลงภาพด้วย AI ยังใช้ไม่ได้ ต้องมี AI_API_KEY หรือ HF_TOKEN "
            "เพราะต้องส่งภาพต้นทางไปให้เจ้านั้นวาดใหม่",
        )
    if detect.analyze(image).is_line_art:
        return AiOutcome(False, image)

    chosen = provider if provider is not None else settings.provider
    problems: list[str] = []

    # แปลงภาพต้องส่งภาพต้นทางไปด้วย จึงใช้ได้เฉพาะเจ้าที่รับภาพแนบ
    # Pollinations กับ Cloudflare รับได้แค่ข้อความ จึงไม่อยู่ในรายการนี้
    for index, current in enumerate(_lineart_chain(settings, chosen)):
        try:
            if current == AiProvider.HUGGINGFACE:
                converted = huggingface.hf_convert_to_lineart(
                    image,
                    settings,
                    prompt=prompt,
                    model_key=model or settings.hf_model,
                )
            else:
                call = AiSettings(
                    provider=current,
                    api_key=settings.api_key,
                    hf_token=settings.hf_token,
                    base_url=settings.base_url,
                    hf_base_url=settings.hf_base_url,
                    timeout=settings.timeout,
                    model=model or settings.model,
                    hf_model=settings.hf_model,
                )
                converted = gemini.convert_to_lineart(image, call, prompt)
        except AiError as exc:
            problems.append(f"{current.value}: {exc}")
            logger.warning("แปลงภาพด้วย %s ไม่สำเร็จ: %s", current.value, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - ต้องกันไม่ให้ขั้นตอนอื่นพัง
            problems.append(f"{current.value}: {type(exc).__name__}: {str(exc)[:160]}")
            logger.exception("เรียก AI ไม่สำเร็จ")
            if isinstance(exc, (AttributeError, NameError, TypeError)):
                problems.append("ดูเหมือนเซิร์ฟเวอร์ยังรันโค้ดเก่าอยู่ ต้อง deploy ใหม่")
            continue

        if converted.size == 0:
            problems.append(f"{current.value}: AI คืนภาพที่ว่างเปล่า")
            continue

        return AiOutcome(True, converted)

    detail = " | ".join(problems) if problems else "ไม่ทราบสาเหตุ"
    return AiOutcome(False, image, f"ใช้ AI ไม่สำเร็จ: {detail}")


def _lineart_chain(
    settings: AiSettings,
    chosen: AiProvider,
) -> list[AiProvider]:
    """เจ้าที่รองรับการแปลงภาพเป็นลายเส้น เรียงตามที่ผู้ใช้เลือก

    ต่างจากการสร้างภาพใหม่ตรงที่ไม่มีเจ้าใดทำได้ฟรีและไม่ต้องมีคีย์
    งานนี้จึงต้องพึ่ง Gemini หรือ Hugging Face อย่างใดอย่างหนึ่ง
    """
    chain: list[AiProvider] = []
    if settings.is_ready(chosen):
        chain.append(chosen)
    for provider in (AiProvider.GEMINI, AiProvider.HUGGINGFACE):
        if provider not in chain and settings.is_ready(provider):
            chain.append(provider)
    return chain
