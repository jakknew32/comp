"""ตั้งค่า AI จากตัวแปรแวดล้อมของเซิร์ฟเวอร์

โปรแกรมไม่ทำงาน AI ให้อยู่ดี ๆ ต้องมีคีย์ของผู้ใช้เสมอ
การไม่มีคีย์ไม่ถือเป็นข้อผิดพลาด เพียงแต่ปิดฟีเจอร์นี้ไว้
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

# ตัวแปรแวดล้อมที่ใช้ตั้งค่า
ENV_AI_PROVIDER = "AI_PROVIDER"
ENV_API_KEY = "AI_API_KEY"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_MODEL = "AI_MODEL"
ENV_HF_MODEL = "AI_HF_MODEL"
ENV_BASE_URL = "AI_BASE_URL"
ENV_HF_BASE_URL = "AI_HF_BASE_URL"
ENV_HF_PROVIDER = "AI_HF_PROVIDER"
ENV_TIMEOUT = "AI_TIMEOUT_SECONDS"

# ผู้ให้บริการฟรีที่ใช้เป็นตัวสำรอง
ENV_POLLINATIONS_TOKEN = "POLLINATIONS_TOKEN"
ENV_POLLINATIONS_BASE_URL = "POLLINATIONS_BASE_URL"
ENV_POLLINATIONS_MODEL = "POLLINATIONS_MODEL"
ENV_CF_TOKEN = "CF_API_TOKEN"
ENV_CF_ACCOUNT_ID = "CF_ACCOUNT_ID"
ENV_CF_MODEL = "CF_IMAGE_MODEL"

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_HF_MODEL = "lineart_sd15"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# Hugging Face ย้ายเข้าร้านค้า unified และปิด api-inference.huggingface.co ไปแล้ว
# ตัวเก่าไม่มี DNS อีกต่อไป ต้องใช้ router.huggingface.co แทน
# และต้องมี path /hf-inference/models ต่อท้ายด้วย
# ถ้าตัด path นี้ออก จะได้ 404 "model not found" ทั้งที่โมเดลมีอยู่จริง
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/hf-inference/models"
DEFAULT_TIMEOUT = 90.0

# ผู้ให้บริการฟรีที่ใช้เป็นตัวสำรอง ไม่ผูกกับเครดิตรายเดือนของผู้ใช้รายคน
DEFAULT_POLLINATIONS_BASE_URL = "https://image.pollinations.ai"
DEFAULT_POLLINATIONS_MODEL = "turbo"
DEFAULT_CF_IMAGE_MODEL = "@cf/black-forest-labs/FLUX.1-schnell"

# ประมาณการค่าใช้จ่ายต่อหนึ่งภาพ ใช้แสดงในหน้าเว็บให้ผู้ใช้ตัดสินใจ
# ตัวเลขนี้มาจากหน้าราคาของ Google/Hugging Face ณ เวลาที่เขียนโค้ด และอาจเปลี่ยนได้
# จึงต้องบอกผู้ใช้เสมอว่าให้เช็คราคาล่าสุดที่ต้นทาง
ESTIMATED_COST_PER_IMAGE_USD = 0.067

PROMPT_TH = (
    "แปลงภาพนี้ให้เป็นภาพระบายสีสำหรับเด็ก "
    "ต้องเป็นภาพเส้นดำบนพื้นหลังขาวเท่านั้น "
    "เส้นต้องหนาและเรียบร้อย เหมือนภาพวาดด้วยมือ "
    "ห้ามมีสี ห้ามมีเงา ไม่มีพื้นหลังสีหรือพื้นหลังมีลาย "
    "รักษารูปร่างและรายละเอียดของสิ่งที่อยู่ในภาพเดิมไว้ทั้งหมด"
)


class AiProvider(str, Enum):
    """ผู้ให้บริการ AI"""
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    POLLINATIONS = "pollinations"
    CLOUDFLARE = "cloudflare"

    @classmethod
    def parse(cls, value: object) -> "AiProvider":
        """แปลงข้อความเป็นผู้ให้บริการ ถ้าไม่รู้จักให้ถือว่าเป็น Gemini

        ค่ามาจากตัวแปรแวดล้อมหรือจากหน้าเว็บ จึงต้องไม่ทำให้เซิร์ฟเวอร์ล้ม
        """
        try:
            return cls((value or "").strip().lower())
        except ValueError:
            return cls.GEMINI


# ลำดับที่ลองใหม่เมื่อเจ้าหลักล้ม เรียงจากที่ไม่เสียเงินและไม่ต้องมีคีย์ก่อน
# เพราะถ้าเจ้าหลักคือเจ้าที่เครดิตจำกัด เว็บจะล้มตามเจ้านั้นไปตลอด
# ตัวที่เสียเงินจริงอย่าง Gemini ต้องอยู่ท้าย ใช้เมื่อทางอื่นไม่เหลือ
FALLBACK_ORDER = (
    AiProvider.POLLINATIONS,
    AiProvider.CLOUDFLARE,
    AiProvider.GEMINI,
    AiProvider.HUGGINGFACE,
)

# ข้อความบอกผู้ใช้เมื่อทุกเจ้าล้มหมด ต้องบอกทางออกที่ทำได้จริง
NO_PROVIDER_READY_MESSAGE = (
    "ยังสร้างภาพด้วย AI ไม่ได้ เพราะไม่มีผู้ให้บริการที่พร้อมใช้งาน "
    "โปรแกรมนี้มีทางฟรีที่ใช้ได้ทันทีคือ Pollinations "
    "ให้ตั้ง AI_PROVIDER=pollinations (หรือเติมโทเคนของเจ้าที่ต้องการ) แล้ว deploy ใหม่"
)


@dataclass
class AiSettings:
    """ผลการอ่านตั้งค่า AI จากสภาพแวดล้อม"""

    provider: AiProvider = AiProvider.GEMINI
    api_key: str | None = None
    hf_token: str | None = None
    model: str = DEFAULT_MODEL
    hf_model: str = "lineart_sd15"
    base_url: str = DEFAULT_BASE_URL
    hf_base_url: str = DEFAULT_HF_BASE_URL
    # ผู้ให้บริการสร้างภาพผ่าน Hugging Face Inference Providers: "auto" = ให้ HF เลือกตัวแรกที่ใช้ได้
    # หรือระบุเอง เช่น fal-ai, together, replicate
    hf_provider: str = "auto"
    # ผู้ให้บริการฟรีที่ใช้เป็นตัวสำรอง ไม่ต้องมีคีย์ก็เรียกได้
    pollinations_token: str | None = None
    pollinations_base_url: str = DEFAULT_POLLINATIONS_BASE_URL
    pollinations_model: str = DEFAULT_POLLINATIONS_MODEL
    cf_token: str | None = None
    cf_account_id: str | None = None
    cf_model: str = DEFAULT_CF_IMAGE_MODEL
    # ค่านี้บอกว่าให้ลองเจ้าอื่นต่อเมื่อเจ้าหลักล้ม ปิดได้ถ้าต้องการความคาดเดาไม่ได้
    allow_fallback: bool = True
    # จุดเริ่มของเลขสุ่ม ทำให้คำสั่งเดิมได้ภาพต่างกันไปเรื่อย ๆ
    seed_base: int = 12345
    timeout: float = DEFAULT_TIMEOUT

    def is_ready(self, provider: "AiProvider | None" = None) -> bool:
        """เจ้านี้เรียกใช้ได้จริงไหม

        Pollinations ใช้ได้เลยโดยไม่ต้องมีคีย์ จึงเป็นตัวสำรองที่ดีที่สุด
        ค่าอื่นต้องดูว่ามีคีย์ของเจ้านั้น ๆ อยู่จริง
        """
        target = provider or self.provider
        if target == AiProvider.GEMINI:
            return bool(self.api_key)
        if target == AiProvider.HUGGINGFACE:
            return bool(self.hf_token)
        if target == AiProvider.POLLINATIONS:
            return True
        if target == AiProvider.CLOUDFLARE:
            return bool(self.cf_token and self.cf_account_id)
        return False

    @property
    def configured(self) -> bool:
        """พร้อมใช้งานหรือยัง

        ต้องเผื่อทางสำรองด้วย เพราะเจ้าหลักอาจไม่ได้ตั้งค่า
        แต่เจ้าฟรียังเรียกได้ เว็บจึงยังสร้างภาพให้ผู้ใช้ได้
        """
        if self.is_ready():
            return True
        return self.allow_fallback and any(self.is_ready(p) for p in FALLBACK_ORDER)

    def describe_missing(self) -> str:
        if self.is_ready():
            return ""
        if self.provider == AiProvider.GEMINI:
            return (
                f"ยังไม่ได้ตั้งค่า AI: ไม่พบตัวแปรแวดล้อม {ENV_API_KEY} "
                "ต้องใส่คีย์ของผู้ให้บริการในการตั้งค่าของ Render "
                "(หรือใช้ผู้ให้บริการฟรีด้วยการตั้ง AI_PROVIDER=pollinations)"
            )
        if self.provider == AiProvider.HUGGINGFACE:
            return (
                f"ยังไม่ได้ตั้งค่า Hugging Face AI: ไม่พบตัวแปรแวดล้อม {ENV_HF_TOKEN} "
                "ต้องใส่ HF_TOKEN ในการตั้งค่าของ Render"
            )
        if self.provider == AiProvider.CLOUDFLARE:
            return (
                "ยังไม่ได้ตั้งค่า Cloudflare Workers AI: ต้องมี "
                f"{ENV_CF_TOKEN} และ {ENV_CF_ACCOUNT_ID} "
                "(สร้างฟรีที่ dash.cloudflare.com)"
            )
        return "ไม่ทราบผู้ให้บริการ AI"

    def masked_key(self) -> str:
        if self.provider == AiProvider.GEMINI and self.api_key:
            return "***" + self.api_key[-4:]
        if self.provider == AiProvider.HUGGINGFACE and self.hf_token:
            return "***" + self.hf_token[-4:]
        if self.provider == AiProvider.POLLINATIONS and self.pollinations_token:
            return "***" + self.pollinations_token[-4:]
        if self.provider == AiProvider.CLOUDFLARE and self.cf_token:
            return "***" + self.cf_token[-4:]
        return ""


def load_settings(env: dict | None = None) -> AiSettings:
    """อ่านค่าจากตัวแปรแวดล้อม

    รับ dict เข้ามาเพื่อให้เทสต์ได้โดยไม่ต้องแก้สภาพแวดล้อมจริง
    """
    source = os.environ if env is None else env

    provider = AiProvider.parse(source.get(ENV_AI_PROVIDER))

    # อ่านคีย์ของทุกเจ้าไว้เสมอ ไม่ใช่เฉพาะเจ้าที่เลือก
    # เพราะต้องใช้เป็นตัวสำรองเมื่อเจ้าหลักล้ม ถ้าอ่านเฉพาะเจ้าเดียว
    # เจ้าอื่นจะกลายเป็นค่าว่างและใช้ไม่ได้เลยตอนที่ต้องการมากที่สุด
    api_key = (source.get(ENV_API_KEY) or "").strip() or None
    hf_token = (source.get(ENV_HF_TOKEN) or "").strip() or None
    pollinations_token = (source.get(ENV_POLLINATIONS_TOKEN) or "").strip() or None
    cf_token = (source.get(ENV_CF_TOKEN) or "").strip() or None
    cf_account_id = (source.get(ENV_CF_ACCOUNT_ID) or "").strip() or None

    model = (source.get(ENV_MODEL) or "").strip() or DEFAULT_MODEL
    hf_model = (source.get(ENV_HF_MODEL) or "").strip() or DEFAULT_HF_MODEL
    base_url = (source.get(ENV_BASE_URL) or "").strip() or DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")
    # Hugging Face ใช้ค่าคนละชื่อกับ AI_BASE_URL เพราะเป็นคนละผู้ให้บริการ
    hf_base_url = (source.get(ENV_HF_BASE_URL) or "").strip() or DEFAULT_HF_BASE_URL
    hf_base_url = hf_base_url.rstrip("/")

    hf_provider = (source.get(ENV_HF_PROVIDER) or "").strip().lower() or "auto"

    pollinations_base_url = (
        (source.get(ENV_POLLINATIONS_BASE_URL) or "").strip()
        or DEFAULT_POLLINATIONS_BASE_URL
    ).rstrip("/")
    pollinations_model = (
        (source.get(ENV_POLLINATIONS_MODEL) or "").strip().lower()
        or DEFAULT_POLLINATIONS_MODEL
    )
    cf_model = (source.get(ENV_CF_MODEL) or "").strip() or DEFAULT_CF_IMAGE_MODEL

    # AI_ALLOW_FALLBACK=false เมื่อต้องการให้ล้มที่เจ้าหลักเลย ไม่เงียบ ๆ ไปใช้เจ้าอื่น
    allow_fallback = _to_bool(source.get("AI_ALLOW_FALLBACK"), True)

    timeout_raw = (source.get(ENV_TIMEOUT) or "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    timeout = min(max(timeout, 5.0), 300.0)

    return AiSettings(
        provider=provider,
        api_key=api_key,
        hf_token=hf_token,
        model=model,
        hf_model=hf_model,
        base_url=base_url,
        hf_base_url=hf_base_url,
        hf_provider=hf_provider,
        pollinations_token=pollinations_token,
        pollinations_base_url=pollinations_base_url,
        pollinations_model=pollinations_model,
        cf_token=cf_token,
        cf_account_id=cf_account_id,
        cf_model=cf_model,
        allow_fallback=allow_fallback,
        timeout=timeout,
    )


def _to_bool(value: object, default: bool) -> bool:
    """อ่านค่าความจริงจากตัวแปรแวดล้อม

    ค่าจาก Render มักเป็นข้อความ ไม่ใช่ bool จริง จึงต้องแปลงเอง
    """
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return default