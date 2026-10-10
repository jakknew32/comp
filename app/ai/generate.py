"""สร้างภาพระบายสีจากข้อความด้วย AI

ต่างจากโมดูลแปลงภาพ (enhance) ตรงที่ไม่ต้องมีภาพต้นฉบับ
ผู้ใช้พิมพ์คำบรรยาย เช่น "แมวการ์ตูนใส่หมวก" แล้ว AI วาดภาพระบายสีใหม่ให้ทั้งหมด

ข้อจำกัดเดียวกับการแปลงภาพ: ต้องมีคีย์ผู้ให้บริการเสมอ
ถ้าไม่มีคีย์ ฟีเจอร์นี้จะไม่แสดงบนหน้าเว็บ
"""

from __future__ import annotations

import base64
import json
import logging
import secrets
import urllib.error
import urllib.request

import cv2
import numpy as np

from .gemini import (
    AiError,
    _decode_image,
    _extract_image,
    _read_http_error,
    describe_connection_error,
)
from . import freebies, huggingface
from .settings import (
    FALLBACK_ORDER,
    NO_PROVIDER_READY_MESSAGE,
    AiProvider,
    AiSettings,
)

logger = logging.getLogger("coloring_book.ai")

# สไตล์สำเร็จรูป แต่ละสไตล์คือคำสั่งเสริมที่แปะต่อท้ายคำบรรยายของผู้ใช้
# ผู้ใช้ไม่ต้องรู้ศัพท์เทคนิค ก็ได้ภาพตามระดับอายุที่ต้องการ
STYLE_PRESETS: dict[str, dict[str, str]] = {
    "kids_easy": {
        "label": "เด็กเล็ก (เส้นหนา ง่ายมาก)",
        "extra": (
            "สำหรับเด็กอนุบาล เส้นขอบหนามากและชัดเจน ชิ้นส่วนน้อย "
            "พื้นที่ว่างสำหรับระบายสีกว้าง ไม่มีรายละเอียดเล็ก ๆ"
        ),
    },
    "classic": {
        "label": "มาตรฐาน (เหมาะกับเด็กโต)",
        "extra": (
            "เส้นขอบหนาปานกลาง รายละเอียดพอดี "
            "มีส่วนประกอบหลักชัดเจน เหมาะกับเด็กประถม"
        ),
    },
    "detailed": {
        "label": "ละเอียด (ผู้ใหญ่ระบายคลายเครียด)",
        "extra": (
            "ลายเส้นละเอียดสวยงาม มีลุคแบบหนังสือระบายสีผู้ใหญ่ "
            "แบ่งพื้นที่ระบายสีเป็นช่อง ๆ จำนวนมาก แต่เส้นต้องไม่ทับกันจนดูงง"
        ),
    },
    "kawaii": {
        "label": "การ์ตูนน่ารัก",
        "extra": (
            "สไตล์การ์ตูนน่ารัก ตัวละครมีหัวโต ตากลม "
            "อารมณ์ดี้ดี้ เส้นโค้งมนนุ่มนวล"
        ),
    },
    "mandala": {
        "label": "ลวดลายวงกลม (Mandala)",
        "extra": (
            "ลวดลายวงกลมแบบแมนดาลา สมมาตรรอบจุดศูนย์กลาง "
            "ลายซ้ำเป็นชั้น ๆ สวยงามน่าระบาย"
        ),
    },
    "scene": {
        "label": "ฉากและเรื่องราว",
        "extra": (
            "ภาพฉากกว้าง มีพื้นหลัง ตัวละคร และองค์ประกอบหลายอย่าง "
            "เล่าเรื่องราวได้ มีมิติตื้น ๆ ไม่ซับซ้อนเกินการระบาย"
        ),
    },
}

DEFAULT_STYLE = "classic"

# แม่แบบคำสั่งหลัก บังคับให้ผลลัพธ์เป็นภาพระบายสีที่ใช้โปรแกรมนี้ต่อได้
BASE_PROMPT_TH = (
    "วาดภาพระบายสี: {text} — {extra} "
    "ต้องเป็นภาพเส้นดำบนพื้นขาวล้วนเท่านั้น เส้นปิดสนิททุกชิ้น "
    "ห้ามมีสี ห้ามมีเงา ห้ามมีไล่แสง ห้ามมีลายพื้นหลัง "
    "ห้ามมีตัวหนังสือหรือลายน้ำในภาพ"
)


def list_styles() -> list[dict[str, str]]:
    """รายการสไตล์สำหรับหน้าเว็บ"""
    return [
        {"key": key, "label": preset["label"]}
        for key, preset in STYLE_PRESETS.items()
    ]


def build_prompt(text: str, style_key: str | None) -> str:
    """ประกอบคำสั่งเต็มจากคำบรรยายผู้ใช้กับสไตล์ที่เลือก"""
    preset = STYLE_PRESETS.get(style_key or "", STYLE_PRESETS[DEFAULT_STYLE])
    return BASE_PROMPT_TH.format(text=text.strip(), extra=preset["extra"])


def generate_with_gemini(
    prompt: str,
    settings: AiSettings,
) -> np.ndarray:
    """สร้างภาพจากข้อความด้วยโมเดลภาพของ Google

    ใช้ endpoint เดียวกับการแปลงภาพ แต่ส่งข้อความล้วนไม่มีภาพแนบ
    คำตอบที่กลับมาเป็นภาพ inlineData เหมือนเดิม
    """
    if not settings.api_key:
        raise AiError(settings.describe_missing())

    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    url = f"{settings.base_url}/models/{settings.model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.api_key,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise AiError(f"AI ตอบกลับข้อผิดพลาด {exc.code}: {_read_http_error(exc)}") from exc
    except urllib.error.URLError as exc:
        raise AiError(describe_connection_error(url, exc)) from exc
    except TimeoutError as exc:
        raise AiError("AI ตอบนานเกินกำหนดเวลา") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AiError("AI ตอบกลับเป็นข้อมูลที่อ่านไม่ได้") from exc

    return _decode_image(_extract_image(payload))


def _hf_candidate_models(settings: AiSettings) -> list[str]:
    """โมเดลที่จะลองตามลำดับ: ตัวที่เลือกก่อน แล้วตามด้วยตัวสำรอง"""
    key = huggingface.HF_MODEL_ALIASES.get(settings.hf_model, settings.hf_model)
    chosen = huggingface.HF_TEXT2IMAGE_MODELS.get(key, key)
    if "/" not in chosen:
        # ชื่อย่อที่ไม่รู้จัก ใช้ค่าเริ่มต้นแทนเพื่อไม่ให้ยิงด้วยชื่อที่ผิด
        chosen = huggingface.HF_TEXT2IMAGE_MODELS[huggingface.DEFAULT_HF_TEXT2IMAGE]
    ordered = [chosen]
    ordered += [m for m in huggingface.HF_TEXT2IMAGE_FALLBACKS if m != chosen]
    return ordered


def _hf_error_message(status: int | None, detail: str) -> str:
    """แปลง error ของ Hugging Face เป็นข้อความที่บอกวิธีแก้"""
    detail = (detail or "").strip().replace("\n", " ")[:200]
    if status == 401:
        return "HF_TOKEN ไม่ถูกต้องหรือหมดอายุ"
    if status == 403:
        return (
            "HF_TOKEN ไม่มีสิทธิ์เรียก Inference Providers "
            "(ตอนสร้าง token ต้องติ๊ก 'Make calls to Inference Providers')"
        )
    if status == 402:
        return (
            "เครดิต Inference Providers หมด บัญชีฟรีมีเครดิตรายเดือนจำกัด "
            "ต้องเติมเครดิต/สมัคร PRO หรือเปลี่ยนไปใช้ Gemini (AI_PROVIDER=gemini)"
        )
    if status == 429:
        return "เรียกถี่เกินกำหนด (rate limit) กรุณารอสักครู่แล้วลองใหม่"
    return detail or "ไม่ทราบสาเหตุ"


def generate_with_huggingface(
    prompt: str,
    settings: AiSettings,
) -> np.ndarray:
    """สร้างภาพจากข้อความผ่าน Hugging Face Inference Providers

    บริการ hf-inference เดิมเลิกให้บริการโมเดลสร้างภาพแล้ว (ตอบ 410)
    ต้องผ่านผู้ให้บริการรายอื่นที่ HF ต่อให้ จึงเรียกผ่านไลบรารี huggingface_hub
    ที่รู้เส้นทางและรูปแบบคำขอของแต่ละเจ้า แทนการยิง HTTP เอง
    """
    if not settings.hf_token:
        raise AiError(settings.describe_missing())

    try:
        from huggingface_hub import InferenceClient
        from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
    except ImportError as exc:
        raise AiError(
            "ยังไม่ได้ติดตั้งไลบรารี huggingface_hub — เพิ่มลงใน requirements.txt แล้ว deploy ใหม่"
        ) from exc

    client = InferenceClient(
        provider=settings.hf_provider or "auto",
        api_key=settings.hf_token,
        timeout=settings.timeout,
    )

    last_problem = ""
    for model_id in _hf_candidate_models(settings):
        try:
            picture = client.text_to_image(prompt, model=model_id)
        except InferenceTimeoutError as exc:
            raise AiError("AI ตอบนานเกินกำหนดเวลา") from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            message = _hf_error_message(status, str(exc))
            if status in (400, 404, 410, 422):
                # โมเดลนี้ไม่มีผู้ให้บริการรับแล้ว → ลองตัวสำรอง
                last_problem = f"{model_id}: {status} {message}"
                logger.warning("โมเดล %s ใช้ไม่ได้ (%s) ลองตัวถัดไป", model_id, status)
                continue
            raise AiError(f"AI ตอบกลับข้อผิดพลาด {status}: {message}") from exc
        except ValueError as exc:
            # ไม่มีผู้ให้บริการสำหรับโมเดลนี้ในบัญชีของคุณ
            last_problem = f"{model_id}: {str(exc)[:160]}"
            continue

        array = cv2.cvtColor(np.asarray(picture.convert("RGB")), cv2.COLOR_RGB2BGR)
        if array.size == 0:
            raise AiError("AI ไม่ได้ส่งภาพกลับมา")
        return array

    raise AiError(
        "ไม่มีโมเดลสร้างภาพที่ใช้ได้ผ่าน Hugging Face"
        + (f" ({last_problem})" if last_problem else "")
        + " — ลองเปลี่ยนไปใช้ Gemini (AI_PROVIDER=gemini)"
    )


class GenerateResult:
    """ผลการสร้างภาพ แยกสถานะสำเร็จกับล้มเหลวให้หน้าเว็บแสดงได้ชัด"""

    def __init__(
        self,
        image: np.ndarray | None,
        note: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.image = image
        self.note = note
        # ผู้ให้บริการที่ใช้จริง อาจไม่ตรงกับที่ตั้งไว้ถ้าเกิดการสำรอง
        self.provider = provider

    @property
    def ok(self) -> bool:
        return self.image is not None


# จับคู่เจ้ากับชื่อฟังก์ชันของเจ้านั้น
# ต้องอ้างชื่อฟังก์ชันเป็นข้อความ ไม่ใช่เก็บตัวฟังก์ชันไว้ตรง ๆ
# เพราะการเก็บไว้ตรง ๆ จะทำให้เทสต์แทนที่ฟังก์ชันในโมดูลนี้ไม่ได้ผล
_DISPATCH = {
    AiProvider.GEMINI: "generate_with_gemini",
    AiProvider.HUGGINGFACE: "generate_with_huggingface",
    AiProvider.POLLINATIONS: "generate_with_pollinations",
    AiProvider.CLOUDFLARE: "generate_with_cloudflare",
}


def _resolve(provider: AiProvider):
    """หาฟังก์ชันของเจ้านั้น จากโมดูลนี้

    ถ้าเจ้าใหม่ถูกเพิ่มใน settings แต่ลืมเพิ่มใน _DISPATCH
    ให้หยุดตอน import แทนที่จะล้มตอนผู้ใช้กด
    """
    name = _DISPATCH.get(provider)
    if name is None:
        raise AiError(f"ยังไม่ได้เขียนส่วนเรียกผู้ให้บริการ {provider.value}")
    if provider == AiProvider.POLLINATIONS:
        return freebies.generate_with_pollinations
    if provider == AiProvider.CLOUDFLARE:
        return freebies.generate_with_cloudflare
    return globals()[name]


def provider_chain(
    settings: AiSettings,
    preferred: str | None = None,
) -> list[AiProvider]:
    """เรียงลำดับเจ้าที่จะลอง ตัวที่ผู้ใช้เลือกมาก่อนเสมอ

    ตัวที่เลือกไม่ได้ทำงาน (ไม่มีคีย์ หรือเจ้ากำลังล่ม) จะถูกข้ามไป
    ไม่ต้องรอให้ล้มจริงแล้วค่อยเปลี่ยน เพราะจะเสียเวลารอผู้ใช้
    """
    chain: list[AiProvider] = []

    head = AiProvider.parse(preferred) if preferred else settings.provider
    if settings.is_ready(head):
        chain.append(head)

    if settings.allow_fallback:
        for provider in FALLBACK_ORDER:
            if provider not in chain and settings.is_ready(provider):
                chain.append(provider)

    return chain


def generate_image(
    text: str,
    style_key: str | None,
    settings: AiSettings,
    provider: str | None = None,
    model: str | None = None,
) -> GenerateResult:
    """สร้างภาพหนึ่งภาพจากคำบรรยาย คืนผลลัพธ์พร้อมข้อความเตือนเมื่อล้มเหลว

    ถ้าเจ้าที่เลือกไว้ล้มจะลองเจ้าถัดไปในลำดับสำรองโดยอัตโนมัติ
    เพราะเครดิตของแต่ละเจ้าหมดคนละเวลา ถ้ายึดเจ้าเดียวเว็บจะล้มทั้งที่ยังมีทางออก

    ไม่โยน exception ออกมาเอง เพราะผู้ใช้ต้องเห็นว่าล้มเหลวเพราะอะไร
    แล้วกดลองใหม่ได้ทันที
    """
    text = (text or "").strip()
    if not text:
        return GenerateResult(None, "กรุณาพิมพ์คำบรรยายภาพที่ต้องการ")
    if len(text) > 600:
        return GenerateResult(None, "คำบรรยายยาวเกินไป (สูงสุด 600 ตัวอักษร)")

    chain = provider_chain(settings, provider)
    if not chain:
        return GenerateResult(None, settings.describe_missing() or NO_PROVIDER_READY_MESSAGE)

    prompt = build_prompt(text, style_key)
    problems: list[str] = []

    # สุ่มใหม่ทุกครั้งที่ผู้ใช้กด ถ้าใช้ค่าเดิมทุกครั้งจะได้ภาพเดิมเป๊ะ ๆ
    # ผู้ใช้จะเข้าใจว่าโปรแกรมค้าง ไม่ใช่ทำงานช้า
    base_seed = (settings.seed_base + secrets.randbelow(100_000)) % 2_147_483_647

    for index, current in enumerate(chain):
        call_settings = _settings_for(settings, current, model, base_seed + index)
        try:
            image = _resolve(current)(prompt, call_settings)
        except AiError as exc:
            problems.append(f"{current.value}: {exc}")
            logger.warning(
                "สร้างภาพด้วย %s ไม่สำเร็จ: %s", current.value, exc
            )
            # ตัวสุดท้ายล้ม = ทุกทางล้มหมด ไม่ต้องลองต่อ
            if index == len(chain) - 1:
                break
            continue
        except Exception as exc:  # noqa: BLE001 - กันไม่ให้เจ้าหนึ่งล้มทั้งเว็บ
            problems.append(f"{current.value}: {type(exc).__name__}: {exc}"[:200])
            logger.exception("สร้างภาพไม่สำเร็จ (%s)", current.value)
            # error ชนิดนี้แปลว่าเซิร์ฟเวอร์ยังรันโค้ดเก่าที่ยังไม่ได้แก้
            # ต้องเตือนเรื่องนี้เฉพาะตอนเจอปัญหานี้จริง ไม่ใช่ทุกครั้งที่ล้ม
            if isinstance(exc, (AttributeError, NameError, TypeError)):
                problems.append("ดูเหมือนเซิร์ฟเวอร์ยังรันโค้ดเก่าอยู่ ต้อง deploy ใหม่")
            if index == len(chain) - 1:
                break
            continue

        if image.size == 0:
            problems.append(f"{current.value}: AI คืนภาพที่ว่างเปล่า")
            continue

        return GenerateResult(image, provider=current.value)

    detail = " | ".join(problems) if problems else "ไม่ทราบสาเหตุ"
    return GenerateResult(None, f"สร้างภาพไม่สำเร็จ: {detail}")


def _settings_for(
    settings: AiSettings,
    provider: AiProvider,
    model: str | None,
    seed: int,
) -> AiSettings:
    """คัดลอกค่าตั้งเป็นชุดของเจ้านั้น

    ค่า model เดียวกันใช้ไม่ได้ทุกเจ้า ถ้าส่งชื่อโมเดลของเจ้าหนึ่งไปให้อีกเจ้า
    จะได้ 404 หรือ 400 แล้วเสียโอกาสที่เจ้าหลังจะทำได้
    """
    return AiSettings(
        provider=provider,
        api_key=settings.api_key,
        hf_token=settings.hf_token,
        base_url=settings.base_url,
        hf_base_url=settings.hf_base_url,
        hf_provider=settings.hf_provider,
        pollinations_token=settings.pollinations_token,
        pollinations_base_url=settings.pollinations_base_url,
        pollinations_model=settings.pollinations_model,
        cf_token=settings.cf_token,
        cf_account_id=settings.cf_account_id,
        cf_model=settings.cf_model,
        allow_fallback=settings.allow_fallback,
        seed_base=seed,
        timeout=settings.timeout,
        # ใส่ชื่อโมเดลที่ผู้ใช้ส่งมาเฉพาะเจ้าที่เลือก ไม่กระจายไปทุกเจ้า
        model=model or settings.model,
        hf_model=(model or settings.hf_model) if provider == AiProvider.HUGGINGFACE else settings.hf_model,
    )


def to_png_data_url(image: np.ndarray) -> str:
    """เข้ารหัสภาพเป็น data URL ส่งกลับหน้าเว็บได้ทันที"""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise AiError("เข้ารหัสภาพไม่สำเร็จ")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return "data:image/png;base64," + encoded
