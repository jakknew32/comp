"""สร้างภาพระบายสีจากข้อความด้วย AI

ต่างจากโมดูลแปลงภาพ (enhance) ตรงที่ไม่ต้องมีภาพต้นฉบับ
ผู้ใช้พิมพ์คำบรรยาย เช่น "แมวการ์ตูนใส่หมวก" แล้ว AI วาดภาพระบายสีใหม่ให้ทั้งหมด

ข้อจำกัดเดียวกับการแปลงภาพ: ต้องมีคีย์ผู้ให้บริการเสมอ
ถ้าไม่มีคีย์ ฟีเจอร์นี้จะไม่แสดงบนหน้าเว็บ
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

from .gemini import AiError, _decode_image, _extract_image, _read_http_error
from .settings import AiProvider, AiSettings

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
        raise AiError("ติดต่อเซิร์ฟเวอร์ AI ไม่สำเร็จ: " + str(exc.reason)) from exc
    except TimeoutError as exc:
        raise AiError("AI ตอบนานเกินกำหนดเวลา") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AiError("AI ตอบกลับเป็นข้อมูลที่อ่านไม่ได้") from exc

    return _decode_image(_extract_image(payload))


def generate_with_huggingface(
    prompt: str,
    settings: AiSettings,
) -> np.ndarray:
    """สร้างภาพจากข้อความด้วย Hugging Face Inference API

    โมเดล text-to-image รับข้อความใน inputs แล้วคืนไบต์ภาพโดยตรง
    ไม่ใช่ JSON จึงต้องลองถอดเป็นภาพทันที
    """
    if not settings.hf_token:
        raise AiError(settings.describe_missing())

    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "color, shading, shadow, gradient, noise, blur, watermark, text, photo",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
        },
    }
    url = f"https://api-inference.huggingface.co/models/{settings.hf_model}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.hf_token}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise AiError(f"AI ตอบกลับข้อผิดพลาด {exc.code}: {_read_http_error(exc)}") from exc
    except urllib.error.URLError as exc:
        raise AiError("ติดต่อเซิร์ฟเวอร์ AI ไม่สำเร็จ: " + str(exc.reason)) from exc
    except TimeoutError as exc:
        raise AiError("AI ตอบนานเกินกำหนดเวลา") from exc

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AiError("AI ไม่ได้ส่งภาพกลับมา")
    return image


class GenerateResult:
    """ผลการสร้างภาพ แยกสถานะสำเร็จกับล้มเหลวให้หน้าเว็บแสดงได้ชัด"""

    def __init__(self, image: np.ndarray | None, note: str | None = None) -> None:
        self.image = image
        self.note = note

    @property
    def ok(self) -> bool:
        return self.image is not None


def generate_image(
    text: str,
    style_key: str | None,
    settings: AiSettings,
    provider: str | None = None,
    model: str | None = None,
) -> GenerateResult:
    """สร้างภาพหนึ่งภาพจากคำบรรยาย คืนผลลัพธ์พร้อมข้อความเตือนเมื่อล้มเหลว

    ไม่โยน exception ออกมาเอง เพราะผู้ใช้ต้องเห็นว่าล้มเหลวเพราะอะไร
    แล้วกดลองใหม่ได้ทันที
    """
    text = (text or "").strip()
    if not text:
        return GenerateResult(None, "กรุณาพิมพ์คำบรรยายภาพที่ต้องการ")
    if len(text) > 600:
        return GenerateResult(None, "คำบรรยายยาวเกินไป (สูงสุด 600 ตัวอักษร)")

    chosen = (provider or settings.provider.value).strip().lower()
    call_settings = AiSettings(
        provider=settings.provider,
        api_key=settings.api_key,
        hf_token=settings.hf_token,
        base_url=settings.base_url,
        timeout=settings.timeout,
        model=model or settings.model,
        hf_model=model or settings.hf_model,
    )

    if not settings.configured:
        return GenerateResult(None, settings.describe_missing())

    prompt = build_prompt(text, style_key)
    try:
        if chosen == AiProvider.HUGGINGFACE.value:
            image = generate_with_huggingface(prompt, call_settings)
        else:
            image = generate_with_gemini(prompt, call_settings)
    except AiError as exc:
        return GenerateResult(None, f"สร้างภาพไม่สำเร็จ: {exc}")
    except Exception as exc:  # noqa: BLE001 - กันไม่ให้คำขอเดียวทำเซิร์ฟเวอร์ล้ม
        return GenerateResult(None, f"สร้างภาพไม่สำเร็จ: {type(exc).__name__}")

    if image.size == 0:
        return GenerateResult(None, "AI คืนภาพที่ว่างเปล่า")
    return GenerateResult(image)


def to_png_data_url(image: np.ndarray) -> str:
    """เข้ารหัสภาพเป็น data URL ส่งกลับหน้าเว็บได้ทันที"""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise AiError("เข้ารหัสภาพไม่สำเร็จ")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return "data:image/png;base64," + encoded
