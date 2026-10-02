"""โมเดลภาษา Gemini ของ Google ใช้แปลงภาพถ่ายเป็นภาพลายเส้น

ใช้ urllib ของไลบรารีมาตรฐาน ไม่ต้องเพิ่มแพ็กเกจใน image
เพราะต้องการ HTTP POST ไปที่ URL เดียวเท่านั้น

หมายเหตุสำคัญ: โค้ดส่วนนี้ยังไม่ได้ทดสอบกับ API จริง
เพราะไม่มีคีย์ในสภาพแวดล้อมการพัฒนา จึงออกแบบให้ล้มเหลวอย่างปลอดภัย
แล้วค่อยใช้การประมวลผลเดิมแทน ไม่ทำให้ทั้งเล่มล้ม
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

from .settings import AiSettings


class AiError(RuntimeError):
    """เรียก AI ไม่สำเร็จ"""


def _encode_png(image: np.ndarray) -> tuple[str, str]:
    """เข้ารหัสภาพเป็น base64 พร้อมชนิดไฟล์ ส่งเข้า API

    บีบอัดด้วย PNG ไม่ใช่ JPEG เพราะภาพลายเส้นมีเส้นบาง
    การบีบอัดแบบมีความสูญเสียจะทำให้เส้นแตกก่อนถึงปลายทาง
    """
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise AiError("เข้ารหัสภาพเพื่อส่งให้ AI ไม่สำเร็จ")
    return "image/png", base64.b64encode(buffer.tobytes()).decode("ascii")


def _decode_image(data_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise AiError("AI ส่งกลับภาพที่อ่านไม่ได้") from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AiError("AI ส่งกลับภาพที่ถอดรหัสไม่ได้")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _extract_image(payload: dict) -> str:
    """ดึงภาพฐานสิบหกจากคำตอบของ API

    API อาจคืนหลายส่วนมา จึงเลือกส่วนที่เป็นภาพชิ้นแรก
    """
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        parts = ((candidate.get("content") or {}).get("parts")) or []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return inline["data"]

    blocked = payload.get("promptFeedback") or payload.get("prompt_feedback")
    if blocked and blocked.get("blockReason"):
        raise AiError(
            "AI ไม่ประมวลผลคำขอนี้: " + str(blocked.get("blockReason"))
        )
    raise AiError("AI ไม่ได้ส่งภาพกลับมา")


def convert_to_lineart(
    image: np.ndarray,
    settings: AiSettings,
    prompt: str,
) -> np.ndarray:
    """ส่งภาพไปให้ AI วาดใหม่เป็นภาพลายเส้น แล้วคืนภาพผลลัพธ์

    ใช้เฉพาะกับภาพถ่ายจริง ส่วนภาพลายเส้นอยู่แล้วไม่ต้องเสียเงิน
    """
    if not settings.configured:
        raise AiError(settings.describe_missing())

    mime, data = _encode_png(image)

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": data}},
                    {"text": prompt},
                ],
            }
        ]
    }

    url = f"{settings.base_url}/models/{settings.model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.api_key or "",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _read_http_error(exc)
        raise AiError(f"AI ตอบกลับข้อผิดพลาด {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AiError("ติดต่อเซิร์ฟเวอร์ AI ไม่สำเร็จ: " + str(exc.reason)) from exc
    except TimeoutError as exc:
        raise AiError("AI ตอบนานเกินกำหนดเวลา") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AiError("AI ตอบกลับเป็นข้อมูลที่อ่านไม่ได้") from exc

    return _decode_image(_extract_image(payload))


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    """ดึงข้อความที่มีประโยชน์จาก error ของ API

    ไม่แสดงรายละเอียดทั้งหมด เพราะอาจมีข้อมูลของคีย์ปนอยู่
    """
    try:
        raw = exc.read().decode("utf-8", "replace")
        payload = json.loads(raw)
        message = ((payload.get("error") or {}).get("message")) or ""
        if message:
            return message[:200]
    except (json.JSONDecodeError, OSError, ValueError):
        pass

    if exc.code in (401, 403):
        return "คีย์ไม่ถูกต้องหรือไม่มีสิทธิ์ใช้งาน"
    if exc.code == 429:
        return "โควตาหมดหรือใช้งานเกินลิมิต"
    if exc.code == 404:
        return "ไม่พบโมเดลที่ระบุ"
    return exc.reason if isinstance(exc.reason, str) else "ไม่ทราบสาเหตุ"
