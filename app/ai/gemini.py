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


def describe_connection_error(url: str, exc: Exception) -> str:
    """แปลง error การเชื่อมต่อให้เป็นข้อความที่ชี้สาเหตุได้

    ข้อความจากระบบอย่าง "[Errno -5] No address associated with hostname"
    ผู้ใช้ทั่วไปอ่านแล้วไม่รู้ว่าต้องแก้อะไร จึงต้องบอกทั้งชื่อเครื่องที่หาไม่เจอ
    และสิ่งที่ควรตรวจ สองสาเหตุที่พบบ่อยคือเครื่องนี้ต่ออินเทอร์เน็ตไม่ได้
    หรือตัวแปร AI_BASE_URL ตั้งชื่อโดเมนผิด
    """
    from urllib.parse import urlparse

    host = urlparse(url).netloc or "ที่ตั้งค่าไว้"
    reason = str(getattr(exc, "reason", exc))
    lowered = reason.lower()

    dns_markers = (
        "no address associated with hostname",
        "name or service not known",
        "getaddrinfo",
        "nodename nor servname provided",
        "temporary failure in name resolution",
    )
    if any(marker in lowered for marker in dns_markers):
        return (
            f"ติดต่อเซิร์ฟเวอร์ AI ไม่สำเร็จ: หาที่อยู่ของ {host} ไม่พบ "
            "— ตรวจว่าเครื่องนี้ต่ออินเทอร์เน็ตได้ "
            f"และค่า AI_BASE_URL ถูกต้อง (ขณะนี้ชี้ที่ {host})"
        )
    if "timed out" in lowered or "timeout" in lowered:
        return (
            f"เซิร์ฟเวอร์ AI ที่ {host} ตอบช้าเกินเวลาที่กำหนด "
            "ลองเพิ่ม AI_TIMEOUT_SECONDS หรือลองใหม่อีกครั้ง"
        )
    if "connection refused" in lowered:
        return (
            f"เซิร์ฟเวอร์ที่ {host} ปฏิเสธการเชื่อมต่อ "
            "— ตรวจค่า AI_BASE_URL ว่าชี้ที่บริการที่เปิดอยู่จริง"
        )
    return f"ติดต่อเซิร์ฟเวอร์ AI ไม่สำเร็จ ({host}): {reason}"


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


def _extract_image(payload: object) -> str:
    """ดึงภาพฐานสิบหกจากคำตอบของ API

    API อาจคืนหลายส่วนมา จึงเลือกส่วนที่เป็นภาพชิ้นแรก
    ตรวจชนิดข้อมูลทุกชั้น เพราะคำตอบที่ผิดรูปแบบต้องกลายเป็นข้อความอธิบาย ไม่ใช่ AttributeError
    """
    if not isinstance(payload, dict):
        detail = message_from_payload(payload)
        raise AiError("AI ตอบกลับรูปแบบที่ไม่รู้จัก" + (f": {detail[:200]}" if detail else ""))

    candidates = payload.get("candidates") or []
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                return inline["data"]

    blocked = payload.get("promptFeedback") or payload.get("prompt_feedback")
    if isinstance(blocked, dict) and blocked.get("blockReason"):
        raise AiError("AI ไม่ประมวลผลคำขอนี้: " + str(blocked.get("blockReason")))
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
        raise AiError(describe_connection_error(url, exc)) from exc
    except TimeoutError as exc:
        raise AiError("AI ตอบนานเกินกำหนดเวลา") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AiError("AI ตอบกลับเป็นข้อมูลที่อ่านไม่ได้") from exc

    return _decode_image(_extract_image(payload))


def message_from_payload(payload: object) -> str:
    """ดึงข้อความ error จากเนื้อ JSON ที่ API ตอบมา ไม่ว่าโครงสร้างจะเป็นแบบไหน

    ผู้ให้บริการแต่ละเจ้าตอบไม่เหมือนกัน เช่น
      {"error": {"message": "..."}}   (Google)
      {"error": "..."}                (Hugging Face)
      "..."                           (ข้อความล้วนที่เข้ารหัสเป็น JSON)
      [{"message": "..."}]            (รายการ)
    เดิมโค้ดสมมติว่า "error" เป็น dict เสมอ พอเจอ string จึงล้มด้วย AttributeError
    แล้วข้อความ error จริงที่ผู้ใช้ต้องการอ่านก็หายไป
    """
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        for item in payload:
            message = message_from_payload(item)
            if message:
                return message
        return ""
    if isinstance(payload, dict):
        for key in ("error", "message", "error_description", "detail"):
            value = payload.get(key)
            if value in (None, ""):
                continue
            message = message_from_payload(value)
            if message:
                return message
    return ""


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    """ดึงข้อความที่มีประโยชน์จาก error ของ API

    ไม่แสดงรายละเอียดทั้งหมด เพราะอาจมีข้อมูลของคีย์ปนอยู่
    """
    try:
        raw = exc.read().decode("utf-8", "replace")
        message = message_from_payload(json.loads(raw))
        if message:
            return message[:200]
    except (json.JSONDecodeError, OSError, ValueError, AttributeError, TypeError):
        pass

    if exc.code in (401, 403):
        return "คีย์ไม่ถูกต้องหรือไม่มีสิทธิ์ใช้งาน"
    if exc.code == 429:
        return "โควตาหมดหรือใช้งานเกินลิมิต"
    if exc.code == 404:
        return "ไม่พบโมเดลที่ระบุ"
    return exc.reason if isinstance(exc.reason, str) else "ไม่ทราบสาเหตุ"
