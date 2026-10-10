"""ผู้ให้บริการสร้างภาพฟรีสองเจ้า ทำงานเป็นตัวสำรองเมื่อเจ้าหลักใช้ไม่ได้

เหตุผลที่ต้องมี:
เครดิตของผู้ให้บริการแต่ละเจ้ามีจำกัด ถ้าพึ่งเจ้าเดียว
เว็บจะล้มทั้งที่ยังมีทางออกอีกหลายทางที่ไม่เสียเงิน

สองเจ้านี้เลือกเพราะไม่ต้องผูกกับเครดิตรายเดือนของผู้ใช้รายคน
- Pollinations  ไม่ต้องมีคีย์ มีโควตาต่อวินาทีแทน (สมัครฟรีได้)
- Cloudflare Workers AI  10,000 neurons/วัน ต่อหนึ่งบัญชี (ฟรี ไม่มีการ์ด)

ใช้ urllib ของไลบรารีมาตรฐานเหมือนโมดูลอื่น ไม่เพิ่มแพ็กเกจใหม่
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
import time

import cv2
import numpy as np

from .gemini import AiError, describe_connection_error, message_from_payload
from .settings import AiSettings


class _RateLimited(Exception):
    """โดนจำกัดอัตราแล้วรอแล้วลองใหม่ได้"""


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    """อ่านข้อความจาก body ของ error แล้วย่อให้สั้น

    บริการฟรีมักไม่ได้ทำตามรูปแบบ JSON เดียวกัน จึงต้องรองรับทั้ง JSON และข้อความล้วน
    """
    try:
        raw = exc.read().decode("utf-8", "replace")
    except OSError:
        return ""

    text = raw.strip()
    if not text:
        return ""
    try:
        message = message_from_payload(json.loads(text))
        if message:
            return message[:200]
    except json.JSONDecodeError:
        pass
    return text[:200]


def _decode_image_bytes(raw: bytes) -> np.ndarray:
    """ถอดไบต์ที่ API ส่งมาเป็นภาพ BGR"""
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AiError("AI ส่งกลับภาพที่ถอดรหัสไม่ได้")
    return image


# เมื่อไม่มีโทเคน Pollinations ให้เรียกได้หนึ่งครั้งทุก 15 วินาที
# ถ้าเจอ 429 แปลว่าโดนจำกัด ต้องรอแล้วลองใหม่ ไม่ใช่บอกผู้ใช้ให้กดเอง
POLLINATIONS_ANON_COOLDOWN = 16.0


# --- Pollinations -------------------------------------------------------------

# โมเดลของ Pollinations ที่ใช้ได้จริง ณ เวลานี้
# flux  คุณภาพดีที่สุด แต่ช้า
# turbo เร็วกว่ามาก เหมาะกับเว็บที่คนกดรัว
POLLINATIONS_MODELS = {
    "flux": "flux",
    "turbo": "turbo",
    "kontext": "kontext",
}
DEFAULT_POLLINATIONS_MODEL = "turbo"

# ขนาดภาพที่ขอ ต้องเป็นเลขหารลงตัว ของ Pollinations บังคับให้เป็น 8 คูณ
POLLINATIONS_WIDTH = 1024
POLLINATIONS_HEIGHT = 1024


def generate_with_pollinations(
    prompt: str,
    settings: AiSettings,
) -> np.ndarray:
    """สร้างภาพจากข้อความผ่าน Pollinations

    ไม่ต้องมีคีย์ก็เรียกได้ ถ้ามีโทเคนจะได้โควตาที่ดีกว่าและเอาลายน้ำออกได้
    ถ้าโดนจำกัดอัตราจะรอแล้วลองใหม่เองหนึ่งครั้ง เพราะเป็นเรื่องปกติมาก
    """
    try:
        return _request_pollinations(prompt, settings)
    except _RateLimited:
        time.sleep(POLLINATIONS_ANON_COOLDOWN)
        try:
            return _request_pollinations(prompt, settings)
        except _RateLimited as exc:
            # รอแล้วยังโดนจำกัดอีก แปลว่าจะต้องรอนานกว่านี้
            # ต้องกลายเป็น AiError ไม่ใช่หลุดออกไป ไม่งั้นชั้นสำรองจะจับไม่ได้
            raise AiError(
                "Pollinations ยังจำกัดอัตราอยู่หลังรอแล้ว "
                "(ไม่มีโทเคนได้ 1 คำขอต่อ 15 วินาที) — "
                "สมัครฟรีที่ auth.pollinations.ai เพื่อได้โควตา 1 คำขอต่อ 5 วินาที"
            ) from exc


def _request_pollinations(
    prompt: str,
    settings: AiSettings,
) -> np.ndarray:
    """ยิงคำขอเดียว คืนภาพ หรือโยน error"""
    base = (settings.pollinations_base_url or "").rstrip("/")
    model = POLLINATIONS_MODELS.get(settings.pollinations_model, settings.pollinations_model)
    query = urllib.parse.urlencode(
        {
            "model": model,
            "width": POLLINATIONS_WIDTH,
            "height": POLLINATIONS_HEIGHT,
            "nologo": "true",
            "seed": settings.seed_base,
        }
    )
    url = f"{base}/prompt/{urllib.parse.quote(prompt)}?{query}"

    headers = {"Accept": "image/*"}
    if settings.pollinations_token:
        headers["Authorization"] = f"Bearer {settings.pollinations_token}"

    request = urllib.request.Request(url, method="GET", headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _read_error_body(exc)
        if exc.code in (429, 402):
            # Pollinations ใช้ 402 ตอนโควตารายวินาทีหมด ไม่ใช่เรื่องเครดิตเงินจริง
            # ตัวที่ไม่มีโทเคนจะโดนบ่อยกว่า และการรอแล้วลองใหม่ก็มีโอกาสสำเร็จ
            if not settings.pollinations_token:
                raise _RateLimited() from exc
            raise AiError(
                "Pollinations เรียกถี่เกินลิมิต "
                "(โทเคนนี้ได้ 1 คำขอต่อ 5 วินาที) — กรุณารอสักครู่แล้วลองใหม่"
            ) from exc
        if exc.code in (401, 403):
            raise AiError("POLLINATIONS_TOKEN ไม่ถูกต้องหรือหมดอายุ") from exc
        raise AiError(
            f"Pollinations ตอบกลับข้อผิดพลาด {exc.code}: {detail or 'ไม่ทราบสาเหตุ'}"
        ) from exc
    except urllib.error.URLError as exc:
        raise AiError(describe_connection_error(url, exc)) from exc
    except TimeoutError as exc:
        raise AiError("Pollinations ตอบนานเกินกำหนดเวลา") from exc

    if not raw:
        raise AiError("Pollinations ไม่ได้ส่งภาพกลับมา")

    image = _decode_image_bytes(raw)
    # ถ้าตอบมาเป็น HTML หรือ JSON แปลว่าเจ้ากำลังล่ม ไม่ใช่รูปถาด
    if image.size == 0:
        raise AiError("Pollinations ส่งกลับภาพที่ว่างเปล่า")
    return image


# --- Cloudflare Workers AI ----------------------------------------------------

DEFAULT_CF_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
CF_API_BASE = "https://api.cloudflare.com/client/v4"

# ความยาวคำสั่งสูงสุดที่ Cloudflare รับได้ ต้องย่อให้สั้นก่อนส่ง
# คำสั่งที่ยาวเกินนี้จะโดนปฏิเสธทั้งที่ผู้ใช้พิมพ์มาไม่เกิน 600 ตัว
CF_MAX_PROMPT = 2048

# FLUX.1-schnell เป็นโมเดลก้าวเร็ว ค่าเริ่มต้นคือ 4 และไม่เกิน 8
# ยิ่งก้าวน้อยยิ่งกินโควตาฟรีน้อย จึงใช้ค่าต่ำสุดที่ยังได้ภาพใช้ได้
CF_STEPS = 4


def generate_with_cloudflare(
    prompt: str,
    settings: AiSettings,
) -> np.ndarray:
    """สร้างภาพจากข้อความผ่าน Cloudflare Workers AI

    ต้องมี CF_API_TOKEN และ CF_ACCOUNT_ID ของบัญชี Cloudflare ที่สมัครฟรี
    โควตาฟรี 10,000 neurons ต่อวัน รีเซ็ตตีน 00:00 UTC

    ราคาประมาณ 9.6 neurons ต่อก้าว + 4.8 ต่อ tile 512x512
    ภาพ 1024x1024 ที่ 4 ก้าว กินราว 75 neurons ต่อภาพ
    คิดเป็นเกือบ 130 ภาพต่อวันที่โควตาฟรี ซึ่งมากกว่าเว็บนี้มาก
    """
    if not settings.cf_token or not settings.cf_account_id:
        raise AiError(
            "ยังไม่ได้ตั้งค่า Cloudflare: ต้องมี CF_API_TOKEN และ CF_ACCOUNT_ID "
            "(สร้างฟรีที่ dash.cloudflare.com ใน Workers AI)"
        )

    model = settings.cf_model or DEFAULT_CF_IMAGE_MODEL
    url = f"{CF_API_BASE}/accounts/{settings.cf_account_id}/ai/run/{model}"

    # ส่งได้เฉพาะ prompt กับ steps เท่านั้น
    # schema ของ Cloudflare ตั้ง additionalProperties เป็น false
    # การส่ง seed หรือ field อื่นจะโดนปฏิเสธด้วย 400 ทันที
    # ตัวอย่างในเอกสารที่มี seed เป็นตัวอย่างสำหรับ Workers binding ไม่ใช่ REST API
    body = {
        "prompt": prompt[:CF_MAX_PROMPT],
        "steps": CF_STEPS,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.cf_token}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _read_error_body(exc)
        if exc.code in (401, 403):
            raise AiError(
                "CF_API_TOKEN ไม่ถูกต้องหรือไม่มีสิทธิ์เรียก Workers AI"
            ) from exc
        if exc.code == 429:
            raise AiError(
                "โควตา Cloudflare หมด (ฟรี 10,000 neurons ต่อวัน รีเซ็ต 00:00 UTC)"
            ) from exc
        if exc.code == 404:
            raise AiError(
                f"CF_ACCOUNT_ID หรือชื่อโมเดลผิด ({model}) — ตรวจว่าเปิดใช้ Workers AI แล้ว"
            ) from exc
        raise AiError(
            f"Cloudflare ตอบกลับข้อผิดพลาด {exc.code}: {detail or 'ไม่ทราบสาเหตุ'}"
        ) from exc
    except urllib.error.URLError as exc:
        raise AiError(describe_connection_error(url, exc)) from exc
    except TimeoutError as exc:
        raise AiError("Cloudflare ตอบนานเกินกำหนดเวลา") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AiError("Cloudflare ตอบกลับเป็นข้อมูลที่อ่านไม่ได้") from exc

    result = payload.get("result") if isinstance(payload, dict) else None
    data_b64 = result.get("image") if isinstance(result, dict) else None
    if not data_b64:
        errors = payload.get("errors") if isinstance(payload, dict) else None
        detail = message_from_payload(errors) or "ไม่ทราบสาเหตุ"
        raise AiError(f"Cloudflare ไม่ได้ส่งภาพกลับมา: {detail[:200]}")

    try:
        image_bytes = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise AiError("Cloudflare ส่งภาพที่ถอดรหัสไม่ได้") from exc

    return _decode_image_bytes(image_bytes)