"""Hugging Face Inference API สำหรับแปลงภาพถ่ายเป็นภาพลายเส้น

ใช้โมเดล ControlNet สำหรับดึงเส้น:
- lllyasviel/control_v11p_sd15_lineart (SD1.5 Lineart)
- lllyasviel/control_v11p_sd15_canny (Canny edges)
- xinsir/controlnet-lineart-sd21 (SD2.1 Lineart)

หมายเหตุ: HF Inference API สำหรับ ControlNet รับภาพ control ผ่าน inputs
และคืนกลับมาเป็นไบต์ภาพโดยตรง ไม่ใช่ JSON เสมอไป จึงต้องลองถอดทั้งสองแบบ
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

from .gemini import describe_connection_error, message_from_payload
from .settings import AiSettings

# ปลายทางมาจาก settings.hf_base_url เสมอ เพื่อให้เปลี่ยนได้ด้วย AI_HF_BASE_URL
# ค่าเริ่มต้นคือ router.huggingface.co/hf-inference/models
# เพราะ api-inference.huggingface.co ถูกปิดแล้วและไม่มี DNS อีกต่อไป
# และ router ต้องมี path /hf-inference/models ต่อท้าย ไม่งั้นได้ 404

# --- โมเดลสองกลุ่ม ห้ามใช้สลับกัน ---
#
# ControlNet รับ "ภาพนำ" ไม่ใช่ข้อความ จึงใช้ได้เฉพาะการแปลงภาพถ่ายเป็นลายเส้น
# ใช้สร้างภาพจากข้อความไม่ได้ เพราะไม่มี text encoder สำหรับสร้างภาพใหม่
# โมเดลในกลุ่มนี้จึงไม่มี tag text-to-image และ router จะตอบ 404 เมือเรียกผ่าน
# เดิมโค้ดใช้โมเดล ControlNet สำหรับสร้างภาพด้วย จึงล้มเหลวเสมอ
HF_CONTROLNET_MODELS = {
    "lineart_sd15": "lllyasviel/control_v11p_sd15_lineart",
    "canny_sd15": "lllyasviel/control_v11p_sd15_canny",
    "lineart_sd21": "xinsir/controlnet-lineart-sd21",
    "depth_sd15": "lllyasviel/control_v11f1p_sd15_depth",
}

# text-to-image จริง ใช้สร้างภาพจากข้อความได้
HF_TEXT2IMAGE_MODELS = {
    "flux_schnell": "black-forest-labs/FLUX.1-schnell",
    "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd_turbo": "stabilityai/sd-turbo",
    "sdxl_turbo": "stabilityai/sdxl-turbo",
}

# ชื่อย่อเดิมที่หน้าเว็บอาจยังส่งมา ต้องแปลงเป็นโมเดล text-to-image ให้ถูกชนิด
HF_MODEL_ALIASES = {
    "lineart_sd15": "flux_schnell",
    "canny_sd15": "flux_schnell",
    "lineart_sd21": "flux_schnell",
    "depth_sd15": "flux_schnell",
    "default": "flux_schnell",
}

DEFAULT_HF_MODEL = "lineart_sd15"

# ชื่อย่อสำหรับสร้างภาพจากข้อความ
DEFAULT_HF_TEXT2IMAGE = "flux_schnell"

DEFAULT_NEGATIVE_PROMPT = (
    "color, shading, shadow, gradient, texture, noise, blur, "
    "watermark, text, logo"
)


class HfError(RuntimeError):
    """เรียก Hugging Face API ไม่สำเร็จ"""


def _encode_png(image: np.ndarray) -> str:
    """เข้ารหัสภาพเป็น base64 PNG"""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise HfError("เข้ารหัสภาพไม่สำเร็จ")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _decode_base64_image(data_b64: str) -> np.ndarray:
    """ถอดรหัส base64 กลับเป็นภาพ BGR"""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HfError("ถอดรหัสภาพจาก AI ไม่สำเร็จ") from exc
    return decode_image_bytes(raw)


def decode_image_bytes(raw: bytes) -> np.ndarray:
    """ถอดไบต์ที่ API ตอบกลับมาเป็นภาพ BGR"""
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HfError("ถอดรหัสภาพไม่สำเร็จ")
    return image


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    """แปลง error ของ API เป็นข้อความที่ผู้ใช้อ่านเข้าใจ"""
    try:
        payload = json.loads(exc.read().decode("utf-8", "replace"))
        message = message_from_payload(payload)
        if message:
            return message[:200]
    except (json.JSONDecodeError, OSError, ValueError, AttributeError, TypeError):
        pass

    if exc.code == 401:
        return "Hugging Face token ไม่ถูกต้องหรือหมดอายุ"
    if exc.code == 403:
        return "Token ไม่มีสิทธิ์เข้าถึงโมเดลนี้"
    if exc.code == 404:
        return "ไม่พบโมเดลที่ระบุ"
    if exc.code == 503:
        return "โมเดลกำลังโหลด กรุณาลองใหม่ในไม่กี่นาที"
    if exc.code == 429:
        return "โควตาหมดหรือเกิน rate limit"
    return exc.reason if isinstance(exc.reason, str) else "ไม่ทราบสาเหตุ"


def _parse_response(raw: bytes) -> np.ndarray:
    """คำตอบของ API อาจเป็นไบต์ภาพตรง ๆ หรือ JSON ที่มี base64 ฝังอยู่"""
    try:
        return decode_image_bytes(raw)
    except HfError:
        pass

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HfError("ตอบกลับจาก API ไม่ใช่ภาพหรือ JSON ที่อ่านได้") from exc

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, str):
            return _decode_base64_image(first)
        if isinstance(first, dict) and isinstance(first.get("image"), str):
            return _decode_base64_image(first["image"])
    elif isinstance(payload, dict):
        for key in ("image", "generated_image", "images", "output"):
            value = payload.get(key)
            if isinstance(value, str):
                return _decode_base64_image(value)
            if isinstance(value, list) and value and isinstance(value[0], str):
                return _decode_base64_image(value[0])

    raise HfError("ไม่พบภาพในคำตอบของ API")


def hf_convert_to_lineart(
    image: np.ndarray,
    settings: AiSettings,
    prompt: str = "lineart, coloring book style, clean lines, white background",
    model_key: str = DEFAULT_HF_MODEL,
) -> np.ndarray:
    """แปลงภาพถ่ายเป็นภาพลายเส้นผ่าน Hugging Face Inference API"""
    token = settings.hf_token or settings.api_key
    if not token:
        raise HfError(settings.describe_missing())

    model_id = HF_CONTROLNET_MODELS.get(
        model_key, HF_CONTROLNET_MODELS[DEFAULT_HF_MODEL]
    )
    # endpoint เดิมถูกปิดไปแล้ว ใช้ค่าจาก settings เพื่อให้เปลี่ยนปลายทางได้
    # โดยไม่ต้องแก้โค้ด และไม่ให้ไปชนกับค่าของผู้ให้บริการอื่น
    url = f"{settings.hf_base_url}/{model_id}"
    payload = {
        "inputs": _encode_png(image),
        "parameters": {
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 1.0,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise HfError(f"HF API ตอบกลับข้อผิดพลาด {exc.code}: {_read_http_error(exc)}") from exc
    except urllib.error.URLError as exc:
        raise HfError(describe_connection_error(url, exc)) from exc
    except TimeoutError as exc:
        raise HfError("HF API ตอบช้าเกินกำหนด") from exc

    return _parse_response(raw)
