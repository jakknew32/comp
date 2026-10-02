"""Hugging Face Inference API Provider สำหรับแปลงภาพเป็นเส้นระบายสี

ใช้ Hugging Face Inference API (Free Tier ~30,000 requests/เดือน)
รองรับ ControlNet models สำหรับ lineart generation:
- lllyasviel/control_v11p_sd15_lineart (SD1.5 Lineart)
- lllyasviel/control_v11p_sd15_canny (Canny edges)
- xinsir/controlnet-lineart-sd21 (SD2.1 Lineart)
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

from ..ai.settings import AiSettings


class HfError(RuntimeError):
    """เรียก Hugging Face API ไม่สำเร็จ"""


HF_MODELS = {
    "lineart_sd15": "lllyasviel/control_v11p_sd15_lineart",
    "canny_sd15": "lllyasviel/control_v11p_sd15_canny",
    "lineart_sd21": "xinsir/controlnet-lineart-sd21",
    "depth_sd15": "lllyasviel/control_v11f1p_sd15_depth",
}

DEFAULT_HF_MODEL = "lineart_sd15"
HF_API_BASE = "https://api-inference.huggingface.co/models"


def _encode_png(image: np.ndarray) -> str:
    """เข้ารหัสภาพเป็น base64 PNG"""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise HfError("เข้ารหัสภาพไม่สำเร็จ")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _decode_image(data_b64: str) -> np.ndarray:
    """ถอดรหัส base64 กลับเป็นภาพ"""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HfError("ถอดรหัสภาพจาก AI ไม่สำเร็จ") from exc

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HfError("ถอดรหัสภาพไม่สำเร็จ")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


class HfError(RuntimeError):
    """เรียก Hugging Face API ไม่สำเร็จ"""


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    """ดึงข้อความ error ที่มีประโยชน์"""
    try:
        raw = exc.read().decode("utf-8", "replace")
        payload = json.loads(raw)
        error = payload.get("error") or payload.get("error_description") or ""
        if error:
            return error[:200]
    except (json.JSONDecodeError, OSError, ValueError):
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
        return "โควต้าหมดหรือเกิน rate limit"
    return exc.reason if isinstance(exc.reason, str) else "ไม่ทราบสาเหตุ"


def hf_convert_to_lineart(
    image: np.ndarray,
    settings,
    prompt: str = "lineart, coloring book style, clean lines, white background",
    model_key: str = "lineart_sd15",
) -> np.ndarray:
    """แปลงภาพเป็น lineart ผ่าน Hugging Face Inference API

    Args:
        image: ภาพ BGR numpy array
        settings: AiSettings (ต้องมี hf_token)
        prompt: คำสั่งสำหรับ generation
        model_key: กุญแจโมเดลจาก HF_MODELS

    Returns:
        np.ndarray: ภาพ lineart BGR

    Raises:
        HfError: เมื่อเรียก API ไม่สำเร็จ
    """
    if not settings.configured:
        raise HfError("ไม่มี HF_TOKEN ในการตั้งค่า")

    token = settings.api_key
    if not token:
        raise HfError("ไม่มี HF_TOKEN ในการตั้งค่า")

    model_id = HF_MODELS.get(model_key, HF_MODELS[DEFAULT_HF_MODEL])
    url = f"{HF_API_BASE}/{model_id}"

    mime, data_b64 = _encode_png(image)

    # HF Inference API รับ payload แบบนี้
    payload = {
        "inputs": data_b64,
        "parameters": {
            "prompt": prompt,
            "negative_prompt": "color, shading, shadow, gradient, texture, noise, blur, watermark, text, logo",
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 1.0,
        },
    }

    # สำหรับ ControlNet ต้องส่งภาพ control image แยก
    # HF Inference API รองรับ controlnetผ่าน parameters
    # แต่บางโมเดลต้องการ format ต่างกัน
    # ใช้รูปแบบมาตรฐาน: inputs = base64 image, parameters มี controlnet_conditioning_scale

    payload = {
        "inputs": _encode_png(image),  # base64 image
        "parameters": {
            "prompt": prompt,
            "negative_prompt": "color, shading, shadow, gradient, texture, noise, blur, watermark, text, logo",
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 1.0,
        },
    }

    url = f"https://api-inference.huggingface.co/models/{model_id}"

    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers=headers,
            ),
            timeout=120,
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _read_hf_http_error(exc)
        raise HfError(f"HF API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HfError(f"ติดต่อ HF API ไม่สำเร็จ: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HfError("HF API ตอบช้าเกินกำหนด") from exc

    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        # บางครั้ง API คืน binary image โดยตรง
        try:
            # ลองถอดรหัสเป็นภาพ
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except Exception:
            pass
        raise HfError("ตอบกลับจาก API ไม่ใช่ JSON ที่อ่านได้")

    # รูปแบบ response ของ HF Inference API สำหรับ image generation
    # อาจเป็น list of images หรือ object ที่มี image
    if isinstance(response, list) and len(response) > 0:
        # response เป็น list of base64 strings
        img_b64 = response[0]
        if isinstance(img_b64, str):
            return _decode_base64_image(img_b64)
    elif isinstance(response, dict):
        # บางครั้งคืน object ที่มี key 'image' หรือ 'generated_image'
        for key in ["image", "generated_image", "images", "output"]:
            if key in response:
                val = response[key]
                if isinstance(val, list) and val:
                    return _decode_base64_image(val[0])
                elif isinstance(val, str):
                    return _decode_base64_image(val)

    raise HfError("ไม่พบภาพในคำตอบของ API")


def _decode_base64_image(data_b64: str) -> np.ndarray:
    """ถอดรหัส base64 กลับเป็นภาพ numpy"""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HfError("ถอดรหัส base64 ไม่สำเร็จ") from exc

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HfError("ถอดรหัสภาพไม่สำเร็จ")
    return image


def _read_hf_http_error(exc: urllib.error.HTTPError) -> str:
    """ดึงข้อความ error จาก HF API"""
    try:
        raw = exc.read().decode("utf-8", "replace")
        payload = json.loads(raw)
        error = payload.get("error") or payload.get("error_description") or ""
        if error:
            return error[:200]
    except (json.JSONDecodeError, OSError, ValueError):
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
        return "โควต้าหมดหรือเกิน rate limit (Free tier ~30k req/month)"
    return exc.reason if isinstance(exc.reason, str) else "ไม่ทราบสาเหตุ"


def hf_convert_to_lineart(
    image: np.ndarray,
    settings,
    prompt: str = "lineart, coloring book style, clean lines, white background, no shading, no color",
    model_key: str = "lineart_sd15",
) -> np.ndarray:
    """แปลงภาพเป็น lineart ผ่าน Hugging Face Inference API

    Args:
        image: ภาพ BGR numpy array
        settings: AiSettings (ต้องมี hf_token)
        prompt: คำสั่งสำหรับ generation
        model_key: กุญแจโมเดลจาก HF_MODELS

    Returns:
        np.ndarray: ภาพ lineart BGR

    Raises:
        HfError: เมื่อเรียก API ไม่สำเร็จ
    """
    if not settings.configured:
        raise HfError("ไม่มี HF_TOKEN ในการตั้งค่า")

    token = settings.api_key
    if not token:
        raise HfError("ไม่มี HF_TOKEN ในการตั้งค่า")

    model_id = HF_MODELS.get(model_key, HF_MODELS["lineart_sd15"])
    url = f"https://api-inference.huggingface.co/models/{model_id}"

    # เข้ารหัสภาพ
    mime, data_b64 = _encode_png(np.array(image))

    payload = {
        "inputs": _encode_png(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),  # HF ใช้ RGB
        "parameters": {
            "prompt": prompt,
            "negative_prompt": "color, shading, shadow, gradient, texture, noise, blur, watermark, text, logo, color",
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": 1.0,
        },
    }

    url = f"https://api-inference.huggingface.co/models/{model_id}"

    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    req = urllib.request.Request(
        f"https://api-inference.huggingface.co/models/{model_id}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                f"https://api-inference.huggingface.co/models/{model_id}",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers=headers,
            ),
            timeout=120,
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _read_hf_http_error(exc)
        raise HfError(f"HF API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HfError(f"ติดต่อ HF API ไม่สำเร็จ: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HfError("HF API ตอบช้าเกินกำหนด") from exc

    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        # บางครั้ง API คืน binary image โดยตรง
        try:
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except Exception:
            pass
        raise HfError("ตอบกลับจาก API ไม่ใช่ JSON ที่อ่านได้")

    # รูปแบบ response ของ HF Inference API สำหรับ image generation
    if isinstance(response, list) and len(response) > 0:
        img_b64 = response[0]
        if isinstance(img_b64, str):
            return _decode_base64_image(img_b64)
    elif isinstance(response, dict):
        for key in ["image", "generated_image", "images", "output"]:
            if key in response:
                val = response[key]
                if isinstance(val, list) and val:
                    return _decode_base64_image(val[0])
                elif isinstance(val, str):
                    return _decode_base64_image(val)

    raise HfError("ไม่พบภาพในคำตอบของ API")


def _decode_base64_image(data_b64: str) -> np.ndarray:
    """ถอดรหัส base64 กลับเป็นภาพ numpy"""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HfError("ถอดรหัส base64 ไม่สำเร็จ") from exc

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HfError("ถอดรหัสภาพไม่สำเร็จ")
    return image