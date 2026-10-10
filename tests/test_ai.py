"""ทดสอบส่วนเชื่อมต่อ AI

จุดสำคัญที่สุดคือ AI ต้องไม่ทำให้ทั้งระบบล้ม
เพราะผู้ใช้อาจไม่ได้ตั้งค่า คีย์อาจหมดอายุ หรือเครือข่ายมีปัญหา
ทุกกรณีต้องกลับไปใช้การประมวลผลเดิมต่อได้
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest

from app import ai
from app.ai import gemini
from app.ai.settings import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_TIMEOUT,
    load_settings,
)
from tests import fixtures


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """กันไม่ให้เทสต์ในไฟล์นี้ติดต่อเครือข่ายจริง

    เคยเกิดกรณีที่เทสต์ยิงไปที่ API จริงด้วยคีย์ปลอม เพราะ monkeypatch
    ทำงานผิดจุด ทำให้เทสต์ช้า ผลไม่น่าเชื่อถือ และเสี่ยงเสียค่าใช้จ่าย

    ปิดที่ urlopen ซึ่งเป็นจุดสัมผัสจริงกับเครือข่าย แทนที่จะปิดฟังก์ชันชั้นบน
    เทสต์ที่ต้องการตรวจรูปแบบคำขอจึงยังเรียกโค้ดจริงได้
    โดยแทนที่ urlopen ของตัวเองทับ
    """
    def forbidden(*_args, **_kwargs):
        raise AssertionError("เทสต์พยายามติดต่อเครือข่าย ต้องแทนที่ urlopen")

    monkeypatch.setattr(gemini.urllib.request, "urlopen", forbidden)


# --- การอ่านค่าตั้งค่า --------------------------------------------------------


def test_no_key_means_not_configured() -> None:
    """ไม่มีคีย์เจ้าหลัก แต่ยังใช้เจ้าฟรีได้ จึงยังถือว่าพร้อมใช้งาน

    ต้องเช็คกับค่าที่ปิด fallback ด้วย ถึงจะเห็นว่าไม่มีเจ้าไหนเลย
    """
    settings = load_settings({})
    assert settings.configured is True, "ต้องมีทางออกฟรีเสมอ ไม่งั้นเว็บจะใช้ไม่ได้"

    offline = load_settings({"AI_ALLOW_FALLBACK": "false"})
    assert offline.configured is False
    assert offline.model == DEFAULT_MODEL
    assert offline.base_url == DEFAULT_BASE_URL
    assert ENV_API_KEY in offline.describe_missing()


def test_key_enables_ai() -> None:
    settings = load_settings({ENV_API_KEY: "  secret-key  "})
    assert settings.configured is True
    assert settings.api_key == "secret-key"


def test_blank_key_is_treated_as_absent() -> None:
    blank = load_settings({ENV_API_KEY: "   ", "AI_ALLOW_FALLBACK": "false"})
    assert blank.api_key is None
    assert blank.configured is False


def test_model_and_url_are_overridable() -> None:
    settings = load_settings(
        {ENV_API_KEY: "k", ENV_MODEL: "custom-model", ENV_BASE_URL: "https://x.test/v1/"}
    )
    assert settings.model == "custom-model"
    assert settings.base_url == "https://x.test/v1", "ตัดเครื่องหมาย / ท้ายออก"


def test_bad_timeout_falls_back_to_default() -> None:
    assert load_settings({ENV_TIMEOUT: "not-a-number"}).timeout > 0
    # ค่าเวลารอต้องอยู่ในกรอบที่ใช้ได้จริง
    assert load_settings({ENV_TIMEOUT: "1"}).timeout >= 5.0
    assert load_settings({ENV_TIMEOUT: "99999"}).timeout <= 300.0


def test_masked_key_never_leaks_secret() -> None:
    settings = load_settings({ENV_API_KEY: "super-secret-value-1234"})
    masked = settings.masked_key()
    assert "super" not in masked
    assert masked.endswith("1234")


def test_status_hides_cost_when_unconfigured() -> None:
    """ไม่มีเจ้าไหนพร้อมเลยจริง ๆ ต้องไม่โชว์ราคาและต้องบอกว่าตั้งค่าอะไร"""
    data = ai.status(load_settings({"AI_ALLOW_FALLBACK": "false"}))
    assert data["configured"] is False
    assert data["model"] is None
    assert data["estimated_cost_per_image_usd"] is None
    assert data["message"]


def test_status_reports_free_provider_when_no_key() -> None:
    """ไม่มีคีย์เลยต้องโชว์ว่าใช้เจ้าฟรี ราคาเป็นศูนย์ ไม่ใช่โชว์ราคา Gemini"""
    data = ai.status(load_settings({}))
    assert data["configured"] is True
    assert data["provider"] == "pollinations"
    assert data["estimated_cost_per_image_usd"] == 0.0
    assert "pollinations" in data["fallback_chain"]


def test_status_reports_cost_when_configured() -> None:
    data = ai.status(load_settings({ENV_API_KEY: "k"}))
    assert data["configured"] is True
    assert data["model"] == DEFAULT_MODEL
    assert data["estimated_cost_per_image_usd"] > 0


# --- การตัดสินใจว่าจะเรียก AI หรือไม่ ---------------------------------------


def test_lineart_is_not_sent_to_ai() -> None:
    """ภาพลายเส้นผ่านการประมวลผลเดิมได้ดีอยู่แล้ว การยิง AI เป็นการเสียเงินเปล่า"""
    settings = load_settings({ENV_API_KEY: "k"})
    art = fixtures.synthetic_lineart()
    assert ai.should_use_ai(art, settings, enabled=True) is False


def test_photo_is_sent_to_ai() -> None:
    settings = load_settings({ENV_API_KEY: "k"})
    assert ai.should_use_ai(fixtures.synthetic_photo(), settings, enabled=True) is True


def test_no_ai_when_disabled_or_unconfigured() -> None:
    photo = fixtures.synthetic_photo()
    assert ai.should_use_ai(photo, load_settings({ENV_API_KEY: "k"}), False) is False
    # ไม่มีคีย์ของเจ้าที่รับภาพแนบ = ทำงานนี้ไม่ได้ ต้องไม่ยิง
    # ต่อให้มี Pollinations ซึ่งรับได้แค่ข้อความก็ตาม
    assert ai.should_use_ai(photo, load_settings({}), True) is False


# --- การไม่ทำให้ระบบพัง -----------------------------------------------------


def test_disabled_returns_original_image_unchanged() -> None:
    photo = fixtures.synthetic_photo()
    outcome = ai.enhance(photo, load_settings({ENV_API_KEY: "k"}), enabled=False)
    assert outcome.used is False
    assert outcome.note is None
    assert np.array_equal(outcome.image, photo)


def test_missing_key_falls_back_with_note() -> None:
    photo = fixtures.synthetic_photo()
    outcome = ai.enhance(photo, load_settings({}), enabled=True)
    assert outcome.used is False
    assert outcome.failed is True
    assert outcome.note and ENV_API_KEY in outcome.note
    assert np.array_equal(outcome.image, photo), "ต้องคืนภาพเดิมเมื่อใช้ AI ไม่ได้"


def test_lineart_skipped_even_when_ai_enabled() -> None:
    art = fixtures.synthetic_lineart()
    outcome = ai.enhance(art, load_settings({ENV_API_KEY: "k"}), enabled=True)
    assert outcome.used is False
    assert outcome.note is None
    assert np.array_equal(outcome.image, art)


def test_api_error_falls_back(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise gemini.AiError("คีย์ไม่ถูกต้อง")

    monkeypatch.setattr(gemini, "convert_to_lineart", boom)
    photo = fixtures.synthetic_photo()
    outcome = ai.enhance(photo, load_settings({ENV_API_KEY: "k"}), enabled=True)
    assert outcome.used is False
    assert "คีย์ไม่ถูกต้อง" in outcome.note
    assert np.array_equal(outcome.image, photo)


def test_unexpected_exception_also_falls_back(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("บั๊กที่ไม่คาดคิด")

    monkeypatch.setattr(gemini, "convert_to_lineart", boom)
    photo = fixtures.synthetic_photo()
    outcome = ai.enhance(photo, load_settings({ENV_API_KEY: "k"}), enabled=True)
    assert outcome.used is False
    assert outcome.note
    assert np.array_equal(outcome.image, photo)


def test_empty_ai_result_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(
        gemini, "convert_to_lineart", lambda *_a, **_k: np.zeros((0, 0, 3), np.uint8)
    )
    photo = fixtures.synthetic_photo()
    outcome = ai.enhance(photo, load_settings({ENV_API_KEY: "k"}), enabled=True)
    assert outcome.used is False
    assert np.array_equal(outcome.image, photo)


# --- การแปลงคำตอบของ API ----------------------------------------------------


def _api_response_with_image(image: np.ndarray) -> dict:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "นี่คือคำอธิบาย"},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64.b64encode(buffer.tobytes()).decode(),
                            }
                        },
                    ]
                }
            }
        ]
    }


def test_extract_image_skips_text_parts() -> None:
    """API อาจคืนทั้งข้อความและภาพ ต้องเลือกเฉพาะภาพ"""
    payload = _api_response_with_image(np.zeros((10, 10, 3), np.uint8))
    assert isinstance(gemini._extract_image(payload), str)


def test_extract_image_accepts_snake_case() -> None:
    payload = {
        "candidates": [
            {"content": {"parts": [{"inline_data": {"data": "abc123"}}]}}
        ]
    }
    assert gemini._extract_image(payload) == "abc123"


def test_extract_image_reports_block_reason() -> None:
    payload = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(gemini.AiError) as info:
        gemini._extract_image(payload)
    assert "SAFETY" in str(info.value)


def test_extract_image_reports_missing_image() -> None:
    with pytest.raises(gemini.AiError):
        gemini._extract_image({"candidates": []})


def test_decode_image_rejects_garbage() -> None:
    with pytest.raises(gemini.AiError):
        gemini._decode_image(base64.b64encode(b"not an image").decode())


def test_decode_image_handles_grayscale() -> None:
    gray = np.full((8, 8), 255, np.uint8)
    ok, buffer = cv2.imencode(".png", gray)
    assert ok
    decoded = gemini._decode_image(base64.b64encode(buffer.tobytes()).decode())
    assert decoded.ndim == 3, "ต้องแปลงเป็นภาพสีเพื่อให้ขั้นถัดไปใช้ต่อได้"
    assert decoded.shape[2] == 3


# --- รูปแบบคำขอที่ส่งออกไป --------------------------------------------------


def test_request_is_well_formed(monkeypatch) -> None:
    """ตรวจคำขอที่ส่งออกไปจริง ไม่ใช่การอ่านช��้อความในซอร์ส

    ยังทดสอบกับเซิร์ฟเวอร์จริงไม่ได้เพราะไม่มีคีย์
    จึงตรวจเฉพาะสิ่งที่เราควบคุมได้ ได้แก่ URL, header, และรูปแบบเนื้อหา
    """
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return json.dumps(_api_response_with_image(np.zeros((6, 6, 3), np.uint8))).encode()

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(gemini.urllib.request, "urlopen", fake_urlopen)

    art = np.full((40, 40), 255, np.uint8)
    art[20, 5:35] = 0
    result = gemini.convert_to_lineart(art, load_settings({ENV_API_KEY: "test-key"}), "ทดสอบ")

    assert result.shape == (6, 6, 3)
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/models/" + DEFAULT_MODEL + ":generateContent")
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["timeout"] is not None

    parts = captured["body"]["contents"][0]["parts"]
    image_part = next(p for p in parts if "inline_data" in p)
    text_part = next(p for p in parts if "text" in p)
    assert image_part["inline_data"]["mime_type"] == "image/png"
    assert base64.b64decode(image_part["inline_data"]["data"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert text_part["text"] == "ทดสอบ"


def test_png_encoding_preserves_thin_lines() -> None:
    """ภาพลายเส้นต้องส่งเป็น PNG ไม่ใช่ JPEG

    JPEG จะทำให้เส้นบางแตกก่อนถึงปลายทาง ซึ่งทำให้ AI วาดตามผิด
    """
    art = np.full((100, 100), 255, np.uint8)
    art[50, 10:90] = 0
    mime, data = gemini._encode_png(art)
    assert mime == "image/png"
    assert base64.b64decode(data)[:8] == b"\x89PNG\r\n\x1a\n"
