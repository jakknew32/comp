"""ข้อความ error จาก AI ต้องแสดงสาเหตุจริงเสมอ ไม่ว่า API จะตอบ JSON โครงสร้างแบบไหน

เดิมถ้า API ตอบ {"error": "ข้อความ"} (Hugging Face ตอบแบบนี้) ตัวอ่าน error จะล้มด้วย
AttributeError: 'str' object has no attribute 'get' แล้วผู้ใช้ไม่เห็นสาเหตุจริงเลย
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from app.ai import gemini, generate
from app.ai.settings import AiProvider, AiSettings


def _http_error(code: int, body: object) -> urllib.error.HTTPError:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    return urllib.error.HTTPError("http://x", code, "Err", {}, io.BytesIO(raw))


BODIES = [
    ({"error": "Invalid credentials in Authorization header"}, "Invalid credentials"),  # HF
    ({"error": {"message": "API key not valid"}}, "API key not valid"),  # Google
    ("Model is currently loading", "currently loading"),  # JSON string ล้วน
    ([{"message": "bad request"}], "bad request"),  # รายการ
    ({"error": ["quota exceeded"]}, "quota exceeded"),  # error เป็นลิสต์
    ({"detail": "not allowed"}, "not allowed"),
]


@pytest.mark.parametrize("body,expected", BODIES)
def test_error_reader_handles_every_shape(body: object, expected: str) -> None:
    assert expected in gemini._read_http_error(_http_error(400, body))
    assert expected in generate.huggingface._read_http_error(_http_error(400, body))


def test_error_reader_falls_back_on_unusable_body() -> None:
    assert "คีย์" in gemini._read_http_error(_http_error(401, b"<html>nope</html>"))
    assert "ไม่พบโมเดล" in gemini._read_http_error(_http_error(404, {"unrelated": 1}))
    assert "โควตา" in gemini._read_http_error(_http_error(429, 12345))


@pytest.mark.parametrize("payload", ["just a string", ["x"], 5, None, {"candidates": "oops"},
                                     {"candidates": [None, "x", {"content": "y"}]},
                                     {"candidates": [{"content": {"parts": ["a", 3]}}]}])
def test_extract_image_never_raises_attribute_error(payload: object) -> None:
    with pytest.raises(gemini.AiError):
        gemini._extract_image(payload)


def _settings(provider: AiProvider) -> AiSettings:
    return AiSettings(provider=provider, api_key="k", hf_token="t")


@pytest.mark.parametrize("provider", [AiProvider.HUGGINGFACE, AiProvider.GEMINI])
def test_generate_shows_real_api_error(monkeypatch, provider: AiProvider) -> None:
    """ผู้ใช้ต้องเห็นข้อความจริงจาก API (เช่น token ผิด) ไม่ใช่ AttributeError"""

    def fake_urlopen(*_args, **_kwargs):
        raise _http_error(401, {"error": "Invalid credentials in Authorization header"})

    monkeypatch.setattr(generate.urllib.request, "urlopen", fake_urlopen)
    result = generate.generate_image("แมว", None, _settings(provider), provider=provider.value)
    assert not result.ok
    assert "Invalid credentials" in result.note
    assert "AttributeError" not in result.note
