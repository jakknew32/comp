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


def test_generate_shows_real_api_error_gemini(monkeypatch) -> None:
    """ผู้ใช้ต้องเห็นข้อความจริงจาก API (เช่นคีย์ผิด) ไม่ใช่ AttributeError"""

    def fake_urlopen(*_args, **_kwargs):
        raise _http_error(401, {"error": "Invalid credentials in Authorization header"})

    monkeypatch.setattr(generate.urllib.request, "urlopen", fake_urlopen)
    result = generate.generate_image(
        "แมว", None, _settings(AiProvider.GEMINI), provider="gemini"
    )
    assert not result.ok
    assert "Invalid credentials" in result.note
    assert "AttributeError" not in result.note


# --- Hugging Face: สร้างภาพผ่าน Inference Providers ---------------------------
# hf-inference เดิมตอบ 410 "model is deprecated" สำหรับโมเดลสร้างภาพ จึงเรียกผ่าน huggingface_hub


def _hf_http_error(status: int, message: str = "boom"):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(status, request=httpx.Request("POST", "http://x"), text=message)
    return HfHubHTTPError(message, response=response)


class _FakeClient:
    """แทน InferenceClient: คืนผลตามสคริปต์ของแต่ละโมเดล"""

    script: dict = {}
    calls: list = []

    def __init__(self, **kwargs) -> None:
        type(self).init_kwargs = kwargs

    def text_to_image(self, prompt, *, model=None, **_kw):
        type(self).calls.append(model)
        outcome = type(self).script.get(model, type(self).script.get("*"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture()
def fake_hf(monkeypatch):
    import huggingface_hub

    _FakeClient.script = {}
    _FakeClient.calls = []
    monkeypatch.setattr(huggingface_hub, "InferenceClient", _FakeClient)
    return _FakeClient


def _picture():
    from PIL import Image

    return Image.new("RGB", (64, 48), (255, 255, 255))


def _hf_settings(**kw) -> AiSettings:
    return AiSettings(provider=AiProvider.HUGGINGFACE, hf_token="hf_x", **kw)


def test_hf_generation_returns_bgr_array(fake_hf) -> None:
    fake_hf.script = {"*": _picture()}
    image = generate.generate_with_huggingface("แมว", _hf_settings())
    assert image.shape == (48, 64, 3)
    assert fake_hf.init_kwargs["provider"] == "auto"
    assert fake_hf.init_kwargs["api_key"] == "hf_x"


def test_hf_provider_can_be_chosen(fake_hf) -> None:
    fake_hf.script = {"*": _picture()}
    generate.generate_with_huggingface("แมว", _hf_settings(hf_provider="fal-ai"))
    assert fake_hf.init_kwargs["provider"] == "fal-ai"


def test_hf_falls_back_when_model_is_gone(fake_hf) -> None:
    """โมเดลแรกตอบ 410 (ถูกเลิก) ต้องลองโมเดลสำรองต่อและสำเร็จ"""
    fake_hf.script = {
        "black-forest-labs/FLUX.1-schnell": _hf_http_error(410, "model is deprecated"),
        "black-forest-labs/FLUX.1-dev": _picture(),
    }
    image = generate.generate_with_huggingface("แมว", _hf_settings())
    assert image.size > 0
    assert fake_hf.calls[:2] == [
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev",
    ]


def test_hf_old_alias_maps_to_text2image_model(fake_hf) -> None:
    fake_hf.script = {"*": _picture()}
    generate.generate_with_huggingface("แมว", _hf_settings(hf_model="lineart_sd15"))
    assert fake_hf.calls[0] == "black-forest-labs/FLUX.1-schnell"


def test_hf_all_models_gone_gives_actionable_message(fake_hf) -> None:
    fake_hf.script = {"*": _hf_http_error(410, "deprecated")}
    result = generate.generate_image(
        "แมว", None, _hf_settings(), provider="huggingface"
    )
    assert not result.ok
    assert "AI_PROVIDER=gemini" in result.note


@pytest.mark.parametrize(
    "status,fragment",
    [(401, "HF_TOKEN"), (403, "Inference Providers"), (402, "เครดิต"), (429, "rate limit")],
)
def test_hf_fatal_errors_stop_immediately_with_clear_message(
    fake_hf, status: int, fragment: str
) -> None:
    fake_hf.script = {"*": _hf_http_error(status, "x")}
    result = generate.generate_image("แมว", None, _hf_settings(), provider="huggingface")
    assert not result.ok
    assert fragment in result.note
    assert len(fake_hf.calls) == 1  # ไม่ลองโมเดลอื่นต่อ เพราะปัญหาอยู่ที่ token/เครดิต


def test_hf_missing_library_gives_clear_message(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    result = generate.generate_image("แมว", None, _hf_settings(), provider="huggingface")
    assert not result.ok
    assert "huggingface_hub" in result.note
