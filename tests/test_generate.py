"""ทดสอบการสร้างภาพระบายสีจากข้อความด้วย AI

หลักการเดียวกับการแปลงภาพ: การสร้างภาพต้องไม่ทำให้เซิร์ฟเวอร์ล้ม
ถ้าไม่มีคีย์หรือ AI ตอบผิดพลาด ต้องคืนข้อความที่ผู้ใช้อ่านเข้าใจ
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import ai
from app.ai import generate
from app.ai.settings import ENV_API_KEY, AiProvider, load_settings
from app.main import app

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """กันไม่ให้เทสต์ยิงไปที่ API จริง"""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("เทสต์พยายามติดต่อเครือข่าย ต้องแทนที่ urlopen")

    monkeypatch.setattr(generate.urllib.request, "urlopen", forbidden)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def sample_image() -> np.ndarray:
    """ภาพสังเคราะห์ขนาดเล็กใช้เป็นผลลัพธ์ของ AI แทนการเรียกจริง"""
    image = np.full((400, 400, 3), 255, np.uint8)
    cv2.rectangle(image, (50, 50), (350, 350), (0, 0, 0), 8)
    return image


# --- สไตล์และการประกอบคำสั่ง ---------------------------------------------------


def test_styles_are_listed_with_keys_and_labels() -> None:
    styles = generate.list_styles()
    assert len(styles) >= 4
    assert all("key" in s and "label" in s for s in styles)
    keys = {s["key"] for s in styles}
    assert "kids_easy" in keys and "classic" in keys


def test_build_prompt_contains_text_and_style() -> None:
    prompt = generate.build_prompt("แมวใส่หมวก", "kids_easy")
    assert "แมวใส่หมวก" in prompt
    assert "เส้นดำ" in prompt, "ต้องบังคับให้ผลลัพธ์เป็นภาพระบายสีเสมอ"


def test_unknown_style_falls_back_to_default() -> None:
    known = generate.build_prompt("x", "classic")
    unknown = generate.build_prompt("x", "ไม่มีสไตล์นี้")
    assert unknown == known


# --- การสร้างภาพแบบไม่ต้องต่อเครือข่าย ------------------------------------------


def test_generate_requires_text() -> None:
    settings = load_settings({ENV_API_KEY: "k"})
    result = generate.generate_image("   ", None, settings)
    assert result.ok is False
    assert "คำบรรยาย" in (result.note or "")


def test_generate_reports_missing_key() -> None:
    """ไม่มีเจ้าไหนพร้อมเลยต้องบอกว่าจะตั้งค่าอะไร

    ต้องปิดทางออกฟรีไว้ด้วย ไม่ใช่บอกให้ใส่คีย์เฉพาะทางเดียว
    เพราะ Pollinations ใช้ได้เลยโดยไม่ต้องมีคีย์
    """
    settings = load_settings({"AI_PROVIDER": "gemini", "AI_ALLOW_FALLBACK": "false"})
    result = generate.generate_image("แมว", None, settings)
    assert result.ok is False
    assert ENV_API_KEY in (result.note or "")


def test_pollinations_needs_no_key_and_is_always_available() -> None:
    """ไม่มีคีย์ใด ๆ ก็ต้องสร้างภาพได้ เพราะ Pollinations ไม่ต้องใช้คีย์"""
    settings = load_settings({})
    assert settings.configured is True
    chain = generate.provider_chain(settings)
    assert [p.value for p in chain] == ["pollinations"]


def test_generate_rejects_overlong_prompt() -> None:
    settings = load_settings({ENV_API_KEY: "k"})
    result = generate.generate_image("ย" * 700, None, settings)
    assert result.ok is False


def test_generate_wraps_api_error_as_note(monkeypatch) -> None:
    settings = load_settings(
        {ENV_API_KEY: "k", "AI_PROVIDER": "gemini", "AI_ALLOW_FALLBACK": "false"}
    )

    def boom(*_args, **_kwargs):
        raise generate.AiError("โควตาหมด")

    monkeypatch.setattr(generate, "generate_with_gemini", boom)
    result = generate.generate_image("แมว", None, settings)
    assert result.ok is False
    assert "โควตาหมด" in (result.note or "")


# --- ระบบสำรองอัตโนมัติ -------------------------------------------------------


def test_falls_back_when_first_provider_has_no_credits(monkeypatch) -> None:
    """เจ้าหลักตอบ 402 ต้องข้ามไปเจ้าฟรีให้เอง ไม่ใช่ล้มให้ผู้ใช้ไปแก้ค่า"""
    settings = load_settings({"AI_PROVIDER": "huggingface", "HF_TOKEN": "t"})
    seen: list[str] = []

    def out_of_credits(prompt, _settings):
        seen.append("huggingface")
        raise generate.AiError("AI ตอบกลับข้อผิดพลาด 402: เครดิตหมด")

    monkeypatch.setattr(generate, "generate_with_huggingface", out_of_credits)
    monkeypatch.setattr(
        generate.freebies,
        "generate_with_pollinations",
        lambda prompt, _settings: seen.append("pollinations") or sample_image(),
    )

    result = generate.generate_image("แมว", None, settings)

    assert result.ok is True
    assert seen == ["huggingface", "pollinations"]
    assert result.provider == "pollinations"


def test_reports_every_provider_that_failed() -> None:
    """ทุกเจ้าล้มหมดต้องเห็นเหตุผลของทุกเจ้า ไม่ใช่แค่เจ้าสุดท้าย"""
    settings = load_settings({ENV_API_KEY: "k"})

    result = generate.generate_image("แมว", None, settings, provider="gemini")
    # เครือข่ายถูกบล็อกโดย fixture ทำให้ทุกเจ้าล้มจริง
    assert result.ok is False
    assert "gemini" in (result.note or "")
    assert "pollinations" in (result.note or "")


def test_fallback_can_be_turned_off() -> None:
    """AI_ALLOW_FALLBACK=false ต้องไม่เรียกเจ้าอื่นแม้เจ้าหลักไม่มีคีย์"""
    settings = load_settings({"AI_ALLOW_FALLBACK": "false"})
    assert settings.configured is False
    assert generate.provider_chain(settings) == []


def test_generate_returns_image_on_success(monkeypatch) -> None:
    settings = load_settings({ENV_API_KEY: "k"})
    monkeypatch.setattr(generate, "generate_with_gemini", lambda *_: sample_image())
    result = generate.generate_image("แมว", "kawaii", settings)
    assert result.ok is True
    assert result.image is not None and result.image.size > 0


def test_generate_uses_huggingface_when_chosen(monkeypatch) -> None:
    settings = load_settings({ENV_API_KEY: "k", "HF_TOKEN": "t"})
    called = {}

    def fake_hf(prompt, _settings):
        called["prompt"] = prompt
        return sample_image()

    monkeypatch.setattr(generate, "generate_with_huggingface", fake_hf)
    result = generate.generate_image("แมว", None, settings, provider="huggingface")
    assert result.ok is True
    # Hugging Face ใช้ FLUX ซึ่งอ่านไทยไม่ออก ต้องได้คำสั่งอังกฤษ
    assert called["prompt"].startswith("black and white coloring book line art")


# --- ภาษาของคำสั่งต้องตรงกับโมเดล -------------------------------------------


def test_english_only_models_get_english_prompt() -> None:
    """FLUX และ Stable Diffusion อ่านไทยไม่ออก ต้องได้คำสั่งอังกฤษ

    ถ้าส่งคำสั่งไทยไป ผลลัพธ์จะไม่ตรงคำบรรยายของผู้ใช้เลย
    """
    for provider in (
        AiProvider.CLOUDFLARE,
        AiProvider.POLLINATIONS,
        AiProvider.HUGGINGFACE,
    ):
        prompt = generate.build_prompt("แมวใส่หมวก", "kawaii", provider)
        # ส่วนที่บังคับรูปแบบภาพต้องเป็นอังกฤษ
        # ส่วนคำบรรยายของผู้ใช้ยังเป็นไทยได้ เพราะแปลงให้ผู้ใช้เองไม่ได้
        assert "coloring book" in prompt
        assert "วาดภาพระบายสี" not in prompt, f"{provider.value} ยังมีคำสั่งไทย"
        assert generate._has_thai(prompt), "คำบรรยายไทยของผู้ใช้ต้องยังอยู่"


def test_gemini_gets_thai_prompt() -> None:
    """Gemini เข้าใจไทยได้ ต้องได้คำสั่งภาษาไทยเพื่อความแม่นยำ"""
    prompt = generate.build_prompt("แมวใส่หมวก", "kawaii", AiProvider.GEMINI)
    assert generate._has_thai(prompt)
    assert "วาดภาพระบายสี" in prompt


def test_prompt_language_follows_the_provider_actually_used(monkeypatch) -> None:
    """คำสั่งต้องถูกประกอบใหม่ตามเจ้าที่เรียกจริง ไม่ใช่ครั้งเดียวตอนต้น

    ถ้าประกอบครั้งเดียวแล้วใช้ทุกเจ้า เจ้าที่อ่านไทยไม่ออกจะได้คำสั่งผิดภาษา
    """
    settings = load_settings({ENV_API_KEY: "k"})
    seen: list[str] = []

    def fake_gemini(prompt, _settings):
        seen.append(prompt)
        return sample_image()

    monkeypatch.setattr(generate, "generate_with_gemini", fake_gemini)
    generate.generate_image("แมว", None, settings, provider="gemini")

    assert len(seen) == 1
    assert generate._has_thai(seen[0]), "ต้องได้คำสั่งไทยเพราะเรียก Gemini จริง"


def test_user_is_warned_when_using_thai_with_english_model(monkeypatch) -> None:
    """พิมพ์ไทยแต่โมเดลอ่านไม่ออก ต้องเตือน ไม่ใช่ให้ผู้ใช้คิดว่าโปรแกรมเพี้ยน"""
    settings = load_settings({"CF_API_TOKEN": "t", "CF_ACCOUNT_ID": "a"})
    monkeypatch.setattr(
        generate.freebies,
        "generate_with_cloudflare",
        lambda *_a, **_k: sample_image(),
    )
    result = generate.generate_image("แมวใส่หมวก", None, settings)

    assert result.ok is True
    assert result.note and "ภาษาอังกฤษ" in result.note


def test_no_warning_when_prompt_is_already_english(monkeypatch) -> None:
    settings = load_settings({"CF_API_TOKEN": "t", "CF_ACCOUNT_ID": "a"})
    monkeypatch.setattr(
        generate.freebies,
        "generate_with_cloudflare",
        lambda *_a, **_k: sample_image(),
    )
    result = generate.generate_image("a cat wearing a hat", None, settings)

    assert result.ok is True
    assert result.note is None, "พิมพ์อังกฤษแล้วต้องไม่เตือน"


def test_to_png_data_url_round_trips() -> None:
    image = sample_image()
    data_url = generate.to_png_data_url(image)
    assert data_url.startswith("data:image/png;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape[0] == 400


# --- รูปแบบคำขอที่ส่งให้ Gemini --------------------------------------------------


def test_gemini_request_is_text_only(monkeypatch) -> None:
    """คำขอสร้างภาพต้องส่งข้อความล้วน ไม่แนบภาพ"""
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            ok, buffer = cv2.imencode(".png", sample_image())
            data = base64.b64encode(buffer.tobytes()).decode("ascii")
            return json.dumps(
                {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": data}}]}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(generate.urllib.request, "urlopen", fake_urlopen)
    settings = load_settings({ENV_API_KEY: "test-key"})
    image = generate.generate_with_gemini("วาดแมว", settings)

    assert image.shape[0] == 400
    assert "test-key" in captured["url"] or captured["url"].endswith(":generateContent")
    parts = captured["body"]["contents"][0]["parts"]
    assert len(parts) == 1
    assert "text" in parts[0], "ต้องเป็นข้อความล้วน ไม่มี inline_data"


# --- เส้นทาง API ---------------------------------------------------------------


def test_generate_endpoint_rejects_unconfigured(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main._ai_settings",
        lambda: load_settings({"AI_PROVIDER": "gemini", "AI_ALLOW_FALLBACK": "false"}),
    )
    response = client.post("/api/generate", json={"prompt": "แมว"})
    assert response.status_code == 400
    assert ENV_API_KEY in response.json()["detail"]


def test_generate_endpoint_returns_data_urls(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("app.main._ai_settings", lambda: load_settings({ENV_API_KEY: "k"}))
    monkeypatch.setattr(ai, "generate_image", lambda *_a, **_k: generate.GenerateResult(sample_image()))
    response = client.post(
        "/api/generate", json={"prompt": "แมวการ์ตูน", "style": "kawaii", "count": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) == 2
    assert data["images"][0]["data"].startswith("data:image/png;base64,")
    assert data["failed"] == []


def test_generate_endpoint_reports_partial_failure(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("app.main._ai_settings", lambda: load_settings({ENV_API_KEY: "k"}))
    outcomes = iter(
        [
            generate.GenerateResult(sample_image()),
            generate.GenerateResult(None, "โควตาหมดชั่วคราว"),
        ]
    )
    monkeypatch.setattr(ai, "generate_image", lambda *_a, **_k: next(outcomes))
    response = client.post("/api/generate", json={"prompt": "แมว", "count": 2})
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) == 1
    assert "โควตาหมดชั่วคราว" in data["failed"]


def test_generate_endpoint_fails_when_all_attempts_fail(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("app.main._ai_settings", lambda: load_settings({ENV_API_KEY: "k"}))
    monkeypatch.setattr(
        ai, "generate_image", lambda *_a, **_k: generate.GenerateResult(None, "AI ล่ม")
    )
    response = client.post("/api/generate", json={"prompt": "แมว"})
    assert response.status_code == 502
    assert "AI ล่ม" in response.json()["detail"]


def test_generate_endpoint_validates_prompt(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("app.main._ai_settings", lambda: load_settings({ENV_API_KEY: "k"}))
    response = client.post("/api/generate", json={"prompt": "   "})
    assert response.status_code == 422


def test_generate_endpoint_rejects_bad_count(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("app.main._ai_settings", lambda: load_settings({ENV_API_KEY: "k"}))
    response = client.post("/api/generate", json={"prompt": "แมว", "count": 99})
    assert response.status_code == 422


def test_dns_failure_message_points_to_cause(monkeypatch) -> None:
    """error แบบ [Errno -5] ต้องแปลงเป็นข้อความที่บอกสาเหตุและวิธีแก้"""
    import urllib.error

    from app.ai.gemini import describe_connection_error

    def reason(msg):
        return urllib.error.URLError(RuntimeError(msg))

    dns = describe_connection_error(
        "https://ai.example.com/v1/models/m:generateContent",
        reason("[Errno -5] No address associated with hostname"),
    )
    assert "ai.example.com" in dns
    assert "AI_BASE_URL" in dns
    assert "อินเทอร์เน็ต" in dns

    timeout = describe_connection_error(
        "https://ai.example.com/v1", reason("timed out")
    )
    assert "AI_TIMEOUT_SECONDS" in timeout

    other = describe_connection_error(
        "https://ai.example.com/v1", reason("weird failure")
    )
    assert "ai.example.com" in other and "weird failure" in other


def test_generate_endpoint_surfaces_dns_note(client: TestClient, monkeypatch) -> None:
    """ล้มเพราะ DNS ต้องเห็นชื่อโดเมนที่หาไม่เจอในข้อความตอบกลับ"""
    import urllib.error

    def fake_urlopen(*_args, **_kwargs):
        raise urllib.error.URLError(RuntimeError("[Errno -5] No address associated with hostname"))

    monkeypatch.setattr("app.main._ai_settings", lambda: load_settings({ENV_API_KEY: "k"}))
    monkeypatch.setattr(generate.urllib.request, "urlopen", fake_urlopen)
    response = client.post("/api/generate", json={"prompt": "แมว"})
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "No address" not in detail or "AI_BASE_URL" in detail
    assert "generativelanguage" in detail or "AI_BASE_URL" in detail


def test_health_lists_styles(client: TestClient) -> None:
    data = client.get("/api/health").json()
    styles = data["ai"]["styles"]
    assert isinstance(styles, list) and len(styles) >= 4
