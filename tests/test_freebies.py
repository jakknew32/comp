"""ทดสอบผู้ให้บริการฟรีที่ใช้เป็นตัวสำรอง

จุดที่ต้องระวังที่สุดคือ API ฟรีมักตอบรูปแบบไม่เหมือนกัน
บางเจ้าตอบเป็นไบต์ภาพตรง ๆ บางเจ้าตอบเป็น JSON ที่มี base64 ซ่อนอยู่
และบางเจ้าตอบ HTML เมื่อล่ม ซึ่งต้องไม่กลายเป็นภาพที่เพี้ยน
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error

import cv2
import numpy as np
import pytest

from app.ai import freebies
from app.ai.settings import AiProvider, load_settings


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """ไม่ให้เทสต์รอจริงเวลาเจอ 429"""
    monkeypatch.setattr(freebies.time, "sleep", lambda _seconds: None)


def png_bytes() -> bytes:
    image = np.full((64, 64, 3), 255, np.uint8)
    cv2.rectangle(image, (8, 8), (56, 56), (0, 0, 0), 3)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._payload


def http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    """สร้าง HTTPError ที่อ่าน body ได้ จำลองการตอบกลับจริง"""
    error = urllib.error.HTTPError("https://example.test", code, "err", {}, None)
    if body:
        error.addinfourl(io.BytesIO(body.encode("utf-8")), None)
    return error


# --- ตั้งค่า -----------------------------------------------------------------


def test_pollinations_is_ready_without_any_key() -> None:
    settings = load_settings({})
    assert settings.is_ready(AiProvider.POLLINATIONS) is True


def test_cloudflare_needs_both_token_and_account_id() -> None:
    assert load_settings({"CF_API_TOKEN": "t"}).is_ready(AiProvider.CLOUDFLARE) is False
    assert load_settings({"CF_ACCOUNT_ID": "a"}).is_ready(AiProvider.CLOUDFLARE) is False
    both = load_settings({"CF_API_TOKEN": "t", "CF_ACCOUNT_ID": "a"})
    assert both.is_ready(AiProvider.CLOUDFLARE) is True


def test_unknown_provider_does_not_crash_loading() -> None:
    settings = load_settings({"AI_PROVIDER": "ไม่มีเจ้านี้"})
    assert settings.provider == AiProvider.GEMINI


# --- Pollinations -------------------------------------------------------------


def test_pollinations_returns_decoded_image(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        return FakeResponse(png_bytes())

    monkeypatch.setattr(freebies.urllib.request, "urlopen", fake_urlopen)
    image = freebies.generate_with_pollinations("แมว", load_settings({}))

    assert image.shape[0] == 64
    assert "แมว" in captured["url"] or "%E0" in captured["url"]
    assert "model=turbo" in captured["url"]


def test_pollinations_sends_token_when_configured(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["auth"] = request.headers.get("Authorization")
        return FakeResponse(png_bytes())

    monkeypatch.setattr(freebies.urllib.request, "urlopen", fake_urlopen)
    settings = load_settings({"POLLINATIONS_TOKEN": "tok-1234"})
    freebies.generate_with_pollinations("แมว", settings)

    assert captured["auth"] == "Bearer tok-1234"


def test_pollinations_waits_once_and_retries_on_rate_limit(monkeypatch) -> None:
    """429 แบบไม่มีโทเคนต้องรอแล้วลองใหม่ ไม่ใช่บอกผู้ใช้ให้กดเอง"""
    attempts = {"n": 0}
    slept: list[float] = []

    def fake_urlopen(_request, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise freebies.urllib.error.HTTPError(
                "https://image.pollinations.ai", 429, "slow down", {}, None
            )
        return FakeResponse(png_bytes())

    monkeypatch.setattr(freebies.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(freebies.time, "sleep", slept.append)

    image = freebies.generate_with_pollinations("แมว", load_settings({}))

    assert image.shape[0] == 64
    assert attempts["n"] == 2
    assert slept and slept[0] >= 10


def test_pollinations_gives_up_after_retry(monkeypatch) -> None:
    def fake_urlopen(_request, timeout=None):
        raise freebies.urllib.error.HTTPError(
            "https://image.pollinations.ai", 429, "slow down", {}, None
        )

    monkeypatch.setattr(freebies.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(freebies.AiError):
        freebies.generate_with_pollinations("แมว", load_settings({}))


def test_pollinations_rate_limit_with_token_does_not_retry(monkeypatch) -> None:
    """มีโทเคนแล้วรอก็ไม่ช่วย ต้องบอกให้ผู้ใช้รอเอง"""
    attempts = {"n": 0}

    def fake_urlopen(_request, timeout=None):
        attempts["n"] += 1
        raise freebies.urllib.error.HTTPError(
            "https://image.pollinations.ai", 429, "slow down", {}, None
        )

    monkeypatch.setattr(freebies.urllib.request, "urlopen", fake_urlopen)
    settings = load_settings({"POLLINATIONS_TOKEN": "tok"})

    with pytest.raises(freebies.AiError) as err:
        freebies.generate_with_pollinations("แมว", settings)

    assert attempts["n"] == 1
    assert "ลิมิต" in str(err.value)


def test_pollinations_rejects_html_error_page(monkeypatch) -> None:
    """ถ้าตอบมาเป็น HTML แปลว่าเจ้ากำลังล่ม ต้องบอกชัด ไม่ใช่คืนภาพเพี้ยน"""
    monkeypatch.setattr(
        freebies.urllib.request,
        "urlopen",
        lambda *_a, **_k: FakeResponse(b"<html>502 Bad Gateway</html>"),
    )
    with pytest.raises(freebies.AiError):
        freebies.generate_with_pollinations("แมว", load_settings({}))


def test_pollinations_reports_bad_token(monkeypatch) -> None:
    def fake_urlopen(_request, timeout=None):
        raise freebies.urllib.error.HTTPError(
            "https://image.pollinations.ai", 401, "nope", {}, None
        )

    monkeypatch.setattr(freebies.urllib.request, "urlopen", fake_urlopen)
    settings = load_settings({"POLLINATIONS_TOKEN": "bad"})
    with pytest.raises(freebies.AiError) as err:
        freebies.generate_with_pollinations("แมว", settings)
    assert "POLLINATIONS_TOKEN" in str(err.value)


# --- Cloudflare ---------------------------------------------------------------


def test_cloudflare_requires_configuration() -> None:
    with pytest.raises(freebies.AiError) as err:
        freebies.generate_with_cloudflare("แมว", load_settings({}))
    assert "CF_API_TOKEN" in str(err.value)


def test_cloudflare_unwraps_base64_from_json(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["auth"] = request.headers.get("Authorization")
        payload = {"result": {"image": base64.b64encode(png_bytes()).decode("ascii")}}
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(freebies.urllib.request, "urlopen", fake_urlopen)
    settings = load_settings({"CF_API_TOKEN": "tok", "CF_ACCOUNT_ID": "acct-1"})
    image = freebies.generate_with_cloudflare("แมว", settings)

    assert image.shape[0] == 64
    assert "acct-1" in captured["url"]
    assert "FLUX.1-schnell" in captured["url"]
    assert captured["auth"] == "Bearer tok"
    assert captured["body"]["prompt"] == "แมว"


def test_cloudflare_reports_api_errors_in_payload(monkeypatch) -> None:
    """Cloudflare ตอบ 200 แต่ body ไม่มีภาพ ต้องบอกเหตุผลจาก errors"""
    monkeypatch.setattr(
        freebies.urllib.request,
        "urlopen",
        lambda *_a, **_k: FakeResponse(
            json.dumps({"errors": [{"message": "model not found"}]}).encode("utf-8")
        ),
    )
    settings = load_settings({"CF_API_TOKEN": "tok", "CF_ACCOUNT_ID": "acct-1"})
    with pytest.raises(freebies.AiError) as err:
        freebies.generate_with_cloudflare("แมว", settings)
    assert "model not found" in str(err.value)


def test_cloudflare_explains_quota_exhausted(monkeypatch) -> None:
    def fake_urlopen(_request, timeout=None):
        raise freebies.urllib.error.HTTPError(
            "https://api.cloudflare.com", 429, "limit", {}, None
        )

    monkeypatch.setattr(freebies.urllib.request, "urlopen", fake_urlopen)
    settings = load_settings({"CF_API_TOKEN": "tok", "CF_ACCOUNT_ID": "acct-1"})
    with pytest.raises(freebies.AiError) as err:
        freebies.generate_with_cloudflare("แมว", settings)
    assert "10,000" in str(err.value)