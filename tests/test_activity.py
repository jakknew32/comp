"""ทดสอบหนังสือกิจกรรม: เขียนตามรอยประ, ฝึกลากเส้น, จับคู่"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.book import activity
from app.main import app
from tests import fixtures


def png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _convert_refs(client: TestClient, count: int) -> list[str]:
    refs = []
    for i in range(count):
        image = fixtures.synthetic_lineart() if i % 2 == 0 else fixtures.photocopied_lineart()
        response = client.post(
            "/api/convert", files={"file": (f"p{i}.png", png_bytes(image), "image/png")}
        )
        assert response.status_code == 200
        refs.append(response.json()["ref"])
    return refs


def test_dashed_glyph_is_hollow_and_dotted() -> None:
    from app.lineart import text

    solid = text.render_text("ก", 200, bold=True)
    dashed = activity._dashed_glyph(solid)
    solid_ink = int(np.count_nonzero(np.array(solid) > 127))
    dashed_ink = int(np.count_nonzero(np.array(dashed) > 127))
    # เส้นประต้องมีหมึกน้อยกว่าตัวอักษรทึบมาก และต้องมีหมึกอยู่ (ไม่ว่างเปล่า)
    assert 0 < dashed_ink < solid_ink * 0.5


def test_tracing_only_book(client: TestClient) -> None:
    response = client.post(
        "/api/activity",
        json={"trace_items": ["ก", "ข", "ค", "แมว", "สวัสดี"], "trace_blank_rows": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert all(p.startswith("data:image/png;base64,") for p in data["pages"])


def test_prewriting_only_book(client: TestClient) -> None:
    response = client.post("/api/activity", json={"prewriting": list(activity.PREWRITING_PATTERNS)})
    assert response.status_code == 200
    assert response.json()["count"] >= 1


def test_unknown_prewriting_pattern_rejected(client: TestClient) -> None:
    response = client.post("/api/activity", json={"prewriting": ["nonsense"]})
    assert response.status_code == 422


def test_empty_request_rejected(client: TestClient) -> None:
    response = client.post("/api/activity", json={})
    assert response.status_code == 422


def test_matching_with_images_and_answer_key(client: TestClient) -> None:
    refs = _convert_refs(client, 4)
    images = [{"ref": r, "caption": f"ชื่อ{i}"} for i, r in enumerate(refs)]
    plain = client.post("/api/activity", json={"images": images, "match_mode": "image_word"})
    with_key = client.post(
        "/api/activity", json={"images": images, "match_mode": "image_word", "answer_key": True}
    )
    assert plain.status_code == with_key.status_code == 200
    # เฉลยเพิ่มมา 1 หน้า (4 คู่ใน 1 หน้า)
    assert with_key.json()["count"] == plain.json()["count"] + 1


def test_matching_shadow_mode(client: TestClient) -> None:
    refs = _convert_refs(client, 3)
    response = client.post(
        "/api/activity",
        json={"images": [{"ref": r, "caption": "x"} for r in refs], "match_mode": "image_shadow"},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_matching_text_pairs(client: TestClient) -> None:
    response = client.post(
        "/api/activity",
        json={
            "match_mode": "text_pairs",
            "text_pairs": [["หมา", "dog"], ["แมว", "cat"], ["ปลา", "fish"]],
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_matching_needs_two_items(client: TestClient) -> None:
    refs = _convert_refs(client, 1)
    response = client.post("/api/activity", json={"images": [{"ref": refs[0], "caption": "x"}]})
    assert response.status_code == 422


def test_matching_reports_missing_refs(client: TestClient) -> None:
    refs = _convert_refs(client, 1)
    response = client.post(
        "/api/activity",
        json={"images": [{"ref": refs[0], "caption": "a"}, {"ref": "0" * 32, "caption": "b"}]},
    )
    assert response.status_code == 409
    assert response.json()["missing"] == [1]


def test_all_sections_together(client: TestClient) -> None:
    refs = _convert_refs(client, 5)
    response = client.post(
        "/api/activity",
        json={
            "prewriting": ["wave", "circle"],
            "trace_items": ["ก", "1"],
            "images": [{"ref": r, "caption": f"ภาพ{i}"} for i, r in enumerate(refs)],
            "pairs_per_page": 3,
            "answer_key": True,
        },
    )
    assert response.status_code == 200
    # ลากเส้น 1 + ตามรอย 1 + จับคู่ 5 ภาพที่ 3 ต่อหน้า = 2 หน้า + เฉลย 2 หน้า
    assert response.json()["count"] == 6


def test_derangement_never_matches_in_place() -> None:
    rng = random.Random(0)
    for n in range(2, 7):
        for _ in range(30):
            order = activity._derangement(n, rng)
            assert sorted(order) == list(range(n))
            assert all(order[i] != i for i in range(n))


def test_silhouette_fills_inside_of_outline() -> None:
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), 150, 255, 10)
    filled = activity._silhouette(mask)
    assert filled[200, 200] == 255  # ใจกลางวงกลมต้องถูกถมเต็ม
    assert filled[5, 5] == 0
