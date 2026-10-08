"""ทดสอบเส้นทาง API ของเว็บแอป"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import MAX_UPLOAD_BYTES
from app.main import app
from tests import fixtures

# starlette.testclient เตือนเรื่อง httpx รุ่นเก่า ไม่เกี่ยวกับผลทดสอบ
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def png_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


# --- หน้าเว็บพื้นฐาน ---------------------------------------------------------


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "สมุดระบายสี" in response.text


def test_static_assets_are_served(client: TestClient) -> None:
    for path in ("/static/app.js", "/static/style.css"):
        assert client.get(path).status_code == 200


def test_health_reports_fonts_ready(client: TestClient) -> None:
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["fonts_ready"] is True
    assert 1 in data["grids"] and 12 in data["grids"]


# --- ตัวอย่างหน้า A4 ---------------------------------------------------------


def test_preview_returns_png(client: TestClient) -> None:
    response = client.post(
        "/api/preview",
        files={"file": ("ยีระแหน.png", png_bytes(fixtures.synthetic_lineart()), "image/png")},
        data={"lineart": json.dumps({"target_line_mm": 2.5})},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image.shape[0] == 3508 and image.shape[1] == 2480


def test_preview_respects_null_params_as_auto(client: TestClient) -> None:
    """ค่าที่เป็น null ต้องถูกตีความว่าให้โปรแกรมตัดสินใจเอง ไม่ใช่ error"""
    response = client.post(
        "/api/preview",
        files={"file": ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")},
        data={"lineart": json.dumps({"target_line_mm": None, "speckle_ratio": None})},
    )
    assert response.status_code == 200


def test_preview_rejects_malformed_json_without_crashing(client: TestClient) -> None:
    response = client.post(
        "/api/preview",
        files={"file": ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")},
        data={"lineart": "{not json"},
    )
    assert response.status_code == 200


def test_preview_rejects_out_of_range_value(client: TestClient) -> None:
    """ค่าที่อยู่นอกช่วงที่ยอมรับต้องถูกปฏิเสธ แล้วใช้ค่าเริ่มต้นแทน"""
    response = client.post(
        "/api/preview",
        files={"file": ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")},
        data={"lineart": json.dumps({"target_line_mm": 999})},
    )
    assert response.status_code == 200


def test_preview_rejects_non_image(client: TestClient) -> None:
    """นามสกุลที่ไม่ใช่ภาพต้องถูกปฏิเสธตั้งแต่ต้น พร้อมข้อความที่เข้าใจได้"""
    response = client.post(
        "/api/preview",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "ไม่ใช่ภาพ" in response.json()["detail"]


def test_preview_accepts_uppercase_extension(client: TestClient) -> None:
    raw = png_bytes(fixtures.synthetic_lineart(400, 400))
    response = client.post(
        "/api/preview", files={"file": ("รูป.PNG", raw, "image/png")}
    )
    assert response.status_code == 200


def test_preview_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/preview", files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_preview_rejects_oversized_file(client: TestClient) -> None:
    oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/api/preview", files={"file": ("big.png", oversized, "image/png")}
    )
    assert response.status_code == 413


def test_preview_rejects_blank_image(client: TestClient) -> None:
    """ภาพขาวล้วนไม่มีเส้นให้ระบาย ต้องแจ้งชัด ไม่ใช่คืนหน้าว่าง"""
    blank = np.full((600, 400), 255, np.uint8)
    response = client.post(
        "/api/preview", files={"file": ("blank.png", png_bytes(blank), "image/png")}
    )
    assert response.status_code == 422
    assert "ไม่พบเส้น" in response.json()["detail"]


# --- สร้างสมุด --------------------------------------------------------------


def test_book_returns_pdf(client: TestClient) -> None:
    response = client.post(
        "/api/book",
        files=[
            ("files", ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")),
            ("files", ("b.png", png_bytes(fixtures.photocopied_lineart()), "image/png")),
        ],
        data={"captions": json.dumps(["ยีระแหน", "ถ่ายจากกระดาษ"])},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert response.headers["x-page-count"] == "3"
    assert "attachment" in response.headers["content-disposition"]


def test_book_uses_supplied_captions(client: TestClient) -> None:
    """ชื่อกำกับที่ผู้ใช้แก้ไขต้องถูกใช้ ไม่ใช่ชื่อไฟล์"""
    response = client.post(
        "/api/book",
        files=[("files", ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png"))],
        data={"captions": json.dumps(["ชื่อที่ผู้ใช้แก้"])},
    )
    assert response.status_code == 200


def test_book_handles_thai_and_spaced_filenames(client: TestClient) -> None:
    """ชื่อไฟล์ไทยและมีช่องว่างต้องถูกอ่านได้"""
    response = client.post(
        "/api/book",
        files=[
            ("files", ("ยีระแหน น่ารัก.png", png_bytes(fixtures.synthetic_lineart()), "image/png")),
            ("files", ("ภาพ ที่ สอง.png", png_bytes(fixtures.photocopied_lineart()), "image/png")),
        ],
    )
    assert response.status_code == 200
    assert response.headers["x-page-count"] == "3"


def test_book_filename_is_ascii(client: TestClient) -> None:
    """ชื่อไฟล์ดาวน์โหลดต้องเป็น ASCII โครงสร้างไฟล์ไม่รองรับภาษาไทย"""
    response = client.post(
        "/api/book",
        files=[("files", ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png"))],
        data={"book": json.dumps({"title": "สมุดระบายสีของฉัน"})},
    )
    disposition = response.headers["content-disposition"]
    assert disposition.isascii(), disposition
    assert ".pdf" in disposition


def test_book_without_any_file_is_a_client_error(client: TestClient) -> None:
    """ไม่ส่งไฟล์มาเลยต้องเป็นข้อผิดพลาดฝั่งผู้ใช้ ไม่ใช่เซิร์ฟเวอร์พัง"""
    response = client.post("/api/book", files=[])
    assert 400 <= response.status_code < 500


def test_book_rejects_malformed_captions(client: TestClient) -> None:
    response = client.post(
        "/api/book",
        files=[("files", ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png"))],
        data={"captions": "not-json"},
    )
    assert response.status_code == 400


def test_book_skips_bad_files_but_keeps_good_ones(client: TestClient) -> None:
    """ไฟล์เสียหนึ่งใบต้องไม่ทำให้ทั้งเล่มล้ม"""
    response = client.post(
        "/api/book",
        files=[
            ("files", ("good.png", png_bytes(fixtures.synthetic_lineart()), "image/png")),
            ("files", ("bad.png", b"not an image", "image/png")),
        ],
    )
    assert response.status_code == 200
    assert response.headers["x-page-count"] == "2"  # ปก + 1 หน้าเนื้อหา


def test_book_fails_when_no_image_is_usable(client: TestClient) -> None:
    response = client.post(
        "/api/book", files=[("files", ("bad.png", b"nope", "image/png"))]
    )
    assert response.status_code == 422


# --- ตรวจสอบค่าพารามิเตอร์ -----------------------------------------------------


def test_validate_echoes_resolved_params(client: TestClient) -> None:
    response = client.post(
        "/api/validate",
        data={"lineart": json.dumps({}), "book": json.dumps({"per_page": 4})},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["book"]["per_page"] == 4
    assert data["grid"] == [2, 2]
    assert data["cells_per_page"] == 4


def test_validate_falls_back_on_bad_per_page(client: TestClient) -> None:
    data = client.post(
        "/api/validate", data={"book": json.dumps({"per_page": 7})}
    ).json()
    assert data["book"]["per_page"] == 1


# --- ขั้นแปลงรูปและขั้นรวมเล่ม (แยกกัน) ------------------------------------


def test_convert_returns_thumbnail(client: TestClient) -> None:
    response = client.post(
        "/api/convert",
        files={"file": ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["thumb"].startswith("data:image/png;base64,")
    assert data["caption"]


def test_convert_rejects_blank_image(client: TestClient) -> None:
    blank = np.full((600, 400), 255, np.uint8)
    response = client.post(
        "/api/convert", files={"file": ("blank.png", png_bytes(blank), "image/png")}
    )
    assert response.status_code == 422


def test_pages_returns_every_page(client: TestClient) -> None:
    """รวมเล่มต้องคืนภาพของทุกหน้า (ปก + 2 หน้าเนื้อหา) ไม่ใช่แค่หน้าเดียว"""
    response = client.post(
        "/api/pages",
        files=[
            ("files", ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")),
            ("files", ("b.png", png_bytes(fixtures.photocopied_lineart()), "image/png")),
        ],
        data={"captions": json.dumps(["หนึ่ง", "สอง"])},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["pages"]) == 3
    assert all(p.startswith("data:image/png;base64,") for p in data["pages"])


def test_pages_without_cover(client: TestClient) -> None:
    response = client.post(
        "/api/pages",
        files=[("files", ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png"))],
        data={"book": json.dumps({"include_cover": False})},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_pages_rejects_when_nothing_usable(client: TestClient) -> None:
    blank = np.full((600, 400), 255, np.uint8)
    response = client.post(
        "/api/pages", files=[("files", ("blank.png", png_bytes(blank), "image/png"))]
    )
    assert response.status_code == 422


# --- รวมเล่มแบบไม่แปลง ------------------------------------------------------


def test_pages_without_conversion(client: TestClient) -> None:
    """เลือก skip_convert แล้วต้องรวมเล่มได้ โดยใช้ภาพตามที่เป็น"""
    response = client.post(
        "/api/pages",
        files=[
            ("files", ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")),
            ("files", ("b.png", png_bytes(fixtures.synthetic_lineart()), "image/png")),
        ],
        data={"lineart": json.dumps({"skip_convert": True})},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 3


def test_skip_convert_keeps_original_border(client: TestClient) -> None:
    """โหมดไม่แปลงต้องไม่ตัดกรอบเดิม ต่างจากโหมดแปลงปกติที่ตัดกรอบออก"""
    from app.config import LineArtParams
    from app.lineart import convert

    image = np.full((1200, 900), 255, np.uint8)
    cv2.rectangle(image, (30, 30), (870, 1170), 0, 12)
    cv2.circle(image, (450, 600), 200, 0, 10)

    kept = convert.passthrough(image)
    stripped = convert.convert(image, LineArtParams(strip_border=True))
    # ภาพต้นฉบับขนาดเดิม และยังมีหมึกของกรอบอยู่ครบ
    assert kept.mask.shape == image.shape
    assert int(np.count_nonzero(kept.mask)) > int(np.count_nonzero(stripped.mask))
