"""ทดสอบการโหลดไฟล์และจัดการชื่อไฟล์

ชื่อไฟล์ภาษาไทยคือปัญหาที่พบบ่อยที่สุดในโปรเจกต์ลักษณะนี้
เพราะ cv2.imread จะอ่านไฟล์ที่มีอักขระนอก ASCII ไม่ได้เลย
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.lineart import imgio
from tests import fixtures


def test_decode_png_bytes() -> None:
    ok, buffer = cv2.imencode(".png", fixtures.synthetic_lineart(200, 200))
    assert ok
    image = imgio.imdecode(buffer.tobytes())
    assert image.shape == (200, 200, 3)


def test_load_upload_rejects_empty() -> None:
    with pytest.raises(imgio.ImageLoadError):
        imgio.load_upload(b"", "a.png")


def test_load_upload_rejects_non_image() -> None:
    """ไฟล์ที่ไม่ใช่ภาพต้องโยน error ที่เราจับได้ ไม่ใช่ error กลางๆ ของ Pillow"""
    with pytest.raises(imgio.ImageLoadError):
        imgio.load_upload(b"this is plain text", "note.png")


def test_load_upload_handles_truncated_file() -> None:
    ok, buffer = cv2.imencode(".png", fixtures.synthetic_lineart(200, 200))
    assert ok
    truncated = buffer.tobytes()[: len(buffer.tobytes()) // 2]
    with pytest.raises(imgio.ImageLoadError):
        imgio.load_upload(truncated, "cut.png")


def test_alpha_is_composited_onto_white() -> None:
    """ภาพโปร่งใสต้องกลายเป็นพื้นขาว เพราะหนังสือระบายสีพิมพ์บนกระดาษขาว"""
    rgba = np.zeros((60, 60, 4), np.uint8)
    rgba[..., 3] = 0  # โปร่งใสหมด
    rgba[20:40, 20:40, :3] = 0  # สี่เหลี่ยมดำ
    rgba[20:40, 20:40, 3] = 255
    ok, buffer = cv2.imencode(".png", rgba)
    assert ok

    image = imgio.imdecode(buffer.tobytes())
    assert image.shape[2] == 3
    assert image[5, 5].tolist() == [255, 255, 255], "พื้นหลังต้องเป็นขาว"
    assert image[30, 30].tolist() == [0, 0, 0], "ส่วนที่ทึบต้องเป็นดำ"


def test_grayscale_input_is_accepted() -> None:
    gray = np.full((40, 40), 128, np.uint8)
    ok, buffer = cv2.imencode(".png", gray)
    assert ok
    image = imgio.imdecode(buffer.tobytes())
    assert image.shape == (40, 40, 3)


def test_is_supported() -> None:
    assert imgio.is_supported("a.png")
    assert imgio.is_supported("a.JPG")
    assert imgio.is_supported("รูป.webp")
    assert not imgio.is_supported("a.pdf")
    assert not imgio.is_supported("a")


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("ยีระแหน.png", "ยีระแหน"),
        ("my cat.jpg", "my cat"),
        ("ภาพที่ 2 เป็นรูป.jpeg", "ภาพที่ 2 เป็นรูป"),
        ("no_extension", "no extension"),
        ("archive.tar.gz", "archive tar gz"),
        ("  spaced  .png", "spaced"),
        ("กระต่าย (น่ารัก)!.png", "กระต่าย น่ารัก"),
    ],
)
def test_caption_from_filename(filename: str, expected: str) -> None:
    assert imgio.caption_from_filename(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "ยีระแหน.png",
        "ภาพที่ 2 เป็นรูป.jpeg",
        "กระต่ายกินผัก.png",
        "ปลาที่นี่มีสองตัว.png",
    ],
)
def test_caption_keeps_thai_vowels_and_tone_marks(filename: str) -> None:
    """สระและวรรณยุกต์ต้องไม่หาย

    เคยพลาดเพราะใช้ \\w กรองชื่อไฟล์ ซึ่งไม่นับอักขระประเภท M
    ทำให้ "ยีระแหน" กลายเป็น "ยระแหน" และพิมพ์ผิดออกไปกับหน้ากระดาษ
    """
    caption = imgio.caption_from_filename(filename)
    stem = filename.rsplit(".", 1)[0]
    # เทียบจำนวนอักขระที่ไม่ใช่ช่องว่าง ชื่อกำกับอาจจัดระยะว่างใหม่ได้
    assert len(caption.split()) == len(stem.split())
    assert "".join(caption.split()) == "".join(stem.split()), (
        f"ชื่อกำกับถูกตัดอักขระทิ้ง: {caption!r}"
    )


def test_caption_never_empty() -> None:
    assert imgio.caption_from_filename("...png")
    assert imgio.caption_from_filename(".png")
    assert imgio.caption_from_filename("___.png")


def test_slugify_strips_non_ascii() -> None:
    assert imgio.slugify("สมุดระบายสี") == "coloring-book"
    assert imgio.slugify("สมุดระบายสี", fallback="fallback") == "fallback"


def test_slugify_keeps_latin() -> None:
    assert imgio.slugify("My Coloring Book") == "My-Coloring-Book"
    assert imgio.slugify("book-2024_v1") == "book-2024_v1"


def test_slugify_is_ascii_and_safe() -> None:
    """ชื่อไฟล์ที่ส่งให้ดาวน์โหลดต้องไม่มีทางหลอกเข้าโครงสร้าง path

    ขึ้นต้นด้วยจุดหรือขีดกลางก็ถือว่าไม่ปลอดภัย
    """
    for raw in ["../../etc/passwd", "a/b/c", "....", "///", "con:aux", "ภาษาไทย"]:
        result = imgio.slugify(raw)
        assert result.isascii(), raw
        assert "/" not in result and "\\" not in result, raw
        assert not result.startswith("."), raw
        assert not result.startswith("-"), raw


def test_slugify_output_is_bounded() -> None:
    assert len(imgio.slugify("x" * 500)) <= 80
