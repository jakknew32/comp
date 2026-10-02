"""โหลดและบันทึกภาพ

จุดที่ต้องระวัง: cv2.imread พังทันทีถ้าชื่อไฟล์เป็นภาษาไทยหรือมีอักขระพิเศษ
โมดูลนี้จึงอ่านไฟล์เป็น bytes ด้วย Python ก่อน แล้วค่อย decode
"""

from __future__ import annotations

import io
import re
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import SUPPORTED_FORMATS

Image.MAX_IMAGE_PIXELS = 200_000_000


class ImageLoadError(ValueError):
    """เกิดข้อผิดพลาดตอนอ่านไฟล์ภาพ"""


def imdecode(data: bytes) -> np.ndarray:
    """ถอดรหัส bytes เป็นภาพ BGR โดยไม่ผ่าน filesystem

    ใช้ตัวนี้แทน cv2.imread / cv2.imdecode ตรงๆ เพราะ imdecode ในบาง build
    ตัดรหัสอักขระนอก ASCII ใน metadata ทิ้ง ทำให้ไฟล์บางชนิด decode ไม่ขึ้น
    """
    if not data:
        raise ImageLoadError("ไฟล์ว่างเปล่า")

    try:
        pil_img = Image.open(io.BytesIO(data))
        # PIL อ่านหัวไฟล์ตอน open แต่ถอดรหัสจริงตอน load
        # ไฟล์ที่ถูกตัดหรือเสียหายจะพังตรงนี้ ไม่ใช่ตอน open
        # จึงต้องอยู่ในบล็อก try เดียวกัน
        pil_img.load()
    except UnidentifiedImageError as exc:
        # ไฟล์ที่ไม่ใช่ภาพจริง เช่น เปลี่ยนนามสกุลจาก .txt เป็น .png
        # ถ้าปล่อยให้หลุดออกไปจะกลายเป็น error 500 ซึ่งไม่มีความหมายกับผู้ใช้
        raise ImageLoadError("ไฟล์นี้ไม่ใช่ภาพที่อ่านได้ กรุณาตรวจสอบว่าเป็นไฟล์ภาพจริง") from exc
    except OSError as exc:
        raise ImageLoadError("เปิดไฟล์ภาพไม่สำเร็จ ไฟล์อาจไม่ครบหรือเสียหาย") from exc

    with pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "LA", "PA"):
            # รวม alpha เป็นพื้นหลังขาว เพราะหนังสือระบายสีต้องพิมพ์บนพื้นขาว
            pil_img = pil_img.convert("RGBA")
            canvas = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            canvas.alpha_composite(pil_img)
            pil_img = canvas.convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")
        rgb = np.asarray(pil_img)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_upload(raw: bytes, filename: str) -> np.ndarray:
    """โหลดไฟล์ที่อัปโหลดมาจากเว็บ โดยตรวจชนิดไฟล์และขนาดก่อน"""
    if not raw:
        raise ImageLoadError("ไฟล์ว่างเปล่า")
    return imdecode(raw)


def is_supported(filename: str) -> bool:
    return filename.lower().endswith(SUPPORTED_FORMATS)


_SLUG_KEPT_PUNCT = " -"


def _keep_char(char: str) -> bool:
    """ตัดสินว่าอักขระควรอยู่ในชื่อกำกับหรือไม่

    ต้องเก็บอักขระประเภท M (สระ วรรณยุกต์ และเครื่องหมายเรียงซ้อน) ด้วย
    เพราะภาษาไทวางสระไว้หลังพยัญชนะ ถ้าใช้ \\w ตามปกติจะไม่นับ
    อักขระเหล่านั้นเป็นตัวอักษร แล้วสระจะหายไปทั้งคำ
    เคยพลาดตรงนี้ ทำให้ชื่อกำกับกลายเป็น "ยระแหน" แทน "ยีระแหน"
    """
    if char in _SLUG_KEPT_PUNCT:
        return True
    if char.isalnum():
        return True
    return unicodedata.category(char).startswith("M")


def caption_from_filename(filename: str) -> str:
    """เดาชื่อกำกับจากชื่อไฟล์ โดยตัดนามสกุลและอักขระไม่พึงประสงค์ออก

    เก็บอักขระไทยไว้ทั้งหมด รวมถึงสระและวรรณยุกต์
    """
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for ext in SUPPORTED_FORMATS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = unicodedata.normalize("NFC", stem)
    # อักขระที่ไม่ต้องการกลายเป็นช่องว่าง เพื่อให้คำไม่ถูกติดกัน
    cleaned = "".join(c if _keep_char(c) else " " for c in stem)
    cleaned = " ".join(cleaned.split())
    return cleaned or "ระบายสี"


def slugify(text: str, fallback: str = "coloring-book") -> str:
    """ทำ slug ปลอดภัยสำหรับชื่อไฟล์ที่จะส่งให้ดาวน์โหลด

    โครงสร้างไฟล์ต้องเป็น ASCII เท่านั้น และห้ามขึ้นต้นด้วยจุดหรือขีดกลาง
    จึงต้องตัดอักขระอื่นทิ้งทั้งหมด ไม่ใช่แค่แทนที่
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return text[:80] or fallback
