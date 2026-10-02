"""โหลดภาพแบบรองรับชื่อไฟล์ภาษาไทยและ EXIF orientation.

``cv2.imread`` พังเมื่อ path ไม่ใช่ ASCII เพราะ OpenCV ใช้ ``fopen``
ซึ่งไม่รองรับชื่อไฟล์ Unicode บน Windows — ทางออกคืออ่านไฟล์เป็น bytes
ด้วย ``np.fromfile`` แล้ว decode ในหน่วยความจำ ทุกเส้นทางในโปรเจกต์นี้
ต้องใช้ :func:`imread_unicode` เท่านั้น
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

PathLike = Union[str, Path, bytes]

#: MIME type ที่ยอมรับ
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


class ImageLoadError(ValueError):
    """เกิดข้อผิดพลาดขณะอ่านภาพ (ไฟล์เสีย, format ไม่รองรับ, ใหญ่เกินกำหนด)"""


def imread_unicode(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """อ่านภาพจาก path ที่อาจเป็นภาษาไทย คืนค่า ``uint8`` BGR.

    ใช้ ``np.fromfile`` + ``cv2.imdecode`` เพื่อเลี่ยงข้อจำกัดของ ``fopen``
    บน Windows ที่ไม่รองรับ Unicode
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:  # pragma: no cover - ขึ้นกับ filesystem
        raise ImageLoadError(f"อ่านไฟล์ไม่สำเร็จ: {exc}") from exc

    if buf.size == 0:
        raise ImageLoadError("ไฟล์ว่างเปล่า")

    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ImageLoadError("ไฟล์นี้ไม่ใช่ภาพที่อ่านได้ หรือไฟล์เสีย")
    return img


def decode_bytes(data: bytes) -> np.ndarray:
    """ถอดภาพจาก bytes (ใช้กับไฟล์ที่อัปโหลด) คืนค่า ``uint8`` BGR."""
    if not data:
        raise ImageLoadError("ไฟล์ว่างเปล่า")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError("ไฟล์นี้ไม่ใช่ภาพที่อ่านได้ หรือไฟล์เสีย")
    return img


def load_bgr(
    source: Union[PathLike, bytes],
) -> tuple[np.ndarray, str]:
    """โหลดภาพเป็น BGR พร้อมจัดการ EXIF orientation

    คืนค่า ``(ภาพ BGR uint8, ชื่อไฟล์ฐาน)``

    ใช้ PIL เป็นตัวหน้าเพราะ PIL อ่านชื่อไฟล์ Unicode ได้เองและจัดการ
    EXIF orientation ได้ถูกต้อง จากนั้นค่อยแปลงกลับเป็น BGR
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        name = "image"
    else:
        path = Path(source)
        if not path.exists():
            raise ImageLoadError(f"ไม่พบไฟล์: {path.name}")
        data = path.read_bytes()
        name = path.name

    try:
        with Image.open(io.BytesIO(data)) as pil:
            fmt = (pil.format or "").upper()
            if fmt and fmt not in ALLOWED_FORMATS:
                raise ImageLoadError(f"รองรับเฉพาะ JPEG/PNG/WEBP (พบ {fmt})")
            pil = ImageOps.exif_transpose(pil)
            pil = pil.convert("RGB")
            rgb = np.asarray(pil, dtype=np.uint8)
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError(f"อ่านภาพไม่สำเร็จ: {exc}") from exc

    if rgb.size == 0 or rgb.ndim != 3:
        raise ImageLoadError("ภาพไม่มีข้อมูล")

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), name


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """แปลง BGR เป็น grayscale uint8."""
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# ชื่อไฟล์
# --------------------------------------------------------------------------

_THAI_EXT_RE = re.compile(
    r"\.(png|jpe?g|webp|bmp|tiff?)\Z", re.IGNORECASE
)


def decode_upload_filename(raw: Optional[str]) -> str:
    """ถอดชื่อไฟล์ที่ Werkzeug ส่งมาให้เป็นชื่อจริง

    Werkzeug ถอด multipart filename ที่ encode มาแบบ latin-1 หรือ utf-8
    ไว้ และทางมาตรฐานคือลอง ``utf-8`` ก่อน ถ้าไม่ผ่านใช้ ``cp437``
    ซึ่งเป็นการถอดแบบ one-to-one ของ byte
    """
    if not raw:
        return ""
    for encoding in ("utf-8", "cp437"):
        try:
            return raw.encode("cp437").decode(encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return raw


def default_caption(filename: str) -> str:
    """ตัดนามสกุลออกจากชื่อไฟล์ เพื่อใช้เป็นชื่อกำกับเริ่มต้น"""
    name = decode_upload_filename(filename)
    stem = _THAI_EXT_RE.sub("", name)
    stem = unicodedata.normalize("NFC", stem).strip()
    # แทนขีดกลางด้วยช่องว่าง แล้วยุบช่องว่างซ้ำ
    stem = re.sub(r"[-_]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "ระบายสี"


def slugify(text: str, max_len: int = 60) -> str:
    """สร้าง slug ปลอดภัยสำหรับใช้เป็นชื่อไฟล์ (ASCII เท่านั้น)

    ถ้าข้อความไม่มีอักขระ ASCII เลย (เช่น ชื่อสมุดภาษาไทยล้วน) จะใช้
    hash แบบ sha1 ซึ่ง**คงที่ข้ามการรัน** ต่างจาก ``hash()`` ของ Python
    ที่สุ่มค่าทุกครั้งที่เริ่มโปรแกรม
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    if not slug:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        slug = f"book-{digest}"
    return slug[:max_len]
