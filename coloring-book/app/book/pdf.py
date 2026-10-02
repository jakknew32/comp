"""รวมหน้าทั้งหมดเป็นไฟล์ PDF

หน้า A4 ทั้งเล่มเป็นภาพขาวดำ 1-bit ตามที่ :mod:`app.lineart.compose`
สร้างไว้ เขียนเป็น PDF แบบไม่ต้องใช้ไลบรารี PDF ภายนอกเลย
ข้อดีคือคุมกรอบมุมมน/grid/ข้อความไทยได้เป๊ะ เพราะไม่มีขั้นตอน PDF
ใด ๆ มากลายเป็นภาพอีกที

**หน่วยความจำ:** หน้าหนึ่งหน้า A4 300 DPI ดิบประมาณ 1.09 MB
โมดูลนี้บีบอัดหน้าทีละหน้าแล้วปล่อย stream ทันที ไม่เก็บภาพดิบทุกหน้า
ไว้ใน RAM พร้อมกัน (ความเสี่ยงข้อ 4)
"""

from __future__ import annotations

import io
import zlib
from typing import Iterable, Iterator, Optional

import numpy as np

from ..config import A4_HEIGHT_PX, A4_WIDTH_PX, MM_PER_INCH

#: 1 จุด = 1/72 นิ้ว
POINTS_PER_MM = 72.0 / MM_PER_INCH
A4_WIDTH_PT = 210.0 * POINTS_PER_MM
A4_HEIGHT_PT = 297.0 * POINTS_PER_MM


class PdfError(RuntimeError):
    """เกิดข้อผิดพลาดขณะสร้าง PDF"""


def pack_1bit(mask: np.ndarray) -> bytes:
    """บีบอัดภาพ binary เป็น 1 bit ต่อพิกเซล (MSB first)

    :param mask: ภาพ uint8 ที่พิกเซล > 127 คือ "หมึก"
    :returns: bytes ที่แต่ละแถวถูกเติมจนหารลงตัว
    """
    if mask.ndim != 2:
        raise PdfError("ต้องเป็นภาพขาวดำ 2 มิติ")

    bits = (mask > 127).astype(np.uint8)
    height, width = bits.shape

    row_bytes = (width + 7) // 8
    if row_bytes * 8 == width:
        # กรณีที่กว้างหารลงตัว 8 พอดี (A4 300 DPI = 2480 = 310 ไบต์)
        return np.packbits(bits, axis=1).tobytes()

    padded = np.zeros((height, row_bytes * 8), dtype=np.uint8)
    padded[:, :width] = bits
    return np.packbits(padded, axis=1).tobytes()


class PdfWriter:
    """เขียน PDF แบบสะสม object ตามลำดับ

    ใช้แบบเพิ่มหน้าทีละหน้า แล้วเรียก :meth:`build` เมื่อครบ
    """

    def __init__(
        self,
        width_pt: float = A4_WIDTH_PT,
        height_pt: float = A4_HEIGHT_PT,
        *,
        title: str = "สมุดระบายสี",
        compress: bool = True,
    ) -> None:
        self.width_pt = float(width_pt)
        self.height_pt = float(height_pt)
        self.title = title
        self.compress = compress
        self._buffer = io.BytesIO()
        self._offsets: list[int] = []
        self._page_refs: list[int] = []
        self._image_refs: list[int] = []
        self._next_ref = 1
        self._started = False

    # -- ภายใน ---------------------------------------------------------

    def _write(self, data: bytes) -> None:
        self._buffer.write(data)

    def _begin(self) -> None:
        """เขียนส่วนหัวของไฟล์ (ต้องทำก่อนเขียน object แรก)"""
        if self._started:
            return
        self._started = True
        self._write(b"%PDF-1.4\n")
        # ไบต์ที่สองต้องเป็นอักขระระดับสูง (binary marker) เพื่อให้ตัวอ่าน
        # ถือว่าเป็นไฟล์ binary
        self._write(b"%\xe2\xe3\xcf\xd3\n")

    def _add_object(self, body: bytes) -> int:
        """เขียน object หนึ่งตัว คืนเลขอ้างอิง"""
        self._begin()
        ref = self._next_ref
        self._next_ref += 1
        self._offsets.append(self._buffer.tell())
        self._write(f"{ref} 0 obj\n".encode("latin-1"))
        self._write(body)
        self._write(b"\nendobj\n")
        return ref

    # -- สาธารณะ --------------------------------------------------------

    def add_page(
        self,
        mask: np.ndarray,
        *,
        width_px: Optional[int] = None,
        height_px: Optional[int] = None,
    ) -> None:
        """เพิ่มหน้ากระดาษหนึ่งหน้าจากภาพลายเส้น

        :param mask: ภาพ uint8 (255 = หมึก) ต้องเป็นขาวดำ
        """
        if mask.ndim != 2:
            raise PdfError("หน้ากระดาษต้องเป็นภาพขาวดำ 2 มิติ")

        height, width = mask.shape[:2]
        if width_px is None:
            width_px = width
        if height_px is None:
            height_px = height

        raw = pack_1bit(mask)
        if self.compress:
            data = zlib.compress(raw, 9)
            filter_name = "/FlateDecode"
        else:
            data = raw
            filter_name = ""

        image_body = (
            f"<< /Type /XObject /Subtype /Image"
            f" /Width {width_px} /Height {height_px}"
            f" /ColorSpace /DeviceGray"
            f" /BitsPerComponent 1"
            f" {filter_name}"
            f" /Decode [1 0]"
            f" /Length {len(data)} >>\n".encode("latin-1")
            + b"stream\n"
            + data
            + b"\nendstream"
        )
        image_ref = self._add_object(image_body)

        page_body = (
            f"<< /Type /Page"
            f" /Parent 0 0 R"
            f" /MediaBox [0 0 {self.width_pt:.2f} {self.height_pt:.2f}]"
            f" /Resources << /XObject << /Im0 {image_ref} 0 R >> >>"
            f" /Contents 0 0 R >>".encode("latin-1")
        ).replace(b"/Parent 0 0 R", b"/Parent __PAGES__")
        page_ref = self._add_object(page_body)

        # เก็บไว้ แล้วแก้ /Parent ในตอน build
        self._page_refs.append(page_ref)
        self._image_refs.append(image_ref)

    def build(self) -> bytes:
        """ปิดไฟล์และคืน bytes ของ PDF ทั้งไฟล์"""
        if not self._page_refs:
            raise PdfError("ยังไม่มีหน้าใดเลย")

        pages_ref = self._next_ref
        self._next_ref += 1
        self._offsets.append(self._buffer.tell())
        kids = " ".join(f"{ref} 0 R" for ref in self._page_refs)
        self._write(
            (
                f"{pages_ref} 0 obj\n"
                f"<< /Type /Pages /Count {len(self._page_refs)}"
                f" /Kids [{kids}] >>\n"
                "endobj\n"
            ).encode("latin-1")
        )

        catalog_ref = self._next_ref
        self._next_ref += 1
        self._offsets.append(self._buffer.tell())
        self._write(
            (
                f"{catalog_ref} 0 obj\n"
                f"<< /Type /Catalog /Pages {pages_ref} 0 R >>\n"
                "endobj\n"
            ).encode("latin-1")
        )

        info_ref = self._next_ref
        self._next_ref += 1
        self._offsets.append(self._buffer.tell())
        title = _pdf_escape(self.title)
        self._write(
            (
                f"{info_ref} 0 obj\n"
                f"<< /Title ({title}) /Producer (coloring-book) >>\n"
                "endobj\n"
            ).encode("latin-1")
        )

        # แก้ /Parent ของทุกหน้าให้ชี้ไปที่ pages_ref
        self._buffer.seek(0)
        data = bytearray(self._buffer.getvalue())
        token = b"/Parent __PAGES__"
        index = data.find(token)
        while index != -1:
            data[index : index + len(token)] = f"/Parent {pages_ref} 0 R".encode(
                "latin-1"
            )
            index = data.find(token, index + 1)
        self._buffer = io.BytesIO(bytes(data))

        # xref
        start = self._buffer.tell()
        count = self._next_ref
        out = self._buffer.getvalue()
        xref = [f"xref\n0 {count}\n".encode("latin-1"), b"0000000000 65535 f \n"]
        for offset in self._offsets:
            xref.append(f"{offset:010d} 00000 n \n".encode("latin-1"))
        xref.append(
            (
                f"trailer\n<< /Size {count} /Root {catalog_ref} 0 R"
                f" /Info {info_ref} 0 R >>\n"
                f"startxref\n{start}\n%%EOF\n"
            ).encode("latin-1")
        )
        return out + b"".join(xref)


def _pdf_escape(text: str) -> str:
    """ escape อักขระที่ PDF ไม่ยอมรับในสตริง (เช่นวงเล็บวงเหลี่ยม)"""
    ascii_text = text.encode("latin-1", "replace").decode("latin-1")
    return ascii_text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(
    pages: Iterable[np.ndarray],
    *,
    title: str = "สมุดระบายสี",
) -> bytes:
    """รวมหน้าทั้งหมดเป็น PDF

    :param pages: iterable ของภาพหน้า (ใช้ generator เพื่อไม่ต้องเก็บทุกหน้า
        ในหน่วยความจำพร้อมกัน)
    :param title: ชื่อเอกสารที่ฝังใน metadata
    """
    writer = PdfWriter(title=title)
    for page in pages:
        writer.add_page(page)
    return writer.build()


def page_count(pdf_bytes: bytes) -> int:
    """นับจำนวนหน้าจาก PDF ที่สร้างเอง (ใช้ในเทสต์)"""
    marker = b"/Type /Pages /Count "
    index = pdf_bytes.find(marker)
    if index == -1:
        return 0
    start = index + len(marker)
    end = pdf_bytes.find(b" ", start)
    if end == -1:
        return 0
    return int(pdf_bytes[start:end])
