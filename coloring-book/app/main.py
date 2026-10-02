"""FastAPI application — เส้นทาง API ของโปรแกรมสร้างสมุดระบายสี

หลักการสำคัญ
------------
* พารามิเตอร์ที่คำนวณอัตโนมัติได้ รับค่า ``null`` ได้ เช่น
  ``{"target_line_mm": null}`` แปลว่า "ให้เซิร์ฟเวอร์ตัดสินใจเอง"
* คำเตือนเรื่องภาษาไทยและข้อจำกัดเรื่องภาพถ่ายต้องส่งไปให้ UI แสดง
  เสมอ ไม่ปล่อยให้ผู้ใช้เห็นผลลัพธ์แล้วรู้สึกว่าโปรแกรมทำงานผิด
* ไฟล์ที่อัปโหลดเก็บในหน่วยความจำเท่านั้น ไม่เขียนลงดิสก์
* ผลลัพธ์ preview ถูก cache ตาม (bytes ของภาพ + พารามิเตอร์) เพราะ
  slider จะยิง request ถี่มาก
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import config
from .book import cover as cover_mod
from .book import layout as layout_mod
from .book import pdf as pdf_mod
from .lineart import convert as convert_mod
from .lineart import detect as detect_mod
from .lineart import imgio
from .lineart import text as text_mod
from .lineart.compose import Cell, compose_page

app = FastAPI(
    title="โปรแกรมสร้างสมุดระบายสี A4",
    version=config.APP_VERSION,
)


# --------------------------------------------------------------------------
# โมเดลคำขอ
# --------------------------------------------------------------------------


class ConvertOptions(BaseModel):
    """พารามิเตอร์การแปลงภาพ — ค่า ``null`` หมายถึงให้เซิร์ฟเวอร์ตัดสินใจเอง"""

    target_line_mm: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    despeckle: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dog_sigma: Optional[float] = Field(default=None, gt=0.0, le=20.0)
    close_rounds: Optional[int] = Field(default=None, ge=0, le=5)
    close_holes: Optional[bool] = None
    upscale: Optional[bool] = None


class LayoutOptions(BaseModel):
    """พารามิเตอร์การจัดหน้า"""

    per_page: int = config.DEFAULT_PER_PAGE
    frame: bool = True
    cover: bool = True
    book_title: str = config.DEFAULT_BOOK_TITLE
    page_numbers: bool = True
    captions: bool = True


class ValidateRequest(BaseModel):
    """คำขอของ ``/api/validate`` — ทุกช่องเป็น optional เพื่อให้ส่งเฉพาะ
    สิ่งที่อยากให้เซิร์ฟเวอร์ช่วยตัดสินใจก็ได้"""

    convert: Optional[ConvertOptions] = None
    layout: Optional[LayoutOptions] = None


def _convert_params(options: Optional[ConvertOptions]) -> convert_mod.ConvertParams:
    if options is None:
        return convert_mod.ConvertParams()
    return convert_mod.ConvertParams(**options.model_dump())


def _layout_params(options: Optional[LayoutOptions]) -> config.LayoutParams:
    if options is None:
        return config.LayoutParams()
    data = options.model_dump()
    # ตรวจโหมดภาพต่อหน้าที่นี่ เพื่อให้ error ชัดเจนก่อนประมวลผล
    data["per_page"] = layout_mod.validate_per_page(int(data["per_page"]))
    return config.LayoutParams(**data)


def _parse_json_field(raw: str, field_name: str) -> Any:
    """ถอดค่า JSON จากฟอร์ม พร้อมข้อความ error ที่บอกชื่อฟิลด์"""
    try:
        return json.loads(raw or "null")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"ฟิลด์ '{field_name}' ไม่ใช่ JSON ที่ถูกต้อง: {exc}"
        )


# --------------------------------------------------------------------------
# cache ของ preview
# --------------------------------------------------------------------------


class PreviewCache:
    """cache ผลลัพธ์ preview ตาม (ภาพ + พารามิเตอร์)

    slider จะยิง request ถี่มาก การประมวลผล XDoG + morphology ซ้ำทุกครั้ง
    จะกิน CPU มาก จึงต้องมี cache
    """

    def __init__(self, max_entries: int = 48) -> None:
        self.max_entries = max_entries
        self._store: dict[str, bytes] = {}
        self._order: list[str] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(image_bytes: bytes, payload: dict[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(image_bytes)
        digest.update(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        return digest.hexdigest()

    def get(self, key: str) -> Optional[bytes]:
        data = self._store.get(key)
        if data is None:
            self.misses += 1
        else:
            self.hits += 1
            # เลื่อนไปท้ายสุด (ใช้ LRU)
            self._order.remove(key)
            self._order.append(key)
        return data

    def put(self, key: str, data: bytes) -> None:
        if key in self._store:
            self._order.remove(key)
        self._store[key] = data
        self._order.append(key)
        while len(self._order) > self.max_entries:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
        }


preview_cache = PreviewCache()


# --------------------------------------------------------------------------
# ตัวช่วย
# --------------------------------------------------------------------------


def _read_upload(upload: UploadFile) -> tuple[bytes, str]:
    """อ่านไฟล์ที่อัปโหลด พร้อมถอดชื่อไฟล์ภาษาไทย

    :returns: ``(bytes ของไฟล์, ชื่อไฟล์ที่ถอดแล้ว)``
    """
    raw_name = upload.filename or "image"
    name = imgio.decode_upload_filename(raw_name)

    if upload.content_type and upload.content_type not in config.ALLOWED_CONTENT_TYPES:
        # บางเบราว์เซอร์ส่ง content_type เป็น octet-stream สำหรับไฟล์บางชนิด
        # จึงไม่ปฏิเสธทันที แต่ยังตรวจ format จริงตอนถอดภาพ
        pass

    data = upload.file.read(config.MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="ไฟล์ว่างเปล่า")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"ไฟล์ใหญ่เกินไป (สูงสุด {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB ต่อไฟล์)"
            ),
        )
    return data, name


def _cell_box_width(cols: int) -> int:
    """ความกว้างของช่องภาพบนหน้า A4 (ใช้คิดสเกลเป้าหมายความหนาเส้น)"""
    from .lineart.compose import compute_geometry

    geometry = compute_geometry(cols, 1)
    if not geometry:  # pragma: no cover - กันพลาด
        return config.A4_WIDTH_PX
    inner_pad = geometry[0].frame_width + 2
    return max(1, geometry[0].w - inner_pad * 2)


def _encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:  # pragma: no cover - กันพลาด
        raise HTTPException(status_code=500, detail="สร้างภาพตัวอย่างไม่สำเร็จ")
    return buffer.tobytes()


def _safe_book_filename(title: str) -> str:
    """สร้างชื่อไฟล์ PDF ที่ปลอดภัย

    ห้ามใช้ชื่อไฟล์จากผู้ใช้ตรง ๆ เพราะเสี่ยง XSS และ path traversal
    (ความเสี่ยงข้อ 7)
    """
    slug = imgio.slugify(title or "coloring-book", max_len=40)
    return f"{slug}.pdf"


# --------------------------------------------------------------------------
# เส้นทาง
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    """สถานะของเซิร์ฟเวอร์และความพร้อมด้านการจัดวางสระภาษาไทย"""
    return {
        "status": "ok",
        "version": config.APP_VERSION,
        "raqm": text_mod.raqm_available(),
        "shaping_warning": text_mod.shaping_warning(),
        "page_px": [config.A4_WIDTH_PX, config.A4_HEIGHT_PX],
        "dpi": config.DPI,
        "preview_cache": preview_cache.stats(),
    }


@app.post("/api/validate")
async def validate(request: Optional[ValidateRequest] = None) -> dict[str, Any]:
    """คืนค่าที่เซิร์ฟเวอร์คำนวณจริง กลับไปแสดงใน UI

    UI ใช้จุดนี้เพื่อบอกผู้ใช้ว่า "ตอนนี้โปรแกรมใช้ค่าอะไรอยู่"
    """
    request = request or ValidateRequest()
    convert = _convert_params(request.convert)
    layout = _layout_params(request.layout)
    cols, rows = layout_mod.grid_for(layout.per_page)

    return {
        "convert": convert.as_dict(),
        "resolved": {
            "target_line_mm": (
                convert.target_line_mm
                if convert.target_line_mm is not None
                else config.TARGET_LINE_MM
            ),
            "despeckle": (
                convert.despeckle if convert.despeckle is not None else 0.0
            ),
            "auto": convert.target_line_mm is None,
        },
        "layout": asdict(layout),
        "grid": {"cols": cols, "rows": rows, "cells": cols * rows},
        "page_px": [config.A4_WIDTH_PX, config.A4_HEIGHT_PX],
        "page_mm": [config.A4_WIDTH_MM, config.A4_HEIGHT_MM],
        "dpi": config.DPI,
        "defaults": {
            "target_line_mm": config.TARGET_LINE_MM,
            "target_line_mm_range": list(config.TARGET_LINE_MM_RANGE),
            "despeckle_range": list(config.DESPECKLE_RANGE),
            "per_page": config.DEFAULT_PER_PAGE,
            "per_page_modes": sorted(config.PER_PAGE_GRID),
            "book_title": config.DEFAULT_BOOK_TITLE,
        },
        "image_output_width_px": _cell_box_width(cols),
    }


@app.post("/api/preview")
async def preview(
    file: UploadFile = File(...),
    options: str = Form("{}"),
    per_page: int = Form(config.DEFAULT_PER_PAGE),
    index: int = Form(0),
) -> Response:
    """คืนภาพหน้า A4 หน้าเดียวสำหรับดูผลแบบทันที

    ใช้ตอนลาก slider — เร็วกว่า ``/api/book`` เพราะไม่ต้องห่อ PDF
    ผลลัพธ์ถูก cache ตาม (bytes ของภาพ + พารามิเตอร์) เพราะ slider
    ยิง request ถี่มาก
    """
    data, _name = _read_upload(file)
    convert_options = _convert_options_from(options)

    try:
        per_page = layout_mod.validate_per_page(per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload = convert_options.model_dump()
    payload["per_page"] = per_page
    key = preview_cache.make_key(data, payload)

    cached = preview_cache.get(key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/png",
            headers={"X-Cache": "HIT", "X-File-Index": str(index)},
        )

    try:
        bgr, _ = imgio.load_bgr(data)
    except imgio.ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cols, _rows = layout_mod.grid_for(per_page)
    box_w = _cell_box_width(cols)

    try:
        result = convert_mod.convert_to_lineart(
            bgr, _convert_params(convert_options), output_width_px=box_w
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    page = compose_page(
        [Cell(mask=result.mask, caption="")],
        cols=cols,
        rows=1,
        page_number=1,
        frame=True,
        captions=False,
    )
    png = _encode_png(page)
    preview_cache.put(key, png)

    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Cache": "MISS", "X-File-Index": str(index)},
    )


def _convert_options_from(options_json: str) -> ConvertOptions:
    """แปลง JSON ของพารามิเตอร์การแปลงภาพเป็นโมเดล

    ค่าที่ไม่ระบุจะเป็น ``None`` = ให้เซิร์ฟเวอร์ตัดสินใจเอง
    """
    raw = _parse_json_field(options_json, "options")
    if raw is None:
        return ConvertOptions()
    if not isinstance(raw, dict):
        raise HTTPException(
            status_code=400, detail="ฟิลด์ 'options' ต้องเป็น JSON object"
        )
    try:
        return ConvertOptions.model_validate(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"พารามิเตอร์ไม่ถูกต้อง: {exc}")


@app.post("/api/book")
async def book(
    files: list[UploadFile] = File(...),
    options: str = Form("{}"),
    captions: str = Form("[]"),
    per_page: int = Form(config.DEFAULT_PER_PAGE),
    frame: bool = Form(True),
    cover: bool = Form(True),
    book_title: str = Form(config.DEFAULT_BOOK_TITLE),
    author: str = Form(""),
    page_numbers: bool = Form(True),
    show_captions: bool = Form(True),
    dry_run: int = Form(0),
) -> Response:
    """สร้าง PDF ทั้งเล่มจากหลายไฟล์

    :param captions: JSON array ของชื่อกำกับ เรียงตามลำดับไฟล์
    :param dry_run: ถ้าเป็น 1 จะไม่ส่ง PDF กลับ แต่คืนสถานะการตรวจ
        ประเภทภาพและค่าที่ระบบคำนวณจริงเป็น JSON (ใช้เรียกหนึ่งครั้งเพื่อ
        แสดงสถานะรายภาพก่อนตัดสินใจดาวน์โหลด)
    """
    convert_options = _convert_options_from(options)

    try:
        per_page = layout_mod.validate_per_page(per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    caption_list = _parse_json_field(captions, "captions") or []
    if not isinstance(caption_list, list):
        raise HTTPException(status_code=400, detail="captions ต้องเป็น JSON array")

    if not files:
        raise HTTPException(status_code=400, detail="ไม่มีไฟล์ให้ประมวลผล")

    params = _convert_params(convert_options)
    title = (book_title or "").strip() or config.DEFAULT_BOOK_TITLE
    cols, rows = layout_mod.grid_for(per_page)
    box_w = _cell_box_width(cols)

    cells: list[Cell] = []
    reports: list[dict[str, Any]] = []

    for position, upload in enumerate(files):
        data, name = _read_upload(upload)
        try:
            bgr, _unused = imgio.load_bgr(data)
        except imgio.ImageLoadError as exc:
            raise HTTPException(status_code=400, detail=f"ไฟล์ '{name}': {exc}")

        try:
            result = convert_mod.convert_to_lineart(
                bgr, params, output_width_px=box_w
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"ไฟล์ '{name}': {exc}")

        caption = ""
        if position < len(caption_list):
            caption = str(caption_list[position] or "").strip()
        if not caption:
            caption = imgio.default_caption(name)

        cells.append(Cell(mask=result.mask, caption=caption))
        reports.append(
            {
                "index": position,
                "filename": name,
                "caption": caption,
                "detection": result.detection.as_dict(),
                "stroke": result.stroke.as_dict(),
                "auto_values": result.auto_values,
                "warnings": result.warnings,
            }
        )

    total_pages = layout_mod.count_pages(len(cells), per_page)

    if dry_run:
        return JSONResponse(
            {
                "page_count": total_pages,
                "has_cover": bool(cover),
                "total_pages_including_cover": total_pages + (1 if cover else 0),
                "grid": {"cols": cols, "rows": rows},
                "image_output_width_px": box_w,
                "raqm": text_mod.raqm_available(),
                "shaping_warning": text_mod.shaping_warning(),
                "scope_notice": SCOPE_NOTICE,
                "files": reports,
            }
        )

    # หน้าทั้งหมดทำเป็น generator เพื่อไม่เก็บทุกหน้าไว้ใน RAM
    # (ความเสี่ยงข้อ 4 — หน้า A4 300 DPI ดิบประมาณ 1.09 MB ต่อหน้า)
    def page_stream():
        if cover:
            yield cover_mod.make_cover(
                cells,
                book_title=title,
                page_count=total_pages,
                author=(author or "").strip(),
            )
        for number, group in enumerate(layout_mod.chunk(cells, per_page), start=1):
            yield compose_page(
                group,
                cols=cols,
                rows=rows,
                page_number=number if page_numbers else None,
                frame=frame,
                captions=show_captions,
            )

    try:
        data = pdf_mod.build_pdf(page_stream(), title=title)
    except pdf_mod.PdfError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            # สร้างชื่อไฟล์เองจาก slug เพื่อไม่รับชื่อผู้ใช้มาใส่ตรง ๆ
            # (กัน XSS และ path traversal — ความเสี่ยงข้อ 7)
            "Content-Disposition": f'attachment; filename="{_safe_book_filename(title)}"',
            "X-Page-Count": str(total_pages + (1 if cover else 0)),
            "X-Raqm": "1" if text_mod.raqm_available() else "0",
        },
    )


# --------------------------------------------------------------------------
# หน้าเว็บ
# --------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    """หน้าเว็บหลัก"""
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/static/{path:path}")
def static_files(path: str) -> FileResponse:
    """ไฟล์ static (จำกัดไว้ใต้โฟลเดอร์ static เท่านั้น)"""
    target = (config.STATIC_DIR / path).resolve()
    try:
        target.relative_to(config.STATIC_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="ไม่อนุญาต")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์")
    return FileResponse(target)


#: ข้อความเตือนที่ต้องแสดงผู้ใช้เสมอ — ไม่ใช่ AI ไม่ทำการ์ตูนให้
SCOPE_NOTICE = (
    "โปรแกรมนี้ทำงานกับภาพลายเส้นเท่านั้น — ภาพถ่ายจริงจะได้เส้นตามภาพ "
    "ไม่ใช่การ์ตูนน่ารักสไตล์ Canva"
)


@app.get("/api/scope")
def scope() -> dict[str, str]:
    """ข้อความขอบเขตการทำงาน ให้ UI ดึงไปแสดงเป็นแถบเตือน"""
    return {
        "notice": SCOPE_NOTICE,
        "shaping_warning": text_mod.shaping_warning() or "",
        "font_available": config.FONT_PATH.exists(),
    }
