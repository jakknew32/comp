"""เว็บแอปสร้างสมุดระบายสี A4

รันด้วย:  uvicorn app.main:app --reload
แล้วเปิด http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime
from functools import lru_cache
from typing import Annotated

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import ai
from .ai import generate as generate_module
from .book import pdf as book_pdf
from .config import (
    GRID_OPTIONS,
    MAX_UPLOAD_BYTES,
    STATIC_DIR,
    BookParams,
    LineArtParams,
)
from .lineart import convert, imgio, text
from .webgrab import WebImageError, download_image, list_images, detect_image_format

logger = logging.getLogger("coloring_book")


def _log_font_status() -> None:
    """แจ้งเตือนตอนเริ่มเซิร์ฟเวอร์ว่าฟอนต์พร้อมใช้หรือไม่

    ถ้า libraqm ไม่ได้ติดตั้งมา ข้อความไทยจะซ้อนกันผิดตำแหน่งโดยไม่มี error
    ผู้ใช้จะเห็นเป็นภาพที่ผิดรูปโดยไม่ทราบสาเหตุ
    จึงต้องขึ้น log ระดับ error ให้เห็นเด่นชัดในหน้า Deploy ของ Render
    """
    try:
        text.check_setup()
        logger.info("ตรวจสอบฟอนต์ผ่าน พร้อมวาดข้อความภาษาไทยได้")
    except text.FontUnavailableError as exc:
        logger.error("ระบบวาดข้อความไทยไม่พร้อม: %s", exc)


_log_font_status()

app = FastAPI(title="โปรแกรมสร้างสมุดระบายสี A4", version="1.0.0")

# ผลลัพธ์ของแต่ละไฟล์ ใช้ซ้ำได้ตามชุดพารามิเตอร์ เพื่อไม่ต้องประมวลผลซ้ำ
# เวลาผู้ใช้ลาก slider ไปมา คีย์คือแฮชของ bytes ไฟล์คู่กับค่าที่เกี่ยวข้อง
_CACHE: dict[str, convert.ConvertResult] = {}
_CACHE_LIMIT = 64


# --- โมเดลคำขอ ---------------------------------------------------------------


class LineArtPayload(BaseModel):
    """ค่าที่ส่งมาจากหน้าเว็บ ฟิลด์ที่เป็น null แปลว่าให้โปรแกรมตัดสินใจเอง"""

    target_line_mm: float | None = Field(default=None, ge=0.2, le=8.0)
    speckle_ratio: float | None = Field(default=None, gt=0.0, le=0.05)
    denoise: int | None = Field(default=None, ge=0, le=15)
    xdog_sigma: float | None = Field(default=None, ge=0.3, le=8.0)
    xdog_tau: float | None = Field(default=None, ge=0.5, le=0.999)
    xdog_phi: float | None = Field(default=None, ge=1.0, le=60.0)
    close_iterations: int = Field(default=1, ge=0, le=4)
    strip_border: bool = True
    use_ai: bool = False
    ai_provider: str = "gemini"
    ai_model: str | None = None
    skip_convert: bool = False


class BookPayload(BaseModel):
    title: str = Field(default="สมุดระบายสีของฉัน", max_length=120)
    author: str = Field(default="", max_length=120)
    per_page: int = Field(default=1)
    show_frame: bool = True
    show_caption: bool = True
    show_page_number: bool = True
    include_cover: bool = True
    caption_size_pt: float = Field(default=20.0, ge=8.0, le=48.0)


class GeneratePayload(BaseModel):
    """คำขอสร้างภาพระบายสีจากข้อความ"""

    prompt: str = Field(min_length=1, max_length=600)
    style: str | None = None
    count: int = Field(default=1, ge=1, le=4)
    provider: str | None = None
    model: str | None = None


def to_lineart_params(payload: LineArtPayload | None) -> LineArtParams:
    if payload is None:
        return LineArtParams()
    return LineArtParams(
        target_line_mm=payload.target_line_mm,
        speckle_ratio=payload.speckle_ratio,
        denoise=payload.denoise,
        xdog_sigma=payload.xdog_sigma,
        xdog_tau=payload.xdog_tau,
        xdog_phi=payload.xdog_phi,
        close_iterations=payload.close_iterations,
        strip_border=payload.strip_border,
        use_ai=payload.use_ai,
        ai_provider=payload.ai_provider,
        ai_model=payload.ai_model,
        skip_convert=payload.skip_convert,
    )


def to_book_params(payload: BookPayload | None) -> BookParams:
    if payload is None:
        return BookParams()
    per_page = payload.per_page if payload.per_page in GRID_OPTIONS else 1
    return BookParams(
        title=payload.title.strip() or "สมุดระบายสีของฉัน",
        author=payload.author.strip(),
        per_page=per_page,
        show_frame=payload.show_frame,
        show_caption=payload.show_caption,
        show_page_number=payload.show_page_number,
        include_cover=payload.include_cover,
        caption_size_pt=payload.caption_size_pt,
    )


def parse_json(raw: str | None, model):
    """อ่านค่า JSON จากฟอร์ม ถ้าไม่ได้ส่งมาหรือผิดรูปแบบให้ใช้ค่าเริ่มต้น

    หน้าเว็บจะส่งค่าที่ยังไม่ได้แตะเป็น null เพื่อให้โปรแกรมตัดสินใจเอง
    จึงต้องยอมรับ null ได้ และถ้าฟิลด์ใดผิดชนิดก็ไม่ควรทำให้ทั้งคำขอล้ม
    """
    if not raw or not raw.strip():
        return None
    try:
        return model.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("อ่านค่าพารามิเตอร์ไม่สำเร็จ ใช้ค่าเริ่มต้นแทน: %s", exc)
        return None


# --- ตัวช่วยประมวลผล ---------------------------------------------------------


def _cache_key(raw: bytes, params: LineArtParams) -> str:
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    return f"{digest}:{sorted(params.to_dict().items())}"


def convert_uploaded(
    raw: bytes, filename: str, params: LineArtParams
) -> tuple[convert.ConvertResult, np.ndarray, str, str]:
    """แปลงไฟล์ที่อัปโหลด ใช้แคชถ้าเคยประมวลผลด้วยค่าเดิม

    คืน (ผลลัพธ์, ภาพต้นฉบับสี, ชื่อกำกับเริ่มต้น, ข้อความเตือนรวม)
    """
    key = _cache_key(raw, params)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached, None, imgio.caption_from_filename(filename), ""

    image = imgio.load_upload(raw, filename)
    notes: list[str] = []

    if params.use_ai and not params.skip_convert:
        outcome = ai.enhance(
            image, _ai_settings(), params.use_ai,
            provider=params.ai_provider,
            model=params.ai_model,
        )
        if outcome.note:
            notes.append(outcome.note)
        image = outcome.image

    if params.skip_convert:
        result = convert.passthrough(image)
    else:
        result = convert.convert(image, params)
    notes.extend(result.warnings)

    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result

    return result, image, imgio.caption_from_filename(filename), " | ".join(notes)


@lru_cache(maxsize=1)
def _ai_settings() -> ai.AiSettings:
    """อ่านค่าตั้งค่�� AI เก็บไว้เพื่อไม่ต้องอ่านซ้ำทุกรูป"""
    return ai.load_settings()


async def read_upload(upload: UploadFile) -> bytes:
    name = upload.filename or ""
    if name and not imgio.is_supported(name):
        raise HTTPException(
            status_code=400,
            detail=f"ไฟล์ {name} ไม่ใช่ภาพที่รองรับ (รองรับ PNG, JPG, WEBP, BMP, TIFF)",
        )

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"ไฟล์ {name or 'ไม่ระบุชื่อ'} ว่างเปล่า")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"ไฟล์ {name} ใหญ่เกินไป "
                f"(สูงสุด {MAX_UPLOAD_BYTES // (1024 * 1024)} MB ต่อไฟล์)"
            ),
        )
    return raw


# --- เส้นทาง -----------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    try:
        text.check_setup()
        fonts_ok = True
        font_error = None
    except text.FontUnavailableError as exc:
        fonts_ok = False
        font_error = str(exc)

    return {
        "status": "ok" if fonts_ok else "degraded",
        "version": app.version,
        "fonts_ready": fonts_ok,
        "font_error": font_error,
        "grids": sorted(GRID_OPTIONS),
        "ai": ai.status(_ai_settings()),
    }


@app.post("/api/validate")
async def validate(
    lineart: Annotated[str | None, Form()] = None,
    book: Annotated[str | None, Form()] = None,
) -> dict:
    """คืนค่าที่โปรแกรมตัดสินใจเอง พร้อมค่าที่ผู้ใช้ส่งมา

    ใช้ตอนหน้าเว็บต้องการรู้ว่าค่าที่ว่างไว้จะถูกตัดสินเป็นอะไร
    """
    lineart_payload = parse_json(lineart, LineArtPayload)
    book_payload = parse_json(book, BookPayload)
    lp = to_lineart_params(lineart_payload)
    bp = to_book_params(book_payload)
    return {
        "lineart": lp.to_dict(),
        "book": bp.to_dict(),
        "grid": bp.grid(),
        "cells_per_page": bp.per_page,
    }


@app.post("/api/preview")
async def preview(
    file: Annotated[UploadFile, File()],
    lineart: Annotated[str | None, Form()] = None,
    book: Annotated[str | None, Form()] = None,
) -> Response:
    """คืนภาพหน้า A4 หนึ่งหน้า ใช้ดูตัวอย่างตอนปรับค่า"""
    lp = to_lineart_params(parse_json(lineart, LineArtPayload))
    bp = to_book_params(parse_json(book, BookPayload))
    bp.per_page = 1
    bp.include_cover = False

    raw = await read_upload(file)
    try:
        result, _, caption, _ = convert_uploaded(raw, file.filename or "ภาพ", lp)
    except imgio.ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result.is_empty:
        raise HTTPException(
            status_code=422,
            detail="ไม่พบเส้นในภาพนี้เลย ลองใช้ภาพที่เป็นภาพลายเส้นหรือเพิ่มความละเอียด",
        )

    page, notes = book_pdf.render_preview(result.mask, caption, bp, lp)
    return Response(
        content=_png_bytes(page),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Notice": _encode_header(notes),
        },
    )


@app.post("/api/book")
async def make_book(
    files: Annotated[list[UploadFile], File()],
    lineart: Annotated[str | None, Form()] = None,
    book: Annotated[str | None, Form()] = None,
    captions: Annotated[str | None, Form()] = None,
) -> Response:
    """ประกอบไฟล์ทั้งหมดเป็น PDF หนึ่งเล่ม"""
    if not files:
        raise HTTPException(status_code=400, detail="ยังไม่ได้เลือกภาพ")

    lp = to_lineart_params(parse_json(lineart, LineArtPayload))
    bp = to_book_params(parse_json(book, BookPayload))

    try:
        caption_list = json.loads(captions) if captions else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="รายการชื่อกำกับไม่ใช่ JSON ที่ถูกต้อง") from None
    if not isinstance(caption_list, list):
        raise HTTPException(status_code=400, detail="รายการชื่อกำกับต้องเป็น JSON array")

    masks: list[np.ndarray] = []
    captions: list[str] = []
    warnings: list[str] = []
    skipped: list[str] = []

    for index, upload in enumerate(files):
        raw = await read_upload(upload)
        name = upload.filename or f"ภาพที่ {index + 1}"
        try:
            result, original_image, default_caption, warn = convert_uploaded(raw, name, lp)
        except imgio.ImageLoadError as exc:
            skipped.append(f"{name}: {exc}")
            continue

        if result.is_empty:
            skipped.append(f"{name}: ไม่พบเส้นในภาพ")
            continue

        caption = default_caption
        if index < len(caption_list) and isinstance(caption_list[index], str):
            caption = caption_list[index].strip() or default_caption

        masks.append(result.mask)
        captions.append(caption)
        if warn:
            warnings.append(f"{name}: {warn}")

    if not masks:
        detail = "ไม่มีภาพใดที่ใช้ได้"
        if skipped:
            detail += " — " + "; ".join(skipped)
        raise HTTPException(status_code=422, detail=detail)

    result = book_pdf.build_book(masks, captions, bp, lp)
    # ชื่อไฟล์ต้องเป็น ASCII ตามโครงสร้างไฟล์ ชื่อสมุดภาษาไทยจถูกตัดทิ้งทั้งหมด
    # จึงใส่วันที่ไว้แทน เพื่อไม่ให้ไฟล์ที่ดาวน์โหลดหลายครั้งชื่อซ้ำกัน
    fallback = "coloring-book-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = imgio.slugify(f"{bp.title}-coloring-book", fallback=fallback)

    all_warnings = list(dict.fromkeys(warnings + result.warnings))

    return Response(
        content=result.pdf_bytes,
        media_type="application/pdf",
        headers={
            # ชื่อไฟล์ต้องมาจากฝั่งเซิร์ฟเวอร์เท่านั้น ห้ามใช้ชื่อที่ผู้ใช้ส่งมา
            "Content-Disposition": f'attachment; filename="{filename}.pdf"',
            "X-Page-Count": str(result.page_count),
            "X-Warnings": _encode_header(all_warnings),
        },
    )


def _data_url_png(image: Image.Image, width_px: int, invert: bool = False) -> str:
    """ย่อภาพเป็นขาวดำ (เฉดเทา) แล้วคืนเป็น data URL ของ PNG ใช้แสดงและพิมพ์ในหน้าเว็บ"""
    gray = image.convert("L")
    if invert:
        gray = Image.eval(gray, lambda v: 255 - v)
    if gray.width > width_px:
        height_px = round(width_px * gray.height / gray.width)
        gray = gray.resize((width_px, height_px), Image.LANCZOS)
    buffer = io.BytesIO()
    gray.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ขนาดภาพหน้าที่ส่งให้เบราว์เซอร์ ~200 DPI พอสำหรับพรีวิวและสั่งพิมพ์ ไฟล์ต่อหน้าเล็กกว่า PDF มาก
PAGE_PREVIEW_WIDTH_PX = 1654
LINEART_THUMB_WIDTH_PX = 360


@app.post("/api/convert")
async def convert_one(
    file: Annotated[UploadFile, File()],
    lineart: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """ขั้นที่ 1: แปลงภาพเดียวเป็นลายเส้น แล้วคืนภาพย่อ

    ผลถูกเก็บในแคชของเซิร์ฟเวอร์ ตอนรวมเล่มภายหลังจึงไม่ต้องแปลงซ้ำ
    และเพราะแปลงทีละรูป คำขอแต่ละครั้งสั้นพอไม่ถูกตัดด้วย timeout ของโฮสต์
    """
    lp = to_lineart_params(parse_json(lineart, LineArtPayload))
    raw = await read_upload(file)
    try:
        result, _, caption, notice = convert_uploaded(raw, file.filename or "ภาพ", lp)
    except imgio.ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result.is_empty:
        raise HTTPException(status_code=422, detail="ไม่พบเส้นในภาพนี้เลย")

    thumb = _data_url_png(
        Image.fromarray(result.mask.astype(np.uint8)),
        LINEART_THUMB_WIDTH_PX,
        invert=True,
    )
    return JSONResponse({"caption": caption, "notice": notice, "thumb": thumb})


@app.post("/api/pages")
async def make_pages(
    files: Annotated[list[UploadFile], File()],
    lineart: Annotated[str | None, Form()] = None,
    book: Annotated[str | None, Form()] = None,
    captions: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """ขั้นที่ 2: รวมภาพเป็นสมุด แล้วคืนภาพของ "ทุกหน้า" สำหรับพรีวิวและสั่งพิมพ์"""
    if not files:
        raise HTTPException(status_code=400, detail="ยังไม่ได้เลือกภาพ")

    lp = to_lineart_params(parse_json(lineart, LineArtPayload))
    bp = to_book_params(parse_json(book, BookPayload))

    try:
        caption_list = json.loads(captions) if captions else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="รายการชื่อกำกับไม่ใช่ JSON ที่ถูกต้อง") from None
    if not isinstance(caption_list, list):
        raise HTTPException(status_code=400, detail="รายการชื่อกำกับต้องเป็น JSON array")

    masks: list[np.ndarray] = []
    names: list[str] = []
    warnings: list[str] = []
    skipped: list[str] = []

    for index, upload in enumerate(files):
        raw = await read_upload(upload)
        name = upload.filename or f"ภาพที่ {index + 1}"
        try:
            result, _, default_caption, warn = convert_uploaded(raw, name, lp)
        except imgio.ImageLoadError as exc:
            skipped.append(f"{name}: {exc}")
            continue
        if result.is_empty:
            skipped.append(f"{name}: ไม่พบเส้นในภาพ")
            continue

        caption = default_caption
        if index < len(caption_list) and isinstance(caption_list[index], str):
            caption = caption_list[index].strip() or default_caption

        masks.append(result.mask)
        names.append(caption)
        if warn:
            warnings.append(f"{name}: {warn}")

    if not masks:
        detail = "ไม่มีภาพใดที่ใช้ได้"
        if skipped:
            detail += " — " + "; ".join(skipped)
        raise HTTPException(status_code=422, detail=detail)

    page_warnings: list[str] = []
    pages: list[str] = []
    # แปลงทีละหน้าแล้วทิ้งภาพเต็ม ไม่ถือทุกหน้า 300 DPI ในหน่วยความจำพร้อมกัน
    for page_image in book_pdf.iter_pages(masks, names, bp, lp, page_warnings):
        pages.append(_data_url_png(page_image, PAGE_PREVIEW_WIDTH_PX))

    return JSONResponse(
        {
            "count": len(pages),
            "pages": pages,
            "warnings": list(dict.fromkeys(warnings + page_warnings)),
            "skipped": skipped,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/web-images")
async def web_images(url: Annotated[str, Form()]) -> JSONResponse:
    """คืนรายการลิงก์รูปที่พบในหน้าเว็บที่ผู้ใช้ใส่มา

    คืนแค่ลิงก์ ไม่ดาวน์โหลดรูปมาทั้งหมด เพราะหน้าเว็บหนึ่งหน้าอาจมีรูปนับร้อย
    การโหลดมาทั้งหมดจะทำให้เซิร์ฟเวอร์ค้าง ผู้ใช้จึงต้องเลือกก่อนดาวน์โหลด
    """
    try:
        images = list_images(url)
    except WebImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse({"images": images, "count": len(images)})


@app.post("/api/web-image")
async def web_image(url: Annotated[str, Form()]) -> Response:
    """ดาวน์โหลดรูปหนึ่งไฟล์จากลิงก์ที่เลือก แล้วส่งกลับเป็นไฟล์ภาพ

    ต้องให้เซิร์ฟเวอร์เป็นคนโหลด เพราะเบราว์เซอร์ถูกกฎ CORS ของเว็บเป้าหมายกัน
    อ่านไฟล์ข้ามโดเมนไม่ได้ ฝั่งหน้าเว็บจึงเอาไฟล์ที่ได้ไปเข้าสู่ขั้นตอนเดียวกับไฟล์ที่อัปโหลด
    """
    try:
        raw, name = download_image(url)
    except WebImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ตรวจจับรูปแบบและส่ง MIME type ที่ถูกต้อง
    mime_type, ext = detect_image_format(raw)
    filename_ascii = imgio.slugify(name, "web-image") + ext

    return Response(
        content=raw,
        media_type=mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename_ascii}"',
            "X-Image-Name": _encode_header([name]),
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/generate")
async def generate_images(payload: GeneratePayload) -> JSONResponse:
    """สร้างภาพระบายสีใหม่จากข้อความด้วย AI

    คืนภาพเป็น data URL ให้หน้าเว็บนำไปเพิ่มเข้าสมุดได้ทันที
    สร้างทีละภาพ ถ้าภาพใดล้มเหลวจะข้ามและรายงานในช่อง failed
    ไม่ทำให้ภาพอื่นที่สำเร็จแล้วหายไป
    """
    settings = _ai_settings()
    if not settings.configured:
        raise HTTPException(status_code=400, detail=settings.describe_missing())

    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="กรุณาพิมพ์คำบรรยายภาพที่ต้องการ")

    images: list[dict] = []
    failed: list[str] = []
    for _ in range(payload.count):
        outcome = ai.generate_image(
            prompt,
            payload.style,
            settings,
            provider=payload.provider,
            model=payload.model,
        )
        if outcome.ok:
            images.append(
                {"data": generate_module.to_png_data_url(outcome.image), "name": prompt}
            )
        elif outcome.note:
            failed.append(outcome.note)

    if not images:
        detail = failed[0] if failed else "สร้างภาพไม่สำเร็จ"
        raise HTTPException(status_code=502, detail=detail)

    return JSONResponse(
        {"images": images, "failed": failed},
        headers={"Cache-Control": "no-store"},
    )


def _encode_header(values: list[str]) -> str:
    """เข้ารหัสข้อความไทยให้พอดีกับข้อจำกัดของ HTTP header

    header ต้องเป็น ASCII เท่านั้น จึงต้องแทนอักขระไทยด้วย escape sequence
    ฝั่งหน้าเว็บจะถอดกลับเป็นข้อความภาษาไทยให้ผู้ใช้อ่านได้
    """
    text = "; ".join(values)
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _png_bytes(page) -> bytes:
    import io

    buffer = io.BytesIO()
    page.convert("L").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


@app.exception_handler(imgio.ImageLoadError)
def handle_image_error(_request, exc: imgio.ImageLoadError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="ไม่พบไฟล์หน้าเว็บ")
    return FileResponse(page)
