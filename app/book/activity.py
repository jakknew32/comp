"""หน้าหนังสือกิจกรรม: เขียนตามรอยประ, แบบฝึกลากเส้น และจับคู่

ทุกหน้าเป็น A4 300 DPI เหมือนหน้าสมุดระบายสี ใช้ฟอนต์ ข้อความ และกรอบชุดเดียวกัน
จึงใช้ขั้นพรีวิวทุกหน้าและสั่งพิมพ์ร่วมกับสมุดระบายสีได้เลย
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import (
    AUTO_TARGET_LINE_MM,
    FRAME_RADIUS_MM,
    FRAME_STROKE_MM,
    mm_to_px,
    pt_to_px,
)
from ..lineart import compose, text
from ..lineart.compose import Box

BLACK = 0
WHITE = 255

# ขนาดเส้นประของตัวอักษรที่ให้เขียนตาม (มม.)
DASH_MM = 1.3
GAP_MM = 1.3
DASH_STROKE_MM = 0.5

# รูปแบบการลากเส้นที่เลือกได้ ตามลำดับที่แสดงในหน้า
PREWRITING_PATTERNS: dict[str, str] = {
    "vertical": "เส้นตรงตั้ง",
    "horizontal": "เส้นตรงนอน",
    "slant": "เส้นเฉียง",
    "wave": "เส้นคลื่น",
    "zigzag": "เส้นซิกแซก",
    "circle": "วงกลม",
    "loop": "เส้นวน",
}

MATCH_MODES = ("image_word", "image_shadow", "text_pairs")

MIN_PAIRS_PER_PAGE = 3
MAX_PAIRS_PER_PAGE = 6


@dataclass
class MatchItem:
    """หนึ่งคู่ที่ต้องจับ: ซ้ายเป็นภาพหรือข้อความ ขวาเป็นข้อความหรือเงา"""

    caption: str
    mask: np.ndarray | None = None  # ภาพลายเส้น (ใช้ในโหมดภาพ)
    right_text: str | None = None  # ข้อความด้านขวา (โหมดคำจับคู่)


@dataclass
class ActivityParams:
    title: str = "สมุดกิจกรรม"
    show_name_line: bool = True
    show_frame: bool = True
    show_page_number: bool = True

    prewriting: list[str] = field(default_factory=list)
    trace_items: list[str] = field(default_factory=list)
    trace_size_mm: float = 20.0
    trace_blank_rows: int = 1
    trace_guides: bool = True

    match_mode: str = "image_word"
    pairs_per_page: int = 4
    answer_key: bool = False


# ---------------------------------------------------------------------------
# เครื่องมือวาดพื้นฐาน
# ---------------------------------------------------------------------------


def _mm(value: float) -> int:
    return mm_to_px(value)


def _frame_inner(params: ActivityParams) -> Box:
    inner = compose.frame_box()
    if params.show_frame:
        inner = inner.inset(_mm(FRAME_STROKE_MM) // 2 + _mm(2))
    return inner


def _new_page(params: ActivityParams) -> Image.Image:
    page = compose.new_page()
    if params.show_frame:
        compose.draw_rounded_frame(
            page,
            compose.frame_box(),
            _mm(FRAME_STROKE_MM),
            _mm(FRAME_RADIUS_MM),
        )
    return page


def _draw_page_number(page: Image.Image, number: int, params: ActivityParams) -> None:
    if not params.show_page_number or number <= 0:
        return
    inset = _mm(FRAME_STROKE_MM) + _mm(2)
    area = compose.frame_box().inset(inset)
    label = text.render_text(f"หน้า {number}", pt_to_px(12), bold=False, max_width=max(1, area.w))
    page.paste(BLACK, (area.right - label.width, area.bottom - label.height), label)


def _paste_mask(page: Image.Image, mask: Image.Image, x: int, y: int) -> None:
    page.paste(BLACK, (int(x), int(y)), mask)


def _line(page: Image.Image, p1: tuple[int, int], p2: tuple[int, int], width_mm: float) -> None:
    ImageDraw.Draw(page).line([p1, p2], fill=BLACK, width=max(1, _mm(width_mm)))


def _dashed_line(
    page: Image.Image,
    p1: tuple[float, float],
    p2: tuple[float, float],
    width_mm: float,
    dash_mm: float = DASH_MM,
    gap_mm: float = GAP_MM,
) -> None:
    draw = ImageDraw.Draw(page)
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if length <= 0:
        return
    dash, gap = _mm(dash_mm), _mm(gap_mm)
    ux, uy = (p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length
    pos = 0.0
    width = max(1, _mm(width_mm))
    while pos < length:
        end = min(length, pos + dash)
        draw.line(
            [(p1[0] + ux * pos, p1[1] + uy * pos), (p1[0] + ux * end, p1[1] + uy * end)],
            fill=BLACK,
            width=width,
        )
        pos += dash + gap


def _dotted_polyline(page: Image.Image, points: np.ndarray, width_mm: float = DASH_STROKE_MM) -> None:
    """วาดเส้นประตามจุดที่ให้ โดยนับระยะตามความยาวเส้นจริง"""
    if len(points) < 2:
        return
    dash, gap = _mm(DASH_MM), _mm(GAP_MM)
    period = dash + gap
    seg = np.hypot(*np.diff(points, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    on = (cum % period) < dash

    canvas = np.array(page)
    width = max(1, _mm(width_mm))
    run: list[np.ndarray] = []
    for index in range(len(points)):
        if on[index]:
            run.append(points[index])
            continue
        if len(run) >= 2:
            cv2.polylines(canvas, [np.round(np.array(run)).astype(np.int32)], False, BLACK, width)
        elif len(run) == 1:
            cv2.circle(canvas, tuple(np.round(run[0]).astype(int)), max(1, width // 2), BLACK, -1)
        run = []
    if len(run) >= 2:
        cv2.polylines(canvas, [np.round(np.array(run)).astype(np.int32)], False, BLACK, width)
    page.paste(Image.fromarray(canvas))


# ---------------------------------------------------------------------------
# หัวหน้า: ชื่อหนังสือ + บรรทัดชื่อ/วันที่
# ---------------------------------------------------------------------------


def _draw_header(page: Image.Image, params: ActivityParams, subtitle: str) -> int:
    """วาดหัวหน้า คืนตำแหน่ง y ที่เนื้อหาเริ่มได้"""
    inner = _frame_inner(params)
    y = inner.y + _mm(5)

    if params.show_name_line:
        label_h = pt_to_px(14)
        name = text.render_text("ชื่อ", label_h, max_width=inner.w)
        date = text.render_text("วันที่", label_h, max_width=inner.w)
        left = inner.x + _mm(6)
        page.paste(BLACK, (left, y), name)
        name_end = left + int(inner.w * 0.52)
        _line(page, (left + name.width + _mm(2), y + name.height - _mm(2)), (name_end, y + name.height - _mm(2)), 0.3)
        date_x = name_end + _mm(8)
        page.paste(BLACK, (date_x, y), date)
        _line(
            page,
            (date_x + date.width + _mm(2), y + name.height - _mm(2)),
            (inner.right - _mm(6), y + name.height - _mm(2)),
            0.3,
        )
        y += name.height + _mm(5)

    title = text.render_text(subtitle or params.title, pt_to_px(28), bold=True, max_width=inner.w - _mm(12))
    page.paste(BLACK, (inner.x + (inner.w - title.width) // 2, y), title)
    y += title.height + _mm(3)
    return y


# ---------------------------------------------------------------------------
# เขียนตามรอยประ
# ---------------------------------------------------------------------------


def _dashed_glyph(mask: Image.Image) -> Image.Image:
    """เปลี่ยนตัวอักษรทึบเป็นตัวอักษรกลวงที่ขอบเป็นเส้นประ (255 = เส้น)"""
    arr = np.array(mask)
    binary = (arr > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    draw_target = Image.new("L", mask.size, 255)  # พื้นขาว วาดเส้นเป็นดำ แล้วกลับค่าตอนท้าย
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float)
        if len(points) < 6:
            continue
        # ปิดรอบ เพื่อให้เส้นประไม่ขาดตรงจุดเริ่ม
        closed = np.vstack([points, points[:1]])
        _dotted_polyline(draw_target, closed)
    return Image.eval(draw_target, lambda v: 255 - v)


def _ink_crop(mask: Image.Image) -> Image.Image:
    box = mask.getbbox()
    return mask.crop(box) if box else mask


def _glyph_row_images(item: str, row_h: int, max_w: int, size_mm: float) -> list[Image.Image]:
    """ภาพตัวอักษรเส้นประของหนึ่งแถว (ตัวสั้นซ้ำเต็มแถว ตัวยาวหนึ่งชุดต่อแถว)"""
    base_px = max(24, round(_mm(size_mm) * 1.45))
    short = len(item.replace(" ", "")) <= 2
    width_for_one = max_w if not short else max(_mm(size_mm) * 2, max_w // 4)
    solid = text.render_text(item, base_px, bold=True, max_width=width_for_one)
    glyph = _ink_crop(_dashed_glyph(solid))

    # ย่อไม่ให้สูงเกินแถว
    limit_h = max(1, row_h - _mm(6))
    if glyph.height > limit_h:
        scale = limit_h / glyph.height
        solid = text.render_text(item, max(12, round(base_px * scale)), bold=True, max_width=width_for_one)
        glyph = _ink_crop(_dashed_glyph(solid))

    if not short:
        return [glyph]

    gap = _mm(10)
    count = max(1, (max_w + gap) // (glyph.width + gap))
    return [glyph] * int(count)


def _draw_guides(page: Image.Image, x0: int, x1: int, top: int, bottom: int) -> None:
    """เส้นบรรทัด: เส้นล่างทึบ เส้นบนและเส้นกลางเป็นเส้นประบาง"""
    mid = (top + bottom) // 2
    _line(page, (x0, bottom), (x1, bottom), 0.35)
    _dashed_line(page, (x0, top), (x1, top), 0.2, dash_mm=2.0, gap_mm=2.0)
    _dashed_line(page, (x0, mid), (x1, mid), 0.2, dash_mm=2.0, gap_mm=2.0)


def _trace_rows(item: str, params: ActivityParams, inner: Box) -> list[str]:
    """รายการแถวของหนึ่งข้อความ: 'dots' = แถวเส้นประ, 'blank' = แถวว่างให้เขียนเอง"""
    return ["dots"] + ["blank"] * max(0, params.trace_blank_rows)


def _render_trace_pages(params: ActivityParams, page_counter: list[int]) -> list[Image.Image]:
    items = [i.strip() for i in params.trace_items if i and i.strip()]
    if not items:
        return []

    inner = _frame_inner(params)
    row_h = _mm(params.trace_size_mm) + _mm(9)
    row_gap = _mm(3)
    side = _mm(8)
    max_w = inner.w - 2 * side

    pages: list[Image.Image] = []
    page: Image.Image | None = None
    y = 0
    bottom_limit = inner.bottom - _mm(10)

    def start_page() -> None:
        nonlocal page, y
        page = _new_page(params)
        y = _draw_header(page, params, "เขียนตามรอยประ") + _mm(3)

    start_page()
    for item in items:
        rows = _trace_rows(item, params, inner)
        block_h = len(rows) * (row_h + row_gap)
        if y + block_h > bottom_limit and y > _draw_header_height_guess(params):
            _finish(page, page_counter, params, pages)
            start_page()
        for kind in rows:
            top, bottom = y, y + row_h
            x0, x1 = inner.x + side, inner.right - side
            if params.trace_guides:
                if kind == "dots":
                    # แถวตัวอักษรเส้นประ: มีแค่เส้นฐาน ไม่ให้เส้นไกด์ตัดกลางตัวอักษร
                    _line(page, (x0, bottom - _mm(2)), (x1, bottom - _mm(2)), 0.35)
                else:
                    _draw_guides(page, x0, x1, top + _mm(2), bottom - _mm(2))
            if kind == "dots":
                images = _glyph_row_images(item, row_h, max_w, params.trace_size_mm)
                gap = _mm(10)
                total_w = sum(g.width for g in images) + gap * (len(images) - 1)
                cursor = x0 + max(0, (max_w - total_w) // 2)
                for glyph in images:
                    gy = top + (row_h - glyph.height) // 2
                    page.paste(BLACK, (cursor, gy), glyph)
                    cursor += glyph.width + gap
            y += row_h + row_gap
    _finish(page, page_counter, params, pages)
    return pages


def _draw_header_height_guess(params: ActivityParams) -> int:
    """ใช้แยกว่าหน้านี้ยังไม่มีเนื้อหาเลยหรือไม่ (กันขึ้นหน้าใหม่วนไม่จบ)"""
    inner = _frame_inner(params)
    name_h = pt_to_px(14) + _mm(5) if params.show_name_line else 0
    return inner.y + _mm(5) + name_h + pt_to_px(28) + _mm(12)


def _finish(page: Image.Image | None, counter: list[int], params: ActivityParams, pages: list[Image.Image]) -> None:
    if page is None:
        return
    counter[0] += 1
    _draw_page_number(page, counter[0], params)
    pages.append(compose.to_bilevel(page))


# ---------------------------------------------------------------------------
# แบบฝึกลากเส้น
# ---------------------------------------------------------------------------


def _pattern_points(kind: str, x0: float, x1: float, top: float, bottom: float) -> list[np.ndarray]:
    """จุดของเส้นแต่ละเส้นในแถบหนึ่งแถบ (หลายเส้นต่อแถบสำหรับเส้นตรง/วงกลม)"""
    h = bottom - top
    w = x1 - x0
    mid = (top + bottom) / 2
    lines: list[np.ndarray] = []

    def sample(p1, p2, n=400):
        return np.stack([np.linspace(p1[0], p2[0], n), np.linspace(p1[1], p2[1], n)], axis=1)

    if kind == "vertical":
        count = 9
        for i in range(count):
            x = x0 + w * (i + 0.5) / count
            lines.append(sample((x, top + h * 0.05), (x, bottom - h * 0.05)))
    elif kind == "horizontal":
        for i in range(3):
            y = top + h * (i + 0.5) / 3
            lines.append(sample((x0, y), (x1, y)))
    elif kind == "slant":
        count = 8
        for i in range(count):
            x = x0 + w * (i + 0.5) / count
            if i % 2 == 0:
                lines.append(sample((x - h * 0.25, bottom - h * 0.05), (x + h * 0.25, top + h * 0.05)))
            else:
                lines.append(sample((x - h * 0.25, top + h * 0.05), (x + h * 0.25, bottom - h * 0.05)))
    elif kind == "wave":
        xs = np.linspace(x0, x1, 900)
        cycles = 6
        ys = mid + (h * 0.38) * np.sin((xs - x0) / w * cycles * 2 * math.pi)
        lines.append(np.stack([xs, ys], axis=1))
    elif kind == "zigzag":
        teeth = 10
        xs = np.linspace(x0, x1, teeth * 2 + 1)
        ys = np.where(np.arange(len(xs)) % 2 == 0, bottom - h * 0.08, top + h * 0.08)
        pts = np.stack([xs, ys], axis=1)
        dense = []
        for a, b in itertools.pairwise(pts):
            dense.append(sample(a, b, 80))
        lines.append(np.vstack(dense))
    elif kind == "circle":
        count = 5
        radius = min(h * 0.42, w / count * 0.38)
        for i in range(count):
            cx = x0 + w * (i + 0.5) / count
            t = np.linspace(-math.pi / 2, 3 * math.pi / 2, 400)
            lines.append(np.stack([cx + radius * np.cos(t), mid + radius * np.sin(t)], axis=1))
    elif kind == "loop":
        xs = np.linspace(0, 1, 1400)
        loops = 6
        phase = xs * loops * 2 * math.pi
        radius = h * 0.30
        step = w / loops
        px = x0 + xs * (w - step) + step * 0.5 + radius * 0.9 * np.sin(phase)
        py = mid - radius * np.cos(phase)
        lines.append(np.stack([px, py], axis=1))
    return lines


def _start_marker(page: Image.Image, point: np.ndarray, nxt: np.ndarray) -> None:
    """จุดเริ่ม (วงกลมทึบ) และลูกศรบอกทิศ"""
    draw = ImageDraw.Draw(page)
    r = _mm(1.6)
    x, y = float(point[0]), float(point[1])
    draw.ellipse([x - r, y - r, x + r, y + r], fill=BLACK)


def _end_arrow(page: Image.Image, prev: np.ndarray, end: np.ndarray) -> None:
    ang = math.atan2(end[1] - prev[1], end[0] - prev[0])
    size = _mm(3.2)
    tip = (float(end[0]), float(end[1]))
    left = (tip[0] - size * math.cos(ang - 0.45), tip[1] - size * math.sin(ang - 0.45))
    right = (tip[0] - size * math.cos(ang + 0.45), tip[1] - size * math.sin(ang + 0.45))
    ImageDraw.Draw(page).polygon([tip, left, right], fill=BLACK)


def _render_prewriting_pages(params: ActivityParams, page_counter: list[int]) -> list[Image.Image]:
    kinds = [k for k in PREWRITING_PATTERNS if k in params.prewriting]
    if not kinds:
        return []

    inner = _frame_inner(params)
    side = _mm(10)
    band_h = _mm(34)
    band_gap = _mm(5)
    bottom_limit = inner.bottom - _mm(14)

    pages: list[Image.Image] = []
    page: Image.Image | None = None
    y = 0

    def start_page() -> None:
        nonlocal page, y
        page = _new_page(params)
        y = _draw_header(page, params, "ฝึกลากเส้น") + _mm(4)

    start_page()
    for kind in kinds:
        if y + band_h > bottom_limit and y > _draw_header_height_guess(params):
            _finish(page, page_counter, params, pages)
            start_page()

        label = text.render_text(PREWRITING_PATTERNS[kind], pt_to_px(14), max_width=inner.w)
        page.paste(BLACK, (inner.x + side, y), label)
        top = y + label.height + _mm(1)
        bottom = y + band_h
        lines = _pattern_points(kind, inner.x + side + _mm(4), inner.right - side - _mm(4), top, bottom)
        for pts in lines:
            _dotted_polyline(page, pts)
            _start_marker(page, pts[0], pts[1])
            if kind in ("vertical", "horizontal", "slant", "zigzag"):
                _end_arrow(page, pts[-8], pts[-1])
        y += band_h + band_gap
    _finish(page, page_counter, params, pages)
    return pages


# ---------------------------------------------------------------------------
# จับคู่
# ---------------------------------------------------------------------------


def _silhouette(mask: np.ndarray) -> np.ndarray:
    """เงาทึบของภาพลายเส้น: ปิดช่องว่างของเส้นแล้วถมสีด้านใน"""
    ink = (mask > 127).astype(np.uint8) * 255
    k = max(3, round(max(ink.shape) * 0.012) | 1)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(ink)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _paste_fit(page: Image.Image, mask: np.ndarray, box: Box) -> None:
    """วางภาพทึบให้พอดีกล่อง รักษาสัดส่วน (ใช้กับเงา)"""
    ys, xs = np.nonzero(mask > 127)
    if len(xs) == 0:
        return
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = min(box.w / crop.shape[1], box.h / crop.shape[0])
    w, h = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(resized)
    page.paste(BLACK, (box.x + (box.w - w) // 2, box.y + (box.h - h) // 2), img)


def _text_in_box(page: Image.Image, label: str, box: Box, size_pt: float = 24, bold: bool = True) -> None:
    rendered = text.render_text(label, pt_to_px(size_pt), bold=bold, max_width=max(1, box.w - _mm(8)))
    page.paste(
        BLACK,
        (box.x + (box.w - rendered.width) // 2, box.y + (box.h - rendered.height) // 2),
        rendered,
    )


def _rounded_box(page: Image.Image, box: Box, stroke_mm: float = 0.8, radius_mm: float = 4.0) -> None:
    layer = Image.new("L", page.size, WHITE)
    ImageDraw.Draw(layer).rounded_rectangle(
        box.as_tuple(), radius=_mm(radius_mm), outline=BLACK, width=max(1, _mm(stroke_mm))
    )
    page.paste(BLACK, (0, 0), layer.point(lambda v: 255 if v < 128 else 0))


def _dot(page: Image.Image, cx: int, cy: int) -> None:
    r = _mm(2.2)
    ImageDraw.Draw(page).ellipse([cx - r, cy - r, cx + r, cy + r], fill=WHITE, outline=BLACK, width=max(1, _mm(0.7)))


def _derangement(count: int, rng: random.Random) -> list[int]:
    """สลับลำดับด้านขวา: พยายามไม่ให้คู่ใดอยู่ตรงข้ามกัน"""
    order = list(range(count))
    if count < 2:
        return order
    for _ in range(60):
        rng.shuffle(order)
        if all(order[i] != i for i in range(count)):
            return order
    # ถ้าสุ่มไม่ได้ ให้หมุนลำดับไปหนึ่งตำแหน่ง รับรองว่าไม่มีคู่ตรงข้ามกัน
    return [(i + 1) % count for i in range(count)]


def _render_matching_page(
    items: list[MatchItem],
    params: ActivityParams,
    rng: random.Random,
    page_number: int,
    answer: bool,
    layout_order: list[int] | None = None,
) -> tuple[Image.Image, list[int]]:
    inner = _frame_inner(params)
    page = _new_page(params)
    subtitle = "เฉลย: จับคู่" if answer else "จับคู่"
    y0 = _draw_header(page, params, subtitle)

    hint_text = {
        "image_word": "ลากเส้นจับคู่ภาพกับชื่อที่ตรงกัน",
        "image_shadow": "ลากเส้นจับคู่ภาพกับเงาที่ตรงกัน",
        "text_pairs": "ลากเส้นจับคู่ซ้ายกับขวาที่ตรงกัน",
    }[params.match_mode]
    if not answer:
        hint = text.render_text(hint_text, pt_to_px(14), max_width=inner.w - _mm(12))
        page.paste(BLACK, (inner.x + (inner.w - hint.width) // 2, y0), hint)
        y0 += hint.height + _mm(3)

    n = len(items)
    order = layout_order if layout_order is not None else _derangement(n, rng)

    area_top = y0 + _mm(2)
    area_bottom = inner.bottom - _mm(14)
    row_h = min(_mm(58), (area_bottom - area_top) // max(1, n))
    block_h = row_h * n
    top = area_top + max(0, (area_bottom - area_top - block_h) // 2)

    box_w = int(inner.w * 0.36)
    left_x = inner.x + _mm(8)
    right_x = inner.right - _mm(8) - box_w
    gap_v = _mm(6)

    left_dots: list[tuple[int, int]] = []
    right_dots: list[tuple[int, int]] = []

    for row in range(n):
        cell_top = top + row * row_h
        cell_h = row_h - gap_v
        left_box = Box(left_x, cell_top, box_w, cell_h)
        right_box = Box(right_x, cell_top, box_w, cell_h)
        left_item = items[row]
        right_item = items[order[row]]

        # ซ้าย
        _rounded_box(page, left_box)
        if params.match_mode == "text_pairs":
            _text_in_box(page, left_item.caption, left_box)
        elif left_item.mask is not None:
            art, _note = compose.place_artwork(
                left_item.mask,
                left_box.inset(_mm(3)),
                _mm(AUTO_TARGET_LINE_MM),
            )
            page.paste(art, (left_box.x + _mm(3), left_box.y + _mm(3)))

        # ขวา
        _rounded_box(page, right_box)
        if params.match_mode == "image_word":
            _text_in_box(page, right_item.caption, right_box)
        elif params.match_mode == "image_shadow" and right_item.mask is not None:
            _paste_fit(page, _silhouette(right_item.mask), right_box.inset(_mm(4)))
        elif params.match_mode == "text_pairs":
            _text_in_box(page, right_item.right_text or "", right_box)

        cy = cell_top + cell_h // 2
        left_dots.append((left_box.right + _mm(4), cy))
        right_dots.append((right_box.x - _mm(4), cy))

    # เฉลย: ลากเส้นจากซ้ายของคู่ไปหาตำแหน่งที่คู่นั้นไปอยู่ทางขวา
    if answer:
        for row in range(n):
            target_row = order.index(row)
            _line(page, left_dots[row], right_dots[target_row], 0.9)

    for point in left_dots + right_dots:
        _dot(page, *point)

    _draw_page_number(page, page_number, params)
    return compose.to_bilevel(page), order


def _render_matching_pages(
    items: list[MatchItem], params: ActivityParams, page_counter: list[int], rng: random.Random
) -> tuple[list[Image.Image], list[Image.Image]]:
    if not items:
        return [], []
    per = min(MAX_PAIRS_PER_PAGE, max(MIN_PAIRS_PER_PAGE, params.pairs_per_page))
    chunks = [items[i : i + per] for i in range(0, len(items), per)]
    # หน้าสุดท้ายที่เหลือคู่เดียวไม่มีอะไรให้จับ → รวมกับหน้าก่อนหน้า
    if len(chunks) > 1 and len(chunks[-1]) < 2:
        chunks[-2].extend(chunks.pop())

    pages: list[Image.Image] = []
    answers: list[Image.Image] = []
    orders: list[list[int]] = []
    for chunk in chunks:
        page_counter[0] += 1
        page, order = _render_matching_page(chunk, params, rng, page_counter[0], answer=False)
        pages.append(page)
        orders.append(order)

    if params.answer_key:
        for chunk, order in zip(chunks, orders):
            page_counter[0] += 1
            page, _ = _render_matching_page(
                chunk, params, rng, page_counter[0], answer=True, layout_order=order
            )
            answers.append(page)
    return pages, answers


# ---------------------------------------------------------------------------
# จุดเข้าหลัก
# ---------------------------------------------------------------------------


def iter_activity_pages(
    params: ActivityParams,
    match_items: list[MatchItem],
    warnings: list[str],
    rng: random.Random | None = None,
):
    """สร้างหน้าตามลำดับ: ฝึกลากเส้น → เขียนตามรอยประ → จับคู่ → เฉลย"""
    rng = rng or random.Random()
    counter = [0]

    for page in _render_prewriting_pages(params, counter):
        yield page
    for page in _render_trace_pages(params, counter):
        yield page

    if params.match_mode not in MATCH_MODES:
        raise ValueError(f"ไม่รู้จักโหมดจับคู่: {params.match_mode}")

    usable = [i for i in match_items if _usable(i, params.match_mode)]
    if match_items and len(usable) < 2:
        warnings.append("ต้องมีอย่างน้อย 2 รายการจึงจะทำหน้าจับคู่ได้ จึงข้ามหน้าจับคู่")
        usable = []
    if len(usable) > 1:
        pages, answers = _render_matching_pages(usable, params, counter, rng)
        for page in pages:
            yield page
        for page in answers:
            yield page


def _usable(item: MatchItem, mode: str) -> bool:
    if mode == "text_pairs":
        return bool(item.caption.strip()) and bool((item.right_text or "").strip())
    if item.mask is None:
        return False
    if mode == "image_word":
        return bool(item.caption.strip())
    return True
