"""สร้างภาพตัวอย่างสำหรับทดสอบ ไม่ต้องพึ่งไฟล์ภาพจริงจากภายนอก

ใช้ทั้งเป็น fixture ในเทสต์ และเป็นตัวอย่างให้คนอื่นรันลองดูว่าโปรแกรมทำอะไรได้
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.lineart import text


def synthetic_lineart(width: int = 900, height: int = 1270) -> np.ndarray:
    """ภาพลายเส้นสังเคราะห์แบบหน้าสมุดระบายสี

    คืนภาพระดับเทา: พื้นหลังขาว (255) เส้นดำ (0) เหมือนภาพจริงที่ผู้ใช้จะอัปโหลด
    มีองค์ประกอบครบตามที่ pipeline ต้องรันได้: เส้นหนาชัด, ก้อนทึบใหญ่ (ตา)
    ลายประดับเล็ก (จุดรบกวน) และกรอบมุมมน
    """
    page = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(page)

    draw.rounded_rectangle((40, 40, width - 40, height - 40), radius=40, outline=0, width=9)

    cx, cy, r = width // 2, int(height * 0.38), int(width * 0.27)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=0, width=8)
    draw.ellipse((cx - r - 60, cy - 60, cx - r + 90, cy + 70), outline=0, width=8)
    draw.ellipse((cx + r - 90, cy - 60, cx + r + 60, cy + 70), outline=0, width=8)

    for dx in (-70, 70):
        draw.ellipse(
            (cx + dx - 32, cy - 20, cx + dx + 32, cy + 44), fill=0, outline=0, width=6
        )

    draw.arc((cx - 45, cy + 55, cx + 45, cy + 115), start=20, end=160, fill=0, width=7)

    body_top = cy + r + 10
    draw.rounded_rectangle(
        (cx - 150, body_top, cx + 150, body_top + 320), radius=70, outline=0, width=8
    )
    for leg in (-100, 0, 100):
        draw.line(
            (cx + leg, body_top + 300, cx + leg, body_top + 430), fill=0, width=10
        )

    # จุดรบกวนขนาดเล็ก ต้องถูกกำจัดไป
    # ภาพต้องใหญ่พอที่จะวางจุดได้ ไม่งั้นช่วงสุ่มจะกลับด้านและพัง
    y_low, y_high = 80, height - 220
    if y_high > y_low and width - 80 > 80:
        rng = np.random.default_rng(7)
        for _ in range(220):
            x = int(rng.integers(80, width - 80))
            y = int(rng.integers(y_low, y_high))
            draw.ellipse((x, y, x + 2, y + 2), fill=0)

    mask = np.array(page)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    label = text.render_text("ยิงทพ", 46, bold=True)
    ink = Image.new("L", (width, height), 0)
    ink.paste(0, ((width - label.width) // 2, height - 130), label)
    mask = np.where((mask > 0) | (np.array(ink) > 127), 255, 0).astype(np.uint8)

    return mask


def synthetic_photo(width: int = 900, height: int = 700) -> np.ndarray:
    """ภาพถ่ายสังเคราะห์ ใช้ทดสอบว่า detect จับได้ว่าไม่ใช่ line art"""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = np.sqrt((xx - width / 2) ** 2 + (yy - height / 2) ** 2)
    bgr = np.zeros((height, width, 3), np.uint8)
    bgr[..., 0] = np.clip(200 - r * 0.6, 0, 255)
    bgr[..., 1] = np.clip(120 + xx * 0.15, 0, 255)
    bgr[..., 2] = np.clip(60 + yy * 0.2, 0, 255)
    noise = rng.normal(0, 18, (height, width, 1))
    return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def photocopied_lineart(width: int = 900, height: int = 1270) -> np.ndarray:
    """ภาพลายเส้นที่ถ่ายจากกระดาษ มีเงาไล่ระดับและไฟฉายสว่าง

    synthetic_lineart คืนภาพระดับเทาโดยพื้นหลังขาว (255) และเส้นดำ (0)
    จึงเพิ่มแสงไล่ระดับตรงๆ ได้โดยไม่ต้องกลับสีก่อน
    """
    page = synthetic_lineart(width, height)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    shade = 40 * (xx / width) - 18 * (yy / height)
    bent = np.clip(page.astype(np.float32) + shade, 0, 255)
    rng = np.random.default_rng(11)
    bent = np.clip(bent + rng.normal(0, 5, bent.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(bent, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent.parent / "samples"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "sample_lineart.png"), synthetic_lineart())
    cv2.imwrite(str(out / "sample_photo.png"), synthetic_photo())
    cv2.imwrite(str(out / "sample_photocopy.png"), photocopied_lineart())
    print("เขียนตัวอย่างลง", out)
