#!/usr/bin/env python3
"""สร้างภาพตัวอย่างสำหรับทดสอบและสาธิต

ภาพที่สร้างครอบคลุมกรณีที่แผนระบุไว้:
  1. line art สะอาด (พื้นหลังขาว เส้นบางสม่ำเสมอ)
  2. line art ถ่ายจากกระดาษมีเงา (ไล่แสง + คราบ)
  3. เส้นหนาผิดปกติ (เส้นหนาไม่สม่ำเสมอ)

เรียกใช้: ``python -m samples.make_samples`` หรือ ``python samples/make_samples.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

#: เก็บสุ่มด้วย seed คงที่ เพื่อให้ผลทดสอบทำซ้ำได้เหมือนเดิม
RNG = np.random.default_rng(20240701)


def _jitter_gray(rng, shape, sigma, mean=255.0):
    noise = rng.normal(mean, sigma, shape).astype(np.float32)
    return np.clip(noise, 0, 255).astype(np.uint8)


def clean_lineart(size: int = 1200) -> np.ndarray:
    """ภาพลายเส้นสะอาด: เส้นบางสม่ำเสมung พื้นหลังขาวบริสุทธิ์"""
    img = np.full((size, size), 255, np.uint8)
    c = size // 2

    # ใบหน้าวงกลม + หูแมว
    cv2.circle(img, (c, int(size * 0.46)), int(size * 0.30), 0, 3)
    cv2.line(img, (int(size * 0.20), int(size * 0.28)),
             (int(size * 0.30), int(size * 0.12)), 0, 3)
    cv2.line(img, (int(size * 0.80), int(size * 0.28)),
             (int(size * 0.70), int(size * 0.12)), 0, 3)
    cv2.line(img, (int(size * 0.27), int(size * 0.16)),
             (int(size * 0.33), int(size * 0.16)), 0, 3)
    cv2.line(img, (int(size * 0.67), int(size * 0.16)),
             (int(size * 0.73), int(size * 0.16)), 0, 3)

    # ตากลมโต (ก้อนทึบ — ต้องไม่หายไปตอนกำจัดจุดรบกวน)
    for dx in (-0.10, 0.10):
        cv2.circle(img, (int(size * (0.5 + dx)), int(size * 0.42)),
                   int(size * 0.045), 0, -1)
    # จมูยิ้ม
    cv2.ellipse(img, (c, int(size * 0.56)),
                (int(size * 0.10), int(size * 0.07)), 200, 340, 0, 3)
    # ครอบฟัน
    for i in range(4):
        x = int(size * (0.44 + 0.03 * i))
        cv2.rectangle(img, (x, int(size * 0.545)),
                      (x + int(size * 0.022), int(size * 0.585)), 0, 2)

    # เส้นประ 3 เส้นเพื่อทดสอบว่าเส้นขาดถูกซ่อม
    for i in range(3):
        y = int(size * (0.78 + 0.05 * i))
        for k in range(4):
            cv2.line(img, (int(size * 0.20) + k * int(size * 0.16), y),
                     (int(size * 0.20) + k * int(size * 0.16) + int(size * 0.10), y),
                     0, 3)
    return img


def photographed(size: int = 1200) -> np.ndarray:
    """line art ที่ถ่ายจากกระดาษ: มีไล่แสง เงา และคราบ"""
    base = clean_lineart(size).astype(np.float32)
    h, w = base.shape

    # ไล่แสงเฉียง (แสงจากบนซ้าย)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = 1.0 - 0.35 * ((xx / w) * 0.5 + (yy / h) * 0.5)
    shaded = base * gradient

    # เงามืดมุมขวาล่าง
    shadow = np.exp(-(((xx - w * 1.05) ** 2 + (yy - h * 1.05) ** 2) / (2 * (w * 0.42) ** 2)))
    shaded = shaded * (1.0 - 0.45 * shadow)

    # เงาของมือทับมุมซ้ายบน
    band = np.clip((yy - h * 0.62) / (h * 0.12), 0, 1)
    shaded = shaded * (1.0 - 0.25 * band)

    # คราบเทา (เปรอะมือ)
    for _ in range(14):
        cx, cy = RNG.integers(0, w), RNG.integers(0, h)
        radius = int(RNG.integers(w // 40, w // 12))
        cv2.circle(shaded, (int(cx), int(cy)), radius, 0, -1)

    # ไฟและสั่นกระเดือนของกล้อง
    noisy = shaded + RNG.normal(0, 6, (h, w)).astype(np.float32)
    noisy = cv2.GaussianBlur(noisy, (0, 0), 0.8)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def thick_irregular(size: int = 1200) -> np.ndarray:
    """เส้นหนาผิดปกติ: บางถึงหนาในภาพเดียว"""
    img = np.full((size, size), 255, np.uint8)
    c = size // 2

    # เส้นหนา 1 วง
    cv2.ellipse(img, (c, int(size * 0.45)),
                (int(size * 0.30), int(size * 0.22)), 0, 0, 360, 0, 18)
    # เส้นบาง 1 วง
    cv2.ellipse(img, (c, int(size * 0.45)),
                (int(size * 0.22), int(size * 0.15)), 0, 0, 360, 0, 3)
    # เส้นหนาเล็กน้อย
    for dx, r, t in ((-0.12, 0.05, 9), (0.12, 0.05, 9)):
        cv2.circle(img, (int(size * (0.5 + dx)), int(size * 0.42)),
                   int(size * r), 0, -1)
    # เส้นที่ค่อนข้างบาง
    for i in range(6):
        x = int(size * (0.16 + 0.13 * i))
        cv2.line(img, (x, int(size * 0.74)), (x, int(size * 0.92)), 0, 2)
    # เส้นหนามาก
    cv2.line(img, (int(size * 0.16), int(size * 0.70)),
             (int(size * 0.84), int(size * 0.70)), 0, 14)
    return img


def grayscale_photo(size: int = 900) -> np.ndarray:
    """ภาพถ่ายจริงแบบเทา (นอกขอบเขต — ใช้ทดสอบว่าการเตือนทำงาน)"""
    img = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img += 40 * np.sin(xx / 40.0) * np.sin(yy / 35.0)
    img += 90 * np.exp(-(((xx - size * 0.4) ** 2 + (yy - size * 0.5) ** 2) / (2 * (size * 0.3) ** 2)))
    img = cv2.GaussianBlur(img, (0, 0), 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def color_photo(size: int = 800) -> np.ndarray:
    """ภาพถ่ายสี (นอกขอบเขต — ใช้ทดสอบว่าการเตือนทำงาน)"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    b = 180 + 60 * np.sin(xx / 50.0) * np.cos(yy / 45.0)
    g = 140 + 50 * np.sin((xx + yy) / 60.0)
    r = 200 + 40 * np.cos(xx / 70.0)
    return np.clip(np.dstack([b, g, r]), 0, 255).astype(np.uint8)


SAMPLES: dict[str, np.ndarray] = {
    "lineart-clean.png": lambda: cv2.cvtColor(clean_lineart(), cv2.COLOR_GRAY2BGR),
    "lineart-photographed.png": lambda: cv2.cvtColor(photographed(), cv2.COLOR_GRAY2BGR),
    "lineart-thick-irregular.png": lambda: cv2.cvtColor(
        thick_irregular(), cv2.COLOR_GRAY2BGR
    ),
    "photo-gray.png": lambda: cv2.cvtColor(grayscale_photo(), cv2.COLOR_GRAY2BGR),
    "photo-color.png": lambda: color_photo(),
    # ชื่อไทย + ชื่อที่มีช่องว่าง — ใช้ทดสอบ imgio
    "รูป ลายเส้น สวย.png": lambda: cv2.cvtColor(clean_lineart(), cv2.COLOR_GRAY2BGR),
}


def write_all(out_dir: Path = OUT_DIR) -> list[Path]:
    """เขียนภาพตัวอย่างทั้งหมดลงโฟลเดอร์ คืนรายการไฟล์"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, maker in SAMPLES.items():
        path = out_dir / name
        image = maker()
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"เขียนไฟล์ไม่สำเร็จ: {path}")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in write_all():
        print(f"เขียน {p.name} ({p.stat().st_size:,} ไบต์)")
    sys.exit(0)
