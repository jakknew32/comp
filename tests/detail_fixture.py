"""สร้างภาพทดสอบที่มีเส้นหลายระดับความหนาและความเข้ม เพื่อหาสาเหตุที่ภาพออกมาขาดๆ"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


def detail_challenge(width: int = 900, height: int = 1270) -> np.ndarray:
    """ภาพที่มีเส้นซับซ้อนตามแบบหน้าสมุดระบายสีจริง

    รวมสิ่งที่มักหายไปในการทำ line art:
    - เส้นบางมาก (1-2 พิกเซล) ซึ่นตรงกับเส้นรายละเอียดของงานจริง
    - เส้นสีเทาอ่อน ไม่ใช่ดำสนิท
    - ลายประดับเล็ก ๆ เช่น จุด หัวใจ ดาว
    - เส้นที่หั่นกันเป็นตาข่าย
    """
    page = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(page)

    # เส้นหนา 3 ระดับความหนา
    draw.rounded_rectangle((60, 60, width - 60, 300), radius=30, outline=0, width=10)
    draw.ellipse((120, 120, width - 120, 300), outline=0, width=5)
    draw.ellipse((200, 180, width - 200, 300), outline=0, width=2)

    # เส้นสีเทาอ่อน ไม่ใช่ดำสนิท
    draw.ellipse((100, 380, width - 100, 800), outline=110, width=4)
    draw.ellipse((150, 430, width - 150, 800), outline=150, width=3)

    # เส้นบางมาก 1-2 พิกเซล
    for i in range(12):
        y = 850 + i * 22
        draw.line((120, y, width - 120, y + 40), fill=0, width=1)
    for i in range(12):
        x = 140 + i * 55
        draw.line((x, 1140, x + 30, 1230), fill=0, width=2)

    # ลายประดับเล็ก ๆ
    for i in range(40):
        x = 100 + (i * 37) % (width - 200)
        y = 850 + (i * 53) % 280
        r = 3 + (i % 4)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=0, width=2)

    # ตาข่ายเส้นหั่นกัน
    for i in range(9):
        x = 130 + i * 78
        draw.line((x, 350, x, 830), fill=0, width=2)
    for i in range(6):
        y = 360 + i * 90
        draw.line((110, y, width - 110, y), fill=0, width=2)

    return np.array(page)


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent.parent / "samples"
    out.mkdir(exist_ok=True)
    img = detail_challenge()
    cv2.imwrite(str(out / "sample_detail_challenge.png"), img)
    print("เขียน", out / "sample_detail_challenge.png", img.shape)
