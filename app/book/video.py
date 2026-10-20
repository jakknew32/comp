"""สร้างวิดีโอ Slideshow (MP4) จากหน้าสมุดระบายสีสำหรับโปรโมท Affiliate

ใช้ OpenCV (cv2) และ Pillow สร้างวิดีโอความละเอียดแนวตั้ง (9:16 - 1080x1920) สำหรับ TikTok / Reels / Shorts
ฟรี 100% ไม่ต้องใช้ API ภายนอก
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import FONT_REGULAR, FONT_BOLD


def create_affiliate_video(
    page_images: list[np.ndarray],
    output_path: str | Path,
    hook_text: str = "🎨 สมุดระบายสีฝึกพัฒนาการเด็ก พิกัดลิงก์ใต้คลิป",
    fps: int = 30,
    seconds_per_page: float = 2.5,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """สร้างวิดีโอ MP4 แนวตั้งจากชุดภาพหน้าสมุดระบายสี

    Args:
        page_images: รายการภาพ NumPy array ของแต่ละหน้า
        output_path: พาธไฟล์วิดีโอผลลัพธ์ (.mp4)
        hook_text: ข้อความพาดหัวดึงดูดใจ (Hook) สำหรับ Affiliate
        fps: เฟรมเรต (ค่ามาตรฐาน 30)
        seconds_per_page: ระยะเวลาแสดงแต่ละภาพ (วินาที)
        width, height: ขนาดวิดีโอ (1080x1920 สำหรับ Reels/TikTok)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frames_per_page = int(fps * seconds_per_page)

    try:
        for idx, img in enumerate(page_images):
            if img is None or img.size == 0:
                continue

            # แปลงภาพ OpenCV เป็น PIL Image เพื่อให้วาดข้อความภาษาไทยได้สวยงาม
            if img.ndim == 2:
                pil_img = Image.fromarray(img).convert("RGB")
            elif img.shape[2] == 4:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # สร้าง Canvas แนวตั้ง (พื้นหลังสีพาสเทลอ่อนๆ)
            canvas = Image.new("RGB", (width, height), color=(245, 247, 250))
            draw = ImageDraw.Draw(canvas)

            # ย่อ/ขยายหน้าสมุดระบายสีให้อยู่ตรงกลางสวยงาม
            orig_w, orig_h = pil_img.size
            target_w = int(width * 0.85)
            target_h = int(orig_h * (target_w / orig_w))

            # จำกัดความสูงไม่ให้เกิน 68% ของจอ
            max_page_height = int(height * 0.68)
            if target_h > max_page_height:
                target_h = max_page_height
                target_w = int(orig_w * (target_h / orig_h))

            resized_page = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            # วางภาพสมุดลงบน Canvas (ตรงกลาง)
            pos_x = (width - target_w) // 2
            pos_y = 380  # เว้นพื้นที่ด้านบนไว้ใส่ข้อความ Hook
            canvas.paste(resized_page, (pos_x, pos_y))

            # วาดกรอบเงาหรือกรอบรองรับภาพให้ดูเป็นเล่มสมุด
            border_box = [pos_x - 6, pos_y - 6, pos_x + target_w + 6, pos_y + target_h + 6]
            draw.rectangle(border_box, outline=(220, 224, 230), width=3)

            # วาดแถบข้อความพาดหัว (Hook Banner) ด้านบน
            banner_box = [40, 140, width - 40, 300]
            draw.rounded_rectangle(banner_box, radius=20, fill=(255, 107, 107), outline=None)

            try:
                font_title = ImageFont.truetype(str(FONT_BOLD), 46)
                font_sub = ImageFont.truetype(str(FONT_REGULAR), 32)
            except IOError:
                font_title = ImageFont.load_default()
                font_sub = ImageFont.load_default()

            # ใส่ข้อความพาดหัว
            draw.text((width // 2, 200), hook_text, fill=(255, 255, 255), font=font_title, anchor="mm")
            draw.text((width // 2, 260), "✨ ภาพคมชัด ไร้ลายน้ำ พร้อมระบายสี", fill=(255, 240, 240), font=font_sub, anchor="mm")

            # หมายเลขหน้า / Call to action ด้านล่าง
            cta_box = [150, height - 200, width - 150, height - 100]
            draw.rounded_rectangle(cta_box, radius=16, fill=(46, 134, 222), outline=None)
            cta_text = f"หน้าที่ {idx + 1} จาก {len(page_images)} — สั่งซื้อพิกัดหน้าโปรไฟล์ 🛒"
            draw.text((width // 2, height - 150), cta_text, fill=(255, 255, 255), font=font_sub, anchor="mm")

            # แปลงกลับเป็น OpenCV BGR เพื่อเขียนลง VideoWriter
            frame_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

            # เขียนซ้ำตามจำนวนเฟรมที่กำหนด
            for _ in range(frames_per_page):
                writer.write(frame_bgr)

    finally:
        writer.release()

    return output_path
