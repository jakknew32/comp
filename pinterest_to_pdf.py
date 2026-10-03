import io
import json
import re
import requests
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ขนาดหน้ากระดาษ A4 ในหน่วย Points (72 points = 1 inch)
A4_WIDTH, A4_HEIGHT = A4


def scrape_pinterest_images(url: str) -> list[str]:
    """ดึง URL รูปภาพจากหน้า Pinterest โดยใช้ Regex และ JSON Parsing"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to fetch Pinterest page. Status: {response.status_code}")
            return []

        html = response.text
        image_urls = set()

        # 1. ค้นหา URL รูปภาพระดับความละเอียดสูง (736x หรือ originals)
        matches = re.findall(
            r"https://i\.pinimg\.com/(?:originals|\d+x)/[a-f0-9/]+\.(?:jpg|png|webp)",
            html,
        )
        for img_url in matches:
            high_res = re.sub(r"/(?:236x|474x|170x)/", "/736x/", img_url)
            image_urls.add(high_res)

        # 2. ค้นหาจาก JSON Script หาก Regex ปกติไม่พบ
        if not image_urls:
            json_match = re.search(
                r'<script id="__PINTEREST_APP_STATE__" type="application/json">(.*?)</script>',
                html,
            )
            if json_match:
                raw_json = json_match.group(1)
                found_imgs = re.findall(
                    r"https://i\.pinimg\.com/[^\"\']+\.(?:jpg|png|webp)",
                    raw_json,
                )
                for img_url in found_imgs:
                    high_res = re.sub(r"/(?:236x|474x|170x)/", "/736x/", img_url)
                    image_urls.add(high_res)

        return list(image_urls)

    except Exception as e:
        print(f"❌ Error scraping Pinterest: {e}")
        return []


def download_image(img_url: str) -> Image.Image | None:
    """ดาวน์โหลดรูปภาพจาก CDN ของ Pinterest และแปลงเป็น PIL Image"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        res = requests.get(img_url, headers=headers, timeout=10)
        if res.status_code == 200:
            img = Image.open(io.BytesIO(res.content))
            # แปลงโหมดภาพเป็น RGB กรณีที่ได้ภาพ RGBA/P มา เพื่อให้สอดคล้องกับ PDF
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            return img
    except Exception as e:
        print(f"⚠️ Failed to download {img_url}: {e}")
    return None


def generate_a4_pdf(image_urls: list[str], output_pdf_path: str):
    """ดาวน์โหลดรูปภาพและจัดวางลงใน A4 PDF (1 รูปต่อ 1 หน้ากระดาษ พอดีขอบ)"""
    c = canvas.Canvas(output_pdf_path, pagesize=A4)

    success_count = 0

    for index, url in enumerate(image_urls):
        print(f"📥 [{index + 1}/{len(image_urls)}] Downloading: {url}")
        img = download_image(url)

        if not img:
            continue

        # คำนวณอัตราส่วนภาพเพื่อให้สเกลลง A4 โดยคง Aspect Ratio ไว้
        img_w, img_h = img.size
        aspect = img_w / img_h

        # กำหนดระยะขอบ (Margin) เล็กน้อย เช่น 20 points
        margin = 20
        max_w = A4_WIDTH - (margin * 2)
        max_h = A4_HEIGHT - (margin * 2)

        if aspect > (max_w / max_h):
            # กว้างเกินไป สเกลตามความกว้าง
            render_w = max_w
            render_h = max_w / aspect
        else:
            # สูงเกินไป สเกลตามความสูง
            render_h = max_h
            render_w = max_h * aspect

        # คำนวณพิกัดให้รูปอยู่ตรงกลางหน้า A4
        x = (A4_WIDTH - render_w) / 2
        y = (A4_HEIGHT - render_h) / 2

        # บันทึกรูปชั่วคราวลง Memory buffer แล้ววาดลง Canvas
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        # วาดรูปภาพลงใน PDF
        from reportlab.lib.utils import ImageReader

        c.drawImage(
            ImageReader(img_buffer),
            x,
            y,
            width=render_w,
            height=render_h,
        )

        # เพิ่มหน้าใหม่ (ถ้ายังมีรูปถัดไป)
        c.showPage()
        success_count += 1

    c.save()
    print(
        f"\n✅ สร้างไฟล์ PDF สำเร็จ: {output_pdf_path} (รวม {success_count} หน้า)"
    )


# --- ตัวอย่างการใช้งาน ---
if __name__ == "__main__":
    pinterest_url = "https://www.pinterest.com/search/pins/?q=%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B9%8C%E0%B8%95%E0%B8%B9%E0%B8%99%E0%B9%80%E0%B8%94%E0%B9%87%E0%B8%81%20%E0%B8%A3%E0%B8%96%E0%B8%9E%E0%B8%A2%E0%B8%B2%E0%B8%A5&rs=typed"

    print("🔍 กำลังค้นหารูปภาพจาก Pinterest...")
    urls = scrape_pinterest_images(pinterest_url)

    if urls:
        print(f"🎯 พบรูปภาพทั้งหมด {len(urls)} รูป")
        generate_a4_pdf(urls, "pinterest_coloring_pages.pdf")
    else:
        print("❌ ไม่พบรูปภาพจากลิงก์ที่ระบุ")