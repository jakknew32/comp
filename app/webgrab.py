"""ดึงรูปจากเว็บไซต์

หน้าเว็บส่ง URL มาให้ โมดูลนี้ไปโหลดหน้านั้น แล้วแยกลิงก์รูปออกมา
เพื่อให้ผู้ใชเลือกได้ว่าว่าจะเอารูปไหนไปทำสมุดระบายสี

ข้อจำกัดสำคัญที่ต้องระวัง:
- เว็บเป้าหมายอาจมีภาพหลายร้อย เกินหน่วยความจำของ Render ถ้าโหลดมาทั้งหมด
  จึงคืนเฉพาะลิงก์ก่อน ผู้ใช้ต้องกดเลือกก่อนจึงจะดาวน์โหลดจริง
- รูปอาจมาจากโดเมนอื่นที่มีนโยบาย CORS เข้ม เบราว์เซอร์ดึงตรงๆ ไม่ได้
  จึงต้องให้เซิร์ฟเวอร์เป็นคนดาวน์โหลดแล้วส่งกลับเป็นไฟล์
"""

from __future__ import annotations

import os
import re
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import MAX_UPLOAD_BYTES
from .lineart import imgio

# เว็บที่มีนโยบาย CORS เข้มจะไม่ยอมให้เบราว์เซอร์อ่านไฟล์ข้ามโดเมน
# เซิร์ฟเวอร์จึงต้องเป็นคนโหลด แล้วส่งข้อมูลกกลับในรูปแบบ base64
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

PAGE_FETCH_TIMEOUT = 20
IMAGE_FETCH_TIMEOUT = 30

# กันเว็บเป้าหมายส่ง HTML ร้อยหลักมาให้ เก็บเฉพาะส่วนที่อ่านเป็นหน้าเว็บได้
MAX_PAGE_BYTES = 8 * 1024 * 1024

# นามสกุลที่เป็นรูปภาพจริง ถ้าไม่ระบุจะตัดออก
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".svg", ".tiff")

# รูปที่เล็กเกินไปมักเป็นไอคอน โลโก้ หรือภาพตกแต่ง ไม่ใช่ภาพระบายสี
# จึงข้ามไปตั้งแต่ตอนค้นหา ไม่ต้องให้ผู้ใช้กดลบทีหลัง
MIN_IMAGE_BYTES = 8 * 1024

# จำนวนลิงก์รูปสูงสุดที่คืนในครั้งเดียว
# ถ้าไม่จำกัด เว็บที่มีแกลเลอรีรูปเยอะจะทำให้หน้าเว็บค้าง
MAX_RESULTS = 60

# ลิงก์ที่ดาวน์โหลดไม่ได้ ไม่ต้องรายงานกลับผู้ใช้ทุกอัน
# เว็บมักมีรูปตัวยัง tracking pixel หรือไอคอนที่ 404 อยู่เสมอ
SKIP_URL_PATTERNS = (
    "data:image",
    "javascript:",
    "mailto:",
    "spacer",
    "blank.gif",
    "1x1",
    "pixel",
    "logo",
    "icon",
    "favicon",
    "avatar",
    "banner",
    "ads",
    "doubleclick",
    "facebook.com",
    "twitter.com",
    "gravatar",
)

# Pinterest ซ่อนรูปไว้ใน JSON ก้อนใหญ่และแอตทริบิวต์ imageSrcSet ของ React
# ไม่ได้อยู่ในแท็ก img ปกติเสมอไป จึงต้องสแกน HTML ดิบด้วย regex เพิ่มจากการแยกแท็กเดิม
# path หลังขนาดลึกได้ตามแฮชของแต่ละรูป เช่น /736x/f7/a9/0a/ชื่อ.jpg ลึก 4 ระดับ
PINIMG_PATTERN = re.compile(
    r"https://i\.pinimg\.com/(?:originals|\d+x)/[^\s\"'<>\\)]+?\.(?:jpg|jpeg|png|webp)",
    re.IGNORECASE,
)

# ขนาดย่อที่ Pinterest ใช้ ต้องย้ายไปขนาดใหญ่กว่าเสมอ
# 736x เป็นขนาดที่มีแทบทุกรูป (originals บางรูปไม่มีและจะ 403)
PINIMG_TARGET_SIZE = "736x"

# หน้าค้นหาของ Pinterest โหลด pin ด้วย JavaScript เมื่อเปิดด้วยเบราว์เซอร์ทั่วไป
# HTML ที่เซิร์ฟเวอร์ส่งให้จึงแทบไม่มีรูป แต่ถ้าแอบอ้างว่าเป็นแอปของ Pinterest เอง
# เซิร์ฟเวอร์จะส่งหน้าแบบ render เสร็จที่มีรูปฝังอยู่ทุก pin กลับมา
PINTEREST_UA = "Pinterest/Android"

# แมพ format ของ PIL เป็น MIME type และ extension
FORMAT_TO_MIME_AND_EXT = {
    "JPEG": ("image/jpeg", ".jpg"),
    "JPG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
    "GIF": ("image/gif", ".gif"),
    "BMP": ("image/bmp", ".bmp"),
    "TIFF": ("image/tiff", ".tiff"),
}


def detect_image_format(data: bytes) -> tuple[str, str]:
    """ตรวจจับรูปแบบของรูปภาพและคืน (mime_type, extension)
    
    ใช้ PIL เพื่ออ่านหัวไฟล์ (แค่ header ไม่ต้องถอดรูปทั้งหมด)
    ถ้าตรวจจับไม่ได้จะคืนค่าเริ่มต้นเป็น JPEG
    """
    from PIL import Image
    import io
    
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").upper()
            mime_type, ext = FORMAT_TO_MIME_AND_EXT.get(fmt, ("image/jpeg", ".jpg"))
            return mime_type, ext
    except Exception:
        # ถ้าตรวจจับไม่ได้ ให้ใช้ค่าเริ่มต้น JPEG
        return "image/jpeg", ".jpg"


def upgrade_pinimg(url: str) -> str:
    """ย้ายรูปของ Pinterest ไปยังขนาดใหญ่ที่สุดที่มีให้ดาวน์โหลด"""
    if "/originals/" in url:
        return url
    return re.sub(r"/(?:\d+x)/", f"/{PINIMG_TARGET_SIZE}/", url)


def extract_pinimg_urls(html: str) -> list[str]:
    """ดึงลิงก์รูป Pinterest จาก HTML ดิบ รวมที่ซ่อนใน JSON"""
    found: list[str] = []
    seen: set[str] = set()
    for match in PINIMG_PATTERN.findall(html):
        upgraded = upgrade_pinimg(match)
        if upgraded not in seen:
            seen.add(upgraded)
            found.append(upgraded)
    return found


class WebImageError(ValueError):
    """เกิดข้อผิดพลาดตอนดึงรูปจากเว็บไซต์"""


def validate_url(url: str) -> str:
    """ตรวจว่า URL ที่ผู้ใช้ใส่มาใช้ดึงรูปได้จริง

    เซิร์ฟเวอร์เป็นคนไปโหลดแทนผู้ใช้ จึงต้องกันไม่ให้ยิงไปที่เครือข่ายภายใน
    มิฉะนั้นผู้ใช้จะใช้เซิร์ฟเวอร์นี้เป็นทางผ่านไปยังระบบอื่นในองค์กรได้
    """
    url = url.strip()
    if not url:
        raise WebImageError("กรุณาใส่ลิงก์เว็บไซต์")

    # ไม่มีโปรโตคอลในสิ่งที่ผู้ใช้พิมพ์ เช่นพิมพ์ว่า example.com
    # ให้เติมให้เองดีกว่าบอกให้เขาเขียนเอง ส่วนใหญ่เขาคิดว่าใส่ได้เลย
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise WebImageError("ลิงก์ต้องขึ้นต้นด้วย http หรือ https")
    if not parsed.netloc:
        raise WebImageError("ลิงก์ไม่มีชื่อเว็บไซต์")

    return url


def _is_probably_private(host: str) -> bool:
    """เดาโดเมนที่ชี้ไปยังเครือข่ายภายใน

    การตรวจ DNS จริงแม่นกว่ามาก แต่ต้องเสียเวลารอ จึงใช้การเดาจากรูปแบบชื่อ
    ซึ่งกันกรณีพื้นฐานได้ ความปลอดภัยจริงยังต้องมาจากการตรวจหลัง resolve
    """
    host = host.lower().strip("[]")
    if host in ("localhost", "localhost.localdomain"):
        return True
    if host.startswith(("127.", "0.", "10.", "192.168.", "169.254.")):
        return True
    if re.match(r"^172\.(1[6-9]|2\d|3[01])\.", host):
        return True
    # โดเมนที่ลงท้ายด้วย .local หรือ .internal มักเป็นเครือข่ายภายใน
    if host.endswith((".local", ".internal", ".lan", ".home")):
        return True
    return False


def _looks_like_image(url: str) -> bool:
    """เดาว่าลิงก์นี้ชี้ไปยังไฟล์รูปภาพหรือไม่

    หลายเว็บไม่ได้ตั้งนามสกุลไว้ จึงต้องตัดสินใจอย่างอื่น
    แต่การเดาผิดว่าเป็นรูปทำให้ต้องดาวน์โหลดแล้วค่อยพัง ซึ่งช้ากว่าการข้าม
    """
    path = urlparse(url).path.lower()
    
    # URLs ที่มี size descriptors มักเป็น incomplete URLs หรือ thumbnail APIs
    # เช่น "/image/20px" หรือ "/250px-name.svg" หรือ "/thumb" ให้ข้าม
    # ต้องเช็คก่อนเช็ค image extension เพราะ Wikimedia URLs อาจมี .png ที่ท้ายแต่ยังคงเป็น incomplete
    if re.search(r"(thumbnail|thumb|/\d+x[^/]*$|\d+w?px(?:[-/]|$))", path):
        return False
    
    # ตรวจสอบนามสกุล - ถ้าเป็นนามสกุลรูปให้ผ่าน
    if path.endswith(IMAGE_EXTENSIONS):
        return True
    
    # ถ้ามีนามสกุลอื่นที่ไม่ใช่รูป ให้ข้าม (เช่น .php, .html, .css)
    if re.search(r"\.[a-z0-9]{2,5}$", path) and not path.endswith("/"):
        return False
    
    # ถ้า URL ไม่มีนามสกุลและไม่ใช่ incomplete URL ให้ลองดาวน์โหลด
    # (เว็บบางแห่งมี CDN ที่ไม่ระบุนามสกุลแต่ส่งรูป)
    return True


def _clean_name(url: str, index: int) -> str:
    """ตั้งชื่อไฟล์จาก URL เพื่อใช้เป็นชื่อกำกับเริ่มต้น

    URL ของเว็บส่วนใหญ่ไม่มีชื่อที่มีความหมาย อย่าง /uploads/a1b2c3.jpg
    จึงต้องมีชื่อกำกับสำรองไว้
    """
    basename = unquote(os.path.basename(urlparse(url).path))
    stem = re.sub(r"\.[a-z0-9]+$", "", basename, flags=re.IGNORECASE)
    stem = re.sub(r"[^\w\-.]", "_", stem).strip("._-")
    if not stem or len(stem) < 2:
        return f"รูปที่ {index + 1}"
    return stem[:60]


def extract_image_urls(html: str, page_url: str) -> list[str]:
    """แยกลิงก์รูปออกจากโค้ด HTML

    เว็บสมัยใหม่ซ่อนรูปไว้หลายที่ ทั้งในแท็ก img ปกติ
    ใน srcset ที่มีหลายความละเอียด ใน data-src ของระบบ lazy load
    และใน background-image ของ CSS จึงต้องไล่ดูทุกทาง
    """
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        """เพิ่มลิงก์รูปถ้าผ่านการกรอง"""
        candidate = candidate.strip()
        if not candidate:
            return
        try:
            absolute = urljoin(page_url, candidate)
        except ValueError:
            return
        # รูปของ Pinterest ถูกอ้างหลายขนาดในหน้าเดียว ยกเป็นขนาดใหญ่ตั้งแต่ตรงนี้
        if "i.pinimg.com" in absolute:
            absolute = upgrade_pinimg(absolute)
        if not absolute.startswith(("http://", "https://")):
            return
        if absolute in seen:
            return

        lowered = absolute.lower()
        if any(pattern in lowered for pattern in SKIP_URL_PATTERNS):
            return
        if not _looks_like_image(absolute):
            return
        host = urlparse(absolute).netloc
        if _is_probably_private(host):
            return

        seen.add(absolute)
        found.append(absolute)

    # <img> ทั้งแบบธรรมดาและแบบ lazy load
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy-src", "data-echo"):
            add(img.get(attr) or "")

        # srcset มีหลายความละเอียดคั่นด้วยจุลภาค
        for attr in ("srcset", "data-srcset"):
            value = img.get(attr)
            if not value:
                continue
            for part in value.split(","):
                url = part.strip().split(" ")[0]
                if url:
                    add(url)

    # <picture> <source> เผื่อเว็บเสิร์ฟ WebP หรือ AVIF
    for source in soup.find_all("source"):
        for attr in ("srcset", "data-srcset", "src"):
            value = source.get(attr)
            if not value:
                continue
            for part in str(value).split(","):
                url = part.strip().split(" ")[0]
                if url:
                    add(url)

    # background-image ใน inline style เช่น style="background-image:url(...)"
    for tag in soup.find_all(style=True):
        for match in re.findall(r"url\(['\"]?(.*?)['\"]?\)", tag["style"], re.IGNORECASE):
            add(match)

    # ลิงก์ตรงไปยังไฟล์รูป เช่นคลิกขวาแล้วเปิดภาพใหม่
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.lower().split("?")[0].endswith(IMAGE_EXTENSIONS):
            add(href)

    # รูปของ Pinterest ที่ฝังใน JSON ของหน้าเว็บ
    # ผ่านการกรองชุดเดียวกับรูปอื่น ๆ โดยใช้ path ของหน้าเป็นฐาน
    for pin_url in extract_pinimg_urls(html):
        add(pin_url)

    return _dedupe_pins(found)[:MAX_RESULTS]


def _dedupe_pins(found: list[str]) -> list[str]:
    """รวมรูป Pinterest ที่เป็น pin เดียวกันให้เหลือขนาดดีที่สุดหนึ่งลิงก์

    หน้าเดียวอ้างรูปเดียวกันทั้ง 236x 474x 736x และ originals
    ยกขนาดให้ 736x หมดแล้วจึงเหลือแค่คู่ 736x กับ originals ที่ชื่อไฟล์เดียวกัน
    เก็บ originals ไว้เสมอเพราะเป็นไฟล์ต้นฉบับคุณภาพสูงสุด
    (ชื่อ hash เดียวกันอาจต่างนามสกุล jpg กับ png จึงต้องตัดนามสกุลออกก่อนเทียบ)
    """
    best: dict[str, str] = {}
    final: list[str] = []
    for url in found:
        if "i.pinimg.com" not in url:
            final.append(url)
            continue
        basename = os.path.basename(urlparse(url).path)
        key = os.path.splitext(basename)[0]
        prior = best.get(key)
        if prior is None:
            best[key] = url
            final.append(url)
        elif "/originals/" in url and "/originals/" not in prior:
            final[final.index(prior)] = url
            best[key] = url
    return final


def fetch_page(url: str, user_agent: str = USER_AGENT) -> str:
    """โหลด HTML ของหน้าเว็บ

    ต้องจำกัดขนาดไว้ เพราะบางเว็บส่งหน้าเว็บกลับมาหลายสิบเมกะไบต์
    ซึ่งกินหน่วยความจำของ Render จนบริการอื่นใช้ไม่ได้
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"},
            timeout=PAGE_FETCH_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise WebImageError("เว็บไซต์ตอบสนองช้าเกินไป ลองใช้เว็บอื่นหรือลองอีกครั้ง") from exc
    except requests.RequestException as exc:
        raise WebImageError(f"เข้าถึงเว็บไซต์ไม่ได้: {exc}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower() and "xml" not in content_type.lower():
        raise WebImageError("ลิงก์นี้ไม่ใช้หน้าเว็บ ให้ใส่ลิงก์หน้าเว็บที่มีรูปภาพ")

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > MAX_PAGE_BYTES:
            break
        chunks.append(chunk)

    try:
        return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
    finally:
        response.close()


def list_images(url: str) -> list[dict]:
    """คืนรายการรูปในหน้าเว็บ พร้อมชื่อที่จะใช้เป็นชื่อกำกับ

    คืนแค่ลิงก์ ไม่ดาวน์โหลดรูปมาทั้งหมด
    เพราะเว็บหนึ่งหน้ามีรูปได้หลายร้อยรูป
    การโหลดมาทั้งหมดจะทำให้เซิร์ฟเวอร์ทำงานช้าหรือค้าง
    """
    validated = validate_url(url)
    host = urlparse(validated).netloc
    if _is_probably_private(host):
        raise WebImageError("ไม่สามารถดึงรูปจากเครือข่ายภายในได้")

    # ลิงก์ที่ชี้ไปยังไฟล์รูปตรง ๆ ใช้ได้ทันที ไม่ต้องโหลดหน้าเว็บ
    if urlparse(validated).path.lower().endswith(IMAGE_EXTENSIONS):
        return [{"url": validated, "name": _clean_name(validated, 0)}]

    # Pinterest ส่งหน้าแบบ render เสร็จพร้อมรูปให้เฉพาะแอปของตัวเอง
    # จึงต้องขอด้วย UA แบบแอป ถ้ายังไม่เจอรูปค่อยลองด้วย UA เบราว์เซอร์ทั่วไป
    is_pinterest = host.lower().endswith("pinterest.com")
    html = fetch_page(validated, user_agent=PINTEREST_UA if is_pinterest else USER_AGENT)
    urls = extract_image_urls(html, validated)
    if not urls and is_pinterest:
        html = fetch_page(validated)
        urls = extract_image_urls(html, validated)

    if not urls:
        raise WebImageError(
            "ไม่พบรูปภาพในหน้าเว็บนี้ "
            "รูปนี้เล็กเกินไป น่าจะเป็นไอคอนหรือภาพตกแต่ง"
        )

    return [
        {"url": image_url, "name": _clean_name(image_url, index)}
        for index, image_url in enumerate(urls)
    ]


def download_image(url: str) -> tuple[bytes, str]:
    """ดาวน์โหลดรูปหนึ่งไฟล์ คืน (bytes, ชื่อไฟล์)

    ต้องดาวน์โหลดทางเซิร์ฟเวอร์ ไม่ใช่ฝั่งเบราว์เซอร์
    เพราะเบราว์เซอร์ถูกกฎ CORS ของเว็บเป้าหมายกันไว้ จะอ่านข้ามโดเมนไม่ได้
    """
    validated = validate_url(url)
    host = urlparse(validated).netloc
    if _is_probably_private(host):
        raise WebImageError("ไม่สามารถดึงรูปจากเครือข่ายภายในได้")

    try:
        response = requests.get(
            validated,
            headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*"},
            timeout=IMAGE_FETCH_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise WebImageError("ดาวน์โหลดรูปไม่สำเร็จ เว็บตอบช้าเกินไป") from exc
    except requests.RequestException as exc:
        raise WebImageError(f"ดาวน์โหลดรูปไม่สำเร็จ: {exc}") from exc

    try:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            # หยุดทันทีที่เกิน ไม่ใช่หลังอ่านครบแล้วค่อยเช็ค
            # เพราะรูปจากเว็บทั่วไปไม่เกิน 25 MB การอ่านเกินไปเป็นการเสียหน่วยความจำเปล่า
            if total > MAX_UPLOAD_BYTES:
                raise WebImageError(
                    f"รูปใหญ่เกินไป (สูงสุด {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
                )
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        response.close()

    if len(raw) < MIN_IMAGE_BYTES:
        raise WebImageError("รูปนี้เล็กเกินไป น่าจะเป็นไอคอนหรือภาพตกแต่ง")

    # ตรวจว่าเป็นภาพจริงด้วย ไม่ใช่หน้าเว็บ HTML ที่สุ่มนามสกุลมา .jpg
    # เว็บหลายแห่งคืน HTML 404 มาพร้อม Content-Type ที่ไม่ตรง ถ้าไม่เช็คไว้
    # รูปนี้จะเข้าไปถึงขั้นตอนวาด PDF แล้วล้มทั้งเล่ม
    try:
        imgio.imdecode(raw)
    except imgio.ImageLoadError as exc:
        raise WebImageError("ลิงก์นี้ไม่ได้ชี้ไปยังไฟล์รูปภาพ") from exc

    return raw, _clean_name(validated, 0)
