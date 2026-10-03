import io
import os
import re
import time
from urllib.parse import unquote, urljoin, urlparse
import zipfile
from bs4 import BeautifulSoup
import requests
import streamlit as st

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Web Image Extractor Pro",
    page_icon=":material/photo_library:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal modern styling เพื่อให้เส้นขอบ card และ gallery สวยคมชัด
st.markdown(
    """
<style>
    /* Card styling */
    .stCard {
        border-radius: 12px;
        border: 1px solid var(--border-color, #E2E8F0);
        padding: 1.25rem;
        background-color: var(--secondary-background-color, #FFFFFF);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    /* Stat box */
    .metric-card {
        background: linear-gradient(135deg, rgba(37,99,235,0.05), rgba(59,130,246,0.02));
        border: 1px solid rgba(37,99,235,0.15);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    /* Image thumbnail card */
    .img-card {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
        background: #fff;
        padding: 6px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .img-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_clean_filename(url: str, index: int, default_ext: str = ".jpg") -> str:
    """สร้างชื่อไฟล์ที่ปลอดภัยและไม่ซ้ำ"""
    parsed = urlparse(url)
    basename = os.path.basename(parsed.path)
    clean_name = unquote(basename).split("?")[0].strip()

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".bmp")
    if not clean_name or not clean_name.lower().endswith(valid_extensions):
        clean_name = f"image_{index:03d}{default_ext}"
    else:
        name_part, ext = os.path.splitext(clean_name)
        name_part = re.sub(r"[^\w\-.]", "_", name_part)[:40]
        clean_name = f"{index:03d}_{name_part}{ext}"

    return clean_name


def extract_images_requests(
    url: str, headers: dict, timeout: int = 15
) -> list[str]:
    """ดึง image URLs ด้วย requests (รวดเร็วและเบา ไม่ต้องพึ่งพา browser)"""
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    found_urls = set()

    # 1. ค้นหาจาก <img> tags
    for img in soup.find_all("img"):
        for attr in [
            "src",
            "data-src",
            "data-original",
            "data-lazy-src",
            "srcset",
            "data-srcset",
        ]:
            val = img.get(attr)
            if not val:
                continue
            # จัดการ srcset (อาจมีหลาย url คั่นด้วย comma)
            if "srcset" in attr:
                parts = [p.strip().split()[0] for p in val.split(",") if p.strip()]
                for p in parts:
                    abs_url = urljoin(url, p)
                    if abs_url.startswith(("http://", "https://")):
                        found_urls.add(abs_url)
            else:
                abs_url = urljoin(url, val.strip())
                if abs_url.startswith(("http://", "https://")):
                    found_urls.add(abs_url)

    # 2. ค้นหาจาก <picture> <source>
    for source in soup.find_all("source"):
        for attr in ["srcset", "data-srcset", "src"]:
            val = source.get(attr)
            if val:
                parts = [p.strip().split()[0] for p in val.split(",") if p.strip()]
                for p in parts:
                    abs_url = urljoin(url, p)
                    if abs_url.startswith(("http://", "https://")):
                        found_urls.add(abs_url)

    # 3. ค้นหาจาก <a> tag ที่ลิงก์ไปรูปภาพตรงๆ
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
        ):
            abs_url = urljoin(url, href)
            found_urls.add(abs_url)

    # 4. ค้นหาจาก background-image ใน inline style
    for tag in soup.find_all(style=True):
        style_val = tag["style"]
        matches = re.findall(r"url\(['\"]?(.*?)['\"]?\)", style_val, re.IGNORECASE)
        for m in matches:
            if m and not m.startswith("data:"):
                abs_url = urljoin(url, m.strip())
                if abs_url.startswith(("http://", "https://")):
                    found_urls.add(abs_url)

    return list(found_urls)


def extract_images_selenium(
    url: str, use_scroll: bool, scroll_count: int = 5
) -> tuple[list[str], str | None]:
    """ดึง image URLs ด้วย Selenium สำหรับเว็บที่ต้องการ render JavaScript / Dynamic DOM"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except ImportError:
        return [], "ไม่พบไลบรารี Selenium ในระบบ"

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        driver.get(url)
        time.sleep(3)

        if use_scroll:
            for _ in range(scroll_count):
                driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )
                time.sleep(1.5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        found_urls = set()

        for img in soup.find_all("img"):
            for attr in [
                "src",
                "data-src",
                "data-original",
                "data-lazy-src",
                "srcset",
            ]:
                val = img.get(attr)
                if val:
                    if "srcset" in attr:
                        parts = [
                            p.strip().split()[0]
                            for p in val.split(",")
                            if p.strip()
                        ]
                        for p in parts:
                            abs_url = urljoin(url, p)
                            if abs_url.startswith(("http://", "https://")):
                                found_urls.add(abs_url)
                    else:
                        abs_url = urljoin(url, val.strip())
                        if abs_url.startswith(("http://", "https://")):
                            found_urls.add(abs_url)

        return list(found_urls), None
    except Exception as e:
        return [], str(e)
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


# ----------------- UI Sidebar Settings -----------------
with st.sidebar:
    st.markdown("### :material/tune: การตั้งค่า (Settings)")
    engine_choice = st.segmented_control(
        "เครื่องมือดึงข้อมูล",
        options=["Requests (เร็ว/มาตรฐาน)", "Selenium (เว็บไดนามิก)"],
        default="Requests (เร็ว/มาตรฐาน)",
    )

    with st.expander("ตัวเลือกเพิ่มเติม (Advanced)", expanded=True):
        min_size_kb = st.number_input(
            "ขนาดไฟล์ขั้นต่ำ (KB)",
            min_value=0,
            max_value=2000,
            value=5,
            help="กรองพวกไอคอนเล็กๆ หรือ pixel tracker ทิ้ง",
        )
        max_images = st.slider(
            "จำกัดจำนวนรูปสูงสุด", min_value=10, max_value=200, value=60, step=10
        )
        use_scroll = st.checkbox(
            "เลื่อนหน้าจออัตโนมัติ (Infinite Scroll)",
            value=True,
            disabled=(engine_choice == "Requests (เร็ว/มาตรฐาน)"),
            help="ใช้ได้เฉพาะโหมด Selenium",
        )
        scroll_steps = st.slider(
            "จำนวนครั้งที่เลื่อนจอ",
            min_value=1,
            max_value=15,
            value=4,
            disabled=not use_scroll or (engine_choice == "Requests (เร็ว/มาตรฐาน)"),
        )

    st.markdown("---")
    st.caption("พัฒนาด้วย Streamlit | รองรับ JPG, PNG, WEBP, GIF, SVG")


# ----------------- Main UI Header -----------------
with st.container(border=True):
    col_icon, col_title = st.columns([1, 11], vertical_alignment="center")
    with col_icon:
        st.markdown(
            "## :material/photo_library:",
            help="Image Scraper",
        )
    with col_title:
        st.subheader("โปรแกรมดูดรูปภาพจากเว็บไซต์ (Web Image Extractor)")
        st.caption(
            "ดึงรูปภาพจากหน้าเว็บได้รวดเร็ว รองรับ Lazy Load, กรองขนาดภาพ, แสดงพรีวิว และดาวน์โหลดเป็นไฟล์ ZIP ได้ทันที"
        )

# URL Input Bar
with st.container(border=True):
    col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_input:
        target_url = st.text_input(
            "ระบุ URL เว็บไซต์ที่ต้องการดึงรูปภาพ",
            placeholder="https://example.com หรือเว็บไซต์ที่คุณต้องการ",
            value="https://en.wikipedia.org/wiki/Cat",
        )
    with col_btn:
        start_scrape = st.button(
            "เริ่มดึงรูป",
            type="primary",
            icon=":material/download:",
            width="stretch",
        )

# Initialize Session State สำหรับเก็บผลลัพธ์
if "scraped_results" not in st.session_state:
    st.session_state.scraped_results = []
if "zip_buffer" not in st.session_state:
    st.session_state.zip_buffer = None
if "last_url" not in st.session_state:
    st.session_state.last_url = ""


# ----------------- Scrape Logic -----------------
if start_scrape:
    if not target_url or not target_url.startswith(("http://", "https://")):
        st.error(
            "กรุณาระบุ URL ที่ถูกต้อง โดยขึ้นต้นด้วย http:// หรือ https://",
            icon=":material/error:",
        )
    else:
        st.session_state.scraped_results = []
        st.session_state.zip_buffer = None
        st.session_state.last_url = target_url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,th;q=0.8",
        }

        with st.status(
            "กำลังเชื่อมต่อและค้นหารูปภาพ...", expanded=True
        ) as status:
            img_urls = []
            if "Selenium" in engine_choice:
                st.write(":material/sync: กำลังเปิด Browser เสมือนเพื่อ Render หน้าเว็บ...")
                urls, error_msg = extract_images_selenium(
                    target_url, use_scroll=use_scroll, scroll_count=scroll_steps
                )
                if error_msg:
                    st.warning(
                        f"Selenium แจ้งเตือน: {error_msg} (กำลังสลับไปใช้ระบบ Requests ให้แทนอัตโนมัติ)"
                    )
                    try:
                        urls = extract_images_requests(target_url, headers=headers)
                    except Exception as ex:
                        st.error(f"เกิดข้อผิดพลาด: {ex}")
                        urls = []
                img_urls = urls
            else:
                st.write(":material/search: กำลังสแกนหาแท็กรูปภาพบนหน้าเว็บ...")
                try:
                    img_urls = extract_images_requests(target_url, headers=headers)
                except Exception as e:
                    st.error(f"ไม่สามารถเข้าถึงเว็บไซต์ได้: {e}")
                    img_urls = []

            # กรองและจำกัดจำนวนรูป
            unique_urls = list(dict.fromkeys(img_urls))[:max_images]
            total_found = len(unique_urls)
            st.write(f":material/check_circle: พบรูปภาพทั้งหมด {total_found} รูป")

            if total_found == 0:
                status.update(
                    label="ไม่พบรูปภาพบนหน้านี้ หรือเว็บไซต์มีการป้องกัน",
                    state="error",
                )
            else:
                st.write(":material/downloading: กำลังดาวน์โหลดและตรวจสอบขนาดรูปภาพ...")
                progress_bar = st.progress(0)

                downloaded_items = []
                zip_bytes_io = io.BytesIO()

                with zipfile.ZipFile(
                    zip_bytes_io, mode="w", compression=zipfile.ZIP_DEFLATED
                ) as zip_file:
                    for idx, img_url in enumerate(unique_urls, 1):
                        progress_bar.progress(idx / total_found)
                        try:
                            req_headers = headers.copy()
                            req_headers["Referer"] = target_url
                            img_resp = requests.get(
                                img_url,
                                headers=req_headers,
                                timeout=10,
                                stream=True,
                            )
                            if img_resp.status_code == 200:
                                content = img_resp.content
                                size_kb = len(content) / 1024.0

                                # กรองขนาดขั้นต่ำ
                                if size_kb >= min_size_kb:
                                    fname = get_clean_filename(img_url, idx)
                                    zip_file.writestr(fname, content)
                                    downloaded_items.append(
                                        {
                                            "name": fname,
                                            "size_kb": f"{size_kb:.1f} KB",
                                            "bytes": content,
                                            "url": img_url,
                                        }
                                    )
                        except Exception:
                            continue

                st.session_state.scraped_results = downloaded_items
                zip_bytes_io.seek(0)
                st.session_state.zip_buffer = zip_bytes_io.getvalue()

                status.update(
                    label=f"ประมวลผลเสร็จสิ้น! บันทึกสำเร็จ {len(downloaded_items)} รูปภาพ",
                    state="complete",
                )


# ----------------- Results Display & Gallery -----------------
results = st.session_state.scraped_results
if results:
    st.space("small")

    # Overview Metrics Row
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("จำนวนรูปที่ดาวน์โหลด", f"{len(results)} รูป")
        m2.metric(
            "เว็บไซต์ต้นทาง",
            urlparse(st.session_state.last_url).netloc or "Unknown",
        )
        total_mb = sum(len(item["bytes"]) for item in results) / (1024 * 1024)
        m3.metric("ขนาดรวมทั้งหมด", f"{total_mb:.2f} MB")
        with m4:
            st.space(5)
            if st.session_state.zip_buffer:
                st.download_button(
                    label="ดาวน์โหลดทั้งหมด (.ZIP)",
                    data=st.session_state.zip_buffer,
                    file_name="downloaded_images.zip",
                    mime="application/zip",
                    type="primary",
                    icon=":material/archive:",
                    width="stretch",
                )

    st.space("small")

    # Gallery View
    st.subheader(
        ":material/grid_view: แกลเลอรีรูปภาพที่ดึงได้ (Image Gallery)"
    )

    # แบ่งเป็นแถวละ 4 รูป อย่างสวยงามและเป็นระเบียบ
    cols_per_row = 4
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            img_index = i + col_idx
            if img_index < len(results):
                item = results[img_index]
                with cols[col_idx]:
                    with st.container(border=True):
                        st.image(
                            item["bytes"],
                            caption=f"{item['name']} ({item['size_kb']})",
                            width="stretch",
                        )
                        st.download_button(
                            label="ดาวน์โหลดรูปนี้",
                            data=item["bytes"],
                            file_name=item["name"],
                            key=f"dl_btn_{img_index}",
                            width="stretch",
                            icon=":material/file_download:",
                        )

elif st.session_state.last_url and not results:
    st.info(
        "ไม่พบรูปภาพที่ตรงตามเงื่อนไข (ลองลด 'ขนาดไฟล์ขั้นต่ำ (KB)' หรือเปลี่ยนไปใช้โหมด Selenium ในแถบด้านข้าง)",
        icon=":material/info:",
    )
