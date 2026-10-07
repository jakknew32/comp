import io
import os
import re
import sys
import time
from urllib.parse import unquote, urljoin, urlparse
import zipfile
from bs4 import BeautifulSoup
import cv2
import numpy as np
from PIL import Image
import requests
import streamlit as st

# เพิ่ม path ให้เรียกใช้ app package จาก repo comp ได้
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# นำเข้า core engine สำหรับทำสมุดระบายสี A4 จาก repo
HAS_COLORING_ENGINE = False
try:
    from app.book import pdf as book_pdf
    from app.config import BookParams, LineArtParams
    from app.lineart import convert, imgio
    HAS_COLORING_ENGINE = True
except Exception as e:
    HAS_COLORING_ENGINE = False
    coloring_engine_error = str(e)

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Web Scraper to A4 Coloring Book",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling ให้ขอบ เส้น และการ์ด คมชัด สวยงาม ทันสมัย
st.markdown(
    """
<style>
    .stCard {
        border-radius: 12px;
        border: 1px solid var(--border-color, #E2E8F0);
        padding: 1.25rem;
        background-color: var(--secondary-background-color, #FFFFFF);
    }
    .step-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        color: #1E293B;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_clean_filename(url: str, index: int, default_ext: str = ".jpg") -> str:
    parsed = urlparse(url)
    basename = os.path.basename(parsed.path)
    clean_name = unquote(basename).split("?")[0].strip()
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
    if not clean_name or not clean_name.lower().endswith(valid_exts):
        clean_name = f"image_{index:03d}{default_ext}"
    else:
        name_part, ext = os.path.splitext(clean_name)
        name_part = re.sub(r"[^\w\-.]", "_", name_part)[:40]
        clean_name = f"{index:03d}_{name_part}{ext}"
    return clean_name


def extract_images_requests(
    url: str, headers: dict, timeout: int = 15
) -> list[str]:
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    found_urls = set()

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

    for source in soup.find_all("source"):
        for attr in ["srcset", "data-srcset", "src"]:
            val = source.get(attr)
            if val:
                parts = [p.strip().split()[0] for p in val.split(",") if p.strip()]
                for p in parts:
                    abs_url = urljoin(url, p)
                    if abs_url.startswith(("http://", "https://")):
                        found_urls.add(abs_url)

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
        ):
            abs_url = urljoin(url, href)
            found_urls.add(abs_url)

    return list(found_urls)


def extract_images_selenium(
    url: str, use_scroll: bool, scroll_count: int = 4
) -> tuple[list[str], str | None]:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        return [], "ไม่พบไลบรารี Selenium ในระบบ"

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
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


# ----------------- Session State -----------------
if "scraped_results" not in st.session_state:
    st.session_state.scraped_results = []
if "zip_buffer" not in st.session_state:
    st.session_state.zip_buffer = None
if "selected_images" not in st.session_state:
    st.session_state.selected_images = set()
if "pdf_buffer" not in st.session_state:
    st.session_state.pdf_buffer = None


# ----------------- Sidebar Options -----------------
with st.sidebar:
    st.markdown("### :material/auto_stories: ตั้งค่าสมุดระบายสี (Book Settings)")
    book_title = st.text_input("ชื่อหน้าปกสมุด", value="สมุดภาพระบายสีของฉัน")
    book_author = st.text_input("ชื่อผู้จัดทำ", value="")
    per_page = st.selectbox(
        "จำนวนรูปต่อหน้า (Grid)", options=[1, 2, 4], index=0
    )
    include_cover = st.checkbox("ใส่หน้าปกสมุดระบายสี (Cover Page)", value=True)
    show_frame = st.checkbox("ใส่กรอบรูปภาพ (Page Frame)", value=True)
    show_caption = st.checkbox("แสดงชื่อภาพใต้รูป (Caption)", value=True)
    show_page_number = st.checkbox("แสดงหมายเลขหน้า (Page Number)", value=True)

    st.markdown("---")
    st.markdown("### :material/tune: การปรับเส้นระบายสี (Line Art)")
    target_line_mm = st.selectbox(
        "ความหนาเส้นเป้าหมาย (มม.)",
        options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        index=4,
    )
    strip_border = st.checkbox(
        "ตัดกรอบเดิมของภาพออกอัตโนมัติ",
        value=True,
        help="ป้องกันกรอบรูปเดิมซ้อนกับกรอบหน้ากระดาษ A4",
    )

    st.markdown("---")
    st.caption("ระบบผสาน Web Scraper และ A4 Line Art Book Engine")


# ----------------- Main Header -----------------
with st.container(border=True):
    col_icon, col_title = st.columns([1, 11], vertical_alignment="center")
    with col_icon:
        st.markdown("## :material/palette:")
    with col_title:
        st.subheader("ระบบดูดรูปภาพจากเว็บ & สร้างสมุดระบายสี A4 (Coloring Book Creator)")
        st.caption(
            "ดึงรูปภาพจากเว็บไซต์ แปลงเป็นลายเส้นขาว-ดำคมชัด และรวมเล่มเป็นไฟล์ PDF ขนาด A4 พร้อมพิมพ์ระบายสีทันที"
        )


# ----------------- Tabs: Workflow -----------------
tab_scrape, tab_convert = st.tabs(
    [
        ":material/download: ขั้นที่ 1: ดึงรูปภาพจากเว็บไซต์",
        ":material/menu_book: ขั้นที่ 2: แปลงลายเส้น & สร้างสมุด PDF",
    ]
)

with tab_scrape:
    with st.container(border=True):
        col_url, col_btn = st.columns([5, 1], vertical_alignment="bottom")
        with col_url:
            target_url = st.text_input(
                "ใส่ URL เว็บไซต์ที่ต้องการดึงรูปภาพ",
                value="https://en.wikipedia.org/wiki/Cat",
                placeholder="https://example.com",
            )
        with col_btn:
            start_scrape = st.button(
                "ดึงรูปภาพ",
                type="primary",
                icon=":material/download:",
                width="stretch",
            )

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            engine_choice = st.selectbox(
                "เอนจิ้นดึงข้อมูล",
                ["Requests (เร็ว/มาตรฐาน)", "Selenium (เว็บไดนามิก)"],
            )
        with col_opt2:
            min_size_kb = st.number_input(
                "ขนาดไฟล์ขั้นต่ำ (KB)", min_value=0, value=5
            )
        with col_opt3:
            max_images = st.number_input(
                "จำกัดจำนวนรูปสูงสุด", min_value=1, max_value=200, value=30
            )

    if start_scrape:
        if not target_url or not target_url.startswith(("http://", "https://")):
            st.error("กรุณาระบุ URL ให้ถูกต้อง (ขึ้นต้นด้วย http:// หรือ https://)")
        else:
            st.session_state.scraped_results = []
            st.session_state.zip_buffer = None
            st.session_state.selected_images = set()

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }

            with st.status("กำลังดึงรูปภาพ...", expanded=True) as status:
                img_urls = []
                if "Selenium" in engine_choice:
                    st.write(":material/sync: ใช้งาน Selenium...")
                    urls, err = extract_images_selenium(
                        target_url, use_scroll=True
                    )
                    if err:
                        st.warning(f"Selenium แจ้งเตือน: {err} สลับไปใช้ Requests")
                        try:
                            urls = extract_images_requests(
                                target_url, headers=headers
                            )
                        except Exception:
                            urls = []
                    img_urls = urls
                else:
                    st.write(":material/search: สแกนหน้าเว็บด้วย Requests...")
                    try:
                        img_urls = extract_images_requests(
                            target_url, headers=headers
                        )
                    except Exception as e:
                        st.error(f"ไม่สามารถเข้าถึงหน้าเว็บได้: {e}")
                        img_urls = []

                unique_urls = list(dict.fromkeys(img_urls))[:max_images]
                total = len(unique_urls)
                st.write(f":material/check_circle: พบทั้งหมด {total} ภาพ")

                if total == 0:
                    status.update(label="ไม่พบรูปภาพ", state="error")
                else:
                    progress = st.progress(0)
                    downloaded = []
                    zip_buf = io.BytesIO()

                    with zipfile.ZipFile(
                        zip_buf, "w", zipfile.ZIP_DEFLATED
                    ) as zf:
                        for idx, url in enumerate(unique_urls, 1):
                            progress.progress(idx / total)
                            try:
                                resp = requests.get(
                                    url,
                                    headers={
                                        **headers,
                                        "Referer": target_url,
                                    },
                                    timeout=10,
                                )
                                if resp.status_code == 200:
                                    content = resp.content
                                    size_kb = len(content) / 1024.0
                                    if size_kb >= min_size_kb:
                                        fname = get_clean_filename(url, idx)
                                        zf.writestr(fname, content)
                                        downloaded.append(
                                            {
                                                "id": idx,
                                                "name": fname,
                                                "size_kb": f"{size_kb:.1f} KB",
                                                "bytes": content,
                                                "url": url,
                                            }
                                        )
                            except Exception:
                                pass

                    st.session_state.scraped_results = downloaded
                    zip_buf.seek(0)
                    st.session_state.zip_buffer = zip_buf.getvalue()
                    # เริ่มต้นเลือกทุกรูปให้ทำสมุดระบายสี
                    st.session_state.selected_images = {
                        item["name"] for item in downloaded
                    }
                    status.update(
                        label=f"ดาวน์โหลดสำเร็จ {len(downloaded)} รูป!",
                        state="complete",
                    )

    # แสดงผล Gallery ของรูปที่ดึงได้
    results = st.session_state.scraped_results
    if results:
        with st.container(border=True):
            r1, r2, r3 = st.columns([2, 2, 2], vertical_alignment="center")
            r1.metric("จำนวนรูปทั้งหมด", f"{len(results)} รูป")
            r2.metric(
                "รูปที่เลือกไปทำสมุด",
                f"{len(st.session_state.selected_images)} รูป",
            )
            with r3:
                if st.session_state.zip_buffer:
                    st.download_button(
                        "ดาวน์โหลดรูปทั้งหมด (.ZIP)",
                        data=st.session_state.zip_buffer,
                        file_name="scraped_images.zip",
                        mime="application/zip",
                        icon=":material/archive:",
                        width="stretch",
                    )

        st.space("small")
        st.write("เลือกรูปที่ต้องการนำไปสร้างสมุดระบายสี:")

        # Action bar เลือกรวดเร็ว
        col_all, col_clear, _ = st.columns([1.5, 1.5, 5])
        with col_all:
            if st.button("เลือกทั้งหมด", icon=":material/select_all:"):
                st.session_state.selected_images = {
                    item["name"] for item in results
                }
                st.rerun()
        with col_clear:
            if st.button("ล้างการเลือก", icon=":material/deselect:"):
                st.session_state.selected_images.clear()
                st.rerun()

        # แสดง 4 คอลัมน์
        cols_per_row = 4
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                img_idx = i + c_idx
                if img_idx < len(results):
                    item = results[img_idx]
                    with cols[c_idx]:
                        with st.container(border=True):
                            st.image(
                                item["bytes"],
                                caption=f"{item['name']} ({item['size_kb']})",
                                width="stretch",
                            )
                            is_checked = item["name"] in st.session_state.selected_images
                            chk = st.checkbox(
                                "ใช้ทำสมุดระบายสี",
                                value=is_checked,
                                key=f"select_{item['name']}",
                            )
                            if chk and item["name"] not in st.session_state.selected_images:
                                st.session_state.selected_images.add(item["name"])
                                st.rerun()
                            elif not chk and item["name"] in st.session_state.selected_images:
                                st.session_state.selected_images.remove(item["name"])
                                st.rerun()


# ----------------- Tab 2: Convert to Line Art & Coloring PDF -----------------
with tab_convert:
    st.subheader(":material/draw: แปลงเป็นภาพลายเส้นและรวมเล่มเป็น PDF (A4)")

    if not HAS_COLORING_ENGINE:
        st.error(
            "ไม่พบโมดูลสร้างสมุดระบายสี (app.lineart / app.book) กรุณาตรวจสอบว่ามีโฟลเดอร์ app อยู่ในโปรเจกต์"
        )
    else:
        results = st.session_state.scraped_results
        selected_names = st.session_state.selected_images
        items_to_convert = [
            item for item in results if item["name"] in selected_names
        ]

        if not items_to_convert:
            st.info(
                "ยังไม่มีรูปภาพที่ถูกเลือก กรุณาดึงรูปภาพจาก 'ขั้นที่ 1' แล้วติ๊กเลือกรูปภาพที่ต้องการครับ",
                icon=":material/info:",
            )
        else:
            with st.container(border=True):
                c_info, c_action = st.columns(
                    [3, 1], vertical_alignment="center"
                )
                with c_info:
                    st.write(
                        f"พร้อมประมวลผลทั้งหมด **{len(items_to_convert)} รูป** เพื่อสร้างเป็นสมุดระบายสี A4"
                    )
                    st.caption(
                        f"ชื่อสมุด: **{book_title}** | จัดหน้า: **{per_page} รูป/หน้า** | ความหนาเส้น: **{target_line_mm} มม.**"
                    )
                with c_action:
                    btn_generate_pdf = st.button(
                        "สร้างสมุดระบายสี PDF",
                        type="primary",
                        icon=":material/picture_as_pdf:",
                        width="stretch",
                    )

            if btn_generate_pdf:
                with st.status(
                    "กำลังแปลงรูปเป็นภาพลายเส้นและจัดหน้า PDF...", expanded=True
                ) as pdf_status:
                    masks = []
                    captions = []
                    lineart_p = LineArtParams(
                        target_line_mm=target_line_mm,
                        strip_border=strip_border,
                    )
                    book_p = BookParams(
                        title=book_title,
                        author=book_author,
                        per_page=per_page,
                        show_frame=show_frame,
                        show_caption=show_caption,
                        show_page_number=show_page_number,
                        include_cover=include_cover,
                    )

                    progress_conv = st.progress(0)
                    for i, itm in enumerate(items_to_convert, 1):
                        progress_conv.progress(i / len(items_to_convert))
                        st.write(f":material/draw: กำลังแปลงภาพ: {itm['name']}...")
                        try:
                            # ถอดรหัสภาพเป็น BGR
                            bgr_img = imgio.imdecode(itm["bytes"])
                            # แปลงเป็นลายเส้น XDoG
                            res = convert.convert(bgr_img, lineart_p)
                            if not res.is_empty:
                                masks.append(res.mask)
                                cap = imgio.caption_from_filename(itm["name"])
                                captions.append(cap)
                        except Exception as e:
                            st.warning(f"ข้ามภาพ {itm['name']}: {e}")

                    st.write(f":material/menu_book: กำลังรวบรวมเข้าเล่ม A4 PDF ({len(masks)} หน้า)...")
                    try:
                        book_res = book_pdf.build_book(
                            masks, captions, book_p, lineart_p
                        )
                        st.session_state.pdf_buffer = book_res.pdf_bytes
                        pdf_status.update(
                            label=f"สร้างสมุดระบายสีสำเร็จ! ทั้งหมด {book_res.page_count} หน้า",
                            state="complete",
                        )
                    except Exception as e:
                        pdf_status.update(
                            label=f"เกิดข้อผิดพลาดในการสร้าง PDF: {e}",
                            state="error",
                        )

            # แสดงผลลัพธ์การสร้าง PDF
            if st.session_state.pdf_buffer:
                st.space("small")
                with st.container(border=True):
                    st.success(
                        ":material/check_circle: สมุดระบายสี A4 พร้อมดาวน์โหลดแล้ว!",
                        icon=":material/verified:",
                    )
                    pdf_size_mb = len(st.session_state.pdf_buffer) / (1024 * 1024)
                    c_meta, c_dl = st.columns([2, 1], vertical_alignment="center")
                    with c_meta:
                        st.write(f"📄 ชื่อไฟล์: **my_coloring.pdf** (ขนาด: {pdf_size_mb:.2f} MB)")
                    with c_dl:
                        st.download_button(
                            label="ดาวน์โหลดไฟล์ PDF สมุดระบายสี",
                            data=st.session_state.pdf_buffer,
                            file_name="my_coloring.pdf",
                            mime="application/pdf",
                            type="primary",
                            icon=":material/download:",
                            width="stretch",
                        )
