import base64
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
import streamlit.components.v1 as components

# เพิ่ม path ให้เรียกใช้ app package จาก repo comp ได้
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# นำเข้า core engine สำหรับทำสมุดระบายสี A4 จาก repo
HAS_COLORING_ENGINE = False
try:
    from app.book import pdf as book_pdf
    from app.config import BookParams, LineArtParams, PAGE_PREVIEW_WIDTH_PX
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

def page_to_png(image: Image.Image, width_px: int = PAGE_PREVIEW_WIDTH_PX) -> bytes:
    """ย่อหน้า A4 เป็น ~200 DPI ขาว-ดำ เพื่อให้ไฟล์เล็กพอสำหรับพรีวิวและสั่งพิมพ์"""
    gray = image.convert("L")
    height_px = round(width_px * gray.height / gray.width)
    if gray.width > width_px:
        gray = gray.resize((width_px, height_px), Image.LANCZOS)
    buf = io.BytesIO()
    gray.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def mask_thumb(mask: np.ndarray, width_px: int = 400) -> bytes:
    """ภาพย่อของลายเส้น (เส้นดำบนพื้นขาว) สำหรับแสดงในหน้าเว็บ"""
    img = Image.fromarray(255 - mask.astype(np.uint8))
    if img.width > width_px:
        img = img.resize((width_px, round(width_px * img.height / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_print_button(label: str = "🖨️ พิมพ์ทุกหน้า") -> None:
    """ปุ่มสั่งพิมพ์ทุกหน้าที่แสดงในพรีวิว (ดึงภาพจากหน้าเว็บ ไม่ต้องส่งไฟล์ใหญ่ไปกลับ)"""
    html = f"""
<style>
  html, body {{ margin: 0; font-family: "Source Sans", sans-serif; }}
  button {{
    width: 100%; padding: 0.55rem 0.75rem; font-size: 16px; font-weight: 600;
    color: #fff; background: #ff4b4b; border: 0; border-radius: 0.5rem; cursor: pointer;
  }}
  button:hover {{ background: #e03e3e; }}
  #msg {{ font-size: 12px; color: #888; margin-top: 4px; min-height: 14px; }}
</style>
<button id="printBtn" type="button">{label}</button>
<div id="msg"></div>
<script>
  const msg = document.getElementById("msg");
  let frame = null;
  function pageUrls() {{
    const doc = window.parent.document;
    let imgs = Array.from(doc.querySelectorAll(".st-key-print_pages img"));
    if (!imgs.length) imgs = Array.from(doc.querySelectorAll('[data-testid="stImage"] img'));
    return imgs.map(i => new URL(i.currentSrc || i.src, window.parent.location.href).href);
  }}
  function buildHtml(urls) {{
    return '<!doctype html><html><head><meta charset="utf-8"><style>' +
      '@page {{ size: A4; margin: 0; }}' +
      'html, body {{ margin: 0; padding: 0; background: #fff; }}' +
      '.pg {{ width: 210mm; height: 296mm; overflow: hidden; break-after: page; page-break-after: always; }}' +
      '.pg:last-child {{ break-after: auto; page-break-after: auto; }}' +
      '.pg img {{ width: 210mm; height: 296mm; object-fit: contain; display: block; }}' +
      '</style></head><body>' +
      urls.map(u => '<div class="pg"><img src="' + u + '"></div>').join("") +
      '</body></html>';
  }}
  document.getElementById("printBtn").addEventListener("click", () => {{
    let urls = [];
    try {{ urls = pageUrls(); }} catch (e) {{}}
    if (!urls.length) {{
      msg.textContent = "ไม่พบภาพหน้าสมุดสำหรับพิมพ์ กรุณาสร้างสมุดใหม่อีกครั้ง";
      return;
    }}
    msg.textContent = "กำลังเตรียมพิมพ์ " + urls.length + " หน้า...";
    if (frame) frame.remove();
    frame = document.createElement("iframe");
    frame.style.cssText = "position:fixed;right:0;bottom:0;width:1px;height:1px;border:0;opacity:0;";
    frame.onload = async () => {{
      try {{
        const w = frame.contentWindow;
        await Promise.all(Array.from(w.document.images).map(img =>
          img.complete ? Promise.resolve() :
          new Promise(r => {{ img.onload = r; img.onerror = r; }})));
        w.focus();
        w.print();
        msg.textContent = "";
      }} catch (e) {{
        msg.textContent = "สั่งพิมพ์ไม่สำเร็จ: " + e;
      }}
    }};
    frame.srcdoc = buildHtml(urls);
    document.body.appendChild(frame);
  }});
</script>
"""
    # Streamlit รุ่นใหม่ใช้ st.iframe (components.html กำลังจะถูกเลิกใช้) รุ่นเก่าถอยกลับไปใช้ของเดิม
    if hasattr(st, "iframe"):
        st.iframe(html, height=70)
    else:
        components.html(html, height=70)


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
    # ลบอักขระควบคุมและอักขระที่ไม่ปลอดภัยในชื่อไฟล์
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', clean_name)
    clean_name = re.sub(r'\s+', '_', clean_name)
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
if "page_pngs" not in st.session_state:
    st.session_state.page_pngs = None
if "converted" not in st.session_state:
    st.session_state.converted = []  # รูปที่แปลงเป็นลายเส้นแล้ว (ผลของขั้นที่ 2)
if "uid_counter" not in st.session_state:
    st.session_state.uid_counter = 0


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
            "ดึงรูปภาพจากเว็บไซต์ แปลงเป็นลายเส้นขาว-ดำคมชัด และรวมเล่มขนาด A4 แล้วสั่งพิมพ์ระบายสีได้ทันที"
        )


# ----------------- Tabs: Workflow -----------------
tab_scrape, tab_convert, tab_book = st.tabs(
    [
        ":material/download: ขั้นที่ 1: ดึงรูปภาพจากเว็บไซต์",
        ":material/draw: ขั้นที่ 2: แปลงรูปเป็นลายเส้น",
        ":material/menu_book: ขั้นที่ 3: รวมเล่ม & พิมพ์",
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


# ----------------- Tab 2: Convert images to line art -----------------
with tab_convert:
    st.subheader(":material/draw: ขั้นที่ 2: แปลงรูปเป็นภาพลายเส้น")
    st.caption(
        "แปลงครั้งเดียว ผลลัพธ์จะถูกเก็บไว้ใช้ในขั้นที่ 3 — ปรับค่าการรวมเล่มกี่ครั้งก็ไม่ต้องแปลงใหม่"
    )

    if not HAS_COLORING_ENGINE:
        st.error(
            "ไม่พบโมดูลสร้างสมุดระบายสี (app.lineart / app.book) กรุณาตรวจสอบว่ามีโฟลเดอร์ app อยู่ในโปรเจกต์"
        )
    else:
        results = st.session_state.scraped_results
        selected_names = st.session_state.selected_images
        sources = [
            (item["name"], item["bytes"])
            for item in results
            if item["name"] in selected_names
        ]

        uploads = st.file_uploader(
            "หรืออัปโหลดรูปเอง (ข้ามขั้นที่ 1 ได้)",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="upload_src",
        )
        for up in uploads or []:
            sources.append((up.name, up.getvalue()))

        if not sources:
            st.info(
                "ยังไม่มีรูปให้แปลง ไปเลือกรูปจาก 'ขั้นที่ 1' หรืออัปโหลดรูปด้านบนได้เลยครับ",
                icon=":material/info:",
            )
        else:
            with st.container(border=True):
                c_info, c_action = st.columns([3, 1], vertical_alignment="center")
                with c_info:
                    st.write(f"พร้อมแปลง **{len(sources)} รูป** เป็นภาพลายเส้น")
                    st.caption(f"ความหนาเส้น: **{target_line_mm} มม.** | ตัดกรอบเดิม: **{'ใช่' if strip_border else 'ไม่'}**")
                with c_action:
                    btn_convert = st.button(
                        "แปลงเป็นลายเส้น",
                        type="primary",
                        icon=":material/draw:",
                        width="stretch",
                    )

            if btn_convert:
                lineart_p = LineArtParams(
                    target_line_mm=target_line_mm,
                    strip_border=strip_border,
                )
                with st.status("กำลังแปลงรูปเป็นภาพลายเส้น...", expanded=True) as conv_status:
                    progress_conv = st.progress(0)
                    done = 0
                    for i, (name, raw) in enumerate(sources, 1):
                        progress_conv.progress(i / len(sources))
                        st.write(f":material/draw: กำลังแปลงภาพ: {name}...")
                        try:
                            res = convert.convert(imgio.imdecode(raw), lineart_p)
                        except Exception as e:
                            st.warning(f"ข้ามภาพ {name}: {e}")
                            continue
                        if res.is_empty:
                            st.warning(f"ข้ามภาพ {name}: ไม่พบเส้นในภาพ")
                            continue
                        existing = next(
                            (c for c in st.session_state.converted if c["name"] == name),
                            None,
                        )
                        if existing:
                            existing["mask"] = res.mask
                            existing["thumb"] = mask_thumb(res.mask)
                        else:
                            st.session_state.uid_counter += 1
                            st.session_state.converted.append(
                                {
                                    "uid": st.session_state.uid_counter,
                                    "name": name,
                                    "caption": imgio.caption_from_filename(name),
                                    "mask": res.mask,
                                    "thumb": mask_thumb(res.mask),
                                }
                            )
                        done += 1
                    conv_status.update(
                        label=f"แปลงสำเร็จ {done} จาก {len(sources)} รูป — ไปต่อที่ 'ขั้นที่ 3' ได้เลย",
                        state="complete",
                    )

        converted = st.session_state.converted
        if converted:
            st.space("small")
            h1, h2 = st.columns([4, 1], vertical_alignment="center")
            h1.write(f"รูปลายเส้นที่แปลงแล้ว **{len(converted)} รูป**")
            if h2.button("ล้างทั้งหมด", icon=":material/delete_sweep:", width="stretch"):
                st.session_state.converted = []
                st.session_state.page_pngs = None
                st.rerun()

            per_row = 4
            for start in range(0, len(converted), per_row):
                cols = st.columns(per_row)
                for offset, col in enumerate(cols):
                    idx = start + offset
                    if idx >= len(converted):
                        break
                    item = converted[idx]
                    with col:
                        with st.container(border=True):
                            st.image(item["thumb"], caption=item["name"], width="stretch")
                            if st.button(
                                "ลบ",
                                key=f"del_{item['uid']}",
                                icon=":material/delete:",
                                width="stretch",
                            ):
                                st.session_state.converted = [
                                    c for c in converted if c["uid"] != item["uid"]
                                ]
                                st.rerun()


# ----------------- Tab 3: Build book & print -----------------
with tab_book:
    st.subheader(":material/menu_book: ขั้นที่ 3: รวมเล่ม & พิมพ์")

    converted = st.session_state.converted
    if not HAS_COLORING_ENGINE:
        st.error("ไม่พบโมดูลสร้างสมุดระบายสี (app.lineart / app.book)")
    elif not converted:
        st.info(
            "ยังไม่มีรูปลายเส้น ไปแปลงรูปที่ 'ขั้นที่ 2' ก่อนครับ",
            icon=":material/info:",
        )
    else:
        st.write("เลือกรูปที่จะใส่ในเล่ม จัดลำดับ และแก้ชื่อกำกับได้ที่นี่")
        for pos, item in enumerate(converted):
            uid = item["uid"]
            with st.container(border=True):
                c_use, c_img, c_cap, c_up, c_down = st.columns(
                    [1, 2, 6, 1, 1], vertical_alignment="center"
                )
                c_use.checkbox("ใส่", value=True, key=f"use_{uid}")
                c_img.image(item["thumb"], width=90)
                c_cap.text_input(
                    "ชื่อกำกับ",
                    value=item["caption"],
                    key=f"cap_{uid}",
                    label_visibility="collapsed",
                )
                if c_up.button("", key=f"up_{uid}", icon=":material/arrow_upward:", disabled=pos == 0):
                    converted[pos - 1], converted[pos] = converted[pos], converted[pos - 1]
                    st.rerun()
                if c_down.button(
                    "", key=f"down_{uid}", icon=":material/arrow_downward:",
                    disabled=pos == len(converted) - 1,
                ):
                    converted[pos + 1], converted[pos] = converted[pos], converted[pos + 1]
                    st.rerun()

        chosen = [c for c in converted if st.session_state.get(f"use_{c['uid']}", True)]

        with st.container(border=True):
            c_info, c_action = st.columns([3, 1], vertical_alignment="center")
            with c_info:
                st.write(f"จะรวมเป็นสมุด **{len(chosen)} รูป**")
                st.caption(
                    f"ชื่อสมุด: **{book_title}** | จัดหน้า: **{per_page} รูป/หน้า** | "
                    f"หน้าปก: **{'มี' if include_cover else 'ไม่มี'}**"
                )
            with c_action:
                btn_build = st.button(
                    "รวมเล่ม (เตรียมพิมพ์)",
                    type="primary",
                    icon=":material/menu_book:",
                    width="stretch",
                    disabled=not chosen,
                )

        if btn_build:
            with st.status("กำลังจัดหน้าสมุด A4...", expanded=True) as build_status:
                book_p = BookParams(
                    title=book_title,
                    author=book_author,
                    per_page=per_page,
                    show_frame=show_frame,
                    show_caption=show_caption,
                    show_page_number=show_page_number,
                    include_cover=include_cover,
                )
                lineart_p = LineArtParams(
                    target_line_mm=target_line_mm,
                    strip_border=strip_border,
                )
                masks = [c["mask"] for c in chosen]
                captions = [
                    (st.session_state.get(f"cap_{c['uid']}") or c["caption"]).strip()
                    or c["caption"]
                    for c in chosen
                ]
                try:
                    page_images, page_warnings = book_pdf.build_pages(
                        masks, captions, book_p, lineart_p
                    )
                    st.session_state.page_pngs = [page_to_png(im) for im in page_images]
                    for w in page_warnings:
                        st.warning(w)
                    build_status.update(
                        label=f"รวมเล่มสำเร็จ! ทั้งหมด {len(page_images)} หน้า",
                        state="complete",
                    )
                except Exception as e:
                    build_status.update(label=f"เกิดข้อผิดพลาดในการรวมเล่ม: {e}", state="error")

        if st.session_state.page_pngs:
            st.space("small")
            pages_png = st.session_state.page_pngs
            with st.container(border=True):
                st.success(
                    ":material/check_circle: สมุดระบายสี A4 พร้อมพิมพ์แล้ว! ตรวจดูทุกหน้าด้านล่าง แล้วกดปุ่มพิมพ์ได้เลย",
                    icon=":material/verified:",
                )
                c_meta, c_dl = st.columns([2, 1], vertical_alignment="center")
                with c_meta:
                    st.write(f"📄 สมุดระบายสี A4 ทั้งหมด **{len(pages_png)} หน้า**")
                with c_dl:
                    render_print_button()

                st.write("ตัวอย่างทุกหน้า:")
                with st.container(key="print_pages"):
                    per_row = 3
                    for start in range(0, len(pages_png), per_row):
                        row = st.columns(per_row)
                        for offset, col in enumerate(row):
                            idx = start + offset
                            if idx >= len(pages_png):
                                break
                            with col:
                                st.image(
                                    pages_png[idx],
                                    caption=f"หน้า {idx + 1}",
                                    width="stretch",
                                )
