"use strict";

/** สถานะของหน้าเว็บทั้งหมดรวมอยู่ที่นี่ */
const state = {
  items: [],
  selectedId: null,
  nextId: 1,
  busy: false,
  // รูปที่ค้นพบจากเว็บไซต์ ยังไม่ได้ดาวน์โหลดเข้าสมุด
  found: [],
};

const GRID_LABELS = {
  1: "1 ภาพเต็มหน้า",
  2: "2 ภาพ (แนวตั้ง)",
  4: "4 ภาพ (2×2)",
  6: "6 ภาพ (2×3)",
  9: "9 ภาพ (3×3)",
  12: "12 ภาพ (3×4)",
};

// ค่าที่ผู้ใช้เลือกสำหรับ "ความไวต่อจุดรบกวน" แปลงเป็น speckle_ratio จริง
// 0 = ให้โปรแกรมตัดสินใจเอง
const SPECKLE_STEPS = [null, 0.00002, 0.00004, 0.00008, 0.00015, 0.0003];

const el = (id) => document.getElementById(id);

const dom = {
  drop: el("drop"),
  picker: el("picker"),
  pick: el("pick"),
  items: el("items"),
  count: el("count"),
  empty: el("empty"),
  clear: el("clear"),
  title: el("title"),
  author: el("author"),
  perPage: el("perPage"),
  line: el("line"),
  lineOut: el("lineOut"),
  speckle: el("speckle"),
  speckleOut: el("speckleOut"),
  frame: el("frame"),
  cover: el("cover"),
  caption: el("caption"),
  pagenum: el("pagenum"),
  previewBtn: el("previewBtn"),
  makeBtn: el("makeBtn"),
  status: el("status"),
  previewWrap: el("previewWrap"),
  previewImg: el("previewImg"),
  broken: el("broken"),
  brokenDetail: el("brokenDetail"),
  aiBox: el("aiBox"),
  useAi: el("useAi"),
  aiHint: el("aiHint"),
  notice: el("notice"),
<<<<<<< HEAD
  webUrl: el("webUrl"),
  fetchBtn: el("fetchBtn"),
  webResults: el("webResults"),
  webImages: el("webImages"),
  webCount: el("webCount"),
  webStatus: el("webStatus"),
  webAddBtn: el("webAddBtn"),
  webSelectAll: el("webSelectAll"),
=======
  // Web scraper elements
  scrapeUrl: el("scrapeUrl"),
  scrapeBtn: el("scrapeBtn"),
  scrapeMinSize: el("scrapeMinSize"),
  scrapeMaxImages: el("scrapeMaxImages"),
  scrapeStatus: el("scrapeStatus"),
>>>>>>> 6f3ccb663956a96eeff79824f691af693e8692ef
};

/** แสดงตัวเลือก AI เฉพาะเมื่อเจ้าของเซิร์ฟเวอร์ตั้งค่าไว้แล้ว

 * ต้องบอกเรื่องค่าใช้จ่ายและการส่งภาพออกนอกเครื่องให้ชัดเจน
 * ผู้ใช้ต้องตัดสินใจอย่างรู้ตัวก่อนกดใช้
 */
function applyAiStatus(status) {
  if (!status || !status.configured) {
    dom.aiBox.hidden = true;
    return;
  }
  const cost = status.estimated_cost_per_image_usd;
  dom.aiBox.hidden = false;
  dom.aiHint.textContent =
    "ใช้เฉพาะกับภาพถ่ายเท่านั้น ภาพลายเส้นจะไม่ถูกส่งให้ AI" +
    (cost ? " · ค่าใช้จ่ายประมาณ " + cost + " USD ต่อภาพ" : "") +
    " · ภาพจะถูกส่งออกนอกเครื่องนี้ผ่านผู้ให้บริการ AI";

  // เมื่อมี AI ให้ใช้ ให้เตือนเรื่องข้อจำกัดเดิมน้อยลง
  dom.notice.querySelector("strong").textContent =
    "โปรแกรมนี้ทำงานดีที่สุดกับภาพลายเส้น";
  dom.notice.lastChild.textContent =
    " ภาพถ่ายจะได้เส้นตามภาพต้นฉบับเป็นค่าเริ่มต้น " +
    "หากอยากได้ภาพการ์ตูนให้เปิดใช้ AI ด้านล่าง";
}

/** ตรวจว่าเซิร์ฟเวอร์พร้อมวาดข้อความไทยหรือไม่

 * ถ้าเซิร์ฟเวอร์ไม่มี libraqm ข้อความไทยจะซ้อนกันผิดตำแหน่ง
 * ผู้ใช้เห็นเป็นภาพผิดรูปแต่ไม่รู้สาเหตุ จึงต้องเตือนให้ชัดตั้งแต่เปิดหน้าเว็บ
 */
async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) return;
    const data = await response.json();

    if (data.ai) applyAiStatus(data.ai);

    if (data.fonts_ready) return;
    dom.brokenDetail.textContent =
      data.font_error || "ไม่ทราบสาเหตุ กรุณาตรวจสอบ log ของเซิร์ฟเวอร์";
    dom.broken.hidden = false;
  } catch (_) {
    // ติดต่อเซิร์ฟเวอร์ไม่ได้ แสดงว่าเครื่องยังไม่พร้อมใช้งานอยู่แล้ว
  }
}

/* ---------- การแสดงผล ---------- */

function setStatus(message, kind = "") {
  dom.status.textContent = message || "";
  dom.status.className = "status" + (kind ? " " + kind : "");
}

function setBusy(busy) {
  state.busy = busy;
  const hasItems = state.items.length > 0;
  dom.previewBtn.disabled = busy || !hasItems;
  dom.makeBtn.disabled = busy || !hasItems;
  dom.previewBtn.textContent = busy ? "กำลังทำงาน" : "ดูตัวอย่างหน้า A4";
  dom.makeBtn.textContent = busy ? "กำลังสร้าง" : "สร้าง PDF";
}

function renderItems() {
  dom.items.replaceChildren();
  dom.count.textContent = String(state.items.length);
  dom.empty.hidden = state.items.length > 0;
  dom.clear.hidden = state.items.length === 0;

  for (const item of state.items) {
    dom.items.appendChild(renderItem(item));
  }
}

function renderItem(item) {
  const li = document.createElement("li");
  li.className = "item" + (item.id === state.selectedId ? " selected" : "");

  const img = document.createElement("img");
  img.className = "thumb";
  img.alt = item.name;
  img.src = item.url;

  const body = document.createElement("div");
  body.className = "item-body";

  // ใช้ textContent เสมอ เพราะชื่อไฟล์มาจากผู้ใช้ อาจมีอักขระ HTML ปนมา
  const name = document.createElement("span");
  name.className = "item-name";
  name.textContent = item.name;
  name.title = item.name;

  const input = document.createElement("input");
  input.type = "text";
  input.value = item.caption;
  input.maxLength = 120;
  input.placeholder = "ชื่อกำกับ";
  input.addEventListener("input", () => {
    item.caption = input.value;
  });
  input.addEventListener("focus", () => select(item.id));

  const tag = document.createElement("span");
  tag.className = "tag" + (item.warning ? " warn" : "");
  tag.textContent = item.warning || "พร้อมใช้";

  body.append(name, input, tag);

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "ghost";
  remove.textContent = "ลบ";
  remove.addEventListener("click", () => removeItem(item.id));

  li.append(img, body, remove);
  li.addEventListener("click", (event) => {
    if (event.target !== input) select(item.id);
  });

  return li;
}

function select(id) {
  state.selectedId = id;
  renderItems();
}

/* ---------- วางรูปจากคลิปบอร์ด ---------- */

/** ชื่อไฟล์ที่เบราว์เซอร์ตั้งให้อัตโนมัติ ไม่ได้มีความหมายกับผู้ใช้ */
const AUTO_NAMES = new Set(["image", "blob", "unnamed", "screenshot"]);

/** ดึงรูปภาพออกจากข้อมูลคลิปบอร์ด

 * เบราว์เซอร์แต่ละตัวให้ข้อมูลคนละแบบ
 * Chrome/Edge ใส่ไฟล์ไว้ใน clipboardData.files
 * บางตัวไม่เติม files แต่มี clipboardData.items ให้แทน
 * จึงต้องเช็กทั้งสองทาง ไม่งั้นวางแล้วไม่มีอะไรเกิดขึ้น
 */
function imagesFromClipboard(clipboardData) {
  if (!clipboardData) return [];

  const files = [];

  if (clipboardData.files && clipboardData.files.length) {
    for (const file of clipboardData.files) {
      if (file.type.startsWith("image/")) files.push(file);
    }
  }
  if (files.length) return files;

  if (clipboardData.items) {
    for (const item of clipboardData.items) {
      if (item.kind !== "file" || !item.type.startsWith("image/")) continue;
      const file = item.getAsFile();
      if (file) files.push(file);
    }
  }
  return files;
}

/** ตั้งชื่อกำกับให้รูปที่วางมา เพราะชื่อไฟล์มักเป็น image.png ซึ่งไม่มีความหมาย */
function pastedCaption(file, index) {
  const base = file.name.replace(/\.[^.]+$/, "").toLowerCase();
  if (file.name && !AUTO_NAMES.has(base)) {
    return defaultCaption(file.name);
  }
  return "ภาพที่วาง " + index;
}

document.addEventListener("paste", (event) => {
  const files = imagesFromClipboard(event.clipboardData);
  if (files.length === 0) return; // ปล่อยให้วางข้อความตามปกติ

  // ถ้ากำลังพิมพ์อยู่ในช่องข้อความ ต้องไม่ขัดจังหวะการวางข้อความ
  const tag = document.activeElement && document.activeElement.tagName;
  const inTextField = tag === "INPUT" || tag === "TEXTAREA";
  if (!inTextField) event.preventDefault();

  const startIndex = state.items.length + 1;
  const prepared = files.map((file, index) => ({
    file,
    name: file.name || "pasted-image",
    caption: pastedCaption(file, startIndex + index),
    url: URL.createObjectURL(file),
    warning: null,
  }));

  state.items.push(...prepared);
  if (state.selectedId === null) state.selectedId = state.items[0].id;
  renderItems();
  setBusy(false);
  setStatus(
    `วางรูปจากคลิปบอร์ด ${files.length} รูป` +
      (inTextField ? " (กดลอกช่องอื่นก่อนวางรูปได้สะดวกขึ้น)" : "")
  );
});

/* ---------- ดึงรูปจากเว็บไซต์ ---------- */

/** แสดงรูปที่ค้นพบจากเว็บไซต์เป็นการ์ดให้ผู้ใช้เลือก */
function renderWebImages() {
  dom.webImages.replaceChildren();

  if (state.found.length === 0) {
    dom.webImages.textContent = "ไม่พบรูปภาพในหน้าเว็บนี้";
    return;
  }

  for (const [index, img] of state.found.entries()) {
    const card = document.createElement("div");
    card.className = "web-image";
    card.dataset.index = index;

    const picture = document.createElement("img");
    picture.src = img.url;
    picture.alt = img.name;
    picture.loading = "lazy";
    // รูปโหลดไม่ขึ้นมักเป็นเพราะเว็บเป้าหมายไม่อนุญาตให้อ่านข้ามโดเมน
    // ส่วนการดาวน์โหลดจริงยังทำงานได้ เพราะเซิร์ฟเวอร์เป็นคนโหลด
    picture.addEventListener("error", () => {
      picture.replaceWith(placeholder());
    });

    const tick = document.createElement("span");
    tick.className = "tick";
    tick.textContent = "✓";

    const name = document.createElement("span");
    name.className = "wname";
    // ใช้ textContent เสมอ เพราะชื่อมาจากเว็บเป้าหมาย อาจมีอักขระ HTML ปนมา
    name.textContent = img.name;
    name.title = img.url;

    card.append(picture, tick, name);
    card.addEventListener("click", () => {
      card.classList.toggle("on");
      syncWebSelection();
    });

    dom.webImages.appendChild(card);
  }
}

/** รูปตัวแทนสำหรับรูปที่เบราว์เซอร์โหลดไม่ขึ้น */
function placeholder() {
  const box = document.createElement("div");
  box.className = "noimg";
  box.textContent = "ดูตัวอย่างไม่ได้";
  return box;
}

/** นับรูปที่เลือกไว้ แล้วเปิดปุ่มเพิ่มเฉพาะเมื่อมีอย่างน้อยหนึ่งรูป */
function syncWebSelection() {
  const chosen = dom.webImages.querySelectorAll(".web-image.on").length;
  dom.webAddBtn.disabled = chosen === 0;
  dom.webAddBtn.textContent =
    chosen === 0 ? "เพิ่มรูปที่เลเพิ่มรูปที่เลือกลงในสมุดกลงในสมุด" : `เพิ่มรูปที่เลือก (${chosen}) ลงในสมุด`;
  dom.webSelectAll.hidden = state.found.length === 0;
}

function setWebStatus(message, kind = "") {
  dom.webStatus.textContent = message || "";
  dom.webStatus.className = "status" + (kind ? " " + kind : "");
}

/** ขอเซิร์ฟเวอร์ไล่หาลิงก์รูปในหน้าเว็บที่ผู้ใช้ใส่มา */
async function fetchWebImages() {
  const url = dom.webUrl.value.trim();
  if (!url) {
    setWebStatus("กรุณาใส่ลิงก์เว็บไซต์ก่อน", "error");
    return;
  }

  dom.fetchBtn.disabled = true;
  setWebStatus("กำลังค้นหารูปในหน้าเว็บ...");

  try {
    const response = await fetch("/api/web-images", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ url }),
    });

    if (!response.ok) {
      setWebStatus(await readError(response), "error");
      return;
    }

    const data = await response.json();
    state.found = data.images || [];
    dom.webCount.textContent = String(state.found.length);
    dom.webResults.hidden = false;

    if (state.found.length === 0) {
      setWebStatus("ไม่พบรูปภาพในหน้าเว็บนี้", "warn");
    } else {
      setWebStatus(
        `พบ ${state.found.length} รูป — เลือกเฉพาะที่ต้องการ`
      );
    }

    renderWebImages();
    syncWebSelection();
  } catch (_) {
    setWebStatus("เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ", "error");
  } finally {
    dom.fetchBtn.disabled = false;
  }
}

/** ดาวน์โหลดรูปที่ผู้ใช้เลือก แล้วเข้าสู่ขั้นตอนเดียวกับไฟล์ที่อัปโหลด */
async function addSelectedWebImages() {
  const chosen = Array.from(dom.webImages.querySelectorAll(".web-image.on"));
  if (chosen.length === 0) {
    setWebStatus("ยังไม่ได้เลือกรูป", "warn");
    return;
  }

  dom.webAddBtn.disabled = true;
  const failures = [];
  let added = 0;

  for (const card of chosen) {
    const target = state.found[Number(card.dataset.index)];
    if (!target) continue;

    setWebStatus(`กำลังดาวน์โหลดรูป ${added + 1} จาก ${chosen.length}...`);
    try {
      const response = await fetch("/api/web-image", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ url: target.url }),
      });
      if (!response.ok) {
        failures.push(`${target.name}: ${await readError(response)}`);
        continue;
      }

      const blob = await response.blob();
      // ต้องตั้งชื่อไฟล์ให้มีนามสกุลที่รองรับ ไม่งั้นเซิร์ฟเวอร์จะปฏิเสธตอนสร้าง PDF
      const filename = withImageExtension(
        target.name,
        blob.type || "image/jpeg"
      );
      const file = new File([blob], filename, {
        type: blob.type || "image/jpeg",
      });

      state.items.push({
        id: state.nextId++,
        file,
        name: filename,
        caption: defaultCaption(filename),
        url: URL.createObjectURL(file),
        warning: null,
      });
      added += 1;
    } catch (_) {
      failures.push(`${target.name}: ดาวน์โหลดไม่สำเร็จ`);
    }
  }

  if (added > 0) {
    if (state.selectedId === null) state.selectedId = state.items[0].id;
    renderItems();
    setBusy(false);
  }

  state.found = [];
  dom.webResults.hidden = true;
  dom.webImages.replaceChildren();
  dom.webAddBtn.textContent = "เพิ่มรูปที่เลือกลงในสมุด";
  dom.webUrl.value = "";

  if (failures.length === 0) {
    setWebStatus("");
  } else {
    setWebStatus(
      `เพิ่มได้ ${added} รูป · ข้าม ${failures.length} รูปที่ดาวน์โหลดไม่ได้: ` +
        failures.join(" | "),
      "warn"
    );
  }
}

/** เติมนามสกุลให้ชื่อที่ได้จาก URL ซึ่งมักไม่มี

 * เซิร์ฟเวอร์ปฏิเสธไฟล์ที่นามสกุลไม่อยู่ในรายการที่รองรับ
 * ชื่อจาก URL มักเป็นแบไม่มีนามสกุล a1b2c3 ไม่มีนามสกุล จึงต้องเติมจากชนิดที่เซิร์ฟเวอร์ส่งมา
 */
function withImageExtension(name, mimeType) {
  const supported = /\.(png|jpe?g|webp|bmp|tiff?)$/i;
  if (supported.test(name)) return name;

  const byMime = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
  };
  return name + "." + (byMime[mimeType] || "jpg");
}

/* ---------- จัดการไฟล์ ---------- */

function addFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (files.length === 0) {
    setStatus("ไม่พบไฟล์ภาพที่รองรับ", "error");
    return;
  }
  for (const file of files) {
    state.items.push({
      id: state.nextId++,
      file,
      name: file.name,
      caption: defaultCaption(file.name),
      url: URL.createObjectURL(file),
      warning: null,
    });
  }
  if (state.selectedId === null) {
    state.selectedId = state.items[0].id;
  }
  renderItems();
  setBusy(false);
  setStatus(`เพิ่ม ${files.length} ภาพแล้ว`);
}

/** ตัดนามสกุลออกจากชื่อไฟล์ เพื่อใช้เป็นชื่อกำกับเริ่มต้น */
function defaultCaption(filename) {
  const base = filename.split("/").pop().split("\\").pop();
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  return stem || "ระบายสี";
}

function removeItem(id) {
  const index = state.items.findIndex((i) => i.id === id);
  if (index === -1) return;
  URL.revokeObjectURL(state.items[index].url);
  state.items.splice(index, 1);
  if (state.selectedId === id) {
    state.selectedId = state.items.length ? state.items[0].id : null;
  }
  dom.previewWrap.hidden = true;
  renderItems();
  setBusy(false);
}

function clearAll() {
  for (const item of state.items) URL.revokeObjectURL(item.url);
  state.items = [];
  state.selectedId = null;
  dom.previewWrap.hidden = true;
  renderItems();
  setBusy(false);
  setStatus("");
}

/* ---------- พารามิเตอร์ ---------- */

function lineArtPayload() {
  const line = parseFloat(dom.line.value);
  const speckleIndex = parseInt(dom.speckle.value, 10);
  return {
    target_line_mm: line === 0 ? null : line,
    speckle_ratio: SPECKLE_STEPS[speckleIndex],
    use_ai: dom.useAi.checked,
  };
}

function bookPayload() {
  return {
    title: dom.title.value,
    author: dom.author.value,
    per_page: parseInt(dom.perPage.value, 10),
    show_frame: dom.frame.checked,
    include_cover: dom.cover.checked,
    show_caption: dom.caption.checked,
    show_page_number: dom.pagenum.checked,
  };
}

function syncLabels() {
  const line = parseFloat(dom.line.value);
  dom.lineOut.textContent = line === 0 ? "อัตโนมัติ" : line.toFixed(1) + " มม.";

  const speckleIndex = parseInt(dom.speckle.value, 10);
  dom.speckleOut.textContent =
    speckleIndex === 0 ? "อัตโนมัติ" : "ระดับ " + speckleIndex;
}

async function readError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail) && data.detail[0]) {
      return data.detail[0].msg || "คำขอไม่ถูกต้อง";
    }
  } catch (_) {
    /* ใช้ข้อความทั่วไปแทน */
  }
  return "ทำรายการไม่สำเร็จ (รหัส " + response.status + ")";
}

/** ถอดข้อความไทยจาก HTTP header กลับเป็นข้อความอ่านได้

 * header ต้องเป็น ASCII เท่านั้น ฝั่งเซิร์ฟเวอร์จึงส่งข้อความภาษาไทยมา
 * ในรูปแบบ \xE0\xB8\xA3 ต้องถอดกลับก่อนนำไปแสดง
 */
function decodeHeader(value) {
  if (!value) return "";
  return value
    .split(";")
    .map((part) =>
      part
        .trim()
        .replace(/\\x([0-9a-fA-F]{2})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
    )
    .filter(Boolean)
    .join(" · ");
}

function downloadName(response) {
  const header = response.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/.exec(header);
  return match ? match[1] : "coloring-book.pdf";
}

/* ---------- ดึงรูปจากเว็บไซต์ ---------- */

async function scrapeImages() {
  const url = dom.scrapeUrl.value.trim();

  if (!url) {
    dom.scrapeStatus.textContent = "กรุณากรอก URL";
    dom.scrapeStatus.className = "status error";
    return;
  }

  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    dom.scrapeStatus.textContent = "URL ต้องขึ้นต้นด้วย http:// หรือ https://";
    dom.scrapeStatus.className = "status error";
    return;
  }

  const minSize = parseInt(dom.scrapeMinSize.value) || 5;
  const maxImages = parseInt(dom.scrapeMaxImages.value) || 30;

  dom.scrapeBtn.disabled = true;
  dom.scrapeStatus.textContent = "กำลังดึงรูปภาพจากเว็บไซต์...";
  dom.scrapeStatus.className = "status busy";

  try {
    const response = await fetch("/api/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: url,
        min_size_kb: minSize,
        max_images: maxImages,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "ไม่สามารถดึงรูปได้");
    }

    const data = await response.json();

    if (data.images && data.images.length > 0) {
      // แปลง base64 เป็นไฟล์
      for (const img of data.images) {
        const response = await fetch(img.data);
        const blob = await response.blob();
        const file = new File([blob], img.name, { type: blob.type });

        state.items.push({
          id: state.nextId++,
          file: file,
          name: img.name,
          caption: defaultCaption(img.name),
          url: URL.createObjectURL(file),
          warning: null,
        });
      }

      if (state.selectedId === null && state.items.length > 0) {
        state.selectedId = state.items[0].id;
      }

      renderItems();
      setBusy(false);
      dom.scrapeStatus.textContent = `ดึงรูปสำเร็จ ${data.images.length} รูป`;
      dom.scrapeStatus.className = "status";
    } else {
      dom.scrapeStatus.textContent = data.message || "ไม่พบรูปภาพในหน้าเว็บนี้";
      dom.scrapeStatus.className = "status warn";
    }
  } catch (error) {
    dom.scrapeStatus.textContent = error.message || "เกิดข้อผิดพลาดในการดึงรูป";
    dom.scrapeStatus.className = "status error";
  } finally {
    dom.scrapeBtn.disabled = false;
  }
}

/* ---------- การเรียกเซิร์ฟเวอร์ ---------- */

async function preview() {
  const item = state.items.find((i) => i.id === state.selectedId) || state.items[0];
  if (!item) return;

  setBusy(true);
  setStatus("กำลังสร้างตัวอย่าง");

  const body = new FormData();
  body.append("file", item.file, item.name);
  body.append("lineart", JSON.stringify(lineArtPayload()));
  body.append("book", JSON.stringify(bookPayload()));

  try {
    const response = await fetch("/api/preview", { method: "POST", body });
    if (!response.ok) {
      setStatus(await readError(response), "error");
      return;
    }
    const blob = await response.blob();
    if (dom.previewImg.dataset.url) URL.revokeObjectURL(dom.previewImg.dataset.url);
    const url = URL.createObjectURL(blob);
    dom.previewImg.dataset.url = url;
    dom.previewImg.src = url;
    dom.previewWrap.hidden = false;

    const notice = decodeHeader(response.headers.get("X-Notice"));
    if (notice) {
      setStatus(notice, "warn");
    } else {
      setStatus("ตัวอย่างพร้อมแล้ว");
    }
  } catch (_) {
    setStatus("เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ", "error");
  } finally {
    setBusy(false);
  }
}

async function makeBook() {
  if (state.items.length === 0) return;

  setBusy(true);
  setStatus("กำลังสร้าง PDF");

  const body = new FormData();
  for (const item of state.items) {
    body.append("files", item.file, item.name);
  }
  body.append(
    "captions",
    JSON.stringify(state.items.map((i) => i.caption))
  );
  body.append("lineart", JSON.stringify(lineArtPayload()));
  body.append("book", JSON.stringify(bookPayload()));

  try {
    const response = await fetch("/api/book", { method: "POST", body });
    if (!response.ok) {
      setStatus(await readError(response), "error");
      return;
    }

    const blob = await response.blob();
    
    // ลองดาวน์โหลดอัตโนมัติ
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = downloadName(response);
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 30000);

    // Fallback: แสดงลิงก์ดาวน์โหลดสำรอง เผื่อ browser block
    const fallbackUrl = URL.createObjectURL(blob);
    const fallbackLink = document.createElement("a");
    fallbackLink.href = fallbackUrl;
    fallbackLink.download = downloadName(response);
    fallbackLink.textContent = "🔗 ดาวน์โหลด PDF (คลิกขวา > Save link as...)";
    fallbackLink.style.cssText = "display:inline-block;margin-top:8px;padding:8px 12px;background:#eef6f1;border:1px solid #2f6f4e;border-radius:6px;color:#2f6f4e;text-decoration:none;font-size:14px;";
    const existing = document.getElementById("fallbackDownload");
    if (existing) existing.remove();
    fallbackLink.id = "fallbackDownload";
    dom.status.insertAdjacentElement("afterend", fallbackLink);
    setTimeout(() => URL.revokeObjectURL(fallbackUrl), 300000);

    const pages = response.headers.get("X-Page-Count");
    const warn = decodeHeader(response.headers.get("X-Warnings"));
    if (warn) {
      setStatus(`สร้างเสร็จแล้ว` + (pages ? " " + pages + " หน้า" : "") + " — " + warn, "warn");
    } else {
      setStatus("สร้างเสร็จแล้ว" + (pages ? " " + pages + " หน้า" : ""));
    }
  } catch (_) {
    setStatus("เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ", "error");
  } finally {
    setBusy(false);
  }
}

/* ---------- เริ่มทำงาน ---------- */

function init() {
  for (const [value, label] of Object.entries(GRID_LABELS)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    if (value === "1") option.selected = true;
    dom.perPage.appendChild(option);
  }

  dom.pick.addEventListener("click", (e) => {
    e.stopPropagation();
    dom.picker.click();
  });
  dom.drop.addEventListener("click", () => dom.picker.click());
  dom.drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      dom.picker.click();
    }
  });

  dom.picker.addEventListener("change", () => {
    addFiles(dom.picker.files);
    dom.picker.value = "";
  });

  for (const type of ["dragenter", "dragover"]) {
    dom.drop.addEventListener(type, (event) => {
      event.preventDefault();
      dom.drop.classList.add("dragging");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    dom.drop.addEventListener(type, () => dom.drop.classList.remove("dragging"));
  }
  dom.drop.addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer && event.dataTransfer.files.length) {
      addFiles(event.dataTransfer.files);
    }
  });
  // กันไม่ให้เบราว์เซอร์เปิดไฟล์ทับหน้าเว็บเมื่อวางนอกพื้นที่
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  dom.clear.addEventListener("click", clearAll);
  dom.previewBtn.addEventListener("click", preview);
  dom.makeBtn.addEventListener("click", makeBook);
  dom.line.addEventListener("input", syncLabels);
  dom.speckle.addEventListener("input", syncLabels);

<<<<<<< HEAD
  dom.fetchBtn.addEventListener("click", fetchWebImages);
  dom.webAddBtn.addEventListener("click", addSelectedWebImages);
  dom.webSelectAll.addEventListener("click", () => {
    const cards = Array.from(dom.webImages.children);
    const allOn = cards.every((card) => card.classList.contains("on"));
    for (const card of cards) card.classList.toggle("on", !allOn);
    syncWebSelection();
  });
  // กด Enter ในช่องลิงก์เพื่อค้นหาได้เลย ไม่ต้องขยับเมาส์ไปกดปุ่ม
  dom.webUrl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      fetchWebImages();
    }
=======
  // Web scraper
  dom.scrapeBtn.addEventListener("click", scrapeImages);
  dom.scrapeUrl.addEventListener("keypress", (event) => {
    if (event.key === "Enter") scrapeImages();
>>>>>>> 6f3ccb663956a96eeff79824f691af693e8692ef
  });

  dom.previewImg.addEventListener("load", () => {
    dom.previewImg.removeAttribute("width");
  });

  syncLabels();
  renderItems();
  setBusy(false);
  checkHealth();
}

init();
