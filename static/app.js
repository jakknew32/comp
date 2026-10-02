"use strict";

/** สถานะของหน้าเว็บทั้งหมดรวมอยู่ที่นี่ */
const state = {
  items: [],
  selectedId: null,
  nextId: 1,
  busy: false,
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
};

/** ตรวจว่าเซิร์ฟเวอร์พร้อมวาดข้อความไทยหรือไม่

 * ถ้าเซิร์ฟเวอร์ไม่มี libraqm ข้อความไทยจะซ้อนกันผิดตำแหน่ง
 * ผู้ใช้เห็นเป็นภาพผิดรูปแต่ไม่รู้สาเหตุ จึงต้องเตือนให้ชัดตั้งแต่เปิดหน้าเว็บ
 */
async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) return;
    const data = await response.json();
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
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = downloadName(response);
    document.body.appendChild(link);
    link.click();
    link.remove();
    // ปล่อย URL หลังเบราว์เซอร์เริ่มดาวน์โหลดแล้ว
    setTimeout(() => URL.revokeObjectURL(link.href), 30000);

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

  dom.previewImg.addEventListener("load", () => {
    dom.previewImg.removeAttribute("width");
  });

  syncLabels();
  renderItems();
  setBusy(false);
  checkHealth();
}

init();
