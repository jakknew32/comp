"use strict";
/** สถานะของหน้าเว็บทั้งหมดรวมอยู่ที่นี่ */
const state = {
  items: [],          // ภาพในสมุด มาจากทุกแหล่ง
  selectedId: null,
  nextId: 1,
  busy: false,
  pages: [],          // ภาพทุกหน้าของสมุดที่รวมเล่มแล้ว (data URL)
  found: [],          // ผลค้นหารูปจากเว็บ
  generated: [],      // ภาพที่ AI สร้างให้รอให้เลือก
  style: "classic",
};
const GRID_LABELS = {
  1: "1 ภาพเต็มหน้า",
  2: "2 ภาพ (แนวตั้ง)",
  4: "4 ภาพ (2×2)",
  6: "6 ภาพ (2×3)",
  9: "9 ภาพ (3×3)",
  12: "12 ภาพ (3×4)",
};
const SPECKLE_STEPS = [null, 0.00002, 0.00004, 0.00008, 0.00015, 0.0003];
const el = (id) => document.getElementById(id);
const dom = {
  // แท็บ
  tabGen: el("tabGen"), tabWeb: el("tabWeb"), tabUpload: el("tabUpload"),
  panelGen: el("panelGen"), panelWeb: el("panelWeb"), panelUpload: el("panelUpload"),
  // สร้างภาพ
  genPrompt: el("genPrompt"), genStyles: el("genStyles"), genCount: el("genCount"),
  genBtn: el("genBtn"), genResults: el("genResults"), genImages: el("genImages"),
  genCountFound: el("genCountFound"), genAddBtn: el("genAddBtn"),
  genStatus: el("genStatus"), genAiNote: el("genAiNote"),
  // ดึงภาพจากเว็บ
  webUrl: el("webUrl"), fetchBtn: el("fetchBtn"),
  webResults: el("webResults"), webImages: el("webImages"),
  webCount: el("webCount"), webStatus: el("webStatus"),
  webAddBtn: el("webAddBtn"), webSelectAll: el("webSelectAll"),
  // อัปโหลด
  drop: el("drop"), picker: el("picker"), pick: el("pick"), uploadStatus: el("uploadStatus"),
  // สมุด
  items: el("items"), count: el("count"), empty: el("empty"), clear: el("clear"),
  // ตั้งค่า
  title: el("title"), author: el("author"), perPage: el("perPage"),
  line: el("line"), speckle: el("speckle"), skipConvert: el("skipConvert"),
  frame: el("frame"), cover: el("cover"),
  caption: el("caption"), pagenum: el("pagenum"),
  previewBtn: el("previewBtn"), convertBtn: el("convertBtn"), bookBtn: el("bookBtn"),
  pagesCard: el("pagesCard"), pagesGrid: el("pagesGrid"),
  pagesCount: el("pagesCount"), printBtn: el("printBtn"),
  status: el("status"), previewWrap: el("previewWrap"), previewImg: el("previewImg"),
  broken: el("broken"), brokenDetail: el("brokenDetail"),
  aiBox: el("aiBox"), useAi: el("useAi"), aiHint: el("aiHint"),
};

/* ---------- แท็บแหล่งภาพ ---------- */
const TABS = [
  { tab: dom.tabGen, panel: dom.panelGen },
  { tab: dom.tabWeb, panel: dom.panelWeb },
  { tab: dom.tabUpload, panel: dom.panelUpload },
];
function activateTab(targetTab) {
  for (const { tab, panel } of TABS) {
    const on = tab === targetTab;
    tab.classList.toggle("active", on);
    tab.setAttribute("aria-selected", on ? "true" : "false");
    panel.hidden = !on;
  }
}
function initTabs() {
  for (const { tab } of TABS) {
    tab.addEventListener("click", () => activateTab(tab));
  }
  activateTab(dom.tabWeb);
}

/* ---------- ตรวจสถานะเซิร์ฟเวอร์ ---------- */
function applyAiStatus(status) {
  if (!status) return;
  if (status.styles && status.styles.length) buildStyleChips(status.styles);
  if (!status.configured) {
    dom.genAiNote.hidden = false;
    dom.genAiNote.textContent =
      "🔑 ยังไม่ได้ตั้งค่า AI — " + (status.message || "ต้องใส่คีย์ผู้ให้บริการที่ตัวแปรแวดล้อมของเซิร์ฟเวอร์ก่อน") +
      " ช่องทางดึงภาพและอัปโหลดยังใช้ได้ปกติ";
    return;
  }
  const cost = status.estimated_cost_per_image_usd;
  dom.genAiNote.hidden = false;
  dom.genAiNote.textContent =
    "โมเดล: " + (status.model || "-") +
    (cost ? " · ค่าใช้จ่ายประมาณ " + cost + " USD ต่อภาพ" : "") +
    " · คำบรรยายและผลลัพธ์จะผ่านผู้ให้บริการ AI";
  dom.aiBox.hidden = false;
  dom.aiHint.textContent =
    "ใช้เฉพาะกับภาพถ่ายที่ดึงมาหรืออัปโหลด ภาพลายเส้นอยู่แล้วจะไม่ถูกส่งให้ AI";
}
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
  } catch (_) {}
}

/* ---------- การแสดงผลทั่วไป ---------- */
function setStatus(message, kind = "") {
  dom.status.textContent = message || "";
  dom.status.className = "status" + (kind ? " " + kind : "");
}
function clearPages() {
  state.pages = [];
  dom.pagesGrid.replaceChildren();
  dom.pagesCount.textContent = "0";
  dom.pagesCard.hidden = true;
}
function showPages(pages) {
  state.pages = pages;
  dom.pagesGrid.replaceChildren();
  pages.forEach((src, index) => {
    const figure = document.createElement("figure");
    figure.className = "page-thumb";
    const img = document.createElement("img");
    img.src = src; img.alt = "หน้า " + (index + 1); img.loading = "lazy";
    img.addEventListener("click", () => window.open(src, "_blank", "noopener"));
    const caption = document.createElement("figcaption");
    caption.textContent = "หน้า " + (index + 1);
    figure.append(img, caption);
    dom.pagesGrid.appendChild(figure);
  });
  dom.pagesCount.textContent = String(pages.length);
  dom.pagesCard.hidden = pages.length === 0;
}
/* คีย์บอกว่าภาพถูกแปลงด้วยค่าชุดไหน ถ้าค่าเปลี่ยนต้องแปลงใหม่ */
function convertKey() { return JSON.stringify(lineArtPayload()); }
/* รวมเล่มแบบไม่แปลง: ทุกรูปถือว่าพร้อมรวมเล่มทันที ไม่ต้องผ่านขั้นแปลง */
function isConverted(item) {
  return dom.skipConvert.checked || item.convertedKey === convertKey();
}
function setBusy(busy, label) {
  state.busy = busy;
  const hasItems = state.items.length > 0;
  dom.previewBtn.disabled = busy || !hasItems;
  dom.convertBtn.disabled = busy || !hasItems || dom.skipConvert.checked;
  dom.bookBtn.disabled = busy || !hasItems;
  dom.previewBtn.textContent = busy && label === "preview" ? "⏳ กำลังทำงาน..." : "👁️ ดูตัวอย่างหน้า A4";
  dom.convertBtn.textContent = busy && label === "convert" ? "⏳ กำลังแปลง..." : "✏️ 1. แปลงเป็นลายเส้น";
  dom.bookBtn.textContent = busy && label === "book" ? "⏳ กำลังรวมเล่ม..." : "📖 2. รวมเล่ม";
}
function renderItems() {
  dom.items.replaceChildren();
  dom.count.textContent = String(state.items.length);
  dom.empty.hidden = state.items.length > 0;
  dom.clear.hidden = state.items.length === 0;
  for (const item of state.items) dom.items.appendChild(renderItem(item));
}
function renderItem(item) {
  const li = document.createElement("li");
  li.className = "item" + (item.id === state.selectedId ? " selected" : "");
  const img = document.createElement("img");
  img.className = "thumb"; img.alt = item.name;
  img.src = !dom.skipConvert.checked && isConverted(item) && item.lineThumb ? item.lineThumb : item.url;
  const body = document.createElement("div");
  body.className = "item-body";
  const name = document.createElement("span");
  name.className = "item-name"; name.textContent = item.name; name.title = item.name;
  const input = document.createElement("input");
  input.type = "text"; input.value = item.caption; input.maxLength = 120;
  input.placeholder = "ชื่อกำกับ";
  input.addEventListener("input", () => { item.caption = input.value; });
  input.addEventListener("focus", () => { select(item.id); });
  const tag = document.createElement("span");
  tag.className = "tag" + (item.warning ? " warn" : "");
  tag.textContent = item.warning || (dom.skipConvert.checked ? "ใช้ภาพตามที่เป็น" : isConverted(item) ? "แปลงเป็นลายเส้นแล้ว ✓" : item.source === "ai" ? "สร้างด้วย AI" : item.source === "web" ? "จากเว็บ" : "ยังไม่ได้แปลง");
  body.append(name, input, tag);
  const remove = document.createElement("button");
  remove.type = "button"; remove.className = "btn btn-ghost"; remove.textContent = "🗑️ ลบ";
  remove.addEventListener("click", (e) => { e.stopPropagation(); removeItem(item.id); });
  li.append(img, body, remove);
  li.addEventListener("click", (event) => {
    if (event.target !== input) select(item.id);
  });
  return li;
}
function select(id) { state.selectedId = id; renderItems(); }
function defaultCaption(filename) {
  const base = filename.split("/").pop().split("\\").pop();
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  return stem || "ระบายสี";
}

/* ---------- ที่ 1: สร้างภาพด้วย AI ---------- */
function buildStyleChips(styles) {
  dom.genStyles.replaceChildren();
  for (const style of styles) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip" + (style.key === state.style ? " on" : "");
    chip.textContent = style.label;
    chip.addEventListener("click", () => {
      state.style = style.key;
      for (const child of dom.genStyles.children) child.classList.remove("on");
      chip.classList.add("on");
    });
    dom.genStyles.appendChild(chip);
  }
}
function setGenStatus(message, kind = "") {
  dom.genStatus.textContent = message || "";
  dom.genStatus.className = "status" + (kind ? " " + kind : "");
}
function renderGenerated() {
  dom.genImages.replaceChildren();
  dom.genCountFound.textContent = String(state.generated.length);
  dom.genResults.hidden = state.generated.length === 0;
  state.generated.forEach((entry, index) => {
    const card = document.createElement("div");
    card.className = "gen-image on"; card.dataset.index = index;
    const picture = document.createElement("img");
    picture.src = entry.data; picture.alt = "ภาพที่ AI สร้าง " + (index + 1);
    const tick = document.createElement("span");
    tick.className = "tick"; tick.textContent = "✓";
    card.append(picture, tick);
    card.addEventListener("click", () => {
      card.classList.toggle("on");
      syncGenSelection();
    });
    dom.genImages.appendChild(card);
  });
  syncGenSelection();
}
function syncGenSelection() {
  const chosen = dom.genImages.querySelectorAll(".gen-image.on").length;
  dom.genAddBtn.disabled = chosen === 0;
  dom.genAddBtn.textContent =
    chosen === 0 ? "➕ เพิ่มที่เลือก (0) ลงในสมุด" : `➕ เพิ่มที่เลือก (${chosen}) ลงในสมุด`;
}
async function dataUrlToFile(dataUrl, name) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return new File([blob], name, { type: "image/png" });
}
async function generateImages() {
  const prompt = dom.genPrompt.value.trim();
  if (!prompt) {
    setGenStatus("กรุณาพิมพ์คำบรรยายภาพที่ต้องการก่อน", "error");
    dom.genPrompt.focus();
    return;
  }
  dom.genBtn.disabled = true;
  const count = parseInt(dom.genCount.value, 10) || 1;
  setGenStatus("⏳ กำลังให้ AI วาดภาพ " + count + " ภาพ อาจใช้เวลาสักครู่...");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        style: state.style,
        count,
      }),
    });
    if (!response.ok) {
      setGenStatus(await readError(response), "error");
      return;
    }
    const data = await response.json();
    state.generated = (data.images || []).map((img) => ({ data: img.data, name: img.name }));
    renderGenerated();
    if (state.generated.length === 0) {
      setGenStatus("ไม่ได้ภาพกลับมา ลองอีกครั้ง", "error");
    } else {
      const extra = data.failed && data.failed.length
        ? " — ล้มเหลวบางส่วน: " + data.failed.join(" | ") : "";
      setGenStatus("✨ ได้ภาพ " + state.generated.length + " ภาพ — เลือกแล้วกดเพิ่มลงในสมุด" + extra);
    }
  } catch (_) {
    setGenStatus("เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ", "error");
  } finally {
    dom.genBtn.disabled = false;
  }
}
async function addGeneratedToBook() {
  const chosen = Array.from(dom.genImages.querySelectorAll(".gen-image.on"));
  if (chosen.length === 0) return;
  dom.genAddBtn.disabled = true;
  const stamp = Date.now();
  let added = 0;
  for (const [i, card] of chosen.entries()) {
    const entry = state.generated[Number(card.dataset.index)];
    if (!entry) continue;
    const caption = promptCaption(entry.name, stamp, added + 1);
    const file = await dataUrlToFile(entry.data, caption + ".png");
    state.items.push({
      id: state.nextId++,
      file,
      name: file.name,
      caption,
      url: entry.data,
      warning: null,
      source: "ai",
    });
    added += 1;
  }
  if (added > 0) {
    if (state.selectedId === null) state.selectedId = state.items[0].id;
    renderItems();
    setBusy(false);
    setStatus("✨ เพิ่มภาพที่ AI สร้าง " + added + " ภาพเข้าสมุดแล้ว");
  }
  state.generated = [];
  dom.genResults.hidden = true;
  dom.genImages.replaceChildren();
  dom.genPrompt.value = "";
  setGenStatus("");
}
function promptCaption(prompt, stamp, order) {
  const stem = prompt.replace(/\s+/g, " ").trim().slice(0, 40);
  return stem ? stem + " " + order : "ภาพ AI " + stamp;
}

/* ---------- ที่ 2: ดึงรูปจากเว็บไซต์ ---------- */
function renderWebImages() {
  dom.webImages.replaceChildren();
  if (state.found.length === 0) {
    dom.webImages.textContent = "ไม่พบรูปภาพในหน้าเว็บนี้";
    return;
  }
  for (const [index, img] of state.found.entries()) {
    const card = document.createElement("div");
    card.className = "web-image"; card.dataset.index = index;
    const picture = document.createElement("img");
    picture.src = img.url; picture.alt = img.name; picture.loading = "lazy";
    picture.addEventListener("error", () => { picture.replaceWith(placeholder()); });
    const tick = document.createElement("span");
    tick.className = "tick"; tick.textContent = "✓";
    const name = document.createElement("span");
    name.className = "wname"; name.textContent = img.name; name.title = img.url;
    card.append(picture, tick, name);
    card.addEventListener("click", () => {
      card.classList.toggle("on");
      syncWebSelection();
    });
    dom.webImages.appendChild(card);
  }
}
function placeholder() {
  const box = document.createElement("div");
  box.className = "noimg"; box.textContent = "ดูตัวอย่างไม่ได้";
  return box;
}
function syncWebSelection() {
  const chosen = dom.webImages.querySelectorAll(".web-image.on").length;
  dom.webAddBtn.disabled = chosen === 0;
  dom.webAddBtn.textContent =
    chosen === 0 ? "➕ เพิ่มรูปที่เลือก (0) ลงในสมุด" : `➕ เพิ่มรูปที่เลือก (${chosen}) ลงในสมุด`;
  dom.webSelectAll.hidden = state.found.length === 0;
}
function setWebStatus(message, kind = "") {
  dom.webStatus.textContent = message || "";
  dom.webStatus.className = "status" + (kind ? " " + kind : "");
}
async function fetchWebImages() {
  const url = dom.webUrl.value.trim();
  if (!url) {
    setWebStatus("กรุณาใส่ลิงก์ก่อน", "error");
    return;
  }
  dom.fetchBtn.disabled = true;
  setWebStatus("🔍 กำลังค้นหารูป...");
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
      setWebStatus("ไม่พบรูปภาพจากลิงก์นี้", "warn");
    } else if (state.found.length === 1) {
      setWebStatus("พบลิงก์รูปตรง 1 รูป — กดเพิ่มลงในสมุดได้เลย");
    } else {
      setWebStatus(`พบ ${state.found.length} รูป — เลือกเฉพาะที่ต้องการ`);
    }
    renderWebImages();
    syncWebSelection();
  } catch (_) {
    setWebStatus("เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ", "error");
  } finally {
    dom.fetchBtn.disabled = false;
  }
}
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
    setWebStatus(`⏳ กำลังดาวน์โหลดรูป ${added + 1} จาก ${chosen.length}...`);
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
        source: "web",
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
  dom.webAddBtn.textContent = "➕ เพิ่มรูปที่เลือก (0) ลงในสมุด";
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

/* ---------- ที่ 3: อัปโหลดไฟล์ ---------- */
function addFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (files.length === 0) {
    setUploadStatus("ไม่พบไฟล์ภาพที่รองรับ", "error");
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
      source: "upload",
    });
  }
  if (state.selectedId === null) {
    state.selectedId = state.items[0].id;
  }
  renderItems();
  setBusy(false);
  setUploadStatus(`✅ เพิ่ม ${files.length} ภาพแล้ว`);
  setStatus(`✅ เพิ่ม ${files.length} ภาพแล้ว`);
}
function setUploadStatus(message, kind = "") {
  dom.uploadStatus.textContent = message || "";
  dom.uploadStatus.className = "status" + (kind ? " " + kind : "");
}

/* ---------- การวางรูปจากคลิปบอร์ด ---------- */
const AUTO_NAMES = new Set(["image", "blob", "unnamed", "screenshot"]);
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
function pastedCaption(file, index) {
  const base = file.name.replace(/\.[^.]+$/, "").toLowerCase();
  if (file.name && !AUTO_NAMES.has(base)) {
    return defaultCaption(file.name);
  }
  return "ภาพที่วาง " + index;
}
document.addEventListener("paste", (event) => {
  const files = imagesFromClipboard(event.clipboardData);
  if (files.length === 0) return;
  const tag = document.activeElement && document.activeElement.tagName;
  const inTextField = tag === "INPUT" || tag === "TEXTAREA";
  if (!inTextField) event.preventDefault();
  const startIndex = state.items.length + 1;
  const prepared = files.map((file, index) => ({
    id: state.nextId++,
    file,
    name: file.name || "pasted-image",
    caption: pastedCaption(file, startIndex + index),
    url: URL.createObjectURL(file),
    warning: null,
    source: "upload",
  }));
  state.items.push(...prepared);
  if (state.selectedId === null) state.selectedId = state.items[0].id;
  renderItems();
  setBusy(false);
  activateTab(dom.tabUpload);
  setUploadStatus(
    `วางรูปจากคลิปบอร์ด ${files.length} รูป` +
      (inTextField ? " (กดลอกช่องอื่นก่อนวางรูปได้สะดวกขึ้น)" : "")
  );
});

/* ---------- จัดการรายการในสมุด ---------- */
function removeItem(id) {
  const index = state.items.findIndex((i) => i.id === id);
  if (index === -1) return;
  const item = state.items[index];
  if (item.url.startsWith("blob:")) URL.revokeObjectURL(item.url);
  state.items.splice(index, 1);
  if (state.selectedId === id) {
    state.selectedId = state.items.length ? state.items[0].id : null;
  }
  dom.previewWrap.hidden = true;
  renderItems();
  setBusy(false);
}
function clearAll() {
  for (const item of state.items) {
    if (item.url.startsWith("blob:")) URL.revokeObjectURL(item.url);
  }
  state.items = [];
  state.selectedId = null;
  dom.previewWrap.hidden = true;
  clearPages();
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
    skip_convert: dom.skipConvert.checked,
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
  /* ค่าทั้งสองเป็น dropdown แล้ว ป้ายบอกค่าอยู่ในตัวเลือกเอง ไม่ต้องซิงก์แยก */
}
async function readError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail) && data.detail[0]) {
      return data.detail[0].msg || "คำขอไม่ถูกต้อง";
    }
  } catch (_) {}
  return "ทำรายการไม่สำเร็จ (รหัส " + response.status + ")";
}
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
/* ---------- การเรียกเซิร์ฟเวอร์: ตัวอย่าง แปลงรูป และรวมเล่ม ---------- */
async function preview() {
  const item = state.items.find((i) => i.id === state.selectedId) || state.items[0];
  if (!item) return;
  setBusy(true, "preview");
  setStatus("⏳ กำลังสร้างตัวอย่าง...");
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
      setStatus("⚠️ " + notice, "warn");
    } else {
      setStatus("✅ ตัวอย่างพร้อมแล้ว");
    }
  } catch (_) {
    setStatus("เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ", "error");
  } finally {
    setBusy(false);
  }
}
/* ขั้นที่ 1: แปลงทีละรูป (คำขอสั้น เห็นความคืบหน้า และผลถูกเก็บไว้ที่เซิร์ฟเวอร์) */
async function convertAll() {
  if (state.items.length === 0) return false;
  const key = convertKey();
  const todo = state.items.filter((item) => item.convertedKey !== key);
  if (todo.length === 0) {
    setStatus("✅ แปลงครบทุกรูปแล้ว กด 'รวมเล่ม' ได้เลย");
    return true;
  }
  setBusy(true, "convert");
  let ok = 0;
  let failed = 0;
  try {
    for (let i = 0; i < todo.length; i++) {
      const item = todo[i];
      setStatus(`⏳ กำลังแปลงรูปที่ ${i + 1} จาก ${todo.length}: ${item.name}`);
      const body = new FormData();
      body.append("file", item.file, item.name);
      body.append("lineart", key);
      try {
        const response = await fetch("/api/convert", { method: "POST", body });
        if (!response.ok) {
          item.warning = await readError(response);
          failed++;
          continue;
        }
        const data = await response.json();
        item.warning = data.notice || null;
        item.lineThumb = data.thumb;
        item.convertedKey = key;
        ok++;
      } catch (_) {
        item.warning = "เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ";
        failed++;
      }
      renderItems();
    }
  } finally {
    renderItems();
    setBusy(false);
  }
  if (failed > 0) {
    setStatus(`แปลงสำเร็จ ${ok} รูป · ไม่สำเร็จ ${failed} รูป (ดูข้อความใต้รูปที่มีปัญหา)`, "warn");
  } else {
    setStatus(`✅ แปลงสำเร็จ ${ok} รูป — กด 'รวมเล่ม' ได้เลย`);
  }
  return ok > 0;
}

/* ขั้นที่ 2: รวมเล่ม ได้ภาพ "ทุกหน้า" มาให้ดูและสั่งพิมพ์ */
async function makeBook() {
  if (state.items.length === 0) return;
  // ถ้ายังมีรูปที่ยังไม่ได้แปลงด้วยค่าปัจจุบัน ให้แปลงให้ก่อนโดยอัตโนมัติ
  if (state.items.some((item) => !isConverted(item))) {
    await convertAll();
  }
  const usable = state.items.filter((item) => isConverted(item));
  if (usable.length === 0) {
    setStatus("ไม่มีรูปที่แปลงสำเร็จ จึงรวมเล่มไม่ได้", "error");
    return;
  }
  setBusy(true, "book");
  clearPages();
  setStatus("⏳ กำลังรวมเล่ม...");
  const body = new FormData();
  for (const item of usable) body.append("files", item.file, item.name);
  body.append("captions", JSON.stringify(usable.map((i) => i.caption)));
  body.append("lineart", convertKey());
  body.append("book", JSON.stringify(bookPayload()));
  try {
    const response = await fetch("/api/pages", { method: "POST", body });
    if (!response.ok) {
      setStatus(await readError(response), "error");
      return;
    }
    const data = await response.json();
    showPages(data.pages);
    dom.pagesCard.scrollIntoView({ behavior: "smooth", block: "start" });
    const extra = (data.warnings || []).concat(data.skipped || []).join(" · ");
    setStatus(
      `✅ รวมเล่มแล้ว ${data.count} หน้า — ตรวจดูแล้วกด 'พิมพ์ทุกหน้า'` + (extra ? " — " + extra : ""),
      extra ? "warn" : ""
    );
  } catch (_) {
    setStatus("เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ", "error");
  } finally {
    setBusy(false);
  }
}

/* สั่งพิมพ์ทุกหน้าจากภาพที่รวมเล่มแล้ว ไม่ต้องใช้ไฟล์ PDF */
let printFrame = null;
function printPages() {
  if (state.pages.length === 0) {
    setStatus("ยังไม่ได้รวมเล่ม กด 'รวมเล่ม' ก่อนพิมพ์", "warn");
    return;
  }
  if (printFrame) printFrame.remove();
  const frame = document.createElement("iframe");
  frame.setAttribute("aria-hidden", "true");
  frame.style.cssText =
    "position:fixed;right:0;bottom:0;width:1px;height:1px;border:0;opacity:0;";
  const pagesHtml = state.pages
    .map((src) => '<div class="pg"><img src="' + src + '"></div>')
    .join("");
  frame.srcdoc =
    '<!doctype html><html><head><meta charset="utf-8"><style>' +
    "@page{size:A4;margin:0}" +
    "html,body{margin:0;padding:0;background:#fff}" +
    ".pg{width:210mm;height:296mm;overflow:hidden;break-after:page;page-break-after:always}" +
    ".pg:last-child{break-after:auto;page-break-after:auto}" +
    ".pg img{display:block;width:210mm;height:296mm;object-fit:contain}" +
    "</style></head><body>" + pagesHtml + "</body></html>";
  frame.addEventListener("load", async () => {
    try {
      const win = frame.contentWindow;
      await Promise.all(
        Array.from(win.document.images).map((img) =>
          img.complete ? Promise.resolve() : new Promise((r) => { img.onload = r; img.onerror = r; })
        )
      );
      win.focus();
      win.print();
    } catch (err) {
      setStatus("สั่งพิมพ์ไม่สำเร็จ: " + err, "error");
    }
  });
  document.body.appendChild(frame);
  printFrame = frame;
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
  initTabs();
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
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
  dom.clear.addEventListener("click", clearAll);
  dom.previewBtn.addEventListener("click", preview);
  dom.convertBtn.addEventListener("click", convertAll);
  dom.bookBtn.addEventListener("click", makeBook);
  dom.printBtn.addEventListener("click", printPages);
  // เปลี่ยนค่าที่มีผลกับการแปลง → ป้าย "แปลงแล้ว" ของรูปต้องอัปเดต
  dom.skipConvert.addEventListener("change", () => {
    renderItems();
    setBusy(state.busy);
  });
  for (const control of [dom.line, dom.speckle, dom.useAi]) {
    control.addEventListener("change", () => renderItems());
  }
  // ดึงภาพจากเว็บ
  dom.fetchBtn.addEventListener("click", fetchWebImages);
  dom.webAddBtn.addEventListener("click", addSelectedWebImages);
  dom.webSelectAll.addEventListener("click", () => {
    const cards = Array.from(dom.webImages.children);
    const allOn = cards.every((card) => card.classList.contains("on"));
    for (const card of cards) card.classList.toggle("on", !allOn);
    syncWebSelection();
  });
  dom.webUrl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      fetchWebImages();
    }
  });
  // สร้างภาพด้วย AI
  dom.genBtn.addEventListener("click", generateImages);
  dom.genAddBtn.addEventListener("click", addGeneratedToBook);
  dom.genPrompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      generateImages();
    }
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
