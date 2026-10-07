/**
 * ทดสอบตรรกะการดึงรูปจากคลิปบอร์ด โดยจำลอง DOM แบบเบา�ๆ
 *
 * รันด้วย: node tests/test_clipboard.mjs
 * ต้องผ่านก่อนจึงจะมั่นใจว่าการวางรูปทำงานกับข้อมูลคลิปบอร์ด
 * ของแต่ละเบราว์เซอร์ได้ถูกต้อง
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "..", "static", "app.js"), "utf8");

// ดึงเฉพาะส่วนที่ต้องการทดสอบออกมา แล้วรันในสภาพแวดล้อมจำลอง
// วิธีนี้ไม่ต้องมี DOM จริง และไม่กระทบกับโค้ดที่ทำงานในเบราว์เซอร์
// AUTO_NAMES ถูกส่งเข้าไปเป็นพารามิเตอร์แทนการประกาศค่าซ้ำ
// จำลองสภาพแวดล้อมของเบราว์เซอร์พอให้โค้ดส่วนนี้ทำงานได้
// โดยจับฟังก์ชันที่ลงทะเบียนไว้ เพื่อนำมาเรียกทดสอบจริง
const registered = {};
const stubState = { items: [], selectedId: null };
let lastStatus = "";

const sandbox = {
  AUTO_NAMES: new Set(["image", "blob", "unnamed", "screenshot"]),
  defaultCaption: (name) => {
    const dot = name.lastIndexOf(".");
    return (dot > 0 ? name.slice(0, dot) : name) || "ระบายสี";
  },
  document: {
    addEventListener: (type, fn) => {
      registered[type] = fn;
    },
    activeElement: null,
  },
  state: stubState,
  URL: { createObjectURL: (f) => "blob:" + (f.name || "x") },
  renderItems: () => {},
  setBusy: () => {},
  setStatus: (msg) => {
    lastStatus = msg;
  },
  setUploadStatus: (msg) => {
    lastStatus = msg;
  },
  activateTab: () => {},
  dom: { tabUpload: {} },
};

const body = source
  .slice(
    source.indexOf("const AUTO_NAMES"),
    source.indexOf("/* ---------- จัดการรายการในสมุด ---------- */")
  )
  .replace(/^const AUTO_NAMES = .*$/m, "");

const load = new Function(
  ...Object.keys(sandbox),
  body + "\nreturn { imagesFromClipboard, pastedCaption };"
);

const { imagesFromClipboard, pastedCaption } = load(
  ...Object.values(sandbox)
);

assert.equal(typeof registered.paste, "function", "ต้องมีตัวฟังเหตุการณ์ paste");

/* ---------- เตรียมตัวช่วยสร้างข้อมูลคลิปบอร์ดปลอม ---------- */

function fileItem(type, name) {
  return { kind: "file", type, getAsFile: () => ({ type, name }) };
}

/* ---------- กรณีที่ต้องผ่าน ---------- */

// 1. Chrome/Edge ใส่ไฟล์ไว้ใน clipboardData.files
{
  const data = {
    files: [{ type: "image/png", name: "a.png" }],
    items: [],
  };
  assert.equal(imagesFromClipboard(data).length, 1, "ต้องอ่านได้จาก files");
}

// 2. บางเบราว์เซอร์ใช้ items แทน files
{
  const data = {
    files: [],
    items: [fileItem("image/png", "b.png")],
  };
  assert.equal(imagesFromClipboard(data).length, 1, "ต้องอ่านได้จาก items");
}

// 3. มีทั้งสองทาง และมีหลายรูป
{
  const data = {
    files: [
      { type: "image/png", name: "1.png" },
      { type: "image/jpeg", name: "2.jpg" },
    ],
    items: [fileItem("image/gif", "3.gif")],
  };
  assert.equal(imagesFromClipboard(data).length, 2, "ต้องรวมเฉพาะที่มาจาก files");
}

// 4. วางข้อความธรรมดา ต้องไม่ได้รูป
{
  const data = {
    files: [{ type: "text/plain", name: "" }],
    items: [{ kind: "string", type: "text/plain", getAsFile: () => null }],
  };
  assert.equal(imagesFromClipboard(data).length, 0, "ข้อความต้องไม่ถูกนับเป็นรูป");
}

// 5. คลิปบอร์ดว่าง
assert.equal(imagesFromClipboard(null).length, 0, "null ต้องไม่พัง");
assert.equal(imagesFromClipboard({}).length, 0, "ว่างเปล่าต้องไม่พัง");
assert.equal(
  imagesFromClipboard({ files: [], items: [] }).length,
  0,
  "ว่างทั้งสองทางต้องไม่พัง"
);

// 6. items ที่ getAsFile คืน null ต้องถูกข้าม ไม่ใช่พัง
{
  const data = {
    files: [],
    items: [
      { kind: "file", type: "image/png", getAsFile: () => null },
      fileItem("image/png", "ok.png"),
    ],
  };
  assert.equal(imagesFromClipboard(data).length, 1, "null ต้องถูกข้าม");
}

// 7. ชื่อกำกับของรูปที่วาง
{
  assert.equal(pastedCaption({ name: "image.png" }, 3), "ภาพที่วาง 3");
  assert.equal(pastedCaption({ name: "blob" }, 1), "ภาพที่วาง 1");
  assert.equal(pastedCaption({ name: "" }, 2), "ภาพที่วาง 2");
  // ชื่อจริงที่มีความหมาย ควรถูกใช้แทน
  assert.equal(pastedCaption({ name: "ยีระแหน.png" }, 4), "ยีระแหน");
}

/* ---------- ทดสอบตัวฟังเหตุการณ์ paste จริง ---------- */

/** เรียกตัวฟัง paste ด้วยข้อมูลคลิปบอร์ดปลอม แล้วคืน state หลังทำงาน */
function firePaste(clipboardData, activeElement = null) {
  sandbox.document.activeElement = activeElement;
  let prevented = false;
  registered.paste({
    clipboardData,
    preventDefault: () => {
      prevented = true;
    },
  });
  return { prevented, items: stubState.items, status: lastStatus };
}

// 1. วางรูปที่คัดลอกมา ต้องเพิ่มเข้ารายการ
{
  const before = stubState.items.length;
  const result = firePaste({
    files: [{ type: "image/png", name: "image.png" }],
    items: [],
  });
  assert.equal(result.items.length, before + 1, "ต้องเพิ่มรูปเข้ารายการ");
  assert.equal(result.prevented, true, "ต้องกันพฤติกรรมเดิมของเบราว์เซอร์");
  assert.ok(result.status.includes("คลิปบอร์ด"), "ต้องแจ้งผู้ใช้ว่าวางสำเร็จ");
  stubState.items.length = before;
}

// 2. วางหลายรูปพร้อมกัน
{
  stubState.items.length = 0;
  firePaste({
    files: [
      { type: "image/png", name: "image.png" },
      { type: "image/jpeg", name: "image.jpg" },
    ],
    items: [],
  });
  assert.equal(stubState.items.length, 2, "ต้องเพิ่มครบทุกรูป");
  // ชื่อกำกับต้องไม่ซ้ำกัน
  const captions = stubState.items.map((i) => i.caption);
  assert.equal(new Set(captions).size, 2, "ชื่อกำกับต้องแยกกัน");
  stubState.items.length = 0;
}

// 3. วางข้อความธรรมดา ต้องไม่เพิ่มอะไรและไม่กันพฤติกรรมเดิม
{
  const result = firePaste({
    files: [{ type: "text/plain", name: "" }],
    items: [{ kind: "string", type: "text/plain", getAsFile: () => null }],
  });
  assert.equal(result.items.length, 0, "ข้อความต้องไม่ถูกเพิ่มเข้ารายการ");
  assert.equal(result.prevented, false, "ต้องปล่อยให้วางข้อความตามปกติ");
}

// 4. อยู่ในช่องพิมพ์ชื่อกำกับแล้ววางรูป ต้องไม่ขัดจังหวะการวางข้อความ
{
  stubState.items.length = 0;
  const input = { tagName: "INPUT" };
  const result = firePaste(
    { files: [{ type: "image/png", name: "a.png" }], items: [] },
    input
  );
  assert.equal(result.items.length, 1, "ยังต้องเพิ่มรูปได้");
  assert.equal(
    result.prevented,
    false,
    "ห้ามกันวางข้อความในช่องพิมพ์ ไม่งั้นผู้ใช้พิมพ์ไม่ได้"
  );
  stubState.items.length = 0;
}

// 5. รูปที่วางต้องมี url สำหรับแสดงรูปย่อ และเลือกไว้ให้แล้ว
{
  stubState.items.length = 0;
  stubState.selectedId = null;
  firePaste({ files: [{ type: "image/png", name: "x.png" }], items: [] });
  assert.ok(stubState.items[0].url, "ต้องมี url สำหรับแสดงรูปย่อ");
  assert.equal(stubState.selectedId, stubState.items[0].id, "ต้องเลือกรูปที่วางไว้ให้");
  stubState.items.length = 0;
}

console.log("ผ่านทั้งหมด: การดึงรูปจากคลิปบอร์ดทำงานถูกต้อง");
