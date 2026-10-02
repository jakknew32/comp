/* ตรรกะฝั่งเบราว์เซอร์ของโปรแกรมสร้างสมุดระบายสี A4
 *
 * หลักการสำคัญ
 * - ใส่ชื่อไฟล์/ชื่อกำกับลง DOM ด้วย textContent เท่านั้น ห้ามใช้ innerHTML
 *   กับข้อมูลจากผู้ใช้ เพราะเป็นความเสี่ยง XSS (ความเสี่ยงข้อ 7 ของแผน)
 * - พารามิเตอร์ที่ "อัตโนมัติ" ส่งเป็น null ไปให้เซิร์ฟเวอร์ตัดสินใจเอง
 * - ตัวเลือก "อัตโนมัติ" เป็นค่าเริ่มต้นของทุก slider
 */
'use strict';

const $ = (id) => document.getElementById(id);

/** รายการไฟล์ที่ผู้ใช้เลือก: {id, file, name, caption, objectUrl, report} */
let items = [];
let nextId = 1;
let busy = false;

/* ------------------------------------------------------------------ */
/* สถานะและข้อความ                                                   */
/* ------------------------------------------------------------------ */

function setStatus(message, kind) {
  const el = $('status');
  el.textContent = message || '';
  el.className = 'status' + (kind ? ' status-' + kind : '');
}

/* ------------------------------------------------------------------ */
/* ค่าจากฟอร์ม                                                        */
/* ------------------------------------------------------------------ */

/** อ่านค่าพารามิเตอร์ปัจจุบัน ค่า null = ให้เซิร์ฟเวอร์ตัดสินใจเอง */
function currentOptions() {
  const line = parseFloat($('target-line').value);
  const speck = parseFloat($('despeckle').value);

  return {
    // ค่าเริ่มต้นของ slider คือค่ากลาง → ถือว่าผู้ใช้ยังไม่ได้ปรับ
    // ให้ส่ง null เพื่อให้เซิร์ฟเวอร์วัดเอง
    target_line_mm: line === autoLineValue() ? null : line,
    despeckle: speck === 0 ? null : speck,
  };
}

function autoLineValue() {
  return parseFloat($('target-line').dataset.center || '2.5');
}

function perPage() {
  return parseInt($('per-page').value, 10) || 1;
}

function captions() {
  return items.map((it) => it.caption);
}

/* ------------------------------------------------------------------ */
/* แสดงค่าที่ "อัตโนมัติ" ในป้ายกำกับ slider                            */
/* ------------------------------------------------------------------ */

function updateSliderLabels() {
  const line = parseFloat($('target-line').value);
  const lineLabel = $('target-line-value');
  lineLabel.textContent =
    line === autoLineValue() ? 'อัตโนมัติ' : line.toFixed(1) + ' มม.';

  const speck = parseFloat($('despeckle').value);
  $('despeckle-value').textContent =
    speck === 0 ? 'อัตโนมัติ' : speck.toFixed(2);

  $('per-page-value').textContent = String(perPage());
}

/* ------------------------------------------------------------------ */
/* รายการไฟล์                                                          */
/* ------------------------------------------------------------------ */

const MAX_BYTES = 25 * 1024 * 1024;

function addFiles(fileList) {
  const incoming = Array.from(fileList || []);
  let rejected = 0;

  for (const file of incoming) {
    if (!/^image\/(jpeg|jpg|png|webp)$/i.test(file.type)) {
      rejected += 1;
      continue;
    }
    if (file.size > MAX_BYTES) {
      rejected += 1;
      continue;
    }
    items.push({
      id: nextId++,
      file: file,
      name: file.name,
      caption: defaultCaption(file.name),
      objectUrl: URL.createObjectURL(file),
      report: null,
    });
  }

  renderFileList();

  if (rejected > 0) {
    setStatus(
      'ข้าม ' + rejected + ' ไฟล์ (ชนิดไม่รองรับหรือเกิน 25 MB)',
      'err'
    );
  } else {
    setStatus('');
  }

  refreshButtons();
  if (items.length > 0) {
    analyseAll();
  }
}

/** ตัดนามสกุลออกจากชื่อไฟล์ เพื่อใช้เป็นชื่อกำกับเริ่มต้น */
function defaultCaption(name) {
  const stem = String(name).replace(/\.(png|jpe?g|webp|bmp|tiff?)$/i, '');
  const cleaned = stem.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim();
  return cleaned || 'ระบายสี';
}

function removeItem(id) {
  const index = items.findIndex((it) => it.id === id);
  if (index === -1) return;
  URL.revokeObjectURL(items[index].objectUrl);
  items.splice(index, 1);
  renderFileList();
  refreshButtons();
  if (items.length > 0) {
    analyseAll();
  } else {
    $('preview').hidden = true;
    $('preview-note').hidden = false;
    $('auto-info').textContent = '';
    setStatus('');
  }
}

function renderFileList() {
  const list = $('file-list');
  list.replaceChildren();

  for (const item of items) {
    const li = document.createElement('li');
    li.className = 'file-item';

    // ภาพตัวอย่าง
    const img = document.createElement('img');
    img.className = 'thumb';
    img.src = item.objectUrl;
    img.alt = 'ตัวอย่าง ' + item.name;
    li.appendChild(img);

    // ข้อมูล + ช่องแก้ชื่อกำกับ
    const meta = document.createElement('div');
    meta.className = 'file-meta';

    const name = document.createElement('div');
    name.className = 'file-name';
    // ใช้ textContent เสมอ — ชื่อไฟล์มาจากผู้ใช้ ห้ามแทรกเป็น HTML
    const strong = document.createElement('b');
    strong.textContent = item.name;
    name.appendChild(strong);
    meta.appendChild(name);

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'caption-input';
    input.value = item.caption;
    input.maxLength = 80;
    input.setAttribute('aria-label', 'ชื่อกำกับสำหรับ ' + item.name);
    input.addEventListener('input', (e) => {
      item.caption = e.target.value;
    });
    meta.appendChild(input);

    meta.appendChild(buildStatusBadge(item));
    li.appendChild(meta);

    // ปุ่มลบ
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'icon-btn';
    del.textContent = '×';
    del.title = 'ลบ ' + item.name;
    del.setAttribute('aria-label', 'ลบ ' + item.name);
    del.addEventListener('click', () => removeItem(item.id));
    li.appendChild(del);

    list.appendChild(li);
  }

  $('file-summary').textContent = items.length
    ? 'เลือก ' + items.length + ' ไฟล์'
    : 'ยังไม่ได้เลือกไฟล์';
}

function buildStatusBadge(item) {
  const holder = document.createElement('div');

  if (!item.report) {
    const pending = document.createElement('span');
    pending.className = 'badge badge-warn';
    pending.textContent = 'กำลังตรวจ…';
    holder.appendChild(pending);
    return holder;
  }

  const det = item.report.detection;
  const badge = document.createElement('span');

  if (det.is_line_art) {
    badge.className = 'badge badge-ok';
    badge.textContent = '✓ ภาพลายเส้น';
  } else {
    badge.className = 'badge badge-warn';
    badge.textContent = '! อาจเป็นภาพถ่าย';
  }
  holder.appendChild(badge);

  const note = document.createElement('span');
  note.className = 'badge-note';
  const parts = [];
  if (det.low_resolution) parts.push('ความละเอียดต่ำ');
  if (item.report.warnings && item.report.warnings.length) {
    parts.push(item.report.warnings.length + ' คำเตือน');
  }
  if (parts.length) {
    note.textContent = parts.join(' · ');
    holder.appendChild(note);
  }
  return holder;
}

function refreshButtons() {
  $('download').disabled = items.length === 0 || busy;
  $('regen').disabled = items.length === 0 || busy;
}

/* ------------------------------------------------------------------ */
/* การเรียก API                                                        */
/* ------------------------------------------------------------------ */

function fileEntries() {
  return items.map((it) => ['files', it.file, it.name]);
}

/** เรียก /api/book แบบ dry_run เพื่อดูสถานะตรวจประเภทภาพทั้งหมด */
async function analyseAll() {
  if (items.length === 0) return;

  setStatus('กำลังตรวจประเภทภาพ…');
  const form = new FormData();
  for (const entry of fileEntries()) form.append(entry[0], entry[1], entry[2]);
  form.append('options', JSON.stringify(currentOptions()));
  form.append('captions', JSON.stringify(captions()));
  form.append('per_page', String(perPage()));
  form.append('dry_run', '1');

  try {
    const res = await fetch('/api/book', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'เรียกเซิร์ฟเวอร์ไม่สำเร็จ');
    }
    const data = await res.json();
    data.files.forEach((rep) => {
      if (items[rep.index]) items[rep.index].report = rep;
    });
    renderFileList();
    describeAuto(data);
    setStatus(
      'ตรวจพบ ' + items.length + ' ไฟล์ · จะได้ ' +
        data.total_pages_including_cover + ' หน้า',
      'ok'
    );
  } catch (err) {
    setStatus(err.message, 'err');
  }
}

/** แสดงค่าที่เซิร์ฟเวอร์คำนวณจริง เพื่อให้ผู้ใช้เห็นว่า "ตอนนี้ใช้ค่าอะไร" */
function describeAuto(data) {
  if (!data.files || !data.files.length) {
    $('auto-info').textContent = '';
    return;
  }
  const first = data.files[0].auto_values || {};
  const strokes = data.files
    .map((f) => f.stroke && f.stroke.measured_px)
    .filter((v) => typeof v === 'number');

  const bits = [];
  if (typeof first.target_line_px_on_page === 'number') {
    bits.push(
      'เป้าหมายเส้นบนกระดาษ ' + first.target_line_px_on_page + ' px'
    );
  }
  if (strokes.length) {
    const min = Math.min.apply(null, strokes);
    const max = Math.max.apply(null, strokes);
    bits.push(
      'วัดเส้นต้นฉบับได้ ' + min + '–' + max + ' px (ปรับอัตโนมัติ)'
    );
  }
  if (first.method) bits.push('วิธี: ' + first.method);

  $('auto-info').textContent = bits.join(' · ');
}

/** ดึงตัวอย่างหน้า A4 ของไฟล์แรก (มี cache ฝั่งเซิร์ฟเวอร์อยู่แล้ว) */
async function refreshPreview() {
  if (items.length === 0) return;

  const form = new FormData();
  const first = items[0];
  form.append('file', first.file, first.name);
  form.append('options', JSON.stringify(currentOptions()));
  form.append('per_page', String(perPage()));

  try {
    const res = await fetch('/api/preview', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'ดูตัวอย่างไม่สำเร็จ');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const img = $('preview');
    if (img.dataset.objectUrl) URL.revokeObjectURL(img.dataset.objectUrl);
    img.src = url;
    img.dataset.objectUrl = url;
    img.hidden = false;
    $('preview-note').hidden = true;
  } catch (err) {
    setStatus(err.message, 'err');
  }
}

/** ส่งคำขอดาวน์โหลด PDF */
async function downloadPdf() {
  if (items.length === 0 || busy) return;

  busy = true;
  refreshButtons();
  setStatus('กำลังสร้าง PDF…');

  const form = new FormData();
  for (const entry of fileEntries()) form.append(entry[0], entry[1], entry[2]);
  form.append('options', JSON.stringify(currentOptions()));
  form.append('captions', JSON.stringify(captions()));
  form.append('per_page', String(perPage()));
  form.append('frame', $('opt-frame').checked ? 'true' : 'false');
  form.append('cover', $('opt-cover').checked ? 'true' : 'false');
  form.append('show_captions', $('opt-captions').checked ? 'true' : 'false');
  form.append('page_numbers', $('opt-page-numbers').checked ? 'true' : 'false');
  form.append('book_title', $('book-title').value);
  form.append('author', $('author').value);

  try {
    const res = await fetch('/api/book', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'สร้าง PDF ไม่สำเร็จ');
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    // ชื่อไฟล์มาจากเซิร์ฟเวอร์ที่สร้าง slug ให้เอง ไม่ใช่ชื่อดิบจากผู้ใช้
    const disposition = res.headers.get('content-disposition') || '';
    const match = /filename="([^"]+)"/.exec(disposition);
    const filename = match ? match[1] : 'coloring-book.pdf';

    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);

    const pages = res.headers.get('x-page-count');
    setStatus(
      'ดาวน์โหลดแล้ว' + (pages ? ' (' + pages + ' หน้า)' : ''),
      'ok'
    );
  } catch (err) {
    setStatus(err.message, 'err');
  } finally {
    busy = false;
    refreshButtons();
  }
}

/* ------------------------------------------------------------------ */
/* การผูกเหตุการณ์                                                   */
/* ------------------------------------------------------------------ */

let debounceTimer = null;

function scheduleRegen() {
  updateSliderLabels();
  clearTimeout(debounceTimer);
  // slider ยิงถี่มาก จึงหน่วงเล็กน้อยก่อนยิงจริง
  debounceTimer = setTimeout(() => {
    analyseAll();
    refreshPreview();
  }, 320);
}

function init() {
  const dropzone = $('dropzone');
  const input = $('file-input');

  dropzone.addEventListener('click', () => input.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      input.click();
    }
  });
  $('pick-files').addEventListener('click', (e) => {
    e.stopPropagation();
    input.click();
  });

  input.addEventListener('change', () => {
    addFiles(input.files);
    input.value = '';
  });

  ['dragenter', 'dragover'].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    })
  );
  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files) {
      addFiles(e.dataTransfer.files);
    }
  });

  ['target-line', 'despeckle', 'per-page'].forEach((id) => {
    $(id).addEventListener('input', scheduleRegen);
  });

  ['opt-frame', 'opt-cover', 'opt-captions', 'opt-page-numbers'].forEach((id) => {
    $(id).addEventListener('change', refreshPreview);
  });

  $('regen').addEventListener('click', () => {
    analyseAll();
    refreshPreview();
  });
  $('download').addEventListener('click', downloadPdf);

  refreshButtons();
  loadDefaults();
}

/** ดึงค่าเริ่มต้นและข้อจำกัดจากเซิร์ฟเวอร์ เพื่อไม่ให้สองฝั่งไม่ตรงกัน */
async function loadDefaults() {
  try {
    const [health, scope, validate] = await Promise.all([
      fetch('/api/health').then((r) => r.json()),
      fetch('/api/scope').then((r) => r.json()),
      fetch('/api/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      }).then((r) => r.json()),
    ]);

    if (scope.notice) $('scope-text').textContent = scope.notice;

    if (!health.raqm && scope.shaping_warning) {
      const el = $('shaping-warning');
      el.textContent = scope.shaping_warning;
      el.hidden = false;
    }

    const defaults = validate.defaults || {};
    const line = $('target-line');
    const [lo, hi] = defaults.target_line_mm_range || [1, 5];
    line.min = lo;
    line.max = hi;
    line.step = 0.1;
    line.value = defaults.target_line_mm;
    // จดจำค่ากลางไว้ เพื่อแยก "อัตโนมัติ" ออกจาก "ผู้ใช้ปรักเอง"
    line.dataset.center = defaults.target_line_mm;

    $('book-title').value = defaults.book_title;

    updateSliderLabels();
  } catch (err) {
    // ใช้ค่าใน HTML ไปก่อนได้ ไม่ถือว่าผิดพลาดร้ายแรง
    updateSliderLabels();
  }
}

document.addEventListener('DOMContentLoaded', init);
