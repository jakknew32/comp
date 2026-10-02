"""ตั้งค่า AI จากตัวแปรแวดล้อมของเซิร์ฟเวอร์

โปรแกรมไม่ทำงาน AI ให้อยู่ดี ๆ ต้องมีคีย์ของผู้ใช้เสมอ
การไม่มีคีย์ไม่ถือเป็นข้อผิดพลาด เพียงแต่ปิดฟีเจอร์นี้ไว้
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ตัวแปรแวดล้อมที่ใช้ตั้งค่า
ENV_API_KEY = "AI_API_KEY"
ENV_MODEL = "AI_MODEL"
ENV_BASE_URL = "AI_BASE_URL"
ENV_TIMEOUT = "AI_TIMEOUT_SECONDS"

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT = 90.0

# ประมาณการค่าใช้จ่ายต่อหนึ่งภาพ ใช้แสดงในหน้าเว็บให้ผู้ใช้ตัดสินใจ
# ตัวเลขนี้มาจากหน้าราคาของ Google ณ เวลาที่เขียนโค้ด และอาจเปลี่ยนได้
# จึงต้องบอกผู้ใช้เสมอว่าให้เช็คราคาล่าสุดที่ต้นทาง
ESTIMATED_COST_PER_IMAGE_USD = 0.067

PROMPT_TH = (
    "แปลงภาพนี้ให้เป็นภาพระบายสีสำหรับเด็ก "
    "ต้องเป็นภาพเส้นดำบนพื้นหลังขาวเท่านั้น "
    "เส้นต้องหนาและเรียบร้อย เหมือนภาพวาดด้วยมือ "
    "ห้ามมีสี ห้ามมีเงา ไม่มีพื้นหลังสีหรือพื้นหลังมีลาย "
    "รักษารูปร่างและรายละเอียดของสิ่งที่อยู่ในภาพเดิมไว้ทั้งหมด"
)


@dataclass
class AiSettings:
    """ผลการอ่านตั้งค่า AI จากสภาพแวดล้อม"""

    api_key: str | None
    model: str
    base_url: str
    timeout: float

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def describe_missing(self) -> str:
        return (
            f"ยังไม่ได้ตั้งค่า AI: ไม่พบตัวแปรแวดล้อม {ENV_API_KEY} "
            "ต้องใส่คีย์ของผู้ให้บริการในการตั้งค่าของ Render"
        )

    def masked_key(self) -> str:
        if not self.api_key:
            return ""
        tail = self.api_key[-4:]
        return "***" + tail


def load_settings(env: dict | None = None) -> AiSettings:
    """อ่านค่าจากตัวแปรแวดล้อม

    รับ dict เข้ามาเพื่อให้เทสต์ได้โดยไม่ต้องแก้สภาพแวดล้อมจริง
    """
    source = os.environ if env is None else env

    key = (source.get(ENV_API_KEY) or "").strip() or None
    model = (source.get(ENV_MODEL) or "").strip() or DEFAULT_MODEL
    base_url = (source.get(ENV_BASE_URL) or "").strip() or DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")

    timeout_raw = (source.get(ENV_TIMEOUT) or "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT
    except ValueError:
        timeout = DEFAULT_TIMEOUT
    timeout = min(max(timeout, 5.0), 300.0)

    return AiSettings(
        api_key=key,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )
