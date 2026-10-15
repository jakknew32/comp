"""คิวงานเบื้องหลังสำหรับงานหนัก (รวมเล่ม) พร้อมรายงานความคืบหน้า

ทำไมต้องมี
  เดิมการรวมเล่มทำจบในคำขอเดียว ถ้าเล่มใหญ่ ใช้เวลานานหรือกินหน่วยความจำมาก
  ตัวกลางของ Render จะตัดการเชื่อมต่อแล้วหน้าเว็บได้ 502 โดยไม่เห็นความคืบหน้าเลย
  ตอนนี้คำขอแรกแค่ "ส่งงานเข้าคิว" แล้วตอบกลับทันที หน้าเว็บถามสถานะเป็นระยะ
  จึงไม่มีคำขอไหนค้างนาน และเห็นเปอร์เซ็นต์จริงว่าทำถึงหน้าไหนแล้ว

ข้อจำกัดโดยตั้งใจ (เหมาะกับแผนฟรีที่ RAM น้อย)
  - ทำงานทีละงานเท่านั้น (worker เดียว) เพื่อไม่ให้หลายงานแย่งหน่วยความจำพร้อมกัน
  - รับงานที่ยังไม่จบได้จำกัดจำนวน เกินแล้วตอบ 429 ให้ลองใหม่
  - เก็บผลไว้ในหน่วยความจำชั่วคราวแล้วลบเองเมื่อหมดเวลา ไม่เขียนลงดิสก์
  - ถ้าเซิร์ฟเวอร์รีสตาร์ท งานที่ค้างอยู่จะหายไป หน้าเว็บจะได้ 404 ตอนถามสถานะ
    และแจ้งผู้ใช้ให้กดรวมเล่มใหม่
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("coloring_book")

# ผลของงานที่เสร็จแล้วถูกเก็บไว้ให้ดึงได้ภายในเวลานี้ แล้วถูกลบทิ้ง
JOB_TTL_SECONDS = 15 * 60
# งานที่ยังรอหรือกำลังทำ รับได้พร้อมกันไม่เกินนี้ (รวมทุกผู้ใช้)
MAX_ACTIVE_JOBS = 4
# จำนวนงานทั้งหมดที่เก็บไว้ในหน่วยความจำ (รวมงานที่เสร็จแล้ว)
MAX_STORED_JOBS = 10


class JobBusy(Exception):
    """คิวเต็ม ยังรับงานใหม่ไม่ได้"""


class JobError(Exception):
    """งานล้มเหลวด้วยเหตุที่รู้อยู่แล้ว พร้อมรหัสสถานะที่จะบอกหน้าเว็บ"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class Job:
    id: str
    state: str = "queued"  # queued | running | done | error
    done: int = 0
    total: int = 0
    message: str = ""
    result: Any = None
    error: dict | None = None
    created: float = field(default_factory=time.monotonic)
    finished: float | None = None


class Progress:
    """ตัวส่งความคืบหน้าให้ฟังก์ชันของงานเรียกใช้ระหว่างทำ"""

    def __init__(self, store: "JobStore", job: Job) -> None:
        self._store = store
        self._job = job

    def set_total(self, total: int) -> None:
        with self._store.lock:
            self._job.total = max(0, int(total))

    def advance(self, message: str = "", step: int = 1) -> None:
        with self._store.lock:
            done = self._job.done + step
            self._job.done = min(self._job.total, done) if self._job.total else done
            if message:
                self._job.message = message

    def note(self, message: str) -> None:
        with self._store.lock:
            self._job.message = message


class JobStore:
    def __init__(
        self,
        workers: int = 1,
        ttl: float = JOB_TTL_SECONDS,
        max_active: int = MAX_ACTIVE_JOBS,
        max_stored: int = MAX_STORED_JOBS,
    ) -> None:
        self.lock = threading.Lock()
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._ttl = ttl
        self._max_active = max_active
        self._max_stored = max_stored

    # --- ภายใน ----------------------------------------------------------

    def _purge_locked(self) -> None:
        """ลบงานที่เสร็จแล้วและหมดอายุ และตัดงานที่เสร็จแล้วเก่าที่สุดถ้าเก็บเกินเพดาน"""
        now = time.monotonic()
        for job_id in [
            j.id for j in self._jobs.values()
            if j.finished is not None and now - j.finished > self._ttl
        ]:
            self._jobs.pop(job_id, None)
        while len(self._jobs) > self._max_stored:
            finished = [j.id for j in self._jobs.values() if j.finished is not None]
            if not finished:
                break
            self._jobs.pop(finished[0], None)

    def _active_count_locked(self) -> int:
        return sum(1 for j in self._jobs.values() if j.finished is None)

    def _run(self, job: Job, fn: Callable[[Progress], Any]) -> None:
        with self.lock:
            job.state = "running"
        progress = Progress(self, job)
        try:
            result = fn(progress)
        except JobError as exc:
            with self.lock:
                job.error = {"status": exc.status, "detail": exc.detail}
                job.state = "error"
                job.finished = time.monotonic()
            return
        except Exception:  # noqa: BLE001 - งานล้มต้องไม่ทำให้ worker ตาย
            logger.exception("job %s failed", job.id)
            with self.lock:
                job.error = {"status": 500, "detail": "สร้างเล่มไม่สำเร็จ ลองใหม่อีกครั้ง หรือลดจำนวนรูปลง"}
                job.state = "error"
                job.finished = time.monotonic()
            return
        with self.lock:
            job.result = result
            job.done = job.total or job.done
            job.state = "done"
            job.finished = time.monotonic()

    # --- ใช้งาน ----------------------------------------------------------

    def submit(self, fn: Callable[[Progress], Any], total: int = 0) -> Job:
        with self.lock:
            self._purge_locked()
            if self._active_count_locked() >= self._max_active:
                raise JobBusy
            job = Job(id=uuid.uuid4().hex, total=total)
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, fn)
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            self._purge_locked()
            return self._jobs.get(job_id)

    def snapshot(self, job: Job) -> dict:
        """สถานะที่ส่งให้หน้าเว็บ (ไม่รวมผลลัพธ์ก้อนใหญ่)"""
        with self.lock:
            position = 0
            if job.state == "queued":
                position = 1 + sum(
                    1 for j in self._jobs.values()
                    if j.state == "queued" and j.created < job.created
                )
            data = {
                "state": job.state,
                "done": job.done,
                "total": job.total,
                "message": job.message,
                "queue_position": position,
            }
            if job.error:
                data["error"] = dict(job.error)
            return data

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
