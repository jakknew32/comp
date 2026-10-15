"""ทดสอบคิวงานเบื้องหลังและเส้นทาง /api/jobs/*"""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.jobs import JobBusy, JobError, JobStore
from app.main import app
from tests import fixtures

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def png_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def wait_until(predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("รอนานเกินไป")


def finish(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    """ถามสถานะจนงานจบ แล้วคืนสถานะสุดท้าย"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get(f"/api/jobs/{job_id}").json()
        if info["state"] in ("done", "error"):
            return info
        time.sleep(0.05)
    raise AssertionError("งานไม่จบภายในเวลาที่กำหนด")


# --- JobStore ----------------------------------------------------------------


def test_store_runs_job_and_reports_progress() -> None:
    store = JobStore()
    gate = threading.Event()

    def work(progress):
        progress.set_total(4)
        progress.advance("สอง", step=2)
        gate.wait(5)
        progress.advance("สี่", step=2)
        return {"ok": True}

    job = store.submit(work, total=4)
    wait_until(lambda: store.snapshot(job)["done"] == 2)
    mid = store.snapshot(job)
    assert mid["state"] == "running" and mid["total"] == 4 and mid["message"] == "สอง"

    gate.set()
    wait_until(lambda: store.get(job.id).state == "done")
    final = store.snapshot(job)
    assert final["done"] == final["total"] == 4
    assert store.get(job.id).result == {"ok": True}
    store.shutdown()


def test_store_progress_never_exceeds_total() -> None:
    store = JobStore()

    def work(progress):
        progress.set_total(2)
        for _ in range(5):
            progress.advance()
        return {}

    job = store.submit(work)
    wait_until(lambda: store.get(job.id).state == "done")
    assert store.snapshot(job)["done"] == 2
    store.shutdown()


def test_store_reports_known_and_unexpected_errors() -> None:
    store = JobStore()

    def known(_progress):
        raise JobError(422, "ไม่มีภาพใดที่ใช้ได้")

    def unexpected(_progress):
        raise RuntimeError("รายละเอียดภายในต้องไม่หลุดไปถึงผู้ใช้")

    a = store.submit(known)
    wait_until(lambda: store.get(a.id).state == "error")
    assert store.snapshot(a)["error"] == {"status": 422, "detail": "ไม่มีภาพใดที่ใช้ได้"}

    b = store.submit(unexpected)
    wait_until(lambda: store.get(b.id).state == "error")
    error = store.snapshot(b)["error"]
    assert error["status"] == 500
    assert "รายละเอียดภายใน" not in error["detail"]

    # worker ต้องยังทำงานต่อได้หลังมีงานล้ม
    c = store.submit(lambda _p: {"still": "alive"})
    wait_until(lambda: store.get(c.id).state == "done")
    store.shutdown()


def test_store_runs_one_job_at_a_time_and_reports_queue_position() -> None:
    store = JobStore(workers=1)
    gate = threading.Event()
    order: list[str] = []

    def first(_p):
        gate.wait(5)
        order.append("first")
        return {}

    def second(_p):
        order.append("second")
        return {}

    a = store.submit(first)
    b = store.submit(second)
    wait_until(lambda: store.get(a.id).state == "running")
    snap = store.snapshot(b)
    assert snap["state"] == "queued" and snap["queue_position"] == 1

    gate.set()
    wait_until(lambda: store.get(b.id).state == "done")
    assert order == ["first", "second"]
    store.shutdown()


def test_store_rejects_when_too_many_active() -> None:
    store = JobStore(max_active=2)
    gate = threading.Event()
    store.submit(lambda _p: gate.wait(5) or {})
    store.submit(lambda _p: gate.wait(5) or {})
    with pytest.raises(JobBusy):
        store.submit(lambda _p: {})
    gate.set()
    store.shutdown()


def test_store_purges_expired_and_caps_stored_jobs() -> None:
    store = JobStore(ttl=0.05, max_stored=3)
    job = store.submit(lambda _p: {})
    wait_until(lambda: store.get(job.id).state == "done")
    time.sleep(0.1)
    assert store.get(job.id) is None  # หมดอายุแล้ว

    store = JobStore(ttl=600, max_stored=3)
    ids = []
    for _ in range(6):
        j = store.submit(lambda _p: {})
        wait_until(lambda j=j: store.get(j.id).state == "done")
        ids.append(j.id)
    assert store.get(ids[0]) is None  # เก่าสุดถูกตัดทิ้ง
    assert store.get(ids[-1]) is not None
    store.shutdown()


# --- เส้นทาง API -------------------------------------------------------------


def test_job_by_ref_returns_every_page(client: TestClient) -> None:
    refs = []
    for name in ("a.png", "b.png"):
        r = client.post(
            "/api/convert",
            files={"file": (name, png_bytes(fixtures.synthetic_lineart()), "image/png")},
        )
        assert r.status_code == 200
        refs.append(r.json()["ref"])

    start = client.post(
        "/api/jobs/pages-by-ref", json={"refs": refs, "captions": ["หนึ่ง", "สอง"]}
    )
    assert start.status_code == 202
    body = start.json()
    assert body["job_id"] and body["total"] == 3  # ปก + 2 หน้า

    info = finish(client, body["job_id"])
    assert info["state"] == "done"
    assert info["done"] == info["total"] == 3

    result = client.get(f"/api/jobs/{body['job_id']}/result")
    assert result.status_code == 200
    data = result.json()
    assert data["count"] == 3
    assert all(p.startswith("data:image/png;base64,") for p in data["pages"])


def test_job_by_ref_reports_missing_with_409(client: TestClient) -> None:
    ok = client.post(
        "/api/convert",
        files={"file": ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")},
    ).json()["ref"]
    response = client.post(
        "/api/jobs/pages-by-ref", json={"refs": [ok, "0" * 32], "captions": []}
    )
    assert response.status_code == 409
    assert response.json()["missing"] == [1]


def test_job_upload_flow_without_conversion(client: TestClient) -> None:
    start = client.post(
        "/api/jobs/pages",
        files=[
            ("files", ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")),
            ("files", ("b.png", png_bytes(fixtures.synthetic_lineart()), "image/png")),
        ],
        data={"lineart": json.dumps({"skip_convert": True})},
    )
    assert start.status_code == 202
    job_id = start.json()["job_id"]

    info = finish(client, job_id)
    assert info["state"] == "done"
    # ช่วงอ่านรูป 2 รูป + ช่วงสร้างหน้า 3 หน้า
    assert info["total"] == 5 and info["done"] == 5
    assert client.get(f"/api/jobs/{job_id}/result").json()["count"] == 3


def test_job_upload_with_nothing_usable_ends_in_error(client: TestClient) -> None:
    blank = np.full((600, 400), 255, np.uint8)
    start = client.post(
        "/api/jobs/pages", files=[("files", ("blank.png", png_bytes(blank), "image/png"))]
    )
    assert start.status_code == 202
    job_id = start.json()["job_id"]

    info = finish(client, job_id)
    assert info["state"] == "error"
    assert info["error"]["status"] == 422
    assert "ไม่มีภาพใดที่ใช้ได้" in info["error"]["detail"]

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 422


def test_job_upload_validates_files_before_queueing(client: TestClient) -> None:
    """ไฟล์ผิดชนิดหรือมากเกินต้องถูกปฏิเสธทันทีในคำขอแรก ไม่ใช่ไปล้มทีหลังในคิว"""
    bad = client.post(
        "/api/jobs/pages", files=[("files", ("note.txt", b"hello", "text/plain"))]
    )
    assert bad.status_code == 400

    tiny = [
        ("files", (f"{i}.png", png_bytes(np.full((8, 8), 255, np.uint8)), "image/png"))
        for i in range(main.MAX_FILES_PER_REQUEST + 1)
    ]
    assert client.post("/api/jobs/pages", files=tiny).status_code == 413


def test_job_by_ref_rejects_too_many_refs(client: TestClient) -> None:
    response = client.post(
        "/api/jobs/pages-by-ref", json={"refs": ["a" * 32] * (main.MAX_FILES_PER_REQUEST + 1)}
    )
    assert response.status_code == 422


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404
    assert client.get("/api/jobs/" + "0" * 32 + "/result").status_code == 404


def test_result_before_finish_is_409(client: TestClient, monkeypatch) -> None:
    gate = threading.Event()
    store = JobStore()
    monkeypatch.setattr(main, "_JOBS", store)
    job = store.submit(lambda _p: gate.wait(5) or {"count": 0, "pages": []})
    assert client.get(f"/api/jobs/{job.id}/result").status_code == 409
    gate.set()
    store.shutdown()


def test_start_returns_429_when_queue_is_full(client: TestClient, monkeypatch) -> None:
    gate = threading.Event()
    store = JobStore(max_active=1)
    monkeypatch.setattr(main, "_JOBS", store)
    store.submit(lambda _p: gate.wait(5) or {})

    ref = client.post(
        "/api/convert",
        files={"file": ("a.png", png_bytes(fixtures.synthetic_lineart()), "image/png")},
    ).json()["ref"]
    response = client.post("/api/jobs/pages-by-ref", json={"refs": [ref]})
    assert response.status_code == 429
    gate.set()
    store.shutdown()
