"""ทดสอบความปลอดภัยและความเสถียร: SSRF, เพดานจำนวนไฟล์, event loop ต้องไม่ค้าง"""

from __future__ import annotations

import http.server
import socketserver
import threading
import time

import cv2
import numpy as np
import pytest
import requests
import uvicorn
from fastapi.testclient import TestClient

from app import main, webgrab
from app.main import app


def _png(size: int = 300) -> bytes:
    rng = np.random.default_rng(0)
    ok, buf = cv2.imencode(".png", rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture()
def local_image_server():
    """เซิร์ฟเวอร์จำลองบนเครื่อง ที่ตอบรูปภาพจริง ใช้เป็น 'เป้าหมายภายใน'"""
    data = _png()
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], hits
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize(
    "address,expected",
    [
        ("8.8.8.8", True),
        ("1.1.1.1", True),
        ("127.0.0.1", False),
        ("10.0.0.5", False),
        ("192.168.1.1", False),
        ("172.16.0.1", False),
        ("169.254.169.254", False),  # metadata ของคลาวด์
        ("100.64.0.1", False),  # CGNAT
        ("0.0.0.0", False),
        ("::1", False),
        ("::ffff:127.0.0.1", False),  # IPv6 ห่อ IPv4
        ("fe80::1", False),
        ("fc00::1", False),
        ("224.0.0.1", False),  # multicast
        ("not-an-ip", False),
    ],
)
def test_ip_is_public(address: str, expected: bool) -> None:
    assert webgrab._ip_is_public(address) is expected


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "2130706433", "0x7f000001", "127.1", "::1", "0.0.0.0"],
)
def test_private_hosts_are_rejected_in_every_notation(host: str) -> None:
    with pytest.raises(webgrab.WebImageError):
        webgrab._assert_public_host(host)


@pytest.mark.parametrize(
    "host", ["localhost", "2130706433", "0x7f000001", "[::1]"]
)
def test_download_blocks_internal_hosts_with_port(local_image_server, host: str) -> None:
    """เดิมช่องโหว่คือ localhost:พอร์ต และเลขฐานสิบผ่านได้ ต้องไม่ถึงเซิร์ฟเวอร์ภายในเลย"""
    port, hits = local_image_server
    with pytest.raises(webgrab.WebImageError):
        webgrab.download_image(f"http://{host}:{port}/a.png")
    assert hits == []


def test_connection_level_guard_blocks_even_if_name_check_is_bypassed(
    local_image_server, monkeypatch
) -> None:
    """จำลอง DNS rebinding / redirect: ชื่อผ่านการตรวจ แต่ IP จริงเป็นภายใน ต้องถูกตัดตอนต่อ socket"""
    port, hits = local_image_server
    monkeypatch.setattr(webgrab, "_assert_public_host", lambda _host: None)
    with pytest.raises(webgrab.WebImageError):
        webgrab.download_image(f"http://127.0.0.1:{port}/a.png")
    assert hits == []  # ไม่มีคำขอถึงเซิร์ฟเวอร์ภายในเลย


def test_api_web_image_rejects_internal_url(local_image_server) -> None:
    port, hits = local_image_server
    client = TestClient(app)
    response = client.post("/api/web-image", data={"url": f"http://localhost:{port}/a.png"})
    assert response.status_code == 400
    assert hits == []


def test_api_web_images_rejects_internal_url(local_image_server) -> None:
    port, hits = local_image_server
    client = TestClient(app)
    response = client.post("/api/web-images", data={"url": f"http://2130706433:{port}/"})
    assert response.status_code == 400
    assert hits == []


# --- เพดานจำนวนไฟล์ ---------------------------------------------------------


def _tiny_files(count: int):
    png = _png(40)
    return [("files", (f"a{i}.png", png, "image/png")) for i in range(count)]


def test_pages_rejects_too_many_files() -> None:
    client = TestClient(app)
    response = client.post("/api/pages", files=_tiny_files(main.MAX_FILES_PER_REQUEST + 1))
    assert response.status_code == 413


def test_book_rejects_too_many_files() -> None:
    client = TestClient(app)
    response = client.post("/api/book", files=_tiny_files(main.MAX_FILES_PER_REQUEST + 1))
    assert response.status_code == 413


def test_pages_by_ref_rejects_too_many_refs() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/pages-by-ref", json={"refs": ["a" * 32] * (main.MAX_FILES_PER_REQUEST + 1)}
    )
    assert response.status_code == 422


# --- event loop ต้องไม่ค้างระหว่างงานหนัก ----------------------------------


@pytest.fixture()
def live_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_health_stays_responsive_during_slow_download(live_server, monkeypatch) -> None:
    """งานดาวน์โหลดช้า ๆ ห้ามทำให้ /api/health ตอบไม่ได้ (ไม่งั้น Render ตัดเป็น 502)"""

    def slow_download(_url: str):
        time.sleep(2.0)
        raise webgrab.WebImageError("ช้ามาก")

    monkeypatch.setattr(main, "download_image", slow_download)

    worker = threading.Thread(
        target=lambda: requests.post(
            live_server + "/api/web-image", data={"url": "https://example.com/a.png"}, timeout=10
        ),
        daemon=True,
    )
    worker.start()
    time.sleep(0.3)  # ให้คำขอช้าเริ่มทำงานก่อน

    started = time.time()
    response = requests.get(live_server + "/api/health", timeout=1.0)
    assert response.status_code == 200
    assert time.time() - started < 1.0
    worker.join(timeout=10)


# --- ต้องไม่ทำให้การดึงรูปจากเว็บสาธารณะพัง -------------------------------


def test_public_host_still_downloads(local_image_server, monkeypatch) -> None:
    """ถ้าที่อยู่ปลายทางเป็นสาธารณะ ต้องดาวน์โหลดได้ตามปกติ (จำลองด้วยการบอกว่า 127.0.0.1 สาธารณะ)"""
    port, hits = local_image_server
    monkeypatch.setattr(webgrab, "_assert_public_host", lambda _host: None)
    monkeypatch.setattr(webgrab, "_ip_is_public", lambda _address: True)
    raw, _name = webgrab.download_image(f"http://127.0.0.1:{port}/photo.png")
    assert len(raw) > webgrab.MIN_IMAGE_BYTES
    assert hits == ["/photo.png"]


def test_https_connection_class_is_constructible() -> None:
    """คลาส connection แบบ HTTPS ที่เราครอบไว้ต้องสร้างได้ (ครอบคลุมเส้นทาง https จริง)"""
    pool = webgrab._SafeHTTPSPool("example.com", port=443)
    connection = pool._new_conn()
    assert isinstance(connection, webgrab._SafeHTTPSConnection)
