"""ทดสอบระบบสร้างวิดีโอ Affiliate Slideshow"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from app.book import video
from tests import fixtures


def test_create_affiliate_video(tmp_path: Path) -> None:
    # สร้างภาพจำลอง 2 หน้า
    img1 = fixtures.synthetic_lineart(400, 400)
    img2 = fixtures.synthetic_lineart(400, 400)

    output_file = tmp_path / "test_promo.mp4"
    result_path = video.create_affiliate_video(
        page_images=[img1, img2],
        output_path=output_file,
        hook_text="ทดสอบสมุดระบายสี",
        fps=10,
        seconds_per_page=0.5,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 0
