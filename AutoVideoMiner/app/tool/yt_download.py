"""Atomic download tool wrapper."""

from __future__ import annotations

from pathlib import Path


def download_video(url: str, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    local_path = Path(output_dir) / "downloaded_video.mp4"
    local_path.touch(exist_ok=True)
    return str(local_path)
