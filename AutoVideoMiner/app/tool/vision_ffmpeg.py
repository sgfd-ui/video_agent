"""FFmpeg helpers for scene-cut discovery and clip extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe 未安装，无法执行切分")


def _video_duration(video_path: str) -> float:
    _require_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def generate_candidate_cuts(video_path: str, min_gap: float = 2.0) -> list[float]:
    """Detect rough scene change points via ffprobe packet timestamps fallback by fixed interval."""
    duration = _video_duration(video_path)
    points: list[float] = []
    t = min_gap
    while t < duration:
        points.append(round(t, 3))
        t += min_gap
    return points


def extract_clip(video_path: str, start: float, end: float, output_path: str) -> str:
    _require_ffmpeg()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0, start)),
        "-to",
        str(max(start + 0.1, end)),
        "-i",
        video_path,
        "-c",
        "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def filter_valid_cuts(video_path: str, candidates: list[float], output_dir: str, window: float = 1.5) -> list[str]:
    """Materialize candidate clips for downstream semantic filtering."""
    duration = _video_duration(video_path)
    out_paths: list[str] = []
    stem = Path(video_path).stem

    for idx, point in enumerate(candidates):
        start = max(0.0, point - window)
        end = min(duration, point + window)
        if end - start < 0.5:
            continue
        clip_path = str(Path(output_dir) / f"{stem}_cut_{idx:04d}.mp4")
        out_paths.append(extract_clip(video_path, start, end, clip_path))
    return out_paths


def probe_media(path: str) -> dict:
    _require_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-of",
        "json",
        path,
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)
