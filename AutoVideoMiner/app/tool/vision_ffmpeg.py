"""Atomic FFmpeg/vision helper tools."""

from __future__ import annotations

from pathlib import Path


def generate_candidate_cuts(video_path: str) -> list[float]:
    return [1.0, 2.5, 4.0]


def filter_valid_cuts(video_path: str, candidates: list[float], output_dir: str) -> list[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output = []
    for index, _ in enumerate(candidates):
        clip = Path(output_dir) / f"{Path(video_path).stem}_clip_{index}.mp4"
        clip.touch(exist_ok=True)
        output.append(str(clip))
    return output
