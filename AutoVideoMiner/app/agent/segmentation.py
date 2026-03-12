"""SegmentationAgent: download videos and produce candidate clips via ffmpeg."""

from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.tool.vision_ffmpeg import filter_valid_cuts, generate_candidate_cuts
from AutoVideoMiner.app.tool.yt_download import download_video

LOGGER = get_logger("agent.segmentation")


@dataclass
class SegmentationAgent:
    workspace: str

    def run(self, urls: list[str]) -> list[str]:
        LOGGER.info("Segmentation start | urls=%s", len(urls))
        clips: list[str] = []
        for url in urls:
            try:
                local_video = download_video(url, self.workspace)
                candidates = generate_candidate_cuts(local_video)
                clips.extend(filter_valid_cuts(local_video, candidates, self.workspace))
            except Exception as exc:
                LOGGER.warning("Segmentation failed for %s | reason=%s", url, exc)
                continue
        LOGGER.info("Segmentation done | clips=%s", len(clips))
        return clips
