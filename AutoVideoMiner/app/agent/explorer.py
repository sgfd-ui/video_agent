"""ExplorerAgent: summarize extracted clips and build event manifest."""

from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.tool.vision_ffmpeg import probe_media

LOGGER = get_logger("agent.explorer")


@dataclass
class ExplorerAgent:
    def summarize(self, clip_paths: list[str]) -> dict:
        LOGGER.info("Explorer start | clips=%s", len(clip_paths))
        events = []
        for clip in clip_paths:
            try:
                meta = probe_media(clip).get("format", {})
                duration = float(meta.get("duration", 0.0))
            except Exception as exc:
                LOGGER.warning("Explorer probe fail | clip=%s reason=%s", clip, exc)
                duration = 0.0
            events.append(
                {
                    "event": "视频片段待复核",
                    "full_description": f"clip={clip}, duration={duration:.2f}s",
                    "level": "需要关注的行为",
                }
            )
        LOGGER.info("Explorer done | events=%s", len(events))
        return {"events": events, "clip_count": len(events)}
