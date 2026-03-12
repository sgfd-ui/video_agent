from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.agent.memory_runtime import AgentMemoryRuntime
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.prompt_loader import get_prompt
from AutoVideoMiner.app.tool.vision_ffmpeg import filter_valid_cuts, generate_candidate_cuts
from AutoVideoMiner.app.tool.yt_download import download_video

LOGGER = get_logger("agent.segmentation")


@dataclass
class SegmentationAgent(AgentMemoryRuntime):
    workspace: str

    def __post_init__(self) -> None:
        self._setup_memory("segmentation_agent")

    def run(self, urls: list[str], logs_dir: str = "") -> list[str]:
        self.memory["system_prompt"] = get_prompt("segmentation", "SYSTEM_PROMPT")
        clips = []
        for u in urls:
            try:
                v = download_video(u, self.workspace)
                clips.extend(filter_valid_cuts(v, generate_candidate_cuts(v), self.workspace))
            except Exception as exc:
                LOGGER.warning("seg fail %s", exc)
        self._append_memory(f"SEG:{len(urls)}->{len(clips)}")
        if logs_dir:
            self._compact_if_needed(logs_dir)
        return clips
