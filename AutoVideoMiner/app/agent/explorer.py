from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.agent.memory_runtime import AgentMemoryRuntime
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.prompt_loader import get_prompt
from AutoVideoMiner.app.tool.vision_ffmpeg import probe_media

LOGGER = get_logger("agent.explorer")


@dataclass
class ExplorerAgent(AgentMemoryRuntime):
    def __post_init__(self) -> None:
        self._setup_memory("explorer_agent")

    def summarize(self, clip_paths: list[str], logs_dir: str = "") -> dict:
        LOGGER.info("explorer start: clips=%s", len(clip_paths))
        self.memory["system_prompt"] = get_prompt("explorer_agent", "SYSTEM_PROMPT")
        events = []
        for c in clip_paths:
            try:
                d = float(probe_media(c).get("format", {}).get("duration", 0.0))
            except Exception:
                d = 0.0
            events.append({"event": "视频片段待复核", "full_description": f"clip={c}, duration={d:.2f}s", "level": "需要关注的行为"})
        self._append_memory(f"EXP:{len(clip_paths)}->{len(events)}")
        if logs_dir:
            self._compact_if_needed(logs_dir)
        LOGGER.info("explorer done: events=%s", len(events))
        return {"events": events, "clip_count": len(events)}
