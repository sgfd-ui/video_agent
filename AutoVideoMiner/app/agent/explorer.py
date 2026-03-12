from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from AutoVideoMiner.app.agent.memory_runtime import AgentMemoryRuntime
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.prompt_loader import get_prompt
from AutoVideoMiner.app.tool.vision_ffmpeg import probe_media

LOGGER = get_logger("agent.explorer")


@dataclass
class ExplorerAgent(AgentMemoryRuntime):
    def summarize(self, clip_paths: list[str], short_memory: dict[str, Any] | None = None, logs_dir: str = "") -> tuple[dict, dict[str, Any]]:
        m = self._init_memory("explorer_agent", short_memory)
        m["system_prompt"] = get_prompt("explorer", "system_main")
        events = []
        for c in clip_paths:
            try:
                d = float(probe_media(c).get("format", {}).get("duration", 0.0))
            except Exception:
                d = 0.0
            events.append({"event": "视频片段待复核", "full_description": f"clip={c}, duration={d:.2f}s", "level": "需要关注的行为"})
        self._append_memory(m, f"EXP:{len(clip_paths)}->{len(events)}")
        if logs_dir:
            self._compact_if_needed(logs_dir, m)
        return {"events": events, "clip_count": len(events)}, m
