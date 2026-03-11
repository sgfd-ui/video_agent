"""ExplorerAgent for event summary output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExplorerAgent:
    def summarize(self, clip_paths: list[str]) -> dict:
        return {
            "events": [
                {
                    "event": "疑似异常行为",
                    "full_description": f"共分析 {len(clip_paths)} 个候选片段并生成事件摘要。",
                    "level": "需要关注的行为",
                }
            ],
            "clip_count": len(clip_paths),
        }
