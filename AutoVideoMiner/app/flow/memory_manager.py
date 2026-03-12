"""Three-layer memory manager for short/mid-term memory compaction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from AutoVideoMiner.app.prompt.memory_prompts import MEMORY_COMPACTION_PROMPT


@dataclass
class MemoryPatch:
    md_delta: str
    context_patch: str


class MemoryManager:
    def __init__(self, logs_dir: str, threshold: float = 0.8) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

    def should_compact(self, token_usage_ratio: float) -> bool:
        return token_usage_ratio >= self.threshold

    def compaction_prompt(self) -> str:
        return MEMORY_COMPACTION_PROMPT

    def compact(self, agent_name: str, stale_messages: list[str]) -> MemoryPatch:
        brief = "\n".join(stale_messages[-15:])[:500]
        delta = f"- {agent_name}: 压缩 {len(stale_messages)} 条历史记录"
        return MemoryPatch(md_delta=delta, context_patch=brief)

    def append_md_delta(self, agent_name: str, patch: MemoryPatch) -> None:
        log_path = self.logs_dir / f"{agent_name}.md"
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"\n{patch.md_delta}\n")
