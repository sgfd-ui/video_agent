from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.prompt_loader import get_prompt
from AutoVideoMiner.app.tool.memory_store import append_log, overwrite_log, read_log

LOGGER = get_logger("flow.memory")


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

    def compact(self, agent_name: str, stale_messages: list[str]) -> MemoryPatch:
        agent = agent_name.replace("_agent", "")
        _ = get_prompt(agent, "memory_compression")
        joined = "\n".join(stale_messages)
        md_delta = f"- {agent_name}: {joined[:1000]}"
        context_patch = joined[-500:]
        try:
            parsed = json.loads('{"md_delta": "' + md_delta.replace('"', "'") + '", "context_patch": "' + context_patch.replace('"', "'") + '"}')
            return MemoryPatch(md_delta=parsed["md_delta"], context_patch=parsed["context_patch"])
        except Exception:
            return MemoryPatch(md_delta=md_delta, context_patch=context_patch)

    def append_md_delta(self, agent_name: str, patch: MemoryPatch) -> None:
        log_path = str(self.logs_dir / f"{agent_name}.md")
        append_log(log_path, f"\n{patch.md_delta}\n")

    def consolidate_task(self, agent_name: str, ram_tail: list[str]) -> str:
        agent = agent_name.replace("_agent", "")
        _ = get_prompt(agent, "task_consolidation")
        log_path = str(self.logs_dir / f"{agent_name}.md")
        old = read_log(log_path)
        final = (old + "\n" + "\n".join(ram_tail)).strip()
        overwrite_log(log_path, final + "\n")
        LOGGER.info("Task consolidated for %s", agent_name)
        return final
