from __future__ import annotations

from typing import Any

from AutoVideoMiner.app.core.config import get_agent_model_config, load_settings
from AutoVideoMiner.app.core.token_usage import estimate_tokens
from AutoVideoMiner.app.flow.memory_manager import MemoryManager


class AgentMemoryRuntime:
    def _init_memory(self, agent_name: str, short_memory: dict[str, Any] | None) -> dict[str, Any]:
        memory = short_memory.copy() if isinstance(short_memory, dict) else {}
        memory.setdefault("system_prompt", "")
        memory.setdefault("context_patch", "")
        memory.setdefault("stale_messages", [])
        memory.setdefault("tail_messages", [])
        memory.setdefault("agent_name", agent_name)
        return memory

    def _append_memory(self, memory: dict[str, Any], msg: str) -> None:
        memory["tail_messages"].append(msg)
        while len(memory["tail_messages"]) > 5:
            memory["stale_messages"].append(memory["tail_messages"].pop(0))

    def _compact_if_needed(self, logs_dir: str, memory: dict[str, Any]) -> None:
        settings = load_settings()
        threshold = float(settings.get("system", {}).get("max_token_threshold", 0.8))
        agent = memory["agent_name"]
        budget = int(get_agent_model_config(agent)["llm"].get("token_budget", 400000))
        content = "\n".join([memory.get("system_prompt", ""), memory.get("context_patch", ""), *memory.get("stale_messages", []), *memory.get("tail_messages", [])])
        ratio = estimate_tokens(content) / max(1, budget)
        if ratio < threshold or not memory.get("stale_messages"):
            return
        manager = MemoryManager(logs_dir=logs_dir, threshold=threshold)
        patch = manager.compact(agent, memory["stale_messages"])
        manager.append_md_delta(agent, patch)
        memory["context_patch"] = patch.context_patch
        memory["stale_messages"] = []

    def _consolidate_task(self, logs_dir: str, memory: dict[str, Any]) -> None:
        manager = MemoryManager(logs_dir=logs_dir)
        manager.consolidate_task(memory["agent_name"], memory.get("tail_messages", []))
