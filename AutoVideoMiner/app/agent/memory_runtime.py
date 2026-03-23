from __future__ import annotations

from AutoVideoMiner.app.core.config import get_agent_model_config, load_settings
from AutoVideoMiner.app.core.token_usage import estimate_tokens
from AutoVideoMiner.app.flow.memory_manager import MemoryManager


class AgentMemoryRuntime:
    agent_name: str
    logs_dir: str
    memory: dict

    def _setup_memory(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.memory = {
            "system_prompt": "",
            "context_patch": "",
            "stale_messages": [],
            "tail_messages": [],
            "agent_name": agent_name,
        }

    def _append_memory(self, msg: str) -> None:
        self.memory["tail_messages"].append(msg)
        while len(self.memory["tail_messages"]) > 5:
            self.memory["stale_messages"].append(self.memory["tail_messages"].pop(0))

    def _compact_if_needed(self, logs_dir: str) -> None:
        settings = load_settings()
        threshold = float(settings.get("system", {}).get("max_token_threshold", 0.8))
        budget = int(get_agent_model_config(self.agent_name)["llm"].get("token_budget", 400000))
        content = "\n".join([
            self.memory.get("system_prompt", ""),
            self.memory.get("context_patch", ""),
            *self.memory.get("stale_messages", []),
            *self.memory.get("tail_messages", []),
        ])
        ratio = estimate_tokens(content) / max(1, budget)
        if ratio < threshold or not self.memory.get("stale_messages"):
            return
        manager = MemoryManager(logs_dir=logs_dir, threshold=threshold)
        patch = manager.compact(self.agent_name, self.memory["stale_messages"])
        manager.append_md_delta(self.agent_name, patch)
        self.memory["context_patch"] = patch.context_patch
        self.memory["stale_messages"] = []

    def _consolidate_task(self, logs_dir: str) -> None:
        manager = MemoryManager(logs_dir=logs_dir)
        manager.consolidate_task(self.agent_name, self.memory.get("tail_messages", []))
