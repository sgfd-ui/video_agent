from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from AutoVideoMiner.app.agent.memory_runtime import AgentMemoryRuntime
from AutoVideoMiner.app.core.config import get_llm_for_agent
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.prompt_loader import get_prompt
from AutoVideoMiner.app.core.token_usage import add_token_usage, estimate_tokens
from AutoVideoMiner.app.tool.sqlite_db import upsert_search_history

LOGGER = get_logger("agent.evaluator")


@dataclass
class EvaluatorAgent(AgentMemoryRuntime):
    db_path: str

    def __post_init__(self) -> None:
        self._setup_memory("evaluator_agent")

    def evaluate(self, platform: str, keyword: str, top_5_results: list[dict], target_scene: str, token_usage: dict[str, Any] | None = None, logs_dir: str = "") -> tuple[float, str]:
        self.memory["system_prompt"] = get_prompt("evaluator", "SYSTEM_PROMPT")
        prompt = f"{get_prompt('evaluator','SYSTEM_PROMPT')}\n{get_prompt('evaluator','SCORING_PROMPT')}\nscene:{target_scene}\nsamples:{[x.get('title','') for x in top_5_results[:5]]}"
        score, reason = 0.0, "probe结果为空"
        try:
            llm = get_llm_for_agent("evaluator_agent")
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", str(resp))
            data = json.loads(content)
            score = float(data.get("score", 0.0))
            reason = str(data.get("reason", ""))
            if token_usage is not None:
                add_token_usage(token_usage, "evaluator_agent", estimate_tokens(prompt) + estimate_tokens(content))
        except Exception as exc:
            LOGGER.warning("evaluator fallback: %s", exc)
            score = 0.0 if not top_5_results else 0.6
            reason = "fallback"
        upsert_search_history(self.db_path, platform, keyword, score, reason)
        self._append_memory(f"EVAL:{platform}:{keyword}:{score}")
        if logs_dir:
            self._compact_if_needed(logs_dir)
        return max(0.0, min(1.0, score)), reason
