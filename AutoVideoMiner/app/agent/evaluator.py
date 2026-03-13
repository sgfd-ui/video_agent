from __future__ import annotations

import json
import re
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

    def _extract_json_payload(self, content: str) -> str:
        text = (content or "").strip()
        if not text:
            raise ValueError("empty llm content")

        block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if block_match:
            cleaned = block_match.group(1).strip()
            if cleaned:
                return cleaned

        arr_match = re.search(r"\[[\s\S]*\]", text)
        if arr_match and arr_match.group(0).strip():
            return arr_match.group(0).strip()

        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match and obj_match.group(0).strip():
            return obj_match.group(0).strip()

        return text

    def _llm_json(self, prompt: str) -> dict[str, Any]:
        llm = get_llm_for_agent("evaluator_agent")
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", str(resp))
        cleaned = self._extract_json_payload(content)
        if not cleaned:
            LOGGER.error("Evaluator _llm_json empty cleaned content")
            raise ValueError("Evaluator _llm_json empty cleaned content")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            LOGGER.error("Evaluator _llm_json parse failed. cleaned_content=%s", cleaned)
            raise

    def evaluate(self, platform: str, keyword: str, top_5_results: list[dict], target_scene: str, token_usage: dict[str, Any] | None = None, logs_dir: str = "") -> tuple[float, str]:
        LOGGER.info("evaluator start: platform=%s keyword=%s samples=%s", platform, keyword, len(top_5_results))
        self.memory["system_prompt"] = get_prompt("evaluator_agent", "SYSTEM_PROMPT")
        prompt = f"{get_prompt('evaluator_agent','SYSTEM_PROMPT')}\n{get_prompt('evaluator_agent','SCORING_PROMPT')}\nscene:{target_scene}\nsamples:{[x.get('title','') for x in top_5_results[:5]]}"
        score, reason = 0.0, "probe结果为空"
        try:
            data = self._llm_json(prompt)
            score = float(data.get("score", 0.0))
            reason = str(data.get("reason", ""))
            if token_usage is not None:
                add_token_usage(token_usage, "evaluator_agent", estimate_tokens(prompt) + estimate_tokens(json.dumps(data, ensure_ascii=False)))
        except Exception as exc:
            LOGGER.warning("evaluator fallback: %s", exc)
            score = 0.0 if not top_5_results else 0.6
            reason = "fallback"
        upsert_search_history(self.db_path, platform, keyword, score, reason)
        self._append_memory(f"EVAL:{platform}:{keyword}:{score}")
        if logs_dir:
            self._compact_if_needed(logs_dir)
        LOGGER.info("evaluator done: platform=%s keyword=%s score=%s", platform, keyword, score)
        return max(0.0, min(1.0, score)), reason
