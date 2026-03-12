"""EvaluatorAgent for title/cover semantic quality checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from AutoVideoMiner.app.core.config import get_llm_for_agent
from AutoVideoMiner.app.core.token_usage import add_token_usage, estimate_tokens
from AutoVideoMiner.app.prompt.agent_prompts import EVALUATOR_PROMPT
from AutoVideoMiner.app.tool.sqlite_db import upsert_search_history


@dataclass
class EvaluatorAgent:
    db_path: str

    def _fallback_score(self, top_5_results: list[dict], target_scene: str) -> tuple[float, str]:
        target_tokens = set(target_scene.lower().split())
        if not top_5_results:
            return 0.0, "probe结果为空"
        ratios = []
        for row in top_5_results:
            title_tokens = set((row.get("title") or "").lower().split())
            ratios.append(len(target_tokens & title_tokens) / max(1, len(target_tokens)))
        score = round(sum(ratios) / len(ratios), 2)
        return score, "fallback lexical score"

    def evaluate(
        self,
        platform: str,
        keyword: str,
        top_5_results: list[dict],
        target_scene: str,
        token_usage: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        score, reason = self._fallback_score(top_5_results, target_scene)
        prompt = (
            f"{EVALUATOR_PROMPT}\n"
            f"target_scene: {target_scene}\n"
            f"samples: {[r.get('title', '') for r in top_5_results[:5]]}"
        )
        try:
            llm = get_llm_for_agent("evaluator_agent")
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", str(resp))
            data = json.loads(content)
            score = float(data.get("score", score))
            reason = str(data.get("reason", reason))
            if token_usage is not None:
                used = estimate_tokens(prompt) + estimate_tokens(content)
                add_token_usage(token_usage, "evaluator_agent", used)
        except Exception:
            if token_usage is not None:
                add_token_usage(token_usage, "evaluator_agent", estimate_tokens(prompt))

        score = max(0.0, min(1.0, score))
        upsert_search_history(self.db_path, platform=platform, keyword=keyword, score=score, reason=reason)
        return score, reason
