"""EvaluatorAgent for title/cover semantic quality checks."""

from __future__ import annotations

import json
from dataclasses import dataclass

from AutoVideoMiner.app.core.config import get_llm_for_agent
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

    def evaluate(self, platform: str, keyword: str, top_5_results: list[dict], target_scene: str) -> tuple[float, str]:
        score, reason = self._fallback_score(top_5_results, target_scene)
        try:
            llm = get_llm_for_agent("evaluator_agent")
            prompt = (
                "你是视频质检助手。根据 target_scene 与 samples 标题相关性打分(0-1)。"
                "仅输出 JSON: {\"score\": float, \"reason\": str}.\n"
                f"target_scene: {target_scene}\n"
                f"samples: {[r.get('title','') for r in top_5_results[:5]]}"
            )
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", str(resp))
            data = json.loads(content)
            score = float(data.get("score", score))
            reason = str(data.get("reason", reason))
        except Exception:
            pass

        score = max(0.0, min(1.0, score))
        upsert_search_history(self.db_path, platform=platform, keyword=keyword, score=score, reason=reason)
        return score, reason
