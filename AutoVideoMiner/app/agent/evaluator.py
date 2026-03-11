"""EvaluatorAgent for multimodal quality checks and persistence."""

from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.tool.sqlite_db import upsert_search_history


@dataclass
class EvaluatorAgent:
    db_path: str

    def evaluate(self, platform: str, keyword: str, top_5_results: list[dict], target_scene: str) -> tuple[float, str]:
        if not top_5_results:
            score, reason = 0.0, "probe结果为空"
        else:
            matched = sum(1 for row in top_5_results if target_scene[:2] in row.get("title", ""))
            score = round(min(1.0, 0.6 + 0.08 * matched), 2)
            reason = "标题与目标场景语义匹配" if score > 0.8 else "相关性偏弱，建议调整关键词"

        upsert_search_history(self.db_path, platform=platform, keyword=keyword, score=score, reason=reason)
        return score, reason
