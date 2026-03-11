"""PlannerAgent: strategy generation and dedup-aware task planning."""

from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.tool.sqlite_db import fetch_search_history_keywords


@dataclass
class PlannerAgent:
    db_path: str
    platforms: tuple[str, ...] = ("bilibili", "youtube", "douyin")

    def plan(self, target_scene: str, event_name: str | None = None) -> list[dict[str, str]]:
        base = f"{target_scene} {event_name}".strip() if event_name else target_scene.strip()
        candidate_keywords = [
            base,
            f"{base} 监控实拍",
            f"{base} 高危行为",
        ]
        history = fetch_search_history_keywords(self.db_path)

        tasks: list[dict[str, str]] = []
        for platform in self.platforms:
            for keyword in candidate_keywords:
                if keyword not in history:
                    tasks.append({"platform": platform, "keyword": keyword})
        return tasks
