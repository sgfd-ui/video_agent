"""PlannerAgent: generate platform/keyword tasks avoiding search-history duplicates."""

from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.tool.sqlite_db import fetch_search_history_keywords


@dataclass
class PlannerAgent:
    db_path: str
    platforms: tuple[str, ...] = ("youtube", "bilibili", "douyin")

    def plan(self, target_scene: str, event_name: str | None = None) -> list[dict[str, str]]:
        base = f"{target_scene} {event_name}".strip() if event_name else target_scene.strip()
        candidates = [base, f"{base} 监控", f"{base} CCTV", f"{base} incident"]
        history = fetch_search_history_keywords(self.db_path)

        tasks: list[dict[str, str]] = []
        for platform in self.platforms:
            for kw in candidates:
                if kw not in history:
                    tasks.append({"platform": platform, "keyword": kw})
        return tasks
