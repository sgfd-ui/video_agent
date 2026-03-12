"""CrawlerAgent: probe/sweep crawling using real yt-dlp search results."""

from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.tool.sqlite_db import add_visited_urls, is_url_visited, update_search_numer
from AutoVideoMiner.app.tool.yt_download import search_videos

LOGGER = get_logger("agent.crawler")


@dataclass
class CrawlerAgent:
    db_path: str
    probe_size: int = 5
    sweep_limit: int = 50

    def crawl(self, platform: str, keyword: str, task_mode: str = "probe") -> list[dict]:
        LOGGER.info("Crawler start | mode=%s platform=%s keyword=%s", task_mode, platform, keyword)
        limit = self.probe_size if task_mode == "probe" else self.sweep_limit
        rows = search_videos(keyword=keyword, limit=limit)

        outputs: list[dict] = []
        low_similarity_count = 0
        target_tokens = set(keyword.lower().split())

        for row in rows:
            url = row.get("url", "")
            if not url or is_url_visited(self.db_path, url):
                continue

            title_tokens = set((row.get("title") or "").lower().split())
            overlap = len(target_tokens & title_tokens) / max(1, len(target_tokens))
            if task_mode == "sweep" and overlap < 0.2:
                low_similarity_count += 1
                if low_similarity_count >= 4:
                    LOGGER.info("Crawler semantic break triggered | platform=%s keyword=%s", platform, keyword)
                    break
            else:
                low_similarity_count = 0

            row["platform"] = platform
            outputs.append(row)

        add_visited_urls(self.db_path, [x["url"] for x in outputs])
        if task_mode == "sweep":
            update_search_numer(self.db_path, platform, keyword, len(outputs))
        LOGGER.info("Crawler done | mode=%s platform=%s keyword=%s count=%s", task_mode, platform, keyword, len(outputs))
        return outputs
