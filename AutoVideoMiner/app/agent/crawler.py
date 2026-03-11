"""CrawlerAgent with probe/sweep and semantic stop heuristics."""

from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.tool.sqlite_db import add_visited_urls, is_url_visited, update_search_numer


@dataclass
class CrawlerAgent:
    db_path: str

    def crawl(self, platform: str, keyword: str, task_mode: str = "probe") -> list[dict[str, str]]:
        max_items = 5 if task_mode == "probe" else 30
        outputs: list[dict[str, str]] = []
        low_similarity_count = 0

        for index in range(max_items):
            similarity = 0.85 if index % 6 else 0.55
            if task_mode == "sweep" and similarity < 0.6:
                low_similarity_count += 1
                if low_similarity_count >= 4:
                    break
            else:
                low_similarity_count = 0

            url = f"https://{platform}.example/video/{keyword.replace(' ', '-')}/{index}"
            if is_url_visited(self.db_path, url):
                continue

            item = {
                "title": f"{keyword} #{index}",
                "cover": "https://example.com/cover.jpg",
                "duration": "00:30",
                "url": url,
            }
            outputs.append(item)

        add_visited_urls(self.db_path, [x["url"] for x in outputs])
        if task_mode == "sweep":
            update_search_numer(self.db_path, keyword, len(outputs))
        return outputs
