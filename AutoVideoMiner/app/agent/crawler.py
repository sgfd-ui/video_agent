from __future__ import annotations

from dataclasses import dataclass

from AutoVideoMiner.app.agent.memory_runtime import AgentMemoryRuntime
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.prompt_loader import get_prompt
from AutoVideoMiner.app.tool.sqlite_db import add_visited_urls, is_url_visited, update_search_numer
from AutoVideoMiner.app.tool.yt_download import search_videos

LOGGER = get_logger("agent.crawler")


@dataclass
class CrawlerAgent(AgentMemoryRuntime):
    db_path: str
    probe_size: int = 5
    sweep_limit: int = 50

    def __post_init__(self) -> None:
        self._setup_memory("crawler_agent")

    def crawl(self, platform: str, keyword: str, task_mode: str = "probe", logs_dir: str = "") -> list[dict]:
        self.memory["system_prompt"] = get_prompt("crawler", "system_main")
        self._append_memory(f"CRAWL:{platform}:{keyword}:{task_mode}")
        rows = search_videos(keyword, self.probe_size if task_mode == "probe" else self.sweep_limit)
        out = []
        low = 0
        toks = set(keyword.lower().split())
        for r in rows:
            u = r.get("url", "")
            if not u or is_url_visited(self.db_path, u):
                continue
            overlap = len(toks & set((r.get("title") or "").lower().split())) / max(1, len(toks))
            if task_mode == "sweep" and overlap < 0.2:
                low += 1
                if low >= 4:
                    break
            else:
                low = 0
            r["platform"] = platform
            out.append(r)
        add_visited_urls(self.db_path, [x["url"] for x in out])
        if task_mode == "sweep":
            update_search_numer(self.db_path, platform, keyword, len(out))
        if logs_dir:
            self._compact_if_needed(logs_dir)
        LOGGER.info("crawler done %s %s %s", platform, keyword, len(out))
        return out
