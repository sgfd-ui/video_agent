from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict


class GlobalState(TypedDict, total=False):
    target_scene: str
    run_mode: Literal["timer", "event"]
    end_time: datetime
    event_snapshot: list[str]
    planner_state: bool
    planner_tasks: list[dict[str, str]]
    planner_reflections: list[dict[str, str]]
    raw_urls: Annotated[list[str], operator.add]
    high_light_clips: list[str]
    manifest: dict
    token_usage: dict
    stop_flag: bool


class CrawlerSubState(TypedDict, total=False):
    platform: str
    current_keyword: str
    retry_count: int
    top_5_results: list[dict]
