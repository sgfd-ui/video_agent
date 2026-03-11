"""Workflow orchestration with map-reduce style crawler execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from AutoVideoMiner.app.agent.crawler import CrawlerAgent
from AutoVideoMiner.app.agent.evaluator import EvaluatorAgent
from AutoVideoMiner.app.agent.explorer import ExplorerAgent
from AutoVideoMiner.app.agent.planner import PlannerAgent
from AutoVideoMiner.app.agent.segmentation import SegmentationAgent
from AutoVideoMiner.app.flow.state import CrawlerSubState, GlobalState


def control_gate(state: GlobalState) -> str:
    if state.get("stop_flag"):
        return "END"
    if state.get("run_mode") == "timer" and state.get("end_time") and datetime.now() > state["end_time"]:
        return "END"
    if state.get("run_mode") == "event" and not state.get("event_snapshot"):
        return "END"
    return "PlannerNode"


def _run_single_task(
    sub_state: CrawlerSubState,
    crawler: CrawlerAgent,
    evaluator: EvaluatorAgent,
    target_scene: str,
) -> list[str]:
    platform = sub_state["platform"]
    keyword = sub_state["current_keyword"]
    retry_count = 0

    while retry_count < 3:
        probe_results = crawler.crawl(platform=platform, keyword=keyword, task_mode="probe")
        score, _reason = evaluator.evaluate(platform, keyword, probe_results[:5], target_scene)
        if score > 0.8:
            sweep_results = crawler.crawl(platform=platform, keyword=keyword, task_mode="sweep")
            return [x["url"] for x in sweep_results]
        retry_count += 1
    return []


def run_once(state: GlobalState, db_path: str, workspace: str) -> GlobalState:
    planner = PlannerAgent(db_path=db_path)
    crawler = CrawlerAgent(db_path=db_path)
    evaluator = EvaluatorAgent(db_path=db_path)
    segmentation_agent = SegmentationAgent(workspace=workspace)
    explorer = ExplorerAgent()

    event_name = None
    if state.get("run_mode") == "event" and state.get("event_snapshot"):
        event_name = state["event_snapshot"].pop(0)

    planner_tasks = planner.plan(target_scene=state["target_scene"], event_name=event_name)
    state["planner_tasks"] = planner_tasks

    sub_states = [
        CrawlerSubState(platform=t["platform"], current_keyword=t["keyword"], retry_count=0, top_5_results=[])
        for t in planner_tasks
    ]

    raw_urls: list[str] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sub_states)))) as pool:
        futures = [pool.submit(_run_single_task, sub, crawler, evaluator, state["target_scene"]) for sub in sub_states]
        for future in futures:
            raw_urls.extend(future.result())

    state["raw_urls"] = raw_urls
    state["high_light_clips"] = segmentation_agent.run(raw_urls)
    state["manifest"] = explorer.summarize(state["high_light_clips"])
    return state
