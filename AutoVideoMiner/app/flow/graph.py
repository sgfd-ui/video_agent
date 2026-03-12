"""Workflow orchestration for AutoVideoMiner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from AutoVideoMiner.app.agent.crawler import CrawlerAgent
from AutoVideoMiner.app.agent.evaluator import EvaluatorAgent
from AutoVideoMiner.app.agent.explorer import ExplorerAgent
from AutoVideoMiner.app.agent.planner import PlannerAgent
from AutoVideoMiner.app.agent.segmentation import SegmentationAgent
from AutoVideoMiner.app.core.config import load_settings
from AutoVideoMiner.app.core.token_usage import add_token_usage, estimate_tokens, init_token_usage
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
    pass_threshold: float,
    token_usage: dict,
) -> list[str]:
    platform = sub_state["platform"]
    keyword = sub_state["current_keyword"]

    add_token_usage(token_usage, "crawler_agent", estimate_tokens(f"crawl:{platform}:{keyword}"))
    for _ in range(3):
        probe_results = crawler.crawl(platform=platform, keyword=keyword, task_mode="probe")
        score, _ = evaluator.evaluate(platform, keyword, probe_results[:5], target_scene, token_usage=token_usage)
        if score > pass_threshold:
            sweep_results = crawler.crawl(platform=platform, keyword=keyword, task_mode="sweep")
            add_token_usage(token_usage, "crawler_agent", estimate_tokens(f"sweep:{len(sweep_results)}"))
            return [x["url"] for x in sweep_results]
    return []


def run_once(state: GlobalState, db_path: str, workspace: str) -> GlobalState:
    settings = load_settings()
    probe_size = int(settings.get("system", {}).get("probe_size", 5))
    sweep_limit = int(settings.get("system", {}).get("sweep_limit", 50))
    pass_threshold = float(settings.get("system", {}).get("evaluator_pass_threshold", 0.8))

    if not state.get("token_usage"):
        state["token_usage"] = init_token_usage(settings)

    planner = PlannerAgent(db_path=db_path)
    crawler = CrawlerAgent(db_path=db_path, probe_size=probe_size, sweep_limit=sweep_limit)
    evaluator = EvaluatorAgent(db_path=db_path)
    segmentation_agent = SegmentationAgent(workspace=workspace)
    explorer = ExplorerAgent()

    event_name = None
    if state.get("run_mode") == "event" and state.get("event_snapshot"):
        event_name = state["event_snapshot"].pop(0)

    planner_tasks = planner.plan(target_scene=state["target_scene"], event_name=event_name)
    state["planner_tasks"] = planner_tasks
    add_token_usage(state["token_usage"], "planner_agent", estimate_tokens(str(planner_tasks)))

    sub_states = [CrawlerSubState(platform=t["platform"], current_keyword=t["keyword"], retry_count=0, top_5_results=[]) for t in planner_tasks]

    raw_urls: list[str] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sub_states)))) as pool:
        futures = [
            pool.submit(_run_single_task, sub, crawler, evaluator, state["target_scene"], pass_threshold, state["token_usage"])
            for sub in sub_states
        ]
        for future in futures:
            raw_urls.extend(future.result())

    seen: set[str] = set()
    deduped = []
    for url in raw_urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)

    state["raw_urls"] = deduped
    state["high_light_clips"] = segmentation_agent.run(deduped)
    add_token_usage(state["token_usage"], "segmentation_agent", estimate_tokens(str(state["high_light_clips"])))

    state["manifest"] = explorer.summarize(state["high_light_clips"])
    add_token_usage(state["token_usage"], "explorer_agent", estimate_tokens(str(state["manifest"])))
    return state
