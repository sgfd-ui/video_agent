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
from AutoVideoMiner.app.tool.ask_human import ask_human


def control_gate(state: GlobalState) -> str:
    if state.get("stop_flag"):
        return "END"
    if state.get("run_mode") == "timer" and state.get("end_time") and datetime.now() > state["end_time"]:
        return "END"
    if state.get("run_mode") == "event" and not state.get("event_snapshot"):
        return "END"
    return "PlannerNode"


def _run_single_task(sub_state: CrawlerSubState, crawler: CrawlerAgent, evaluator: EvaluatorAgent, target_scene: str, pass_threshold: float, token_usage: dict, logs_dir: str) -> list[str]:
    platform = sub_state["platform"]
    keyword = sub_state["current_keyword"]
    add_token_usage(token_usage, "crawler_agent", estimate_tokens(f"crawl:{platform}:{keyword}"))
    for _ in range(3):
        probe_results = crawler.crawl(platform, keyword, "probe", logs_dir)
        score, _reason = evaluator.evaluate(platform, keyword, probe_results[:5], target_scene, token_usage, logs_dir)
        if score > pass_threshold:
            sweep_results = crawler.crawl(platform, keyword, "sweep", logs_dir)
            add_token_usage(token_usage, "crawler_agent", estimate_tokens(f"sweep:{len(sweep_results)}"))
            return [x["url"] for x in sweep_results]
    return []


def run_once(state: GlobalState, db_path: str, workspace: str, logs_dir: str) -> GlobalState:
    settings = load_settings()
    probe_size = int(settings.get("system", {}).get("probe_size", 5))
    sweep_limit = int(settings.get("system", {}).get("sweep_limit", 50))
    pass_threshold = float(settings.get("system", {}).get("evaluator_pass_threshold", 0.8))

    if not state.get("token_usage"):
        state["token_usage"] = init_token_usage(settings)

    planner = PlannerAgent(db_path=db_path, logs_dir=logs_dir)
    crawler = CrawlerAgent(db_path=db_path, probe_size=probe_size, sweep_limit=sweep_limit)
    evaluator = EvaluatorAgent(db_path=db_path)
    segmentation_agent = SegmentationAgent(workspace=workspace)
    explorer = ExplorerAgent()

    event_name = state["event_snapshot"].pop(0) if state.get("run_mode") == "event" and state.get("event_snapshot") else None

    try:
        planner_result = planner.plan(state["target_scene"], event_name)
    except RuntimeError as exc:
        state["planner_tasks"] = []
        state["planner_state"] = False
        state["planner_reflections"] = []
        state["hitl"] = ask_human(str(exc))
        state["raw_urls"] = []
        state["high_light_clips"] = []
        state["manifest"] = {"events": []}
        state["stop_flag"] = True
        planner._consolidate_task(logs_dir)
        return state

    state["planner_tasks"] = planner_result.list
    state["planner_state"] = planner_result.state
    state["planner_reflections"] = planner_result.reflections
    add_token_usage(state["token_usage"], "planner_agent", estimate_tokens(str(planner_result.list)))

    if not planner_result.state or not planner_result.list:
        state["raw_urls"] = []
        state["high_light_clips"] = []
        state["manifest"] = {"events": []}
        state["stop_flag"] = True
        planner._consolidate_task(logs_dir)
        return state

    sub_states = [CrawlerSubState(platform=t["platform"], current_keyword=t["keyword"], retry_count=0, top_5_results=[]) for t in planner_result.list]

    raw_urls: list[str] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sub_states)))) as pool:
        futures = [pool.submit(_run_single_task, sub, crawler, evaluator, state["target_scene"], pass_threshold, state["token_usage"], logs_dir) for sub in sub_states]
        for f in futures:
            raw_urls.extend(f.result())

    deduped = list(dict.fromkeys(raw_urls))
    state["raw_urls"] = deduped
    clips = segmentation_agent.run(deduped, logs_dir)
    state["high_light_clips"] = clips
    add_token_usage(state["token_usage"], "segmentation_agent", estimate_tokens(str(clips)))

    manifest = explorer.summarize(clips, logs_dir)
    state["manifest"] = manifest
    add_token_usage(state["token_usage"], "explorer_agent", estimate_tokens(str(manifest)))

    for agent_obj in [planner, crawler, evaluator, segmentation_agent, explorer]:
        agent_obj._consolidate_task(logs_dir)

    return state
