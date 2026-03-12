from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from AutoVideoMiner.app.agent.crawler import CrawlerAgent
from AutoVideoMiner.app.agent.evaluator import EvaluatorAgent
from AutoVideoMiner.app.agent.explorer import ExplorerAgent
from AutoVideoMiner.app.agent.planner import PlannerAgent
from AutoVideoMiner.app.agent.segmentation import SegmentationAgent
from AutoVideoMiner.app.core.config import load_settings
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.token_usage import add_token_usage, estimate_tokens, init_token_usage
from AutoVideoMiner.app.flow.state import CrawlerSubState, GlobalState
from AutoVideoMiner.app.tool.memory_store import read_log

LOGGER = get_logger("flow.graph")


def control_gate(state: GlobalState) -> str:
    if state.get("stop_flag"):
        return "END"
    if state.get("run_mode") == "timer" and state.get("end_time") and datetime.now() > state["end_time"]:
        return "END"
    if state.get("run_mode") == "event" and not state.get("event_snapshot"):
        return "END"
    return "PlannerNode"


def _run_single_task(sub_state: CrawlerSubState, crawler: CrawlerAgent, evaluator: EvaluatorAgent, target_scene: str, pass_threshold: float, token_usage: dict, memories: dict, logs_dir: str) -> tuple[list[str], dict]:
    platform = sub_state["platform"]
    keyword = sub_state["current_keyword"]
    add_token_usage(token_usage, "crawler_agent", estimate_tokens(f"crawl:{platform}:{keyword}"))
    for _ in range(3):
        probe_results, memories["crawler_agent"] = crawler.crawl(platform, keyword, "probe", memories.get("crawler_agent"), logs_dir)
        score, _reason, memories["evaluator_agent"] = evaluator.evaluate(platform, keyword, probe_results[:5], target_scene, token_usage, memories.get("evaluator_agent"), logs_dir)
        if score > pass_threshold:
            sweep_results, memories["crawler_agent"] = crawler.crawl(platform, keyword, "sweep", memories.get("crawler_agent"), logs_dir)
            add_token_usage(token_usage, "crawler_agent", estimate_tokens(f"sweep:{len(sweep_results)}"))
            return [x["url"] for x in sweep_results], memories
    return [], memories


def run_once(state: GlobalState, db_path: str, workspace: str, logs_dir: str) -> GlobalState:
    settings = load_settings()
    probe_size = int(settings.get("system", {}).get("probe_size", 5))
    sweep_limit = int(settings.get("system", {}).get("sweep_limit", 50))
    pass_threshold = float(settings.get("system", {}).get("evaluator_pass_threshold", 0.8))

    if not state.get("token_usage"):
        state["token_usage"] = init_token_usage(settings)
    memories = state.get("agent_short_memories", {})

    planner = PlannerAgent(db_path=db_path, logs_dir=logs_dir)
    crawler = CrawlerAgent(db_path=db_path, probe_size=probe_size, sweep_limit=sweep_limit)
    evaluator = EvaluatorAgent(db_path=db_path)
    segmentation_agent = SegmentationAgent(workspace=workspace)
    explorer = ExplorerAgent()

    event_name = state["event_snapshot"].pop(0) if state.get("run_mode") == "event" and state.get("event_snapshot") else None
    mid_memory = read_log(str(Path(logs_dir) / "planner_agent.md"))

    planner_result = planner.plan(state["target_scene"], event_name, memories.get("planner_agent"), mid_memory)
    memories["planner_agent"] = planner_result.short_memory
    state["planner_tasks"] = planner_result.list
    state["planner_state"] = planner_result.state
    state["planner_reflections"] = planner_result.reflections
    add_token_usage(state["token_usage"], "planner_agent", estimate_tokens(str(planner_result.list)))

    if not planner_result.state or not planner_result.list:
        state["raw_urls"] = []
        state["high_light_clips"] = []
        state["manifest"] = {"events": []}
        state["stop_flag"] = True
        state["agent_short_memories"] = memories
        return state

    sub_states = [CrawlerSubState(platform=t["platform"], current_keyword=t["keyword"], retry_count=0, top_5_results=[]) for t in planner_result.list]

    raw_urls: list[str] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sub_states)))) as pool:
        futures = [pool.submit(_run_single_task, sub, crawler, evaluator, state["target_scene"], pass_threshold, state["token_usage"], memories, logs_dir) for sub in sub_states]
        for f in futures:
            urls, memories = f.result()
            raw_urls.extend(urls)

    deduped = list(dict.fromkeys(raw_urls))
    state["raw_urls"] = deduped
    clips, memories["segmentation_agent"] = segmentation_agent.run(deduped, memories.get("segmentation_agent"), logs_dir)
    state["high_light_clips"] = clips
    add_token_usage(state["token_usage"], "segmentation_agent", estimate_tokens(str(clips)))

    manifest, memories["explorer_agent"] = explorer.summarize(clips, memories.get("explorer_agent"), logs_dir)
    state["manifest"] = manifest
    add_token_usage(state["token_usage"], "explorer_agent", estimate_tokens(str(manifest)))

    # post-task consolidation for all agents
    for agent_obj, name in [
        (planner, "planner_agent"),
        (crawler, "crawler_agent"),
        (evaluator, "evaluator_agent"),
        (segmentation_agent, "segmentation_agent"),
        (explorer, "explorer_agent"),
    ]:
        if name in memories:
            agent_obj._consolidate_task(logs_dir, memories[name])

    state["agent_short_memories"] = memories
    return state
