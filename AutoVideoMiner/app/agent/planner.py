"""PlannerAgent: generate-retrieve-reflect-revise loop (max 5 iterations)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from AutoVideoMiner.app.core.config import get_llm_for_agent
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.prompt.agent_prompts import PLANNER_PROMPT
from AutoVideoMiner.app.tool.sqlite_db import (
    fetch_search_records_exact,
    fetch_search_records_similar,
)

LOGGER = get_logger("agent.planner")


@dataclass
class PlanResult:
    state: bool
    list: list[dict[str, str]]
    reflections: list[dict[str, str]]


@dataclass
class PlannerAgent:
    db_path: str
    logs_dir: str
    platforms: tuple[str, ...] = ("bilibili", "youtube", "douyin")

    def _load_mid_memory(self) -> str:
        file_path = Path(self.logs_dir) / "planner_agent.md"
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8")[-2000:]

    def _generate_initial(self, scene: str, event_name: str | None) -> list[dict[str, str]]:
        seed = f"{scene} {event_name}".strip() if event_name else scene.strip()
        base_keywords = [seed, f"{seed} 监控", f"{seed} CCTV", f"{seed} incident"]
        tasks = [{"platform": p, "keyword": k} for p in self.platforms for k in base_keywords]

        try:
            llm = get_llm_for_agent("planner_agent")
            prompt = (
                f"{PLANNER_PROMPT}\n"
                "输出 JSON 数组，每项含 platform, keyword。"
                f"\n场景: {seed}\n中期记忆: {self._load_mid_memory()}\n"
            )
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", str(resp))
            parsed = json.loads(content)
            if isinstance(parsed, list):
                valid = [x for x in parsed if isinstance(x, dict) and x.get("platform") and x.get("keyword")]
                if valid:
                    tasks = [{"platform": str(x["platform"]), "keyword": str(x["keyword"])} for x in valid]
        except Exception as exc:
            LOGGER.warning("Planner initial generation fallback | reason=%s", exc)

        return tasks

    def _retrieve_evidence(self, tasks: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
        evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
        exact = fetch_search_records_exact(self.db_path, tasks)
        exact_map = {(r["platform"], r["keyword"]): [r] for r in exact}

        for task in tasks:
            key = (task["platform"], task["keyword"])
            if key in exact_map:
                evidence[key] = exact_map[key]
            else:
                evidence[key] = fetch_search_records_similar(self.db_path, task["platform"], task["keyword"], threshold=0.8)
        return evidence

    def _reflect(self, tasks: list[dict[str, str]], evidence: dict[tuple[str, str], list[dict[str, Any]]]) -> dict:
        reflections = []
        status = "OPTIMAL"

        for task in tasks:
            records = evidence.get((task["platform"], task["keyword"]), [])
            advice = ""
            if records:
                best = sorted(records, key=lambda x: float(x.get("score") or 0), reverse=True)[0]
                if float(best.get("score") or 0) < 0.8:
                    advice = "历史低分，建议增加场景限定词并避免旧关键词"
                    status = "REVISE"
                elif best.get("similarity", 1.0) > 0.9:
                    advice = "历史高相似，建议替换同义词避免重复"
                    status = "REVISE"
            reflections.append({"platform": task["platform"], "keyword": task["keyword"], "advice": advice})

        return {"status": status, "reflections": reflections}

    def _revise(self, tasks: list[dict[str, str]], reflections: list[dict[str, str]], short_memory: list[dict]) -> list[dict[str, str]]:
        revised: list[dict[str, str]] = []
        for task, rf in zip(tasks, reflections):
            advice = rf.get("advice", "")
            if not advice:
                revised.append(task)
                continue
            candidate = {"platform": task["platform"], "keyword": f"{task['keyword']} 最新现场"}
            if candidate not in revised:
                revised.append(candidate)

        short_memory.extend(reflections)
        return revised

    def plan(self, target_scene: str, event_name: str | None = None) -> PlanResult:
        LOGGER.info("Planner start | scene=%s event=%s", target_scene, event_name)
        tasks = self._generate_initial(target_scene, event_name)
        short_memory: list[dict] = []
        reflections: list[dict[str, str]] = []

        for loop in range(1, 6):
            LOGGER.info("Planner loop=%s tasks=%s", loop, len(tasks))
            evidence = self._retrieve_evidence(tasks)
            verdict = self._reflect(tasks, evidence)
            reflections = verdict["reflections"]

            if verdict["status"] == "OPTIMAL":
                LOGGER.info("Planner optimal reached at loop=%s", loop)
                return PlanResult(state=True, list=tasks, reflections=reflections)

            tasks = self._revise(tasks, reflections, short_memory)

        LOGGER.warning("Planner exhausted 5 loops -> circuit break")
        return PlanResult(state=False, list=[], reflections=reflections)
