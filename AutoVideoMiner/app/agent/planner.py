"""PlannerAgent: generate-retrieve-reflect-revise loop (max 5 iterations)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from AutoVideoMiner.app.core.config import get_agent_model_config, get_llm_for_agent, load_settings
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.token_usage import estimate_tokens
from AutoVideoMiner.app.flow.memory_manager import MemoryManager
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
    short_memory: dict[str, Any]


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

    def _new_short_memory(self, short_memory: dict[str, Any] | None = None) -> dict[str, Any]:
        memory = short_memory.copy() if isinstance(short_memory, dict) else {}
        memory.setdefault("system_prompt", PLANNER_PROMPT)
        memory.setdefault("context_patch", "")
        memory.setdefault("stale_messages", [])
        memory.setdefault("tail_messages", [])
        return memory

    def _append_short_memory(self, memory: dict[str, Any], message: str) -> None:
        tail = memory["tail_messages"]
        stale = memory["stale_messages"]
        tail.append(message)
        while len(tail) > 5:
            stale.append(tail.pop(0))

    def _maybe_compact_short_memory(self, memory: dict[str, Any]) -> None:
        settings = load_settings()
        threshold = float(settings.get("system", {}).get("max_token_threshold", 0.8))
        token_budget = int(get_agent_model_config("planner_agent")["llm"].get("token_budget", 400000))

        total_text = "\n".join(
            [
                memory["system_prompt"],
                memory.get("context_patch", ""),
                *memory.get("stale_messages", []),
                *memory.get("tail_messages", []),
            ]
        )
        ratio = estimate_tokens(total_text) / max(1, token_budget)
        if ratio < threshold:
            return

        manager = MemoryManager(logs_dir=self.logs_dir, threshold=threshold)
        stale = memory.get("stale_messages", [])
        if not stale:
            return

        patch = manager.compact("planner_agent", stale)
        manager.append_md_delta("planner_agent", patch)
        memory["context_patch"] = patch.context_patch
        memory["stale_messages"] = []
        LOGGER.info("Planner short-memory compacted | ratio=%.4f threshold=%.2f", ratio, threshold)

    def _generate_initial(self, scene: str, event_name: str | None, short_memory: dict[str, Any]) -> list[dict[str, str]]:
        seed = f"{scene} {event_name}".strip() if event_name else scene.strip()
        base_keywords = [seed, f"{seed} 监控", f"{seed} CCTV", f"{seed} incident"]
        tasks = [{"platform": p, "keyword": k} for p in self.platforms for k in base_keywords]

        try:
            llm = get_llm_for_agent("planner_agent")
            prompt = (
                f"{PLANNER_PROMPT}\n"
                "输出 JSON 数组，每项含 platform, keyword。"
                f"\n场景: {seed}\n中期记忆: {self._load_mid_memory()}\n"
                f"短期记忆(摘要+最近5轮): {short_memory.get('context_patch','')} | {short_memory.get('tail_messages', [])}\n"
            )
            self._append_short_memory(short_memory, f"INIT_PROMPT:{prompt[:1000]}")
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", str(resp))
            self._append_short_memory(short_memory, f"INIT_RESP:{content[:1000]}")
            parsed = json.loads(content)
            if isinstance(parsed, list):
                valid = [x for x in parsed if isinstance(x, dict) and x.get("platform") and x.get("keyword")]
                if valid:
                    tasks = [{"platform": str(x["platform"]), "keyword": str(x["keyword"])} for x in valid]
        except Exception as exc:
            LOGGER.warning("Planner initial generation fallback | reason=%s", exc)

        self._maybe_compact_short_memory(short_memory)
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

    def _revise(self, tasks: list[dict[str, str]], reflections: list[dict[str, str]], short_memory: dict[str, Any]) -> list[dict[str, str]]:
        revised: list[dict[str, str]] = []
        for task, rf in zip(tasks, reflections):
            advice = rf.get("advice", "")
            if not advice:
                revised.append(task)
                continue
            candidate = {"platform": task["platform"], "keyword": f"{task['keyword']} 最新现场"}
            if candidate not in revised:
                revised.append(candidate)

        self._append_short_memory(short_memory, f"REFLECT:{json.dumps(reflections, ensure_ascii=False)[:1000]}")
        self._maybe_compact_short_memory(short_memory)
        return revised

    def plan(
        self,
        target_scene: str,
        event_name: str | None = None,
        short_memory: dict[str, Any] | None = None,
    ) -> PlanResult:
        LOGGER.info("Planner start | scene=%s event=%s", target_scene, event_name)
        memory = self._new_short_memory(short_memory)
        tasks = self._generate_initial(target_scene, event_name, memory)
        reflections: list[dict[str, str]] = []

        for loop in range(1, 6):
            LOGGER.info("Planner loop=%s tasks=%s tail=%s stale=%s", loop, len(tasks), len(memory['tail_messages']), len(memory['stale_messages']))
            evidence = self._retrieve_evidence(tasks)
            verdict = self._reflect(tasks, evidence)
            reflections = verdict["reflections"]

            if verdict["status"] == "OPTIMAL":
                LOGGER.info("Planner optimal reached at loop=%s", loop)
                return PlanResult(state=True, list=tasks, reflections=reflections, short_memory=memory)

            tasks = self._revise(tasks, reflections, memory)

        LOGGER.warning("Planner exhausted 5 loops -> circuit break")
        return PlanResult(state=False, list=[], reflections=reflections, short_memory=memory)
