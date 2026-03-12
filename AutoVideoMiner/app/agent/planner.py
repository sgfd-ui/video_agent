"""PlannerAgent: internal generate-retrieve-reflect-revise loop (max 5 iterations)."""

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

    def _invoke_llm_json(self, prompt: str) -> Any:
        llm = get_llm_for_agent("planner_agent")
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", str(resp))
        return json.loads(content), content

    def _generate_initial(self, scene: str, event_name: str | None, short_memory: dict[str, Any]) -> list[dict[str, str]]:
        seed = f"{scene} {event_name}".strip() if event_name else scene.strip()
        base_keywords = [seed, f"{seed} 监控", f"{seed} CCTV", f"{seed} incident"]
        tasks = [{"platform": p, "keyword": k} for p in self.platforms for k in base_keywords]

        prompt = (
            f"{PLANNER_PROMPT}\n"
            "输出 JSON 数组，每项含 platform, keyword。"
            f"\n场景: {seed}\n中期记忆: {self._load_mid_memory()}\n"
            f"短期记忆(摘要+最近5轮): {short_memory.get('context_patch','')} | {short_memory.get('tail_messages', [])}\n"
            f"平台候选: {self.platforms}"
        )

        try:
            self._append_short_memory(short_memory, f"INIT_PROMPT:{prompt[:1000]}")
            parsed, content = self._invoke_llm_json(prompt)
            self._append_short_memory(short_memory, f"INIT_RESP:{str(content)[:1000]}")
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

    def _reflect(self, tasks: list[dict[str, str]], evidence: dict[tuple[str, str], list[dict[str, Any]]], short_memory: dict[str, Any]) -> dict:
        """Reflection is internal to Planner and uses Planner's own LLM."""
        default_reflections = [{"platform": t["platform"], "keyword": t["keyword"], "advice": ""} for t in tasks]
        verdict = {"status": "OPTIMAL", "reflections": default_reflections}

        prompt = (
            f"{PLANNER_PROMPT}\n"
            "你现在是规划裁决者。请基于任务与检索证据判断是否需要修改。"
            "输出 JSON: {\"status\":\"OPTIMAL|REVISE\",\"reflections\":[{\"platform\":\"...\",\"keyword\":\"...\",\"advice\":\"...\"}]}\n"
            f"tasks: {tasks}\n"
            f"evidence: {evidence}\n"
            f"short_memory_tail: {short_memory.get('tail_messages',[])}"
        )

        try:
            parsed, content = self._invoke_llm_json(prompt)
            self._append_short_memory(short_memory, f"REFLECT_RESP:{str(content)[:1000]}")
            if isinstance(parsed, dict) and parsed.get("status") in {"OPTIMAL", "REVISE"}:
                refs = parsed.get("reflections", [])
                if isinstance(refs, list):
                    fixed = [
                        {
                            "platform": str(r.get("platform", "")),
                            "keyword": str(r.get("keyword", "")),
                            "advice": str(r.get("advice", "")),
                        }
                        for r in refs
                        if isinstance(r, dict)
                    ]
                    if fixed:
                        verdict = {"status": parsed["status"], "reflections": fixed}
        except Exception as exc:
            LOGGER.warning("Planner reflect llm fallback | reason=%s", exc)
            # heuristic fallback
            refs = []
            status = "OPTIMAL"
            for t in tasks:
                records = evidence.get((t["platform"], t["keyword"]), [])
                advice = ""
                if records:
                    best = sorted(records, key=lambda x: float(x.get("score") or 0), reverse=True)[0]
                    if float(best.get("score") or 0) < 0.8 or best.get("similarity", 1.0) > 0.9:
                        advice = "历史表现风险，建议改写关键词"
                        status = "REVISE"
                refs.append({"platform": t["platform"], "keyword": t["keyword"], "advice": advice})
            verdict = {"status": status, "reflections": refs}

        self._maybe_compact_short_memory(short_memory)
        return verdict

    def _revise(self, tasks: list[dict[str, str]], reflections: list[dict[str, str]], short_memory: dict[str, Any]) -> list[dict[str, str]]:
        revised: list[dict[str, str]] = []
        prompt = (
            f"{PLANNER_PROMPT}\n"
            "根据 reflections 对任务进行修正并仅输出 JSON 数组[{platform,keyword}]。\n"
            f"tasks: {tasks}\nreflections: {reflections}\n"
            f"short_memory_tail: {short_memory.get('tail_messages',[])}"
        )
        try:
            parsed, content = self._invoke_llm_json(prompt)
            self._append_short_memory(short_memory, f"REVISE_RESP:{str(content)[:1000]}")
            if isinstance(parsed, list):
                valid = [x for x in parsed if isinstance(x, dict) and x.get("platform") and x.get("keyword")]
                if valid:
                    revised = [{"platform": str(x["platform"]), "keyword": str(x["keyword"])} for x in valid]
        except Exception as exc:
            LOGGER.warning("Planner revise llm fallback | reason=%s", exc)

        if not revised:
            # fallback rewrite
            for task, rf in zip(tasks, reflections):
                advice = rf.get("advice", "")
                if not advice:
                    revised.append(task)
                else:
                    revised.append({"platform": task["platform"], "keyword": f"{task['keyword']} 最新现场"})

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
            verdict = self._reflect(tasks, evidence, memory)
            reflections = verdict["reflections"]

            if verdict["status"] == "OPTIMAL":
                LOGGER.info("Planner optimal reached at loop=%s", loop)
                return PlanResult(state=True, list=tasks, reflections=reflections, short_memory=memory)

            tasks = self._revise(tasks, reflections, memory)

        LOGGER.warning("Planner exhausted 5 loops -> circuit break")
        return PlanResult(state=False, list=[], reflections=reflections, short_memory=memory)
