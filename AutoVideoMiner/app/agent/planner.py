from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from AutoVideoMiner.app.agent.memory_runtime import AgentMemoryRuntime
from AutoVideoMiner.app.core.config import get_llm_for_agent
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.prompt_loader import get_prompt
from AutoVideoMiner.app.tool.sqlite_db import fetch_search_records_exact, fetch_search_records_similar

LOGGER = get_logger("agent.planner")


@dataclass
class PlanResult:
    state: bool
    list: list[dict[str, str]]
    reflections: list[dict[str, str]]
    short_memory: dict[str, Any]


@dataclass
class PlannerAgent(AgentMemoryRuntime):
    db_path: str
    logs_dir: str
    platforms: tuple[str, ...] = ("bilibili", "youtube", "douyin")

    def _llm_json(self, text: str):
        llm = get_llm_for_agent("planner_agent")
        resp = llm.invoke(text)
        content = getattr(resp, "content", str(resp))
        return json.loads(content)

    def _retrieve(self, tasks: list[dict[str, str]]) -> dict:
        ev = {}
        exact = fetch_search_records_exact(self.db_path, tasks)
        em = {(r["platform"], r["keyword"]): [r] for r in exact}
        for t in tasks:
            k = (t["platform"], t["keyword"])
            ev[k] = em.get(k) or fetch_search_records_similar(self.db_path, t["platform"], t["keyword"], 0.8)
        return ev

    def plan(self, target_scene: str, event_name: str | None = None, short_memory: dict[str, Any] | None = None, mid_memory_text: str = "") -> PlanResult:
        m = self._init_memory("planner_agent", short_memory)
        seed = f"{target_scene} {event_name}".strip() if event_name else target_scene.strip()
        tasks = [{"platform": p, "keyword": k} for p in self.platforms for k in [seed, f"{seed} 监控", f"{seed} CCTV", f"{seed} incident"]]

        system_main = get_prompt("planner", "system_main")
        task_logic = get_prompt("planner", "task_logic")
        self._append_memory(m, f"GUI:{seed}")

        try:
            init_prompt = f"{system_main}\n{task_logic}\nmid_memory:{mid_memory_text}\nshort:{m.get('context_patch','')} {m.get('tail_messages',[])}\n返回任务JSON数组"
            parsed = self._llm_json(init_prompt)
            if isinstance(parsed, list):
                vals = [x for x in parsed if isinstance(x, dict) and x.get("platform") and x.get("keyword")]
                if vals:
                    tasks = [{"platform": str(x["platform"]), "keyword": str(x["keyword"])} for x in vals]
            self._append_memory(m, f"INIT_OK:{len(tasks)}")
        except Exception as e:
            LOGGER.warning("planner init fallback: %s", e)

        reflections: list[dict[str, str]] = []
        for _ in range(5):
            ev = self._retrieve(tasks)
            try:
                rp = f"{system_main}\n{task_logic}\n对tasks+evidence反思，输出{{status,reflections}} JSON\ntasks:{tasks}\nevidence:{ev}"
                verdict = self._llm_json(rp)
            except Exception:
                verdict = {"status": "OPTIMAL", "reflections": [{"platform": t["platform"], "keyword": t["keyword"], "advice": ""} for t in tasks]}

            reflections = verdict.get("reflections", []) if isinstance(verdict, dict) else []
            self._append_memory(m, f"REFLECT:{str(reflections)[:200]}")
            self._compact_if_needed(self.logs_dir, m)

            if isinstance(verdict, dict) and verdict.get("status") == "OPTIMAL":
                return PlanResult(True, tasks, reflections, m)

            try:
                vp = f"{system_main}\n{task_logic}\n根据reflections修正tasks，返回任务JSON数组\ntasks:{tasks}\nreflections:{reflections}"
                revised = self._llm_json(vp)
                if isinstance(revised, list):
                    vals = [x for x in revised if isinstance(x, dict) and x.get("platform") and x.get("keyword")]
                    if vals:
                        tasks = [{"platform": str(x["platform"]), "keyword": str(x["keyword"])} for x in vals]
            except Exception:
                tasks = [{"platform": t["platform"], "keyword": f"{t['keyword']} 最新现场"} for t in tasks]

        return PlanResult(False, [], reflections, m)
