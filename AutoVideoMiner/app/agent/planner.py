from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class PlannerAgent(AgentMemoryRuntime):
    db_path: str
    logs_dir: str
    platforms: tuple[str, ...] = ("bilibili", "youtube", "douyin")

    def __post_init__(self) -> None:
        self._setup_memory("planner_agent")

    def _llm_json(self, text: str):
        llm = get_llm_for_agent("planner_agent", force_reload=True)
        resp = llm.invoke(text)
        return json.loads(getattr(resp, "content", str(resp)))

    def _retrieve(self, tasks: list[dict[str, str]]) -> dict:
        ev = {}
        exact = fetch_search_records_exact(self.db_path, tasks)
        em = {(r["platform"], r["keyword"]): [r] for r in exact}
        for t in tasks:
            k = (t["platform"], t["keyword"])
            ev[k] = em.get(k) or fetch_search_records_similar(self.db_path, t["platform"], t["keyword"], 0.8)
        return ev

    def plan(self, target_scene: str, event_name: str | None = None) -> PlanResult:
        seed = f"{target_scene} {event_name}".strip() if event_name else target_scene.strip()
        mid_memory_path = Path(self.logs_dir) / "planner_agent.md"
        mid_memory_text = mid_memory_path.read_text(encoding="utf-8") if mid_memory_path.exists() else ""

        tasks = [{"platform": p, "keyword": k} for p in self.platforms for k in [seed, f"{seed} 监控", f"{seed} CCTV", f"{seed} incident"]]
        system_main = get_prompt("planner", "SYSTEM_PROMPT")
        task_logic = get_prompt("planner", "STRATEGY_GEN_PROMPT")
        self.memory["system_prompt"] = system_main
        self._append_memory(f"GUI:{seed}")

        try:
            init_prompt = f"{system_main}\n{task_logic}\nmid_memory:{mid_memory_text}\nshort:{self.memory.get('context_patch','')} {self.memory.get('tail_messages',[])}\n返回任务JSON数组"
            parsed = self._llm_json(init_prompt)
            if isinstance(parsed, list):
                vals = [x for x in parsed if isinstance(x, dict) and x.get("platform") and x.get("keyword")]
                if vals:
                    tasks = [{"platform": str(x["platform"]), "keyword": str(x["keyword"])} for x in vals]
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
            self._append_memory(f"REFLECT:{str(reflections)[:200]}")
            self._compact_if_needed(self.logs_dir)
            if isinstance(verdict, dict) and verdict.get("status") == "OPTIMAL":
                return PlanResult(True, tasks, reflections)

            try:
                vp = f"{system_main}\n{task_logic}\n根据REVISION_PROMPT修正tasks，返回任务JSON数组\ntasks:{tasks}\nreflections:{reflections}"
                revised = self._llm_json(vp)
                if isinstance(revised, list):
                    vals = [x for x in revised if isinstance(x, dict) and x.get("platform") and x.get("keyword")]
                    if vals:
                        tasks = [{"platform": str(x["platform"]), "keyword": str(x["keyword"])} for x in vals]
            except Exception:
                tasks = [{"platform": t["platform"], "keyword": f"{t['keyword']} 最新现场"} for t in tasks]

        return PlanResult(False, [], reflections)
