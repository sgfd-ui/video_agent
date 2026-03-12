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

    def __post_init__(self) -> None:
        self._setup_memory("planner_agent")

    def _llm_json(self, text: str):
        llm = get_llm_for_agent("planner_agent", force_reload=True)
        resp = llm.invoke(text)
        return json.loads(getattr(resp, "content", str(resp)))

    def _retrieve(self, tasks: list[dict[str, str]]) -> dict:
        evidence: dict[tuple[str, str], list[dict]] = {}
        exact = fetch_search_records_exact(self.db_path, tasks)
        exact_map = {(r["platform"], r["keyword"]): [r] for r in exact}
        for task in tasks:
            task_key = (task["platform"], task["keyword"])
            evidence[task_key] = exact_map.get(task_key) or fetch_search_records_similar(
                self.db_path,
                task["platform"],
                task["keyword"],
                0.8,
            )
        return evidence

    def plan(self, target_scene: str, event_name: str | None = None) -> PlanResult:
        seed = f"{target_scene} {event_name}".strip() if event_name else target_scene.strip()
        mid_memory_path = Path(self.logs_dir) / "planner_agent.md"
        mid_memory_text = mid_memory_path.read_text(encoding="utf-8") if mid_memory_path.exists() else ""

        system_main = get_prompt("planner", "SYSTEM_PROMPT")
        task_logic = get_prompt("planner", "STRATEGY_GEN_PROMPT")
        self.memory["system_prompt"] = system_main
        self._append_memory(f"GUI:{seed}")

        init_prompt = (
            f"{system_main}\n{task_logic}\n"
            f"mid_memory:{mid_memory_text}\n"
            f"short:{self.memory.get('context_patch', '')} {self.memory.get('tail_messages', [])}\n"
            "返回任务JSON数组"
        )

        try:
            parsed = self._llm_json(init_prompt)
        except Exception as exc:
            LOGGER.exception("Planner LLM initial generation failed")
            raise RuntimeError("PlannerAgent: 无法连接大模型，任务已中断，请检查模型与密钥配置。") from exc

        if not isinstance(parsed, list):
            raise RuntimeError("PlannerAgent: 初稿生成返回格式错误，任务已中断。")

        tasks = [item for item in parsed if isinstance(item, dict) and item.get("platform") and item.get("keyword")]
        if not tasks:
            raise RuntimeError("PlannerAgent: 初稿任务为空，任务已中断。")
        tasks = [{"platform": str(item["platform"]), "keyword": str(item["keyword"])} for item in tasks]

        reflections: list[dict[str, str]] = []
        for _ in range(5):
            evidence = self._retrieve(tasks)
            reflect_prompt = (
                f"{system_main}\n{task_logic}\n"
                "对tasks+evidence反思，输出{status,reflections} JSON\n"
                f"tasks:{tasks}\n"
                f"evidence:{evidence}"
            )
            try:
                verdict = self._llm_json(reflect_prompt)
            except Exception as exc:
                LOGGER.exception("Planner LLM reflection failed")
                raise RuntimeError("PlannerAgent: 反思阶段无法连接大模型，任务已中断。") from exc

            if not isinstance(verdict, dict):
                raise RuntimeError("PlannerAgent: 反思阶段返回格式错误，任务已中断。")

            reflections = verdict.get("reflections", [])
            self._append_memory(f"REFLECT:{str(reflections)[:200]}")
            self._compact_if_needed(self.logs_dir)

            if verdict.get("status") == "OPTIMAL":
                return PlanResult(True, tasks, reflections)

            revise_prompt = (
                f"{system_main}\n{task_logic}\n"
                "根据REVISION_PROMPT修正tasks，返回任务JSON数组\n"
                f"tasks:{tasks}\n"
                f"reflections:{reflections}"
            )
            try:
                revised = self._llm_json(revise_prompt)
            except Exception as exc:
                LOGGER.exception("Planner LLM revision failed")
                raise RuntimeError("PlannerAgent: 修改阶段无法连接大模型，任务已中断。") from exc

            if not isinstance(revised, list):
                raise RuntimeError("PlannerAgent: 修改阶段返回格式错误，任务已中断。")

            revised_tasks = [item for item in revised if isinstance(item, dict) and item.get("platform") and item.get("keyword")]
            if not revised_tasks:
                raise RuntimeError("PlannerAgent: 修改后任务为空，任务已中断。")
            tasks = [{"platform": str(item["platform"]), "keyword": str(item["keyword"])} for item in revised_tasks]

        return PlanResult(False, [], reflections)
