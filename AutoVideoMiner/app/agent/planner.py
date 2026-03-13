from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from AutoVideoMiner.app.agent.memory_runtime import AgentMemoryRuntime
from AutoVideoMiner.app.core.config import get_llm_for_agent
from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.core.prompt_loader import get_prompt
from AutoVideoMiner.app.flow.memory_manager import MemoryManager
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

    def _extract_json_payload(self, content: str) -> str:
        text = (content or "").strip()
        if not text:
            raise ValueError("empty llm content")
        block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if block_match and block_match.group(1).strip():
            return block_match.group(1).strip()
        arr_match = re.search(r"\[[\s\S]*\]", text)
        if arr_match and arr_match.group(0).strip():
            return arr_match.group(0).strip()
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match and obj_match.group(0).strip():
            return obj_match.group(0).strip()
        return text

    def _llm_text(self, prompt: str) -> str:
        llm = get_llm_for_agent("planner_agent", force_reload=True)
        resp = llm.invoke(prompt)
        return str(getattr(resp, "content", str(resp))).strip()

    def _llm_json(self, prompt: str) -> Any:
        content = self._llm_text(prompt)
        cleaned = self._extract_json_payload(content).strip()
        if not cleaned:
            LOGGER.error("Planner _llm_json empty cleaned content")
            raise ValueError("Planner _llm_json empty cleaned content")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            LOGGER.error("Planner _llm_json parse failed. cleaned_content=%s", cleaned)
            raise

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
        LOGGER.info("planner evidence collected: tasks=%s", len(tasks))
        return evidence

    def _load_prompts(self) -> dict[str, str]:
        return {
            "system": get_prompt("planner_agent", "SYSTEM_PROMPT"),
            "memory_route": get_prompt("planner_agent", "MEMORY_ROUTING_PROMPT"),
            "issue": get_prompt("planner_agent", "TASK_ISSUANCE_PROMPT"),
            "reflect": get_prompt("planner_agent", "REFLECTION_PROMPT"),
            "revise": get_prompt("planner_agent", "REVISION_PROMPT"),
            "compress": get_prompt("planner_agent", "MEMORY_COMPRESSION_SYSTEM_PROMPT"),
            "consolidate": get_prompt("planner_agent", "MEMORY_CONSOLIDATION_SYSTEM_PROMPT"),
        }

    def _maybe_compact(
        self,
        manager: MemoryManager,
        prompts: dict[str, str],
        *,
        scene: str,
        task_id: str,
        scene_md_path: str,
    ) -> None:
        if not self.memory.get("stale_messages"):
            return
        existing = manager.read_task_fragments(scene_md_path, task_id)
        raw = "\n".join(self.memory["stale_messages"]).strip()
        patch = manager.planner_compact(
            self._llm_text,
            scene=scene,
            existing_memory=existing,
            raw_procedure=raw,
            prompt_system=prompts["compress"],
        )
        manager.append_task_patch(scene_md_path, task_id, patch.md_delta)
        self.memory["context_patch"] = patch.context_patch
        self.memory["stale_messages"] = []
        LOGGER.info("planner compacted memory for task_id=%s", task_id)

    def plan(self, target_scene: str, event_name: str | None = None) -> PlanResult:
        scene = f"{target_scene} {event_name}".strip() if event_name else target_scene.strip()
        manager = MemoryManager(logs_dir=self.logs_dir)
        task_id = manager.allocate_task_id()
        global_rules_path, scene_md_path = manager.init_planner_memory_files(scene)
        LOGGER.info("planner start: scene=%s task_id=%s", scene, task_id)

        prompts = self._load_prompts()
        self.memory["system_prompt"] = f"{prompts['system']}\n{prompts['issue']}\nscene:{scene}"
        self._append_memory(f"TASK_ID:{task_id}")

        candidate_files = manager.list_scene_memory_files()
        global_rules = manager.read_file(global_rules_path)
        route_prompt = (
            f"{prompts['system']}\n{prompts['memory_route']}\n"
            f"<SCENE>{scene}</SCENE>\n"
            f"<CANDIDATE_FILES>{candidate_files}</CANDIDATE_FILES>\n"
            f"<GLOBAL_RULES>{global_rules}</GLOBAL_RULES>"
        )
        try:
            route_data = self._llm_json(route_prompt)
            selected_paths = route_data.get("selected_paths", []) if isinstance(route_data, dict) else []
        except Exception:
            selected_paths = []
        selected_contents = [manager.read_file(p) for p in selected_paths if isinstance(p, str)]
        memory_context = "\n\n".join(selected_contents)

        reflections: list[dict[str, str]] = []
        tasks: list[dict[str, str]] = []

        for loop_idx in range(5):
            if loop_idx >= 5:
                break
            issue_prompt = (
                f"{prompts['system']}\n{prompts['issue']}\n"
                f"<SCENE>{scene}</SCENE>\n"
                f"<MEMORY_CONTEXT>{memory_context}</MEMORY_CONTEXT>\n"
                f"<BUFFER_DATA>{self.memory.get('stale_messages', [])}</BUFFER_DATA>\n"
                f"<SHORT_TERM_REFLECTIONS>{self.memory.get('tail_messages', [])}</SHORT_TERM_REFLECTIONS>"
            )
            try:
                issued = self._llm_json(issue_prompt)
            except Exception as exc:
                LOGGER.exception("Planner issuance failed")
                raise RuntimeError("PlannerAgent: 任务发布阶段失败，任务中断。") from exc

            task_list = issued.get("list", issued) if isinstance(issued, dict) else issued
            if not isinstance(task_list, list):
                raise RuntimeError("PlannerAgent: 任务发布返回格式错误，任务中断。")
            tasks = [{"platform": str(i["platform"]), "keyword": str(i["keyword"])} for i in task_list if isinstance(i, dict) and i.get("platform") and i.get("keyword")]
            if not tasks:
                raise RuntimeError("PlannerAgent: 任务发布为空，任务中断。")

            evidence = self._retrieve(tasks)
            reflect_prompt = (
                f"{prompts['system']}\n{prompts['reflect']}\n"
                f"<SCENE>{scene}</SCENE>\n"
                f"<CANDIDATE_LIST>{tasks}</CANDIDATE_LIST>\n"
                f"<SQLITE_EVIDENCE>{evidence}</SQLITE_EVIDENCE>"
            )
            try:
                verdict = self._llm_json(reflect_prompt)
            except Exception as exc:
                LOGGER.exception("Planner reflection failed")
                raise RuntimeError("PlannerAgent: 反思阶段失败，任务中断。") from exc

            if not isinstance(verdict, dict):
                raise RuntimeError("PlannerAgent: 反思返回格式错误，任务中断。")
            reflections = verdict.get("reflections", [])
            self._append_memory(json.dumps({"loop": loop_idx + 1, "reflections": reflections}, ensure_ascii=False))

            self._compact_if_needed(self.logs_dir)
            self._maybe_compact(manager, prompts, scene=scene, task_id=task_id, scene_md_path=scene_md_path)

            if verdict.get("status") == "OPTIMAL":
                final_tail = "\n".join(self.memory.get("tail_messages", []) + self.memory.get("stale_messages", []))
                fragments = manager.read_task_fragments(scene_md_path, task_id)
                final_summary = manager.planner_consolidate(
                    self._llm_text,
                    scene=scene,
                    end_status="成功下发",
                    history_fragments=fragments,
                    latest_ram=final_tail,
                    prompt_system=prompts["consolidate"],
                )
                manager.replace_task_summary(scene_md_path, task_id, final_summary)
                LOGGER.info("planner optimal: task_id=%s tasks=%s", task_id, len(tasks))
                return PlanResult(True, tasks, reflections)

            revise_prompt = (
                f"{prompts['system']}\n{prompts['revise']}\n"
                f"当前列表:{tasks}\n"
                f"反思:{reflections}"
            )
            revised = self._llm_json(revise_prompt)
            revised_list = revised.get("list", revised) if isinstance(revised, dict) else revised
            if not isinstance(revised_list, list):
                raise RuntimeError("PlannerAgent: 修订返回格式错误，任务中断。")
            tasks = [{"platform": str(i["platform"]), "keyword": str(i["keyword"])} for i in revised_list if isinstance(i, dict) and i.get("platform") and i.get("keyword")]
            if not tasks:
                raise RuntimeError("PlannerAgent: 修订后任务为空，任务中断。")

        final_tail = "\n".join(self.memory.get("tail_messages", []) + self.memory.get("stale_messages", []))
        fragments = manager.read_task_fragments(scene_md_path, task_id)
        final_summary = manager.planner_consolidate(
            self._llm_text,
            scene=scene,
            end_status="5次失败强制熔断",
            history_fragments=fragments,
            latest_ram=final_tail,
            prompt_system=prompts["consolidate"],
        )
        manager.replace_task_summary(scene_md_path, task_id, final_summary)
        LOGGER.warning("planner fused after max loops: task_id=%s", task_id)
        return PlanResult(False, [], reflections)
