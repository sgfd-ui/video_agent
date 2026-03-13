from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from AutoVideoMiner.app.core.logger import get_logger
from AutoVideoMiner.app.tool.memory_store import append_log, extract_section_by_task_id, overwrite_log, read_log, replace_section_by_task_id

LOGGER = get_logger("flow.memory")


@dataclass
class MemoryPatch:
    md_delta: str
    context_patch: str


class MemoryManager:
    def __init__(self, logs_dir: str, threshold: float = 0.8) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

    def should_compact(self, token_usage_ratio: float) -> bool:
        return token_usage_ratio >= self.threshold

    def init_planner_memory_files(self, scene: str) -> tuple[str, str]:
        planner_root = self.logs_dir.parent.parent / "memory" / "planner_agent"
        planner_root.mkdir(parents=True, exist_ok=True)
        global_rules_path = planner_root / "memory.md"
        scene_path = planner_root / f"{self._safe_scene_name(scene)}.md"
        if not global_rules_path.exists():
            global_rules_path.write_text("# Planner Global Rules\n- 记录跨场景选取规则与避雷策略。\n", encoding="utf-8")
        if not scene_path.exists():
            scene_path.write_text(f"# Scene Memory\nscene: {scene}\n", encoding="utf-8")
        return str(global_rules_path), str(scene_path)

    def allocate_task_id(self) -> str:
        return str(int(time.time()))

    def ensure_scene_memory_file(self, scene: str) -> tuple[str, str]:
        return self.init_planner_memory_files(scene)

    def insert_task_anchor(self, scene_md_path: str, task_id: str) -> None:
        text = read_log(scene_md_path)
        header = f"## timeStamp：{task_id}"
        if header in text:
            return
        append_log(scene_md_path, ("\n" if text and not text.endswith("\n") else "") + f"{header}\n")

    def append_task_patch(self, scene_md_path: str, task_id: str, md_delta: str) -> None:
        existing = extract_section_by_task_id(scene_md_path, task_id)
        merged = (existing + "\n" + md_delta.strip()).strip() if existing else md_delta.strip()
        replace_section_by_task_id(scene_md_path, task_id, merged)

    def read_task_fragments(self, scene_md_path: str, task_id: str) -> str:
        return extract_section_by_task_id(scene_md_path, task_id)

    def replace_task_summary(self, scene_md_path: str, task_id: str, final_markdown: str) -> None:
        replace_section_by_task_id(scene_md_path, task_id, final_markdown)

    def planner_compact(
        self,
        llm_call: Callable[[str], Any],
        *,
        scene: str,
        existing_memory: str,
        raw_procedure: str,
        prompt_system: str,
    ) -> MemoryPatch:
        user_prompt = (
            "请根据 System Prompt 的要求，对以下数据执行脱水合并操作：\n"
            f"<SCENE>\n{scene}\n</SCENE>\n"
            f"<EXISTING_MEMORY>\n{existing_memory}\n</EXISTING_MEMORY>\n"
            f"<RAW_PROCEDURE>\n{raw_procedure}\n</RAW_PROCEDURE>\n"
            "仅输出 JSON。"
        )
        raw = llm_call(f"{prompt_system}\n{user_prompt}")
        parsed = self._parse_json_object(raw)
        md_delta = str(parsed.get("md_delta", "")).strip()
        context_patch = str(parsed.get("context_patch", "")).strip()
        if not md_delta:
            md_delta = "- 无可合并事实"
        return MemoryPatch(md_delta=md_delta, context_patch=context_patch)

    def planner_consolidate(
        self,
        llm_call: Callable[[str], Any],
        *,
        scene: str,
        end_status: str,
        history_fragments: str,
        latest_ram: str,
        prompt_system: str,
    ) -> str:
        user_prompt = (
            "请根据 System Prompt 的要求，为本次任务生成最终定稿总结。\n"
            f"<SCENE>\n{scene}\n</SCENE>\n"
            f"结束原因：{end_status}\n"
            f"<HISTORY_FRAGMENTS>\n{history_fragments}\n</HISTORY_FRAGMENTS>\n"
            f"<LATEST_REALTIME_DATA>\n{latest_ram}\n</LATEST_REALTIME_DATA>\n"
            "输出 Markdown。"
        )
        return str(llm_call(f"{prompt_system}\n{user_prompt}")).strip()


    # Backward-compatible generic APIs for non-planner agents
    def compact(self, agent_name: str, stale_messages: list[str]) -> MemoryPatch:
        joined = "\n".join(stale_messages).strip()
        return MemoryPatch(md_delta=f"- {agent_name}: {joined[:1000]}", context_patch=joined[-500:])

    def append_md_delta(self, agent_name: str, patch: MemoryPatch) -> None:
        log_path = str(self.logs_dir / f"{agent_name}.md")
        append_log(log_path, f"\n{patch.md_delta}\n")

    def consolidate_task(self, agent_name: str, ram_tail: list[str]) -> str:
        log_path = str(self.logs_dir / f"{agent_name}.md")
        old = read_log(log_path)
        final = (old + "\n" + "\n".join(ram_tail)).strip()
        overwrite_log(log_path, final + "\n")
        LOGGER.info("Task consolidated for %s", agent_name)
        return final

    def list_scene_memory_files(self) -> list[str]:
        planner_root = self.logs_dir.parent.parent / "memory" / "planner_agent"
        planner_root.mkdir(parents=True, exist_ok=True)
        return sorted([str(p) for p in planner_root.glob("*.md") if p.name != "memory.md"])

    def read_file(self, path: str) -> str:
        return read_log(path)

    def _safe_scene_name(self, scene: str) -> str:
        cleaned = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", scene.strip())
        return cleaned[:80] or "default_scene"

    def _parse_json_object(self, raw: Any) -> dict[str, Any]:
        text = str(raw).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        match = re.search(r"\{[\s\S]*\}", text)
        payload = match.group(0) if match else text
        return json.loads(payload)
