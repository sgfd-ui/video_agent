from __future__ import annotations

import re
from pathlib import Path


def read_log(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def append_log(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)


def overwrite_log(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def extract_section_by_task_id(path: str, task_id: str) -> str:
    text = read_log(path)
    if not text:
        return ""
    pattern = re.compile(rf"## ID: {re.escape(task_id)}\n([\s\S]*?)(?=\n## ID: |\Z)")
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def replace_section_by_task_id(path: str, task_id: str, new_content: str) -> None:
    text = read_log(path)
    header = f"## ID: {task_id}\n"
    replacement = f"{header}{new_content.strip()}\n"
    pattern = re.compile(rf"## ID: {re.escape(task_id)}\n([\s\S]*?)(?=\n## ID: |\Z)")
    if pattern.search(text):
        text = pattern.sub(replacement.strip("\n"), text)
        if not text.endswith("\n"):
            text += "\n"
        overwrite_log(path, text)
    else:
        append_log(path, ("\n" if text and not text.endswith("\n") else "") + replacement)
