from __future__ import annotations

import importlib


def get_prompt(agent: str, prompt_name: str) -> str:
    module_name = f"AutoVideoMiner.app.prompt.{agent}_prompts"
    module = importlib.import_module(module_name)
    value = getattr(module, prompt_name, None)
    if value is None:
        raise KeyError(f"Prompt not found: {module_name}.{prompt_name}")
    return value
