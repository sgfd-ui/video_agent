from __future__ import annotations

import importlib


def get_prompt(agent: str, key: str) -> str:
    module_name = f"AutoVideoMiner.app.prompt.{agent}_prompts"
    module = importlib.import_module(module_name)
    var_name = f"{agent.upper()}_PROMPT"
    prompt_dict = getattr(module, var_name)
    value = prompt_dict.get(key)
    if value is None:
        raise KeyError(f"Prompt key not found: {agent}.{key}")
    return value
