from __future__ import annotations

import importlib


def _import_prompt_module(agent: str):
    candidates = [
        f"AutoVideoMiner.app.prompts.{agent}.{agent}_prompts",
        f"AutoVideoMiner.app.prompt.{agent.replace('_agent', '')}_prompts",
    ]
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(f"Prompt module not found for agent={agent}")


def get_prompt(agent: str, prompt_name: str) -> str:
    module = _import_prompt_module(agent)
    value = getattr(module, prompt_name, None)
    if value is None:
        raise KeyError(f"Prompt not found: {module.__name__}.{prompt_name}")
    return value
