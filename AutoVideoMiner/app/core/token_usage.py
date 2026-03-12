"""Token usage accounting helpers."""

from __future__ import annotations

from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars/token for mixed zh/en safe approximation)."""
    return max(1, len(text) // 4)


def init_token_usage(settings: dict[str, Any]) -> dict[str, Any]:
    agents_cfg = settings.get("agents", {})
    agents = {}
    total_budget = 0
    for agent_name, cfg in agents_cfg.items():
        budget = int(cfg.get("llm", {}).get("token_budget", 400_000))
        agents[agent_name] = {"used": 0, "budget": budget}
        total_budget += budget
    return {"agents": agents, "total_used": 0, "total_budget": total_budget}


def add_token_usage(token_usage: dict[str, Any], agent_name: str, used_tokens: int) -> dict[str, Any]:
    if not token_usage:
        return token_usage
    agents = token_usage.setdefault("agents", {})
    agent = agents.setdefault(agent_name, {"used": 0, "budget": 400_000})
    agent["used"] += int(used_tokens)
    token_usage["total_used"] = int(token_usage.get("total_used", 0)) + int(used_tokens)
    return token_usage
