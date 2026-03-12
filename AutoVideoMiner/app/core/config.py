"""Configuration loader and per-agent Bedrock model factories."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import yaml
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrock

from AutoVideoMiner.app.core.logger import get_logger

LOGGER = get_logger("core.config")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / "config" / ".env", override=False)
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("settings.yaml 格式错误")
    return data


def reload_settings() -> dict[str, Any]:
    load_settings.cache_clear()
    return load_settings()


def _env(name: str, required: bool = False, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise EnvironmentError(f"缺少环境变量: {name}")
    return value


def _build_client_from_cfg(model_cfg: dict[str, Any]):
    region = model_cfg.get("region") or _env(model_cfg["region_env"], required=False, default="us-east-1")
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=_env(model_cfg["access_key_env"], required=False),
        aws_secret_access_key=_env(model_cfg["secret_key_env"], required=False),
        aws_session_token=_env(model_cfg["session_token_env"], required=False),
    )


def get_agent_model_config(agent_name: str, force_reload: bool = False) -> dict[str, Any]:
    settings = reload_settings() if force_reload else load_settings()
    cfg = settings.get("agents", {}).get(agent_name)
    if not cfg:
        raise KeyError(f"agent 未配置: {agent_name}")
    if "llm" not in cfg or "embedding" not in cfg:
        raise KeyError(f"agent {agent_name} 缺少 llm/embedding 配置")
    return cfg


def get_llm_for_agent(agent_name: str, force_reload: bool = False) -> ChatBedrock:
    cfg = get_agent_model_config(agent_name, force_reload=force_reload)["llm"]
    if cfg["provider"] != "aws_bedrock":
        raise ValueError("仅支持 aws_bedrock")
    client = _build_client_from_cfg(cfg)
    model_kwargs = {
        "temperature": cfg.get("temperature", 0),
        "top_p": cfg.get("top_p", 1),
        "max_tokens": cfg.get("max_tokens", 4096),
    }
    LOGGER.info("LLM init from settings | agent=%s model=%s kwargs=%s", agent_name, cfg.get("model_id"), model_kwargs)
    return ChatBedrock(
        client=client,
        model_id=cfg["model_id"],
        model_kwargs=model_kwargs,
    )


def get_embedding_for_agent(agent_name: str, force_reload: bool = False) -> BedrockEmbeddings:
    cfg = get_agent_model_config(agent_name, force_reload=force_reload)["embedding"]
    if cfg["provider"] != "aws_bedrock":
        raise ValueError("仅支持 aws_bedrock")
    client = _build_client_from_cfg(cfg)
    LOGGER.info("Embedding init from settings | agent=%s model=%s", agent_name, cfg.get("model_id"))
    return BedrockEmbeddings(client=client, model_id=cfg["model_id"])
