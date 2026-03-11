"""Configuration loader and Bedrock model factory."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import yaml
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrock

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


def _build_boto3_client(region: str):
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )


def _resolve_model(alias: str) -> dict[str, Any]:
    settings = load_settings()
    if alias not in settings.get("models", {}):
        raise KeyError(f"模型别名不存在: {alias}")
    return settings["models"][alias]


def get_llm_for_agent(agent_name: str) -> ChatBedrock:
    settings = load_settings()
    agent_cfg = settings.get("agents", {}).get(agent_name)
    if not agent_cfg:
        raise KeyError(f"agent 未配置: {agent_name}")

    model_config = _resolve_model(agent_cfg["llm"])
    if model_config["provider"] != "aws_bedrock":
        raise ValueError(f"Unsupported provider: {model_config['provider']}")

    region = model_config.get("region", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    client = _build_boto3_client(region)
    return ChatBedrock(
        client=client,
        model_id=model_config["name"],
        model_kwargs={"temperature": model_config.get("temperature", 0.0)},
    )


def get_embedding_model() -> BedrockEmbeddings:
    model_config = _resolve_model("embedding_model")
    region = model_config.get("region", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    client = _build_boto3_client(region)
    return BedrockEmbeddings(client=client, model_id=model_config["name"])
