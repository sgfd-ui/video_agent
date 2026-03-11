"""MCP client bootstrap placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCPClients:
    browser: str
    sqlite: str
    filesystem: str


def init_mcp_clients() -> MCPClients:
    return MCPClients(
        browser="Browser MCP configured",
        sqlite="SQLite MCP configured",
        filesystem="FileSystem MCP configured",
    )
