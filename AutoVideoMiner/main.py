"""AutoVideoMiner entrypoint."""

from __future__ import annotations

from pathlib import Path

from AutoVideoMiner.app.gui.streamlit_app import run_app
from AutoVideoMiner.app.tool.sqlite_db import init_db


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    init_db(str(root / "data" / "db" / "autovidminer.db"))
    run_app()
