"""Streamlit GUI for AutoVideoMiner control plane."""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import streamlit as st

from AutoVideoMiner.app.flow.graph import control_gate, run_once
from AutoVideoMiner.app.tool.sqlite_db import fetch_event_snapshot


def run_app() -> None:
    st.set_page_config(page_title="AutoVideoMiner", layout="wide")
    st.title("AutoVideoMiner 控制台")

    target_scene = st.text_input("目标场景", "欧洲 住宅 室外监控")
    run_mode = st.radio("运行模式", ["event", "timer"], horizontal=True)

    end_time = None
    if run_mode == "timer":
        hour_range = st.slider("运行时间段", 0, 24, (9, 18))
        end_time = datetime.combine(datetime.today(), time(hour=hour_range[1] % 24))

    start_clicked = st.button("Start Agent", type="primary")
    stop_clicked = st.button("Graceful Stop")

    db_path = str(Path(__file__).resolve().parents[2] / "data" / "db" / "autovidminer.db")
    workspace = str(Path(__file__).resolve().parents[2] / "data" / "workspace")

    if "state" not in st.session_state:
        st.session_state.state = {
            "target_scene": target_scene,
            "run_mode": run_mode,
            "end_time": end_time,
            "event_snapshot": fetch_event_snapshot(db_path),
            "stop_flag": False,
        }

    if start_clicked:
        st.session_state.state.update(
            {
                "target_scene": target_scene,
                "run_mode": run_mode,
                "end_time": end_time,
                "event_snapshot": fetch_event_snapshot(db_path),
                "stop_flag": False,
            }
        )
        if control_gate(st.session_state.state) != "END":
            st.session_state.state = run_once(st.session_state.state, db_path=db_path, workspace=workspace)

    if stop_clicked:
        st.session_state.state["stop_flag"] = True

    state = st.session_state.state

    st.subheader("实时指标")
    a, b, c = st.columns(3)
    a.metric("并发线程", len(state.get("planner_tasks", [])))
    b.metric("URL 入库数量", len(state.get("raw_urls", [])))
    c.metric("高光片段数量", len(state.get("high_light_clips", [])))

    st.subheader("Agent OS 日志流")
    st.write(
        [
            f"ControlGate => {control_gate(state)}",
            f"Planner tasks => {len(state.get('planner_tasks', []))}",
            f"Manifest ready => {'manifest' in state}",
        ]
    )

    st.subheader("结果清单")
    st.json(state.get("manifest", {"events": []}))

    st.subheader("HITL 接管中心")
    st.info("若某节点返回 ask_human 中断标记，可在此提供人工判定并继续流程。")
