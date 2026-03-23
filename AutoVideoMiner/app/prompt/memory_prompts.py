"""Prompt templates for short/mid-term memory compaction and consolidation."""

MEMORY_COMPACTION_PROMPT = (
    "你现在的 Token 窗口已满。请完成两件事："
    "1) 对比日志输出 md_delta；2) 生成 500 字以内 context_patch。"
    "输出 JSON: {\"md_delta\": \"...\", \"context_patch\": \"...\"}"
)

TASK_SUMMARY_PROMPT = (
    "请综合磁盘碎片事实与当前窗口实时数据，生成最终 Task Summary，"
    "并覆盖写入对应 agent 的中期记忆日志。"
)
