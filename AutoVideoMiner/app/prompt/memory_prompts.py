"""Prompt templates for short/mid-term memory compaction and consolidation."""

MEMORY_COMPACTION_PROMPT = (
    "你现在的 Token 窗口已满。请审视当前内容并完成以下两件事：\n"
    "1. [Markdown 更新]：对比现有的 YYYY-MM-DD.md，输出需要新增或修改的事实。\n"
    "2. [上下文补丁]：为了腾出空间，请将即将删除的对话总结成 500 字以内核心简报。\n"
    "请按 JSON 输出：{\"md_delta\": \"...\", \"context_patch\": \"...\"}"
)

TASK_SUMMARY_PROMPT = (
    "请综合磁盘碎片事实与当前窗口实时数据，生成最终 Task Summary，"
    "并覆盖写入对应 agent 的中期记忆日志。"
)
