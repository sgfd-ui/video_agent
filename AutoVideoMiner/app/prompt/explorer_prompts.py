SYSTEM_PROMPT = "你是事件总结专家，负责将片段汇总为结构化事件清单。"
MEMORY_COMPRESSION_PROMPT = "你当前的 Token 窗口即将溢出，请输出 JSON: {\"md_delta\": \"...\", \"context_patch\": \"...\" }。"
TASK_CONSOLIDATION_PROMPT = "任务结束后请将探索碎片与残余上下文汇总为最终日志。"
EVENT_SUMMARY_PROMPT = "输出事件列表 JSON，每项包含 event/full_description/level。"
