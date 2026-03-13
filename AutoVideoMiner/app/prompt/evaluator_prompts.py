SYSTEM_PROMPT = "你是视频质检专家，负责评分和原因解释。"
MEMORY_COMPRESSION_PROMPT = "你当前的 Token 窗口即将溢出，请输出 JSON: {\"md_delta\": \"...\", \"context_patch\": \"...\" }。"
TASK_CONSOLIDATION_PROMPT = "任务结束后请将评估碎片与残余上下文汇总为最终日志。"
SCORING_PROMPT = "根据场景与样本输出 JSON: {\"score\": 0-1, \"reason\": \"...\"}。"
