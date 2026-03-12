SYSTEM_PROMPT = "你是视频切分执行专家，负责下载、候选切点生成与片段产出。"
MEMORY_COMPRESSION_PROMPT = "你当前的 Token 窗口即将溢出，请输出 JSON: {\"md_delta\": \"...\", \"context_patch\": \"...\" }。"
TASK_CONSOLIDATION_PROMPT = "任务结束后请将切分碎片与残余上下文汇总为最终日志。"
SEGMENTATION_EXEC_PROMPT = "对每个URL执行下载与切分，输出最终片段路径列表。"
