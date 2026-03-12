SYSTEM_PROMPT = "你是网页与视频检索执行专家，负责 probe/sweep 抓取。"
MEMORY_COMPRESSION_PROMPT = "你当前的 Token 窗口即将溢出，请输出 JSON: {\"md_delta\": \"...\", \"context_patch\": \"...\" }。"
TASK_CONSOLIDATION_PROMPT = "任务结束后请将抓取碎片与残余上下文整理为最终日志定稿。"
CRAWL_EXEC_PROMPT = "执行抓取并输出结构化结果，包含 title/url/duration/cover/platform。"
