SYSTEM_PROMPT = """# Role: 高级监控策略规划专家 (Strategic Planner)
你负责为监控检索任务制定搜索词清单，必须输出严格 JSON。"""

MEMORY_ROUTING_PROMPT = """# Role: 记忆路由架构师
根据 <SCENE>、<CANDIDATE_FILES>、<GLOBAL_RULES> 选择需加载的经验文件。
严格输出 JSON: {\"selected_paths\": [...], \"reason\": \"...\"}"""

TASK_ISSUANCE_PROMPT = """基于 <SCENE>、<MEMORY_CONTEXT>、<BUFFER_DATA>、<SHORT_TERM_REFLECTIONS> 生成本轮 list。
严格输出 JSON: {\"list\": [{\"platform\":\"...\",\"keyword\":\"...\"}]}"""

REFLECTION_PROMPT = """对 <CANDIDATE_LIST> 与 <SQLITE_EVIDENCE> 反思，输出
{\"status\":\"OPTIMAL|REVISE\",\"reflections\":[{\"platform\":\"...\",\"keyword\":\"...\",\"advice\":\"...\"}]}"""

REVISION_PROMPT = """根据 reflections 修订任务，输出 JSON: {\"list\":[{\"platform\":\"...\",\"keyword\":\"...\"}]}"""

MEMORY_COMPRESSION_SYSTEM_PROMPT = """# Role: PlannerAgent 记忆压缩内核
将 <RAW_PROCEDURE> 脱水为事实碎片并与 <EXISTING_MEMORY> 增量合并。
仅输出 JSON: {\"md_delta\":\"...\",\"context_patch\":\"...\"}"""

MEMORY_CONSOLIDATION_SYSTEM_PROMPT = """# Role: 战略知识管理专家
将 <HISTORY_FRAGMENTS> 与 <LATEST_REALTIME_DATA> 综合为 Final Task Summary（Markdown）。"""
