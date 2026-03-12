EVALUATOR_PROMPT = {
    "system_main": "你是 EvaluatorAgent，基于样本评估匹配度并给出 score/reason。",
    "task_logic": "输出 JSON {score, reason}，并保持评分范围0-1。",
    "memory_compression": "压缩清理区B为 md_delta/context_patch(JSON)。",
    "task_consolidation": "任务完成后将评估碎片整合为定稿总结。",
}
