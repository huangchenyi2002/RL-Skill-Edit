# CONTEXT

- 当前内容：仓库已改为唯一 `RL-Skill-Edit` CLI；`initial_skill` 只作为输入和配对报告基线。
- 当前状态：Task 5 已完成单流程训练、事务冻结、严格 provenance、`--test-only` 和 API-free smoke；下一步清理剩余旧 OSD 文件与文档。
- 关键决定：依赖文件统一为 `requirements.txt`；训练先写 staging bundle，所有持久化路径使用 bundle 内相对路径，完整成功后才替换既有冻结结果。
