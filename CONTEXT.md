# CONTEXT

- 当前内容：仓库已改为唯一 `RL-Skill-Edit` CLI；`initial_skill` 只作为输入和配对报告基线。
- 当前状态：Task 5 已完成单流程训练、严格 provenance、全输出事务提交、`--test-only` 和 API-free smoke；下一步清理剩余旧 OSD 文件与文档。
- 关键决定：依赖文件统一为 `requirements.txt`；训练或复测都先写同一个 staging 输出树，严格校验全部 RL 与报告产物后才整体替换旧结果，任何失败恢复旧输出并保留不可回滚证据。
