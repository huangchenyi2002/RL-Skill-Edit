# CONTEXT

- 当前内容：仓库已改为唯一 `RL-Skill-Edit` CLI；`initial_skill` 只作为输入和配对报告基线。
- 当前状态：Task 5 已完成单流程训练、严格 provenance、全输出事务提交、`--test-only` 和 API-free smoke；下一步清理剩余旧 OSD 文件与文档。
- 关键决定：训练或复测都先写并校验完整 staging 输出树；成功替换后把调用前输出完整保留为同级 `.<output-name>.previous`，下次提交先清理该快照且不触碰当前结果，安装失败则用经指纹验证的快照恢复。
