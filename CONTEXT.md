# CONTEXT

- 当前内容：仓库只包含一个优化器 `RL-Skill-Edit`；初始 Skill 只是输入与配对报告基线。
- 当前状态：单一 CLI、真实 Spreadsheet 运行时、API-free smoke、严格 provenance、事务发布和边界测试均已落地。
- 关键决定：Train 更新策略，Validation 选 checkpoint，Test 只在 Skill 冻结后读取；已移除所有不属于该流程的运行文件和公开说明。
