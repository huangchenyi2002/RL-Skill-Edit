# CONTEXT

- 当前内容：仓库只包含一个优化器 `RL-Skill-Edit`；初始 Skill 只是输入与配对报告基线。
- 当前状态：单一 CLI、真实 Spreadsheet 运行时、API-free smoke、严格 provenance、事务发布和边界测试均已落地。
- 关键决定：优化前不读取 Test 内容；Validation 选定 Skill 后，唯一 loader 形成 Test digest，训练写入或 test-only 完整校验 provenance，成功后才执行 Test。
