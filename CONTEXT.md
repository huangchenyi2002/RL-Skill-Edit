# CONTEXT

- 当前内容：仓库只包含一个优化器 `RL-Skill-Edit`；初始 Skill 只是输入与配对报告基线。
- 当前状态：功能代码截至 `0b2173e`；347 项测试、Ruff、compileall、API-free smoke 和 test-only 均通过，已发布。
- 发布位置：`git@github.com:huangchenyi2002/RL-Skill-Edit.git`；默认 `main` 与 `kaggle_data` 均保存同一 RL-only 版本。
- 关键决定：优化前不读取 Test 内容；Validation 选定 Skill 后才绑定并执行 Test。不同 split 的 task ID 与任一 init/golden workbook 内容都必须互不重叠，答案范围变化不能绕过检查。
