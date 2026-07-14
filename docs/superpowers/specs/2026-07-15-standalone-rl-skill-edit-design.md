# Standalone RL-Skill-Edit Repository Design

## 目标与边界

该仓库只实现一种外部 Skill 优化方法：`RL-Skill-Edit`。`initial_skill`
只是一份不可缺少的起始输入，并在最终报告中作为配对基线；它不是第二种优化方法。

仓库不得包含原 OSD 方法的入口、估计器或运行链，包括 `study1_main.py`、
Teacher、Reference、Parser、label-space、lambda projection、current-method
artifact import 和原方法 gate。也不保留 random-edit-search，因为它是另一种搜索方法。

## 方案选择

采用“独立且可运行”的方案：保留 RL 算法，抽取真实 Student、Editor 和
Spreadsheet 评分所必需的通用运行能力，放到 RL 包自己的 adapter 中。不会仅仅隐藏
原方法文档，也不会把仓库缩成只能运行 mock 的算法样例。

未采用的方案：

- 只删除原方法入口但继续保留完整 `src/`：依赖边界仍然混合，不符合目标。
- 只保留抽象 RL 核心和 mock：结构最小，但不能直接进行真实 SpreadsheetBench 实验。

## 目标目录结构

```text
RL-Skill-Edit/
├── rl_skill_edit/
│   ├── adapters/
│   │   ├── openrouter.py       # Student/Editor 共用的最小 API 客户端
│   │   └── spreadsheet.py      # 单 Skill Student rollout、代码执行与 Excel 评分
│   ├── action_space.py
│   ├── budget.py
│   ├── cache.py
│   ├── evaluation.py           # RL evaluator protocol、mock 与真实 adapter
│   ├── manifest.py
│   ├── modules.py
│   ├── optimizer.py
│   ├── patch_generator.py
│   ├── patch_validator.py
│   ├── policy.py
│   ├── reporting.py            # 仅 initial 与 frozen RL 的配对报告
│   ├── reward.py
│   ├── state_encoder.py
│   ├── types.py
│   ├── cli.py
│   └── __main__.py
├── configs/
│   ├── rl_skill_edit.yaml
│   └── rl_skill_edit_smoke.yaml
├── data/mock_rl_skill_edit/
├── docs/
├── scripts/run_smoke.sh
├── tests/
├── README.md
└── requirements.txt
```

`baselines/`、`src/`、`experiments/` 和根目录 `config.yaml` 不再存在。

## 模块设计

### RL 核心

现有 action space、state encoder、actor-critic、局部 patch、reward、budget、cache、
manifest 和 optimizer 保持算法含义不变，只移动到顶层 `rl_skill_edit` 包。公共预算只记录
Student rollouts、Editor calls、Evaluator calls、tokens、cost、cache 和 wall time，不再暴露
Teacher 或 Reference 字段。

### 通用运行 adapter

`adapters/openrouter.py` 只提供 Student 和 Editor 所需的 chat 调用、seed、usage 与 cost
计数。`adapters/spreadsheet.py` 只支持单个强制激活 Skill：构造 prompt、提取 Python、在
隔离的临时 workbook 上执行、比较 golden range，并返回可见轨迹与 hard/soft reward。
它不包含 dispatch、no-skill rollout、Teacher/Reference endpoint、原方法 gate 或
history/label 估计，也不保留 implicit activation、启发式评分或自动补写保存代码。

`SkillArtifact` 直接向 adapter 提供 metadata 和 Markdown body，不再转换成原项目的
`SkillLibrary`。

### CLI 与报告

唯一入口为：

```bash
python -m rl_skill_edit --config configs/rl_skill_edit.yaml
```

CLI 不接受 `--methods`。它固定执行一条流程：加载 initial Skill，训练
`rl_skill_edit`，冻结 Validation 最优 Skill，再对 initial 和 frozen RL 进行统一 blind
Test。`--test-only` 只加载并校验既有 RL provenance，然后重复同一报告流程。

`reporting.py` 只输出 initial 与 RL 的 task-level rows、均值、成功率、配对差、标准误、
bootstrap interval 和 win/tie/loss。删除 current-method artifact importer、通用 method
registry 和 random-search 分支。

两个 `configs/rl_skill_edit*.yaml` 必须直接包含 OpenRouter、Student、Editor、评分
adapter、预算和路径所需的完整设置，不再通过 `repository_config` 或根目录配置间接继承。

## 数据流与隔离

1. 启动时只加载并校验 Train 与 Validation manifests，包括精确大小、唯一 ID、文件哈希
   和 split 不重叠。
2. 每个 episode 从相同 initial Skill 开始。策略只读取 Train 可见结果并选择
   `(module, operator)`；Editor 只收到该 Train evidence。
3. paired Train reward 更新策略；Validation 只选择 checkpoint，不进入策略更新。
4. RL Skill 和所有 provenance 冻结后才加载 Test manifest。
5. Test 对 initial 和 RL 使用相同 task 顺序、Student 配置、seed、repetitions 和 blind
   prompt，并禁用 cache read。

## 失败规则

- manifest 缺失、大小错误、ID 重复或 split 重叠：立即失败。
- workbook、golden range 或真实运行配置缺失：立即失败，不用启发式 reward 代替。
- Student 返回空响应或没有可执行代码：显式记录失败，不自动补答案或修改代码。
- Editor 不是严格 JSON、目标不唯一、修改越界或改变受保护结构：记录 invalid action，
  不修复、不生成兜底 patch。
- Student/API bundle 不完整：整次 evaluation 失败，不用部分样本继续训练或报告。
- `--test-only` 的 Skill、config、实现、依赖、seed 或 split digest 不匹配：立即失败。

## 删除与迁移

必须删除：

- `study1_main.py`、根目录 `config.yaml`、完整 `src/`；
- `current_method` 配置、mock artifacts、加载逻辑、usage 解析、测试和文档；
- `random_policy.py`、`random_edit_search` CLI/配置/测试/文档；
- 原 OSD README、架构图和旧的混合方法 implementation note；
- 旧的 `docs/superpowers/plans/2026-07-15-rl-skill-edit.md` 和
  `docs/superpowers/specs/2026-07-15-rl-skill-edit-design.md`；
- `data/spreadsheet/` 中随原方法复制的 Skill，以及只测试原方法兼容性的测试。

现有 RL 文件迁移到新包后，旧 `baselines/` 和 `experiments/` 整体删除。
`README.md`、`ARCHITECTURE.md`、`CONTEXT.md` 和 implementation note 均重写为单一
RL 流程。`requirements-rl.txt` 重命名为 `requirements.txt`；provenance 的 dependency
hash 改为该文件，implementation hash 覆盖完整 `rl_skill_edit/` 包和唯一 CLI，不再引用
任何旧路径。

## 验收标准

- `git ls-files` 中不存在 `study1_main.py`、`src/`、current-method artifacts、
  `random_policy.py` 或原 OSD 配置。
- 运行时代码不 import `src.*`，不含 Teacher/Reference/Parser/lambda 执行路径。
- CLI 只有 RL 训练与 `--test-only`；initial 只作为输入和报告基线。
- API-free smoke 从干净输出目录完成真实 policy sampling/update、Validation selection、
  freeze 和 blind Test；initial 与 RL 的结果可重复。
- 全部保留测试、Ruff、compileall 和 staged secret/path scan 通过。
- README 只说明 RL-Skill-Edit，并明确真实实验需要用户提供 manifests/workbooks 和 API key。
