## 文档阅读导航

> 本指南面向编程与机器学习零基础读者，帮你按循序渐进的顺序阅读 `docs/` 目录中的资料。每个阶段都列出了推荐文档、适合的读者角色以及阅读目标，完成上一阶段再继续往下，可逐步搭建对项目的完整理解。

---

### 1. 入门与整体概览
- **先读**：
  - `README.md` — 了解项目能做什么、如何一键运行。
  - `docs/beginner_step_by_step.md` — 按生活化比喻理解“从零到跑通”的全过程。
  - `docs/beginner_function_walkthrough.md` — 对照代码，弄清每个脚本具体调用了哪些函数。
- **适合人群**：所有初次接触仓库的同学。
- **阅读目标**：知道项目解决的问题、核心流程和关键入口命令。

### 2. 需求背景与业务语境
- **推荐顺序**：
  1. `docs/requirements_analysis.md` — 弄清业务目标、角色分工、验收标准。
  2. `docs/project_structure.md` — 掌握仓库目录划分，知道代码与数据放在哪里。
  3. `docs/project_summary.md` — 快速了解当前成果与下一步规划。
- **适合人群**：准备写报告或想与业务方沟通的同学。
- **阅读目标**：明确“为什么要做”“角色各自负责什么”，为后续技术文档打基础。

### 3. 数据理解与处理流程
- **推荐顺序**：
  1. `docs/data_understanding.md` — 熟悉原始 LC/LP/LCIS 数据结构与字段意义。
  2. `docs/data_processing_design.md` — 查看数据清洗、特征工程的设计原则。
  3. `docs/data_pipeline_summary.md` — 对照真实运行日志，确认流水线产物。
- **结合代码**：阅读 `src/ai_homework/data/` 与 `src/ai_homework/features/` 下的模块；对照 `configs/data_processing.yaml` 理解可配置项。
- **适合人群**：对数据清洗、特征工程有兴趣或要扩展特征的同学。
- **阅读目标**：搞清楚“原始数据如何变成建模数据”以及关键决策点。

### 4. 模型训练、调优与可解释性
- **推荐顺序**：
  1. `docs/modeling_design.md` — 理解模型选型、阈值策略与评估指标。
  2. `docs/modeling_results.md` — 查看最新实验结果与性能对比。
  3. `docs/interpretability_analysis.md` — 了解特征重要性、SHAP 分析的结论。
- **结合代码**：关注 `src/ai_homework/pipelines/train_models.py`、`src/ai_homework/models/training.py`、`src/ai_homework/evaluation/metrics.py`，以及 `configs/model_training.yaml`。
- **产物位置**：所有自动生成的指标、图表在 `outputs/reports/`；模型与调参历史在 `outputs/artifacts/`。
- **阅读目标**：掌握模型训练流程、调参依据与结果解释，便于复现或继续优化。

### 5. 测试、交付与团队协作
- **推荐顺序**：
  1. `docs/testing_strategy.md` — 了解测试覆盖范围与验证方法。
  2. `docs/collaboration_guide.md` — 熟悉多角色协作、提交作业或 PR 的流程。
  3. `docs/project_summary.md`（再次回顾）— 整合全局信息，为交付或答辩做准备。
- **扩展资料**：
  - `docs/role_*_report.md` — 各角色输出的阶段性总结。
  - `docs/migration/2025-structure-refactor.md` — 关注目录重构等历史变更。
- **阅读目标**：确保交付完整、测试可靠，团队成员对各自职责有清晰认识。

### 6. 后续深入与自定义探索
- 如果要改造流程：先调整 `configs/`，再通过 `docs/beginner_function_walkthrough.md` 定位对应函数。
- 如果要写总结或展示：结合 `docs/project_summary.md`、`outputs/reports/` 生成 PPT 或演示脚本。
- 如果要扩展数据源或模型：先在对应设计文档中记录假设，再更新代码与文档保持一致。

---

**使用建议**：把本指南视为“导航页”，首次接触项目时先快速扫一遍；之后每当文档更新或有新成员加入，也可以把这份导航分享给他们，确保大家遵循同一套阅读顺序和项目语境。祝学习顺利！ 

