## 项目目录结构说明

- `configs/`：集中存放 YAML 配置（数据处理、模型训练等），所有路径均以项目根目录为基准的相对路径，便于跨环境复现。
- `data/`
  - `raw/`：原始数据，`source_data/` 内请放置老师提供的 `P2P_PPDAI_DATA` 文件夹（含 LC/LP/LCIS 与数据字典）；新的原始数据可按来源或日期再建子目录。
  - `interim/`：数据清洗、特征构造等中间结果，支持回溯。
  - `processed/`：建模所需的最终训练/验证/测试数据集。
  - `external/`：外部辅助数据源（宏观经济、风控指标等）。
- `docs/`：项目文档（需求、设计、测试、结构说明等），新增 `migration/` 子目录记录重大结构变更。
- `environments/`：虚拟环境与依赖说明，仅保留 `requirements.txt` 与环境搭建指南。
- `notebooks/`：探索性分析与实验 Notebook，命名建议为 `YYYYMMDD_author_topic.ipynb`。
- `outputs/`：统一存放可再生成的运行产物，默认被 `.gitignore` 忽略。
  - `artifacts/`：模型调参记录、预测结果、特征分析输出等。
  - `models/`：训练生成的模型权重文件。
  - `experiments/`：实验追踪信息（如 CatBoost 日志、训练断点）。
  - `logs/`：流水线运行日志。
  - `reports/figures/`、`reports/tables/`：自动生成的图表与表格。
- `reports/`：手写或总结类报告文件，通常为 Markdown。
- `src/ai_homework/`：项目主包，采用 `src` layout 方便发布与导入。
  - `cli/`：命令行入口，可在 Agent 中说“请执行 python -m ai_homework.cli.run_pipeline”调用。
  - `data/`：原始数据加载、清洗、标签生成模块。
  - `features/`：特征工程与构造逻辑。
  - `models/`：模型训练与调参工具。
  - `pipelines/`：端到端流水线封装，供脚本与 CLI 复用。
  - `utils/`：通用工具（配置、IO、日志等）。
  - `evaluation/`：评估与可视化工具（指标计算、曲线绘制等）。
  - `visualization/`：后续可补充的可视化脚本/模板。
- `tests/`：按包结构划分的测试用例目录。

> 说明：`outputs/`、`data/processed/` 等易变产物默认不纳入 Git，需要时可通过配置文件重新生成。
