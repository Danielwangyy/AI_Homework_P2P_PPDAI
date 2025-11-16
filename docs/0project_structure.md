## 项目目录结构说明

以下说明基于当前仓库实际结构，便于新成员快速定位资料与代码。

### 根目录
- `README.md`：项目概览、运行指引与常见问题。
- `CONTRIBUTING.md`：协作规范与提交要求。
- `pyproject.toml`：Python 打包与依赖配置。
- `finalize_report.py`：生成最终报告的脚本入口。

### 配置与环境
- `configs/`：集中存放 YAML 配置；当前包含 `data_processing.yaml`、`model_training.yaml`、`data_pipeline.yml` 等数据与建模流水线参数。
- `environments/`：环境说明与 `requirements.txt`，用于重建运行环境。
- `scripts/`：常用自动化脚本，如 `run_pipeline.sh`、`setup.sh`、`setup.ps1`。

### 数据资产
- `data/`
  - `raw/source_data/`：原始数据与数据字典（LC/LP/LCIS 等），保持只读。
  - `interim/`：数据清洗与特征构造后的中间产物（如 `loan_master.parquet`），用于调试复现。
  - `processed/`：建模阶段直接使用的 `train/valid/test.parquet` 等最终数据集。
- `catboost_info/`：CatBoost 训练产生的中间日志与指标文件，供调参追踪。

### 文档与报告
- `docs/`：需求分析、数据理解、建模设计、角色视角报告等 Markdown 文档。
- `outputs/reports/`：流水线自动生成的分析图表与表格（`figures/`、`tables/`）。

### 运行产物
- `outputs/`
  - `artifacts/`：模型调参记录、预测结果、阈值搜索及标准化器等工件。
  - `models/`：不同算法输出的权重文件（joblib）。
  - `middata/`：数据处理流水线的中间输出（清洗后的 CSV、异常记录等）。
  - `logs/`：数据准备与模型训练的运行日志。

### 源码与测试
- `src/ai_homework/`：主代码包，采用 `src` 布局，包含：
  - `cli/`：命令行入口模块。
  - `data/`：原始数据加载、清洗、标签生成逻辑。
  - `features/`：特征工程与特征构造工具。
  - `models/`：模型训练、调参与推理相关代码。
  - `pipelines/`：端到端流水线封装，供 CLI 与脚本复用。
  - `evaluation/`：评估指标与可视化函数。
  - `utils/`：配置、日志、IO 等通用工具。
  - `visualization/`：可视化脚本与模板。
- `src/ai_homework_p2p.egg-info/`：打包生成的元数据文件，便于 pip 安装与版本追踪。
- `tests/`：单元测试与配置校验，目前包含 `tests/configs/test_configs.py`。

> 说明：`data/processed/`、`outputs/` 等易变产物可通过流水线重新生成，默认不纳入版本控制。若需清理或备份，请先确认不会影响复现实验。
