## 测试方案设计

### 1. 测试范围
- **数据准备流水线**：`src/ai_homework/pipelines/prepare_data.py`（调用 `data/loading.py`、`data/labeling.py`、`data/cleaning.py`、`features/engineering.py` 等模块）。
- **模型训练流水线**：`src/ai_homework/pipelines/train_models.py` 及其依赖的 `models/training.py`、`evaluation/metrics.py`、`utils/data_io.py`。
- **命令行入口**：`src/ai_homework/cli/run_pipeline.py`，覆盖“完整流程”和“跳过单环节”两种模式。
- **公共工具**：`src/ai_homework/utils/{config,data_io,logger}.py`，保证配置解析、数据加载、日志初始化稳定可复用。

### 2. 测试类型与示例
- **单元测试（待补充）**
  - 标签生成：`_label_from_lcis`、`_label_from_lp`、`generate_labels`。
  - 数据清洗：`clean_lc`（年龄裁剪、二值字段标准化、缺失填补）。
  - 特征构造：`build_feature_dataframe`（输出列、缺失率、数值字段类型）。
  - 模型训练：`_train_with_cv`、`train_model`（参数网格遍历、异常路径）。
  - 指标函数：`compute_classification_metrics`、`plot_confusion_matrix`、`plot_roc_curve`。
  - 数据读取：`load_dataset`（ID / label 拆分正确性）。
- **集成测试（建议使用精简数据集）**
  1. 执行 `python -m ai_homework.pipelines.prepare_data --config configs/data_processing.yaml`，核对 `data/interim/loan_master.parquet`、`data/processed/{train,valid,test}.parquet` 与 `outputs/artifacts/numeric_scaler.pkl`。
  2. 执行 `python -m ai_homework.pipelines.train_models --config configs/model_training.yaml` 或 `python -m ai_homework.cli.run_pipeline`，核对 `outputs/models/`、`outputs/reports/{tables,figures}/`、`outputs/artifacts/`。
  3. 为上述步骤编写 pytest fixture，可通过 `python -m pytest -m integration` 调度。
- **回归测试**
  - 变更配置或特征后重放流水线，比对 `outputs/reports/tables/model_metrics.csv`、`outputs/reports/tables/threshold_summary.csv`。
  - 检查 `outputs/logs/data_preparation.log`、`outputs/logs/model_training.log` 的关键节点。
- **性能/稳定性测试**
  - 记录两条流水线在标准数据集下的耗时（目标 < 5 分钟）与内存峰值（目标 < 4GB）。
  - 关注 macOS 上的 `libomp`、字体等依赖是否正常加载。

### 3. 指标与验收准则
- **定量指标**（取自当前 `outputs/reports/tables/model_metrics.csv`，阈值调优成本比 FN:FP=5:1）：
  - LightGBM（Test）：Recall ≈ 0.884、Precision ≈ 0.392、F1 ≈ 0.543、ROC-AUC ≈ 0.760；
  - XGBoost / CatBoost（Test）：Recall ≈ 0.898-0.902、Precision ≈ 0.386、F1 ≈ 0.540-0.541、ROC-AUC ≈ 0.762-0.763；
  - Logistic Regression（Test）：Recall ≈ 0.883、Precision ≈ 0.372、F1 ≈ 0.523、ROC-AUC ≈ 0.730。
- **定性准则**
  - 输出目录结构完整（`outputs/models|artifacts|reports|logs`）。
  - 日志无未捕获异常；若出现字体警告需通过字体注册或英文标签处理。
  - CLI 参数组合（`--skip-data`、`--skip-train`、自定义配置路径）均运行成功。
- **业务验收建议**
  - 与客户角色确认成本权重 5:1 是否符合业务场景；若需提升 Precision，可调高阈值或改用 `strategy=f_beta`。
  - 以“测试集 ROC-AUC ≥ 0.75 且 Recall ≥ 0.88”作为当前作业版本的达标标准。

### 4. 工具与运行方式
- 虚拟环境：`./scripts/setup.sh` 自动创建 `.venv` 并安装 `environments/requirements.txt`。
- 常用测试命令（可直接对 Cursor Agent 说）：
  - `请执行 python -m ai_homework.pipelines.prepare_data --config configs/data_processing.yaml`
  - `请执行 python -m ai_homework.pipelines.train_models --config configs/model_training.yaml`
  - `请执行 python -m ai_homework.cli.run_pipeline --skip-train`
  - `请执行 python -m ai_homework.cli.run_pipeline --skip-data`
  - `请执行 python -m pytest`
- 日志与产物位置：
  - 日志：`outputs/logs/{data_preparation,model_training}.log`
  - 指标/阈值：`outputs/reports/tables/{model_metrics,threshold_summary}.csv`
  - 图表：`outputs/reports/figures/*`
  - SHAP 与特征重要性：`outputs/reports/tables/shap_values_{model}.csv`、`outputs/reports/tables/feature_importance_{model}.csv`

### 5. 风险与对策
- **依赖风险**：macOS 平台需确保 `libomp` 可用；脚本已自动扩展 `DYLD_LIBRARY_PATH`，若仍失败需手动通过 Homebrew 安装。
- **数据风险**：原始 CSV 修改可能导致字段缺失；建议保留 `data/raw/source_data` 压缩包备份。
- **模型风险**：阈值变化影响 Precision/Recall，务必保留 `outputs/artifacts/*_threshold_search.csv` 与 `summary_{model}.json`。

### 6. 改进计划
1. 为核心函数补充 pytest 单测，并接入 `pytest --cov=src/ai_homework`。
2. 提供压缩版演示数据集，缩短集成测试耗时。
3. 在 CI（GitHub Actions 或本地脚本）中串联 `prepare_data` → `train_models` → `pytest`。
4. 编写指标同步脚本（或 checklist），确保 `docs/` 与 `outputs/reports/` 指标一致。

