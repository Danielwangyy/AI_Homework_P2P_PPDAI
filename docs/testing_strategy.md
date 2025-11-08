## 测试方案设计

### 1. 测试范围
- 数据流水线（`ai_homework/pipelines/prepare_data.py`）：验证数据加载、清洗、特征构造、分层划分及导出流程。
- 模型训练流水线（`ai_homework/pipelines/train_models.py`）：验证模型训练、调参、评估、产物输出。
- 模型结果：核对指标、图表、特征重要性、预测结果文件是否齐全且格式正确。
- 公共工具模块：`ai_homework/evaluation/metrics.py`、`ai_homework/utils/data_io.py`、`ai_homework/models/training.py` 等关键函数应具备单元测试覆盖。

### 2. 测试类型与用例
- **单元测试（需补充）**：
  - 标签生成函数 `_label_from_lp`、`_label_from_lcis`：使用极端样例与边界条件断言标签正确。
  - 特征构建函数 `build_feature_dataframe`：断言输出列集合、缺失率与数值范围。
  - 训练工具 `ai_homework/models/training.py`：
    - `_train_with_cv`：使用小型 DataFrame 和固定随机种子，断言返回的最佳参数与 history 记录长度。
    - `train_model`：针对未知模型类型抛出 `ValueError`，确保异常路径被覆盖。
  - 指标工具 `ai_homework/evaluation/metrics.py`：
    - `compute_classification_metrics`：基于手工构造的混淆矩阵断言 Accuracy/F1/expected_cost。
    - `plot_confusion_matrix`、`plot_roc_curve`：借助临时目录断言图像文件生成且尺寸非空。
  - 数据加载工具 `ai_homework/utils/data_io.py`：使用 tmp_path 构造样例 Parquet，验证 ID、标签和特征列拆分正确。
- **集成测试（建议强化）**：
  1. 准备精简版输入数据（几十行），在 Agent 模式下说明：“请执行 python -m ai_homework.pipelines.prepare_data --config configs/data_processing.yaml”，随后检查是否生成中间/最终文件与日志。
  2. 在同一精简数据上，对 Agent 说：“请执行 python -m ai_homework.pipelines.train_models --config configs/model_training.yaml”，或“请执行 python -m ai_homework.cli.run_pipeline --skip-data”，确认 `outputs/` 下的模型、指标、图表、SHAP 文件齐全，并核对阈值摘要的成本字段。
  3. 结合 `pytest` fixtures，可以让 Agent 运行 “请执行 python -m pytest -m integration” 来触发集成测试分组。
- **性能测试**：记录脚本运行时长（约 2 分钟以内），监控内存使用，确保在课程环境可复现。
- **回归测试**：变更特征或参数后重复执行流水线，核对输出文件时间戳与差异。

### 3. 指标与验收准则
- 定量指标：Accuracy、Precision、Recall、F1、ROC-AUC。
- 定性准则：
  - 输出目录结构完整，文件命名符合规范。
  - 模型日志中无未捕获异常；若存在警告（如字体），已通过替换标签等方式消除。
- 客户验收标准：测试集 ROC-AUC ≥ 0.90；最终模型 XGBoost 达到 0.9649。

### 4. 工具与环境
- 虚拟环境：建议使用 `.venv/` 或 Conda 环境，依赖列于 `environments/requirements.txt`（包含 `pytest`、`pytest-cov` 等测试工具）。需要安装时，可直接告诉 Agent：“请在当前环境执行 pip install -r environments/requirements.txt”。
- 常用操作示例（均可直接对 Agent 说）：
  - “请执行 python -m ai_homework.cli.run_pipeline”
  - “请执行 python -m ai_homework.cli.run_pipeline --skip-data”
  - “请执行 python -m ai_homework.pipelines.prepare_data --config configs/data_processing.yaml”
  - “请执行 python -m ai_homework.pipelines.train_models --config configs/model_training.yaml”
  - “请执行 python -m pytest”
  - “请执行 python -m pytest -m integration”
- 日志与结果：
  - 日志：`outputs/logs/*.log`
  - 指标：`outputs/reports/tables/model_metrics.csv`
  - 图表：`outputs/reports/figures/*.png`

### 5. 风险与对策
- **依赖风险**：XGBoost 依赖 `libomp`，如遇缺失，可请 Agent 执行 “请通过 Homebrew 安装 libomp” 并确认环境变量设置。
- **数据异常风险**：保留原始数据与中间产物备份；若需修正，可直接回放流水线。
- **模型稳定性风险**：引入多模型评估比较，防止单模型失效；后续可加入交叉验证与阈值调优。

### 6. 改进计划
- 将核心函数纳入 `pytest` 框架，自动化运行单元测试。
- 为集成测试提供缩略数据集与 Fixture，避免长时间运行。
- 建立持续集成流程（GitHub Actions 或本地脚本），提交代码后自动执行数据与模型流水线，并收集覆盖率（`pytest --cov=src --cov-report=term-missing`）。
- 新增可解释性输出（如 SHAP）后，将文件生成校验纳入测试范围，确保解释结果同步更新。

### 7. 执行顺序建议
1. 让 Agent 更新依赖，并确认“请执行 python -m pytest” 能顺利完成。
2. 优先补充 `src/utils`、`src/models/training.py` 的单元测试，覆盖主要逻辑分支。
3. 构建缩略版数据集及集成测试，纳入 `pytest` 标记体系。
4. 将上述 Agent 话术整理进 CI 或开发 checklist，做到提交前自动运行。

