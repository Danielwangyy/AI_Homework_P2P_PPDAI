## 模型训练与评估设计

### 1. 目标与指标
- 通过对 `train/valid/test` 数据划分，构建并评估违约预测模型，重点关注：Accuracy、Precision、Recall、F1-score、ROC-AUC。
- 基线模型使用逻辑回归，提升模型扩展为 LightGBM、XGBoost、CatBoost，并比较性能。
- 对验证集执行成本敏感阈值调优（默认为 FN:FP=5:1），兼顾业务关注的召回率。
- 输出混淆矩阵与 ROC 曲线图，结合阈值摘要表分析分类策略影响。

### 2. 数据输入与特征
- 数据来源：`data/processed/{train,valid,test}.parquet`。
- 每份数据包含 `ListingId`、特征列与 `label`。
- 特征经过前序流水线的一致预处理（缺失填补、One-Hot、标准化），此阶段直接使用。
- 若后续新增特征或重新执行清洗流程，仅需保持列名对齐，即可复用训练脚本。
- 可通过 `configs/data_processing.yaml` 中的 `feature_drops` 段落按“exact/prefix”等规则移除特征，实现快速的消融实验（当前示例：去除 `loan_date_year_*` 相关维度）。

### 3. 模型方案
1. **逻辑回归（Baseline）**
   - 使用 `sklearn.linear_model.LogisticRegression`，设置 `class_weight='balanced'` 与较高 `max_iter` 以收敛。
   - 网格搜索正则化强度 `C`，最终模型在验证集 F1 最优。
2. **LightGBM 分类器**
   - 依赖 `lightgbm.LGBMClassifier`，网格搜索 `num_leaves`、`max_depth`、`min_child_samples` 等结构超参。
   - 自动根据正负样本比例与成本权重设置 `scale_pos_weight`。
3. **XGBoost 分类器**
   - 使用 `xgboost.XGBClassifier`，调参维度包含 `max_depth`、`learning_rate`、`min_child_weight`、`gamma` 与抽样率。
   - 同样根据成本权重调整 `scale_pos_weight`，提升少数类召回。
4. **CatBoost 分类器**
   - 依赖 `catboost.CatBoostClassifier`，参数覆盖 `depth`、`l2_leaf_reg` 等；禁用默认日志输出以保持训练安静。
   - 通过 `class_weights` 调节正负样本成本比。

### 4. 评估与可视化
- 评估流程：
  1. 使用交叉验证搜索参数并在训练集重训最佳模型；
  2. 在验证集计算概率，执行阈值网格搜索与成本指标统计；
  3. 固定最佳阈值后对 train/valid/test 输出最终指标。
- 指标输出：
  - `reports/tables/model_metrics.{csv,json}` 保存多模型多数据集的核心指标、成本统计与阈值；
  - `reports/tables/threshold_summary.csv` 汇总每个模型的最佳阈值与关键召回/精度。
- 可视化：
  - ROC 曲线图、混淆矩阵图存放于 `reports/figures/`；
  - SHAP Summary 图展示树模型的特征贡献度。

### 5. 代码结构规划
- `configs/model_training.yaml`：定义数据路径、模型列表、调参网格、指标名称等。
- `src/utils/data_io.py`：封装 processed 数据加载逻辑。
- `src/utils/metrics.py`：集中实现指标计算、图表绘制辅助函数。
- `src/models/training.py`：实现统一的交叉验证训练、调参流程，并返回训练历史。
- `src/pipelines/train_models.py`：命令行入口，负责配置加载、阈值调优、指标导出、模型持久化与 SHAP 可解释性。

### 6. 日志与产物
- 训练日志写入 `logs/model_training.log`，包含参数设置、评分结果等信息。
- 模型文件（`joblib` 或 `json` 格式）输出到 `models/`，命名体现模型类型与时间戳。
- 网格搜索结果、阈值搜索、特征重要性、SHAP 数值等写入 `artifacts/` 与 `reports/tables/` 目录，便于复用。

### 7. 测试与验证
- 本阶段通过 `tests/test_project_configs.py` 保障配置完整性；后续可补充小样本集成测试验证阈值与指标函数。

### 8. 后续扩展方向
- 引入更多模型（如随机森林、深度学习）并统一评估接口。
- 探索多目标优化（Precision-Recall 曲线、收益函数）或分层阈值策略。
- 结合 SHAP、Permutation Importance、LIME 等方法加强可解释与稳定性分析。

### 9. 特征消融实验流程
- **设计目标**：逐步移除（或替换）某类特征，评估其对模型训练时间、验证指标、阈值与特征重要性的影响，从而判断该特征族的业务价值与上线必要性。
- **配置方式**：在 `configs/data_processing.yaml` 的 `feature_drops` 段落中分别声明 `before_encoding`（原始列）与 `after_encoding`（One-Hot 等衍生列）的移除规则，支持 `exact`、`prefixes`、`suffixes`、`contains`、`regex` 等形式，并在日志中自动记录被移除的字段列表。
- **执行步骤**：
  1. 基线运行：保留全部特征执行一次流水线，记录模型指标、阈值、SHAP 等基准结果；
  2. 修改配置：按实验计划更新 `feature_drops`，或结合白名单直接取消特定特征；
  3. 重跑流水线：使用统一命令 `python3 -m ai_homework.cli.run_pipeline`，系统会重新生成数据与模型；
  4. 归档结果：将最新的 `model_metrics.csv`、`threshold_summary.csv`、特征/SHAP 报告与关键日志片段整理入文档或实验表格，便于对比。
- **记录与版本管理**：推荐在 `docs/modeling_results.md` 中增设“消融实验记录”小节，包含实验名称、移除特征描述、主要指标差异、结论/后续动作；同时可在 Git 分支或笔记中注明配置编号与运行时间，保证可追溯性。

