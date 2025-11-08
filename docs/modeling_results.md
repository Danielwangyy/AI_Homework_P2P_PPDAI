## 模型训练结果摘要

-### 1. 运行信息
- 启动方式：在 Agent 模式下说明“请执行 python -m ai_homework.pipelines.train_models”
- 配置：`configs/model_training.yaml`
- 最近执行时间：2025-11-08 20:24-20:40（本地日志）
- 输入数据：`data/processed/{train,valid,test}.parquet`
- 输出目录：
  - 模型文件：`outputs/models/`
  - 评估指标：`outputs/reports/tables/model_metrics.{csv,json}`
  - 图表：`outputs/reports/figures/`
  - 预测与调参记录：`outputs/artifacts/`

### 2. 主要指标

| 模型 | 数据集 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Train | 0.540 | 0.374 | 0.887 | 0.526 | 0.733 |
| Logistic Regression | Valid | 0.539 | 0.373 | 0.882 | 0.524 | 0.731 |
| Logistic Regression | Test | 0.536 | 0.372 | 0.883 | 0.523 | 0.730 |
| LightGBM | Train | 0.602 | 0.415 | 0.934 | 0.575 | 0.804 |
| LightGBM | Valid | 0.573 | 0.393 | 0.881 | 0.544 | 0.759 |
| LightGBM | Test | 0.571 | 0.392 | 0.884 | 0.543 | 0.760 |
| XGBoost | Train | 0.572 | 0.396 | 0.920 | 0.553 | 0.785 |
| XGBoost | Valid | 0.559 | 0.386 | 0.898 | 0.540 | 0.761 |
| XGBoost | Test | 0.558 | 0.386 | 0.898 | 0.540 | 0.762 |
| CatBoost | Train | 0.564 | 0.390 | 0.910 | 0.546 | 0.775 |
| CatBoost | Valid | 0.556 | 0.385 | 0.898 | 0.539 | 0.761 |
| CatBoost | Test | 0.559 | 0.386 | 0.902 | 0.541 | 0.763 |

> 更多字段（阈值、成本等）见 `outputs/reports/tables/model_metrics.csv`

### 3. 关键观察
- 所有模型均使用成本敏感阈值调优（FN:FP=5:1），因此 Recall 极高且 Precision 较低，Accuracy 明显下降；需要和业务方确认该成本假设是否符合预期。
- 树模型（LightGBM / XGBoost / CatBoost）在 Recall 与 F1 上略优于逻辑回归，但指标差距不大，提示特征区分度仍有限。
- LightGBM 以 0.544（验证集 F1）和 0.760（验证集 ROC-AUC）领先，其最优阈值约 0.846，对默认 0.5 阈值的偏离较大。
- SHAP 分析已对三类树模型输出，重点特征集中在历史逾期、负债比、借款金额等变量；详细排名见 `reports/tables/shap_values_{model}.csv`。

### 4. 可视化
- ROC 曲线、混淆矩阵已覆盖四个模型：
  - `outputs/reports/figures/roc_curve_{model}.png`
  - `outputs/reports/figures/confusion_matrix_{model}.png`
- 另外导出 `outputs/reports/figures/shap_summary_{model}.png` 展示特征影响力。

### 5. 产物
- 调参与阈值搜索：`outputs/artifacts/{model}_tuning_history.csv`、`outputs/artifacts/{model}_threshold_search.csv`
- 测试集预测：`outputs/artifacts/test_predictions_{model}.parquet`
- 模型文件：`outputs/models/{model}_{timestamp}.joblib`
- SHAP 详细数值：`outputs/reports/tables/shap_values_{model}.csv`

### 6. 后续改进思路
- 复核成本权重与业务 KPI，必要时重新定义阈值或切换到 F-beta（β<1）目标以提升 Precision。
- 引入更多特征（例如用户信用历史时间序列聚合、交易行为指标），提升模型区分度。
- 在树模型基础上尝试集成策略（Stacking、Blending）或 AutoML，评估是否带来增益。
- 结合阈值调优结果，设计不同风险等级的分层策略，用于业务决策。

