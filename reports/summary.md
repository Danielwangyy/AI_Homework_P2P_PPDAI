## 模型评估报告摘要（2025-11-08 20:40）

- **模型范围**：逻辑回归、LightGBM、XGBoost、CatBoost；均使用成本敏感阈值调优（FN:FP=5:1）。
- **核心指标（验证集）**：
  - LightGBM：Precision 0.393、Recall 0.881、F1 0.544、ROC-AUC 0.759。
  - XGBoost：Precision 0.386、Recall 0.898、F1 0.540、ROC-AUC 0.761。
  - CatBoost：Precision 0.385、Recall 0.898、F1 0.539、ROC-AUC 0.761。
  - Logistic Regression：Precision 0.373、Recall 0.882、F1 0.524、ROC-AUC 0.731。
- **阈值选择**：最佳阈值范围 0.35-0.85，显著高于/低于默认 0.5；需结合业务验证 FN 成本假设。
- **产出位置**：
  - 指标表：`outputs/reports/tables/model_metrics.{csv,json}`
  - 阈值摘要：`outputs/reports/tables/threshold_summary.csv`
  - 图表：`outputs/reports/figures/`（ROC、混淆矩阵、SHAP Summary）
  - SHAP 数值：`outputs/reports/tables/shap_values_{model}.csv`
- **下一步建议**：
  - 复盘 Precision 下降原因，可调整成本比或引入校准策略；
  - 增补特征（交易频率、资产负债率洞察）以提升区分度；
  - 针对高召回结果设计分层策略（高、中、低风险）并结合业务试点。

