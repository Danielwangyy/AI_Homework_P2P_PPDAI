## 模型解释性分析

### 1. 目标
- 识别对违约预测贡献最大的特征，支持风控策略与人工复核。
- 评估模型是否符合业务直觉，避免“黑箱”风险。

### 2. 方法
- 基于最终 XGBoost 模型提取 Gain 型特征重要性，并补充 `feature_importances_` 作为兜底。
- 输出结果：
  - `reports/tables/feature_importance_xgboost.csv`
  - `reports/figures/feature_importance_xgboost.png`
- 若需要重新生成这些文件，可在 Agent 模式下说：“请执行 python -m ai_homework.pipelines.train_models --config configs/model_training.yaml”，Agent 会自动调用训练流程并刷新解释性产物。

### 3. 主要发现
- 特征 Top10 包括（截取部分）：
  1. `历史逾期率`
  2. `历史还款能力指数`
  3. `借款杠杆比`
  4. `历史逾期还款期数`
  5. `按期还款率`
- 这些特征聚焦于借款人历史表现与当前负债状况，与业务知识高度一致。
- 时间特征（如 `借款成功日期_month`）在模型中仍有一定权重，说明季节性或批次效应可能存在。

### 4. 解读建议
- 对高风险客群重点关注“历史逾期率”“借款杠杆比”，可作为人工复核的重点。
- 结合认证信息（手机、征信等）与历史成功借款次数，进一步分层客户信用等级。
- 后续可引入 SHAP 值，提供单笔样本级别解释，增强模型透明度。

