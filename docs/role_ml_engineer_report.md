## 机器学习算法工程师工作报告

### 1. 模型设计
- **基线模型：逻辑回归**
  - 选用 `LogisticRegression`，设置 `class_weight='balanced'` 以缓解类别不平衡。
  - 网格搜索 `C ∈ {0.1, 1.0, 10.0}`，最终选择 `C=1.0`。
- **提升模型：XGBoost 分类器**
  - 关键参数：`max_depth`、`learning_rate`、`min_child_weight`、`gamma`。
  - 采用网格搜索组合，并通过验证集 F1 指标选取最佳组合（`max_depth=5, learning_rate=0.1, min_child_weight=1, gamma=0.0`）。
- 训练流程封装在 `src/pipelines/train_models.py`，读取 `configs/model_training.yaml` 实现参数化管理。

### 2. 训练与评估
- 数据输入为 `data/processed/` 下的训练/验证/测试集，特征已完成编码与标准化。
- 指标计算及图表输出由 `src/utils/metrics.py` 完成，生成混淆矩阵、ROC 曲线、指标表格。
- 模型表现：
  - 逻辑回归测试集 ROC-AUC 0.8747，F1 0.7228。
  - XGBoost 测试集 ROC-AUC 0.9649，F1 0.8209，显著优于基线。
- 训练日志记录在 `logs/model_training.log`，模型与调参历史保存在 `models/`、`artifacts/`。

### 3. 可解释性与输出
- 生成特征重要性 `reports/tables/feature_importance_xgboost.csv`，图表位于 `reports/figures/feature_importance_xgboost.png`。
- 重要特征包括历史逾期率、借款杠杆比、历史还款行为等，与业务经验一致。
- 测试集预测结果保存为 `artifacts/test_predictions_{model}.parquet`，可供后续策略分析。

### 4. 后续改进建议
- 引入 LightGBM 或 CatBoost 对比不同梯度提升方案。
- 尝试基于阈值搜索优化 F1 或成本函数，满足不同策略需求。
- 引入 SHAP 值或 Permutation Importance，进一步增强解释能力。

