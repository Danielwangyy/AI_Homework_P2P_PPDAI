## 模型训练结果摘要

### 1. 运行信息

- 启动方式：终端执行 `python3 -m ai_homework.cli.run_pipeline --skip-data`
- 配置文件：数据准备 `configs/data_processing.yaml`（本次运行已跳过），模型训练 `configs/model_training.yaml`
- 最近执行时间：2025-11-16 08:28-08:33（本地日志）
- 输入数据：`data/processed/{train,valid,test}.parquet`
- 输出目录：
  - 模型文件：`outputs/models/`
  - 评估指标：`outputs/reports/tables/model_metrics.{csv,json}`
  - 图表：`outputs/reports/figures/`
  - 预测与调参记录：`outputs/artifacts/`

### 2. 主要指标

| 模型                | 数据集 | Accuracy | Precision | Recall | F1    | ROC-AUC |
| ------------------- | ------ | -------- | --------- | ------ | ----- | ------- |
| Logistic Regression | Train  | 0.433    | 0.431     | 0.999  | 0.602 | 0.683   |
| Logistic Regression | Valid  | 0.433    | 0.431     | 0.998  | 0.602 | 0.669   |
| Logistic Regression | Test   | 0.432    | 0.430     | 0.998  | 0.601 | 0.669   |
| LightGBM            | Train  | 0.472    | 0.448     | 1.000  | 0.619 | 0.738   |
| LightGBM            | Valid  | 0.455    | 0.440     | 0.985  | 0.608 | 0.680   |
| LightGBM            | Test   | 0.451    | 0.438     | 0.982  | 0.605 | 0.679   |
| XGBoost             | Train  | 0.430    | 0.429     | 1.000  | 0.601 | 0.776   |
| XGBoost             | Valid  | 0.430    | 0.429     | 1.000  | 0.601 | 0.680   |
| XGBoost             | Test   | 0.430    | 0.429     | 1.000  | 0.601 | 0.681   |
| CatBoost            | Train  | 0.432    | 0.430     | 1.000  | 0.601 | 0.745   |
| CatBoost            | Valid  | 0.431    | 0.430     | 1.000  | 0.601 | 0.687   |
| CatBoost            | Test   | 0.430    | 0.430     | 0.999  | 0.601 | 0.686   |

> 更多字段（阈值、期望成本等）见 `outputs/reports/tables/model_metrics.csv`

### 3. 超参数调优概览

- **Logistic Regression**：保持 `class_weight=balanced`、`solver=lbfgs`、`max_iter=500`，仅对 `C∈{0.1, 1, 10, 20}` 网格搜索；`C=0.1` 取得最高 5 折 F1 均值 0.595（std≈0.003），说明较强的 L2 正则仍是最稳的选择。
- **LightGBM**：在 `num_leaves∈{31,63,127}`、`max_depth=-1`、`min_child_samples∈{5,10,20,40}` 与 `min_split_gain∈{0.0,0.1}` 之间搜索，基础参数包含 `n_estimators=40`、`learning_rate=0.05`、`scale_pos_weight≈6.66`。最佳组合为 `num_leaves=127`、`min_child_samples=10`、`min_split_gain=0.0`，5 折 F1 均值约 0.609（std≈0.0011），说明移除 `loan_date_year` 后性能仍保持稳定。
- **XGBoost**：针对 `max_depth∈{3,5}`、`min_child_weight∈{1,5}`、`learning_rate∈{0.03,0.1}`、`gamma∈{0.0,0.1}`、`subsample∈{0.6,0.8}` 搜索；`max_depth=5`、`min_child_weight=1`、`learning_rate=0.1`、`gamma=0.0`、`subsample=0.6` 的组合获得最高 5 折 F1 均值约 0.610（std≈0.0011）。
- **CatBoost**：固定 `iterations=400`、`learning_rate=0.05`、`class_weights=[1.0,4.0]`，比较 `depth∈{5,7}` 与 `l2_leaf_reg∈{3,7}`。最佳参数 `depth=7`、`l2_leaf_reg=3`，5 折 F1 均值约 0.615（std≈0.0014），兼顾稳定性与泛化。
- **阈值调优**：统一使用成本敏感策略（`fn_cost=5`、`fp_cost=1`）扫描 201 个阈值。最优阈值分别为 Logistic 0.185、LightGBM 0.500、XGBoost 0.068、CatBoost 0.212，验证集期望成本控制在 6.6k 左右，差异 <1%。

### 4. 关键观察

- 成本敏感阈值让四个模型的 Recall 保持在 0.982~1.000 区间，Precision 聚集在 0.429~0.440；若业务希望进一步提高精度，需要重新评估 FN:FP=5:1 的设定或考虑分层阈值。
- 树模型在 AUC 上仍占优势，CatBoost 测试集达到 0.686、期望成本约 6.56k，为目前最稳的一档；LightGBM 与 XGBoost 分别为 0.679/0.681，差距维持在 0.007 以内。
- 去掉 `loan_date_year_*` 后，LightGBM 的验证/测试召回从 ~0.997 降至 ~0.985/0.982，Precision 提升约 0.007，F1 仅下降 ≈0.003，说明年份特征贡献有限但对召回有轻微帮助。
- 将 LightGBM 的 `n_estimators` 保持在 40 后，训练时长依旧较短，`No further splits with positive gain` 警告偶发但数量可控，参数约束仍在限制树深。
- 整体 Accuracy 约 0.43，源于类别权重与样本分布；在此策略下 F1≈0.601 已接近当前特征上限，需依靠新增特征或集成策略突破。

### 5. 消融实验记录

- **设计与执行**：每次消融需明确“移除的特征集”和“对比的基线版本”，推荐在 `feature_drops` 配置中通过 exact/prefix/regex 规则落地，并在模型训练日志中观察“特征消融[...]：移除列 ...”提示，确认配置生效。
- **指标对比**：对比 `model_metrics.csv` 与 `threshold_summary.csv`，重点观察 F1、ROC-AUC、成本指标及阈值变化；若差异小于预期，应考虑特征冗余或噪声情况；若差异较大，需结合业务解释正负面的影响。
- **可解释性分析**：通过 `feature_importance_{model}.csv` 与 SHAP 输出确认剩余特征的重要性排序是否发生迁移，防止消融后出现意料之外的“替代特征”驱动模型。
- **记录方式**：建议以表格或清单形式记录“实验编号 → 消融特征 → 关键指标变化 → 结论/后续动作”，并在文档中保留链接或引用，确保团队成员可以复现与复核。
- **结果应用**：若某类特征对指标贡献有限，可考虑在生产环境永久移除以简化采集与维护；若对召回/精准度影响显著，则应保留，或进一步优化其质量与稳定性（例如异常值处理、实时口径校验）。

### 5. 可视化

#### ROC 曲线

![Logistic Regression ROC](../outputs/reports/figures/roc_curve_logistic_regression.png)
![LightGBM ROC](../outputs/reports/figures/roc_curve_lightgbm.png)
![XGBoost ROC](../outputs/reports/figures/roc_curve_xgboost.png)
![CatBoost ROC](../outputs/reports/figures/roc_curve_catboost.png)

#### 混淆矩阵

![Logistic Regression CM](../outputs/reports/figures/confusion_matrix_logistic_regression.png)
![LightGBM CM](../outputs/reports/figures/confusion_matrix_lightgbm.png)
![XGBoost CM](../outputs/reports/figures/confusion_matrix_xgboost.png)
![CatBoost CM](../outputs/reports/figures/confusion_matrix_catboost.png)

#### SHAP 特征影响力

![LightGBM SHAP](../outputs/reports/figures/shap_summary_lightgbm.png)
![XGBoost SHAP](../outputs/reports/figures/shap_summary_xgboost.png)
![CatBoost SHAP](../outputs/reports/figures/shap_summary_catboost.png)

### 6. 产物

- 调参与阈值搜索：`outputs/artifacts/{model}_tuning_history.csv`、`outputs/artifacts/{model}_threshold_search.csv`
- 测试集预测：`outputs/artifacts/test_predictions_{model}.parquet`
- 模型文件：`outputs/models/{model}_{timestamp}.joblib`
- SHAP 明细：`outputs/reports/tables/shap_values_{model}.csv`

### 7. 后续改进思路

- 复核成本假设与业务 KPI，若需提升 Precision，可尝试调整成本比或改用 F-beta（β<1）作为调优目标。
- 深入特征工程，引入时序画像、额度使用率等变量，突破现有 AUC 天花板。
- 扩大树模型调参范围（如 `num_leaves`、`gamma`、`l2_leaf_reg`），或尝试 Stacking/Blending 探索集成增益。
- 基于阈值调优结果设计多档风控策略，支持差异化运营决策。
