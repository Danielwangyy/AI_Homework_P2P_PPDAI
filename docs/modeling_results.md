## 模型训练结果摘要

> 本文档记录最近一次完整流水线的训练产出，持续更新时请在原有结构上追加内容并保留基线。每次实验完成后即刻补充对应章节，保持团队同步。

## 1. 运行概览

### 1.1 执行与配置

- 启动方式：`python3 -m ai_homework.cli.run_pipeline --skip-data`
- 配置文件：数据准备 `configs/data_processing.yaml`，模型训练 `configs/model_training.yaml`
- 最近执行时间：2025-11-16 09:50-09:55（本地日志）

### 1.2 数据与产出

- 输入数据：`data/processed/{train,valid,test}.parquet`
- 输出目录：
  - 模型文件：`outputs/models/`
  - 评估指标：`outputs/reports/tables/model_metrics.{csv,json}`
  - 图表：`outputs/reports/figures/`
  - 预测与调参记录：`outputs/artifacts/`

## 2. 消融实验记录

### 2025-11-16 09:50 + 基线

> ⚠️ 以下指标为 2025-11-16 09:50-09:55（本地时间）在默认配置、保留全部特征情况下执行 `python3 -m ai_homework.cli.run_pipeline --skip-data` 所得基线。除非重新跑该基线，请勿改动本段记录及对应表格；所有消融实验对照均应引用此批次产出（配置与阈值已在同批次日志与 `outputs/reports/tables/model_metrics.*` 中固化）。

#### 1.关键观察

#### 2.超参搜索摘要

##### 2.1.阈值调优

- 成本敏感策略：`fn_cost=5`、`fp_cost=1`
- 扫描 201 个阈值
- 最优阈值：Logistic 0.185、LightGBM 0.500、XGBoost 0.068、CatBoost 0.212，验证集期望成本≈6.6k，差异 <1%

##### 2.2.Logistic Regression

- 保持 `class_weight=balanced`、`solver=lbfgs`、`max_iter=500`
- 网格搜索 `C∈{0.1, 1, 10, 20}`
- `C=0.1` 的 5 折 F1 均值最高（≈0.595，std≈0.003），说明较强的 L2 正则仍最稳

##### 2.3.LightGBM

- 搜索空间：`num_leaves∈{31,63,127}`、`max_depth=-1`、`min_child_samples∈{5,10,20,40}`、`min_split_gain∈{0.0,0.1}`
- 关键基础参数：`n_estimators=40`、`learning_rate=0.05`、`scale_pos_weight≈6.66`
- 最优组合：`num_leaves=127`、`min_child_samples=10`、`min_split_gain=0.0`，5 折 F1 均值≈0.609（std≈0.0011），移除 `loan_date_year` 后性能保持稳定

##### 2.4.XGBoost

- 搜索空间：`max_depth∈{3,5}`、`min_child_weight∈{1,5}`、`learning_rate∈{0.03,0.1}`、`gamma∈{0.0,0.1}`、`subsample∈{0.6,0.8}`
- 最优组合：`max_depth=5`、`min_child_weight=1`、`learning_rate=0.1`、`gamma=0.0`、`subsample=0.6`
- 5 折 F1 均值≈0.610（std≈0.0011）

##### 2.5.CatBoost

- 固定参数：`iterations=400`、`learning_rate=0.05`、`class_weights=[1.0,4.0]`
- 比较范围：`depth∈{5,7}`、`l2_leaf_reg∈{3,7}`
- 最优组合：`depth=7`、`l2_leaf_reg=3`，5 折 F1 均值≈0.615（std≈0.0014），兼顾稳定性与泛化

#### 3.各模型结果对比

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

#### 4.可视化索引

##### 4.1.ROC 曲线

![Logistic Regression ROC](../outputs/reports/figures/roc_curve_logistic_regression.png)
![LightGBM ROC](../outputs/reports/figures/roc_curve_lightgbm.png)
![XGBoost ROC](../outputs/reports/figures/roc_curve_xgboost.png)
![CatBoost ROC](../outputs/reports/figures/roc_curve_catboost.png)

##### 4.2.混淆矩阵

![Logistic Regression CM](../outputs/reports/figures/confusion_matrix_logistic_regression.png)
![LightGBM CM](../outputs/reports/figures/confusion_matrix_lightgbm.png)
![XGBoost CM](../outputs/reports/figures/confusion_matrix_xgboost.png)
![CatBoost CM](../outputs/reports/figures/confusion_matrix_catboost.png)

##### 4.3.SHAP 特征影响力

![LightGBM SHAP](../outputs/reports/figures/shap_summary_lightgbm.png)
![XGBoost SHAP](../outputs/reports/figures/shap_summary_xgboost.png)
![CatBoost SHAP](../outputs/reports/figures/shap_summary_catboost.png)

### 示例：2025-11-16 10:30 + 消融实验1（移除 `loan_date_year_*`）

#### 1.关键观察

- 成本敏感的调优策略下，阈值保持在FN:FP=5:1时， Recall 在 0.982 ~ 1.000，Precision 聚集于 0.429 ~ 0.440；如需提升 Precision，需与业务复核成本假设与业务 KPI ，可调整成本比或或考虑分层阈值或尝试 F-beta（β<1）作为调优目标
- 树模型 AUC 领先，CatBoost 测试集 AUC=0.686、期望成本≈6.56k，当前最稳；LightGBM、XGBoost 分别为 0.679/0.681，差距 <0.01
- 移除 `loan_date_year_*` 后，LightGBM 召回略降（~0.997→~0.985/0.982），Precision +0.007，F1 轻微下降，指向年份特征贡献有限
- LightGBM 维持 `n_estimators=40` 可控制训练时长，`No further splits with positive gain` 警告偶发但可接受
- 整体 Accuracy ≈0.43 由类别权重与样本分布决定；F1≈0.601 接近现有特征上限，突破需依赖新特征或集成策略

#### 2.超参搜索摘要

##### 2.1.阈值调优

##### 2.2.Logistic Regression

##### 2.3.LightGBM

##### 2.4.XGBoost

##### 2.5.CatBoost

#### 3各模型结果对比

#### 4可视化索引

##### 3.1.ROC 曲线

##### 4.2.混淆矩阵

##### 4.3.SHAP 特征影响力

> （后续实验依次追加，保持同一结构，必要时可插入表格或简短结论）

## 3. 产物索引

- 调参与阈值搜索：`outputs/artifacts/{model}_tuning_history.csv`、`outputs/artifacts/{model}_threshold_search.csv`
- 测试集预测：`outputs/artifacts/test_predictions_{model}.parquet`
- 模型文件：`outputs/models/{model}_{timestamp}.joblib`
- SHAP 明细：`outputs/reports/tables/shap_values_{model}.csv`

## 4. 后续改进方向

- 复核成本假设与业务 KPI；若需提升 Precision，可调整成本比或尝试 F-beta（β<1）作为调优目标
- 深入特征工程，引入时序画像、额度使用率等变量，突破现有 AUC 天花板
- 扩大树模型调参范围（如 `num_leaves`、`gamma`、`l2_leaf_reg`），或探索 Stacking/Blending 以获取集成增益
- 基于阈值调优结果设计多档风控策略，支持差异化运营决策

## 5. 附录：消融实验步骤

- **基线步骤（勿删改）**：消融前确保完成一次“全特征”流水线运行（命令 `python3 -m ai_homework.cli.run_pipeline --skip-data`，默认 `feature_drops` 为空），并沿用该批次指标对照
- **设计与执行**：使用 `feature_drops` 的 exact/prefix/regex 规则明确移除的特征集；训练日志出现“特征消融[...]：移除列 ...”以确认生效
- **指标对比**：对比 `model_metrics.csv` 与 `threshold_summary.csv` 中的 F1、ROC-AUC、成本及阈值；差异过小需排查特征冗余，差异过大需结合业务解读
- **可解释性分析**：结合 `feature_importance_{model}.csv` 与 SHAP 输出，关注重要性排序是否迁移，避免意料之外的替代特征
- **记录方式**：为每个批次新增标题“日期时间 + 消融标签”，并按以下层级补充关键信息
- **结果应用**：对贡献有限的特征考虑在生产环境移除；对关键特征需进一步优化质量或监控
