## 模型训练结果摘要

> 本文档记录最近一次完整流水线的训练产出，持续更新时请在原有结构上追加内容并保留基线。每次实验完成后即刻补充对应章节，保持团队同步。

## 1. 运行概览

### 1.1 执行与配置

- 启动方式：`python3 -m ai_homework.cli.run_pipeline`
- 配置文件：数据准备 `configs/data_processing.yaml`，模型训练 `configs/model_training.yaml`
- 最近执行时间：2025-11-16 11:20-11:26（本地日志）

### 1.2 数据与产出

- 输入数据：`data/processed/{train,valid,test}.parquet`
- 输出目录：
  - 模型文件：`outputs/models/`
  - 评估指标：`outputs/reports/tables/model_metrics.{csv,json}`
  - 图表：`outputs/reports/figures/`
  - 预测与调参记录：`outputs/artifacts/`

## 2. 消融实验记录

### 2025-11-16 11:20 + 基线

> ⚠️ 以下指标为 2025-11-16 11:20-11:26（本地时间）在默认配置、保留全部特征情况下执行 `python3 -m ai_homework.cli.run_pipeline` 所得基线。除非重新跑该基线，请勿改动本段记录及对应表格；所有消融实验对照均应引用此批次产出（配置与阈值已在同批次日志与 `outputs/reports/tables/model_metrics.*` 中固化）。

#### 1.关键观察

- 全量流水线重跑后，四个模型的验证 F1 仍集中在 0.601~0.603，说明特征工程与数据划分稳定，整体性能波动 <0.3pp。
- 成本敏感阈值收敛到更低的分数：Logistic 阈值降至 0.0815，树模型阈值落在 0.19~0.38 区间，同时维持召回 ≥0.997。
- CatBoost 继续产出最高验证 AUC（0.692）且期望成本最低（≈6.53k），LightGBM 则在验证 F1（0.603）上略占优势。
- 四个模型的误判主要来自 FP（约 6.5k 条），Precision 保持在 0.429~0.432，若需提升精度仍需重新评估 FN:FP=5:1 的成本设定或调整特征集。

#### 2.超参搜索摘要

##### 2.1.阈值调优

- 验证集概率通过 `_tune_threshold` 按成本策略（`fn_cost=5`，`fp_cost=1`）扫描 201 个阈值；
- 最优阈值依次为：Logistic 0.0815、LightGBM 0.3785、XGBoost 0.1940、CatBoost 0.3245；
- 搜索详情记录在 `outputs/artifacts/{model}_threshold_search.csv`，综合指标写入 `summary_{model}.json`。
- 默认 `grid_size=201` 在 0.05~0.95 区间生成等距候选阈值（步长≈0.0045），既覆盖默认的 0.5，又兼顾精度与计算成本，可在 `configs/model_training.yaml` 中调整。
- 评估阶段同步输出 `expected_cost = fp_cost × FP + fn_cost × FN`，其中 FP 代表误拒好客户、FN 代表放行坏客户。当前成本设定（FN:FP=5:1）体现“坏账风险优先”的业务策略；模型的 expected_cost 越低，说明在相同权重下整体业务损失越小，可作为模型选型与阈值调优的共用标准。

##### 2.2.Logistic Regression

- 保持 `class_weight=balanced`、`solver=lbfgs`、`max_iter=500`
- 网格搜索 `C∈{0.1, 1, 10, 20}`
- `C=0.1` 的 5 折 F1 均值最高（≈0.595，std≈0.003），说明较强的 L2 正则仍最稳

##### 2.3.LightGBM

- 搜索空间：`num_leaves∈{31,63,127}`、`max_depth=-1`、`min_child_samples∈{5,10,20,40}`、`min_split_gain∈{0.0,0.1}`
- 关键基础参数：`n_estimators=40`、`learning_rate=0.05`、`scale_pos_weight≈6.66`
- 最优组合：`num_leaves=127`、`min_child_samples=10`、`min_split_gain=0.0`，5 折 F1 均值≈0.609（std≈0.0011）

##### 2.4.XGBoost

- 搜索空间：`max_depth∈{3,5}`、`min_child_weight∈{1,5}`、`learning_rate∈{0.03,0.1}`、`gamma∈{0.0,0.1}`、`subsample∈{0.6,0.8}`
- 最优组合：`max_depth=5`、`min_child_weight=1`、`learning_rate=0.1`、`gamma=0.0`、`subsample=0.6`
- 5 折 F1 均值≈0.609（std≈0.0011）

##### 2.5.CatBoost

- 固定参数：`iterations=400`、`learning_rate=0.05`、`class_weights=[1.0,4.0]`
- 比较范围：`depth∈{5,7}`、`l2_leaf_reg∈{3,7}`
- 最优组合：`depth=7`、`l2_leaf_reg=3`，5 折 F1 均值≈0.615（std≈0.0014），兼顾稳定性与泛化

#### 3.各模型结果对比评估

| 模型                | 数据集 | Accuracy | Precision | Recall | F1    | ROC-AUC |
| ------------------- | ------ | -------- | --------- | ------ | ----- | ------- |
| Logistic Regression | Train  | 0.429    | 0.429     | 1.000  | 0.600 | 0.686   |
| Logistic Regression | Valid  | 0.429    | 0.429     | 1.000  | 0.600 | 0.671   |
| Logistic Regression | Test   | 0.429    | 0.429     | 1.000  | 0.600 | 0.673   |
| LightGBM            | Train  | 0.440    | 0.434     | 1.000  | 0.605 | 0.741   |
| LightGBM            | Valid  | 0.436    | 0.432     | 0.997  | 0.603 | 0.687   |
| LightGBM            | Test   | 0.435    | 0.431     | 0.998  | 0.602 | 0.688   |
| XGBoost             | Train  | 0.433    | 0.431     | 1.000  | 0.602 | 0.777   |
| XGBoost             | Valid  | 0.431    | 0.430     | 0.999  | 0.601 | 0.690   |
| XGBoost             | Test   | 0.432    | 0.430     | 0.999  | 0.601 | 0.689   |
| CatBoost            | Train  | 0.437    | 0.433     | 1.000  | 0.604 | 0.748   |
| CatBoost            | Valid  | 0.435    | 0.432     | 0.999  | 0.603 | 0.692   |
| CatBoost            | Test   | 0.435    | 0.431     | 0.998  | 0.602 | 0.693   |

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

| 指标缩写 | 全称 | 标签判定 | 混淆矩阵位置 | 业务描述 |
| --- | --- | --- | --- | --- |
| TN | True Negative | 真值=未逾期、预测=未逾期 | 左上角 | 正常客户被正确放行 |
| FP | False Positive | 真值=未逾期、预测=逾期 | 右上角 | 误杀好客户（错拒或限额） |
| FN | False Negative | 真值=逾期、预测=未逾期 | 左下角 | 没拦住坏客户，潜在坏账 |
| TP | True Positive | 真值=逾期、预测=逾期 | 右下角 | 成功识别坏客户，可提前干预 |

##### 4.3.SHAP 特征影响力

![LightGBM SHAP](../outputs/reports/figures/shap_summary_lightgbm.png)
![XGBoost SHAP](../outputs/reports/figures/shap_summary_xgboost.png)
![CatBoost SHAP](../outputs/reports/figures/shap_summary_catboost.png)

### 2025-11-16 16:10 + 消融实验（移除 `loan_date_year_*`）

#### 1.关键观察

- `feature_drops` 配置将原始列与 One-Hot 展开列一并剔除，LightGBM 训练日志仅剩 67 个有效特征，确认 `loan_date_year_*` 已从训练集中移除。
- LightGBM 验证集 F1 提升至 0.608（较基线 +0.005），Precision 升至 0.440（+0.008），但 Recall 下降到 0.985（-0.012），表明年份特征主要贡献在召回端。
- Logistic Regression 的最优阈值上调至 0.185（基线 0.0815），Precision 与 F1 略有提升，但 Recall 从 1.000 微降至 0.998，概率分布右移更明显。
- CatBoost 与 XGBoost 的 F1 基本持平（≈0.601），然而 ROC-AUC 分别下滑约 0.005~0.01，说明年份特征对排序能力仍有边际贡献。
- 四个模型的 expected_cost 仍集中在 6.55k 左右，成本差异 <+0.13k，消融未显著恶化整体业务损失。

#### 2.阈值与调参摘要

- `logistic_regression`：阈值 0.185，验证集 expected_cost ≈6.56k，Recall 0.998；低概率段被进一步压缩。
- `lightgbm`：阈值提升至 0.500（原 0.3785），优化后 expected_cost ≈6.56k，Precision 提升但需关注 Recall 下探。
- `xgboost`：阈值降至 0.068（原 0.1940），保持近乎满召回，expected_cost ≈6.57k；概率分布更偏左。
- `catboost`：阈值 0.212（原 0.3245），expected_cost ≈6.56k，Recall 0.9996；需要额外特征来弥补 AUC 损失。

#### 3.各模型结果对比

| 模型 | 数据集 | Accuracy | Precision | Recall | F1 | ROC-AUC | Expected Cost | 阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Valid | 0.433 | 0.431 | 0.998 | 0.602 | 0.669 | 6.56k | 0.185 |
| Logistic Regression | Test | 0.432 | 0.430 | 0.998 | 0.601 | 0.669 | 6.57k | 0.185 |
| LightGBM | Valid | 0.455 | 0.440 | 0.985 | 0.608 | 0.680 | 6.56k | 0.500 |
| LightGBM | Test | 0.451 | 0.438 | 0.982 | 0.605 | 0.679 | 6.68k | 0.500 |
| XGBoost | Valid | 0.430 | 0.429 | 1.000 | 0.601 | 0.680 | 6.57k | 0.068 |
| XGBoost | Test | 0.430 | 0.429 | 1.000 | 0.601 | 0.681 | 6.57k | 0.068 |
| CatBoost | Valid | 0.431 | 0.430 | 1.000 | 0.601 | 0.687 | 6.56k | 0.212 |
| CatBoost | Test | 0.430 | 0.430 | 0.999 | 0.601 | 0.686 | 6.57k | 0.212 |

#### 4.可视化索引

- ROC 曲线、混淆矩阵与 SHAP Summary 均已按同名路径重新渲染（`../outputs/reports/figures/`），用于与基线视图进行对照。

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
