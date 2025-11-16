## 模型解释性分析

### 1. 目标
- 识别对违约预测贡献最大的特征，支持风控策略与人工复核。
- 评估模型是否符合业务直觉，避免“黑箱”风险。

### 2. 方法
- 对三类树模型（LightGBM、XGBoost、CatBoost）输出 Gain/Permutation 特征重要性，并统一生成为 CSV 与柱状图。
  - `outputs/reports/tables/feature_importance_{model}.csv`
  - `outputs/reports/figures/feature_importance_{model}.png`
- 基于同一批次训练结果，抽样 2,000 条样本计算 SHAP 值，生成全局 summary 图与明细表：
  - `outputs/reports/figures/shap_summary_{model}.png`
  - `outputs/reports/tables/shap_values_{model}.csv`
- 再次训练可执行 `python3 -m ai_homework.cli.run_pipeline --skip-data`，流程会自动刷新上述解释性产物。

### 3. 主要发现
- **跨模型共识特征**：`history_repay_ratio`、`loan_term`、`loan_amount`/`loan_amount_per_term`、`history_avg_loan_amount` 与 `outstanding_to_history_amount_ratio` 在三套模型的重要性依旧领先，显示历史偿付能力与额度结构是区分违约风险的核心。
- **LightGBM 侧重额度结构**：在移除 `loan_date_year_*` 之后，模型几乎完全依赖额度类派生指标（如 `loan_amount_to_history_amount_ratio`、`history_avg_term_payment`），高额借款且相对历史额度偏大的样本 SHAP 值仍明显为正。
- **XGBoost 仍保留时间切片信息**：除了历史行为指标外，`loan_date_quarter_*` 与部分 `loan_date_month_*` 特征继续进入前列，说明季度/月份层面的批次效应对 XGBoost 决策仍有帮助。
- **CatBoost 更聚焦额度与行为**：去掉年份后，`loan_term`、`history_repay_ratio`、`loan_term_rating_interaction`、`outstanding_to_history_amount_ratio` 等指标占据前十，时间 Dummy 的存在感明显下降，评分等级 (`rating_numeric`) 依旧体现风险分层。
- 整体来看，正向 SHAP 值多由“借款额相对历史偏大 + 逾期比例高 + 部分时间段”驱动，反向 SHAP 值则对应“还款率高 + 等级优秀 + 借款额度适中”的人群。

### 4. 解读建议
- 人工复核可优先关注“历史还款率快速下降”“当前借款额显著高于历史均值”且落在高风险时间窗口（如 15 年、16 年一季度）的样本。
- 将评分等级、认证信息与额度比指标结合，构建多维评分卡或规则引擎，以形成更细粒度的风险分层。
- 针对业务关切的单笔样本，可直接查询 `shap_values_{model}.csv` 或使用 summary 图定位关键驱动因素，在客服或风控沟通中提升透明度。

