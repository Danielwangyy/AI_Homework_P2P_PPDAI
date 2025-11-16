## 数据准备流水线执行摘要

### 运行概览
- **命令**：`python3 -m ai_homework.cli.run_pipeline --data-config configs/data_processing.yaml --skip-train`
- **配置版本**：`configs/data_processing.yaml`
- **运行时间**：2025-11-16 08:18（UTC+8）
- **日志位置**：`outputs/logs/data_preparation.log`

### 输入快照
- 原始目录：`data/raw/source_data/`
- 数据文件：`LC.csv`（合同主数据）、`LP.csv`（还款明细）、`LCIS.csv`（投资快照）
- 预处理前关键指标：合同总数 328,553，LP 明细 3,203,276，LCIS 快照 292,539

### 核心执行步骤
1. 读取原始数据并生成异常统计报告，落地至 `outputs/reports/`。
2. 对 LC、LP、LCIS 进行清洗：标准化认证字段、截尾金额与年龄、统一状态枚举。
3. 构建样本与标签：筛除未到期合同、计算 `sum_DPD` 判定违约，并追加 LCIS “逾期中”合同后仅保留有效样本。
4. 基于 `LC_labeled_samples.csv` 构造特征：应用白名单/黑名单策略剔除放款后字段，仅保留申请时可得信息，同时保留 `loan_date` 元数据用于后续切分。
5. 执行自动化特征筛选与缩放：在筛选前再次校验黑名单，综合相关系数、卡方检验、随机森林重要性，生成标准化矩阵与 scaler。
6. 优先尝试按借款成功日期进行时间切分，如遇空集自动回退至 70%/15%/15% 的分层拆分，最终产出 train/valid/test 及配套元数据。

### 主要输出
- 中间数据：`outputs/middata/LC_cleaned.csv`、`LP_cleaned.csv`、`LCIS_cleaned.csv`
- 样本文件：`outputs/middata/LC_labeled_samples.csv`（有效样本 76,725 条）、`LC_invalid_samples.csv`（未到期 255,529 条）
- 特征矩阵：`data/interim/loan_master.parquet`
- 模型输入：`data/processed/train.parquet`、`valid.parquet`、`test.parquet`
- 模型资产：`outputs/artifacts/numeric_scaler.pkl`
- 筛选报告：`outputs/reports/tables/feature_selection_summary.csv`

### 运行结论与观察
- 特征工程完全基于 `LC_labeled_samples.csv` 中的申请时信息，白名单/黑名单双重校验确保所有放款后字段与衍生项在进入模型前即被剔除。构造完成后会记录被移除列，便于审计追踪。
- `select_features()` 在运行期若检测到黑名单列会立刻报错，本次运行未触发告警；自动筛选后保留 70 个特征（相关 54、卡方 70、随机森林 70），特征清单已写入报告。
- 76,725 条有效样本按 70/15/15 拆分为 53,707 / 11,509 / 11,509；标签分布约为 57.1% 正常、42.9% 违约，数据质量指标保持稳定。
- 下一阶段需基于新特征集重新训练 LightGBM 等模型，并关注训练日志中的评估指标与潜在过拟合信号。

### 下一步建议
- 建模阶段直接引用输出的 Parquet 与 scaler，保证列顺序一致。若需减少特征，可在配置调整筛选阈值后重跑。
- 针对约 28.8% 的逾期率，可在训练流程中探索阈值调节或采样策略。
- 若新增外部特征或调整标签逻辑，请同步更新配置并记录版本，以保持流水线可追溯。