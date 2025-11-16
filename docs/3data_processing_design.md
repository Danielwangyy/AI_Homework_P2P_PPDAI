## 数据预处理与特征工程设计

### 设计目标与范围
- 支撑风险建模的数据基础设施，覆盖原始数据接入、样本构建、特征工程与数据集划分的全生命周期。
- 兼顾可复现性与可配置性，通过统一配置驱动不同批次的数据处理。
- 在保持业务可解释性的前提下，为上层模型提供结构化、标准化且无泄露风险的特征矩阵与标签。

### 流程架构概览
1. **数据接入层**  
   - 从 `data/raw/source_data` 读取 LC、LP、LCIS 三类原始表，完成编码、数据类型标准化，并在 `data/interim` 中生成快照备份。  
   - 通过 `load_raw_data()` 抽象读取逻辑，屏蔽不同来源差异。
2. **清洗与一致性校验层**  
   - 认证、状态等枚举字段统一映射；金额、年龄等连续字段执行截尾与越界处理。  
   - `generate_anomaly_statistics()` 产出质量报告；`validate_consistency()` 比对借款金额与还款本金、借款期限与应还期数，异常写入 `outputs/middata`。
3. **样本与标签建模层**  
   - 结合 LP 还款轨迹判断贷款是否到期、是否违约，形成基础标签。  
   - 利用 LCIS 快照补充仍处于“逾期中”的合同，形成多来源标签并记录 `label_source`。  
   - 结果输出至 `outputs/middata/LC_labeled_samples.csv` 与 `LC_invalid_samples.csv`。
4. **特征构建层**  
   - `build_feature_dataframe()` 仅依赖 `LC_labeled_samples.csv` 中“申请时即可获取”的字段构造基础、统计、时间与交互特征。  
   - 通过集中维护的白名单/黑名单控制准入，剔除所有放款后行为字段并记录处理日志。
5. **数据集切分与落地层**  
   - `split_datasets()` 优先按借款成功日期执行时间切分，若任一集合为空则自动回退至 ListingId 分层的 70%/15%/15% 拆分。  
   - 输出 Parquet 文件、标准化器与特征清单等配套工件，供建模流水线直接引用。

### 核心处理策略
- **缺失与异常治理**：
  - 数值字段优先使用截尾 + 中位数/分位数填补，保留缺失指示列。  
  - 类别字段采用众数、`Unknown` 标签或独立分桶策略，确保不引入虚假信息。
- **时间与状态处理**：
  - LP 的统一 `recorddate` 作为观察截面；借款理论到期日基于期限计算，用于筛除未到期样本。  
  - 状态字段维持标准字典并记录异常枚举，便于回溯。
- **信息防泄露与特征准入**：
  - 在 `src/ai_homework/features/constants.py` 中集中维护可用字段白名单与泄露黑名单，覆盖标签、LP/LCIS 行为及其衍生特征。  
  - `build_feature_dataframe()` 会在特征构造前删除黑名单列、过滤白名单之外的字段，并将处理结果写入日志；若仍检测到黑名单项会直接抛错。  
  - `select_features()` 在自动筛选阶段再次校验黑名单，确保配置层面的疏漏不会流入模型训练。  
  - 仅保留 `loan_date` 作为元数据便于时间切分，最终训练矩阵不包含该字段。
- **特征体系规划**：
  - **基础特征**：金额、期限、利率、评级、认证、借款人画像等。  
  - **历史行为比率**：`history_total_loans`、`history_total_amount`、`history_repay_ratio`、`history_overdue_rate`、`history_avg_loan_amount`、`history_avg_term_payment` 等全部由样本字段按比率或均值构造。  
  - **结构衍生**：金额/历史成功金额比、总待还本金占比、评级×借款类型交互等均在样本内计算完成。  
  - **时间特征**：借款成功日期的年份、季度、月份、星期几等派生离散列，仅依赖可用的申请时间戳。  
  - **交互特征**：将评级数值化生成 `rating_numeric`，并派生 `loan_amount_rating_interaction`、`loan_term_rating_interaction`、`loan_amount_history_repay_ratio`、`loan_term_history_overdue_rate` 等组合指标。  
  - **审计留存字段**：`overdue_days_sum`、`lp_max_term`、`lp_last_due_date` 等放款后信息仅保留在样本导出中用于审核，不会进入建模矩阵；相关衍生计算（如 `loan_to_lp_term_ratio`、`loan_date_to_lp_last_repay_days` 等）全部列入黑名单。  
  - `label_source` 仅在样本导出中保留用于追溯标签来源，建模阶段完全移除，避免与目标变量产生直接映射。
  - 对仍无法从样本表推出的候选特征（如跨标的历史 DPD、近期贷款间隔等）保留“数据缺失”标记，待后续数据源扩展时补充。
  - 特征清单通过配置维护，可按版本号切换。
- **特征筛选与缩放**：
  - 支持皮尔逊相关阈值、卡方检验 Top-K、树模型重要性 Top-K 等组合策略。  
  - 数值特征使用 StandardScaler/MinMaxScaler；类别 One-Hot/目标编码输出统一矩阵与列顺序。

### 配置化与可复用性
- 所有规则集中在 `configs/data_processing.yaml`：包括清洗阈值、特征列表、特征选择策略、数据集拆分参数等；与之配套的白名单/黑名单定义统一维护在 `src/ai_homework/features/constants.py`。  
- 模块化函数在 `src/data/` 中实现，并在 `src/pipelines/prepare_data.py` 暴露 CLI，支持命令行参数覆盖默认配置。  
- 通过版本化目录（`outputs/artifacts/`、`outputs/reports/`）存放 scalers、特征重要性、筛选摘要等，便于追溯。

### 质量控制与测试设计
- 构建针对关键函数的单元测试（`tests/test_data_pipeline.py`），覆盖：
  - 标签生成边界场景（尚未到期、逾期天数缺失、LCIS 补充）。
  - 清洗后列类型、缺失率、枚举映射正确性。
  - 数据集拆分后无 `ListingId` 交叉污染，标签分布稳定。
- 对每次运行输出数据处理日志（结合 `loguru`），记录样本量、异常记录比例、特征筛选结果等核心指标。  
- 借助数据质量报告与中间产物，可在 Notebook 中进行抽样复核和可视化检查。

### 迭代规划
- 引入双侧 Winsorize、分布自适应变换（如 Box-Cox、QuantileTransformer）以提升特征稳健性。  
- 扩展外部数据接入（如宏观经济、征信补充），并通过配置开关逐步上线。  
- 针对 LCIS 补充样本开展漂移监控，评估其对模型的长期影响。  
- 预留自动化数据对账和质量告警（邮件/飞书）模块，支持生产环境的持续监控。
