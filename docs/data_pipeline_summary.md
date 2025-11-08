## 数据准备流水线执行摘要

- 启动方式：在 Agent 模式下说明“请执行 python -m ai_homework.pipelines.prepare_data --config configs/data_processing.yaml”
- 配置文件：`configs/data_processing.yaml`
- 运行时间：2025-11-03 01:21（UTC+8，本地日志时间）

### 输入数据
- 原始目录：`data/raw/source_data/`
- 涉及文件：`LC.csv`、`LP.csv`、`LCIS.csv`

### 处理步骤回顾
1. 读取原始数据并统一日期格式。
2. 按照 LCIS + LP 的联合规则生成违约标签。
3. 清洗 LC 表异常值（年龄异常置空、认证字段二值化等）。
4. 构造时间衍生特征与行为特征，并聚合 LP 期次统计（正常/逾期期数、按期还款率）。
5. 进行类别独热编码、数值特征标准化，完成分层抽样（70%/15%/15%）。

### 输出结果
- 中间数据：`data/interim/loan_master.parquet`
- 训练集：`data/processed/train.parquet`（229,987 条，label 分布与全量一致）
- 验证集：`data/processed/valid.parquet`（49,283 条）
- 测试集：`data/processed/test.parquet`（49,283 条）
- 数值特征标准化器：`artifacts/numeric_scaler.pkl`

### 后续工作建议
- 在模型训练阶段读取上述 Parquet 与标准化器，保证特征列顺序一致。
- 针对类别不平衡（逾期率约 28.8%）可在建模流程中尝试阈值调节或采样策略。
- 若需扩展特征，可在 `configs/data_processing.yaml` 更新特征清单并重跑流水线。

