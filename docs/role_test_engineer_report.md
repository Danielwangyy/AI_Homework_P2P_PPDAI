## 测试工程师工作报告

### 1. 测试目标
- 验证违约预测模型在训练、验证、测试集上的分类性能是否满足客户指标。
- 检查数据流水线输出的稳定性，确保无缺失列、特征分布异常或标签泄露。
- 输出图表与报告，为产品验收和后续优化提供依据。

### 2. 测试方案
- **功能测试**：
  - 运行 `src/pipelines/prepare_data.py` 及 `src/pipelines/train_models.py`，确认脚本可在虚拟环境中一键执行。
  - 校验输出文件完整性（Parquet、CSV、PNG、Joblib 等）。
- **模型测试**：
  - 主要监控指标：Accuracy、Precision、Recall、F1、ROC-AUC。
  - 利用验证集进行调参，最终在测试集评估模型泛化能力。
  - 绘制混淆矩阵与 ROC 曲线，辅助分析正负样本的识别情况。
- **数据一致性测试**：
  - 确认训练/验证/测试集的 `ListingId` 无交叉。
  - 随机抽样检查特征取值，确保数值型无 NaN、类别型无异常标签。

### 3. 关键结果
- 评估结果汇总在 `reports/tables/model_metrics.csv`：
  - 逻辑回归测试集 F1=0.7228，Recall=0.7781，满足基础要求。
  - XGBoost 测试集 F1=0.8209，Recall=0.7945，整体优于基线，并达到客户提出的 ROC-AUC ≥ 0.90 目标（0.9649）。
- 混淆矩阵与 ROC 曲线：
  - `reports/figures/confusion_matrix_logistic_regression.png`
  - `reports/figures/confusion_matrix_xgboost.png`
  - `reports/figures/roc_curve_logistic_regression.png`
  - `reports/figures/roc_curve_xgboost.png`
- 特征重要性验证：`reports/tables/feature_importance_xgboost.csv`，用于检视模型关注的业务要素。

### 4. 发现与建议
- XGBoost 存在少量字体警告（图表中文标签），已在可视化中改用英文标签以消除风险。
- 建议后续增加阈值敏感性测试，评估不同 Recall/Precision 平衡点对业务的影响。
- 可补充 Stress Test（重放缺失或异常数据）验证流水线鲁棒性，并加入单元测试保障模型预测接口的稳定性。

### 5. 交付物
- 测试脚本：同建模脚本共用流水线，详见 `src/pipelines/train_models.py`。
- 测试记录：日志 `logs/model_training.log`、指标表格 `reports/tables/model_metrics.*`、图表 `reports/figures/`。
- 建议事项及复现说明均记录于本报告，方便评审与复盘。

