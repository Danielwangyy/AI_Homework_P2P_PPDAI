## 函数级流程拆解指南

> 本文与 `docs/beginner_step_by_step.md` 搭配阅读：前者告诉你“做什么”，本文聚焦“每一步背后具体调用了哪些函数、解决了哪些子问题”。阅读时，可在 Cursor 中按住 `⌘` + 点击函数名快速跳转源代码。

---

### 1. 环境准备：`scripts/setup.sh`

**子问题域：定位 Python 解释器与虚拟环境**
- **考虑因素**：不同操作系统的 Python 路径不一致；需要保证所有后续命令都在 `.venv` 中执行，确保依赖版本一致。
- **技术选择**：脚本依次检测 `$PYTHON`、`.venv/bin/python`、`python3`、`python`，并在缺失 `.venv` 时自动创建；随后激活虚拟环境。
- **完成工作**：创建并激活 `.venv`，后续命令统一使用虚拟环境中的解释器，避免“系统 Python 污染”。

```9:31:/Users/admin/Documents/AI_Homework_P2P_PPDAI/scripts/setup.sh
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] 未发现 .venv，正在创建虚拟环境..."
  "$BASE_PYTHON" -m venv "$VENV_DIR"
fi
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  source "$VENV_DIR/Scripts/activate"
else
  echo "[ERROR] 找不到虚拟环境的激活脚本，请删除 .venv 后重试。" >&2
  exit 1
fi
```

**子问题域：依赖安装与 CLI 验证**
- **考虑因素**：一次性安装 requirements 与项目本身；自动检查 CLI 是否能运行，第一时间暴露环境问题。
- **技术选择**：使用虚拟环境内的 Python 执行 `pip install -r environments/requirements.txt` 与 `pip install -e .`。最后运行 `python -m ai_homework.cli.run_pipeline --help` 验证。
- **完成工作**：打印 CLI 帮助，如果失败会输出详细日志，便于新手排查。

---

### 2. 原始数据就位：配置 + 数据读取

**子问题域：明确需要放置的数据目录**
- **考虑因素**：保证数据准备流水线能找到原始 CSV；同时允许自定义路径。
- **技术选择**：在 `configs/data_processing.yaml` 中约定 `raw_data_dir: "data/raw/source_data"`，并在代码中统一通过 `_resolve_path` 转成绝对路径。
- **完成工作**：只要按配置放置 `LC.csv`、`LP.csv`、`LCIS.csv`，流水线即可正确加载。

**子问题域：解析 CSV 并处理日期列**
- **考虑因素**：三张表的日期列不同，需统一转换为 `datetime`，避免后续比较日期时报错。
- **技术选择**：`load_raw_data` 针对每张表定义日期列清单并调用 `_load_csv`。
- **完成工作**：返回包含三份 DataFrame 的字典，供后续流程使用。

```23:40:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/data/loading.py
def load_raw_data(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    data = {
        "lc": _load_csv(raw_dir / "LC.csv", DATE_COLUMNS_LC),
        "lp": _load_csv(raw_dir / "LP.csv", DATE_COLUMNS_LP),
        "lcis": _load_csv(raw_dir / "LCIS.csv", DATE_COLUMNS_LCIS),
    }
    return data
```

---

### 3. 生成干净数据：`ai_homework.pipelines.prepare_data.run_pipeline`

**子问题域：读取配置并初始化日志**
- **考虑因素**：多平台路径差异，需保证配置里写相对路径也能被正确解析；同时写日志方便定位问题。
- **技术选择**：`load_yaml` 读取配置；`_resolve_path` 将路径转换为绝对路径；`setup_logger` 创建 `outputs/logs/data_preparation.log`。
- **完成工作**：确保每一次运行都能复现相同的目录结构与日志输出。

```159:207:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/pipelines/prepare_data.py
cfg = load_yaml(config_path)
raw_dir = _resolve_path(cfg.get("raw_data_dir"))
...
setup_logger(log_dir, name="data_preparation")
logger = get_logger()
logger.info("开始数据准备流水线")
```

**子问题域：生成违约标签**
- **考虑因素**：LCIS 表提供借款状态，LP 表提供逐期还款记录，两者逻辑存在差异；优先使用更细粒度的 LP 信息。
- **技术选择**：`generate_labels` 分别调用 `_label_from_lcis`、`_label_from_lp`，再合并。LP 标签缺失时回退到 LCIS，且始终对 `ListingId` 求并集，避免丢样本。
- **完成工作**：得到带有 `label`、`lp_label`、`lcis_label` 的标签表，为后续特征拼接做准备。

```53:78:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/data/labeling.py
labels = pd.DataFrame({id_col: all_ids})
labels = labels.merge(lp_labels.rename("lp_label"), left_on=id_col, right_index=True, how="left")
labels = labels.merge(lcis_labels.rename("lcis_label"), left_on=id_col, right_index=True, how="left")
labels["label"] = labels["lp_label"].fillna(labels["lcis_label"])
labels["label"] = labels["label"].fillna(0).astype(int)
```

**子问题域：清洗 LC 主表**
- **考虑因素**：年龄异常值、认证字段的字符串标识、类别缺失值等都会影响模型稳定性。
- **技术选择**：`clean_lc` 对年龄设定合法区间、将二值字段映射成 0/1、统一类别缺失值为“未知”、使用中位数填补数值缺失。
- **完成工作**：为特征工程提供整洁一致的基础数据。

```50:83:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/data/cleaning.py
if "年龄" in cleaned.columns:
    cleaned.loc[~cleaned["年龄"].between(18, 70), "年龄"] = np.nan
...
for col in cfg.get("categorical_features", []):
    if col in cleaned.columns:
        cleaned[col] = cleaned[col].fillna("未知")
```

**子问题域：特征构造与合并**
- **考虑因素**：需要兼顾业务含义（历史逾期率、借款杠杆比）与模型可读性（One-Hot 编码）。还要把 LP 明细汇总成借款级特征。
- **技术选择**：`build_feature_dataframe` 依次调用 `add_time_features`、`add_derived_features`、`compute_lp_features`；将 LP 聚合结果按 ID 左连接，填补缺失。
- **完成工作**：得到包含标签与所有候选特征的主表，并写入 `data/interim/loan_master.parquet` 方便排查。

```109:148:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/features/engineering.py
df = add_time_features(cleaned_lc)
df = add_derived_features(df)
df = df.merge(labels[[id_col, "label"]], on=id_col, how="left")
lp_features = compute_lp_features(lp, id_col)
if not lp_features.empty:
    df = df.merge(lp_features, on=id_col, how="left")
```

**子问题域：数据集划分与标准化**
- **考虑因素**：配置默认为时间切分，以模拟真实上线场景；若时间切分导致某集合为空，则回退到分层随机切分。同时仅对存在的数值列做标准化，保留 One-Hot 列。
- **技术选择**：`_stratified_split` 复用 `train_test_split` 进行两段式分层；`StandardScaler` 只拟合数据集中存在的数值列，并保存到 `outputs/artifacts/numeric_scaler.pkl`。
- **完成工作**：输出 `train.parquet`、`valid.parquet`、`test.parquet`，并确保数值特征处于同一尺度。

```237:311:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/pipelines/prepare_data.py
if split_strategy == "time":
    ...
    if min(len(X_train), len(X_val), len(X_test)) == 0:
        logger.warning("时序切分结果存在空集...回退到随机分层切分策略")
        ...
else:
    ...
scaler = StandardScaler()
numeric_in_X = [col for col in numeric_cols if col in X_train.columns]
...
joblib.dump({"scaler": scaler, "numeric_features": numeric_in_X}, scaler_output_path)
```

---

### 4. 训练模型：`ai_homework.pipelines.train_models.run_pipeline`

**子问题域：加载数据与基础配置**
- **考虑因素**：需要和数据准备阶段的输出保持路径一致；任何路径差异都应通过配置调整。
- **技术选择**：`load_yaml` 解析 `configs/model_training.yaml`，将输出目录、日志目录转成绝对路径；`load_dataset` 负责拆分特征、标签、ID。
- **完成工作**：日志初始化后输出训练、验证、测试样本量，帮助确认数据是否齐全。

```275:288:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/pipelines/train_models.py
train_path = _resolve_path(data_cfg["train_path"])
...
X_train, y_train, _ = load_dataset(train_path, id_col, label_col)
X_valid, y_valid, _ = load_dataset(valid_path, id_col, label_col)
X_test, y_test, test_ids = load_dataset(test_path, id_col, label_col)
```

**子问题域：应对类别不平衡**
- **考虑因素**：逾期率约 28.8%，若直接训练会导致模型偏向预测“未逾期”；同时成本敏感权重要求 FN 代价更高。
- **技术选择**：根据正负样本数量和阈值调优的成本权重动态计算 `scale_pos_weight`（XGBoost/LightGBM）或 `class_weights`（CatBoost）。
- **完成工作**：训练阶段自动调整类别权重，提高召回率。

```312:329:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/pipelines/train_models.py
pos_count = float(np.sum(y_train == 1))
neg_count = float(np.sum(y_train == 0))
if pos_count > 0:
    cost_ratio = float(threshold_cfg.get("fn_cost", 5.0)) / ...
    scale_weight = max((neg_count / pos_count) * cost_ratio, 1.0)
    if model_name in {"xgboost", "lightgbm"}:
        params.setdefault("scale_pos_weight", scale_weight)
```

**子问题域：交叉验证选择最佳参数**
- **考虑因素**：需要一个统一入口管理不同模型的调参与训练。
- **技术选择**：`train_model` 根据 `type` 调用 `train_logistic_regression`、`train_xgboost` 等；核心 `_train_with_cv` 使用 `StratifiedKFold` 和 `ParameterGrid` 遍历参数。
- **完成工作**：返回包含 `best_params`、`validation_score`、`history` 的 `ModelResult`，并保存调参过程到 CSV。

```216:271:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/models/training.py
def train_model(model_name: str, ...):
    model_type = config.get("type")
    ...
    if model_type == "xgboost":
        return train_xgboost(...)
    ...
    raise ValueError(f"不支持的模型类型: {model_type}")
```

**子问题域：阈值调优与业务成本平衡**
- **考虑因素**：误判坏客户（FN）的成本远高于误判好客户（FP），需要自定义阈值而不是固定 0.5。
- **技术选择**：`_tune_threshold` 遍历网格，按 `fn_cost:fp_cost = 5:1` 计算成本或 F-β；返回最佳阈值、指标以及搜索记录。
- **完成工作**：模型在验证集上获得更符合业务目标的标签划分，结果写入 `threshold_summary.csv`。

```91:157:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/pipelines/train_models.py
best_threshold, threshold_metrics, threshold_history = _tune_threshold(
    y_valid,
    valid_proba,
    threshold_cfg,
)
...
valid_metrics = compute_classification_metrics(y_valid, valid_preds, valid_proba, cost_weights=cost_weights)
```

**子问题域：评估、可视化与解释**
- **考虑因素**：需要统一输出指标表、混淆矩阵、ROC、SHAP，便于写报告与做答辩。
- **技术选择**：`compute_classification_metrics` 计算 Accuracy/Precision/Recall/F1/ROC-AUC；`plot_confusion_matrix`、`plot_roc_curve` 生成图像；`_export_shap_values` 对树模型输出特征重要性图表与 CSV。
- **完成工作**：`outputs/reports/tables/model_metrics.csv` 汇总所有指标，`outputs/reports/figures/` 存放图表，`outputs/artifacts/` 保存模型与预测记录。

```387:501:/Users/admin/Documents/AI_Homework_P2P_PPDAI/src/ai_homework/pipelines/train_models.py
metrics_records.append(metrics_to_dataframe(train_metrics, model_name, "train"))
...
plt.savefig(fig_path, dpi=300)
...
threshold_df.to_csv(tables_dir / "threshold_summary.csv", index=False)
logger.info("模型训练流程结束")
```

---

### 5. 如何验证产物与文档一致？

- **日志对照**：`outputs/logs/data_preparation.log` 和 `outputs/logs/model_training.log` 会打印上述子问题域的关键节点。若日志缺少某一步（例如“标签生成完成”），说明该环节未成功执行。
- **表格交叉**：`outputs/reports/tables/model_metrics.csv` 中的指标应与 `docs/project_summary.md`、`docs/modeling_results.md` 的最新描述一致；若你重新训练模型，请同步更新这些文档。
- **脚本复现**：运行 `./scripts/run_pipeline.sh`（或加上 `--skip-data` / `--skip-train`）应按本文顺序依次调用各函数。如果中途失败，可根据本文定位到具体子问题域，结合源代码和配置排查。

---

### 6. 继续探索的建议

- 打开 `docs/data_pipeline_summary.md`，将日志中的产物路径与本文的函数步骤进行核对，加深对数据流的理解。
- 如果想修改特征或模型参数，先更新 `configs/*.yaml`，再对照本文查找对应函数，确认改动会影响的子问题域。
- 遇到陌生术语，利用 Cursor 让 Agent 解读具体函数（例如：“请解释 `_tune_threshold` 的作用”），逐步构建自己的知识笔记。

有了这份函数级拆解，你可以自信地追踪每一个“按钮”背后发生了什么，既能保持文档一致性，也能在调参或排障时快速找到正确的代码入口。祝你在机器学习之路上越走越顺！

