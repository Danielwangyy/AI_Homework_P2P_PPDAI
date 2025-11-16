"""数据准备流水线入口。

该文件负责将“原始数据”一步步加工为“建模数据”，并产出中间文件。
整体流程适合初学者理解：读取 → 清洗 → 特征构造 → 划分数据集 → 标准化 → 保存结果。
建议搭配日志输出与配置文件一起阅读，能快速掌握数据流向。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..data.cleaning import clean_lc, clean_lp, clean_lcis, generate_anomaly_statistics
from ..data.labeling import generate_labels
from ..data.loading import load_raw_data
from ..features.engineering import build_feature_dataframe
from ..utils.config import load_yaml
from ..utils.logger import get_logger, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """确保待训练的特征列存在。

    如果缺失某个字段，就补上一列默认值 0.0，避免后续模型训练报错。
    """
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0


def _resolve_path(path_str: str | None) -> Path | None:
    """将配置中的路径统一转换为绝对路径。

    返回 None 表示配置项为空或没有提供。
    """
    if not path_str:
        return None
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


def _export_feature_analysis(
    feature_df: pd.DataFrame,
    numeric_cols: Iterable[str],
    label_col: str,
    analysis_cfg: dict,
) -> None:
    """根据配置导出特征统计与高相关性特征对。

    该步骤在初始建模时很有帮助，可快速了解特征质量与与标签的关联性。
    配置项为空时自动跳过，减少对主流程的干扰。
    """
    summary_path = _resolve_path(analysis_cfg.get("summary_path"))
    high_corr_path = _resolve_path(analysis_cfg.get("high_corr_path"))

    if not summary_path:
        return

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    available_numeric = [col for col in numeric_cols if col in feature_df.columns]
    if not available_numeric:
        return

    summary_records = []
    numeric_subset = feature_df[available_numeric]

    for col in available_numeric:
        series = numeric_subset[col]
        summary_records.append(
            {
                "feature": col,
                "missing_rate": float(series.isna().mean()),
                "mean": float(series.mean()),
                "std": float(series.std()),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    if label_col in feature_df.columns:
        corr_df = pd.concat(
            [numeric_subset, feature_df[[label_col]]],
            axis=1,
        )
        correlation = corr_df.corr(numeric_only=True)[label_col].drop(label_col, errors="ignore")
        summary_df["abs_corr_with_label"] = summary_df["feature"].map(
            lambda col: float(abs(correlation.get(col, float("nan"))))
        )
    else:
        summary_df["abs_corr_with_label"] = float("nan")

    summary_df = summary_df.sort_values("abs_corr_with_label", ascending=False, na_position="last")
    summary_df.to_csv(summary_path, index=False)

    if high_corr_path:
        threshold = float(analysis_cfg.get("high_corr_threshold", 0.85))
        high_corr_path.parent.mkdir(parents=True, exist_ok=True)
        corr_matrix = numeric_subset.corr(numeric_only=True).abs()
        records = []
        for i, col in enumerate(available_numeric):
            for j in range(i + 1, len(available_numeric)):
                other = available_numeric[j]
                value = corr_matrix.loc[col, other]
                if pd.notna(value) and value >= threshold:
                    records.append(
                        {
                            "feature_a": col,
                            "feature_b": other,
                            "abs_corr": float(value),
                        }
                    )
        pd.DataFrame(records, columns=["feature_a", "feature_b", "abs_corr"]).to_csv(high_corr_path, index=False)


def _stratified_split(
    X: pd.DataFrame,
    target: pd.Series,
    ids: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """执行分层抽样切分。

    返回训练集、验证集、测试集以及对应的标签与 ID，
    保证每个集合中标签分布一致。
    """
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X,
        target,
        ids,
        test_size=test_size + val_size,
        stratify=target,
        random_state=random_state,
    )

    relative_test_size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp,
        y_temp,
        ids_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test


def run_pipeline(config_path: Path) -> None:
    """主函数：根据配置执行数据准备流程。

    为方便理解，按顺序概括如下：

    1. 读取配置，定位各类输入/输出目录；
    2. 调用 `setup_logger` 初始化日志文件；
    3. 加载原始数据集，并生成标签；
    4. 清洗数据、拼接三张表，构造特征；
    5. 可选导出特征统计报告；
    6. 根据配置决定是“时间切分”还是“随机分层切分”；
       - 时间切分失败（某个集合为空）时，会自动退回随机切分；
    7. 对数值特征做标准化，并保存标准化器（后续推理可复用）；
    8. 将训练/验证/测试数据分别保存至 `data/processed/`；
    9. 结合日志观察执行情况，方便排查问题。
    """
    cfg = load_yaml(config_path)
    raw_dir = _resolve_path(cfg.get("raw_data_dir"))
    interim_dir = _resolve_path(cfg.get("interim_dir"))
    processed_dir = _resolve_path(cfg.get("processed_dir"))
    outputs_root = PROJECT_ROOT / "outputs"
    artifacts_dir = _resolve_path(cfg.get("artifacts_dir")) or (outputs_root / "artifacts")
    log_dir = _resolve_path(cfg.get("logs_dir")) or (outputs_root / "logs")
    scaler_output_path = _resolve_path(cfg.get("scaler_output_path")) or (artifacts_dir / "numeric_scaler.pkl")

    if raw_dir is None or interim_dir is None or processed_dir is None:
        raise ValueError("配置文件缺少必要的目录字段：raw_data_dir、interim_dir、processed_dir")

    id_col = cfg.get("id_column", "ListingId")
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(log_dir, name="data_preparation")
    logger = get_logger()
    logger.info("开始数据准备流水线")

    # 创建 middata 目录
    middata_dir = outputs_root / "middata"
    middata_dir.mkdir(parents=True, exist_ok=True)
    logger.info("中间数据目录已创建: {}".format(middata_dir))

    data = load_raw_data(raw_dir)
    
    # 生成异常值统计表（基于原始数据）
    logger.info("开始生成异常值统计表")
    anomaly_stats = generate_anomaly_statistics(data["lc"], data["lp"], data["lcis"], cfg)
    anomaly_stats_path = middata_dir / "anomaly_statistics.csv"
    anomaly_stats.to_csv(anomaly_stats_path, index=False, encoding='utf-8-sig')
    logger.info("异常值统计表已保存: {} ({} 条记录)".format(anomaly_stats_path, len(anomaly_stats)))

    # 清洗三张表（根据规范严格清洗）
    logger.info("开始清洗数据表")
    cleaned_lc = clean_lc(data["lc"], cfg)
    logger.info("LC 表清洗完成 (添加了新特征: 正常还款比)")
    
    cleaned_lp = clean_lp(data["lp"])
    logger.info("LP 表清洗完成 (还款状态已重新赋值)")
    
    cleaned_lcis = clean_lcis(data["lcis"])
    logger.info("LCIS 表清洗完成 (已处理异常值和无效状态)")

    # 使用清洗后的数据生成标签
    logger.info("开始生成标签（使用清洗后的数据）")
    cleaned_data = {
        "lc": cleaned_lc,
        "lp": cleaned_lp,
        "lcis": cleaned_lcis
    }
    labels = generate_labels(cleaned_data, id_col=id_col)
    
    # 统计标签分布
    valid_labels = labels[labels["is_valid"] == True]
    invalid_labels = labels[labels["is_valid"] == False]
    logger.info("标签生成完成：总记录数={}，有效标签={}，无效标签={}".format(
        len(labels), len(valid_labels), len(invalid_labels)
    ))
    if len(valid_labels) > 0:
        label_counts = valid_labels["label"].value_counts()
        logger.info("有效标签分布：")
        for label_val, count in label_counts.items():
            label_name = "已还清（正常）" if label_val == 0 else "逾期中（违约）"
            logger.info("  - {} (标签={}): {} 条".format(label_name, label_val, count))

    # 保存清洗后的三张CSV表
    logger.info("开始保存清洗后的CSV表")
    
    # LC表：保存清洗后的数据
    cleaned_lc_path = middata_dir / "LC_cleaned.csv"
    lc_to_save = cleaned_lc.copy()
    # 将日期字段转换为字符串格式以便在CSV中可读（处理NaN值）
    if "借款成功日期" in lc_to_save.columns:
        if pd.api.types.is_datetime64_any_dtype(lc_to_save["借款成功日期"]):
            lc_to_save["借款成功日期"] = lc_to_save["借款成功日期"].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
    lc_to_save.to_csv(cleaned_lc_path, index=False, encoding='utf-8-sig')
    logger.info("清洗后的LC表已保存: {} ({} 条记录, {} 个字段)".format(
        cleaned_lc_path, len(cleaned_lc), len(cleaned_lc.columns)
    ))

    # LP表：保存清洗后的数据
    lp_cleaned_path = middata_dir / "LP_cleaned.csv"
    lp_to_save = cleaned_lp.copy()
    # 将日期字段转换为字符串格式（处理NaN值）
    date_cols_lp = ["到期日期", "还款日期", "recorddate"]
    for col in date_cols_lp:
        if col in lp_to_save.columns and pd.api.types.is_datetime64_any_dtype(lp_to_save[col]):
            lp_to_save[col] = lp_to_save[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
    lp_to_save.to_csv(lp_cleaned_path, index=False, encoding='utf-8-sig')
    logger.info("清洗后的LP表已保存: {} ({} 条记录, {} 个字段)".format(
        lp_cleaned_path, len(cleaned_lp), len(cleaned_lp.columns)
    ))

    # LCIS表：保存清洗后的数据
    lcis_cleaned_path = middata_dir / "LCIS_cleaned.csv"
    lcis_to_save = cleaned_lcis.copy()
    # 将日期字段转换为字符串格式（处理NaN值）
    date_cols_lcis = ["借款成功日期", "上次还款日期", "下次计划还款日期", "recorddate"]
    for col in date_cols_lcis:
        if col in lcis_to_save.columns and pd.api.types.is_datetime64_any_dtype(lcis_to_save[col]):
            lcis_to_save[col] = lcis_to_save[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
    lcis_to_save.to_csv(lcis_cleaned_path, index=False, encoding='utf-8-sig')
    logger.info("清洗后的LCIS表已保存: {} ({} 条记录, {} 个字段)".format(
        lcis_cleaned_path, len(cleaned_lcis), len(cleaned_lcis.columns)
    ))

    feature_df = build_feature_dataframe(cleaned_lc, cleaned_lp, labels, cfg, id_col)
    
    # 只保留有效标签的记录（排除"正常还款中"等未产生结果的记录）
    if "is_valid" in feature_df.columns:
        feature_df_valid = feature_df[feature_df["is_valid"] == True].copy()
        logger.info("过滤无效标签后：{} 条有效记录（原始记录：{} 条）".format(
            len(feature_df_valid), len(feature_df)
        ))
        feature_df = feature_df_valid
    
    # 确保标签列存在且有效
    if "label" not in feature_df.columns:
        raise ValueError("特征表中缺少标签列")
    
    # 检查是否有有效标签
    valid_label_mask = feature_df["label"].notna()
    if valid_label_mask.sum() == 0:
        raise ValueError("没有有效的标签记录，无法进行模型训练")
    
    feature_df = feature_df[valid_label_mask].copy()
    logger.info("最终用于建模的记录数：{} 条".format(len(feature_df)))
    
    analysis_cfg = cfg.get("feature_analysis", {})
    if analysis_cfg:
        _export_feature_analysis(feature_df, cfg.get("numeric_features", []), cfg.get("label_column", "label"), analysis_cfg)
    interim_path = interim_dir / "loan_master.parquet"
    feature_df.to_parquet(interim_path, index=False)
    logger.info("中间数据写入 {}".format(interim_path))

    categorical_cols = cfg.get("categorical_features", [])
    binary_cols = cfg.get("binary_features", [])
    numeric_cols = cfg.get("numeric_features", [])
    extra_numeric_cols = cfg.get("lp_numeric_features", [])
    if extra_numeric_cols:
        numeric_cols = list(dict.fromkeys(list(numeric_cols) + list(extra_numeric_cols)))

    _ensure_columns(feature_df, binary_cols + numeric_cols)

    # 检查binary_cols中是否有object类型的列，如果有，移到categorical_cols中进行one-hot编码
    binary_cols_to_convert = []
    binary_cols_numeric = []
    for col in binary_cols:
        if col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                # object类型的binary列应该进行one-hot编码
                binary_cols_to_convert.append(col)
                if col not in categorical_cols:
                    categorical_cols.append(col)
                    logger.info("将object类型的binary特征 '{}' 移到categorical特征中进行one-hot编码".format(col))
            else:
                # 数值类型的binary列保持不变
                binary_cols_numeric.append(col)
        else:
            binary_cols_numeric.append(col)
    
    # 更新binary_cols，只保留数值类型的列
    binary_cols = binary_cols_numeric

    # 对categorical_cols进行one-hot编码
    dummy_df = pd.get_dummies(
        feature_df[categorical_cols],
        columns=categorical_cols,
        prefix=categorical_cols,
        dtype=float,
    )
    
    # 确保所有binary_cols都是数值类型
    binary_features_df = feature_df[binary_cols].copy() if binary_cols else pd.DataFrame(index=feature_df.index)
    for col in binary_cols:
        if col in binary_features_df.columns:
            # 确保是float类型
            binary_features_df[col] = pd.to_numeric(binary_features_df[col], errors='coerce').fillna(0.0).astype(float)
    
    # 将原始 ID + 数值/二值特征 + One-Hot 结果拼接在一起
    features = pd.concat(
        [feature_df[[id_col]], feature_df[numeric_cols], binary_features_df, dummy_df],
        axis=1,
    )
    features = features.loc[:, ~features.columns.duplicated()]
    target = feature_df["label"].astype(int)

    X = features.drop(columns=[id_col])
    ids = feature_df[id_col]
    
    # 最终检查：确保所有特征都是数值类型
    object_cols = [col for col in X.columns if X[col].dtype == 'object']
    if object_cols:
        logger.warning("发现object类型的特征列，将尝试转换为数值类型: {}".format(object_cols))
        for col in object_cols:
            # 尝试转换为数值，如果失败则设为0
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)

    split_strategy = str(cfg.get("split_strategy", "random")).lower()
    if split_strategy == "time":
        logger.info("使用时序切分策略")
        # 读取切分边界
        train_end_str = cfg.get("train_end_date", "2016-12-31")
        val_end_str = cfg.get("val_end_date", "2017-06-30")
        train_end = pd.to_datetime(train_end_str)
        val_end = pd.to_datetime(val_end_str)

        # 基于借款成功日期进行切分
        dates = pd.to_datetime(feature_df["借款成功日期"], errors="coerce")
        unknown_mask = dates.isna()
        train_mask = (dates <= train_end) | unknown_mask
        val_mask = (~unknown_mask) & (dates > train_end) & (dates <= val_end)
        test_mask = (~unknown_mask) & (dates > val_end)

        X_train, y_train, ids_train = X[train_mask], target[train_mask], ids[train_mask]
        X_val, y_val, ids_val = X[val_mask], target[val_mask], ids[val_mask]
        X_test, y_test, ids_test = X[test_mask], target[test_mask], ids[test_mask]

        if min(len(X_train), len(X_val), len(X_test)) == 0:
            logger.warning(
                "时序切分结果存在空集（train={}, val={}, test={}），回退到随机分层切分策略",
                len(X_train),
                len(X_val),
                len(X_test),
            )
            test_size = float(cfg.get("test_size", 0.15))
            val_size = float(cfg.get("val_size", 0.15))
            random_state = int(cfg.get("random_state", 42))
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                ids_train,
                ids_val,
                ids_test,
            ) = _stratified_split(X, target, ids, test_size, val_size, random_state)
    else:
        logger.info("使用随机切分策略")
        test_size = float(cfg.get("test_size", 0.15))
        val_size = float(cfg.get("val_size", 0.15))
        random_state = int(cfg.get("random_state", 42))

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            ids_train,
            ids_val,
            ids_test,
        ) = _stratified_split(X, target, ids, test_size, val_size, random_state)

    scaler = StandardScaler()
    numeric_in_X = [col for col in numeric_cols if col in X_train.columns]
    # 仅对存在且确认为数值的列执行标准化
    for dataset in (X_train, X_val, X_test):
        for col in numeric_in_X:
            dataset[col] = dataset[col].astype(float)
    if len(numeric_in_X) > 0 and len(X_train) > 0:
        X_train.loc[:, numeric_in_X] = scaler.fit_transform(X_train[numeric_in_X])
        if len(X_val) > 0:
            X_val.loc[:, numeric_in_X] = scaler.transform(X_val[numeric_in_X])
        if len(X_test) > 0:
            X_test.loc[:, numeric_in_X] = scaler.transform(X_test[numeric_in_X])

    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "numeric_features": numeric_in_X}, scaler_output_path)
    logger.info("数值特征标准化器已保存: {}", scaler_output_path)

    def assemble(ids_series, X_df, y_series):
        return pd.concat(
            [ids_series.reset_index(drop=True), X_df.reset_index(drop=True)], axis=1
        ).assign(label=y_series.reset_index(drop=True))

    train_df = assemble(ids_train, X_train, y_train)
    val_df = assemble(ids_val, X_val, y_val)
    test_df = assemble(ids_test, X_test, y_test)

    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "valid.parquet"
    test_path = processed_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.info("数据集已生成：train {}，valid {}，test {}".format(len(train_df), len(val_df), len(test_df)))
    logger.info("数据准备流水线完成")


def run_cleaning_only(config_path: Path) -> None:
    """仅执行数据清洗，输出中间表，不进行后续的特征工程和模型训练。
    
    该函数会：
    1. 加载原始数据
    2. 生成异常值统计表
    3. 清洗LC表
    4. 保存清洗后的三张CSV表到 outputs/middata/
    """
    cfg = load_yaml(config_path)
    raw_dir = _resolve_path(cfg.get("raw_data_dir"))
    outputs_root = PROJECT_ROOT / "outputs"
    log_dir = _resolve_path(cfg.get("logs_dir")) or (outputs_root / "logs")

    if raw_dir is None:
        raise ValueError("配置文件缺少必要的目录字段：raw_data_dir")

    id_col = cfg.get("id_column", "ListingId")

    setup_logger(log_dir, name="data_preparation")
    logger = get_logger()
    logger.info("开始数据清洗流程（仅清洗，不进行特征工程）")

    # 创建 middata 目录
    middata_dir = outputs_root / "middata"
    middata_dir.mkdir(parents=True, exist_ok=True)
    logger.info("中间数据目录已创建: {}".format(middata_dir))

    # 加载原始数据
    data = load_raw_data(raw_dir)
    logger.info("原始数据加载完成: LC={}条, LP={}条, LCIS={}条".format(
        len(data["lc"]), len(data["lp"]), len(data["lcis"])
    ))

    # 生成异常值统计表（基于原始数据）
    logger.info("开始生成异常值统计表")
    anomaly_stats = generate_anomaly_statistics(data["lc"], data["lp"], data["lcis"], cfg)
    anomaly_stats_path = middata_dir / "anomaly_statistics.csv"
    
    # 如果文件存在且被占用，先尝试删除或使用临时文件名
    if anomaly_stats_path.exists():
        try:
            anomaly_stats_path.unlink()
            logger.info("已删除旧的异常值统计表文件")
        except PermissionError:
            logger.warning("异常值统计表文件被占用，尝试使用带时间戳的文件名")
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            anomaly_stats_path = middata_dir / f"anomaly_statistics_{timestamp}.csv"
    
    try:
        anomaly_stats.to_csv(anomaly_stats_path, index=False, encoding='utf-8-sig')
        logger.info("异常值统计表已保存: {} ({} 条记录)".format(anomaly_stats_path, len(anomaly_stats)))
    except PermissionError as e:
        logger.error("无法保存异常值统计表，文件可能被占用: {}".format(e))
        logger.info("请关闭Excel或其他打开该文件的程序后再试")

    # 清洗三张表（根据规范严格清洗）
    logger.info("开始清洗数据表")
    cleaned_lc = clean_lc(data["lc"], cfg)
    logger.info("LC 表清洗完成 (添加了新特征: 正常还款比)")
    
    cleaned_lp = clean_lp(data["lp"])
    logger.info("LP 表清洗完成 (还款状态已重新赋值)")
    
    cleaned_lcis = clean_lcis(data["lcis"])
    logger.info("LCIS 表清洗完成 (已处理异常值和无效状态)")

    # 保存清洗后的三张CSV表
    logger.info("开始保存清洗后的CSV表")
    
    # LC表：保存清洗后的数据
    cleaned_lc_path = middata_dir / "LC_cleaned.csv"
    lc_to_save = cleaned_lc.copy()
    # 将日期字段转换为字符串格式以便在CSV中可读（处理NaN值）
    if "借款成功日期" in lc_to_save.columns:
        if pd.api.types.is_datetime64_any_dtype(lc_to_save["借款成功日期"]):
            lc_to_save["借款成功日期"] = lc_to_save["借款成功日期"].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
    try:
        lc_to_save.to_csv(cleaned_lc_path, index=False, encoding='utf-8-sig')
        logger.info("清洗后的LC表已保存: {} ({} 条记录, {} 个字段)".format(
            cleaned_lc_path, len(cleaned_lc), len(cleaned_lc.columns)
        ))
    except PermissionError as e:
        logger.error("无法保存LC表，文件可能被占用: {}".format(e))

    # LP表：保存清洗后的数据
    lp_cleaned_path = middata_dir / "LP_cleaned.csv"
    lp_to_save = cleaned_lp.copy()
    # 将日期字段转换为字符串格式（处理NaN值）
    date_cols_lp = ["到期日期", "还款日期", "recorddate"]
    for col in date_cols_lp:
        if col in lp_to_save.columns and pd.api.types.is_datetime64_any_dtype(lp_to_save[col]):
            lp_to_save[col] = lp_to_save[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
    try:
        lp_to_save.to_csv(lp_cleaned_path, index=False, encoding='utf-8-sig')
        logger.info("清洗后的LP表已保存: {} ({} 条记录, {} 个字段)".format(
            lp_cleaned_path, len(cleaned_lp), len(cleaned_lp.columns)
        ))
    except PermissionError as e:
        logger.error("无法保存LP表，文件可能被占用: {}".format(e))

    # LCIS表：保存清洗后的数据
    lcis_cleaned_path = middata_dir / "LCIS_cleaned.csv"
    lcis_to_save = cleaned_lcis.copy()
    # 将日期字段转换为字符串格式（处理NaN值）
    date_cols_lcis = ["借款成功日期", "上次还款日期", "下次计划还款日期", "recorddate"]
    for col in date_cols_lcis:
        if col in lcis_to_save.columns and pd.api.types.is_datetime64_any_dtype(lcis_to_save[col]):
            lcis_to_save[col] = lcis_to_save[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
    try:
        lcis_to_save.to_csv(lcis_cleaned_path, index=False, encoding='utf-8-sig')
        logger.info("清洗后的LCIS表已保存: {} ({} 条记录, {} 个字段)".format(
            lcis_cleaned_path, len(cleaned_lcis), len(cleaned_lcis.columns)
        ))
    except PermissionError as e:
        logger.error("无法保存LCIS表，文件可能被占用: {}".format(e))

    logger.info("数据清洗流程完成！中间表已保存至: {}".format(middata_dir))
    logger.info("生成的文件：")
    logger.info("  - 异常值统计表: {}".format(anomaly_stats_path))
    logger.info("  - 清洗后的LC表: {}".format(cleaned_lc_path))
    logger.info("  - 清洗后的LP表: {}".format(lp_cleaned_path))
    logger.info("  - 清洗后的LCIS表: {}".format(lcis_cleaned_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PPDAI dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "data_processing.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--cleaning-only",
        action="store_true",
        help="仅执行数据清洗，输出中间表，不进行后续的特征工程和模型训练",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cleaning_only:
        run_cleaning_only(Path(args.config))
    else:
        run_pipeline(Path(args.config))

