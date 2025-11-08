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

from ..data.cleaning import clean_lc
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

    data = load_raw_data(raw_dir)
    labels = generate_labels(data, id_col=id_col)
    logger.info("标签生成完成：{} 条记录".format(labels.shape[0]))

    cleaned_lc = clean_lc(data["lc"], cfg)
    logger.info("LC 表清洗完成")

    feature_df = build_feature_dataframe(cleaned_lc, data["lp"], labels, cfg, id_col)
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

    dummy_df = pd.get_dummies(
        feature_df[categorical_cols],
        columns=categorical_cols,
        prefix=categorical_cols,
        dtype=float,
    )
    # 将原始 ID + 数值/二值特征 + One-Hot 结果拼接在一起
    features = pd.concat(
        [feature_df[[id_col]], feature_df[binary_cols + numeric_cols], dummy_df],
        axis=1,
    )
    features = features.loc[:, ~features.columns.duplicated()]
    target = feature_df["label"].astype(int)

    X = features.drop(columns=[id_col])
    ids = feature_df[id_col]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PPDAI dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "data_processing.yaml"),
        help="配置文件路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.config))

