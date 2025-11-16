"""数据准备流水线入口。

该文件负责将“原始数据”一步步加工为“建模数据”，并产出中间文件。
整体流程适合初学者理解：读取 → 清洗 → 特征构造 → 划分数据集 → 标准化 → 保存结果。
建议搭配日志输出与配置文件一起阅读，能快速掌握数据流向。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..data.cleaning import clean_lc, clean_lp, clean_lcis, generate_anomaly_statistics
from ..data.labeling import build_samples_with_labels, generate_labels
from ..data.loading import load_raw_data
from ..features.engineering import build_feature_dataframe
from ..features.selection import select_features
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


def _resolve_drop_columns(
    columns: Sequence[str],
    rules: Dict[str, Sequence[str]] | None,
    logger,
) -> Set[str]:
    drop: Set[str] = set()
    if not rules:
        return drop

    for name in rules.get("exact", []) or []:
        if name in columns:
            drop.add(name)

    for prefix in rules.get("prefixes", []) or []:
        drop.update(col for col in columns if col.startswith(prefix))

    for suffix in rules.get("suffixes", []) or []:
        drop.update(col for col in columns if col.endswith(suffix))

    for substring in rules.get("contains", []) or []:
        drop.update(col for col in columns if substring in col)

    for pattern in rules.get("regex", []) or []:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            if logger:
                logger.warning("特征消融：正则表达式 %s 无效，跳过。错误：%s", pattern, exc)
            continue
        drop.update(col for col in columns if compiled.search(col))

    return drop


def _apply_feature_drops(
    df: pd.DataFrame,
    rules: Dict[str, Sequence[str]] | None,
    logger,
    stage: str,
    *,
    protected_cols: Set[str] | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    """按配置规则移除特征列，用于快速执行消融实验。"""

    drop_candidates = _resolve_drop_columns(df.columns, rules, logger)
    if protected_cols:
        drop_candidates -= {col for col in protected_cols if col in drop_candidates}

    drop_existing = sorted(col for col in drop_candidates if col in df.columns)
    if not drop_existing:
        return df, []

    logger.info("特征消融[%s]：移除列 %s", stage, drop_existing)
    return df.drop(columns=drop_existing, errors="ignore"), drop_existing


def _check_consistency_between_lc_lp(
    lc_df: pd.DataFrame,
    lp_df: pd.DataFrame,
    id_col: str,
    *,
    principal_rel_tol: float = 0.05,
    principal_abs_tol: float = 1.0,
    term_tol: float = 1.0,
) -> tuple[dict, dict]:
    """校验借款金额/期限与 LP 汇总指标的一致性。

    返回:
        summary: 各项校验的摘要统计
        details: 具体不一致记录的 DataFrame 字典
    """
    summary: dict = {}
    details: dict = {}

    # 借款金额 vs LP 应还本金总和
    required_lc_cols = {id_col, "借款金额"}
    required_lp_cols = {id_col, "应还本金"}
    if required_lc_cols.issubset(lc_df.columns) and required_lp_cols.issubset(lp_df.columns):
        lc_amount = lc_df[list(required_lc_cols)].copy()
        lc_amount["借款金额"] = pd.to_numeric(lc_amount["借款金额"], errors="coerce")

        lp_principal = lp_df[list(required_lp_cols)].copy()
        lp_principal["应还本金"] = pd.to_numeric(lp_principal["应还本金"], errors="coerce")
        lp_grouped = (
            lp_principal.groupby(id_col, dropna=False)["应还本金"]
            .sum(min_count=1)
            .reset_index(name="lp应还本金合计")
        )

        merged_principal = lc_amount.merge(lp_grouped, on=id_col, how="inner")
        available_mask = merged_principal["借款金额"].notna() & merged_principal["lp应还本金合计"].notna()
        checked_principal = merged_principal[available_mask].copy()

        total_checked = len(checked_principal)
        if total_checked > 0:
            diff = (checked_principal["lp应还本金合计"] - checked_principal["借款金额"]).abs()
            denom = checked_principal["借款金额"].abs()
            relative_diff = pd.Series(np.nan, index=checked_principal.index, dtype=float)
            positive_denom = denom > 0
            relative_diff.loc[positive_denom] = diff.loc[positive_denom] / denom.loc[positive_denom]

            mismatch_mask = pd.Series(False, index=checked_principal.index)
            mismatch_mask.loc[positive_denom] = relative_diff.loc[positive_denom] > principal_rel_tol
            mismatch_mask.loc[~positive_denom] = diff.loc[~positive_denom] > principal_abs_tol

            mismatch_count = int(mismatch_mask.sum())
            summary["principal"] = {
                "status": "ok",
                "total_checked": total_checked,
                "mismatch_count": mismatch_count,
                "mismatch_rate": round(mismatch_count / total_checked * 100, 4),
                "relative_tolerance": principal_rel_tol,
                "absolute_tolerance": principal_abs_tol,
            }

            if mismatch_count > 0:
                issue_cols = [
                    id_col,
                    "借款金额",
                    "lp应还本金合计",
                ]
                detail_df = checked_principal.loc[mismatch_mask, issue_cols].copy()
                detail_df["差值"] = (
                    detail_df["lp应还本金合计"] - detail_df["借款金额"]
                )
                detail_df["相对误差"] = detail_df["差值"] / detail_df["借款金额"].replace(0, np.nan)
                details["principal"] = detail_df
            else:
                details["principal"] = pd.DataFrame(columns=[id_col, "借款金额", "lp应还本金合计", "差值", "相对误差"])
        else:
            summary["principal"] = {
                "status": "ok",
                "total_checked": 0,
                "mismatch_count": 0,
                "mismatch_rate": 0.0,
                "relative_tolerance": principal_rel_tol,
                "absolute_tolerance": principal_abs_tol,
            }
            details["principal"] = pd.DataFrame(columns=[id_col, "借款金额", "lp应还本金合计", "差值", "相对误差"])
    else:
        summary["principal"] = {
            "status": "skipped",
            "reason": "缺少借款金额或应还本金列",
        }
        details["principal"] = pd.DataFrame(columns=[id_col, "借款金额", "lp应还本金合计", "差值", "相对误差"])

    # 借款期限 vs LP 最大期数
    required_lp_term_cols = {id_col, "期数"}
    if {id_col, "借款期限"}.issubset(lc_df.columns) and required_lp_term_cols.issubset(lp_df.columns):
        lc_term = lc_df[[id_col, "借款期限"]].copy()
        lc_term["借款期限"] = pd.to_numeric(lc_term["借款期限"], errors="coerce")

        lp_periods = lp_df[list(required_lp_term_cols)].copy()
        lp_periods["期数"] = pd.to_numeric(lp_periods["期数"], errors="coerce")
        lp_max_period = (
            lp_periods.groupby(id_col, dropna=False)["期数"]
            .max()
            .reset_index(name="lp最大期数")
        )

        merged_term = lc_term.merge(lp_max_period, on=id_col, how="inner")
        available_mask = merged_term["借款期限"].notna() & merged_term["lp最大期数"].notna()
        checked_term = merged_term[available_mask].copy()

        total_checked = len(checked_term)
        if total_checked > 0:
            diff = (checked_term["lp最大期数"] - checked_term["借款期限"]).abs()
            mismatch_mask = diff > term_tol
            mismatch_count = int(mismatch_mask.sum())
            summary["term"] = {
                "status": "ok",
                "total_checked": total_checked,
                "mismatch_count": mismatch_count,
                "mismatch_rate": round(mismatch_count / total_checked * 100, 4),
                "tolerance": term_tol,
            }
            if mismatch_count > 0:
                detail_df = checked_term.loc[mismatch_mask, [id_col, "借款期限", "lp最大期数"]].copy()
                detail_df["差值"] = detail_df["lp最大期数"] - detail_df["借款期限"]
                details["term"] = detail_df
            else:
                details["term"] = pd.DataFrame(columns=[id_col, "借款期限", "lp最大期数", "差值"])
        else:
            summary["term"] = {
                "status": "ok",
                "total_checked": 0,
                "mismatch_count": 0,
                "mismatch_rate": 0.0,
                "tolerance": term_tol,
            }
            details["term"] = pd.DataFrame(columns=[id_col, "借款期限", "lp最大期数", "差值"])
    else:
        summary["term"] = {
            "status": "skipped",
            "reason": "缺少借款期限或期数列",
        }
        details["term"] = pd.DataFrame(columns=[id_col, "借款期限", "lp最大期数", "差值"])

    return summary, details


def _export_feature_analysis(
    feature_df: pd.DataFrame,
    numeric_cols: Iterable[str],
    label_col: str,
    analysis_cfg: dict,
) -> None:
    """根据配置导出特征统计与高相关性特征对。

    该步骤在初始建模时很有帮助，可快速了解特征质量与与标签的关联性。
    配置项为空时自动跳过，减少对主流程的干扰。

    业务解读：
    - 特征统计表能快速让业务方了解每个指标的分布与缺失情况，判断是否符合业务认知；
    - 与标签的相关度提醒业务人员关注可能影响违约率的关键因素；
    - 高相关特征对提示存在信息重复的指标，帮助业务在口径或指标体系上做取舍。
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
        # 业务含义：缺失率反映指标采集是否稳定，均值/方差帮助快速评估客户整体结构

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
        # 业务含义：相关系数提示该指标与违约结果的线性关系强弱，便于业务侧识别重点风险指标
    else:
        summary_df["abs_corr_with_label"] = float("nan")

    summary_df = summary_df.sort_values("abs_corr_with_label", ascending=False, na_position="last")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

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
        # 业务含义：高相关特征对往往来源于相同业务口径，提示需要合并或选取代表性指标，避免重复判断
        pd.DataFrame(records, columns=["feature_a", "feature_b", "abs_corr"]).to_csv(
            high_corr_path,
            index=False,
            encoding="utf-8-sig",
        )


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

    # 清洗三张表（根据规范严格清洗）
    logger.info("开始清洗数据表")
    cleaned_lc = clean_lc(data["lc"], cfg)
    logger.info("LC 表清洗完成")

    cleaned_lp = clean_lp(data["lp"])
    logger.info("LP 表清洗完成 (还款状态已重新赋值)")

    cleaned_lcis = clean_lcis(data["lcis"])
    logger.info("LCIS 表清洗完成 (已处理异常值和无效状态)")
    lcis_notes = getattr(cleaned_lcis, "attrs", {}).get("cleaning_notes", {})
    status_invalid_summary = lcis_notes.get("标当前状态_invalid")
    if status_invalid_summary:
        total_removed = status_invalid_summary.get("total_removed", 0)
        value_counts = status_invalid_summary.get("value_counts", {})
        logger.info(
            "LCIS 标当前状态剔除非法取值: 共 {} 条，分布={}".format(
                total_removed, value_counts
            )
        )

    lc_total_after_cleaning = len(cleaned_lc)
    lcis_total_after_cleaning = len(cleaned_lcis)
    clip_records = []
    for df in (cleaned_lc, cleaned_lcis):
        records = getattr(df, "attrs", {}).get("clip_records", [])
        if records:
            clip_records.extend(records)

    if clip_records:
        extra_rows = []
        for record in clip_records:
            table = record.get("table", "")
            column = record.get("column", "")
            count = int(record.get("count", 0))
            if count <= 0 or not column:
                continue
            if table == "LC":
                total = lc_total_after_cleaning
            elif table == "LCIS":
                total = lcis_total_after_cleaning
            else:
                total = len(cleaned_lp)
            threshold = record.get("threshold")
            if threshold is not None and not pd.isna(threshold):
                data_item = f"{column}_P99裁剪(阈值={threshold:.2f})"
            else:
                data_item = f"{column}_P99裁剪"
            extra_rows.append(
                {
                    "表名": table or "未标注",
                    "数据项": data_item,
                    "异常值数量": count,
                    "总记录数": total,
                    "异常值占比": round(count / total * 100, 2) if total > 0 else 0.0,
                }
            )
        if extra_rows:
            anomaly_stats = pd.concat([anomaly_stats, pd.DataFrame(extra_rows)], ignore_index=True)
            anomaly_stats = anomaly_stats.sort_values(["表名", "数据项"])

    anomaly_stats.to_csv(anomaly_stats_path, index=False, encoding='utf-8-sig')
    logger.info("异常值统计表已保存: {} ({} 条记录)".format(anomaly_stats_path, len(anomaly_stats)))

    consistency_summary, consistency_details = _check_consistency_between_lc_lp(
        cleaned_lc, cleaned_lp, id_col=id_col
    )
    notes = cleaned_lc.attrs.setdefault("cleaning_notes", {})
    notes["consistency_checks"] = consistency_summary
    cleaned_lc.attrs["cleaning_notes"] = notes

    principal_summary = consistency_summary.get("principal", {})
    if principal_summary.get("status") == "ok":
        total_checked = principal_summary.get("total_checked", 0)
        mismatch_count = principal_summary.get("mismatch_count", 0)
        mismatch_rate = principal_summary.get("mismatch_rate", 0.0)
        logger.info(
            "借款金额 vs LP 应还本金一致性：检查 {} 条，超出容差 {} 条，占比 {:.4f}% (相对容差={}, 绝对容差={})".format(
                total_checked,
                mismatch_count,
                mismatch_rate,
                principal_summary.get("relative_tolerance"),
                principal_summary.get("absolute_tolerance"),
            )
        )
        principal_issues = consistency_details.get("principal")
        if principal_issues is not None and not principal_issues.empty:
            issues_path = middata_dir / "principal_inconsistencies.csv"
            principal_issues.to_csv(issues_path, index=False, encoding="utf-8-sig")
            logger.warning(
                "借款金额与 LP 应还本金不一致记录已导出: {} ({} 条)".format(
                    issues_path, len(principal_issues)
                )
            )
    else:
        logger.warning(
            "借款金额 vs LP 应还本金一致性检查跳过：{}".format(principal_summary.get("reason", "未知原因"))
        )

    term_summary = consistency_summary.get("term", {})
    if term_summary.get("status") == "ok":
        total_checked = term_summary.get("total_checked", 0)
        mismatch_count = term_summary.get("mismatch_count", 0)
        mismatch_rate = term_summary.get("mismatch_rate", 0.0)
        logger.info(
            "借款期限 vs LP 最大期数一致性：检查 {} 条，超出容差 {} 条，占比 {:.4f}% (容差={})".format(
                total_checked,
                mismatch_count,
                mismatch_rate,
                term_summary.get("tolerance"),
            )
        )
        term_issues = consistency_details.get("term")
        if term_issues is not None and not term_issues.empty:
            issues_path = middata_dir / "term_inconsistencies.csv"
            term_issues.to_csv(issues_path, index=False, encoding="utf-8-sig")
            logger.warning(
                "借款期限与 LP 最大期数不一致记录已导出: {} ({} 条)".format(issues_path, len(term_issues))
            )
    else:
        logger.warning(
            "借款期限 vs LP 最大期数一致性检查跳过：{}".format(term_summary.get("reason", "未知原因"))
        )

    # 使用清洗后的数据构建有效样本并生成标签
    logger.info("开始构建有效样本与标签")
    cleaned_data = {
        "lc": cleaned_lc,
        "lp": cleaned_lp,
        "lcis": cleaned_lcis
    }
    samples_df = build_samples_with_labels(cleaned_lc, cleaned_lp, cleaned_lcis, id_col=id_col)
    invalid_samples = samples_df.attrs.get("invalid_samples")
    if samples_df.empty:
        raise ValueError("未找到符合样本定义的记录，无法继续数据准备")

    # 保存样本数据集
    samples_output_path = middata_dir / "LC_labeled_samples.csv"
    samples_to_save = samples_df.copy()
    sample_date_cols = [
        "借款成功日期",
        "借款理论到期日期",
        "lp_last_due_date",
        "lp_last_repay_date",
        "lp_recorddate",
        "lcis_recorddate",
    ]
    for col in sample_date_cols:
        if col in samples_to_save.columns and pd.api.types.is_datetime64_any_dtype(samples_to_save[col]):
            samples_to_save[col] = samples_to_save[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
    sample_column_rename = {
        "sum_DPD": "逾期天数总和",
        "label": "违约标签",
        "label_source": "标签来源",
        "is_effective": "是否周期结束样本",
        "lp_max_period": "LP最大期数",
        "lp_last_due_date": "LP最后到期日",
        "lp_last_repay_date": "LP最后还款日",
        "lp_recorddate": "LP记录日期",
        "lcis_recorddate": "LCIS记录日期",
        "is_valid": "是否有效样本",
    }
    samples_export = samples_to_save.rename(
        columns={k: v for k, v in sample_column_rename.items() if k in samples_to_save.columns}
    )
    try:
        samples_export.to_csv(samples_output_path, index=False, encoding='utf-8-sig')
        logger.info(
            "样本集已保存: {} ({} 条记录, {} 个字段)".format(
                samples_output_path, len(samples_df), len(samples_export.columns)
            )
        )
    except PermissionError as e:
        logger.error("无法保存样本集，文件可能被占用: {}".format(e))

    invalid_output_path = middata_dir / "LC_invalid_samples.csv"
    if isinstance(invalid_samples, pd.DataFrame) and not invalid_samples.empty:
        invalid_to_save = invalid_samples.copy()
        for col in sample_date_cols:
            if col in invalid_to_save.columns and pd.api.types.is_datetime64_any_dtype(invalid_to_save[col]):
                invalid_to_save[col] = invalid_to_save[col].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
                )
        invalid_export = invalid_to_save.rename(
            columns={k: v for k, v in sample_column_rename.items() if k in invalid_to_save.columns}
        )
        try:
            invalid_export.to_csv(invalid_output_path, index=False, encoding='utf-8-sig')
            logger.info(
                "未到期样本已保存: {} ({} 条记录, {} 个字段)".format(
                    invalid_output_path, len(invalid_export), len(invalid_export.columns)
                )
            )
        except PermissionError as e:
            logger.error("无法保存未到期样本，文件可能被占用: {}".format(e))
    else:
        logger.info("未到期样本为空，跳过导出")

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

    feature_df = build_feature_dataframe(samples_df, cfg, id_col, logger=logger)
    feature_attrs = getattr(feature_df, "attrs", {}) or {}
    dropped_blacklist = feature_attrs.get("dropped_blacklist_columns") or []
    if dropped_blacklist:
        logger.info("特征工程：已剔除潜在泄露字段 %s", dropped_blacklist)
    dropped_whitelist = feature_attrs.get("dropped_whitelist_columns") or []
    if dropped_whitelist:
        logger.info("特征工程：白名单外字段在构建时被过滤 %s", dropped_whitelist)
    missing_whitelist = feature_attrs.get("missing_whitelist_columns") or []
    if missing_whitelist:
        logger.warning("特征工程：缺少白名单字段 %s，已使用缺省值填充", missing_whitelist)
    metadata_columns = set(feature_attrs.get("metadata_columns") or [])
    feature_drop_cfg = cfg.get("feature_drops") or {}
    feature_drop_records: Dict[str, List[str]] = dict(feature_attrs.get("feature_drops") or {})
    if metadata_columns:
        logger.info("特征工程：保留以下元数据列以供后续处理 %s", sorted(metadata_columns))

    before_drop_cfg = feature_drop_cfg.get("before_encoding")
    if before_drop_cfg:
        feature_df, dropped_before = _apply_feature_drops(
            feature_df,
            before_drop_cfg,
            logger,
            stage="before_encoding",
            protected_cols={id_col},
        )
        if dropped_before:
            feature_drop_records["before_encoding"] = dropped_before
            metadata_columns = {col for col in metadata_columns if col in feature_df.columns}
            feature_attrs["retained_columns"] = [col for col in feature_df.columns]
    feature_attrs["feature_drops"] = feature_drop_records

    labels = generate_labels(cleaned_data, id_col=id_col, samples_df=samples_df)
    if "is_valid" not in labels.columns:
        raise ValueError("标签数据缺少 is_valid 字段，无法识别有效记录")

    valid_labels = labels[labels["is_valid"] == True].copy()
    invalid_label_count = len(labels) - len(valid_labels)
    logger.info(
        "标签生成完成：总记录数={}，有效标签={}，无效标签={}".format(
            len(labels), len(valid_labels), invalid_label_count
        )
    )

    label_column = cfg.get("label_column", "label")
    if label_column not in valid_labels.columns:
        raise ValueError(f"标签数据缺少目标列 {label_column}")

    label_counts = valid_labels[label_column].value_counts(dropna=True)
    if not label_counts.empty:
        logger.info("有效标签分布：")
        for label_val, count in label_counts.items():
            logger.info("  - 标签={}：{} 条".format(label_val, count))

    label_merge_cols = [id_col, label_column]
    if "is_valid" in valid_labels.columns:
        label_merge_cols.append("is_valid")
    feature_df = feature_df.merge(valid_labels[label_merge_cols], on=id_col, how="inner")
    if feature_df.empty:
        raise ValueError("特征表与有效标签合并后为空，请检查样本与标签定义")
    logger.info("特征表合并有效标签后剩余 {} 条记录".format(len(feature_df)))

    target = feature_df[label_column].astype(int)

    analysis_cfg = cfg.get("feature_analysis", {})
    if analysis_cfg:
        _export_feature_analysis(feature_df, cfg.get("numeric_features", []), label_column, analysis_cfg)

    interim_path = interim_dir / "loan_master.parquet"
    feature_snapshot = feature_df.copy()
    feature_snapshot.to_parquet(interim_path, index=False)
    logger.info("中间数据写入 {}".format(interim_path))

    if "is_valid" in feature_df.columns:
        feature_df = feature_df.drop(columns=["is_valid"])
    feature_df = feature_df.drop(columns=[label_column])
    feature_attrs["metadata_columns"] = list(metadata_columns)
    feature_df.attrs = feature_attrs

    categorical_cols = cfg.get("categorical_features", [])
    binary_cols = cfg.get("binary_features", [])
    numeric_cols = cfg.get("numeric_features", [])
    extra_numeric_cols = cfg.get("lp_numeric_features", [])
    if extra_numeric_cols:
        numeric_cols = list(dict.fromkeys(list(numeric_cols) + list(extra_numeric_cols)))

    raw_categorical_cols = list(categorical_cols)
    raw_binary_cols = list(binary_cols)
    raw_numeric_cols = list(numeric_cols)

    categorical_cols = [col for col in raw_categorical_cols if col in feature_df.columns]
    binary_cols = [col for col in raw_binary_cols if col in feature_df.columns]
    numeric_cols = [col for col in raw_numeric_cols if col in feature_df.columns]

    removed_categorical = sorted(set(raw_categorical_cols) - set(categorical_cols))
    removed_binary = sorted(set(raw_binary_cols) - set(binary_cols))
    removed_numeric = sorted(set(raw_numeric_cols) - set(numeric_cols))

    if removed_categorical:
        logger.info("特征消融：分类特征中以下字段不存在或已被移除，将跳过 %s", removed_categorical)
    if removed_binary:
        logger.info("特征消融：二值特征中以下字段不存在或已被移除，将跳过 %s", removed_binary)
    if removed_numeric:
        logger.info("特征消融：数值特征中以下字段不存在或已被移除，将跳过 %s", removed_numeric)

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

    after_drop_cfg = feature_drop_cfg.get("after_encoding")
    if after_drop_cfg:
        features, dropped_after = _apply_feature_drops(
            features,
            after_drop_cfg,
            logger,
            stage="after_encoding",
            protected_cols={id_col},
        )
        if dropped_after:
            feature_drop_records["after_encoding"] = dropped_after
            feature_attrs["feature_drops"] = feature_drop_records

    X = features.drop(columns=[id_col])
    ids = feature_df[id_col]

    selection_cfg = cfg.get("feature_selection")
    X, selected_columns, selection_summary = select_features(X, target, selection_cfg, logger)
    features = pd.concat([features[[id_col]], X], axis=1)
    ids = feature_df[id_col]

    if selection_cfg and selection_cfg.get("enable") and selection_summary is not None and not selection_summary.empty:
        report_path = _resolve_path(selection_cfg.get("report_path"))
        if report_path is None:
            reports_dir = PROJECT_ROOT / "outputs" / "reports" / "tables"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / "feature_selection_summary.csv"
        else:
            report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            selection_summary.to_csv(report_path, index=False, encoding="utf-8-sig")
            logger.info("特征选择报告已保存: {}".format(report_path))
        except PermissionError as exc:
            logger.warning("无法写入特征选择报告: {}".format(exc))
    
    # 最终检查：确保所有特征都是数值类型
    object_cols = [col for col in X.columns if X[col].dtype == 'object']
    if object_cols:
        logger.warning("发现object类型的特征列，将尝试转换为数值类型: {}".format(object_cols))
        for col in object_cols:
            # 尝试转换为数值，如果失败则设为0
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)

    split_strategy = str(cfg.get("split_strategy", "random")).lower()
    test_size = float(cfg.get("test_size", 0.15))
    val_size = float(cfg.get("val_size", 0.15))
    random_state = int(cfg.get("random_state", 42))

    if split_strategy == "time":
        logger.info("使用时序切分策略")
        use_random_split = False
        if "loan_date" not in feature_df.columns:
            logger.warning("特征表缺少 loan_date 列，无法执行时序切分，将回退到随机分层切分")
            use_random_split = True
        else:
            train_end_str = cfg.get("train_end_date", "2016-12-31")
            val_end_str = cfg.get("val_end_date", "2017-06-30")
            train_end = pd.to_datetime(train_end_str)
            val_end = pd.to_datetime(val_end_str)

            dates = pd.to_datetime(feature_df["loan_date"], errors="coerce")
            unknown_mask = dates.isna()
            train_mask = (dates <= train_end) | unknown_mask
            val_mask = (~unknown_mask) & (dates > train_end) & (dates <= val_end)
            test_mask = (~unknown_mask) & (dates > val_end)

            X_train, y_train, ids_train = X[train_mask], target[train_mask], ids[train_mask]
            X_val, y_val, ids_val = X[val_mask], target[val_mask], ids[val_mask]
            X_test, y_test, ids_test = X[test_mask], target[test_mask], ids[test_mask]

            if min(len(X_train), len(X_val), len(X_test)) == 0:
                logger.warning(
                    "时序切分结果存在空集（train=%s, val=%s, test=%s），将改用随机分层切分",
                    len(X_train),
                    len(X_val),
                    len(X_test),
                )
                use_random_split = True

        if use_random_split:
            logger.info("改用随机分层切分策略")
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
    
    # 清洗三张表（根据规范严格清洗）
    logger.info("开始清洗数据表")
    cleaned_lc = clean_lc(data["lc"], cfg)
    logger.info("LC 表清洗完成")
    
    cleaned_lp = clean_lp(data["lp"])
    logger.info("LP 表清洗完成 (还款状态已重新赋值)")
    
    cleaned_lcis = clean_lcis(data["lcis"])
    logger.info("LCIS 表清洗完成 (已处理异常值和无效状态)")

    lc_total_after_cleaning = len(cleaned_lc)
    lcis_total_after_cleaning = len(cleaned_lcis)
    clip_records = []
    for df in (cleaned_lc, cleaned_lcis):
        records = getattr(df, "attrs", {}).get("clip_records", [])
        if records:
            clip_records.extend(records)

    if clip_records:
        extra_rows = []
        for record in clip_records:
            table = record.get("table", "")
            column = record.get("column", "")
            count = int(record.get("count", 0))
            if count <= 0 or not column:
                continue
            if table == "LC":
                total = lc_total_after_cleaning
            elif table == "LCIS":
                total = lcis_total_after_cleaning
            else:
                total = len(cleaned_lp)
            threshold = record.get("threshold")
            if threshold is not None and not pd.isna(threshold):
                data_item = f"{column}_P99裁剪(阈值={threshold:.2f})"
            else:
                data_item = f"{column}_P99裁剪"
            extra_rows.append(
                {
                    "表名": table or "未标注",
                    "数据项": data_item,
                    "异常值数量": count,
                    "总记录数": total,
                    "异常值占比": round(count / total * 100, 2) if total > 0 else 0.0,
                }
            )
        if extra_rows:
            anomaly_stats = pd.concat([anomaly_stats, pd.DataFrame(extra_rows)], ignore_index=True)
            anomaly_stats = anomaly_stats.sort_values(["表名", "数据项"])

    try:
        anomaly_stats.to_csv(anomaly_stats_path, index=False, encoding='utf-8-sig')
        logger.info("异常值统计表已保存: {} ({} 条记录)".format(anomaly_stats_path, len(anomaly_stats)))
    except PermissionError as e:
        logger.error("无法保存异常值统计表，文件可能被占用: {}".format(e))
        logger.info("请关闭Excel或其他打开该文件的程序后再试")

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

