"""样本构建与标签生成模块。

遵循业务给定的统一口径：
- 预测目标为“某一笔放款合同（ListingId）未来逾期概率”；
- 有效样本需满足：借款成功日期 + 借款期限 < recorddate（贷款理论周期已经走完）；
- 负样本：有效样本在周期内出现过逾期（sum_DPD>0 或记录状态为“逾期中”）；
- 正样本：有效样本在周期内从未逾期（sum_DPD=0）。

注：清洗逻辑保持不变，本模块只负责样本圈定与打标签。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

STATUS_OVERDUE = "逾期中"
LABEL_SOURCE_LP_OVERDUE = "lp_sum_dpd"
LABEL_SOURCE_LP_CLEAN = "lp_clean"
LABEL_SOURCE_LCIS_OVERDUE = "lcis_overdue"

LC_LIKE_COLUMNS = [
    "借款金额",
    "借款期限",
    "借款利率",
    "借款成功日期",
    "初始评级",
    "借款类型",
    "是否首标",
    "年龄",
    "性别",
    "手机认证",
    "户口认证",
    "视频认证",
    "学历认证",
    "征信认证",
    "淘宝认证",
    "历史成功借款次数",
    "历史成功借款金额",
    "总待还本金",
    "历史正常还款期数",
    "历史逾期还款期数",
]


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """将任意 Series 转换为 datetime64[ns]，失败置为 NaT。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _prepare_lc(lc: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """为 LC 表补充借款理论到期日期。"""
    required_cols = {id_col, "借款成功日期", "借款期限"}
    missing = required_cols - set(lc.columns)
    if missing:
        raise KeyError(f"LC 表缺少必要字段: {missing}")

    df = lc[lc[id_col].notna()].copy()
    df["借款成功日期"] = _coerce_datetime(df["借款成功日期"])
    df["借款期限"] = pd.to_numeric(df["借款期限"], errors="coerce")

    def _calc_due(start: pd.Timestamp, term: float) -> pd.Timestamp:
        if pd.isna(start) or pd.isna(term):
            return pd.NaT
        try:
            term_int = int(round(float(term)))
        except (TypeError, ValueError):
            return pd.NaT
        if term_int < 0:
            return pd.NaT
        return start + pd.DateOffset(months=term_int)

    df["借款理论到期日期"] = [
        _calc_due(start, term) for start, term in zip(df["借款成功日期"], df["借款期限"])
    ]
    return df


def _summarize_lp(lp: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """按 ListingId 汇总 LP 表，得到 sum_DPD 及最后一期信息。"""
    required_cols = {id_col, "期数", "到期日期", "recorddate"}
    missing = required_cols - set(lp.columns)
    if missing:
        raise KeyError(f"LP 表缺少必要字段: {missing}")

    subset_cols = [id_col, "期数", "到期日期", "还款日期", "recorddate"]
    subset_cols = [col for col in subset_cols if col in lp.columns]
    df = lp[lp[id_col].notna()].copy()
    df = df[subset_cols]

    df["期数"] = pd.to_numeric(df.get("期数"), errors="coerce")
    for col in ("到期日期", "还款日期", "recorddate"):
        if col in df.columns:
            df[col] = _coerce_datetime(df[col])

    sort_keys = [col for col in ("期数", "recorddate", "到期日期") if col in df.columns]
    df_sorted = df.sort_values(sort_keys)
    last_records = df_sorted.groupby(id_col, dropna=False).tail(1)

    rename_map = {
        "期数": "lp_max_period",
        "到期日期": "lp_last_due_date",
        "还款日期": "lp_last_repay_date",
        "recorddate": "lp_recorddate",
    }
    last_records_renamed = last_records.rename(
        columns={col: rename_map[col] for col in rename_map if col in last_records.columns}
    )
    keep_cols = [id_col] + [
        rename_map[col] for col in rename_map if rename_map[col] in last_records_renamed.columns
    ]

    fallback_col = None
    if "lp_recorddate" in last_records_renamed.columns:
        fallback_col = "_lp_recorddate_fallback"
        fallback_df = last_records_renamed[[id_col, "lp_recorddate"]].rename(
            columns={"lp_recorddate": fallback_col}
        )
        df = df.merge(fallback_df, on=id_col, how="left")
    else:
        df[fallback_col or "_lp_recorddate_fallback"] = pd.NaT
        fallback_col = "_lp_recorddate_fallback"

    if "还款日期" in df.columns:
        df["effective_repay_date"] = df["还款日期"]
    else:
        df["effective_repay_date"] = pd.NaT
    df["effective_repay_date"] = df["effective_repay_date"].fillna(df[fallback_col])
    if "recorddate" in df.columns:
        df["effective_repay_date"] = df["effective_repay_date"].fillna(df["recorddate"])

    df["dpd"] = (df["effective_repay_date"] - df["到期日期"]).dt.days
    df["dpd"] = df["dpd"].apply(lambda x: max(x, 0) if pd.notna(x) else 0.0)

    dpd_summary = (
        df.groupby(id_col, dropna=False)["dpd"]
        .sum(min_count=1)
        .fillna(0.0)
        .reset_index(name="sum_DPD")
    )

    df.drop(columns=[fallback_col, "effective_repay_date"], inplace=True, errors="ignore")

    summary = last_records_renamed[keep_cols].merge(dpd_summary, on=id_col, how="left")
    return summary


def _extract_overdue_from_lcis(lcis: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """获取 LCIS 中最新状态为“逾期中”的 ListingId。"""
    if lcis is None or lcis.empty or id_col not in lcis.columns:
        return pd.DataFrame(columns=[id_col, "lcis_recorddate"])
    if "标当前状态" not in lcis.columns:
        return pd.DataFrame(columns=[id_col, "lcis_recorddate"])

    subset_cols = [id_col, "标当前状态"] + [col for col in LC_LIKE_COLUMNS if col in lcis.columns]
    if "recorddate" in lcis.columns:
        subset_cols.append("recorddate")
    df = lcis[subset_cols].copy()
    df = df[df[id_col].notna()]

    overdue_only = df[df["标当前状态"] == STATUS_OVERDUE].copy()
    if overdue_only.empty:
        return pd.DataFrame(columns=[id_col, "lcis_recorddate"])

    if "recorddate" in overdue_only.columns:
        overdue_only["recorddate"] = _coerce_datetime(overdue_only["recorddate"])
        overdue_only = overdue_only.sort_values("recorddate")
        latest_overdue = overdue_only.groupby(id_col, dropna=False).tail(1)
        latest_overdue = latest_overdue.rename(columns={"recorddate": "lcis_recorddate"})
    else:
        # 如果没有记录日期，按原始顺序保留最后一条逾期记录
        latest_overdue = overdue_only.groupby(id_col, dropna=False).tail(1)
        latest_overdue["lcis_recorddate"] = pd.NaT

    columns_to_return = [id_col, "lcis_recorddate"] + [
        col for col in LC_LIKE_COLUMNS if col in latest_overdue.columns
    ]
    return latest_overdue[columns_to_return].reset_index(drop=True)


def _base_columns(id_col: str) -> list[str]:
    return [
        id_col,
        "sum_DPD",
        "label",
        "label_source",
        "is_effective",
        "lp_max_period",
        "lp_last_due_date",
        "lp_last_repay_date",
        "lp_recorddate",
        "lcis_recorddate",
        "借款理论到期日期",
    ]


def _ensure_base_columns(
    df: pd.DataFrame,
    id_col: str,
    *,
    is_effective_default: bool,
) -> pd.DataFrame:
    """补齐样本信息必备列，保持列顺序统一。"""
    datetime_cols = {
        "lp_last_due_date",
        "lp_last_repay_date",
        "lp_recorddate",
        "lcis_recorddate",
        "借款理论到期日期",
    }
    numeric_cols = {"sum_DPD", "lp_max_period"}

    result = df.copy()
    for col in _base_columns(id_col):
        if col in result.columns:
            continue
        if col in datetime_cols:
            result[col] = pd.NaT
        elif col in numeric_cols:
            result[col] = np.nan
        elif col == "label":
            result[col] = pd.NA
        elif col == "label_source":
            result[col] = ""
        elif col == "is_effective":
            result[col] = is_effective_default
        else:
            result[col] = np.nan
    return result[_base_columns(id_col)]


def _empty_sample_frame(lc: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """当无有效样本时返回一个结构完整但无行的数据框。"""
    base = lc.head(0).copy()
    for col in _base_columns(id_col):
        if col in base.columns:
            continue
        if col == "label":
            base[col] = pd.Series(dtype="Int64")
        elif col in {"sum_DPD"}:
            base[col] = pd.Series(dtype=float)
        elif col == "lp_max_period":
            base[col] = pd.Series(dtype=float)
        elif col == "is_effective":
            base[col] = pd.Series(dtype=bool)
        elif col == "label_source":
            base[col] = pd.Series(dtype=object)
        else:
            base[col] = pd.Series(dtype="datetime64[ns]")
    base["is_valid"] = pd.Series(dtype=bool)
    return base


def build_samples_with_labels(
    lc: pd.DataFrame,
    lp: pd.DataFrame,
    lcis: pd.DataFrame | None,
    id_col: str = "ListingId",
) -> pd.DataFrame:
    """构建“LC字段 + 标签”的样本数据集。

    返回的 DataFrame 至少包含：
    - LC 清洗后的全部字段；
    - sum_DPD、is_effective、label_source 等中间字段；
    - label（0/1，可空整型）、is_valid（是否有效样本）。
    """
    if lc is None or lp is None:
        raise ValueError("LC/LP 数据不能为空")

    prepared_lc = _prepare_lc(lc, id_col)
    lp_summary = _summarize_lp(lp, id_col)

    candidate = prepared_lc[[id_col, "借款理论到期日期"]].merge(lp_summary, on=id_col, how="inner")
    if candidate.empty:
        return _empty_sample_frame(prepared_lc, id_col)

    if "lp_recorddate" in candidate.columns:
        candidate["lp_recorddate"] = _coerce_datetime(candidate["lp_recorddate"])
    else:
        candidate["lp_recorddate"] = pd.NaT

    candidate["is_effective"] = (
        candidate["借款理论到期日期"].notna()
        & candidate["lp_recorddate"].notna()
        & (candidate["借款理论到期日期"] < candidate["lp_recorddate"])
    )

    effective = candidate[candidate["is_effective"]].copy()
    effective["label"] = np.where(effective["sum_DPD"] > 0, 1, 0)
    effective["label_source"] = np.where(
        effective["label"] == 1, LABEL_SOURCE_LP_OVERDUE, LABEL_SOURCE_LP_CLEAN
    )
    effective["lcis_recorddate"] = pd.NaT
    effective = _ensure_base_columns(effective, id_col=id_col, is_effective_default=True)

    invalid = candidate[~candidate["is_effective"]].copy()
    if not invalid.empty:
        invalid["label"] = pd.NA
        invalid["label_source"] = "lp_not_matured"
        invalid["is_effective"] = False
        invalid["lcis_recorddate"] = pd.NaT
        invalid = _ensure_base_columns(invalid, id_col=id_col, is_effective_default=False)
    else:
        invalid = _ensure_base_columns(
            pd.DataFrame(columns=[id_col]), id_col=id_col, is_effective_default=False
        ).iloc[0:0]

    overdue = _extract_overdue_from_lcis(lcis, id_col)
    if overdue.empty:
        overdue = _ensure_base_columns(
            pd.DataFrame(columns=[id_col]), id_col=id_col, is_effective_default=False
        )
        overdue = overdue.iloc[0:0]
    else:
        overdue["sum_DPD"] = np.nan
        overdue["label"] = 1
        overdue["label_source"] = LABEL_SOURCE_LCIS_OVERDUE
        overdue["is_effective"] = False
        overdue["lp_max_period"] = np.nan
        overdue["lp_last_due_date"] = pd.NaT
        overdue["lp_last_repay_date"] = pd.NaT
        overdue["lp_recorddate"] = pd.NaT
        overdue["借款理论到期日期"] = pd.NaT
        overdue = _ensure_base_columns(overdue, id_col=id_col, is_effective_default=False)

    sample_info = pd.concat([effective, overdue], ignore_index=True)
    if sample_info.empty:
        return _empty_sample_frame(prepared_lc, id_col)

    sample_info = sample_info.sort_values(
        by=["label", "label_source"], ascending=[False, True]
    ).drop_duplicates(subset=id_col, keep="first")
    sample_info["label"] = sample_info["label"].astype("Int64")
    if "lp_max_period" in sample_info.columns:
        sample_info["lp_max_period"] = sample_info["lp_max_period"].astype("Float64")

    sample_mergable = sample_info.drop(columns=["借款理论到期日期"], errors="ignore")
    merged = prepared_lc.merge(sample_mergable, on=id_col, how="right")
    merged = merged.sort_values(id_col).reset_index(drop=True)
    merged["is_valid"] = True

    if not invalid.empty:
        invalid_mergable = invalid.drop(columns=["借款理论到期日期"], errors="ignore")
        invalid_merged = prepared_lc.merge(invalid_mergable, on=id_col, how="right")
        invalid_merged = invalid_merged.sort_values(id_col).reset_index(drop=True)
        invalid_merged["is_valid"] = False
    else:
        invalid_merged = invalid.copy()
        invalid_merged["is_valid"] = False

    merged.attrs["invalid_samples"] = invalid_merged

    def _fill_from_lcis(target: pd.DataFrame) -> pd.DataFrame:
        if lcis is None or lcis.empty or target.empty:
            return target
        enrich_cols = [col for col in LC_LIKE_COLUMNS if col in lcis.columns]
        if not enrich_cols:
            return target
        working = lcis.copy()
        if "recorddate" in working.columns:
            working["recorddate"] = _coerce_datetime(working["recorddate"])
            working = working.sort_values("recorddate")
        working = working.drop_duplicates(subset=id_col, keep="last")
        working = working.set_index(id_col)
        for col in enrich_cols:
            if col not in working.columns:
                continue
            series = working[col]
            if col in target.columns:
                target[col] = target[col].where(target[col].notna(), target[id_col].map(series))
            else:
                target[col] = target[id_col].map(series)
        return target

    merged = _fill_from_lcis(merged)
    invalid_merged = _fill_from_lcis(invalid_merged)
    merged.attrs["invalid_samples"] = invalid_merged

    return merged


def generate_labels(
    data: Dict[str, pd.DataFrame],
    id_col: str = "ListingId",
    samples_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """生成标签明细表，用于与特征表合并。

    参数
    ----
    data: Dict[str, DataFrame]
        包含清洗后的 lc / lp / lcis 数据。
    id_col: str
        唯一标识列，默认 ListingId。
    samples_df: DataFrame | None
        已经构建好的样本集（可选，提供可避免重复计算）。

    返回
    ----
    DataFrame
        至少包含 [id_col, label, is_valid]，外加 sum_DPD / label_source 等辅助字段。
    """
    if samples_df is None:
        samples_df = build_samples_with_labels(
            data["lc"],
            data["lp"],
            data.get("lcis"),
            id_col=id_col,
        )

    if samples_df.empty:
        return samples_df[[id_col, "label", "is_valid"]] if "label" in samples_df.columns else pd.DataFrame(
            columns=[id_col, "label", "is_valid"]
        )

    label_cols = [id_col, "label", "is_valid"]
    optional_cols = [
        "sum_DPD",
        "is_effective",
        "label_source",
        "lp_max_period",
        "lp_last_due_date",
        "lp_last_repay_date",
        "lp_recorddate",
        "lcis_recorddate",
        "借款理论到期日期",
    ]
    label_cols.extend([col for col in optional_cols if col in samples_df.columns])
    return samples_df[label_cols].copy()

