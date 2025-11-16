"""特征工程模块。

文件结构遵循“单一职责”原则：
- `add_time_features`：日期字段拆分；
- `add_derived_features`：基于历史记录构造比率、均值等；
- `compute_lp_features`：将还款计划按借款 ID 汇总；
- `build_feature_dataframe`：将所有特征合并到一起。

函数名和注释都尽量贴近业务含义，帮助初学者理解。
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """从借款成功日期中衍生出 year/month/quarter/weekday。"""
    if "借款成功日期" not in df.columns:
        return df
    temp = pd.to_datetime(df["借款成功日期"], errors="coerce")
    df["借款成功日期_year"] = temp.dt.year.fillna(-1).astype(int).astype(str)
    df["借款成功日期_month"] = temp.dt.month.fillna(-1).astype(int).astype(str)
    df["借款成功日期_quarter"] = temp.dt.quarter.fillna(-1).astype(int).astype(str)
    df["借款成功日期_weekday"] = temp.dt.weekday.fillna(-1).astype(int).astype(str)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造一系列基于历史表现的比率与均值特征。"""
    df = df.copy()

    def _get_series(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].astype(float)
        return pd.Series(0.0, index=df.index, dtype=float)

    history_normal = _get_series("历史正常还款期数")
    history_overdue = _get_series("历史逾期还款期数")
    total_periods = history_normal + history_overdue
    total_periods = total_periods.replace(0, np.nan)

    df["历史逾期率"] = (history_overdue / total_periods).fillna(0.0)
    df["历史还款能力指数"] = (history_normal / total_periods).fillna(0.0)

    history_amount = _get_series("历史成功借款金额")
    current_amount = _get_series("借款金额")
    total_principal_outstanding = _get_series("总待还本金")

    denom_amount = history_amount.replace({0: np.nan})
    df["借款杠杆比"] = (
        current_amount / denom_amount
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["待还本金占比"] = (
        total_principal_outstanding / denom_amount
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    loan_term = _get_series("借款期限").replace(0, np.nan)
    df["借款金额每期"] = (current_amount / loan_term).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    history_count = _get_series("历史成功借款次数").replace(0, np.nan)
    history_avg_amount = (
        history_amount / history_count
    ).replace([np.inf, -np.inf], np.nan)
    df["历史平均借款金额"] = history_avg_amount.fillna(0.0)

    df["借款金额相对历史平均"] = (
        current_amount / history_avg_amount.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["历史平均每期还款额"] = (
        history_amount / total_periods
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["历史还款净差"] = history_normal - history_overdue

    return df


def compute_lp_features(lp: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """将 LP 明细表聚合为“借款计划”级别的特征。"""
    if lp.empty or id_col not in lp.columns:
        return pd.DataFrame(columns=[id_col])

    temp = lp.copy()
    grouped = temp.groupby(id_col, dropna=False)
    features = grouped.agg(
        lp总期数=("期数", "max"),
        lp计划总本金=("应还本金", "sum"),
        lp计划总利息=("应还利息", "sum"),
        lp平均期本金=("应还本金", "mean"),
        lp平均期利息=("应还利息", "mean"),
    )

    features["lp平均期本息"] = (
        features["lp平均期本金"].fillna(0.0) + features["lp平均期利息"].fillna(0.0)
    )

    if "到期日期" in temp.columns:
        schedule_span = grouped["到期日期"].agg(lambda s: (s.max() - s.min()).days if s.notna().any() else 0)
        features["lp计划周期天数"] = schedule_span.fillna(0)

    return features.reset_index()


def build_feature_dataframe(
    cleaned_lc: pd.DataFrame,
    lp: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: Dict[str, Iterable[str]],
    id_col: str,
) -> pd.DataFrame:
    """综合 LC 清洗结果、LP 聚合特征与标签，生成最终特征表。"""
    categorical_cols = cfg.get("categorical_features", [])
    binary_cols = cfg.get("binary_features", [])
    numeric_cols = list(cfg.get("numeric_features", []))
    extra_numeric_cols = cfg.get("lp_numeric_features", [])

    df = add_time_features(cleaned_lc)
    df = add_derived_features(df)

    # 合并标签，包括is_valid列（如果存在）
    label_cols = [id_col, "label"]
    if "is_valid" in labels.columns:
        label_cols.append("is_valid")
    if "lcis_label" in labels.columns:
        label_cols.append("lcis_label")
    if "lp_label" in labels.columns:
        label_cols.append("lp_label")
    
    df = df.merge(labels[label_cols], on=id_col, how="left")

    lp_features = compute_lp_features(lp, id_col)
    if not lp_features.empty:
        df = df.merge(lp_features, on=id_col, how="left")

    inferred_numeric_cols = set(extra_numeric_cols or [])
    if not lp_features.empty:
        inferred_numeric_cols.update(
            col for col in lp_features.columns if col != id_col
        )

    for col in set(numeric_cols).union(inferred_numeric_cols):
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("未知").astype(str)

    return df

