"""Feature engineering module restricted to `LC_labeled_samples.csv`.

All derived features must originate from the sample dataset itself
to avoid reintroducing data leakage from raw LC/LP/LCIS tables.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .constants import (
    ALIAS_MAP,
    ALL_BLACKLIST_COLUMNS,
    BINARY_ALIAS_COLUMNS,
    CATEGORICAL_ALIAS_COLUMNS,
    METADATA_COLUMNS,
    NUMERIC_ALIAS_COLUMNS,
    SAFE_RAW_COLUMNS,
)

YES_VALUES = {"是", "Y", "YES", "TRUE", "T", "1", "1.0"}
NO_VALUES = {"否", "N", "NO", "FALSE", "F", "0", "0.0"}

RATING_SCORE = {
    "AAA": 8,
    "AA": 7,
    "A": 6,
    "B": 5,
    "C": 4,
    "D": 3,
    "E": 2,
    "F": 1,
}


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: np.nan})
    result = numerator.astype(float) / denom
    return result.replace([np.inf, -np.inf], np.nan)


def _normalize_binary(series: pd.Series) -> pd.Series:
    def mapper(value: object) -> float | np.nan:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if np.isnan(value):
                return np.nan
            return 1.0 if value > 0 else 0.0
        text = str(value).strip().upper()
        if text in YES_VALUES or value is True:
            return 1.0
        if text in NO_VALUES or value is False:
            return 0.0
        return np.nan

    return series.apply(mapper)


def _rating_to_numeric(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)

    def mapper(value: object) -> float:
        if pd.isna(value):
            return 0.0
        text = str(value).strip().upper()
        if text in RATING_SCORE:
            return float(RATING_SCORE[text])
        if text and text[0] in RATING_SCORE:
            return float(RATING_SCORE[text[0]])
        return 0.0

    return series.apply(mapper).astype(float)


def add_domain_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for alias, source_col in ALIAS_MAP.items():
        if alias in df.columns:
            continue
        if source_col in df.columns:
            df[alias] = df[source_col]
        else:
            df[alias] = np.nan

    for col in NUMERIC_ALIAS_COLUMNS:
        if col in df.columns:
            df[col] = _to_numeric(df[col]).fillna(0.0).astype(float)

    if "loan_date" in df.columns:
        df["loan_date"] = _to_datetime(df["loan_date"])

    for col in BINARY_ALIAS_COLUMNS:
        if col in df.columns:
            df[col] = _normalize_binary(df[col]).fillna(0.0)

    for col in CATEGORICAL_ALIAS_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("未知").astype(str)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    loan_date = df["loan_date"]

    df["loan_date_year"] = loan_date.dt.year.fillna(-1).astype(int).astype(str)
    df["loan_date_quarter"] = loan_date.dt.quarter.fillna(-1).astype(int).astype(str)
    df["loan_date_month"] = loan_date.dt.month.fillna(-1).astype(int).astype(str)
    df["loan_date_weekday"] = loan_date.dt.weekday.fillna(-1).astype(int).astype(str)

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    history_normal = df["history_normal_terms"]
    history_overdue = df["history_overdue_terms"]
    total_terms = history_normal + history_overdue
    total_terms = total_terms.replace(0, np.nan)

    df["history_repay_ratio"] = _safe_divide(history_normal, total_terms).fillna(0.0)
    df["history_overdue_rate"] = _safe_divide(history_overdue, total_terms).fillna(0.0)

    df["loan_amount_per_term"] = _safe_divide(df["loan_amount"], df["loan_term"]).fillna(0.0)
    df["history_avg_loan_amount"] = _safe_divide(df["history_total_amount"], df["history_total_loans"]).fillna(0.0)
    df["loan_amount_ratio_to_history_avg"] = _safe_divide(df["loan_amount"], df["history_avg_loan_amount"]).fillna(0.0)
    df["history_avg_term_payment"] = _safe_divide(df["history_total_amount"], total_terms).fillna(0.0)

    df["loan_amount_to_history_amount_ratio"] = _safe_divide(
        df["loan_amount"], df["history_total_amount"]
    ).fillna(0.0)
    df["outstanding_to_history_amount_ratio"] = _safe_divide(
        df["outstanding_principal"], df["history_total_amount"]
    ).fillna(0.0)

    df["loan_amount_history_repay_ratio"] = (df["loan_amount"] * df["history_repay_ratio"]).fillna(0.0)
    df["loan_term_history_overdue_rate"] = (df["loan_term"] * df["history_overdue_rate"]).fillna(0.0)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rating_numeric"] = _rating_to_numeric(df["rating"])

    df["loan_amount_rating_interaction"] = (df["loan_amount"] * df["rating_numeric"]).fillna(0.0)
    df["loan_term_rating_interaction"] = (df["loan_term"] * df["rating_numeric"]).fillna(0.0)

    return df


def build_feature_dataframe(
    samples: pd.DataFrame,
    _cfg: Dict[str, Iterable[str]],
    id_col: str,
    *,
    logger: Optional[object] = None,
) -> pd.DataFrame:
    """Construct feature table using only application-time information."""
    df = samples.copy()
    df.attrs = {}

    if id_col not in df.columns:
        raise KeyError(f"样本数据缺少唯一标识列 {id_col}")

    initial_columns = set(df.columns)
    blacklist_cols = sorted(initial_columns & ALL_BLACKLIST_COLUMNS)
    if blacklist_cols:
        df = df.drop(columns=blacklist_cols, errors="ignore")
        if logger:
            logger.info("特征工程：剔除黑名单字段 %s", blacklist_cols)

    whitelist = SAFE_RAW_COLUMNS | {id_col}
    whitelist_drop = sorted(set(df.columns) - whitelist)
    if whitelist_drop:
        df = df.drop(columns=whitelist_drop, errors="ignore")
        if logger:
            logger.info("特征工程：因白名单限制排除字段 %s", whitelist_drop)

    missing_required = sorted((whitelist - {id_col}) - set(df.columns))
    for col in missing_required:
        df[col] = np.nan
    if missing_required and logger:
        logger.warning("特征工程：白名单字段缺失，已填充缺省值 %s", missing_required)

    date_col = ALIAS_MAP.get("loan_date", "借款成功日期")
    if date_col in df.columns:
        df[date_col] = _to_datetime(df[date_col])

    df = add_domain_aliases(df)
    df = add_time_features(df)
    df = add_ratio_features(df)
    df = add_interaction_features(df)

    raw_cols_to_remove = (SAFE_RAW_COLUMNS - {id_col}) & set(df.columns)
    if raw_cols_to_remove:
        df = df.drop(columns=raw_cols_to_remove)

    remaining_leak = sorted(set(df.columns) & ALL_BLACKLIST_COLUMNS)
    if remaining_leak:
        raise AssertionError(f"检测到未剔除的泄露字段: {remaining_leak}")

    for meta_col in METADATA_COLUMNS:
        if meta_col not in df.columns:
            df[meta_col] = pd.NaT

    df.attrs = {
        "dropped_blacklist_columns": blacklist_cols,
        "dropped_whitelist_columns": whitelist_drop,
        "missing_whitelist_columns": missing_required,
        "metadata_columns": list(METADATA_COLUMNS),
        "retained_columns": list(df.columns),
    }

    return df

