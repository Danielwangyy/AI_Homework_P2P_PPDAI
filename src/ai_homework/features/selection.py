"""自动化特征筛选模块。

该模块实现多种特征选择手段：
- 皮尔逊相关系数：衡量数值特征与标签的线性关联；
- 卡方检验：衡量离散/非负特征与标签的统计关联；
- 随机森林重要性：通过树模型捕捉非线性关系。

每种方法的入选结果会被记录并汇总，最终按照配置策略（并集/投票/交集）
确定需要保留的特征列，同时支持导出分析报告供人工审阅。
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2

from .constants import ALL_BLACKLIST_COLUMNS


def _log(logger, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if log_fn is None:
        logger.info(message)
    else:
        log_fn(message)


def _calc_correlations(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for col in X.columns:
        series = X[col]
        if series.nunique(dropna=False) <= 1:
            scores[col] = 0.0
            continue
        corr_val = series.corr(y)
        if pd.isna(corr_val):
            corr_val = 0.0
        scores[col] = float(abs(corr_val))
    return scores


def _correlation_select(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> Tuple[List[str], Dict[str, float]]:
    scores = _calc_correlations(X, y)
    threshold = float(cfg.get("threshold", 0.0))
    max_features = cfg.get("max_features")
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [col for col, score in sorted_items if score >= threshold]
    if max_features is not None:
        max_k = int(max_features)
        selected = selected[:max_k]
    return selected, scores


def _chi2_select(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> Tuple[List[str], Dict[str, float]]:
    if X.empty:
        return [], {}
    X_adj = X.copy()
    for col in X_adj.columns:
        min_val = X_adj[col].min()
        if pd.notna(min_val) and min_val < 0:
            X_adj[col] = X_adj[col] - min_val
    non_constant_cols = [col for col in X_adj.columns if X_adj[col].var() > 0]
    if not non_constant_cols:
        return [], {}
    X_use = X_adj[non_constant_cols]
    scores_arr, _ = chi2(X_use, y)
    scores = {col: float(score) for col, score in zip(non_constant_cols, scores_arr)}
    top_k = int(cfg.get("top_k", len(non_constant_cols)))
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [col for col, score in sorted_items if not np.isnan(score)]
    selected = selected[: min(top_k, len(selected))]
    return selected, scores


def _model_importance_select(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> Tuple[List[str], Dict[str, float]]:
    if X.empty:
        return [], {}
    n_estimators = int(cfg.get("n_estimators", 200))
    random_state = cfg.get("random_state", 42)
    max_depth = cfg.get("max_depth")
    params = {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "n_jobs": cfg.get("n_jobs", -1),
    }
    if max_depth is not None:
        params["max_depth"] = max_depth
    clf = RandomForestClassifier(**params)
    clf.fit(X, y)
    importances = clf.feature_importances_
    scores = {col: float(score) for col, score in zip(X.columns, importances)}
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_k = int(cfg.get("top_k", len(sorted_items)))
    selected = [col for col, score in sorted_items if score > 0]
    selected = selected[: min(top_k, len(selected))]
    return selected, scores


def _combine_results(
    candidates: Dict[str, List[str]],
    cfg: dict,
) -> List[str]:
    if not candidates:
        return []
    strategy = str(cfg.get("strategy", "union")).lower()
    min_methods = int(cfg.get("min_methods", 1))
    votes = Counter()
    for cols in candidates.values():
        for col in cols:
            votes[col] += 1
    total_methods = len(candidates)
    if strategy == "intersection":
        selected = [col for col, count in votes.items() if count == total_methods]
    elif strategy == "vote":
        threshold = max(1, min(min_methods, total_methods))
        selected = [col for col, count in votes.items() if count >= threshold]
    else:  # 默认并集
        selected = list(votes.keys())
    return selected


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict | None,
    logger=None,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """根据配置执行自动化特征筛选。

    返回：
        X_selected: 筛选后的特征矩阵
        selected_columns: 保留的特征列名
        summary_df: 筛选过程汇总表
    """
    if cfg is None or not cfg.get("enable", False):
        _log(logger, "info", "特征选择未启用，保留所有特征列")
        return X.copy(), list(X.columns), pd.DataFrame()

    forbidden = sorted(set(X.columns) & ALL_BLACKLIST_COLUMNS)
    if forbidden:
        message = f"检测到禁止进入模型的字段：{forbidden}"
        _log(logger, "error", message)
        raise ValueError(message)

    X = X.copy()
    method_results: Dict[str, List[str]] = {}
    method_scores: Dict[str, Dict[str, float]] = {}

    if cfg.get("correlation", {}).get("enable", True):
        try:
            selected, scores = _correlation_select(X, y, cfg.get("correlation", {}))
            method_results["correlation"] = selected
            method_scores["correlation"] = scores
            _log(
                logger,
                "info",
                f"相关系数筛选保留 {len(selected)} 个特征（阈值={cfg.get('correlation', {}).get('threshold', 0.0)}）",
            )
        except Exception as exc:  # noqa: BLE001
            _log(logger, "warning", f"相关系数筛选失败：{exc}")

    if cfg.get("chi2", {}).get("enable", False):
        try:
            selected, scores = _chi2_select(X, y, cfg.get("chi2", {}))
            method_results["chi2"] = selected
            method_scores["chi2"] = scores
            _log(
                logger,
                "info",
                f"卡方检验筛选保留 {len(selected)} 个特征（top_k={cfg.get('chi2', {}).get('top_k')}）",
            )
        except Exception as exc:  # noqa: BLE001
            _log(logger, "warning", f"卡方检验筛选失败：{exc}")

    if cfg.get("model_importance", {}).get("enable", False):
        try:
            selected, scores = _model_importance_select(X, y, cfg.get("model_importance", {}))
            method_results["model_importance"] = selected
            method_scores["model_importance"] = scores
            _log(
                logger,
                "info",
                "模型重要性筛选保留 {} 个特征（top_k={}）".format(
                    len(selected), cfg.get("model_importance", {}).get("top_k")
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _log(logger, "warning", f"基于模型的重要性筛选失败：{exc}")

    selected_columns = _combine_results(method_results, cfg)
    if not selected_columns:
        _log(logger, "warning", "自动化筛选未选出任何特征，回退至完整特征集")
        selected_columns = list(X.columns)

    selected_columns = [col for col in selected_columns if col in X.columns]
    X_selected = X[selected_columns].copy()
    _log(logger, "info", f"最终保留特征列数量：{len(selected_columns)}")

    summary_records = []
    for col in X.columns:
        record = {"feature": col, "selected": col in selected_columns}
        for method_name, scores in method_scores.items():
            record[f"{method_name}_score"] = scores.get(col, np.nan)
            record[f"{method_name}_selected"] = col in method_results.get(method_name, [])
        summary_records.append(record)
    summary_df = pd.DataFrame(summary_records)

    return X_selected, selected_columns, summary_df

