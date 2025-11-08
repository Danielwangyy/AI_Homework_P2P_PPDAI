"""模型训练与调参工具。

这里提供若干“模型训练工厂函数”，供流水线调用。
重点在于 `_train_with_cv`：它负责执行交叉验证、记录历史结果，并返回最佳模型。
初学者可以从该函数入手，理解网格搜索与交叉验证的基本流程。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from xgboost import XGBClassifier


@dataclass
class ModelResult:
    """统一的模型训练结果数据结构。"""
    name: str
    estimator: Any
    best_params: Dict[str, Any]
    validation_score: float
    history: Iterable[Dict[str, Any]]
    best_threshold: Optional[float] = None


def _train_with_cv(
    name: str,
    model_constructor: Callable[[Dict[str, Any]], Any],
    base_params: Dict[str, Any],
    param_grid: Dict[str, Iterable[Any]],
    X_train,
    y_train,
    scoring_cfg: Dict[str, Any],
    cv_cfg: Dict[str, Any],
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> ModelResult:
    """执行交叉验证并搜寻最佳超参数。

    关键思路：
    1. 根据 `param_grid` 枚举所有候选组合；
    2. 使用 `StratifiedKFold` 保持类别分布；
    3. 根据配置决定评估指标，默认 F-beta；
    4. 记录每一组参数的平均分与标准差；
    5. 重新训练最佳参数模型，返回 `ModelResult`。
    """
    grid = list(ParameterGrid(param_grid)) if param_grid else [{}]
    folds = int(cv_cfg.get("folds", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = cv_cfg.get("random_state", 42 if shuffle else None)
    skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)

    metric_type = scoring_cfg.get("type", "f_beta").lower()
    beta = float(scoring_cfg.get("beta", 1.0))

    best_score = -np.inf
    best_params = None
    best_estimator = None
    history: list[Dict[str, Any]] = []

    for param_set in grid:
        params = {**base_params, **param_set}
        fold_scores = []
        for train_idx, valid_idx in skf.split(X_train, y_train):
            estimator = model_constructor(params)
            estimator.fit(
                X_train.iloc[train_idx],
                y_train.iloc[train_idx],
                **(fit_kwargs or {}),
            )
            preds = estimator.predict(X_train.iloc[valid_idx])

            if metric_type == "f_beta":
                score = fbeta_score(
                    y_train.iloc[valid_idx],
                    preds,
                    beta=beta,
                    zero_division=0,
                )
            else:
                score = fbeta_score(
                    y_train.iloc[valid_idx],
                    preds,
                    beta=1.0,
                    zero_division=0,
                )
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        history.append(
            {**params, "cv_mean_score": mean_score, "cv_std_score": std_score}
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is not None:
        best_estimator = model_constructor(best_params)
        best_estimator.fit(X_train, y_train, **(fit_kwargs or {}))
    else:
        best_params = base_params

    return ModelResult(
        name=name,
        estimator=best_estimator,
        best_params=best_params,
        validation_score=float(best_score),
        history=history,
    )


def train_logistic_regression(
    X_train,
    y_train,
    base_params: Dict[str, Any],
    param_grid: Dict[str, Iterable[Any]],
    scoring_cfg: Dict[str, Any],
    cv_cfg: Dict[str, Any],
) -> ModelResult:
    """训练逻辑回归模型，返回 `ModelResult`。"""
    return _train_with_cv(
        name="logistic_regression",
        model_constructor=lambda params: LogisticRegression(**params),
        base_params=base_params,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        scoring_cfg=scoring_cfg,
        cv_cfg=cv_cfg,
    )


def train_xgboost(
    X_train,
    y_train,
    base_params: Dict[str, Any],
    param_grid: Dict[str, Iterable[Any]],
    scoring_cfg: Dict[str, Any],
    cv_cfg: Dict[str, Any],
    fit_kwargs: Optional[Dict[str, Any]],
) -> ModelResult:
    """训练 XGBoost 模型。"""
    return _train_with_cv(
        name="xgboost",
        model_constructor=lambda params: XGBClassifier(**params),
        base_params=base_params,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        scoring_cfg=scoring_cfg,
        cv_cfg=cv_cfg,
        fit_kwargs=fit_kwargs,
    )


def train_lightgbm(
    X_train,
    y_train,
    base_params: Dict[str, Any],
    param_grid: Dict[str, Iterable[Any]],
    scoring_cfg: Dict[str, Any],
    cv_cfg: Dict[str, Any],
    fit_kwargs: Optional[Dict[str, Any]],
) -> ModelResult:
    """训练 LightGBM 模型（按需导入依赖）。"""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("需要安装 lightgbm 库以训练 LightGBM 模型") from exc

    return _train_with_cv(
        name="lightgbm",
        model_constructor=lambda params: LGBMClassifier(**params),
        base_params=base_params,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        scoring_cfg=scoring_cfg,
        cv_cfg=cv_cfg,
        fit_kwargs=fit_kwargs,
    )


def train_catboost(
    X_train,
    y_train,
    base_params: Dict[str, Any],
    param_grid: Dict[str, Iterable[Any]],
    scoring_cfg: Dict[str, Any],
    cv_cfg: Dict[str, Any],
    fit_kwargs: Optional[Dict[str, Any]],
) -> ModelResult:
    """训练 CatBoost 模型，并默认关闭 training verbose。"""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("需要安装 catboost 库以训练 CatBoost 模型") from exc

    return _train_with_cv(
        name="catboost",
        model_constructor=lambda params: CatBoostClassifier(**params),
        base_params=base_params,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        scoring_cfg=scoring_cfg,
        cv_cfg=cv_cfg,
        fit_kwargs={"verbose": False, **(fit_kwargs or {})},
    )


def train_model(
    model_name: str,
    X_train,
    y_train,
    X_valid,
    y_valid,
    config: Dict[str, Any],
) -> ModelResult:
    """按照配置中声明的 `type` 选择具体训练函数。"""
    model_type = config.get("type")
    base_params = config.get("params", {})
    param_grid = config.get("param_grid", {})
    scoring_cfg = config.get("scoring", {})
    cv_cfg = config.get("cv", {})
    fit_kwargs = config.get("fit_kwargs", {})

    if model_type == "logistic_regression":
        return train_logistic_regression(
            X_train,
            y_train,
            base_params,
            param_grid,
            scoring_cfg,
            cv_cfg,
        )
    if model_type == "xgboost":
        return train_xgboost(
            X_train,
            y_train,
            base_params,
            param_grid,
            scoring_cfg,
            cv_cfg,
            fit_kwargs,
        )
    if model_type == "lightgbm":
        return train_lightgbm(
            X_train,
            y_train,
            base_params,
            param_grid,
            scoring_cfg,
            cv_cfg,
            fit_kwargs,
        )
    if model_type == "catboost":
        return train_catboost(
            X_train,
            y_train,
            base_params,
            param_grid,
            scoring_cfg,
            cv_cfg,
            fit_kwargs,
        )
    raise ValueError(f"不支持的模型类型: {model_type}")

