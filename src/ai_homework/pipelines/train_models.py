"""模型训练流水线。

主要职责：
1. 读取经过预处理的数据；
2. 根据配置训练多种模型（逻辑回归、XGBoost、LightGBM、CatBoost）；
3. 进行阈值调优、评估与可视化；
4. 将模型、指标、SHAP 分析等产物写入 `outputs/`。

文件较长，建议初学者先浏览 `run_pipeline` 最后一段的执行顺序，
再回头查看辅助函数的具体实现。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import fbeta_score

from ..evaluation.metrics import (
    compute_classification_metrics,
    metrics_to_dataframe,
    plot_confusion_matrix,
    plot_roc_curve,
)
from ..models.training import train_model
from ..utils.config import load_yaml
from ..utils.data_io import load_dataset
from ..utils.logger import get_logger, setup_logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LIBOMP_PATH = Path("/opt/homebrew/opt/libomp/lib")
if LIBOMP_PATH.exists():
    current = os.environ.get("DYLD_LIBRARY_PATH", "")
    path_str = str(LIBOMP_PATH)
    if path_str not in current:
        os.environ["DYLD_LIBRARY_PATH"] = f"{path_str}:{current}" if current else path_str
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def _resolve_path(path_str: str) -> Path:
    """将配置中的相对路径转换为绝对路径。"""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def parse_args() -> argparse.Namespace:
    """命令行参数解析，便于单独运行本脚本。"""
    parser = argparse.ArgumentParser(description="Train default risk models")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "model_training.yaml"),
        help="模型训练配置文件路径",
    )
    return parser.parse_args()


def _predict_with_proba(estimator, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """统一处理模型预测。

    某些模型只提供 `predict_proba`，也有少数只提供 `predict`。
    该函数负责屏蔽差异，并返回二分类标签与概率值。
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "predict"):
        # 部分模型仅返回标签，无法绘制 ROC
        proba = estimator.predict(X)
    else:
        raise AttributeError("模型缺少 predict/predict_proba 方法")
    preds = (proba >= threshold).astype(int)
    return preds, proba


def _apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """按照给定阈值将概率转成 0/1 标签。"""
    return (proba >= threshold).astype(int)


def _tune_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cfg: Dict[str, float],
) -> Tuple[float, Dict[str, float], list[Dict[str, float]]]:
    """通过网格搜索找到最优阈值。

    - `strategy=cost`：最小化 FN/FP 对应的成本；
    - `strategy=f_beta`：最大化指定 beta 的 F-score。
    返回最佳阈值、其对应指标以及完整搜索记录。
    """
    if len(y_true) == 0:
        return float(cfg.get("default", 0.5)), {}, []

    strategy = str(cfg.get("strategy", "cost")).lower()
    beta = float(cfg.get("beta", 2.0))
    fn_cost = float(cfg.get("fn_cost", 5.0))
    fp_cost = float(cfg.get("fp_cost", 1.0))

    grid = cfg.get("grid")
    if not grid:
        grid_size = int(cfg.get("grid_size", 201))
        min_threshold = float(cfg.get("min_threshold", 0.05))
        max_threshold = float(cfg.get("max_threshold", 0.95))
        grid = np.linspace(min_threshold, max_threshold, grid_size)

    best_threshold = float(cfg.get("default", 0.5))
    best_score = -np.inf
    search_records: list[Dict[str, float]] = []

    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    for threshold in grid:
        threshold = float(threshold)
        preds = _apply_threshold(y_score_arr, threshold)
        fn = float(np.sum((preds == 0) & (y_true_arr == 1)))
        fp = float(np.sum((preds == 1) & (y_true_arr == 0)))

        if strategy == "cost":
            cost = fn_cost * fn + fp_cost * fp
            score = -cost / max(len(y_true_arr), 1)
        elif strategy == "f_beta":
            score = fbeta_score(y_true_arr, preds, beta=beta, zero_division=0)
        else:
            score = fbeta_score(y_true_arr, preds, beta=1.0, zero_division=0)

        search_records.append(
            {
                "threshold": threshold,
                "objective_score": float(score),
                "fn": fn,
                "fp": fp,
            }
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    best_preds = _apply_threshold(y_score_arr, best_threshold)
    best_metrics = compute_classification_metrics(y_true_arr, best_preds, y_score_arr)
    best_metrics["threshold"] = best_threshold
    best_metrics["objective_score"] = float(best_score)
    best_metrics["fn_cost"] = fn_cost
    best_metrics["fp_cost"] = fp_cost
    return best_threshold, best_metrics, search_records


def _configure_matplotlib_font(logger) -> None:
    """尝试加载常见中文字体，避免图形出现乱码。"""
    preferred_font_paths = [
        Path.home() / "Library" / "Fonts" / "NotoSansSC-Regular.otf",
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ]
    preferred_font_names = [
        "Noto Sans SC",
        "PingFang SC",
        "STHeiti",
        "Microsoft YaHei",
        "Heiti SC",
    ]

    discovered_names: list[str] = []
    for font_path in preferred_font_paths:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
                props = font_manager.FontProperties(fname=str(font_path))
                name = props.get_name()
                if name:
                    discovered_names.append(name)
            except Exception as exc:
                logger.warning("加载字体 %s 失败：%s", font_path, exc)

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in discovered_names + preferred_font_names:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            logger.info("使用 Matplotlib 字体: %s", font_name)
            return

    logger.warning("未找到可用于中文显示的字体，将继续使用默认字体（可能出现缺字）")


FEATURE_DISPLAY_NAME_MAP: Dict[str, str] = {
    "loan_amount": "借款金额",
    "loan_term": "借款期限",
    "interest_rate": "借款利率",
    "loan_type": "借款类型",
    "rating": "初始评级",
    "first_loan_flag": "是否首标",
    "user_age": "借款人年龄",
    "user_gender": "借款人性别",
    "phone_verified": "手机认证",
    "hukou_verified": "户口认证",
    "video_verified": "视频认证",
    "education_verified": "学历认证",
    "credit_verified": "征信认证",
    "taobao_verified": "淘宝认证",
    "history_total_loans": "历史成功借款次数",
    "history_total_amount": "历史成功借款金额",
    "outstanding_principal": "总待还本金",
    "history_normal_terms": "历史正常还款期数",
    "history_overdue_terms": "历史逾期还款期数",
    "history_repay_ratio": "历史还款率",
    "history_overdue_rate": "历史逾期率",
    "loan_amount_per_term": "单期借款金额",
    "history_avg_loan_amount": "历史平均借款金额",
    "loan_amount_ratio_to_history_avg": "借款金额/历史平均",
    "history_avg_term_payment": "历史平均期还金额",
    "loan_amount_to_history_amount_ratio": "借款金额/历史总额",
    "outstanding_to_history_amount_ratio": "待还本金/历史总额",
    "rating_numeric": "评级数值化",
    "loan_amount_rating_interaction": "借款金额×评级",
    "loan_term_rating_interaction": "借款期限×评级",
    "loan_amount_history_repay_ratio": "借款金额×历史还款率",
    "loan_term_history_overdue_rate": "借款期限×历史逾期率",
    "loan_date_year": "借款年份",
    "loan_date_quarter": "借款季度",
    "loan_date_month": "借款月份",
    "loan_date_weekday": "借款星期",
}

CATEGORY_PREFIX_DISPLAY: Dict[str, str] = {
    "loan_type": "借款类型",
    "rating": "初始评级",
    "user_gender": "借款人性别",
    "loan_date_year": "借款年份",
    "loan_date_quarter": "借款季度",
    "loan_date_month": "借款月份",
    "loan_date_weekday": "借款星期",
}

WEEKDAY_NAME_MAP: Dict[str, str] = {
    "0": "周一",
    "1": "周二",
    "2": "周三",
    "3": "周四",
    "4": "周五",
    "5": "周六",
    "6": "周日",
}

UNKNOWN_CATEGORY_VALUES = {"", "nan", "none", "null", "unknown", "-1", "-1.0", "NaN", "None", "UNK"}


def _format_category_value(base: str, raw_value: str) -> str:
    value = str(raw_value).strip()
    lower_value = value.lower()
    if value in UNKNOWN_CATEGORY_VALUES or lower_value in UNKNOWN_CATEGORY_VALUES:
        return "未知"

    if base == "loan_date_year":
        return f"{value}年"
    if base == "loan_date_quarter":
        return f"第{value}季度"
    if base == "loan_date_month":
        return f"{value}月"
    if base == "loan_date_weekday":
        return WEEKDAY_NAME_MAP.get(value, f"周{value}")

    return value.replace("__", "/").replace("_", " ")


def _build_feature_display_map(columns: Iterable[str]) -> Dict[str, str]:
    display_map: Dict[str, str] = {}
    used_names: set[str] = set()

    for col in columns:
        display_name = FEATURE_DISPLAY_NAME_MAP.get(col)

        if display_name is None:
            for base, prefix_display in CATEGORY_PREFIX_DISPLAY.items():
                prefix = f"{base}_"
                if col.startswith(prefix):
                    raw_value = col[len(prefix) :]
                    formatted_value = _format_category_value(base, raw_value)
                    display_name = f"{prefix_display}={formatted_value}"
                    break

        if display_name is None:
            display_name = FEATURE_DISPLAY_NAME_MAP.get(col.split("__")[0], col)

        original_display_name = display_name
        duplicate_index = 1
        while display_name in used_names:
            duplicate_index += 1
            display_name = f"{original_display_name} ({duplicate_index})"

        display_map[col] = display_name
        used_names.add(display_name)

    return display_map


MODEL_DISPLAY_NAME_MAP: Dict[str, str] = {
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "gbdt": "GBDT",
    "logistic_regression": "Logistic Regression",
}


def _get_model_display_name(model_name: str) -> str:
    key = str(model_name).lower()
    return MODEL_DISPLAY_NAME_MAP.get(key, model_name)


def _export_shap_values(
    estimator,
    X_reference: pd.DataFrame,
    model_name: str,
    figures_dir: Path,
    tables_dir: Path,
    shap_cfg: Dict[str, float],
    logger,
) -> None:
    """根据 SHAP 值解释模型。

    只要 `shap` 库已安装、且模型支持树模型解释，就会输出：
    - Top 特征重要性条形图；
    - 每个样本的 SHAP 数值表格。
    """
    try:
        import shap
    except ImportError:
        logger.warning("缺少 shap 库，跳过模型 %s 的 SHAP 分析", model_name)
        return

    sample_size = int(shap_cfg.get("sample_size", 2000))
    if sample_size > 0 and len(X_reference) > sample_size:
        sample = X_reference.sample(sample_size, random_state=42)
    else:
        sample = X_reference.copy()

    feature_display_map = _build_feature_display_map(sample.columns)
    display_sample = sample.rename(columns=feature_display_map)
    display_columns = [feature_display_map.get(col, col) for col in sample.columns]

    logger.info("生成模型 %s 的 SHAP 分析样本量=%s", model_name, len(sample))
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap.summary_plot(
            shap_values,
            display_sample,
            show=False,
            plot_type="bar",
            max_display=min(25, sample.shape[1]),
        )
        model_display_name = _get_model_display_name(model_name)
        ax = plt.gca()
        ax.set_title(f"{model_display_name} SHAP 特征影响力", fontweight="bold")
        fig_path = figures_dir / f"shap_summary_{model_name}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        shap_table = pd.DataFrame(shap_values, columns=display_columns)
        shap_table.insert(0, "model", model_name)
        shap_table.to_csv(
            tables_dir / f"shap_values_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logger.info("SHAP 结果已输出：%s, %s", fig_path, tables_dir / f"shap_values_{model_name}.csv")
    except (ValueError, AttributeError, TypeError) as e:
        logger.warning("模型 %s 的 SHAP 分析失败（可能是版本兼容性问题）：%s", model_name, str(e))
        logger.warning("跳过 SHAP 分析，继续处理其他模型")
    except Exception as e:
        logger.warning("模型 %s 的 SHAP 分析失败：%s", model_name, str(e))
        logger.warning("跳过 SHAP 分析，继续处理其他模型")


def run_pipeline(config_path: Path) -> None:
    """主入口：按配置执行模型训练与评估。

    执行顺序概览：
    1. 解析配置与基础目录；
    2. 初始化日志、字体等全局设置；
    3. 读取训练/验证/测试数据；
    4. 遍历每个模型配置：
       4.1 调整类别权重参数；
       4.2 调用 `train_model` 训练并做交叉验证；
       4.3 在验证集上调阈值，记录最佳阈值与指标；
       4.4 保存模型、调参历史、阈值搜索、预测结果；
       4.5 生成混淆矩阵、ROC 曲线和（可选）SHAP 图表；
    5. 汇总所有模型的指标、阈值摘要，写入报告目录；
    6. 日志中会输出每一步的大致耗时与产物路径。
    """
    cfg = load_yaml(config_path)

    logs_dir = _resolve_path(cfg["output"]["logs_dir"])
    setup_logger(logs_dir, name="model_training")
    logger = get_logger()

    run_label_raw = cfg.get("experiment_label") or cfg.get("run_label") or cfg.get("experiment_name") or config_path.stem
    run_label = re.sub(r"[^0-9A-Za-z_-]+", "_", str(run_label_raw).strip()) or "run"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_identifier = f"{run_label}_{run_timestamp}"
    logger.info("本次训练运行标识：%s", run_identifier)

    _configure_matplotlib_font(logger)

    logger.info("开始模型训练流程")

    data_cfg = cfg["data"]
    id_col = data_cfg["id_column"]
    label_col = data_cfg["label_column"]

    # 读取准备好的训练/验证/测试数据
    train_path = _resolve_path(data_cfg["train_path"])
    valid_path = _resolve_path(data_cfg["valid_path"])
    test_path = _resolve_path(data_cfg["test_path"])

    X_train, y_train, _ = load_dataset(train_path, id_col, label_col)
    X_valid, y_valid, _ = load_dataset(valid_path, id_col, label_col)
    X_test, y_test, test_ids = load_dataset(test_path, id_col, label_col)

    logger.info("数据加载完成：train=%s, valid=%s, test=%s", len(X_train), len(X_valid), len(X_test))

    models_cfg: Dict[str, Dict] = cfg["models"]
    metrics_cfg = cfg.get("metrics", [])
    cv_default = cfg.get("cross_validation", {})
    scoring_default = cfg.get("scoring", {"type": "f_beta", "beta": 1.0})
    threshold_default = cfg.get("threshold_tuning", {"strategy": "cost", "fn_cost": 5.0, "fp_cost": 1.0})

    evaluation_cfg = cfg.get("evaluation", {})
    shap_cfg = evaluation_cfg.get("shap", {})

    metrics_records = []
    threshold_summaries: list[Dict[str, float]] = []
    artifacts_dir = _resolve_path(cfg["output"]["artifacts_dir"])
    models_dir = _resolve_path(cfg["output"]["models_dir"])
    figures_dir = _resolve_path(cfg["output"]["figures_dir"])
    tables_dir = _resolve_path(cfg["output"]["tables_dir"])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 遍历配置文件中声明的每个模型
    for model_name, model_cfg in models_cfg.items():
        logger.info("训练模型：%s", model_name)
        merged_cfg = dict(model_cfg)
        merged_cfg.setdefault("cv", cv_default)
        merged_cfg.setdefault("scoring", scoring_default)
        threshold_cfg = merged_cfg.pop("threshold_tuning", threshold_default)

        params = merged_cfg.setdefault("params", {})
        pos_count = float(np.sum(y_train == 1))
        neg_count = float(np.sum(y_train == 0))
        if pos_count > 0:
            cost_ratio = float(threshold_cfg.get("fn_cost", 5.0)) / max(float(threshold_cfg.get("fp_cost", 1.0)), 1e-6)
            scale_weight = max((neg_count / pos_count) * cost_ratio, 1.0)
            if model_name in {"xgboost", "lightgbm"}:
                params.setdefault("scale_pos_weight", scale_weight)
            if model_name == "catboost":
                params.setdefault("class_weights", [1.0, scale_weight])

        try:
            result = train_model(
                model_name,
                X_train,
                y_train,
                X_valid,
                y_valid,
                merged_cfg,
            )
        except ImportError as exc:
            logger.error("模型 %s 训练失败，缺少依赖：%s", model_name, exc)
            continue
        logger.info("模型 %s 验证集最佳F1=%.4f，参数=%s", model_name, result.validation_score, result.best_params)

        estimator = result.estimator
        if estimator is None:
            logger.error("模型 %s 训练失败，跳过后续评估", model_name)
            continue

        _, train_proba = _predict_with_proba(estimator, X_train, threshold=0.5)
        _, valid_proba = _predict_with_proba(estimator, X_valid, threshold=0.5)
        has_test = len(X_test) > 0
        if has_test:
            _, test_proba = _predict_with_proba(estimator, X_test, threshold=0.5)

        best_threshold, threshold_metrics, threshold_history = _tune_threshold(
            y_valid,
            valid_proba,
            threshold_cfg,
        )
        # 小贴士：
        # - “阈值调优”可以理解为：当模型给出“预测概率”后，我们要决定多少分以上算“坏客户”。
        #   这个分数（阈值）高一些，坏客户更难被识别，但好客户也更安全；阈值低一些，能抓住
        #   更多坏客户，但也可能误伤好客户。
        # - 通过遍历不同阈值，并计算成本或 F-score，我们就能找到更符合业务目标的平衡点。
        # - `_tune_threshold` 会返回最佳阈值、对应指标以及整个搜索过程，方便我们事后复盘分析。
        result.best_threshold = best_threshold
        cost_weights = {"fn_cost": threshold_cfg.get("fn_cost", 5.0), "fp_cost": threshold_cfg.get("fp_cost", 1.0)}

        train_preds = _apply_threshold(train_proba, best_threshold)
        valid_preds = _apply_threshold(valid_proba, best_threshold)
        if has_test:
            test_preds = _apply_threshold(test_proba, best_threshold)

        train_metrics = compute_classification_metrics(y_train, train_preds, train_proba, cost_weights=cost_weights)
        valid_metrics = compute_classification_metrics(y_valid, valid_preds, valid_proba, cost_weights=cost_weights)
        train_metrics["threshold"] = best_threshold
        valid_metrics["threshold"] = best_threshold
        valid_metrics["threshold_objective"] = threshold_metrics.get("objective_score")
        valid_metrics["threshold_strategy"] = threshold_cfg.get("strategy", "cost")
        result.validation_score = float(valid_metrics.get("f1", result.validation_score))
        if has_test:
            test_metrics = compute_classification_metrics(y_test, test_preds, test_proba, cost_weights=cost_weights)
            test_metrics["threshold"] = best_threshold
        else:
            test_metrics = {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "roc_auc": float("nan")}

        # 记录三份数据集的指标，后续会被汇总成一张表
        metrics_records.append(metrics_to_dataframe(train_metrics, model_name, "train"))
        metrics_records.append(metrics_to_dataframe(valid_metrics, model_name, "valid"))
        metrics_records.append(metrics_to_dataframe(test_metrics, model_name, "test"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"{model_name}_{timestamp}.joblib"
        joblib.dump(estimator, model_path)
        logger.info("模型已保存：%s", model_path)

        history_df = pd.DataFrame(result.history)
        history_df.to_csv(
            artifacts_dir / f"{model_name}_tuning_history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        threshold_search_path = None
        if threshold_history:
            threshold_search_path = artifacts_dir / f"{model_name}_threshold_search.csv"
            pd.DataFrame(threshold_history).to_csv(
                threshold_search_path,
                index=False,
                encoding="utf-8-sig",
            )

        if has_test:
            cm_filename = f"confusion_matrix_{model_name}_{run_identifier}.png"
            plot_confusion_matrix(
                y_test,
                test_preds,
                figures_dir / cm_filename,
                title=f"{model_name} - Confusion Matrix",
            )

        if has_test and not np.isnan(test_metrics.get("roc_auc", np.nan)):
            plot_roc_curve(
                y_test,
                test_proba,
                figures_dir / f"roc_curve_{model_name}.png",
                title=f"{model_name} - ROC Curve",
            )

        # 保存测试集预测结果（用于后续分析）
        if has_test:
            predictions_df = pd.DataFrame({
                id_col: test_ids,
                "y_true": y_test,
                "y_pred": test_preds,
                "y_score": test_proba,
            })
            predictions_df.to_parquet(artifacts_dir / f"test_predictions_{model_name}.parquet", index=False)

        metadata = {
            "model": model_name,
            "best_params": result.best_params,
            "best_threshold": result.best_threshold,
            "threshold_metrics": threshold_metrics,
            "threshold_config": threshold_cfg,
            "threshold_search_path": str(threshold_search_path) if threshold_search_path else None,
            "metrics": {
                "train": train_metrics,
                "valid": valid_metrics,
                "test": test_metrics,
            },
            "timestamp": timestamp,
            "run_identifier": run_identifier,
            "run_label": run_label,
            "config": model_cfg,
        }
        with (artifacts_dir / f"summary_{model_name}.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 额外导出特征重要性（仅对支持的模型）
        try:
            if model_name in {"xgboost", "lightgbm", "catboost"} and hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                fi_df = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": importances,
                }).sort_values("importance", ascending=False)
                fi_path = tables_dir / f"feature_importance_{model_name}.csv"
                fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")
                logger.info("特征重要性已输出：%s", fi_path)
        except Exception as e:
            logger.warning("导出特征重要性失败：%s", e)

        if shap_cfg.get("enabled", False) and model_name in {"xgboost", "lightgbm", "catboost"}:
            reference_df = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
            _export_shap_values(
                estimator,
                reference_df,
                model_name,
                figures_dir,
                tables_dir,
                shap_cfg,
                logger,
            )

        threshold_summaries.append(
            {
                "model": model_name,
                "threshold": best_threshold,
                "strategy": threshold_cfg.get("strategy", "cost"),
                "fn_cost": cost_weights["fn_cost"],
                "fp_cost": cost_weights["fp_cost"],
                "valid_expected_cost": valid_metrics.get("expected_cost"),
                "valid_recall": valid_metrics.get("recall"),
                "valid_precision": valid_metrics.get("precision"),
                "valid_f1": valid_metrics.get("f1"),
            }
        )

    if metrics_records:
        metrics_table = pd.concat(metrics_records, ignore_index=True)
        metrics_table.to_csv(
            tables_dir / "model_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metrics_table.to_json(tables_dir / "model_metrics.json", orient="records", force_ascii=False)
        logger.info("模型评估指标已写入 outputs/reports/tables/model_metrics.csv")

    if threshold_summaries:
        threshold_df = pd.DataFrame(threshold_summaries)
        threshold_df.to_csv(
            tables_dir / "threshold_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logger.info("阈值调优摘要已写入 outputs/reports/tables/threshold_summary.csv")

    logger.info("模型训练流程结束")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.config))
