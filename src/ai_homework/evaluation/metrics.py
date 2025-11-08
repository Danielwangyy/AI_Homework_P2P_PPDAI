"""模型评估指标与可视化工具。

所有函数都尽量做到“输入即输出”，避免隐藏状态，便于单元测试。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    cost_weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """计算常用分类指标，并可选纳入成本信息。"""
    results = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        results["roc_auc"] = metrics.roc_auc_score(y_true, y_score)
    else:
        results["roc_auc"] = float("nan")

    if cost_weights is not None:
        fp_cost = float(cost_weights.get("fp_cost", 1.0))
        fn_cost = float(cost_weights.get("fn_cost", 1.0))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        results["fp"] = fp
        results["fn"] = fn
        results["expected_cost"] = fp_cost * fp + fn_cost * fn

    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """绘制并保存混淆矩阵图像。"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["No-Default", "Default"]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """绘制 ROC 曲线，并在图例中显示 AUC 指标。"""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def metrics_to_dataframe(
    metrics_dict: Dict[str, float],
    model_name: str,
    split: str,
    suffix: str | None = None,
) -> pd.DataFrame:
    """将字典形式的指标转换为 DataFrame，方便汇总与保存。"""
    df = pd.DataFrame([metrics_dict])
    df.insert(0, "model", model_name)
    split_label = split if suffix is None else f"{split}@{suffix}"
    df.insert(1, "split", split_label)
    return df

