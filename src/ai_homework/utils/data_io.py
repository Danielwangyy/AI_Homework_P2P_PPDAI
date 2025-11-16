"""数据加载工具函数。

提供读取处理后数据集的统一接口，避免在业务代码中重复编写
`pd.read_parquet` + 列拆分的逻辑。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_dataset(path: Path, id_column: str, label_column: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """加载处理后的数据集并拆分特征/标签。

    返回值说明：
        X: 特征 DataFrame（不包含 id/label）。
        y: 标签 Series。
        ids: ListingId Series。

    小贴士：
    - 若读取时报错，可检查是否已执行数据准备流水线；
    - 也可以传入自定义 `id_column` / `label_column`，默认兼容配置文件。
    """
    df = pd.read_parquet(path)
    ids = df[id_column]
    y = df[label_column]
    feature_cols = [col for col in df.columns if col not in {id_column, label_column}]
    X = df[feature_cols]
    return X, y, ids

