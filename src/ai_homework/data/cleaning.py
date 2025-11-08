"""数据清洗与预处理模块。

负责整理 LC 主表中的原始字段，包括：
- 二值字段统一为 0/1；
- 缺失值处理；
- 数值字段截断填补等。

为了让初学者更容易理解，函数内部加入了逐步注释。
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


BINARY_TRUE_KEYWORDS = {"成功", "是", "Y", "YES", "TRUE", "1", "已认证", "已完成"}
BINARY_FALSE_KEYWORDS = {"否", "N", "NO", "FALSE", "0"}


def _normalize_binary(series: pd.Series) -> pd.Series:
    """将任意表示形式的布尔/认证字段转成 0 或 1。"""

    # 统一转换逻辑封装在内部函数，便于 apply 调用
    def mapper(val: object) -> float:
        if pd.isna(val):
            return np.nan
        text = str(val).strip().upper()
        if text in BINARY_TRUE_KEYWORDS:
            return 1.0
        if text in BINARY_FALSE_KEYWORDS:
            return 0.0
        if "成功" in text or text == "是":
            return 1.0
        return 0.0

    return series.apply(mapper)


def clean_lc(df: pd.DataFrame, cfg: Dict[str, Iterable[str]]) -> pd.DataFrame:
    """对 LC 主表执行一系列清洗步骤。

    参数说明：
    - df：原始 LC 数据；
    - cfg：配置文件中关于特征列的约定。
    """
    cleaned = df.copy()

    # 1. 年龄异常值处理：限定在 [18, 70]，超出范围视为缺失
    if "年龄" in cleaned.columns:
        cleaned.loc[~cleaned["年龄"].between(18, 70), "年龄"] = np.nan

    # 2. 二值字段统一为 0/1，避免出现 “是/否” 等字符串
    binary_cols = cfg.get("binary_features", [])
    for col in binary_cols:
        if col in cleaned.columns:
            cleaned[col] = _normalize_binary(cleaned[col])

    # 3. “是否首标” 在原始数据中经常缺失，这里额外做一次兜底处理
    if "是否首标" in cleaned.columns:
        cleaned["是否首标"] = cleaned["是否首标"].fillna("否")
        cleaned["是否首标"] = cleaned["是否首标"].apply(
            lambda x: 1.0 if str(x).strip().upper() in {"是", "1", "Y", "YES"} else 0.0
        )

    # 4. 类别型字段的缺失值统一填充为 “未知”，便于 one-hot 编码
    for col in cfg.get("categorical_features", []):
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("未知")

    # 5. 性别字段常出现空字符串，需要额外处理
    if "性别" in cleaned.columns:
        cleaned["性别"] = cleaned["性别"].replace({np.nan: "未知", "": "未知"})
        cleaned["性别"] = cleaned["性别"].apply(lambda x: x if str(x).strip() else "未知")

    # 6. 数值字段缺失值使用中位数填充，避免极端值影响均值
    numeric_cols = cfg.get("numeric_features", [])
    for col in numeric_cols:
        if col in cleaned.columns:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)

    return cleaned

