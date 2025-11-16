"""原始数据加载模块。

与 `loaders.py` 不同，这里专注于“同时读取三张表并返回字典”，
方便上层流水线一次性拿到所有数据。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

DATE_COLUMNS_LC = ["借款成功日期"]
DATE_COLUMNS_LP = ["到期日期", "还款日期", "recorddate"]
DATE_COLUMNS_LCIS = [
    "借款成功日期",
    "上次还款日期",
    "下次计划还款日期",
    "recorddate",
]


def _load_csv(path: Path, date_columns: list[str] | None = None) -> pd.DataFrame:
    """读取 CSV 并按需解析日期列。"""
    df = pd.read_csv(path)
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_raw_data(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """读取 LC、LP、LCIS 数据，并以字典形式返回。"""
    data = {
        "lc": _load_csv(raw_dir / "LC.csv", DATE_COLUMNS_LC),
        "lp": _load_csv(raw_dir / "LP.csv", DATE_COLUMNS_LP),
        "lcis": _load_csv(raw_dir / "LCIS.csv", DATE_COLUMNS_LCIS),
    }
    return data

