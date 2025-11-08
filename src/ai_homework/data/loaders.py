"""数据加载模块：封装原始数据读取逻辑。

初学者可以把它理解为“读取原始 CSV 文件的工具箱”，
每个函数都只负责一张表，便于单元测试与重用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "source_data"


def _resolve_path(path: Optional[Path | str], filename: str) -> Path:
    """组合目录与文件名，确保文件存在。"""
    base = Path(path) if path is not None else DEFAULT_RAW_DIR
    full_path = base / filename
    if not full_path.exists():
        raise FileNotFoundError(
            f"未找到文件: {full_path}。请确认已将老师提供的 P2P_PPDAI_DATA "
            "文件夹（含 LC/LP/LCIS 等数据）放入 data/raw/source_data/。"
        )
    return full_path


def load_lc(path: Optional[Path | str] = None, **read_kwargs) -> pd.DataFrame:
    """读取 LC.csv 标的特征数据。

    支持传入自定义参数（编码、分隔符等），否则使用默认编码 utf-8。
    """
    full_path = _resolve_path(path, "LC.csv")
    defaults: Dict = {"encoding": "utf-8"}
    defaults.update(read_kwargs)
    df = pd.read_csv(full_path, **defaults)
    return df


def load_lp(path: Optional[Path | str] = None, parse_dates: bool = True, **read_kwargs) -> pd.DataFrame:
    """读取 LP.csv 还款记录数据，可选解析日期列。"""
    full_path = _resolve_path(path, "LP.csv")
    defaults: Dict = {"encoding": "utf-8"}
    if parse_dates:
        defaults["parse_dates"] = ["到期日期", "还款日期", "recorddate"]
        defaults["infer_datetime_format"] = True
    defaults.update(read_kwargs)
    df = pd.read_csv(full_path, **defaults)
    if parse_dates is False:
        for col in ["到期日期", "还款日期", "recorddate"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_lcis(path: Optional[Path | str] = None, parse_dates: bool = True, **read_kwargs) -> pd.DataFrame:
    """读取 LCIS.csv 投资快照数据。"""
    full_path = _resolve_path(path, "LCIS.csv")
    defaults: Dict = {"encoding": "utf-8"}
    if parse_dates:
        defaults["parse_dates"] = [
            "借款成功日期",
            "上次还款日期",
            "下次计划还款日期",
            "recorddate",
        ]
        defaults["infer_datetime_format"] = True
    defaults.update(read_kwargs)
    df = pd.read_csv(full_path, **defaults)
    if parse_dates is False:
        for col in ["借款成功日期", "上次还款日期", "下次计划还款日期", "recorddate"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


__all__ = ["load_lc", "load_lp", "load_lcis", "DEFAULT_RAW_DIR"]

