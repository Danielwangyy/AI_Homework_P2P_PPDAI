"""违约标签生成模块。

LCIS 表（标状态）和 LP 表（还款记录）共同决定最终的违约标签。
为帮助新同学理解，这里拆成两个子函数：分别计算 “借款状态” 与 “还款表现”，
最后再合并。
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def _label_from_lcis(lcis: pd.DataFrame, id_col: str) -> pd.Series:
    """依靠 LCIS 表中的“标当前状态”判断是否逾期。"""
    if "标当前状态" not in lcis.columns:
        return pd.Series(dtype="int64")
    status = lcis[[id_col, "标当前状态"]].copy()
    status["标当前状态"] = (
        status["标当前状态"].astype(str).str.contains("逾期", na=False).astype(int)
    )
    return status.groupby(id_col)["标当前状态"].max()


def _label_from_lp(lp: pd.DataFrame, id_col: str) -> pd.Series:
    """根据 LP 还款明细推断是否逾期。

    - 还款状态为 2：表示逾期；
    - 正常已还：比较实际还款日期是否晚于到期日；
    - 未还：利用 recorddate 判断是否已经逾期。
    """
    lp_required = {"还款状态", "到期日期", "还款日期", "recorddate"}
    if not lp_required.issubset(set(lp.columns)):
        raise ValueError("LP 表缺少标签计算所需字段")

    temp = lp[[id_col, "还款状态", "到期日期", "还款日期", "recorddate"]].copy()
    temp["到期日期"] = pd.to_datetime(temp["到期日期"], errors="coerce")
    temp["还款日期"] = pd.to_datetime(temp["还款日期"], errors="coerce")
    temp["recorddate"] = pd.to_datetime(temp["recorddate"], errors="coerce")

    temp["lp_overdue"] = 0
    temp.loc[temp["还款状态"] == 2, "lp_overdue"] = 1

    paid_mask = temp["还款状态"].isin([1, 2]) & temp["还款日期"].notna()
    temp.loc[paid_mask & (temp["还款日期"] > temp["到期日期"]), "lp_overdue"] = 1

    unpaid_mask = temp["还款状态"].isin([0, 4]) & temp["到期日期"].notna()
    temp.loc[unpaid_mask & (temp["recorddate"] > temp["到期日期"]), "lp_overdue"] = 1

    return temp.groupby(id_col)["lp_overdue"].max()


def generate_labels(
    data: Dict[str, pd.DataFrame],
    id_col: str = "ListingId",
) -> pd.DataFrame:
    """融合 LCIS 和 LP 标签逻辑，返回统一的标签表。"""
    lcis_labels = _label_from_lcis(data["lcis"], id_col)
    lp_labels = _label_from_lp(data["lp"], id_col)

    # 取出 LC 表中出现过的全部 ListingId，防止丢失样本
    lc_ids = data["lc"][id_col].dropna().unique() if id_col in data["lc"].columns else []
    all_ids = (
        pd.Index(lc_ids)
        .union(lp_labels.index)
        .union(lcis_labels.index)
    )

    # 汇总各来源标签，并以 LP 标签优先（没有 LP 时回退到 LCIS）
    labels = pd.DataFrame({id_col: all_ids})
    labels = labels.merge(lp_labels.rename("lp_label"), left_on=id_col, right_index=True, how="left")
    labels = labels.merge(lcis_labels.rename("lcis_label"), left_on=id_col, right_index=True, how="left")
    labels["lcis_label"] = labels["lcis_label"].fillna(0)
    labels["label"] = labels["lp_label"].fillna(labels["lcis_label"])
    labels["label"] = labels["label"].fillna(0).astype(int)
    labels["lp_label"] = labels["lp_label"].fillna(0).astype(int)
    labels["lcis_label"] = labels["lcis_label"].astype(int)
    return labels

