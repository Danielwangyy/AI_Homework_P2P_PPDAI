"""违约标签生成模块。

根据新的业务逻辑：
1. 优先使用LCIS表的"标当前状态"作为因变量
2. 使用LP表的"还款状态"作为补充
3. 只保留"已还清"（标签=0）和"逾期中"（标签=1）作为有效的因变量
4. 排除"正常还款中"的记录（还未产生结果）
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _label_from_lcis(lcis: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """从LCIS表的"标当前状态"生成标签。
    
    规则：
    - "已还清" → 标签 = 0 (正常还款)
    - "逾期中" → 标签 = 1 (违约)
    - "正常还款中" → 排除（还未产生结果）
    - 缺失值 → 排除
    
    返回：
    - DataFrame，包含 id_col 和 lcis_label 列，以及 valid 列（表示是否为有效标签）
    """
    if "标当前状态" not in lcis.columns:
        return pd.DataFrame(columns=[id_col, "lcis_label", "lcis_valid"])
    
    # 获取每个ListingId的最新状态（使用recorddate或取最后一条记录）
    lcis_subset = lcis[[id_col, "标当前状态"]].copy()
    
    # 如果有多条记录，取最后一条（通常recorddate最新的）
    if "recorddate" in lcis.columns:
        lcis_subset = lcis[[id_col, "标当前状态", "recorddate"]].copy()
        lcis_subset["recorddate"] = pd.to_datetime(lcis_subset["recorddate"], errors="coerce")
        # 按ListingId分组，取recorddate最新的记录
        lcis_latest = lcis_subset.sort_values("recorddate").groupby(id_col).tail(1)
    else:
        # 如果没有recorddate，取最后一条记录
        lcis_latest = lcis_subset.groupby(id_col).tail(1)
    
    # 生成标签
    def map_status_to_label(status):
        """将状态映射为标签"""
        if pd.isna(status):
            return np.nan, False  # 缺失值，无效
        status_str = str(status).strip()
        if status_str == "已还清":
            return 0, True  # 正常还款，有效标签
        elif status_str == "逾期中":
            return 1, True  # 违约，有效标签
        elif status_str == "正常还款中":
            return np.nan, False  # 还未产生结果，排除
        else:
            return np.nan, False  # 其他状态，排除
    
    labels = []
    for _, row in lcis_latest.iterrows():
        listing_id = row[id_col]
        status = row["标当前状态"]
        label, valid = map_status_to_label(status)
        labels.append({
            id_col: listing_id,
            "lcis_label": label,
            "lcis_valid": valid
        })
    
    return pd.DataFrame(labels)


def _label_from_lp(lp: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """从LP表的"还款状态"生成标签。
    
    规则：
    - "已还清" → 标签 = 0 (正常还款)
    - "逾期中" → 标签 = 1 (违约)
    - "正常还款中" → 排除（还未产生结果）
    - 缺失值 → 排除
    
    对于每个ListingId，如果有任何一期是"逾期中"，则标记为1（违约）
    如果所有期都是"已还清"，则标记为0（正常还款）
    如果有"正常还款中"但没有"逾期中"，则排除（还未产生结果）
    
    返回：
    - DataFrame，包含 id_col 和 lp_label 列，以及 valid 列（表示是否为有效标签）
    """
    if "还款状态" not in lp.columns:
        return pd.DataFrame(columns=[id_col, "lp_label", "lp_valid"])
    
    lp_subset = lp[[id_col, "还款状态"]].copy()
    
    # 按ListingId分组，检查还款状态
    labels = []
    for listing_id, group in lp_subset.groupby(id_col):
        statuses = group["还款状态"].dropna().unique()
        
        # 检查是否有"逾期中"
        has_overdue = any(str(s).strip() == "逾期中" for s in statuses)
        # 检查是否有"已还清"
        has_paid_off = any(str(s).strip() == "已还清" for s in statuses)
        # 检查是否有"正常还款中"
        has_repaying = any(str(s).strip() == "正常还款中" for s in statuses)
        
        if has_overdue:
            # 有逾期，标记为违约（标签=1）
            labels.append({
                id_col: listing_id,
                "lp_label": 1,
                "lp_valid": True
            })
        elif has_paid_off and not has_repaying:
            # 全部已还清且没有正常还款中，标记为正常（标签=0）
            labels.append({
                id_col: listing_id,
                "lp_label": 0,
                "lp_valid": True
            })
        elif has_paid_off and has_repaying:
            # 部分已还清但还有正常还款中，排除（还未产生结果）
            # 即使部分期已还清，但只要还有正常还款中的期，说明还未完全结束
            labels.append({
                id_col: listing_id,
                "lp_label": np.nan,
                "lp_valid": False
            })
        elif has_repaying:
            # 只有正常还款中，排除（还未产生结果）
            labels.append({
                id_col: listing_id,
                "lp_label": np.nan,
                "lp_valid": False
            })
        else:
            # 其他情况（可能只有缺失值或其他未知状态），排除
            labels.append({
                id_col: listing_id,
                "lp_label": np.nan,
                "lp_valid": False
            })
    
    return pd.DataFrame(labels)


def generate_labels(
    data: Dict[str, pd.DataFrame],
    id_col: str = "ListingId",
) -> pd.DataFrame:
    """生成标签：优先使用LCIS表的标当前状态，LP表的还款状态作为补充。
    
    规则：
    1. 优先使用LCIS表的"标当前状态"
    2. 如果LCIS没有有效标签，则使用LP表的"还款状态"作为补充
    3. 只保留"已还清"（标签=0）和"逾期中"（标签=1）作为有效的因变量
    4. 排除"正常还款中"的记录（还未产生结果）
    
    参数：
    - data: 包含lc、lp、lcis的字典
    - id_col: ID列名
    
    返回：
    - DataFrame，包含 id_col、label、lcis_label、lp_label 列
      以及 is_valid 列（表示是否为有效标签，True表示可以用于建模）
    """
    # 从LCIS表生成标签
    lcis_labels = _label_from_lcis(data["lcis"], id_col)
    
    # 从LP表生成标签
    lp_labels = _label_from_lp(data["lp"], id_col)
    
    # 获取所有可能的ListingId（从LC表）
    lc_ids = data["lc"][id_col].dropna().unique() if id_col in data["lc"].columns else []
    all_ids = pd.Index(lc_ids)
    
    # 合并标签
    labels = pd.DataFrame({id_col: all_ids})
    
    # 合并LCIS标签
    if not lcis_labels.empty:
        labels = labels.merge(
            lcis_labels[[id_col, "lcis_label", "lcis_valid"]],
            on=id_col,
            how="left"
        )
    else:
        labels["lcis_label"] = np.nan
        labels["lcis_valid"] = False
    
    # 合并LP标签
    if not lp_labels.empty:
        labels = labels.merge(
            lp_labels[[id_col, "lp_label", "lp_valid"]],
            on=id_col,
            how="left"
        )
    else:
        labels["lp_label"] = np.nan
        labels["lp_valid"] = False
    
    # 生成最终标签：优先使用LCIS，如果没有有效标签则使用LP
    def get_final_label(row):
        """获取最终标签"""
        # 优先使用LCIS标签
        if pd.notna(row["lcis_label"]) and row.get("lcis_valid", False):
            return row["lcis_label"], True
        # 如果LCIS没有有效标签，使用LP标签
        elif pd.notna(row["lp_label"]) and row.get("lp_valid", False):
            return row["lp_label"], True
        # 都没有有效标签，返回缺失
        else:
            return np.nan, False
    
    labels[["label", "is_valid"]] = labels.apply(
        lambda row: pd.Series(get_final_label(row)),
        axis=1
    )
    
    # 确保标签为整数类型（有效标签）或NaN（无效标签）
    labels["label"] = labels["label"].astype("Int64")  # 使用可空整数类型
    
    return labels

