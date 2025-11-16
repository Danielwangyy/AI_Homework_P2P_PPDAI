"""数据清洗与预处理模块。

根据数据处理方法Excel规范，严格清洗三张数据表的所有68个维度。
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


# 认证成功的关键词（全部为字符串类型）
SUCCESS_KEYWORDS = {"成功", "成功认证", "已认证", "是", "Y", "YES", "TRUE", "1", "1.0", "已完成", "认证成功"}
# 认证失败的关键词（全部为字符串类型）
FAILURE_KEYWORDS = {"否", "未成功认证", "未认证", "N", "NO", "FALSE", "0", "0.0", "", "未成功"}

LC_CATEGORY_ALLOWED_VALUES = {
    "初始评级": {"A", "B", "C", "D", "E", "F"},
    "借款类型": {"应收安全标", "电商", "APP闪电", "普通", "其他"},
    "是否首标": {"是", "否"},
    "性别": {"男", "女"},
}

LP_CATEGORY_ALLOWED_VALUES = {
    "还款状态": {"0", "1", "2", "3", "4"},
}

LCIS_CATEGORY_ALLOWED_VALUES = {
    "初始评级": {"AAA", "AA", "A", "B", "C", "D", "E", "F"},
    "借款类型": {"应收安全标", "电商", "APP闪电", "普通", "其他"},
    "是否首标": {"是", "否"},
    "性别": {"男", "女"},
    "标当前状态": {"正常还款中", "逾期中", "已还清", "已债转"},
}


def _convert_auth_to_binary(series: pd.Series) -> pd.Series:
    """将认证字段转换为0/1：成功认证=1，未成功认证=0。
    
    根据规范：手机认证、户口认证、视频认证、学历认证、征信认证、淘宝认证
    需要转换为：成功认证=1，未成功认证=0
    """
    def mapper(val: object) -> float:
        if pd.isna(val):
            return 0.0  # 缺失值视为未成功认证
        
        # 转换为字符串进行处理
        val_str = str(val).strip()
        
        # 先检查是否为数值类型（可能是1.0或0.0）
        try:
            val_float = float(val)
            if val_float == 1.0 or val_float == 1:
                return 1.0
            elif val_float == 0.0 or val_float == 0:
                return 0.0
        except (ValueError, TypeError):
            pass
        
        val_str_upper = val_str.upper()
        val_str_normalized = val_str_upper.replace(" ", "")
        
        for keyword in FAILURE_KEYWORDS:
            keyword_upper = keyword.upper()
            keyword_normalized = keyword_upper.replace(" ", "")
            if keyword_normalized == "":
                if val_str_normalized == "":
                    return 0.0
                continue
            if keyword_normalized in val_str_normalized or keyword_upper in val_str_upper:
                return 0.0
        
        for keyword in SUCCESS_KEYWORDS:
            keyword_upper = keyword.upper()
            keyword_normalized = keyword_upper.replace(" ", "")
            if keyword_normalized == "":
                continue
            if keyword_normalized in val_str_normalized or keyword_upper in val_str_upper:
                return 1.0
        
        if ("成功" in val_str and
                "未" not in val_str and
                "無" not in val_str and
                "无" not in val_str and
                "不" not in val_str):
            return 1.0
        
        return 0.0
    
    return series.apply(mapper)


def clean_lc(df: pd.DataFrame, cfg: Dict[str, Iterable[str]] | None = None) -> pd.DataFrame:
    """根据规范清洗LC表。
    
    清洗规则：
    1. 大部分数值字段保持不变
    2. 认证字段（手机认证、户口认证、视频认证、学历认证、征信认证、淘宝认证）：
       成功认证=1，未成功认证=0
    3. 类别字段（初始评级、借款类型、是否首标、性别）保持原值，后续进行one-hot编码
    
    参数说明：
    - df：原始 LC 数据
    - cfg：配置文件中关于特征列的约定（可选，用于兼容性）
    """
    cleaned = df.copy()
    
    if "年龄" in cleaned.columns:
        numeric_age = pd.to_numeric(cleaned["年龄"], errors="coerce")
        out_of_range_mask = numeric_age.notna() & ~numeric_age.between(18, 70)
        affected_count = int(out_of_range_mask.sum())
        cleaned["年龄"] = numeric_age
        if affected_count > 0:
            cleaned.loc[out_of_range_mask, "年龄"] = np.nan
        total_rows = len(cleaned)
        notes = cleaned.attrs.setdefault("cleaning_notes", {})
        notes["年龄_out_of_range"] = {
            "total_rows": total_rows,
            "affected_count": affected_count,
            "affected_rate": round(affected_count / total_rows * 100, 4) if total_rows > 0 else 0.0,
            "min_allowed": 18,
            "max_allowed": 70,
            "action": "set_to_nan",
        }
        cleaned.attrs["cleaning_notes"] = notes
    
    # 1. 认证字段转换：成功认证=1，未成功认证=0
    auth_cols = ["手机认证", "户口认证", "视频认证", "学历认证", "征信认证", "淘宝认证"]
    for col in auth_cols:
        if col in cleaned.columns:
            cleaned[col] = _convert_auth_to_binary(cleaned[col])
    
    # 2. 类别字段保持不变（后续进行one-hot编码）
    # 包括：初始评级、借款类型、是否首标、性别
    # 这些字段在清洗阶段保持原值，不进行填充或转换
    
    # 3. 数值字段保持不变（根据规范，LC表所有数值字段都不变）
    # 包括：ListingId、借款金额、借款期限、借款利率、借款成功日期、年龄、
    #       历史成功借款次数、历史成功借款金额、总待还本金、
    #       历史正常还款期数、历史逾期还款期数
    
    clip_records = []
    for col in ["借款金额", "总待还本金"]:
        if col in cleaned.columns:
            numeric_series = pd.to_numeric(cleaned[col], errors='coerce')
            cleaned[col] = numeric_series
            if numeric_series.notna().any():
                threshold = numeric_series.quantile(0.99)
                if pd.notna(threshold):
                    exceed_mask = numeric_series > threshold
                    count = int(exceed_mask.sum())
                    if count > 0:
                        cleaned.loc[exceed_mask, col] = threshold
                        clip_records.append(
                            {
                                "table": "LC",
                                "column": col,
                                "threshold": float(threshold),
                                "count": count,
                                "method": "P99",
                            }
                        )

    if clip_records:
        existing = cleaned.attrs.get("clip_records", [])
        cleaned.attrs["clip_records"] = existing + clip_records
    
    return cleaned


def clean_lp(df: pd.DataFrame) -> pd.DataFrame:
    """根据规范清洗LP表。
    
    清洗规则：
    1. 还款状态重新赋值：0=正常还款中；1，3=已还清；2，4=逾期中
    2. 其他字段保持不变
    
    参数说明：
    - df：原始 LP 数据
    """
    cleaned = df.copy()
    
    # 还款状态重新赋值
    if "还款状态" in cleaned.columns:
        def map_repayment_status(val):
            """将还款状态映射为新值"""
            if pd.isna(val):
                return np.nan
            val = int(val) if pd.notna(val) else val
            # 0=正常还款中；1，3=已还清；2，4=逾期中
            if val == 0:
                return "正常还款中"
            elif val in [1, 3]:
                return "已还清"
            elif val in [2, 4]:
                return "逾期中"
            else:
                return np.nan  # 未知状态设为缺失
        
        cleaned["还款状态"] = cleaned["还款状态"].apply(map_repayment_status)
    
    # 其他字段保持不变
    return cleaned


def clean_lcis(df: pd.DataFrame) -> pd.DataFrame:
    """根据规范清洗LCIS表。
    
    清洗规则：
    1. 认证字段处理：手机认证、户口认证、视频认证、学历认证、征信认证、淘宝认证
       统一转换为成功认证=1，未成功认证=0（异常值视为未成功认证）
    2. 历史正常还款期数、历史逾期还款期数：以99%分位数值为极大值，超过的取极大值
    3. 标当前状态：只保留'正常还款中'，'逾期中'，'已还清'，其他直接剔除
    4. 上次还款日期：删除非日期项（设为缺失）
    5. 历史成功借款次数、历史成功借款金额、总待还本金：保留缺失值
    6. 其他字段保持不变
    
    参数说明：
    - df：原始 LCIS 数据
    """
    cleaned = df.copy()
    
    # 1. 认证字段处理：全部转换为成功认证=1，未成功认证=0
    # 将非认证状态（婚姻、学历等错位值）统一视为未成功认证
    auth_cols_binary = ["手机认证", "户口认证", "视频认证", "学历认证", "征信认证", "淘宝认证"]
    for col in auth_cols_binary:
        if col in cleaned.columns:
            cleaned[col] = _convert_auth_to_binary(cleaned[col])
    
    # 2. 历史正常还款期数、历史逾期还款期数：以99%分位数值为极大值
    period_cols = ["历史正常还款期数", "历史逾期还款期数"]
    clip_records = []
    for col in period_cols:
        if col in cleaned.columns:
            # 转换为数值类型
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
            # 计算99%分位数
            p99 = cleaned[col].quantile(0.99)
            if pd.notna(p99):
                exceed_mask = cleaned[col] > p99
                count = int(exceed_mask.sum())
                if count > 0:
                    # 超过99%分位数的值替换为99%分位数
                    cleaned.loc[exceed_mask, col] = p99
                    clip_records.append(
                        {
                            "table": "LCIS",
                            "column": col,
                            "threshold": float(p99),
                            "count": count,
                            "method": "P99",
                        }
                    )
    
    invalid_status_summary = None

    # 3. 标当前状态：只保留'正常还款中'，'逾期中'，'已还清'，其他删除
    if "标当前状态" in cleaned.columns:
        valid_statuses = {"正常还款中", "逾期中", "已还清"}

        original_status = cleaned["标当前状态"].copy()
        normalized_status = original_status.apply(_normalize_category_value)
        allowed_status = {str(val).strip() for val in valid_statuses if val is not None}
        invalid_mask = normalized_status.isna() | ~normalized_status.isin(allowed_status)

        if invalid_mask.any():
            invalid_counts = (
                original_status[invalid_mask]
                .fillna("缺失")
                .astype(str)
                .value_counts()
                .to_dict()
            )
            invalid_status_summary = {
                "total_removed": int(invalid_mask.sum()),
                "value_counts": {str(key): int(val) for key, val in invalid_counts.items()},
            }

        def filter_status(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip()
            if val_str in valid_statuses:
                return val_str
            return np.nan  # 其他状态删除（设为缺失）
        
        cleaned["标当前状态"] = cleaned["标当前状态"].apply(filter_status)
    
    # 4. 上次还款日期：删除非日期项
    if "上次还款日期" in cleaned.columns:
        # 尝试转换为日期，失败则设为缺失
        date_series = pd.to_datetime(cleaned["上次还款日期"], errors='coerce')
        cleaned["上次还款日期"] = date_series
    
    # 5. 历史成功借款次数、历史成功借款金额、总待还本金：保留缺失值（不做填充）
    # 这些字段保持原值，不进行填充
    
    # 6. 清除标当前状态为缺失的记录，避免后续流程再行剔除
    if "标当前状态" in cleaned.columns:
        cleaned = cleaned[cleaned["标当前状态"].notna()].copy()
        if invalid_status_summary:
            notes = cleaned.attrs.setdefault("cleaning_notes", {})
            notes["标当前状态_invalid"] = invalid_status_summary
    
    if clip_records:
        existing = cleaned.attrs.get("clip_records", [])
        cleaned.attrs["clip_records"] = existing + clip_records

    return cleaned


def _detect_lc_anomalies(df: pd.DataFrame, cfg: Dict[str, Iterable[str]]) -> Dict[str, int]:
    """检测LC表中的异常值。
    
    参数说明：
    - df：原始LC数据
    - cfg：配置文件中关于特征列的约定
    
    返回：
    - 字典，键为列名，值为异常值数量
    """
    anomalies = {}
    total_rows = len(df)
    
    # 1. 年龄异常值：超出 [18, 70] 范围
    if "年龄" in df.columns:
        age_anomalies = (~df["年龄"].between(18, 70)) & df["年龄"].notna()
        anomalies["年龄_超出范围"] = int(age_anomalies.sum())
    
    # 2. 缺失值统计（排除已经单独处理的列）
    excluded_cols = {"年龄", "借款成功日期", "性别"}
    for col in df.columns:
        if col in excluded_cols:
            continue
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            anomalies[f"{col}_缺失值"] = missing_count
    
    # 3. 空字符串统计（对于非空字符串列，排除性别字段）
    for col in df.columns:
        if col == "性别":
            continue
        if df[col].dtype == 'object':
            empty_str_count = int((df[col] == "").sum())
            if empty_str_count > 0:
                anomalies[f"{col}_空字符串"] = empty_str_count
    
    # 4. 性别字段空值或空字符串（特殊处理）
    if "性别" in df.columns:
        gender_empty = ((df["性别"] == "") | (df["性别"].isna())).sum()
        if gender_empty > 0:
            anomalies["性别_空值或空字符串"] = int(gender_empty)
    
    # 5. 日期字段转换失败
    if "借款成功日期" in df.columns:
        date_col = pd.to_datetime(df["借款成功日期"], errors='coerce')
        invalid_dates = date_col.isna() & df["借款成功日期"].notna()
        if invalid_dates.sum() > 0:
            anomalies["借款成功日期_格式错误"] = int(invalid_dates.sum())
    
    # 6. 数值字段的负数（对于应该为正数的字段）
    positive_numeric_cols = ["借款金额", "借款期限", "借款利率", "历史成功借款次数", 
                              "历史成功借款金额", "总待还本金", "历史正常还款期数",
                              "历史逾期还款期数"]
    for col in positive_numeric_cols:
        if col in df.columns:
            negative_count = int((df[col] < 0).sum())
            if negative_count > 0:
                anomalies[f"{col}_负数"] = negative_count

    # 7. 分类字段非法取值
    anomalies.update(_detect_category_anomalies(df, LC_CATEGORY_ALLOWED_VALUES))
    
    return anomalies


def _detect_lp_anomalies(df: pd.DataFrame) -> Dict[str, int]:
    """检测LP表中的异常值。
    
    参数说明：
    - df：原始LP数据
    
    返回：
    - 字典，键为列名，值为异常值数量
    """
    anomalies = {}
    total_rows = len(df)
    
    # 1. 缺失值统计
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            anomalies[f"{col}_缺失值"] = missing_count
    
    # 2. 日期字段转换失败
    date_cols = ["到期日期", "还款日期", "recorddate"]
    for col in date_cols:
        if col in df.columns:
            date_series = pd.to_datetime(df[col], errors='coerce')
            invalid_dates = date_series.isna() & df[col].notna()
            if invalid_dates.sum() > 0:
                anomalies[f"{col}_格式错误"] = int(invalid_dates.sum())
    
    # 3. 数值字段的负数
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        negative_count = int((df[col] < 0).sum())
        if negative_count > 0:
            anomalies[f"{col}_负数"] = negative_count
    
    # 4. 空字符串
    for col in df.columns:
        if df[col].dtype == 'object':
            empty_str_count = int((df[col] == "").sum())
            if empty_str_count > 0:
                anomalies[f"{col}_空字符串"] = empty_str_count

    # 5. 分类字段非法取值
    anomalies.update(_detect_category_anomalies(df, LP_CATEGORY_ALLOWED_VALUES))
    
    return anomalies


def _detect_lcis_anomalies(df: pd.DataFrame) -> Dict[str, int]:
    """检测LCIS表中的异常值。
    
    参数说明：
    - df：原始LCIS数据
    
    返回：
    - 字典，键为列名，值为异常值数量
    """
    anomalies = {}
    total_rows = len(df)
    
    # 1. 缺失值统计
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            anomalies[f"{col}_缺失值"] = missing_count
    
    # 2. 日期字段转换失败
    date_cols = ["借款成功日期", "上次还款日期", "下次计划还款日期", "recorddate"]
    for col in date_cols:
        if col in df.columns:
            date_series = pd.to_datetime(df[col], errors='coerce')
            invalid_dates = date_series.isna() & df[col].notna()
            if invalid_dates.sum() > 0:
                anomalies[f"{col}_格式错误"] = int(invalid_dates.sum())
    
    # 3. 数值字段的负数
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        negative_count = int((df[col] < 0).sum())
        if negative_count > 0:
            anomalies[f"{col}_负数"] = negative_count
    
    # 4. 空字符串
    for col in df.columns:
        if df[col].dtype == 'object':
            empty_str_count = int((df[col] == "").sum())
            if empty_str_count > 0:
                anomalies[f"{col}_空字符串"] = empty_str_count
    
    # 5. 分类字段非法取值统计
    anomalies.update(_detect_category_anomalies(df, LCIS_CATEGORY_ALLOWED_VALUES))
    
    return anomalies


def _normalize_category_value(value: object) -> str | None:
    """标准化分类字段取值为可比较的字符串。"""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _detect_category_anomalies(
    df: pd.DataFrame,
    allowed_map: Dict[str, Iterable[object]],
) -> Dict[str, int]:
    """针对给定的字段-取值集合，检测非法分类取值。"""
    anomalies: Dict[str, int] = {}

    for col, allowed_values in allowed_map.items():
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        normalized_series = series.apply(_normalize_category_value)
        allowed_normalized = {
            value
            for value in (
                _normalize_category_value(v) for v in allowed_values
            )
            if value is not None and value != ""
        }

        invalid_mask = ~normalized_series.isin(allowed_normalized)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            anomalies[f"{col}_非法取值"] = invalid_count

    return anomalies


def generate_anomaly_statistics(
    lc_df: pd.DataFrame,
    lp_df: pd.DataFrame,
    lcis_df: pd.DataFrame,
    cfg: Dict[str, Iterable[str]]
) -> pd.DataFrame:
    """生成异常值统计表。
    
    参数说明：
    - lc_df：原始LC数据
    - lp_df：原始LP数据
    - lcis_df：原始LCIS数据
    - cfg：配置文件中关于特征列的约定
    
    返回：
    - DataFrame，包含：表名、数据项、异常值数量、总记录数、异常值占比
    """
    records = []
    
    # LC表异常值统计
    lc_anomalies = _detect_lc_anomalies(lc_df, cfg)
    lc_total = len(lc_df)
    for anomaly_type, count in lc_anomalies.items():
        records.append({
            "表名": "LC",
            "数据项": anomaly_type,
            "异常值数量": count,
            "总记录数": lc_total,
            "异常值占比": round(count / lc_total * 100, 2) if lc_total > 0 else 0.0
        })
    
    # LP表异常值统计
    lp_anomalies = _detect_lp_anomalies(lp_df)
    lp_total = len(lp_df)
    for anomaly_type, count in lp_anomalies.items():
        records.append({
            "表名": "LP",
            "数据项": anomaly_type,
            "异常值数量": count,
            "总记录数": lp_total,
            "异常值占比": round(count / lp_total * 100, 2) if lp_total > 0 else 0.0
        })
    
    # LCIS表异常值统计
    lcis_anomalies = _detect_lcis_anomalies(lcis_df)
    lcis_total = len(lcis_df)
    for anomaly_type, count in lcis_anomalies.items():
        records.append({
            "表名": "LCIS",
            "数据项": anomaly_type,
            "异常值数量": count,
            "总记录数": lcis_total,
            "异常值占比": round(count / lcis_total * 100, 2) if lcis_total > 0 else 0.0
        })
    
    if not records:
        # 如果没有异常值，返回空表但保持列结构
        return pd.DataFrame(columns=["表名", "数据项", "异常值数量", "总记录数", "异常值占比"])
    
    result_df = pd.DataFrame(records)
    # 按表名和数据项排序
    result_df = result_df.sort_values(["表名", "数据项"])
    return result_df