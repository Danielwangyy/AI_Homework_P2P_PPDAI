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
        
        # 检查是否为成功认证（检查字符串是否包含成功关键词）
        val_str_upper = val_str.upper()
        for keyword in SUCCESS_KEYWORDS:
            if keyword.upper() in val_str_upper or keyword in val_str:
                return 1.0
        
        # 检查是否为失败认证
        for keyword in FAILURE_KEYWORDS:
            if keyword.upper() in val_str_upper or keyword in val_str:
                return 0.0
        
        # 默认：如果包含"成功"关键字，返回1，否则返回0
        if "成功" in val_str:
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
    4. 添加新特征：正常还款比 = 历史正常还款期数 / (历史正常还款期数 + 历史逾期还款期数)
    
    参数说明：
    - df：原始 LC 数据
    - cfg：配置文件中关于特征列的约定（可选，用于兼容性）
    """
    cleaned = df.copy()
    
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
    
    # 4. 添加新特征：正常还款比
    if "历史正常还款期数" in cleaned.columns and "历史逾期还款期数" in cleaned.columns:
        normal_periods = pd.to_numeric(cleaned["历史正常还款期数"], errors='coerce').fillna(0)
        overdue_periods = pd.to_numeric(cleaned["历史逾期还款期数"], errors='coerce').fillna(0)
        total_periods = normal_periods + overdue_periods
        
        # 计算正常还款比，避免除零
        cleaned["正常还款比"] = np.where(
            total_periods > 0,
            normal_periods / total_periods,
            0.0  # 如果总期数为0，则正常还款比为0
        )
    
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
    1. 认证字段处理：手机认证、户口认证使用one-hot编码（保持原值），
       视频认证、学历认证、征信认证、淘宝认证：成功认证=1，未成功认证=0
    2. 历史正常还款期数、历史逾期还款期数：以99%分位数值为极大值，超过的取极大值
    3. 标当前状态：只保留'正常还款中'，'逾期中'，'已还清'，其他删除（设为缺失）
    4. 上次还款日期：删除非日期项（设为缺失）
    5. 历史成功借款次数、历史成功借款金额、总待还本金：保留缺失值
    6. 其他字段保持不变
    
    参数说明：
    - df：原始 LCIS 数据
    """
    cleaned = df.copy()
    
    # 1. 认证字段处理
    # 手机认证、户口认证：one-hot编码（保持原值，不做转换）
    # 视频认证、学历认证、征信认证、淘宝认证：成功认证=1，未成功认证=0
    auth_cols_binary = ["视频认证", "学历认证", "征信认证", "淘宝认证"]
    for col in auth_cols_binary:
        if col in cleaned.columns:
            cleaned[col] = _convert_auth_to_binary(cleaned[col])
    
    # 2. 历史正常还款期数、历史逾期还款期数：以99%分位数值为极大值
    period_cols = ["历史正常还款期数", "历史逾期还款期数"]
    for col in period_cols:
        if col in cleaned.columns:
            # 转换为数值类型
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
            # 计算99%分位数
            p99 = cleaned[col].quantile(0.99)
            if pd.notna(p99):
                # 超过99%分位数的值替换为99%分位数
                cleaned.loc[cleaned[col] > p99, col] = p99
    
    # 3. 标当前状态：只保留'正常还款中'，'逾期中'，'已还清'，其他删除
    if "标当前状态" in cleaned.columns:
        valid_statuses = {"正常还款中", "逾期中", "已还清"}
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