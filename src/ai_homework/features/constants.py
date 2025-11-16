"""特征工程阶段使用的常量定义。

该模块集中维护以下几类信息，方便统一管控：
- 原始可用字段白名单（以借款成功时即可获取的信息为准）；
- 潜在泄露字段黑名单（包含标签、本金回收、LP/LCIS 记录等放款后才出现的数据）；
- 列别名映射，统一特征列的英文命名；
- 数值/二值/类别型特征在工程阶段需要的规范化配置；
- 元数据列（例如借款成功日期）用于后续时间切分，但不会进入模型训练。
"""
from __future__ import annotations

from typing import Dict, Iterable, Set

SAFE_RAW_COLUMNS: Set[str] = {
    "ListingId",
    "借款金额",
    "借款期限",
    "借款利率",
    "借款成功日期",
    "初始评级",
    "借款类型",
    "是否首标",
    "年龄",
    "性别",
    "手机认证",
    "户口认证",
    "视频认证",
    "学历认证",
    "征信认证",
    "淘宝认证",
    "历史成功借款次数",
    "历史成功借款金额",
    "总待还本金",
    "历史正常还款期数",
    "历史逾期还款期数",
}

# 放款后或基于结果生成的字段，禁止进入特征矩阵。
LEAKAGE_RAW_COLUMNS: Set[str] = {
    "违约标签",
    "标签来源",
    "是否有效样本",
    "是否周期结束样本",
    "逾期天数总和",
    "LP最大期数",
    "LP最后到期日",
    "LP最后还款日",
    "LP记录日期",
    "LCIS记录日期",
    "sum_DPD",
    "is_valid",
    "is_effective",
    "label",
    "label_source",
    "lp_max_period",
    "lp_last_due_date",
    "lp_last_repay_date",
    "lp_recorddate",
    "lcis_recorddate",
    "借款理论到期日期",
}

# 由黑名单字段衍生出的特征，同样需要阻断。
LEAKAGE_FEATURE_COLUMNS: Set[str] = {
    "overdue_days_sum",
    "is_valid_flag",
    "is_cycle_finished_flag",
    "loan_date_to_theoretical_due_days",
    "loan_date_to_lp_last_due_days",
    "loan_date_to_lp_last_repay_days",
    "loan_date_to_lp_record_days",
    "loan_date_to_lcis_record_days",
    "loan_to_lp_term_ratio",
    "lp_vs_declared_term_diff",
}

# 用于统一英文特征命名的映射关系。
ALIAS_MAP: Dict[str, str] = {
    "loan_amount": "借款金额",
    "loan_term": "借款期限",
    "interest_rate": "借款利率",
    "loan_date": "借款成功日期",
    "loan_type": "借款类型",
    "rating": "初始评级",
    "first_loan_flag": "是否首标",
    "user_age": "年龄",
    "user_gender": "性别",
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
}

NUMERIC_ALIAS_COLUMNS: Iterable[str] = (
    "loan_amount",
    "loan_term",
    "interest_rate",
    "user_age",
    "history_total_loans",
    "history_total_amount",
    "outstanding_principal",
    "history_normal_terms",
    "history_overdue_terms",
)

BINARY_ALIAS_COLUMNS: Iterable[str] = (
    "first_loan_flag",
    "phone_verified",
    "hukou_verified",
    "video_verified",
    "education_verified",
    "credit_verified",
    "taobao_verified",
)

CATEGORICAL_ALIAS_COLUMNS: Iterable[str] = (
    "loan_type",
    "rating",
    "user_gender",
)

# 元数据列用于切分等辅助逻辑，最终不会输入模型。
METADATA_COLUMNS: Set[str] = {
    "loan_date",
}

ALL_BLACKLIST_COLUMNS: Set[str] = LEAKAGE_RAW_COLUMNS | LEAKAGE_FEATURE_COLUMNS


