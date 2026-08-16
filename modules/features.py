"""
modules/features.py — 企业融资评分模型「单一事实源」特征定义

设计目标（修复此前 bug：特征顺序在 train / runtime / feature_meta 三处强耦合）：
- FEATURE_NAMES 为唯一权威特征列表（顺序即模型输入顺序）。
- 训练管线（scripts/train_real.py）与运行时（modules/ml_model.py）均从此导入，
  杜绝特征错位导致的“无声错误”。
- 提供两种特征构造入口：
    statement_to_features : 从用户上传/录入的财务报表（金额单位：万元）构造 12 维特征；
    dataset_row_to_features: 从真实公开数据集（台湾经济新报企业破产预测）的列构造同一 12 维特征。
- 真实数据集来源：https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction
  （6,819 家真实上市公司、95 个财务比率、二元破产标签，1999–2009）。

本模块不依赖任何 ML 库，纯 Python + numpy，可在运行时被 Streamlit 直接 import。
"""
import math

# —— 12 维企业财务特征（顺序即模型输入顺序，禁止随意改动）——
FEATURE_NAMES = [
    "资产负债率",      # 总负债 / 总资产
    "流动比率",        # 流动资产 / 流动负债
    "速动比率",        # (流动资产-存货) / 流动负债
    "毛利率",          # (营收-营业成本) / 营收
    "资产净利率",      # 净利润 / 总资产
    "现金流比率",      # 货币资金 / 流动负债（近似经营现金流覆盖）
    "总资产周转率",    # 营收 / 总资产
    "应收账款周转率",  # 营收 / 应收账款
    "存货周转率",      # 营业成本 / 存货
    "净利润增长率",    # 同比净利润增长率（小数）
    "利息保障倍数",    # (净利润+利息费用) / 利息费用
    "权益负债率",      # (总资产-总负债) / 总负债（正值=权益>负债）
]

# 真实数据集列名 -> 本项目特征名（精确匹配，已按上面 12 个特征对齐）
DATASET_COLMAP = {
    "资产负债率": "Debt ratio %",
    "流动比率": "Current Ratio",
    "速动比率": "Quick Ratio",
    "毛利率": "Operating Gross Margin",
    "资产净利率": "Net Income to Total Assets",
    "现金流比率": "Cash/Current Liability",
    "总资产周转率": "Total Asset Turnover",
    "应收账款周转率": "Accounts Receivable Turnover",
    "存货周转率": "Inventory Turnover Rate (times)",
    "净利润增长率": "After-tax Net Profit Growth Rate",
    "利息保障倍数": "Interest Coverage Ratio (Interest expense to EBIT)",
    "权益负债率": "Equity to Liability",
}


def _safe(x, default=0.0):
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def statement_to_features(stmt, clip=None):
    """
    从财务报表 dict 构造 12 维特征向量。
    stmt 约定（金额单位：万元；增长率/占比为百分数或小数均可，这里统一按“百分数→小数”处理增长率）。
    含除零保护；clip 为可选 {特征名:(low,high)} 边界（来自训练集分位数），用于让单条记录与训练分布对齐。
    """
    A = _safe(stmt.get("总资产", 0))
    L = _safe(stmt.get("总负债", 0))
    R = _safe(stmt.get("营业收入", 0))
    CA = _safe(stmt.get("流动资产", 0))
    CL = max(_safe(stmt.get("流动负债", 1)), 1e-6)
    INV = _safe(stmt.get("存货", 0))
    AR = _safe(stmt.get("应收账款", 0))
    C = _safe(stmt.get("营业成本", 0))
    NP = _safe(stmt.get("净利润", 0))
    I = _safe(stmt.get("利息费用", 0))
    OCF = _safe(stmt.get("经营活动现金流净额", 0))
    cash = _safe(stmt.get("货币资金", OCF))   # 优先用货币资金，缺失回退经营现金流
    gr = _safe(stmt.get("净利润增长率", 0)) / 100.0

    feats = {
        "资产负债率": L / A if A > 0 else 0.0,
        "流动比率": CA / CL,
        "速动比率": (CA - INV) / CL,
        "毛利率": (R - C) / R if R > 0 else 0.0,
        "资产净利率": NP / A if A > 0 else 0.0,
        "现金流比率": cash / CL,
        "总资产周转率": R / A if A > 0 else 0.0,
        "应收账款周转率": R / AR if AR > 0 else 0.0,
        "存货周转率": C / INV if INV > 0 else 0.0,
        "净利润增长率": gr,
        "利息保障倍数": (NP + I) / I if I > 0 else 0.0,
        "权益负债率": (A - L) / L if L > 0 else 0.0,
    }
    arr = [feats[k] for k in FEATURE_NAMES]
    if clip:
        arr = [min(max(v, clip[k][0]), clip[k][1]) for k, v in zip(FEATURE_NAMES, arr)]
    return list(arr)


def dataset_row_to_features(row, clip=None):
    """
    从真实数据集的一行（dict: 列名->值）构造同一 12 维特征。
    与 statement_to_features 共享 FEATURE_NAMES 与语义，确保训练/推理口径一致。
    注意：UCI / 公开数据集列名常带前导空格，这里统一按去空格键匹配。
    """
    r = {k.strip(): v for k, v in row.items()}
    def col(name):
        return _safe(r.get(name, 0))

    feats = {
        "资产负债率": col("Debt ratio %") / 100.0,          # 数据集为百分数
        "流动比率": col("Current Ratio"),
        "速动比率": col("Quick Ratio"),
        "毛利率": col("Operating Gross Margin") / 100.0,     # 数据集为百分数
        "资产净利率": col("Net Income to Total Assets"),
        "现金流比率": col("Cash/Current Liability"),
        "总资产周转率": col("Total Asset Turnover"),
        "应收账款周转率": col("Accounts Receivable Turnover"),
        "存货周转率": col("Inventory Turnover Rate (times)"),
        "净利润增长率": col("After-tax Net Profit Growth Rate") / 100.0,  # 百分数
        "利息保障倍数": col("Interest Coverage Ratio (Interest expense to EBIT)"),
        "权益负债率": col("Equity to Liability"),
    }
    arr = [feats[k] for k in FEATURE_NAMES]
    if clip:
        arr = [min(max(v, clip[k][0]), clip[k][1]) for k, v in zip(FEATURE_NAMES, arr)]
    return list(arr)
