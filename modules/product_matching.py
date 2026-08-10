"""
模块3：信贷产品匹配
根据企业指标与产品库准入条件做规则匹配
"""

import pandas as pd

def match_products(metrics, df_products):
    """
    参数：
        metrics: dict，企业的各项指标
        df_products: DataFrame，银行产品库
    返回：
        list of dict，每个dict包含：匹配度、产品名、银行、额度、利率、准入条件、差距说明
    """
    results = []

    # 提取企业指标（安全转换）
    revenue = safe_float(metrics.get("营业收入", 0))
    years = safe_float(metrics.get("经营年限", 0))
    customer_con = metrics.get("客户集中度", "")
    # 暂时默认企业可提供抵押（后续可加选项）
    can_provide_collateral = True

    for _, row in df_products.iterrows():
        # 提取产品准入条件
        rev_req = parse_range_lower(row["营收门槛（万元）"])   # 取最小值
        year_req = safe_float(row["成立年限门槛"])
        mortgage_req = row["抵押要求"]
        customer_req = row["客户集中度要求"]

        # 差距列表
        gaps = []

        # 1. 营收检查
        if revenue < rev_req:
            gaps.append(f"营收需达到 {rev_req} 万元，当前为 {revenue} 万元")

        # 2. 成立年限检查
        if years < year_req:
            gaps.append(f"企业需成立满 {year_req} 年，当前为 {years} 年")

        # 3. 客户集中度检查（简单关键词匹配）
        if customer_req and "无" not in customer_req:
            if "低" in customer_req and "低" not in customer_con:
                gaps.append(f"客户集中度要求低（前五大客户占比<30%），当前为 {customer_con}")
            elif "科技" in customer_req and "科技" not in metrics.get("行业", ""):
                gaps.append("该产品面向科技型企业，当前行业可能不符合")

        # 4. 抵押要求检查
        if "需" in mortgage_req and not can_provide_collateral:
            gaps.append(f"该产品需要：{mortgage_req}，当前企业无法提供")

        # 判断匹配度
        if not gaps:
            match_degree = "🟢 完全匹配"
            gap_desc = "无"
        elif len(gaps) <= 2:
            match_degree = "🟡 差距匹配"
            gap_desc = "；".join(gaps)
        else:
            continue  # 不匹配，跳过不展示

        results.append({
            "匹配度": match_degree,
            "产品名": row["产品名"],
            "银行": row["银行"],
            "额度": row["额度范围（万元）"],
            "利率": row["利率范围（%）"],
            "准入条件": f"营收≥{rev_req}万, 成立≥{year_req}年, {mortgage_req}",
            "差距说明": gap_desc,
            "数据来源": row["数据来源"],
            "采集日期": row["采集日期"]
        })

    # 排序：完全匹配在前
    results.sort(key=lambda x: 0 if "完全" in x["匹配度"] else 1)
    return results

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def parse_range_lower(range_str):
    """从 '10-200' 提取最小值 10"""
    try:
        return float(range_str.split("-")[0])
    except:
        return 0.0