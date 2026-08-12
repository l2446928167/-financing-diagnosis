"""
模块3：信贷产品匹配
根据企业指标与产品库准入条件做规则匹配
v1.3：新增纳税评级匹配、融资机构数匹配、征信/法院执行一票否决逻辑
"""

import pandas as pd


def match_products(metrics, df_products):
    """
    参数：
        metrics: dict，企业的各项指标
        df_products: DataFrame，银行产品库
    返回：
        list of dict，每个dict包含：匹配度、产品名、银行、额度、利率、准入条件、差距说明
    v1.3新增：
        - 纳税评级匹配：产品有纳税评级要求时检查企业评级是否达标
        - 融资机构数匹配：产品有机构数上限时检查企业是否超标
        - 征信/法院执行一票否决：有法院执行记录的产品匹配度降级
    """
    results = []

    # 提取企业指标（安全转换）
    revenue = safe_float(metrics.get("营业收入", 0))
    years = safe_float(metrics.get("经营年限", 0))
    customer_con = metrics.get("客户集中度", "")
    # 暂时默认企业可提供抵押（后续可加选项）
    can_provide_collateral = True

    # v1.3：信用相关指标
    tax_rating = metrics.get("纳税信用评级", "")
    credit_status = metrics.get("实控人征信状态", "")
    court_execution = metrics.get("法院执行记录", "无")
    financing_count = safe_float(metrics.get("融资机构数量", 0))

    # 纳税评级优先级排序：A > B > M > C > D
    tax_rating_order = {"A": 5, "B": 4, "M": 3, "C": 2, "D": 1}

    for _, row in df_products.iterrows():
        # 提取产品准入条件
        rev_req = parse_range_lower(row["营收门槛（万元）"])
        year_req = safe_float(row["成立年限门槛"])
        mortgage_req = row["抵押要求"]
        customer_req = row["客户集中度要求"]

        # v1.3：新增列的提取（带兼容性，若列不存在则默认无要求）
        tax_rating_req = row.get("纳税评级要求", "")
        if pd.isna(tax_rating_req):
            tax_rating_req = ""
        financing_count_limit = safe_float(row.get("融资机构数上限", ""), default=0)
        if pd.isna(row.get("融资机构数上限", 0)):
            financing_count_limit = 0

        # 差距列表
        gaps = []
        # 是否有一票否决级差距
        veto = False

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

        # 5. v1.3：纳税评级匹配
        if tax_rating_req:
            # 产品的纳税评级要求，企业评级需达到或超过
            req_order = tax_rating_order.get(tax_rating_req.strip(), 0)
            ent_order = tax_rating_order.get(tax_rating.strip(), 0)
            if ent_order < req_order:
                gaps.append(f"纳税评级要求{tax_rating_req}级及以上，当前为{'未评级' if not tax_rating else tax_rating}级")

        # 6. v1.3：融资机构数匹配
        if financing_count_limit > 0 and financing_count > financing_count_limit:
            gaps.append(f"融资机构数上限为{int(financing_count_limit)}家，当前为{int(financing_count)}家")

        # 7. v1.3：法院执行一票否决
        if court_execution and "有" in court_execution and "无" not in court_execution:
            veto = True
            gaps.append("企业有法院执行/诉讼记录，影响信贷审批")

        # 判断匹配度
        if veto:
            # 有法院执行，匹配度降级为"差距匹配"
            match_degree = "🟡 差距匹配"
            gap_desc = "；".join(gaps)
        elif not gaps:
            match_degree = "🟢 完全匹配"
            gap_desc = "无"
        elif len(gaps) <= 2:
            match_degree = "🟡 差距匹配"
            gap_desc = "；".join(gaps)
        else:
            continue  # 不匹配，跳过不展示

        # 构建准入条件描述（含纳税评级和融资机构数）
        cond_parts = [f"营收≥{rev_req}万", f"成立≥{year_req}年", mortgage_req]
        if tax_rating_req:
            cond_parts.append(f"纳税评级≥{tax_rating_req}")
        if financing_count_limit > 0:
            cond_parts.append(f"融资机构≤{int(financing_count_limit)}家")
        cond_str = ", ".join(cond_parts)

        results.append({
            "匹配度": match_degree,
            "产品名": row["产品名"],
            "银行": row["银行"],
            "额度": row["额度范围（万元）"],
            "利率": row["利率范围（%）"],
            "准入条件": cond_str,
            "差距说明": gap_desc,
            "数据来源": row["数据来源"],
            "采集日期": row["采集日期"]
        })

    # 排序：完全匹配在前，差距匹配在后
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
