"""
模块3：信贷产品匹配
根据企业指标与产品库准入条件做规则匹配
v1.4：适配新Schema（额度下限/上限等结构化字段），集成差距分析模块
"""

import pandas as pd
from modules.gap_analysis import analyze_gaps, _check_product_gaps


def match_products(metrics, df_products):
    """
    参数：
        metrics: dict，企业的各项指标
        df_products: DataFrame，银行产品库（新Schema）
    返回：
        list of dict，每个dict包含：匹配度、产品名、银行、产品类型、额度、利率、准入条件、差距说明、差距分析
    v1.4改动：
        - 适配新Schema列名（额度下限/上限、利率下限/上限等）
        - 调用gap_analysis模块获取差距详情
        - 在匹配结果中增加"差距分析"字段
    """
    results = []

    # 提取企业指标（安全转换）
    revenue = safe_float(metrics.get("营业收入", 0))
    years = safe_float(metrics.get("经营年限", 0))
    # 暂时默认企业可提供抵押（后续可加选项）
    can_provide_collateral = metrics.get("可提供抵押", True)

    # 信用相关指标
    tax_rating = str(metrics.get("纳税信用评级", "")).strip()
    credit_status = metrics.get("实控人征信状态", "")
    court_execution = metrics.get("法院执行记录", "无")
    financing_count = safe_float(metrics.get("融资机构数量", 0))

    # 纳税评级优先级排序
    tax_rating_order = {"A": 5, "B": 4, "M": 3, "C": 2, "D": 1}

    for _, row in df_products.iterrows():
        # 提取产品准入条件（新Schema列名）
        rev_req = safe_float(row.get("营收门槛", 0))
        year_req = safe_float(row.get("成立年限门槛", 0))
        mortgage_req = str(row.get("抵押要求", ""))
        product_type = str(row.get("产品类型", ""))

        # 新增列
        tax_rating_req = str(row.get("纳税评级要求", "")).strip()
        if pd.isna(row.get("纳税评级要求", "")) or not tax_rating_req:
            tax_rating_req = ""
        financing_count_limit = safe_float(row.get("融资机构数上限", 0))
        if pd.isna(row.get("融资机构数上限", 0)):
            financing_count_limit = 0
        debt_ratio_limit = safe_float(row.get("资产负债率上限", 0))
        current_ratio_min = safe_float(row.get("流动比率下限", 0))
        industry_limit = str(row.get("行业限制", ""))
        if pd.isna(row.get("行业限制", "")):
            industry_limit = ""

        # 差距列表
        gaps = []
        # 是否有一票否决级差距
        veto = False

        # 1. 营收检查
        if rev_req > 0 and revenue < rev_req:
            gaps.append(f"营收需达到 {rev_req:.0f} 万元，当前为 {revenue:.0f} 万元")

        # 2. 成立年限检查
        if year_req > 0 and years < year_req:
            gaps.append(f"企业需成立满 {year_req:.1f} 年，当前为 {years:.1f} 年")

        # 3. 纳税评级匹配
        if tax_rating_req:
            req_order = tax_rating_order.get(tax_rating_req, 0)
            ent_order = tax_rating_order.get(tax_rating, 0)
            if ent_order < req_order:
                current_label = tax_rating if tax_rating else "未评级"
                gaps.append(f"纳税评级要求{tax_rating_req}级及以上，当前为{current_label}级")

        # 4. 融资机构数匹配
        if financing_count_limit > 0 and financing_count > financing_count_limit:
            gaps.append(f"融资机构数上限为{int(financing_count_limit)}家，当前为{int(financing_count)}家")

        # 5. 资产负债率检查
        if debt_ratio_limit > 0:
            total_assets = safe_float(metrics.get("总资产", 0))
            total_liabilities = safe_float(metrics.get("总负债", 0))
            if total_assets > 0 and total_liabilities > 0:
                debt_ratio = (total_liabilities / total_assets) * 100
                if debt_ratio > debt_ratio_limit:
                    gaps.append(f"资产负债率要求≤{debt_ratio_limit:.0f}%，当前为{debt_ratio:.1f}%")

        # 6. 流动比率检查
        if current_ratio_min > 0:
            ent_current_ratio = safe_float(metrics.get("流动比率", 0))
            if ent_current_ratio < current_ratio_min:
                gaps.append(f"流动比率要求≥{current_ratio_min:.2f}，当前为{ent_current_ratio:.2f}")

        # 7. 抵押要求检查
        if "需" in mortgage_req and not can_provide_collateral:
            gaps.append(f"该产品需要：{mortgage_req}，当前企业无法提供")

        # 8. 行业限制检查
        if industry_limit:
            enterprise_industry = str(metrics.get("行业", ""))
            industry_keywords = [kw for kw in industry_limit.split("/") if kw]
            matched = any(kw in enterprise_industry for kw in industry_keywords)
            if not matched:
                gaps.append(f"该产品面向{industry_limit}行业，当前行业可能不符合")

        # 9. 法院执行一票否决
        if court_execution and "有" in court_execution and "无" not in court_execution:
            veto = True
            gaps.append("企业有法院执行/诉讼记录，影响信贷审批")

        # 判断匹配度
        if veto:
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

        # 构建额度、利率展示（新Schema）
        amount_lower = safe_float(row.get("额度下限", 0))
        amount_upper = safe_float(row.get("额度上限", 0))
        rate_lower = safe_float(row.get("利率下限", 0))
        rate_upper = safe_float(row.get("利率上限", 0))
        amount_str = f"{amount_lower:.0f}-{amount_upper:.0f}"
        rate_str = f"{rate_lower:.1f}-{rate_upper:.1f}"

        # 构建准入条件描述
        cond_parts = [f"营收≥{rev_req:.0f}万", f"成立≥{year_req:.1f}年", mortgage_req]
        if tax_rating_req:
            cond_parts.append(f"纳税评级≥{tax_rating_req}")
        if financing_count_limit > 0:
            cond_parts.append(f"融资机构≤{int(financing_count_limit)}家")
        if debt_ratio_limit > 0:
            cond_parts.append(f"负债率≤{debt_ratio_limit:.0f}%")
        if current_ratio_min > 0:
            cond_parts.append(f"流动比率≥{current_ratio_min:.2f}")
        if industry_limit:
            cond_parts.append(f"行业:{industry_limit}")
        cond_str = ", ".join(cond_parts)

        # v1.4：获取该产品的差距分析详情
        product_gap_detail = _check_product_gaps(metrics, row)

        results.append({
            "匹配度": match_degree,
            "产品名": row["产品名"],
            "银行": row["银行"],
            "产品类型": product_type,
            "额度": amount_str,
            "利率": rate_str,
            "准入条件": cond_str,
            "差距说明": gap_desc,
            "差距分析": product_gap_detail,  # v1.4新增
            "数据来源": row.get("数据来源", ""),
            "采集日期": row.get("采集日期", ""),
        })

    # 排序：完全匹配在前，差距匹配在后
    results.sort(key=lambda x: 0 if "完全" in x["匹配度"] else 1)
    return results


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
