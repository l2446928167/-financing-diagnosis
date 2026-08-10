"""
模块2：金融健康诊断
基于5个维度的规则引擎评分
"""

def safe_float(value, default=0.0):
    """安全转换为浮点数，失败则返回默认值"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def score_liquidity(metrics):
    """
    流动性健康（满分10）
    指标：流动比率、速动比率（若无则用流动比率代替）、现金流覆盖（短期借款/营业收入）
    """
    current_ratio = safe_float(metrics.get("流动比率", 0))
    # 假设速动比率未提供，我们仅用流动比率
    score = 0
    if current_ratio >= 2.0:
        score = 10
    elif current_ratio >= 1.5:
        score = 8
    elif current_ratio >= 1.0:
        score = 6
    elif current_ratio >= 0.5:
        score = 4
    else:
        score = 2

    # 现金流覆盖：短期借款 / 营业收入（取倒数，越小越好）
    revenue = safe_float(metrics.get("营业收入", 1))
    short_debt = safe_float(metrics.get("短期借款", 0))
    if revenue > 0:
        debt_ratio = short_debt / revenue
        if debt_ratio < 0.3:
            score = min(10, score + 1)
        elif debt_ratio > 0.7:
            score = max(1, score - 1)

    return round(score, 1)

def score_solvency(metrics):
    """
    偿债能力（满分10）
    指标：资产负债率、利息保障倍数（简化）、短期偿债压力
    """
    total_assets = safe_float(metrics.get("总资产", 1))
    total_liabilities = safe_float(metrics.get("总负债", 0))
    debt_ratio = total_liabilities / total_assets if total_assets > 0 else 0.8

    score = 0
    if debt_ratio < 0.3:
        score = 10
    elif debt_ratio < 0.5:
        score = 8
    elif debt_ratio < 0.7:
        score = 6
    elif debt_ratio < 0.85:
        score = 4
    else:
        score = 2

    # 利息保障倍数：用 净利润/(利息费用) 近似，但我们没有利息费用，可假设为0或略
    # 这里简化，只看短期偿债：短期借款/流动资产？没有流动资产，跳过。
    return score

def score_stability(metrics):
    """
    经营稳定性（满分10）
    指标：营收趋势（无历史数据，用营收规模近似），经营年限，客户集中度
    """
    revenue = safe_float(metrics.get("营业收入", 0))
    years = safe_float(metrics.get("经营年限", 1))
    customer_con = metrics.get("客户集中度", "中（30%~60%）")

    # 年限评分
    if years >= 5:
        year_score = 10
    elif years >= 3:
        year_score = 7
    elif years >= 2:
        year_score = 5
    else:
        year_score = 3

    # 营收规模评分（假设年营收500万以上较稳定）
    if revenue >= 1000:
        rev_score = 10
    elif revenue >= 500:
        rev_score = 8
    elif revenue >= 200:
        rev_score = 6
    else:
        rev_score = 4

    # 客户集中度评分
    if "低" in customer_con:
        con_score = 10
    elif "中" in customer_con:
        con_score = 7
    else:
        con_score = 4

    # 平均
    score = (year_score + rev_score + con_score) / 3
    return round(score, 1)

def score_receivable(metrics):
    """
    应收账款质量（满分10）
    指标：回款周期（应收账款/营业收入*365），账龄结构
    """
    revenue = safe_float(metrics.get("营业收入", 1))
    receivables = safe_float(metrics.get("应收账款", 0))
    # 回款周期
    if revenue > 0:
        dso = (receivables / revenue) * 365
    else:
        dso = 180  # 默认较长

    if dso < 30:
        turnover_score = 10
    elif dso < 60:
        turnover_score = 8
    elif dso < 90:
        turnover_score = 6
    elif dso < 120:
        turnover_score = 4
    else:
        turnover_score = 2

    # 账龄结构评分：超12月比例
    over_12 = safe_float(metrics.get("应收账款_超12月占比", 10))
    if over_12 < 5:
        aging_score = 10
    elif over_12 < 15:
        aging_score = 7
    elif over_12 < 30:
        aging_score = 4
    else:
        aging_score = 2

    score = (turnover_score + aging_score) / 2
    return round(score, 1)

def score_financing(metrics):
    """
    融资结构（满分10）
    指标：负债类型（定性），融资成本（平均利率），杠杆水平（资产负债率）
    """
    total_assets = safe_float(metrics.get("总资产", 1))
    total_liabilities = safe_float(metrics.get("总负债", 0))
    debt_ratio = total_liabilities / total_assets if total_assets > 0 else 0.8

    rate = safe_float(metrics.get("平均融资利率", 5.0))

    # 杠杆评分
    if debt_ratio < 0.4:
        leverage_score = 10
    elif debt_ratio < 0.6:
        leverage_score = 7
    elif debt_ratio < 0.8:
        leverage_score = 4
    else:
        leverage_score = 2

    # 成本评分（利率）
    if rate <= 4.0:
        cost_score = 10
    elif rate <= 5.5:
        cost_score = 8
    elif rate <= 7.0:
        cost_score = 6
    else:
        cost_score = 3

    score = (leverage_score + cost_score) / 2
    return round(score, 1)

def traffic_light(score):
    """红黄绿灯：>=7绿，4~7黄，<4红"""
    if score >= 7:
        return "🟢 绿色"
    elif score >= 4:
        return "🟡 黄色"
    else:
        return "🔴 红色"

def generate_risks_and_suggestions(metrics, scores):
    """生成风险点（最多5条）和改善建议"""
    risks = []
    suggestions = []

    # 流动性
    if scores["流动性健康"] < 4:
        risks.append("流动比率过低，短期偿债压力大")
        suggestions.append("建议增加流动资产或减少短期负债，改善流动性。")
    elif scores["流动性健康"] < 7:
        risks.append("流动性一般，需关注现金流管理")
        suggestions.append("建议优化库存和应收账款周转，提升现金储备。")

    # 偿债能力
    if scores["偿债能力"] < 4:
        risks.append("资产负债率过高，偿债能力弱")
        suggestions.append("建议控制新增负债，通过增资或利润留存降低杠杆。")
    elif scores["偿债能力"] < 7:
        risks.append("负债水平中等，存在一定偿债压力")
        suggestions.append("建议合理安排债务结构，避免短借长投。")

    # 经营稳定性
    if scores["经营稳定性"] < 4:
        risks.append("经营稳定性较差，营收或客户结构风险高")
        suggestions.append("建议拓展客户群体，降低单一客户依赖。")
    elif scores["经营稳定性"] < 7:
        risks.append("经营稳定性有改善空间")
        suggestions.append("建议提升营收规模并优化客户结构。")

    # 应收账款
    if scores["应收账款质量"] < 4:
        risks.append("应收账款质量差，坏账风险高")
        suggestions.append("建议加强账龄管理，对超期款项加紧催收，或考虑保理融资。")
    elif scores["应收账款质量"] < 7:
        risks.append("应收账款回款较慢，需关注账龄变化")
        suggestions.append("建议优化信用政策，缩短账期。")

    # 融资结构
    if scores["融资结构"] < 4:
        risks.append("融资成本过高或杠杆水平危险")
        suggestions.append("建议寻求政策性低息贷款置换高息负债，降低财务成本。")
    elif scores["融资结构"] < 7:
        risks.append("融资结构可优化，成本略高或杠杆偏高")
        suggestions.append("建议比较不同银行产品，争取更低利率。")

    # 限制条数
    risks = risks[:5]
    suggestions = suggestions[:5]
    return risks, suggestions

def diagnose(metrics):
    """
    主诊断函数，返回一个包含所有诊断信息的字典。
    """
    dims = {
        "流动性健康": score_liquidity(metrics),
        "偿债能力": score_solvency(metrics),
        "经营稳定性": score_stability(metrics),
        "应收账款质量": score_receivable(metrics),
        "融资结构": score_financing(metrics)
    }

    weights = {
        "流动性健康": 0.25,
        "偿债能力": 0.25,
        "经营稳定性": 0.20,
        "应收账款质量": 0.15,
        "融资结构": 0.15
    }

    overall = sum(dims[k] * weights[k] for k in dims)
    overall = round(overall, 1)

    risks, suggestions = generate_risks_and_suggestions(metrics, dims)

    result = {
        "overall_score": overall,
        "dimension_scores": dims,
        "traffic_lights": {k: traffic_light(v) for k, v in dims.items()},
        "risks": risks,
        "suggestions": suggestions,
        "metrics_used": metrics
    }
    return result