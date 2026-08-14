"""
模块2：金融健康诊断
基于8个维度的规则引擎评分（v1.3 升级版）
维度与权重：
  现金流健康(0.18) / 偿债能力(0.14) / 盈利质量(0.12) / 运营效率(0.10)
  成长性(0.08) / 经营稳定性(0.10) / 应收账款质量(0.08) / 信用与融资(0.20)
"""

def safe_float(value, default=0.0):
    """安全转换为浮点数，失败则返回默认值"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ============================================================
#  维度1：现金流健康（权重 0.18）
#  指标：经营现金流净额、现金流量比率（经营现金流/流动负债）、
#        盈利现金比（经营现金流/净利润）
# ============================================================
def score_cashflow(metrics):
    """
    现金流健康（满分10）
    分段映射 + 缺值给中间分5
    """
    ocf = safe_float(metrics.get("经营活动现金流净额", ""), default=None)
    current_liabilities = safe_float(metrics.get("流动负债", ""), default=None)
    net_profit = safe_float(metrics.get("净利润", ""), default=None)

    # 子评分1：经营现金流净额为正且规模合理
    if ocf is None:
        ocf_score = 5  # 缺值中间分
    else:
        if ocf > 0:
            ocf_score = 8
            # 净额超过50万给满分
            if ocf >= 50:
                ocf_score = 10
            elif ocf >= 20:
                ocf_score = 9
        else:
            # 负现金流
            if ocf > -10:
                ocf_score = 4
            elif ocf > -50:
                ocf_score = 3
            else:
                ocf_score = 1

    # 子评分2：现金流量比率 = 经营现金流 / 流动负债
    if ocf is None or current_liabilities is None or current_liabilities <= 0:
        ratio_score = 5  # 缺值中间分
    else:
        cashflow_ratio = ocf / current_liabilities
        if cashflow_ratio > 0.3:
            ratio_score = 10
        elif cashflow_ratio > 0.2:
            ratio_score = 8
        elif cashflow_ratio > 0.1:
            ratio_score = 6
        elif cashflow_ratio > 0:
            ratio_score = 4
        else:
            ratio_score = 2

    # 子评分3：盈利现金比 = 经营现金流 / 净利润
    if ocf is None or net_profit is None or net_profit <= 0:
        profit_cash_score = 5  # 缺值或亏损中间分
    else:
        profit_cash_ratio = ocf / net_profit
        if profit_cash_ratio > 1.2:
            profit_cash_score = 10
        elif profit_cash_ratio > 0.8:
            profit_cash_score = 8
        elif profit_cash_ratio > 0.5:
            profit_cash_score = 6
        elif profit_cash_ratio > 0:
            profit_cash_score = 4
        else:
            profit_cash_score = 2

    score = (ocf_score * 0.3 + ratio_score * 0.4 + profit_cash_score * 0.3)
    return round(score, 1)


# ============================================================
#  维度2：偿债能力（权重 0.14）
#  指标：资产负债率、流动比率、利息保障倍数（EBIT/利息费用）
# ============================================================
def score_solvency(metrics):
    """
    偿债能力（满分10）
    """
    total_assets = safe_float(metrics.get("总资产", ""), default=None)
    total_liabilities = safe_float(metrics.get("总负债", ""), default=None)
    current_ratio = safe_float(metrics.get("流动比率", ""), default=None)
    interest_expense = safe_float(metrics.get("利息费用", ""), default=None)

    # 子评分1：资产负债率
    if total_assets is not None and total_assets > 0 and total_liabilities is not None:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.3:
            debt_score = 10
        elif debt_ratio < 0.5:
            debt_score = 8
        elif debt_ratio < 0.7:
            debt_score = 6
        elif debt_ratio < 0.85:
            debt_score = 4
        else:
            debt_score = 2
    else:
        debt_score = 5  # 缺值中间分

    # 子评分2：流动比率
    if current_ratio is not None and current_ratio > 0:
        if current_ratio >= 2.0:
            cr_score = 10
        elif current_ratio >= 1.5:
            cr_score = 8
        elif current_ratio >= 1.0:
            cr_score = 6
        elif current_ratio >= 0.5:
            cr_score = 4
        else:
            cr_score = 2
    else:
        cr_score = 5  # 缺值中间分

    # 子评分3：利息保障倍数 = EBIT / 利息费用（简化用净利润近似EBIT）
    net_profit = safe_float(metrics.get("净利润", ""), default=None)
    if interest_expense is not None and interest_expense > 0 and net_profit is not None:
        # 简化：EBIT ≈ 净利润 + 利息费用（忽略所得税差异）
        ebit = net_profit + interest_expense
        icr = ebit / interest_expense
        if icr >= 5:
            icr_score = 10
        elif icr >= 3:
            icr_score = 8
        elif icr >= 2:
            icr_score = 6
        elif icr >= 1:
            icr_score = 4
        else:
            icr_score = 2
    else:
        icr_score = 5  # 缺值中间分

    score = (debt_score * 0.4 + cr_score * 0.3 + icr_score * 0.3)
    return round(score, 1)


# ============================================================
#  维度3：盈利质量（权重 0.12）
#  指标：毛利率、净利率、盈利现金比（与现金流维度共享）
# ============================================================
def score_profitability(metrics):
    """
    盈利质量（满分10）
    """
    revenue = safe_float(metrics.get("营业收入", ""), default=None)
    net_profit = safe_float(metrics.get("净利润", ""), default=None)
    operating_cost = safe_float(metrics.get("营业成本", ""), default=None)
    ocf = safe_float(metrics.get("经营活动现金流净额", ""), default=None)

    # 子评分1：毛利率 = (营收 - 营业成本) / 营收
    if revenue is not None and revenue > 0 and operating_cost is not None:
        gross_margin = (revenue - operating_cost) / revenue
        if gross_margin > 0.4:
            gm_score = 10
        elif gross_margin > 0.25:
            gm_score = 8
        elif gross_margin > 0.15:
            gm_score = 6
        elif gross_margin > 0:
            gm_score = 4
        else:
            gm_score = 2
    else:
        gm_score = 5  # 缺值中间分

    # 子评分2：净利率 = 净利润 / 营收
    if revenue is not None and revenue > 0 and net_profit is not None:
        net_margin = net_profit / revenue
        if net_margin > 0.15:
            nm_score = 10
        elif net_margin > 0.08:
            nm_score = 8
        elif net_margin > 0.03:
            nm_score = 6
        elif net_margin > 0:
            nm_score = 4
        else:
            nm_score = 2
    else:
        nm_score = 5  # 缺值中间分

    # 子评分3：盈利现金比（与现金流维度共享，但此处权重较低）
    if ocf is not None and net_profit is not None and net_profit > 0:
        pcr = ocf / net_profit
        if pcr > 1.0:
            pcr_score = 10
        elif pcr > 0.7:
            pcr_score = 8
        elif pcr > 0.4:
            pcr_score = 6
        elif pcr > 0:
            pcr_score = 4
        else:
            pcr_score = 2
    else:
        pcr_score = 5  # 缺值中间分

    score = (gm_score * 0.35 + nm_score * 0.35 + pcr_score * 0.3)
    return round(score, 1)


# ============================================================
#  维度4：运营效率（权重 0.10）
#  指标：DSO（应收账款周转天数）、存货周转天数、总资产周转率
# ============================================================
def score_operation_efficiency(metrics):
    """
    运营效率（满分10）
    """
    revenue = safe_float(metrics.get("营业收入", ""), default=None)
    receivables = safe_float(metrics.get("应收账款", ""), default=None)
    inventory = safe_float(metrics.get("存货", ""), default=None)
    total_assets = safe_float(metrics.get("总资产", ""), default=None)

    # 子评分1：DSO = 应收账款 / 营收 * 365
    if revenue is not None and revenue > 0 and receivables is not None:
        dso = (receivables / revenue) * 365
        if dso < 30:
            dso_score = 10
        elif dso < 60:
            dso_score = 8
        elif dso < 90:
            dso_score = 6
        elif dso < 120:
            dso_score = 4
        else:
            dso_score = 2
    else:
        dso_score = 5  # 缺值中间分

    # 子评分2：存货周转天数 = 存货 / 营业成本 * 365（无营业成本用营收近似）
    operating_cost = safe_float(metrics.get("营业成本", ""), default=None)
    cost_basis = operating_cost if (operating_cost is not None and operating_cost > 0) else revenue
    if inventory is not None and cost_basis is not None and cost_basis > 0:
        inv_days = (inventory / cost_basis) * 365
        if inv_days < 30:
            inv_score = 10
        elif inv_days < 60:
            inv_score = 8
        elif inv_days < 90:
            inv_score = 6
        elif inv_days < 180:
            inv_score = 4
        else:
            inv_score = 2
    else:
        inv_score = 5  # 缺值中间分

    # 子评分3：总资产周转率 = 营收 / 总资产
    if revenue is not None and total_assets is not None and total_assets > 0:
        tat = revenue / total_assets
        if tat > 2.0:
            tat_score = 10
        elif tat > 1.0:
            tat_score = 8
        elif tat > 0.5:
            tat_score = 6
        elif tat > 0.2:
            tat_score = 4
        else:
            tat_score = 2
    else:
        tat_score = 5  # 缺值中间分

    score = (dso_score * 0.4 + inv_score * 0.3 + tat_score * 0.3)
    return round(score, 1)


# ============================================================
#  维度5：成长性（权重 0.08）
#  指标：营收增长率、净利润增长率
# ============================================================
def score_growth(metrics):
    """
    成长性（满分10）
    """
    rev_growth = safe_float(metrics.get("营收增长率", ""), default=None)
    profit_growth = safe_float(metrics.get("净利润增长率", ""), default=None)

    # 子评分1：营收增长率
    if rev_growth is not None:
        if rev_growth > 30:
            rg_score = 10
        elif rev_growth > 15:
            rg_score = 8
        elif rev_growth > 5:
            rg_score = 6
        elif rev_growth > 0:
            rg_score = 4
        elif rev_growth > -10:
            rg_score = 3
        else:
            rg_score = 1
    else:
        rg_score = 5  # 缺值中间分

    # 子评分2：净利润增长率
    if profit_growth is not None:
        if profit_growth > 30:
            pg_score = 10
        elif profit_growth > 15:
            pg_score = 8
        elif profit_growth > 5:
            pg_score = 6
        elif profit_growth > 0:
            pg_score = 4
        elif profit_growth > -10:
            pg_score = 3
        else:
            pg_score = 1
    else:
        pg_score = 5  # 缺值中间分

    score = (rg_score * 0.5 + pg_score * 0.5)
    return round(score, 1)


# ============================================================
#  维度6：经营稳定性（权重 0.10）
#  指标：经营年限、营收规模、客户集中度
# ============================================================
def score_stability(metrics):
    """
    经营稳定性（满分10）
    """
    revenue = safe_float(metrics.get("营业收入", ""), default=None)
    years = safe_float(metrics.get("经营年限", ""), default=None)
    customer_con = metrics.get("客户集中度", "")

    # 子评分1：经营年限
    if years is not None and years > 0:
        if years >= 5:
            year_score = 10
        elif years >= 3:
            year_score = 7
        elif years >= 2:
            year_score = 5
        else:
            year_score = 3
    else:
        year_score = 5  # 缺值中间分

    # 子评分2：营收规模
    if revenue is not None:
        if revenue >= 1000:
            rev_score = 10
        elif revenue >= 500:
            rev_score = 8
        elif revenue >= 200:
            rev_score = 6
        else:
            rev_score = 4
    else:
        rev_score = 5  # 缺值中间分

    # 子评分3：客户集中度
    if customer_con:
        if "低" in customer_con:
            con_score = 10
        elif "中" in customer_con:
            con_score = 7
        elif "高" in customer_con:
            con_score = 4
        else:
            con_score = 5
    else:
        con_score = 5  # 缺值中间分

    score = (year_score * 0.35 + rev_score * 0.35 + con_score * 0.3)
    return round(score, 1)


# ============================================================
#  维度7：应收账款质量（权重 0.08）
#  指标：DSO、账龄结构（超12月占比）
# ============================================================
def score_receivable(metrics):
    """
    应收账款质量（满分10）
    """
    revenue = safe_float(metrics.get("营业收入", ""), default=None)
    receivables = safe_float(metrics.get("应收账款", ""), default=None)

    # 子评分1：DSO
    if revenue is not None and revenue > 0 and receivables is not None:
        dso = (receivables / revenue) * 365
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
    else:
        turnover_score = 5  # 缺值中间分

    # 子评分2：账龄结构 - 超12月占比
    over_12 = safe_float(metrics.get("应收账款_超12月占比", ""), default=None)
    if over_12 is not None:
        if over_12 < 5:
            aging_score = 10
        elif over_12 < 15:
            aging_score = 7
        elif over_12 < 30:
            aging_score = 4
        else:
            aging_score = 2
    else:
        aging_score = 5  # 缺值中间分

    score = (turnover_score * 0.5 + aging_score * 0.5)
    return round(score, 1)


# ============================================================
#  维度8：信用与融资（权重 0.20）
#  指标：纳税信用评级、实控人征信、法院执行、融资机构数、融资利率、杠杆水平
# ============================================================
def score_credit_financing(metrics):
    """
    信用与融资（满分10）
    """
    tax_rating = metrics.get("纳税信用评级", "")
    credit_status = metrics.get("实控人征信状态", "")
    court_execution = metrics.get("法院执行记录", "无")
    financing_count = safe_float(metrics.get("融资机构数量", ""), default=None)
    avg_rate = safe_float(metrics.get("平均融资利率", ""), default=None)
    total_assets = safe_float(metrics.get("总资产", ""), default=None)
    total_liabilities = safe_float(metrics.get("总负债", ""), default=None)

    # 子评分1：纳税信用评级
    tax_rating_map = {"A": 10, "B": 8, "M": 6, "C": 3, "D": 1}
    if tax_rating and tax_rating in tax_rating_map:
        tax_score = tax_rating_map[tax_rating]
    else:
        tax_score = 5  # 未评级或缺值中间分

    # 子评分2：实控人征信状态
    if credit_status:
        if "良好" in credit_status:
            credit_score = 10
        elif "一般" in credit_status:
            credit_score = 6
        elif "逾期" in credit_status:
            credit_score = 2
        else:
            credit_score = 5
    else:
        credit_score = 5  # 缺值中间分

    # 子评分3：法院执行/诉讼记录（一票否决级）
    if court_execution:
        if "有" in court_execution and "无" not in court_execution:
            court_score = 1  # 有法院执行，严重扣分
        else:
            court_score = 10
    else:
        court_score = 5  # 缺值中间分

    # 子评分4：融资机构数量（越多越分散风险越大）
    if financing_count is not None:
        if financing_count <= 1:
            fc_score = 10
        elif financing_count <= 3:
            fc_score = 8
        elif financing_count <= 5:
            fc_score = 5
        else:
            fc_score = 2
    else:
        fc_score = 5  # 缺值中间分

    # 子评分5：融资利率
    if avg_rate is not None and avg_rate > 0:
        if avg_rate <= 4.0:
            rate_score = 10
        elif avg_rate <= 5.5:
            rate_score = 8
        elif avg_rate <= 7.0:
            rate_score = 6
        else:
            rate_score = 3
    else:
        rate_score = 5  # 缺值中间分

    # 子评分6：杠杆水平（资产负债率）
    if total_assets is not None and total_assets > 0 and total_liabilities is not None:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.4:
            leverage_score = 10
        elif debt_ratio < 0.6:
            leverage_score = 7
        elif debt_ratio < 0.8:
            leverage_score = 4
        else:
            leverage_score = 2
    else:
        leverage_score = 5  # 缺值中间分

    # 加权：纳税评级(0.20) + 征信(0.20) + 法院执行(0.15) + 机构数(0.10) + 利率(0.15) + 杠杆(0.20)
    score = (
        tax_score * 0.20 +
        credit_score * 0.20 +
        court_score * 0.15 +
        fc_score * 0.10 +
        rate_score * 0.15 +
        leverage_score * 0.20
    )
    return round(score, 1)


def traffic_light(score):
    """红黄绿灯：>=7绿，4~7黄，<4红"""
    if score >= 7:
        return "绿色"
    elif score >= 4:
        return "黄色"
    else:
        return "红色"


def generate_risks_and_suggestions(metrics, scores):
    """生成风险点（最多8条）和改善建议，覆盖全部8个维度"""
    risks = []
    suggestions = []

    # 维度1：现金流健康
    if scores["现金流健康"] < 4:
        risks.append("经营现金流为负或极低，企业面临现金流断裂风险")
        suggestions.append("建议加速应收账款回款、压缩存货周转天数，必要时申请短期流动资金贷款。")
    elif scores["现金流健康"] < 7:
        risks.append("现金流覆盖能力偏弱，需关注现金流量比率")
        suggestions.append("建议优化收支节奏，保持至少3个月经营支出的现金储备。")

    # 维度2：偿债能力
    if scores["偿债能力"] < 4:
        risks.append("资产负债率过高，偿债能力严重不足")
        suggestions.append("建议控制新增负债，通过增资或利润留存降低杠杆，优先偿还高息债务。")
    elif scores["偿债能力"] < 7:
        risks.append("负债水平中等偏上，存在一定偿债压力")
        suggestions.append("建议合理安排债务结构，长短期搭配，避免短借长投。")

    # 维度3：盈利质量
    if scores["盈利质量"] < 4:
        risks.append("盈利质量差，毛利率或净利率过低，盈利现金比不足")
        suggestions.append("建议审视成本结构，提升产品附加值，确保利润有真实现金流支撑。")
    elif scores["盈利质量"] < 7:
        risks.append("盈利质量有改善空间，利润现金转化率偏低")
        suggestions.append("建议关注应收账款回款质量，减少账面利润与现金流偏差。")

    # 维度4：运营效率
    if scores["运营效率"] < 4:
        risks.append("运营效率低下，应收账款和存货周转缓慢")
        suggestions.append("建议优化信用政策缩短DSO，加强存货管理降低库存积压。")
    elif scores["运营效率"] < 7:
        risks.append("运营效率有提升空间，周转天数偏长")
        suggestions.append("建议设置回款KPI，定期清理呆滞库存，提升资产周转率。")

    # 维度5：成长性
    if scores["成长性"] < 4:
        risks.append("营收或利润出现下滑，成长性不足")
        suggestions.append("建议开拓新客户或新市场，优化产品线，寻找增长第二曲线。")
    elif scores["成长性"] < 7:
        risks.append("成长性一般，增速放缓")
        suggestions.append("建议关注行业趋势，适度投入研发或渠道拓展以维持增长。")

    # 维度6：经营稳定性
    if scores["经营稳定性"] < 4:
        risks.append("经营稳定性较差，经营年限短或客户集中度高")
        suggestions.append("建议拓展客户群体降低单一客户依赖，争取签订长期合同。")
    elif scores["经营稳定性"] < 7:
        risks.append("经营稳定性有改善空间")
        suggestions.append("建议提升营收规模并分散客户来源，增强抗风险能力。")

    # 维度7：应收账款质量
    if scores["应收账款质量"] < 4:
        risks.append("应收账款质量差，回款周期长且超期占比高，坏账风险大")
        suggestions.append("建议加强账龄管理，对超期款项加紧催收，或考虑保理融资转移风险。")
    elif scores["应收账款质量"] < 7:
        risks.append("应收账款回款偏慢，需关注账龄变化")
        suggestions.append("建议优化信用政策缩短账期，定期评估客户信用等级。")

    # 维度8：信用与融资
    if scores["信用与融资"] < 4:
        risks.append("信用状况不佳或融资结构危险，存在法院执行或征信逾期")
        suggestions.append("建议优先处理法院执行和征信逾期记录，减少融资机构数降低综合成本，争取纳税评级提升。")
    elif scores["信用与融资"] < 7:
        risks.append("信用与融资有优化空间，融资成本偏高或机构分散")
        suggestions.append("建议比较不同银行产品置换高息负债，提升纳税信用评级争取银税贷优惠。")

    # 限制条数（覆盖8个维度，最多8条）
    risks = risks[:8]
    suggestions = suggestions[:8]
    return risks, suggestions


def diagnose(metrics):
    """
    主诊断函数，返回一个包含所有诊断信息的字典。
    8维评分体系，权重之和严格为1.0
    """
    dims = {
        "现金流健康": score_cashflow(metrics),
        "偿债能力": score_solvency(metrics),
        "盈利质量": score_profitability(metrics),
        "运营效率": score_operation_efficiency(metrics),
        "成长性": score_growth(metrics),
        "经营稳定性": score_stability(metrics),
        "应收账款质量": score_receivable(metrics),
        "信用与融资": score_credit_financing(metrics),
    }

    # 权重之和 = 0.18 + 0.14 + 0.12 + 0.10 + 0.08 + 0.10 + 0.08 + 0.20 = 1.00
    weights = {
        "现金流健康": 0.18,
        "偿债能力": 0.14,
        "盈利质量": 0.12,
        "运营效率": 0.10,
        "成长性": 0.08,
        "经营稳定性": 0.10,
        "应收账款质量": 0.08,
        "信用与融资": 0.20,
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
