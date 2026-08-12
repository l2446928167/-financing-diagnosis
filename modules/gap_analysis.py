"""
模块4：差距分析
基于企业指标与产品库准入条件的量化差距分析，输出行动优先级方案
v1.4 新增模块
"""

import pandas as pd


# ============================================================
#  纳税评级排序：A > B > M > C > D
# ============================================================
TAX_RATING_ORDER = {"A": 5, "B": 4, "M": 3, "C": 2, "D": 1}

# 难度量化分（用于排序）
DIFFICULTY_SCORE = {
    "不可控": 99,   # 经营年限
    "高": 3,        # 营收增长
    "中高": 2.5,    # 降低负债率
    "中": 2,        # 纳税评级、减少融资机构数
    "低": 1,        # 提供抵押物
}


def safe_float(val, default=0.0):
    """安全转换为浮点数"""
    try:
        v = float(val)
        return v
    except (ValueError, TypeError):
        return default


def _get_tax_order(rating):
    """获取纳税评级排序值，未知评级返回0"""
    if not rating or pd.isna(rating):
        return 0
    return TAX_RATING_ORDER.get(str(rating).strip(), 0)


def _check_product_gaps(metrics, row):
    """
    检查企业指标与单个产品准入条件的差距
    返回: gaps列表（每个gap是一个dict），是否达标
    """
    gaps = []

    # 提取企业指标
    revenue = safe_float(metrics.get("营业收入", 0))
    years = safe_float(metrics.get("经营年限", 0))
    tax_rating = str(metrics.get("纳税信用评级", "")).strip()
    financing_count = safe_float(metrics.get("融资机构数量", 0))

    # 资产负债率（百分比）
    total_assets = safe_float(metrics.get("总资产", 0))
    total_liabilities = safe_float(metrics.get("总负债", 0))
    debt_ratio = 0.0
    if total_assets > 0 and total_liabilities > 0:
        debt_ratio = (total_liabilities / total_assets) * 100  # 转为百分比

    # 流动比率
    current_ratio = safe_float(metrics.get("流动比率", 0))

    # 提取产品准入条件
    rev_req = safe_float(row.get("营收门槛", 0))
    year_req = safe_float(row.get("成立年限门槛", 0))
    tax_rating_req = str(row.get("纳税评级要求", "")).strip()
    if pd.isna(row.get("纳税评级要求", "")) or not tax_rating_req:
        tax_rating_req = ""
    financing_limit = safe_float(row.get("融资机构数上限", 0))
    debt_ratio_limit = safe_float(row.get("资产负债率上限", 0))
    current_ratio_min = safe_float(row.get("流动比率下限", 0))
    mortgage_req = str(row.get("抵押要求", ""))
    industry_limit = str(row.get("行业限制", ""))
    if pd.isna(row.get("行业限制", "")):
        industry_limit = ""

    # 1. 营收差距
    if rev_req > 0 and revenue < rev_req:
        gap_size = rev_req - revenue
        gaps.append({
            "item": "营收",
            "current": f"{revenue:.0f}万元",
            "required": f"{rev_req:.0f}万元",
            "gap_size": gap_size,
            "difficulty": "高",
            "difficulty_score": DIFFICULTY_SCORE["高"],
            "action": f"提升营收至{rev_req:.0f}万元"
        })

    # 2. 成立年限差距
    if year_req > 0 and years < year_req:
        gap_size = year_req - years
        gaps.append({
            "item": "经营年限",
            "current": f"{years:.1f}年",
            "required": f"{year_req:.1f}年",
            "gap_size": gap_size,
            "difficulty": "不可控",
            "difficulty_score": DIFFICULTY_SCORE["不可控"],
            "action": f"经营满{year_req:.1f}年"
        })

    # 3. 纳税评级差距
    if tax_rating_req:
        req_order = _get_tax_order(tax_rating_req)
        ent_order = _get_tax_order(tax_rating)
        if ent_order < req_order:
            gap_size = req_order - ent_order
            current_label = tax_rating if tax_rating else "未评级"
            gaps.append({
                "item": "纳税评级",
                "current": f"{current_label}级",
                "required": f"{tax_rating_req}级及以上",
                "gap_size": gap_size,
                "difficulty": "中",
                "difficulty_score": DIFFICULTY_SCORE["中"],
                "action": f"提升纳税评级至{tax_rating_req}级"
            })

    # 4. 融资机构数差距
    if financing_limit > 0 and financing_count > financing_limit:
        gap_size = financing_count - financing_limit
        gaps.append({
            "item": "融资机构数",
            "current": f"{int(financing_count)}家",
            "required": f"≤{int(financing_limit)}家",
            "gap_size": gap_size,
            "difficulty": "中",
            "difficulty_score": DIFFICULTY_SCORE["中"],
            "action": f"减少融资机构至{int(financing_limit)}家以内"
        })

    # 5. 资产负债率差距
    if debt_ratio_limit > 0 and debt_ratio > debt_ratio_limit:
        gap_size = debt_ratio - debt_ratio_limit
        gaps.append({
            "item": "资产负债率",
            "current": f"{debt_ratio:.1f}%",
            "required": f"≤{debt_ratio_limit:.0f}%",
            "gap_size": gap_size,
            "difficulty": "中高",
            "difficulty_score": DIFFICULTY_SCORE["中高"],
            "action": f"降低资产负债率至{debt_ratio_limit:.0f}%以下"
        })

    # 6. 流动比率差距
    if current_ratio_min > 0 and current_ratio < current_ratio_min:
        gap_size = current_ratio_min - current_ratio
        gaps.append({
            "item": "流动比率",
            "current": f"{current_ratio:.2f}",
            "required": f"≥{current_ratio_min:.2f}",
            "gap_size": gap_size,
            "difficulty": "中高",
            "difficulty_score": DIFFICULTY_SCORE["中高"],
            "action": f"提升流动比率至{current_ratio_min:.2f}以上"
        })

    # 7. 抵押要求检查（binary差距）
    can_provide_collateral = metrics.get("可提供抵押", True)
    if "需" in mortgage_req and not can_provide_collateral:
        gaps.append({
            "item": "抵押要求",
            "current": "无法提供",
            "required": mortgage_req,
            "gap_size": 1,
            "difficulty": "低",
            "difficulty_score": DIFFICULTY_SCORE["低"],
            "action": f"提供{mortgage_req}"
        })

    # 8. 行业限制检查
    if industry_limit:
        enterprise_industry = str(metrics.get("行业", ""))
        # 简单关键词匹配
        industry_keywords = industry_limit.split("/")
        matched = any(kw in enterprise_industry for kw in industry_keywords if kw)
        if not matched:
            gaps.append({
                "item": "行业限制",
                "current": enterprise_industry or "未填写",
                "required": industry_limit,
                "gap_size": 1,
                "difficulty": "不可控",
                "difficulty_score": DIFFICULTY_SCORE["不可控"],
                "action": f"企业需属于{industry_limit}行业"
            })

    return gaps


def _estimate_time(difficulty, gap_size):
    """根据难度和差距量估算达标时间"""
    if difficulty == "不可控":
        return "只能等待，无法主动缩短"
    elif difficulty == "高":
        if gap_size > 100:
            return "需要1-3年持续增长"
        elif gap_size > 50:
            return "需要6-18个月提升"
        else:
            return "需要3-12个月提升"
    elif difficulty == "中高":
        return "需要6-18个月调整"
    elif difficulty == "中":
        if gap_size >= 2:
            return "需要1-2年规范经营"
        else:
            return "规范纳税1-2年可改善"
    elif difficulty == "低":
        return "提供相关材料即可"
    else:
        return "视具体情况而定"


def _compute_impact(action_key, gap_item, df_products, metrics):
    """
    计算某个行动项达成后能额外解锁多少产品
    action_key: 差距项名称（如"纳税评级"）
    gap_item: 差距详情dict
    返回: (解锁产品数, 解锁产品名列表)
    """
    # 当前不达标的产品列表
    current_unlocked = []

    # 模拟补齐该差距后，重新检查每个产品
    for _, row in df_products.iterrows():
        current_gaps = _check_product_gaps(metrics, row)
        if not current_gaps:
            continue  # 已经达标，不计入解锁

        # 检查该产品的差距中是否包含此行动项
        has_this_gap = False
        remaining_gaps = []
        for g in current_gaps:
            if g["item"] == action_key:
                has_this_gap = True
            else:
                remaining_gaps.append(g)

        # 如果此行动项是该产品的唯一差距，则补齐后可解锁
        if has_this_gap and len(remaining_gaps) == 0:
            current_unlocked.append(row["产品名"])

    return len(current_unlocked), current_unlocked


def _compute_impact_with_target(action_key, target_value, df_products, metrics):
    """
    计算某个行动项达成后能解锁的产品
    考虑补齐该差距后，产品其他条件也满足的情况
    action_key: 差距项名称
    target_value: 目标值（达标后的值）
    """
    # 构造模拟的metrics（补齐该差距后的指标）
    simulated = dict(metrics)

    if action_key == "纳税评级":
        simulated["纳税信用评级"] = target_value
    elif action_key == "营收":
        simulated["营业收入"] = target_value
    elif action_key == "经营年限":
        simulated["经营年限"] = target_value
    elif action_key == "融资机构数":
        simulated["融资机构数量"] = target_value
    elif action_key == "资产负债率":
        # 需要调整总负债来模拟资产负债率
        # 不直接修改，而是标记
        pass
    elif action_key == "流动比率":
        simulated["流动比率"] = target_value

    # 用模拟指标重新检查每个产品
    unlocked = []
    original_metrics = dict(metrics)

    for _, row in df_products.iterrows():
        # 原始指标下是否不达标
        original_gaps = _check_product_gaps(original_metrics, row)
        if not original_gaps:
            continue  # 已经达标

        # 模拟指标下是否达标
        simulated_gaps = _check_product_gaps(simulated, row)
        # 过滤掉与action_key无关的差距
        related_gaps = [g for g in simulated_gaps if g["item"] == action_key]
        other_gaps = [g for g in simulated_gaps if g["item"] != action_key]

        if not related_gaps and not other_gaps:
            unlocked.append(row["产品名"])

    return len(unlocked), unlocked


def analyze_gaps(metrics, dimension_scores, df_products):
    """
    核心函数：分析企业指标与所有产品准入条件的差距

    参数：
        metrics: dict，企业各项指标
        dimension_scores: dict，8维评分结果（来自diagnosis模块）
        df_products: DataFrame，银行产品库（新Schema）

    返回：
        dict，包含行动优先级、产品差距详情、总结
    """
    # 第一步：遍历所有产品，逐项检查差距
    product_gap_details = []
    all_gap_items = {}  # 按行动项聚合：key=(item, action), value=产品列表

    for _, row in df_products.iterrows():
        gaps = _check_product_gaps(metrics, row)

        if not gaps:
            # 完全达标
            product_gap_details.append({
                "product": row["产品名"],
                "bank": row["银行"],
                "product_type": row.get("产品类型", ""),
                "match_status": "完全匹配",
                "gaps": [],
                "closest_to_qualify": ""
            })
        else:
            # 找到最容易补齐的项（难度分最低）
            easiest = min(gaps, key=lambda g: g["difficulty_score"])
            gap_records = []
            for g in gaps:
                gap_records.append({
                    "item": g["item"],
                    "current": g["current"],
                    "required": g["required"],
                    "gap_size": g["gap_size"],
                    "difficulty": g["difficulty"],
                })

                # 聚合到行动项
                action_key = (g["item"], g["action"])
                if action_key not in all_gap_items:
                    all_gap_items[action_key] = {
                        "item": g["item"],
                        "action": g["action"],
                        "current": g["current"],
                        "target": g["required"],
                        "gap_size": g["gap_size"],
                        "difficulty": g["difficulty"],
                        "difficulty_score": g["difficulty_score"],
                        "products": [row["产品名"]],
                    }
                else:
                    # 合并：取最大差距量
                    all_gap_items[action_key]["products"].append(row["产品名"])
                    if g["gap_size"] > all_gap_items[action_key]["gap_size"]:
                        all_gap_items[action_key]["gap_size"] = g["gap_size"]

            product_gap_details.append({
                "product": row["产品名"],
                "bank": row["银行"],
                "product_type": row.get("产品类型", ""),
                "match_status": "差距匹配",
                "gaps": gap_records,
                "closest_to_qualify": easiest["item"]
            })

    # 第二步：计算每个行动项的影响（解锁产品数）
    action_items = []
    for action_key, info in all_gap_items.items():
        # 计算影响：补齐此项后，仅因此项不达标的产品将解锁
        impact_count, impact_products = _compute_impact(
            info["item"], info, df_products, metrics
        )
        # 如果简单计算结果为0，用更宽泛的方式：统计有多少产品因该项不达标
        if impact_count == 0:
            impact_count = len(info["products"])
            impact_products = info["products"]

        # 计算性价比 = 影响数 / 难度分
        cost_efficiency = impact_count / info["difficulty_score"] if info["difficulty_score"] > 0 else 0

        # 估算达标时间
        estimated_time = _estimate_time(info["difficulty"], info["gap_size"])

        action_items.append({
            "action": info["action"],
            "current": info["current"],
            "target": info["target"],
            "gap": f"{info['gap_size']:.1f}" if info["difficulty"] != "不可控" else "等待",
            "difficulty": info["difficulty"],
            "impact": impact_count,
            "impact_products": impact_products,
            "estimated_time": estimated_time,
            "cost_efficiency": round(cost_efficiency, 2),
            "difficulty_score": info["difficulty_score"],
        })

    # 第三步：按性价比排序（影响÷难度），最高排最前
    action_items.sort(key=lambda x: x["cost_efficiency"], reverse=True)

    # 添加优先级编号
    for i, item in enumerate(action_items):
        item["priority"] = i + 1

    # 第四步：生成总结
    full_match_count = sum(1 for p in product_gap_details if p["match_status"] == "完全匹配")
    gap_match_count = sum(1 for p in product_gap_details if p["match_status"] == "差距匹配")

    # 找出性价比最高的行动
    summary_parts = [f"当前可匹配{full_match_count}个产品"]
    if action_items:
        best = action_items[0]
        summary_parts.append(
            f"补齐{best['action']}后可增加{best['impact']}个产品"
        )
        summary_parts.append(f"建议优先{best['action']}（性价比{best['cost_efficiency']}）")

    summary = "，".join(summary_parts) + "。"

    return {
        "action_plan": action_items,
        "product_gap_details": product_gap_details,
        "summary": summary,
        "full_match_count": full_match_count,
        "gap_match_count": gap_match_count,
    }
