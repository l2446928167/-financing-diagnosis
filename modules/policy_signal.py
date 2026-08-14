"""
modules/policy_signal.py — 政策信号分析模型（对话式 v4.0 新增）

把「目标产业相关政策」量化为可比较的「行业景气程度」等数学模型指标，
并融入整体诊断，而不是单独开一个政策查询窗口。

设计要点：
  - 内置分行业、带时间戳的政策事件库（knowledge 风格的零网络数据集），
    覆盖维度：税收、补贴、信贷/金融、法规/监管、产业规划。
  - compute_policy_signal(industry, as_of)：在 as_of 时点，
    对「近 12 个月」与「前 12 个月」两个窗口分别聚合，输出：
        · 政策景气指数（0-100，越高越利好）
        · 景气等级（利好 / 中性 / 承压）
        · 趋势（上行 / 平稳 / 下行）
        · 对经营稳定性的定性影响
        · 最近若干条政策摘编
    —— 由于事件带时间戳，指数会随 as_of 自然变化，体现「政策随时间变化」。
  - industry_policy_factor(industry, as_of)：返回 [-1, 1] 的行业政策因子，
    供诊断结论做定性融合（如「行业政策偏利好，对成长性形成支撑」）。
  - collect_latest_policies(industry)：实时采集扩展点（默认返回 None，
    由内置数据集兜底）；如需接入官方政策 API / 爬虫，可在此实现并合并。

说明：事件数据为面向演示维护的结构化样本，模型本身为通用量化引擎；
接入真实数据源后无需改动计算逻辑。
"""
import datetime

# 行业键
IND_NEW_ENERGY = "新能源"
IND_TECH = "科技"
IND_MFG = "制造"
IND_GENERAL = "通用"

# 维度
DIM_TAX = "税收"
DIM_SUBSIDY = "补贴"
DIM_CREDIT = "信贷"
DIM_REG = "法规"
DIM_PLAN = "产业规划"

# 政策事件库：每项 (行业, 日期, 维度, 标题, 情感+/-1, 强度0-1, 摘编)
# 情感为正表示利好企业融资/经营，为负表示合规成本或退坡。
POLICY_EVENTS = [
    # ---------- 通用（作用于所有行业） ----------
    ("通用", "2023-08-15", DIM_CREDIT, "普惠金融定向降准", +1, 0.6,
     "央行对聚焦普惠金融的金融机构实施定向降准，释放长期资金。"),
    ("通用", "2024-11-25", DIM_REG, "小微金融监管评价办法", +1, 0.4,
     "建立小微金融服务监管评价体系，引导银行加大普惠投放。"),
    ("通用", "2024-12-31", DIM_CREDIT, "支小再贷款与普惠小微工具", +1, 0.9,
     "支小再贷款余额 1.75 万亿，普惠小微贷款支持工具扩至单户 2000 万。"),
    ("通用", "2025-06-01", DIM_REG, "保障中小企业款项支付条例", +1, 0.7,
     "机关/大型企业应付款项，鼓励应收账款融资确权，改善回款。"),
    ("通用", "2026-03-27", DIM_CREDIT, "深化和规范银税互动", +1, 0.8,
     "将纳税信用转化为融资信用，扩大诚信纳税企业信贷供给。"),

    # ---------- 新能源 ----------
    ("新能源", "2023-01-01", DIM_SUBSIDY, "新能源汽车国补退出", -1, 0.8,
     "延续十余年的国家购车补贴正式退出，行业进入市场化阶段。"),
    ("新能源", "2024-01-01", DIM_TAX, "新能源汽车购置税减免延续", +1, 0.9,
     "新能源汽车购置税减免政策延续，稳定终端需求。"),
    ("新能源", "2024-06-01", DIM_PLAN, "新能源汽车下乡活动", +1, 0.6,
     "组织开展新能源汽车下乡，拓展下沉市场。"),
    ("新能源", "2025-01-15", DIM_SUBSIDY, "汽车以旧换新补贴加码", +1, 0.8,
     "提高汽车以旧换新补贴标准，拉动置换需求。"),
    ("新能源", "2025-09-10", DIM_REG, "新能源汽车碳足迹管理", -1, 0.3,
     "建立碳足迹管理体系，出口与合规成本小幅上升。"),
    ("新能源", "2026-03-20", DIM_SUBSIDY, "充电基础设施补贴", +1, 0.7,
     "加大充换电基础设施财政补贴，完善使用环境。"),

    # ---------- 科技 ----------
    ("科技", "2024-04-01", DIM_CREDIT, "科技创新和技术改造再贷款", +1, 0.9,
     "设立 5000 亿再贷款，支持科技型中小企业首贷与设备更新。"),
    ("科技", "2024-07-01", DIM_PLAN, "专精特新中小企业支持", +1, 0.6,
     "梯度培育专精特新，强化融资与要素保障。"),
    ("科技", "2025-05-10", DIM_TAX, "研发费用加计扣除比例提升", +1, 0.8,
     "提高研发费用加计扣除比例，降低创新成本。"),
    ("科技", "2025-11-01", DIM_REG, "平台经济合规指引", 0, 0.1,
     "明确平台经济常态化监管，合规预期趋稳。"),
    ("科技", "2026-02-15", DIM_PLAN, "数据要素市场化配置", +1, 0.5,
     "推进数据要素市场化，利好数据类科技企业。"),

    # ---------- 制造 ----------
    ("制造", "2024-03-01", DIM_PLAN, "制造业重点产业链高质量发展", +1, 0.7,
     "实施重点产业链高质量发展行动，稳定供应链。"),
    ("制造", "2025-02-01", DIM_CREDIT, "设备更新改造专项再贷款", +1, 0.8,
     "设立设备更新改造专项再贷款，支持技改投资。"),
    ("制造", "2025-08-01", DIM_TAX, "制造业减税降费延续", +1, 0.6,
     "制造业减税降费政策延续，减轻经营负担。"),
    ("制造", "2026-04-01", DIM_PLAN, "绿色制造体系", +1, 0.4,
     "推进绿色制造体系，引导低碳转型投资。"),
]


def resolve_industry(industry_text):
    """把用户输入的行业文本映射到跟踪行业键。"""
    if not industry_text:
        return IND_MFG
    t = industry_text
    if any(k in t for k in ("新能源", "电动", "锂电", "光伏", "电池", "汽车")):
        return IND_NEW_ENERGY
    if any(k in t for k in ("科技", "软件", "互联网", "电子", "半导体", "人工智能", "AI", "大数据", "信息")):
        return IND_TECH
    return IND_MFG


def _events_for(industry_key, as_of, window_start, window_end):
    """取 (行业==通用 或 ==industry_key) 且日期落在 [window_start, window_end] 的事件。"""
    out = []
    for ev in POLICY_EVENTS:
        ind, date_s, dim, title, sent, inten, note = ev
        if ind not in (IND_GENERAL, industry_key):
            continue
        try:
            d = datetime.date.fromisoformat(date_s)
        except Exception:
            continue
        if window_start <= d <= window_end:
            out.append((d, dim, title, sent, inten, note))
    return out


def _aggregate(events):
    """聚合事件 → (平均情感强度, 事件数)。平均情感强度 ∈ [-1, 1]。"""
    if not events:
        return 0.0, 0
    total = sum(sent * inten for (_d, _dm, _t, sent, inten, _n) in events)
    return total / len(events), len(events)


def compute_policy_signal(industry_text, as_of=None):
    """计算 as_of 时点的行业政策信号。as_of 缺省为今天。"""
    if as_of is None:
        as_of = datetime.date.today()
    elif isinstance(as_of, str):
        as_of = datetime.date.fromisoformat(as_of[:10])
    industry_key = resolve_industry(industry_text)

    cur_start = as_of - datetime.timedelta(days=365)
    prev_start = as_of - datetime.timedelta(days=730)
    cur_events = _events_for(industry_key, as_of, cur_start, as_of)
    prev_events = _events_for(industry_key, as_of, prev_start, cur_start)

    cur_avg, cur_n = _aggregate(cur_events)
    prev_avg, _ = _aggregate(prev_events)

    # 景气指数：50 为中性，每单位平均情感强度映射到 ±40
    index = max(0.0, min(100.0, round(50 + 40 * cur_avg, 1)))
    if index >= 65:
        level = "利好"
    elif index >= 45:
        level = "中性"
    else:
        level = "承压"

    diff = cur_avg - prev_avg
    if diff > 0.08:
        trend = "上行"
    elif diff < -0.08:
        trend = "下行"
    else:
        trend = "平稳"

    if cur_avg >= 0.15:
        effect = "偏正面，对成长性与经营稳定性形成支撑"
    elif cur_avg <= -0.1:
        effect = "偏负面，需关注合规成本与需求波动"
    else:
        effect = "中性，暂无明显方向性影响"

    recent = sorted(cur_events, key=lambda x: x[0], reverse=True)[:4]
    recent_list = [
        f"{d.strftime('%Y-%m')} {title}（{'利好' if s > 0 else '承压' if s < 0 else '中性'}）"
        for (d, _dm, title, s, _i, _n) in recent
    ]
    if not recent_list:
        recent_list = ["近 12 个月暂无收录的专项政策事件（以通用政策为主）"]

    return {
        "industry": industry_key,
        "as_of": as_of.isoformat(),
        "index": index,
        "level": level,
        "trend": trend,
        "effect": effect,
        "factor": round(cur_avg, 3),
        "event_count": cur_n,
        "recent": recent_list,
    }


def industry_policy_factor(industry_text, as_of=None):
    """返回 [-1, 1] 的行业政策因子，供诊断结论融合。"""
    return compute_policy_signal(industry_text, as_of)["factor"]


def collect_latest_policies(industry_text, as_of=None):
    """
    实时采集扩展点：默认返回 None，由内置数据集兜底。
    如需接入官方政策 API / 爬虫，可在此实现（请求、解析、归一化为
    POLICY_EVENTS 同构记录并合并），保持 compute_policy_signal 逻辑不变。
    """
    return None
