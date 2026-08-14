"""
app.py v4.0 — 对话式融资诊断助手

相对 v3.0 的改造（均不改动算法模块 modules/ 与 utils/vector_store.py）：
- 移除侧边栏：API Key 改为通过环境变量 / .env 加载（config.py），界面不再填写。
- 聊天框直接支持上传文件（PDF / Excel / CSV），主流 AI 助手式交互。
- 增加对话持久化：可「从历史对话继续」或「新建对话」；每次诊断自动保存企业数据，
  同企业再次上传时自动比对前后变化，输出经营与资金健康度演变趋势。
- 政策改为「信号分析模型」：内置分行业时间序列政策库量化行业景气指数并融入诊断，
  不再提供独立的政策查询窗口。
- 修复乱码：清理 Emoji 与含 Emoji 的字段；字体栈兜底稳健。
- 去除界面内部架构表述，只呈现结果与结论。
"""
import re
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from config import DEEPSEEK_API_KEY, MODEL_NAME   # API Key 仅来自环境变量 / .env
from utils import persistence as PS
from utils.ui_style import inject_css, badge, score_level, dim_level

load_dotenv()

st.set_page_config(page_title="融资诊断助手", layout="centered")

inject_css()

APP_VERSION = "v4.0 (2026-08-14)"
PRODUCT_CSV = "knowledge/bank_products/products.csv"
RAG_INDEX = "knowledge/rag_corpus/bm25_index.json"

# ======================== session_state 初始化 ========================
_DEFAULTS = {
    "metrics": {}, "metrics_rev": 0, "raw_text": "", "df": None,
    "diagnosis_result": None, "full_metrics": None, "llm_text": None,
    "ai_extracted": False, "match_cache": None, "pdf_bytes": None,
    "ai_rec": None, "qa_result": None, "_ai_changes": [], "_file_sig": None,
    "messages": [], "stage": "init", "ml": None, "gap_cache": None,
    "api_key": DEEPSEEK_API_KEY, "model": MODEL_NAME,
    "conv_id": None, "conv_title": "新对话", "enterprise_name": "默认企业",
    "policy_result": None, "_upload_name": "",
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


# ======================== 会话持久化 ========================
def _new_conversation():
    """新建一个空白会话（同时清掉当前数据）。"""
    _reset_data()
    st.session_state.conv_id = PS.new_conv_id()
    st.session_state.conv_title = "新对话"
    st.session_state.enterprise_name = "默认企业"
    st.session_state.messages = []
    st.session_state.stage = "init"
    _save_current_conversation()


def _load_conversation_state(conv_id):
    conv = PS.load_conversation(conv_id)
    if not conv:
        _new_conversation()
        return
    st.session_state.conv_id = conv.get("id", conv_id)
    st.session_state.conv_title = conv.get("title", "对话")
    st.session_state.enterprise_name = conv.get("enterprise_name", "默认企业")
    st.session_state.messages = conv.get("messages", [])
    st.session_state.stage = conv.get("stage", "init")
    st.session_state.full_metrics = conv.get("full_metrics")
    st.session_state.diagnosis_result = conv.get("diagnosis_result")
    st.session_state.ml = conv.get("ml")
    st.session_state.llm_text = conv.get("llm_text")
    st.session_state.gap_cache = conv.get("gap_cache")
    st.session_state.metrics = conv.get("metrics", {})
    st.session_state.policy_result = conv.get("policy_result")


def _save_current_conversation():
    cid = st.session_state.get("conv_id")
    if not cid:
        return
    title = st.session_state.get("conv_title") or "新对话"
    if title == "新对话":
        for _m in st.session_state.messages:
            if _m.get("role") == "user" and _m.get("content"):
                title = _m["content"][:18]
                break
    conv = {
        "id": cid,
        "title": title,
        "enterprise": st.session_state.get("enterprise_name", "默认企业"),
        "messages": st.session_state.messages,
        "stage": st.session_state.stage,
        "full_metrics": st.session_state.full_metrics,
        "diagnosis_result": st.session_state.diagnosis_result,
        "ml": st.session_state.ml,
        "llm_text": st.session_state.llm_text,
        "gap_cache": st.session_state.gap_cache,
        "enterprise_name": st.session_state.get("enterprise_name", "默认企业"),
        "policy_result": st.session_state.get("policy_result"),
    }
    PS.save_conversation(conv)
    st.session_state.conv_title = title


# ======================== 辅助函数 ========================
def _to_num(v):
    """把指标字符串安全转成 float。"""
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(v).replace(",", "").replace("，", ""))
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


def generate_diagnosis_text(full_metrics, dims, overall, model_choice, api_key):
    """用 LLM 生成诊断总结、风险点、改善建议，失败返回 None。"""
    from utils.llm_helper import call_llm

    system_prompt = """
    你是一位资深的小微企业融资顾问。根据提供的企业财务指标和8维度健康评分，
    请输出以下内容（用中文）：
    1. 总体评价：100字以内的概括性评价。
    2. 风险点：不超过8条具体风险，每条以"- "开头，语气客观，使用"建议"而非"你应该"。
    3. 改善建议：针对每个不健康维度给出2-3条具体、可操作的建议，以"- "开头。
    请严格按照以下格式输出，不要多余文字：
    【总体评价】
    <内容>
    【风险点】
    - 风险1
    【改善建议】
    - 建议1
    """
    user_prompt = f"""
    企业指标：
    总资产：{full_metrics.get('总资产', '未填写')} 万元
    营业收入：{full_metrics.get('营业收入', '未填写')} 万元
    净利润：{full_metrics.get('净利润', '未填写')} 万元
    经营年限：{full_metrics.get('经营年限', '未填写')} 年
    行业：{full_metrics.get('行业', '未填写')}
    客户集中度：{full_metrics.get('客户集中度', '未填写')}
    平均融资利率：{full_metrics.get('平均融资利率', '未填写')}%
    纳税信用评级：{full_metrics.get('纳税信用评级', '未填写')}
    实控人征信状态：{full_metrics.get('实控人征信状态', '未填写')}
    法院执行记录：{full_metrics.get('法院执行记录', '未填写')}
    融资机构数量：{full_metrics.get('融资机构数量', '未填写')}
    营收增长率：{full_metrics.get('营收增长率', '未填写')}%
    净利润增长率：{full_metrics.get('净利润增长率', '未填写')}%

    各维度评分（满分10）：
    """
    for dim, score in dims.items():
        user_prompt += f"\n- {dim}：{score}"
    user_prompt += f"\n总体健康评分：{overall}/10"

    return call_llm(system_prompt, user_prompt, model_choice, api_key)


def _parse_llm_sections(llm_text):
    """解析 LLM 输出为 (总体评价, 风险点列表, 建议列表)。"""
    if not llm_text:
        return "", [], []
    sm = re.search(r"【总体评价】\s*(.*?)\s*【风险点】", llm_text, re.DOTALL)
    rm = re.search(r"【风险点】\s*(.*?)\s*【改善建议】", llm_text, re.DOTALL)
    gm = re.search(r"【改善建议】\s*(.*)", llm_text, re.DOTALL)

    def _lines(m):
        if not m:
            return []
        return [ln.lstrip("- ").strip() for ln in m.group(1).strip().split("\n")
                if ln.strip().startswith("-")]

    return (sm.group(1).strip() if sm else ""), _lines(rm), _lines(gm)


@st.cache_data
def _load_products():
    return pd.read_csv(PRODUCT_CSV, encoding="utf-8")


@st.cache_resource
def _load_rag_index():
    import json
    from utils.vector_store import load_index
    idx = load_index(RAG_INDEX)
    try:
        with open(RAG_INDEX, encoding="utf-8") as f:
            meta = json.load(f).get("meta", {})
    except Exception:
        meta = {}
    return idx, meta


def _ensure_matches():
    if st.session_state.get("match_cache") is not None:
        return st.session_state.match_cache
    from modules.product_matching import match_products
    matches = match_products(st.session_state.full_metrics, _load_products())
    st.session_state.match_cache = matches
    return matches


def _report_rag_citations():
    try:
        from utils.vector_store import retrieve
        idx, meta = _load_rag_index()
        hits = retrieve(idx, "小微企业融资政策支持", k=3, category="policy")
        cites = [
            f"{h['title']}（{h['source']}）"
            f"{(' ' + h['clause']) if h.get('clause') else ''}（条款摘编）"
            for h in hits
        ]
        return cites, meta.get("asof", "")
    except Exception:
        return [], ""


def _gen_ai_recommendation(full_metrics, matches):
    """智能产品推荐 / 无匹配时的改善建议。"""
    from utils.llm_helper import call_llm
    if matches:
        product_list = "\n".join([
            f"- {m['匹配度']} {m['产品名']}（{m['银行']}），额度：{m['额度']}万元，"
            f"利率：{m['利率']}%，差距：{m['差距说明']}"
            for m in matches
        ])
        prompt = f"""
        根据以下企业情况和匹配到的银行产品，请你作为融资顾问，
        用中文给出简明扼要的推荐建议（200字以内），包括：
        - 最推荐哪1-2个产品，为什么适合该企业？
        - 申请时需要注意什么（如材料准备、时间节点）？
        - 如果产品是"差距匹配"，企业应优先补齐哪个条件？

        企业情况：
        总资产：{full_metrics.get('总资产', '未填写')}万元
        营业收入：{full_metrics.get('营业收入', '未填写')}万元
        经营年限：{full_metrics.get('经营年限', '未填写')}年
        行业：{full_metrics.get('行业', '未填写')}
        客户集中度：{full_metrics.get('客户集中度', '未填写')}
        现有融资利率：{full_metrics.get('平均融资利率', '未填写')}%
        纳税信用评级：{full_metrics.get('纳税信用评级', '未填写')}
        征信状态：{full_metrics.get('实控人征信状态', '未填写')}

        匹配产品列表：
        {product_list}

        请直接输出推荐内容，不要使用markdown标题。
        """
        system = "你是资深小微企业信贷顾问，语言简洁专业，用'建议'而非'你应该'。"
        return call_llm(system, prompt, st.session_state.model,
                        st.session_state.api_key, max_tokens=500)
    prompt = f"""
    企业当前未匹配到任何信贷产品，请根据企业情况给出2-3条具体改善建议，
    帮助其未来达到银行准入条件。企业情况：
    总资产：{full_metrics.get('总资产', '未填写')}万元
    营业收入：{full_metrics.get('营业收入', '未填写')}万元
    经营年限：{full_metrics.get('经营年限', '未填写')}年
    行业：{full_metrics.get('行业', '未填写')}
    纳税信用评级：{full_metrics.get('纳税信用评级', '未填写')}
    征信状态：{full_metrics.get('实控人征信状态', '未填写')}
    """
    system = "你是小微企业融资改善顾问，给出可操作的建议。"
    return call_llm(system, prompt, st.session_state.model,
                    st.session_state.api_key, max_tokens=400)


# ======================== 图表 ========================
def _dims_chart(dims):
    try:
        import altair as alt
        df = pd.DataFrame({"维度": list(dims.keys()), "评分": list(dims.values())})
        df["等级"] = df["评分"].apply(
            lambda s: "健康" if s >= 7 else ("关注" if s >= 4 else "高风险"))
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("评分:Q", scale=alt.Scale(domain=[0, 10]), title=None),
                y=alt.Y("维度:N", sort="-x", title=None),
                color=alt.Color(
                    "等级:N",
                    scale=alt.Scale(domain=["健康", "关注", "高风险"],
                                    range=["#2BA471", "#D98B1F", "#E5484D"]),
                    legend=None),
                tooltip=["维度", "评分"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(pd.DataFrame({"评分": dims}))


def _shap_chart(contribs):
    rows = sorted(
        ((k, v) for k, v in contribs.items() if k not in ("bias", "违约概率")),
        key=lambda kv: -abs(kv[1]))
    df = pd.DataFrame({"因子": [k for k, _ in rows],
                       "贡献": [round(v, 3) for _, v in rows]})
    try:
        import altair as alt
        df["方向"] = df["贡献"].apply(lambda v: "推高风险" if v >= 0 else "拉低风险")
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("贡献:Q", title=None),
                y=alt.Y("因子:N", sort="-x", title=None),
                color=alt.Color(
                    "方向:N",
                    scale=alt.Scale(domain=["推高风险", "拉低风险"],
                                    range=["#E5484D", "#2BA471"]),
                    legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["因子", "贡献"],
            )
            .properties(height=max(160, 26 * len(df)))
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(df.set_index("因子"))


# ======================== 文件解析 / 录入 ========================
_MANUAL_KEYS = ["总资产", "总负债", "营业收入", "营业成本", "净利润",
                "流动负债", "经营活动现金流净额", "利息费用",
                "应收账款", "存货", "流动比率"]


def _handle_file(uploaded_file):
    from modules.data_input import parse_financial_data, auto_extract_metrics
    sig = (uploaded_file.name, uploaded_file.size)
    if st.session_state._file_sig != sig:
        st.session_state._file_sig = sig
        st.session_state._upload_name = uploaded_file.name
        st.session_state.ai_extracted = False
        st.session_state.diagnosis_result = None
        st.session_state.full_metrics = None
        st.session_state.llm_text = None
        st.session_state.match_cache = None
        st.session_state.pdf_bytes = None
        st.session_state.ai_rec = None
        st.session_state.ml = None
        st.session_state.gap_cache = None
        st.session_state._ai_changes = []
        with st.spinner("正在解析文件…"):
            try:
                raw_text, df = parse_financial_data(uploaded_file)
            except Exception as e:
                st.error(f"解析失败：{e}")
                return
        st.session_state.raw_text = raw_text
        st.session_state.df = df
        st.session_state.metrics = auto_extract_metrics(raw_text, df)
        st.session_state.metrics_rev += 1
        st.toast("文件解析完成，请确认下方指标")


def _render_ai_extract():
    if not st.session_state.get("api_key"):
        st.caption("未配置 API Key（环境变量 DEEPSEEK_API_KEY），暂不可用智能提取；"
                   "规则提取仍可用。")
        return
    if st.button("智能提取财务指标", key="ai_extract"):
        with st.spinner("正在分析文件内容…"):
            try:
                from modules.data_input import llm_extract_metrics
                from utils.llm_helper import call_llm
                ai_metrics = llm_extract_metrics(
                    st.session_state.raw_text, st.session_state.df,
                    st.session_state.model, st.session_state.api_key, call_llm)
            except Exception as e:
                st.error(f"智能提取异常：{type(e).__name__} – {e}")
                return
        if not ai_metrics:
            st.error("智能提取未能获得有效结果。")
            return
        old = st.session_state.metrics
        changes = []
        for k, nf in ai_metrics.items():
            if k.startswith("__"):
                continue
            of = old.get(k, "")
            ov = of.get("value", "") if isinstance(of, dict) else str(of)
            nv = nf.get("value", "") if isinstance(nf, dict) else str(nf)
            if nv and ov != nv:
                changes.append(f"{k}: {ov or '(空)'} → {nv}")
        st.session_state.metrics = ai_metrics
        st.session_state.ai_extracted = True
        st.session_state.metrics_rev += 1
        st.session_state._ai_changes = changes
        st.toast(f"智能提取完成，更新 {len(changes)} 个指标")
        st.rerun()
    if st.session_state.get("_ai_changes"):
        with st.expander(f"智能提取变更（{len(st.session_state._ai_changes)} 项）"):
            for c in st.session_state._ai_changes:
                st.markdown(f"- `{c}`")


def _render_verification():
    verification = st.session_state.metrics.get("__verification__")
    if not verification:
        return
    with st.expander("会计校验结果", expanded=False):
        for check in verification:
            status = check.get("是否通过")
            name = check.get("检查项", "")
            actual = check.get("实际值", "")
            expected = check.get("预期值", "")
            if status is True:
                st.markdown(f"{badge('通过', 'green')} **{name}**：{actual}")
            elif status is False:
                st.markdown(f"{badge('偏差', 'red')} **{name}**：{actual}，预期：{expected}")
            else:
                st.markdown(f"{badge('无法校验', 'amber')} **{name}**：{actual}")


def _seed_manual_template():
    st.session_state.metrics = {
        k: {"value": "", "unit": ("倍" if k == "流动比率" else "万元"), "page": ""}
        for k in _MANUAL_KEYS
    }
    st.session_state.metrics_rev += 1


def _run_diagnosis(edited, s, enterprise_name):
    from modules.diagnosis import diagnose

    metrics = {}
    for _, row in edited.iterrows():
        metrics[row["指标"]] = {"value": str(row["数值"]),
                                "unit": str(row["单位"]), "page": str(row["来源页"])}
    for k in st.session_state.metrics:
        if k.startswith("__"):
            metrics[k] = st.session_state.metrics[k]
    st.session_state.metrics = metrics

    flat = {k: (v.get("value", "") if isinstance(v, dict) else str(v))
            for k, v in metrics.items() if not k.startswith("__")}
    full_metrics = {
        **flat,
        "经营年限": s["operating_years"],
        "客户集中度": s["customer_concentration"],
        "平均融资利率": s["avg_interest_rate"],
        "行业": s["industry"],
        "应收账款_3月内占比": s["ar_less_3m"],
        "应收账款_3_12月占比": s["ar_3_12m"],
        "应收账款_超12月占比": s["ar_over_12m"],
        "纳税信用评级": "" if s["tax_credit_rating"] == "未评级" else s["tax_credit_rating"],
        "实控人征信状态": s["controller_credit"],
        "法院执行记录": s["court_execution"],
        "融资机构数量": s["financing_institution_count"],
        "营收增长率": s["revenue_growth_rate"],
        "净利润增长率": s["profit_growth_rate"],
        "可提供抵押": s["can_collateral"] == "可提供",
        "行业周期信号": s["industry_cycle"],
    }
    st.session_state.enterprise_name = enterprise_name
    with st.spinner("正在分析…"):
        result = diagnose(full_metrics)

    st.session_state.full_metrics = full_metrics
    st.session_state.diagnosis_result = result
    st.session_state.match_cache = None
    st.session_state.pdf_bytes = None
    st.session_state.ai_rec = None
    st.session_state.gap_cache = None

    if st.session_state.get("api_key"):
        with st.spinner("正在生成诊断总结…"):
            st.session_state.llm_text = generate_diagnosis_text(
                full_metrics, result["dimension_scores"], result["overall_score"],
                st.session_state.model, st.session_state.api_key)
    else:
        st.session_state.llm_text = None


def _clean_conclusion(score, proba):
    """把内部结论映射为中性、不暴露架构的对外表述。"""
    if proba is None:
        if score >= 7:
            return "综合评分健康，融资可行性较好。"
        if score >= 4:
            return "综合评分中等，存在若干可改善项。"
        return "综合评分偏低，建议优先改善财务结构。"
    if score >= 7 and proba < 0.35:
        return "综合评估为低风险，建议优先推荐。"
    if score < 4 and proba > 0.60:
        return "综合评估为高风险，建议谨慎。"
    if score >= 7 and proba >= 0.50:
        return "评估存在预警信号，建议补充材料并人工复核。"
    if score < 4 and proba <= 0.35:
        return "评估存在分歧，建议补充材料并人工复核。"
    return "建议补充材料后再评估。"


def _after_diagnose():
    res = st.session_state.diagnosis_result
    full = st.session_state.full_metrics
    dims = res["dimension_scores"]
    score = res["overall_score"]
    level, label = score_level(score)

    proba = conclusion = None
    contribs = None
    try:
        from modules.ml_model import (predict_default_proba, explain_statement)
        _CAT = {"纳税信用评级", "实控人征信状态", "法院执行记录",
                "客户集中度", "行业", "可提供抵押"}
        statement = {k: (v if k in _CAT else _to_num(v))
                     for k, v in full.items()}
        proba = predict_default_proba(statement)
        try:
            contribs = explain_statement(statement)
        except Exception:
            contribs = None
    except Exception:
        pass
    st.session_state.ml = {"proba": proba, "conclusion": None, "contribs": contribs}

    # 政策信号（融入整体分析，不单独开窗口）
    policy = None
    try:
        from modules.policy_signal import compute_policy_signal
        policy = compute_policy_signal(full.get("行业", ""))
    except Exception:
        policy = None
    st.session_state.policy_result = policy

    # 企业数据快照 + 跨期对比
    enterprise = st.session_state.get("enterprise_name") or "默认企业"
    snap = PS.build_snapshot(full, res, st.session_state.get("_upload_name", ""), enterprise)
    comparison = PS.save_enterprise_snapshot(enterprise, snap)

    summary, risks, suggestions = _parse_llm_sections(st.session_state.get("llm_text"))
    snap_payload = {
        "overall": score, "level": level, "label": label, "proba": proba,
        "conclusion": _clean_conclusion(score, proba), "dims": dims,
        "contribs": contribs, "summary": summary, "risks": risks,
        "suggestions": suggestions, "policy": policy,
    }
    st.session_state.stage = "diagnosed"
    add_assistant("已完成金融健康诊断，结果如下：",
                  payload={"type": "diagnosis", "data": snap_payload})
    if comparison and comparison.get("has_prev"):
        add_assistant("已记录本次企业数据，并与上次上传自动对比：",
                      payload={"type": "trend", "data": comparison})
    add_assistant("接下来可以：查看信贷产品匹配，或生成 PDF 报告。")


# ======================== 对话消息工具 ========================
def add_msg(role, content=None, payload=None):
    st.session_state.messages.append(
        {"role": role, "content": content, "payload": payload})


def add_user(content):
    add_msg("user", content)


def add_assistant(content=None, payload=None):
    add_msg("assistant", content, payload)


def _reset_data():
    keys = ["metrics", "raw_text", "df", "diagnosis_result", "full_metrics",
            "llm_text", "ai_extracted", "match_cache", "pdf_bytes", "ai_rec",
            "qa_result", "ml", "gap_cache", "_file_sig", "_ai_changes"]
    for k in keys:
        if k == "metrics":
            st.session_state.metrics = {}
        elif k == "_file_sig":
            st.session_state._file_sig = None
        elif k == "_ai_changes":
            st.session_state._ai_changes = []
        else:
            st.session_state[k] = None
    st.session_state.stage = "init"


def _render_message(m):
    with st.chat_message(m["role"]):
        if m.get("content"):
            st.markdown(m["content"], unsafe_allow_html=True)
        p = m.get("payload")
        if p:
            _render_payload(p)


def _render_payload(p):
    t = p.get("type")
    if t == "diagnosis":
        _render_diagnosis_payload(p["data"])
    elif t == "products":
        _render_products_payload(p["data"])
    elif t == "qa":
        _render_qa_payload(p["data"])
    elif t == "report":
        _render_report_payload()
    elif t == "trend":
        _render_trend_payload(p["data"])


# ---------------- 诊断结果（助手消息内渲染） ----------------
def _render_diagnosis_payload(d):
    res_score = d["overall"]
    level, label = d["level"], d["label"]
    o1, o2, o3 = st.columns([1, 1, 2])
    with o1:
        st.metric("总体健康评分", f"{res_score} / 10")
        st.markdown(badge(label, level), unsafe_allow_html=True)
    with o2:
        st.metric("违约风险概率", "未接入" if d["proba"] is None else f"{d['proba'] * 100:.1f}%")
    with o3:
        st.markdown("**综合结论**")
        st.info(d["conclusion"] if d["conclusion"]
                else "外部风险模型未接入，当前为综合评分结论。")

    st.markdown("**八维健康评分**")
    _dims_chart(d["dims"])
    badge_cols = st.columns(4)
    for i, (dim, sc) in enumerate(d["dims"].items()):
        with badge_cols[i % 4]:
            st.markdown(f"{dim}　{badge(f'{sc}/10', dim_level(sc))}",
                        unsafe_allow_html=True)

    if d["contribs"]:
        with st.expander("风险归因（各因子对违约概率的贡献方向与强度）"):
            _shap_chart(d["contribs"])
            st.caption("正值推高风险、负值拉低风险；以下为风险归因分析，仅供对照参考。")

    summary, risks, suggestions = d["summary"], d["risks"], d["suggestions"]
    if summary:
        st.markdown("**诊断总结**")
        st.markdown(summary)
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**核心风险点**")
        shown = risks or st.session_state.diagnosis_result.get("risks", [])
        if shown:
            for r_ in shown:
                st.markdown(f"- {r_}")
        else:
            st.success("未发现明显风险点。")
    with r2:
        st.markdown("**改善行动建议**")
        shown = suggestions or st.session_state.diagnosis_result.get("suggestions", [])
        for s_ in shown:
            st.markdown(f"- {s_}")

    # 行业政策环境（政策信号模型，已融入分析）
    pol = d.get("policy")
    if pol:
        st.markdown("**行业政策环境**")
        st.markdown(
            f"政策景气指数 <b>{pol['index']} / 100</b>（{pol['level']}）　"
            f"趋势：{pol['trend']}　对经营稳定性：{pol['effect']}",
            unsafe_allow_html=True)
        with st.expander("近期政策摘编", expanded=False):
            for ev in pol.get("recent", []):
                st.markdown(f"- {ev}")

    st.caption("以上诊断由智能分析引擎生成，仅供参考，不构成金融建议。")


# ---------------- 跨期趋势（助手消息内渲染） ----------------
def _render_trend_payload(data):
    verdict = data.get("verdict", "持平")
    vlevel = {"改善": "green", "恶化": "red", "持平": "amber"}.get(verdict, "amber")
    st.markdown(f"**经营与资金健康度演变趋势**　{badge(verdict, vlevel)}")
    rows = data.get("rows", [])
    if rows:
        tdf = pd.DataFrame([{
            "指标": r["name"], "上次": r["before"], "本次": r["after"],
            "变化": r["delta"], "变化%": r["pct"],
            "方向": "改善" if r["improved"] else "走弱",
        } for r in rows])
        st.dataframe(tdf, use_container_width=True, hide_index=True)
    st.markdown(data.get("summary", ""))
    st.caption("跨期对比基于同一企业历史上传数据自动生成；指标口径以当次上传文件为准。")


# ---------------- 产品匹配（助手消息内渲染） ----------------
def _gap_action_rows(gap):
    if not gap:
        return []
    action_plan = gap.get("action_plan", [])
    rows = []
    for act in action_plan:
        mode = act.get("impact_mode", "unlock")
        if mode == "unlock":
            impact_label = str(act["impact"])
            products_label = ("、".join(act["impact_products"][:5])
                              if act["impact_products"] else "—")
        else:
            impact_label = f"{act['impact']}（尚有其他差距）"
            products_label = (("、".join(act["impact_products"][:5]) + "…")
                              if act["impact_products"] else "—")
        rows.append({
            "优先级": act["priority"], "行动": act["action"],
            "当前值": act["current"], "目标值": act["target"],
            "难度": act["difficulty"], "影响产品数": impact_label,
            "相关产品": products_label, "预计时间": act["estimated_time"],
            "性价比": act["cost_efficiency"],
        })
    return rows


def _render_products_payload(data):
    matches = data.get("matches") or []
    if matches:
        display_cols = ["匹配度", "产品名", "银行", "产品类型",
                        "额度", "利率", "准入条件", "差距说明"]
        st.dataframe(pd.DataFrame(matches)[display_cols],
                     use_container_width=True, hide_index=True)
    else:
        st.warning("未找到匹配的信贷产品，建议改善财务状况后再查询。")

    if data.get("gap_summary"):
        st.markdown(f"**差距分析总结**　{data['gap_summary']}")

    action_rows = data.get("action_rows") or []
    if action_rows:
        with st.expander("差距分析与行动方案（按性价比排序）"):
            st.caption("性价比 = 解锁产品数 ÷ 提升难度分，数值越高越应优先。")
            st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

    if st.session_state.get("api_key") and not st.session_state.get("ai_rec"):
        if st.button("生成智能产品推荐说明", key="ai_rec_btn2"):
            with st.spinner("正在分析最佳产品方案…"):
                st.session_state.ai_rec = _gen_ai_recommendation(
                    st.session_state.full_metrics, matches)
            st.rerun()
    if st.session_state.get("ai_rec"):
        st.markdown("**智能产品推荐说明**")
        st.markdown(st.session_state.ai_rec)


# ---------------- 问答（助手消息内渲染） ----------------
def _render_qa_payload(data):
    if data.get("empty"):
        st.info("该问题未被本模块语料收录（诚实返回空，不编造）。"
                "可改问信贷产品或普惠政策。")
        return
    if data.get("mock"):
        st.caption("（当前为本地检索结果）")
    st.markdown(data.get("text", ""))
    cites = data.get("citations") or []
    if cites:
        with st.expander(f"参考清单（{len(cites)} 条；政策条目为条款摘编）"):
            for c in cites:
                st.markdown(f"- {c}")
    if data.get("asof"):
        st.caption(f"检索截至 {data['asof']}")


# ---------------- 报告（助手消息内渲染） ----------------
def _render_report_payload():
    if st.session_state.get("pdf_bytes"):
        st.download_button("下载 PDF 报告", data=st.session_state.pdf_bytes,
                           file_name="融资诊断报告.pdf",
                           mime="application/pdf", type="primary")
    else:
        st.info("报告尚未生成，请稍候或重试。")


# ======================== 行为分发 ========================
def do_diagnose():
    if not st.session_state.metrics:
        add_assistant("还没有可诊断的数据，请在对话框上传财务文件，或点击下方「手动录入指标」。")
        return
    if st.session_state.diagnosis_result and st.session_state.stage == "diagnosed":
        _after_diagnose()
        return
    add_assistant("请在下方指标表单确认后，点击「开始金融健康诊断」。",)


def do_products():
    try:
        matches = _ensure_matches()
    except Exception as e:
        add_assistant(f"产品匹配出错：{e}")
        return
    try:
        from modules.gap_analysis import analyze_gaps
        gap = analyze_gaps(st.session_state.full_metrics,
                           st.session_state.diagnosis_result["dimension_scores"],
                           _load_products())
        st.session_state.gap_cache = gap
    except Exception:
        gap = None
    snap = {"matches": matches,
            "gap_summary": gap.get("summary") if gap else None,
            "action_rows": _gap_action_rows(gap) if gap else []}
    add_assistant("已为你匹配信贷产品并量化差距：",
                  payload={"type": "products", "data": snap})


def do_report():
    full = st.session_state.full_metrics
    res = st.session_state.diagnosis_result
    matches = _ensure_matches()
    rag_cites, rag_asof = _report_rag_citations()
    ml = st.session_state.ml or {}
    gap = st.session_state.gap_cache
    if gap is None:
        try:
            from modules.gap_analysis import analyze_gaps
            gap = analyze_gaps(full, res["dimension_scores"], _load_products())
        except Exception:
            gap = None
    if st.session_state.get("api_key") and not st.session_state.get("ai_rec"):
        with st.spinner("生成智能产品推荐…"):
            st.session_state.ai_rec = _gen_ai_recommendation(full, matches)
    summary, risks, suggestions = _parse_llm_sections(st.session_state.get("llm_text"))
    try:
        from modules.report_generator import generate_pdf
        with st.spinner("正在生成 PDF 报告…"):
            st.session_state.pdf_bytes = generate_pdf(
                full, res, matches, _load_products(),
                ai_summary=summary,
                ai_risks="\n".join(f"- {r}" for r in risks),
                ai_suggestions="\n".join(f"- {x}" for x in suggestions),
                ai_recommendation=st.session_state.get("ai_rec") or "",
                rag_citations=rag_cites, rag_asof=rag_asof,
                ml_proba=ml.get("proba"), ml_conclusion=ml.get("conclusion"),
                shap_contribs=ml.get("contribs"), gap_result=gap,
                policy_result=st.session_state.get("policy_result"))
        add_assistant("诊断报告已生成，点击下方按钮下载：",
                      payload={"type": "report"})
    except Exception as e:
        add_assistant(f"报告生成失败：{e}")


def _file_qa(question):
    """基于本次上传的财务文件内容作答；无法命中则返回 None 以回落到政策检索。"""
    raw = st.session_state.get("raw_text") or ""
    df = st.session_state.get("df")
    has = bool(raw.strip()) or (df is not None and not getattr(df, "empty", True))
    if not has:
        return None

    _STOP = {"的", "了", "是", "在", "和", "与", "及", "或", "这", "那", "我", "你",
             "他", "它", "吗", "呢", "啊", "吧", "请", "问", "什么", "多少", "如何",
             "怎么", "为什么", "分析", "对比", "说明", "计算", "企业", "公司", "这家", "资料"}
    kws = [w for w in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", question)
           if w not in _STOP]
    if not kws:
        return None

    if raw:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        hits = []
        for l in lines:
            if any(k in l for k in kws):
                hits.append(l)
        ctx = "\n".join(hits[:30]) if hits else raw[:2500]
        local_hits = hits
    else:
        ctx = df.head(30).to_csv(index=False)
        local_hits = []

    if st.session_state.get("api_key"):
        from utils.llm_helper import call_llm
        prompt = (
            "根据以下企业财务资料回答问题，仅基于资料作答；"
            "资料中没有的信息请明确说明无法回答。\n\n"
            f"资料：\n{ctx}\n\n问题：{question.strip()}"
        )
        ans = call_llm("你是财务分析助手，仅基于给定的企业资料作答，语言简洁。",
                       prompt, st.session_state.model, st.session_state.api_key,
                       max_tokens=500)
        return ans if ans else None

    if local_hits:
        return "（基于本地上传文件内容检索，未接入大模型）\n" + "\n".join(
            f"- {h}" for h in local_hits[:15])
    return None


def do_qa(question):
    # 优先基于本次上传的财务文件作答（满足「上传后基于文件问答」）
    file_ans = _file_qa(question)
    if file_ans is not None:
        add_assistant(file_ans)
        return

    try:
        from utils.vector_store import retrieve, grounded_answer
        from utils.llm_helper import call_llm as rag_call_llm
        rag_index, rag_meta = _load_rag_index()
    except Exception as e:
        add_assistant(f"问答模块不可用：{e}")
        return
    q = question.strip()
    hits = retrieve(rag_index, q, k=5)
    if not hits:
        add_assistant("", payload={"type": "qa",
                     "data": {"empty": True, "text": "", "citations": [],
                              "mock": False, "asof": ""}})
        return
    use_mock = not st.session_state.get("api_key")

    def _rag_llm(prompt, temperature=0):
        return rag_call_llm(
            "你是小微金融政策与银行产品问答助手，仅基于给定资料回答。",
            prompt, st.session_state.model, st.session_state.api_key,
            temperature=temperature, max_tokens=600)

    with st.spinner("检索中…" if use_mock else "检索与生成中…"):
        ans = grounded_answer(None if use_mock else _rag_llm, q, hits,
                              use_mock=use_mock)
    add_assistant("", payload={"type": "qa",
                 "data": {"empty": False, "text": ans.text,
                          "citations": ans.citations, "mock": use_mock,
                          "asof": rag_meta.get("asof", "")}})


def _enter_manual():
    if st.session_state.stage in ("need_confirm", "diagnosed") and st.session_state.metrics:
        add_assistant("当前已有数据。如需重新手动录入，请先点「重新录入」。")
        return
    _seed_manual_template()
    st.session_state.stage = "need_confirm"
    add_user("我想手动录入财务指标")
    add_assistant("已为你生成指标模板，请在下方填写（单位：万元 / 倍），确认后开始诊断。")


def _dispatch(cmd):
    if cmd == "manual":
        _enter_manual()
    elif cmd == "diagnose":
        do_diagnose()
    elif cmd == "products":
        if not st.session_state.diagnosis_result:
            add_assistant("请先完成诊断，我再为你匹配信贷产品。")
        else:
            do_products()
    elif cmd == "report":
        if not st.session_state.diagnosis_result:
            add_assistant("请先完成诊断，我再为你生成 PDF 报告。")
        else:
            do_report()
    elif cmd == "reset":
        _reset_data()
        add_assistant("已清空数据，可重新上传文件或手动录入。")


def _classify(text):
    t = text.strip()
    if any(k in t for k in ("诊断", "分析", "评分", "健康", "开始")):
        return "diagnose"
    if any(k in t for k in ("产品", "匹配", "信贷", "推荐", "贷款")):
        return "products"
    if any(k in t for k in ("报告", "pdf", "PDF", "下载报告")):
        return "report"
    return "qa"


# ======================== live 面板：指标确认 ========================
def _render_confirm_panel():
    m = st.session_state.metrics
    if not m:
        return
    src = "智能提取" if st.session_state.ai_extracted else "自动提取 / 手动录入"
    st.markdown(f"**确认指标与补充信息**　{badge(src, 'blue')}", unsafe_allow_html=True)

    with st.form("diagnosis_form"):
        rows = []
        for k in m:
            if k.startswith("__"):
                continue
            f = m[k]
            if isinstance(f, dict):
                rows.append({"指标": k, "数值": str(f.get("value", "")),
                             "单位": f.get("unit", ""), "来源页": f.get("page", "")})
            else:
                rows.append({"指标": k, "数值": str(f), "单位": "", "来源页": ""})
        edited = st.data_editor(
            pd.DataFrame(rows),
            key=f"metrics_editor_{st.session_state.metrics_rev}",
            hide_index=True, use_container_width=True,
            column_config={
                "指标": st.column_config.TextColumn(disabled=True),
                "单位": st.column_config.TextColumn(disabled=True, width="small"),
                "来源页": st.column_config.TextColumn(disabled=True, width="small"),
            },
        )

        st.markdown("**经营与信用信息**")
        b1, b2 = st.columns(2)
        with b1:
            operating_years = st.number_input("企业经营年限", min_value=0.0, step=0.5, value=3.0)
            industry = st.text_input("所属行业", value="制造业")
        with b2:
            avg_interest_rate = st.number_input("现有融资平均年利率（%）",
                                                min_value=0.0, step=0.1, value=5.0)
            customer_concentration = st.selectbox(
                "客户集中度", ["低（前五大客户占比<30%）", "中（30%~60%）", "高（>60%）"])

        st.markdown("**应收账款账龄结构（%）**")
        a1, a2, a3 = st.columns(3)
        with a1:
            ar_less_3m = st.number_input("3 个月内", 0, 100, 60)
        with a2:
            ar_3_12m = st.number_input("3~12 个月", 0, 100, 30)
        with a3:
            ar_over_12m = st.number_input("超过 12 个月", 0, 100, 10)

        with st.expander("高级选项（信用评级、增长率、行业周期信号、企业名称）"):
            g1, g2, g3 = st.columns(3)
            with g1:
                tax_credit_rating = st.selectbox("纳税信用评级",
                                                 ["未评级", "A", "B", "M", "C", "D"])
                revenue_growth_rate = st.number_input("营收增长率（%）",
                                                      -100.0, 500.0, 10.0, 1.0)
            with g2:
                controller_credit = st.selectbox("实控人征信状态",
                                                 ["良好", "一般", "有逾期记录"])
                profit_growth_rate = st.number_input("净利润增长率（%）",
                                                     -100.0, 500.0, 8.0, 1.0)
            with g3:
                court_execution = st.selectbox("法院执行/诉讼记录", ["无", "有"])
                financing_institution_count = st.number_input("融资机构数量", 0, 10, 1, 1)
            g4, g5 = st.columns(2)
            with g4:
                can_collateral = st.selectbox(
                    "是否可提供抵押/担保", ["可提供", "暂无法提供"])
            with g5:
                industry_cycle = st.slider(
                    "行业周期信号（外部输入）", -1.0, 1.0, 0.0, 0.1)
            enterprise_name = st.text_input(
                "企业名称（用于跨期对比，选填）",
                value=st.session_state.get("enterprise_name", "默认企业"))

        submitted = st.form_submit_button("开始金融健康诊断", type="primary",
                                          use_container_width=True)

    if submitted:
        _run_diagnosis(edited, {
            "operating_years": operating_years,
            "industry": industry,
            "avg_interest_rate": avg_interest_rate,
            "customer_concentration": customer_concentration,
            "ar_less_3m": ar_less_3m,
            "ar_3_12m": ar_3_12m,
            "ar_over_12m": ar_over_12m,
            "tax_credit_rating": tax_credit_rating,
            "controller_credit": controller_credit,
            "court_execution": court_execution,
            "financing_institution_count": financing_institution_count,
            "revenue_growth_rate": revenue_growth_rate,
            "profit_growth_rate": profit_growth_rate,
            "can_collateral": can_collateral,
            "industry_cycle": industry_cycle,
        }, enterprise_name)
        _after_diagnose()
        st.rerun()


# ======================== 快捷操作 ========================
def _quick_actions():
    stage = st.session_state.stage
    if stage == "init":
        c1, c2 = st.columns(2)
        with c1:
            if st.button("手动录入指标", use_container_width=True, key="qa_manual"):
                st.session_state._pending = "manual"
        with c2:
            if st.button("重新诊断", use_container_width=True, key="qa_rediag"):
                st.session_state._pending = "diagnose"
    elif stage == "need_confirm":
        st.caption("在上方表单确认指标后，点击「开始金融健康诊断」")
    elif stage == "diagnosed":
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("查看产品匹配", use_container_width=True, key="qa_prod"):
                st.session_state._pending = "products"
        with c2:
            if st.button("生成 PDF 报告", use_container_width=True, key="qa_rep"):
                st.session_state._pending = "report"
        with c3:
            if st.button("重新录入", use_container_width=True, key="qa_reset"):
                st.session_state._pending = "reset"


# ======================== 主区渲染 ========================
st.markdown(
    '<div class="fd-topbar"><div class="dot">融</div>'
    '<div><div class="name">融资诊断助手</div>'
    '<div class="sub">上传财务资料，获取健康诊断、产品匹配与诊断报告</div></div></div>',
    unsafe_allow_html=True)
st.markdown('<div class="fd-divider"></div>', unsafe_allow_html=True)

# ----- 会话管理：从历史对话继续 / 新建对话 -----
convs = PS.list_conversations()
_conv_options = [("＋ 新建对话", None)] + [
    (f"{c['title']} · {c['updated'][:10]}", c["id"]) for c in convs]
_conv_labels = [o[0] for o in _conv_options]
_conv_map = {o[0]: o[1] for o in _conv_options}
_col_sel, _col_new = st.columns([4, 1])
with _col_sel:
    sel = st.selectbox("对话历史", _conv_labels, index=0, key="conv_selector",
                       label_visibility="collapsed")
with _col_new:
    if st.button("新建对话", use_container_width=True, key="btn_new_conv"):
        _new_conversation()
        st.rerun()

if st.session_state.get("_conv_sel_last") != sel:
    st.session_state._conv_sel_last = sel
    cid = _conv_map.get(sel)
    if cid is None:
        _new_conversation()
    else:
        _load_conversation_state(cid)
        st.rerun()

# 冷启动：尚无会话则建一个空白会话
if not st.session_state.get("conv_id"):
    _new_conversation()

# 首次进入的欢迎语（messages 为空时仅加一次）
if not st.session_state.messages:
    add_assistant("你好，我是融资诊断助手。")
    add_assistant(
        "上传一份财务资料（PDF / Excel / CSV），或点击下方「手动录入指标」开始。"
        "我会完成健康诊断、产品匹配与诊断报告。")

# 处理快捷操作回调
if st.session_state.get("_pending"):
    cmd = st.session_state._pending
    st.session_state._pending = None
    _dispatch(cmd)

# 渲染对话历史
for _m in st.session_state.messages:
    _render_message(_m)

# 当前步 live 面板
if st.session_state.stage == "need_confirm":
    _render_confirm_panel()
    with st.expander("解析详情 / 智能提取", expanded=False):
        _render_ai_extract()
        _render_verification()

# 快捷操作
_quick_actions()

# 对话输入（含文件上传）
prompt = st.chat_input(
    "描述需求，或上传财务文件（PDF / Excel / CSV）…",
    accept_file=True, file_type=["pdf", "csv", "xlsx", "xls"])
if prompt is not None:
    files = getattr(prompt, "files", None) or []
    text = getattr(prompt, "text", "") or ""
    if files:
        f = files[0]
        _handle_file(f)
        add_user(f"我上传了文件：{f.name}")
        add_assistant("已解析财务资料，请在下方指标表中确认或补充，确认后开始诊断。")
        st.session_state.stage = "need_confirm"
    if text.strip():
        add_user(text)
        cmd = _classify(text)
        if cmd == "qa":
            do_qa(text)
        else:
            _dispatch(cmd)
    st.rerun()

# 落盘当前会话（消息 / 诊断态）
_save_current_conversation()
