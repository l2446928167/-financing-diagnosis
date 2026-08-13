"""
app.py v2.0 — 界面重构版

相对 v1.6 的改动（算法模块 modules/ 与 utils/vector_store.py 均未改动）：
- 信息架构：单页长滚动 → 四标签页（数据录入 / 诊断报告 / 产品匹配与差距 / 政策问答）
- P0 修复：诊断结果、匹配、报告全部从 session_state 渲染，rerun 不再丢失
  （v1.6 全部渲染在 `if st.button()` 块内，任意交互即清空）
- P0 修复：录入控件包进 st.form，一次提交一次 rerun；问答区 st.fragment 独立刷新
- 指标修正墙 → st.data_editor 表格；补充信息分「基础 + 高级选项」
- 新增纯手动录入路径（无文件也可演示）；空状态 hero 引导
- 视觉：config.toml 金融蓝主题 + utils/ui_style.py 卡片/徽章；8 维与 SHAP 升级为配色图表
- 侧边栏精简：不再回显 Key 片段；移除单选项模型下拉
"""
import re
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv, set_key

APP_VERSION = "v2.0 (2026-08-13)"
MODEL_NAME = "DeepSeek-V4-Flash"
PRODUCT_CSV = "knowledge/bank_products/products.csv"
RAG_INDEX = "knowledge/rag_corpus/bm25_index.json"

load_dotenv()

st.set_page_config(page_title="小微企业融资诊断", page_icon="🏦", layout="wide")

from utils.ui_style import inject_css, badge, score_level, dim_level, hero, empty_state  # noqa: E402

inject_css()

# ======================== session_state 初始化 ========================
for _k, _v in {
    "metrics": {}, "metrics_rev": 0, "raw_text": "", "df": None,
    "diagnosis_result": None, "full_metrics": None, "llm_text": None,
    "ai_extracted": False, "match_cache": None, "pdf_bytes": None,
    "ai_rec": None, "qa_result": None, "_ai_changes": [], "_file_sig": None,
}.items():
    st.session_state.setdefault(_k, _v)

# ======================== 侧边栏 ========================
with st.sidebar:
    st.header("配置")
    api_key = st.text_input("DeepSeek API Key", type="password",
                            value=os.environ.get("DEEPSEEK_API_KEY", ""))
    st.session_state.api_key = api_key
    st.session_state.model = MODEL_NAME

    if api_key:
        st.caption("✅ 已配置 API Key，AI 功能可用")
    else:
        st.caption("⚠️ 未配置 API Key：AI 提取/生成不可用，规则引擎与检索问答仍可用")

    s1, s2 = st.columns(2)
    with s1:
        if st.button("保存到本地", use_container_width=True):
            if api_key:
                set_key(os.path.join(os.path.dirname(__file__), ".env"),
                        "DEEPSEEK_API_KEY", api_key)
                st.toast("API Key 已保存，下次启动自动加载")
            else:
                st.toast("请先输入 API Key")
    with s2:
        if st.button("测试连接", use_container_width=True):
            from utils.llm_helper import test_api_connection
            with st.spinner("测试连接中…"):
                ok, msg = test_api_connection(api_key)
            (st.success if ok else st.error)(msg)

    st.caption(f"模型：{MODEL_NAME} ｜ 版本：{APP_VERSION}")
    with st.expander("使用说明"):
        st.markdown(
            "1. 「数据录入」上传文件或手动录入，确认指标后开始诊断\n"
            "2. 「诊断报告」查看评分、双轨对照与 SHAP 归因，下载 PDF\n"
            "3. 「产品匹配与差距」查看信贷产品匹配与行动方案\n"
            "4. 「政策问答」可溯源查询政策与产品"
        )

# ======================== 辅助函数 ========================


def _to_num(v):
    """把指标字符串安全转成 float：去千分位逗号与非法后缀，失败为 0。"""
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
    - 风险2
    ...
    【改善建议】
    - 建议1
    - 建议2
    ...
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
    """解析 LLM 输出为 (总体评价, 风险点列表, 建议列表)；无文本时全空。"""
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
    """产品匹配结果缓存：诊断更新后自动失效重算。"""
    if st.session_state.get("match_cache") is not None:
        return st.session_state.match_cache
    from modules.product_matching import match_products
    matches = match_products(st.session_state.full_metrics, _load_products())
    st.session_state.match_cache = matches
    return matches


def _report_rag_citations():
    """报告附录用政策依据（条款摘编 + 检索截至日期）。"""
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
    """AI 产品推荐 / 无匹配时的改善建议（按钮触发，避免每次 rerun 调 LLM）。"""
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
    """8 维评分：横向条形 + 红黄绿色带（阈值与规则卡一致 7/4）。"""
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
                                    range=["#16A34A", "#D97706", "#DC2626"]),
                    legend=None),
                tooltip=["维度", "评分"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(pd.DataFrame({"评分": dims}))


def _shap_chart(contribs):
    """SHAP 归因：正值红（推高违约）、负值绿（拉低违约）。"""
    rows = sorted(
        ((k, v) for k, v in contribs.items() if k not in ("bias", "违约概率")),
        key=lambda kv: -abs(kv[1]))
    df = pd.DataFrame({"因子": [k for k, _ in rows],
                       "贡献": [round(v, 3) for _, v in rows]})
    try:
        import altair as alt
        df["方向"] = df["贡献"].apply(lambda v: "推高违约" if v >= 0 else "拉低违约")
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("贡献:Q", title=None),
                y=alt.Y("因子:N", sort="-x", title=None),
                color=alt.Color(
                    "方向:N",
                    scale=alt.Scale(domain=["推高违约", "拉低违约"],
                                    range=["#DC2626", "#16A34A"]),
                    legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["因子", "贡献"],
            )
            .properties(height=max(160, 26 * len(df)))
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(df.set_index("因子"))


# ======================== Tab1：数据录入 ========================

# diagnose() 消费的原始财务键（与 modules/diagnosis.py 一致），手动录入模板用
_MANUAL_KEYS = ["总资产", "总负债", "营业收入", "营业成本", "净利润",
                "流动负债", "经营活动现金流净额", "利息费用",
                "应收账款", "存货", "流动比率"]


def _handle_file(uploaded_file):
    """解析文件（仅在新文件时重解析），并展示文件信息。"""
    from modules.data_input import parse_financial_data, auto_extract_metrics
    sig = (uploaded_file.name, uploaded_file.size)
    if st.session_state._file_sig != sig:
        st.session_state._file_sig = sig
        st.session_state.ai_extracted = False
        st.session_state.diagnosis_result = None
        st.session_state.full_metrics = None
        st.session_state.llm_text = None
        st.session_state.match_cache = None
        st.session_state.pdf_bytes = None
        st.session_state.ai_rec = None
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

    raw_text, df = st.session_state.raw_text, st.session_state.df
    type_info = {"pdf": "PDF", "xlsx": "Excel", "xls": "Excel", "csv": "CSV"}
    parts = [type_info.get(uploaded_file.name.split(".")[-1].lower(), "文件")]
    if raw_text and raw_text.strip():
        parts.append(f"文本 {len(raw_text)} 字符")
    if df is not None:
        parts.append(f"表格 {df.shape[0]} 行 × {df.shape[1]} 列")
    st.caption("已解析：" + " ｜ ".join(parts))


def _render_ai_extract():
    """AI 智能提取（表单外，需要即时触发），差异清单持久展示。"""
    if not st.session_state.get("api_key"):
        st.caption("在侧边栏配置 API Key 后，可用 AI 智能提取财务指标（比正则更准）。")
        return
    if st.button("AI 智能提取财务指标", key="ai_extract"):
        with st.spinner("AI 正在分析文件内容…"):
            try:
                from modules.data_input import llm_extract_metrics
                from utils.llm_helper import call_llm
                ai_metrics = llm_extract_metrics(
                    st.session_state.raw_text, st.session_state.df,
                    st.session_state.model, st.session_state.api_key, call_llm)
            except Exception as e:
                st.error(f"AI 提取异常：{type(e).__name__} – {e}")
                return
        if not ai_metrics:
            st.error("AI 提取未能获得有效结果。")
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
        st.session_state.metrics_rev += 1   # 触发 data_editor 重建，展示新值
        st.session_state._ai_changes = changes
        st.toast(f"AI 提取完成，更新 {len(changes)} 个指标")
        st.rerun()
    if st.session_state.get("_ai_changes"):
        with st.expander(f"AI 提取变更（{len(st.session_state._ai_changes)} 项）"):
            for c in st.session_state._ai_changes:
                st.markdown(f"- `{c}`")


def _render_verification():
    verification = st.session_state.metrics.get("__verification__")
    if not verification:
        return
    with st.expander("会计校验结果", expanded=True):
        for check in verification:
            status = check.get("是否通过")
            name = check.get("检查项", "")
            actual = check.get("实际值", "")
            expected = check.get("预期值", "")
            if status is True:
                st.markdown(f"✅ **{name}**：通过（{actual}）")
            elif status is False:
                st.markdown(f"⚠️ **{name}**：偏差 — {actual}，预期：{expected}")
            else:
                st.markdown(f"ℹ️ **{name}**：无法校验（{actual}）")


def _render_raw_data():
    with st.expander("查看原始解析数据"):
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df, use_container_width=True)
        if st.session_state.raw_text:
            st.text_area("提取的文本（前 10000 字符）",
                         st.session_state.raw_text[:10000], height=260)
            st.download_button("下载完整原始文本", data=st.session_state.raw_text,
                               file_name="raw_text.txt", mime="text/plain")


def _seed_manual_template():
    st.session_state.metrics = {
        k: {"value": "", "unit": ("倍" if k == "流动比率" else "万元"), "page": ""}
        for k in _MANUAL_KEYS
    }
    st.session_state.metrics_rev += 1


def _render_diagnosis_form():
    """指标确认（data_editor）+ 补充信息 + 提交诊断。st.form 内一次提交一次 rerun。"""
    m = st.session_state.metrics
    src = "AI 提取" if st.session_state.ai_extracted else "自动提取 / 手动录入"
    st.subheader("确认指标与补充信息")
    st.caption(f"指标来源：{src}；「数值」列可直接编辑。")

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

        with st.expander("高级选项（信用评级、增长率、行业周期信号）"):
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
                    "是否可提供抵押/担保", ["可提供", "暂无法提供"],
                    help="影响需要抵押物的产品匹配与差距分析")
            with g5:
                industry_cycle = st.slider(
                    "行业周期信号（ML 外部输入）", -1.0, 1.0, 0.0, 0.1,
                    help="-1=行业深度下行，0=平稳，1=高度景气；正式版接行业景气度数据源")

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
        })


def _run_diagnosis(edited, s):
    """计算并写入 session_state；渲染全部在按钮块外（修 v1.6 结果消失 bug）。"""
    from modules.diagnosis import diagnose

    metrics = {}
    for _, row in edited.iterrows():
        metrics[row["指标"]] = {"value": str(row["数值"]),
                                "unit": str(row["单位"]), "page": str(row["来源页"])}
    for k in st.session_state.metrics:          # 保留 __verification__ 等特殊键
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
    with st.spinner("规则引擎诊断中…"):
        result = diagnose(full_metrics)

    st.session_state.full_metrics = full_metrics
    st.session_state.diagnosis_result = result
    st.session_state.match_cache = None
    st.session_state.pdf_bytes = None
    st.session_state.ai_rec = None

    if st.session_state.get("api_key"):
        with st.spinner("AI 正在生成诊断总结…"):
            st.session_state.llm_text = generate_diagnosis_text(
                full_metrics, result["dimension_scores"], result["overall_score"],
                st.session_state.model, st.session_state.api_key)
    else:
        st.session_state.llm_text = None
    st.toast("诊断完成，切换到「诊断报告」标签页查看")


# ======================== 页面结构 ========================

st.title("AI + 小微企业融资诊断")
st.caption("规则引擎 × 违约 ML 双轨对照 · 政策/产品可溯源问答 · 一键生成诊断报告")

tab_input, tab_diag, tab_match, tab_qa = st.tabs(
    ["数据录入", "诊断报告", "产品匹配与差距", "政策问答"])

# ---------------- Tab 1 ----------------
with tab_input:
    uploaded_file = st.file_uploader(
        "上传财务报表 / 银行流水 / 应收账款明细（PDF / Excel / CSV）",
        type=["pdf", "xlsx", "xls", "csv"])

    if uploaded_file is not None:
        _handle_file(uploaded_file)
        if st.session_state.raw_text or st.session_state.df is not None:
            _render_ai_extract()
            _render_verification()
            _render_raw_data()
    elif st.session_state._file_sig is not None:
        # 文件被移除 → 清空派生状态
        st.session_state._file_sig = None
        st.session_state.metrics = {}
        st.session_state.raw_text = ""
        st.session_state.df = None
        st.session_state.diagnosis_result = None
        st.session_state.full_metrics = None
        st.session_state.llm_text = None
        st.session_state.ai_extracted = False
        st.session_state.match_cache = None
        st.session_state.pdf_bytes = None
        st.session_state.ai_rec = None

    if st.session_state.metrics:
        _render_diagnosis_form()
    else:
        if uploaded_file is not None:
            st.warning("未能从该文件提取到可读内容，可更换文件，或手动录入指标。")
        else:
            hero()
        if st.checkbox("手动录入指标", key="manual_mode"):
            _seed_manual_template()
            st.rerun()

# ---------------- Tab 2：诊断报告 ----------------
with tab_diag:
    res = st.session_state.diagnosis_result
    if not res:
        empty_state("尚无诊断结果",
                    "请先在「数据录入」页确认指标，点击「开始金融健康诊断」。")
    else:
        full_metrics = st.session_state.full_metrics
        dims = res["dimension_scores"]
        score = res["overall_score"]
        level, label = score_level(score)

        # 双轨（ML 概率 + 四态结论），模型缺失自动降级
        statement, proba, conclusion = None, None, None
        try:
            from modules.ml_model import (predict_default_proba,
                                          dual_track_conclusion, explain_statement)
            _CAT = {"纳税信用评级", "实控人征信状态", "法院执行记录",
                    "客户集中度", "行业", "可提供抵押"}
            statement = {k: (v if k in _CAT else _to_num(v))
                         for k, v in full_metrics.items()}
            proba = predict_default_proba(statement)
            conclusion, _tag = dual_track_conclusion(score, proba)
        except Exception:
            statement = None

        o1, o2, o3 = st.columns([1, 1, 2])
        with o1:
            st.metric("总体健康评分", f"{score} / 10")
            st.markdown(badge(label, level), unsafe_allow_html=True)
        with o2:
            st.metric("ML 违约概率", "不可用" if proba is None else f"{proba * 100:.1f}%")
        with o3:
            st.markdown("**双轨结论（规则卡 × ML）**")
            st.info(conclusion if conclusion else "ML 模型不可用，当前为单轨规则卡结论。")

        st.subheader("8 维健康评分")
        _dims_chart(dims)
        badge_cols = st.columns(4)
        for i, (dim, sc) in enumerate(dims.items()):
            with badge_cols[i % 4]:
                st.markdown(f"{dim}　{badge(f'{sc}/10', dim_level(sc))}",
                            unsafe_allow_html=True)

        if statement is not None:
            try:
                contribs = explain_statement(statement)
            except Exception:
                contribs = None
            if contribs:
                with st.expander("SHAP 归因（各因子对违约概率的贡献方向与强度）"):
                    _shap_chart(contribs)
                    st.caption("正值推高违约概率、负值拉低。模型为合成数据方法论演示，仅供对照参考。")

        summary, risks, suggestions = _parse_llm_sections(st.session_state.get("llm_text"))
        if summary:
            st.subheader("AI 诊断总结")
            st.markdown(summary)

        r1, r2 = st.columns(2)
        with r1:
            st.subheader("核心风险点")
            shown = risks or res["risks"]
            if shown:
                for r in shown:
                    st.markdown(f"- {r}")
            else:
                st.success("未发现明显风险点。")
        with r2:
            st.subheader("改善行动建议")
            shown = suggestions or res["suggestions"]
            for s_ in shown:
                st.markdown(f"- {s_}")
        st.caption("以上诊断基于规则引擎与 AI 生成内容，仅供参考，不构成金融建议。")

        st.subheader("下载诊断报告")
        with st.container(border=True):
            st.caption("报告包含：指标与评分、双轨结论、风险与建议、产品匹配，"
                       "以及政策与产品依据附录（条款摘编）。")
            if st.button("生成 PDF 报告", key="gen_pdf", type="primary"):
                with st.spinner("正在生成 PDF…"):
                    try:
                        from modules.report_generator import generate_pdf
                        matches = _ensure_matches()
                        rag_cites, rag_asof = _report_rag_citations()
                        st.session_state.pdf_bytes = generate_pdf(
                            full_metrics, res, matches, _load_products(),
                            ai_summary=summary,
                            ai_risks="\n".join(f"- {r}" for r in risks),
                            ai_suggestions="\n".join(f"- {x}" for x in suggestions),
                            ai_recommendation=st.session_state.get("ai_rec") or "",
                            rag_citations=rag_cites,
                            rag_asof=rag_asof)
                        st.toast("PDF 报告已生成")
                    except Exception as e:
                        st.error(f"报告生成失败：{e}")
            if st.session_state.get("pdf_bytes"):
                st.download_button("下载 PDF 报告", data=st.session_state.pdf_bytes,
                                   file_name="融资诊断报告.pdf", mime="application/pdf",
                                   key="dl_pdf")

# ---------------- Tab 3：产品匹配与差距 ----------------
with tab_match:
    if not st.session_state.diagnosis_result:
        empty_state("尚无诊断结果", "完成诊断后，这里会给出信贷产品匹配与差距分析。")
    else:
        full_metrics = st.session_state.full_metrics
        res = st.session_state.diagnosis_result

        st.subheader("匹配银行信贷产品")
        try:
            matches = _ensure_matches()
        except Exception as e:
            st.error(f"产品匹配出错：{e}")
            matches = []

        if matches:
            display_cols = ["匹配度", "产品名", "银行", "产品类型",
                            "额度", "利率", "准入条件", "差距说明"]
            st.dataframe(pd.DataFrame(matches)[display_cols],
                         use_container_width=True, hide_index=True)
            st.caption("数据来源及采集日期见产品库明细。")
        else:
            st.warning("未找到匹配的信贷产品，建议改善财务状况后再查询。")

        if st.session_state.get("api_key"):
            if st.button("生成 AI 产品推荐说明", key="ai_rec_btn"):
                with st.spinner("AI 正在分析最佳产品方案…"):
                    st.session_state.ai_rec = _gen_ai_recommendation(full_metrics, matches)
            if st.session_state.get("ai_rec"):
                with st.container(border=True):
                    st.markdown("**AI 产品推荐说明**")
                    st.markdown(st.session_state.ai_rec)
        else:
            st.caption("配置 API Key 后可生成 AI 产品推荐说明。")

        st.subheader("差距分析与行动方案")
        try:
            from modules.gap_analysis import analyze_gaps
            gap_result = analyze_gaps(full_metrics, res["dimension_scores"], _load_products())

            action_plan = gap_result.get("action_plan", [])
            if action_plan:
                st.markdown("性价比 = 解锁产品数 ÷ 提升难度分，数值越高越应优先。")
                action_rows = []
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
                    action_rows.append({
                        "优先级": act["priority"], "行动": act["action"],
                        "当前值": act["current"], "目标值": act["target"],
                        "难度": act["difficulty"], "影响产品数": impact_label,
                        "相关产品": products_label, "预计时间": act["estimated_time"],
                        "性价比": act["cost_efficiency"],
                    })
                st.dataframe(pd.DataFrame(action_rows),
                             use_container_width=True, hide_index=True)
            else:
                st.success("所有匹配产品均无差距，无需额外行动。")

            gap_products = [p for p in gap_result.get("product_gap_details", [])
                            if p["match_status"] == "差距匹配"]
            if gap_products:
                st.markdown("**各产品差距明细**")
                for gp in gap_products:
                    with st.expander(f"{gp['product']}（{gp['bank']}）— {gp['product_type']}"):
                        if gp["gaps"]:
                            gap_table = [{
                                "差距项": g["item"], "当前值": g["current"],
                                "准入要求": g["required"],
                                "差距量": (f"{g['gap_size']:.1f}"
                                          if isinstance(g["gap_size"], (int, float))
                                          else g["gap_size"]),
                                "提升难度": g["difficulty"],
                            } for g in gp["gaps"]]
                            st.dataframe(pd.DataFrame(gap_table), hide_index=True,
                                         use_container_width=True)
                            st.caption(f"最容易补齐：{gp['closest_to_qualify']}")
                        else:
                            st.info("该产品已达标")

            if gap_result.get("summary"):
                st.markdown("**总结**")
                st.markdown(gap_result["summary"])
        except Exception as e:
            st.error(f"差距分析出错：{e}")

        _products = _load_products()
        with st.expander(f"查看完整产品库（{len(_products)} 条）"):
            st.dataframe(_products, use_container_width=True, hide_index=True)
            st.caption("数据来源：各银行官网，采集日期见表中字段。")

# ---------------- Tab 4：政策问答 ----------------


def _qa_body():
    """RAG 问答：BM25 检索 + 溯源生成；无 Key 降级检索片段；OOV 诚实空。"""
    try:
        from utils.vector_store import retrieve, grounded_answer
        from utils.llm_helper import call_llm as rag_call_llm
        rag_index, rag_meta = _load_rag_index()
    except Exception as e:
        st.warning(f"RAG 模块不可用：{e}")
        return

    q = st.text_input("问题", key="qa_q",
                      placeholder="例：科技型小微企业无抵押能贷多少？")
    cat = st.radio("语料范围", ["全部", "政策", "产品", "研报"],
                   horizontal=True, key="qa_cat")
    cat_map = {"全部": None, "政策": "policy", "产品": "product", "研报": "research"}

    if st.button("检索回答", key="qa_go", type="primary") and q.strip():
        hits = retrieve(rag_index, q.strip(), k=5, category=cat_map[cat])
        if not hits:
            st.session_state.qa_result = {"q": q.strip(), "empty": True}
        else:
            use_mock = not st.session_state.get("api_key")

            def _rag_llm(prompt, temperature=0):
                return rag_call_llm(
                    "你是小微金融政策与银行产品问答助手，仅基于给定资料回答。",
                    prompt, st.session_state.model, st.session_state.api_key,
                    temperature=temperature, max_tokens=600)

            with st.spinner("检索中…" if use_mock else "检索与生成中…"):
                ans = grounded_answer(None if use_mock else _rag_llm,
                                      q.strip(), hits, use_mock=use_mock)
            st.session_state.qa_result = {
                "q": q.strip(), "empty": False, "mock": use_mock,
                "text": ans.text, "citations": ans.citations,
            }

    result = st.session_state.get("qa_result")
    if not result:
        return
    if result["empty"]:
        st.info("该问题未被本模块语料收录（诚实返回空，不编造）。"
                "可改问信贷产品或普惠政策。")
        return
    if result["mock"]:
        st.caption("未配置 API Key：当前展示检索片段（配置后由 DeepSeek 做溯源生成）")
    with st.container(border=True):
        st.markdown(result["text"])
    with st.expander(f"参考清单（{len(result['citations'])} 条；政策条目为条款摘编）"):
        for c in result["citations"]:
            st.markdown(f"- {c}")
    if rag_meta.get("asof"):
        st.caption(f"检索截至 {rag_meta['asof']} ｜ 产品库指纹 "
                   f"{str(rag_meta.get('products_csv_sha256', ''))[:12]}…")


with tab_qa:
    st.subheader("政策 / 产品智能问答（可溯源）")
    st.caption("基于 BM25 检索 + 溯源生成：只用语料回答，未知即如实返回空。")
    _fragment = getattr(st, "fragment", None)
    try:
        (_fragment(_qa_body) if _fragment else _qa_body)()
    except Exception as e:
        st.warning(f"问答区渲染异常：{e}")
