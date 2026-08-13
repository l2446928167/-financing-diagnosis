import streamlit as st
import pandas as pd
import re
import os
from dotenv import load_dotenv, set_key

APP_VERSION = "v1.6 (2026-08-13)"

# 加载 .env（用于本地保存 API Key）
load_dotenv()
saved_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

st.set_page_config(page_title="小微企业融资诊断工具", layout="wide")
st.title("🏦 AI + 小微企业融资诊断系统")
st.markdown("上传企业财务数据，获取健康诊断与信贷产品匹配建议。")

# ======================== 侧边栏 ========================
with st.sidebar:
    st.header("⚙️ 配置区")
    st.caption(f"版本：{APP_VERSION}")
    api_key = st.text_input("请输入 API Key", type="password", value=saved_api_key)
    model = st.selectbox("选择模型", ["DeepSeek-V4-Flash"])

    if st.button("💾 保存 API Key 到本地"):
        if api_key:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            set_key(env_path, "DEEPSEEK_API_KEY", api_key)
            st.success("API Key 已保存，下次启动自动加载。")
        else:
            st.warning("请先输入 API Key")

    st.session_state.api_key = api_key
    st.session_state.model = model

    # 测试 API 连接按钮
    if st.button("🔌 测试 API 连接"):
        from utils.llm_helper import test_api_connection
        with st.spinner("正在测试连接..."):
            ok, msg = test_api_connection(api_key)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    # 显示 API Key 状态
    if api_key:
        st.markdown(f"🔑 API Key 已填写（{api_key[:6]}...{api_key[-4:]}）")
    else:
        st.warning("⚠️ 未填写 API Key，AI 功能不可用")

# ======================== 初始化 session_state ========================
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "df" not in st.session_state:
    st.session_state.df = None
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "ai_extracted" not in st.session_state:
    st.session_state.ai_extracted = False

# ======================== 辅助函数 ========================
def generate_diagnosis_text(full_metrics, dims, overall, model_choice, api_key):
    """用 LLM 生成诊断总结、风险点、改善建议，覆盖8个维度，失败返回 None"""
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

def _to_num(v):
    """把指标输入框的字符串安全转成 float：去千分位逗号与非法后缀，失败为 0"""
    import re as _re
    try:
        cleaned = _re.sub(r'[^\d.\-]', '', str(v).replace(",", "").replace("，", ""))
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


# ======================== 模块1：数据录入 ========================
st.header("📤 第一步：上传企业财务数据")

uploaded_file = st.file_uploader("支持 PDF / Excel / CSV 格式", type=["pdf", "xlsx", "xls", "csv"])

if uploaded_file is not None:
    from modules.data_input import parse_financial_data, auto_extract_metrics

    # v1.5.1 修复：更换上传文件时重置 AI 提取标记，避免沿用上一个文件的指标
    file_sig = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("_file_sig") != file_sig:
        st.session_state._file_sig = file_sig
        st.session_state.ai_extracted = False

    with st.spinner("正在解析文件..."):
        try:
            raw_text, df = parse_financial_data(uploaded_file)
            st.session_state.raw_text = raw_text
            st.session_state.df = df
            # 如果 AI 已提取过指标，不覆盖（防止 rerun 时被正则结果冲掉）
            if not st.session_state.ai_extracted:
                extracted = auto_extract_metrics(raw_text, df)
                st.session_state.metrics = extracted
                st.success("文件解析完成！")
            else:
                st.success("文件解析完成！（保留 AI 提取结果）")
        except Exception as e:
            st.error(f"解析失败：{e}")
            st.stop()

    # 显示文件类型信息
    file_type = uploaded_file.name.split(".")[-1].lower()
    type_info = {"pdf": "📄 PDF文件", "xlsx": "📊 Excel文件", "xls": "📊 Excel文件", "csv": "📊 CSV文件"}
    has_text = bool(raw_text and raw_text.strip())
    has_df = df is not None
    status_parts = [type_info.get(file_type, "📁 文件")]
    if has_text:
        status_parts.append(f"文本 {len(raw_text)} 字符")
    if has_df:
        status_parts.append(f"表格 {df.shape[0]}行×{df.shape[1]}列")
    if not has_text and not has_df:
        status_parts.append("⚠️ 未提取到可读内容")
    st.info(" ｜ ".join(status_parts))

    # 显示并允许手动修正指标
    if st.session_state.ai_extracted:
        st.subheader("🤖 AI 提取的关键财务指标（可手动修正）")
        st.caption("以下指标由 AI 智能提取，比正则匹配更精准。")
    else:
        st.subheader("📊 自动提取的关键财务指标（可手动修正）")

    # v1.5：适配新的含unit/page的指标格式
    metrics_dict = st.session_state.metrics
    col1, col2, col3 = st.columns(3)
    updated_metrics = {}
    keys = [k for k in metrics_dict if not k.startswith("__")]
    for i, key in enumerate(keys):
        with [col1, col2, col3][i % 3]:
            field = metrics_dict[key]
            # 兼容新旧格式
            if isinstance(field, dict):
                val = field.get("value", "")
                unit = field.get("unit", "")
                page = field.get("page", "")
                label = f"{key}"
                caption_parts = []
                if unit:
                    caption_parts.append(f"单位: {unit}")
                if page:
                    caption_parts.append(f"第{page}页")
                new_val = st.text_input(
                    label=label,
                    value=val,
                    key=f"metric_{key}"
                )
                if caption_parts:
                    st.caption(" | ".join(caption_parts))
                # 更新时保留unit和page
                updated_metrics[key] = {"value": new_val, "unit": unit, "page": page}
            else:
                # 旧格式（纯字符串）
                new_val = st.text_input(
                    label=key,
                    value=str(field),
                    key=f"metric_{key}"
                )
                updated_metrics[key] = {"value": new_val, "unit": "", "page": ""}
    # 保留特殊键（如 __verification__）
    for k in metrics_dict:
        if k.startswith("__"):
            updated_metrics[k] = metrics_dict[k]
    st.session_state.metrics = updated_metrics

    # v1.5：显示会计校验结果
    verification = st.session_state.metrics.get("__verification__")
    if verification:
        st.subheader("🔍 会计校验结果")
        for check in verification:
            status = check.get("是否通过")
            check_name = check.get("检查项", "")
            actual = check.get("实际值", "")
            expected = check.get("预期值", "")
            if status is True:
                st.markdown(f"✅ **{check_name}**：通过（{actual}）")
            elif status is False:
                st.markdown(f"⚠️ **{check_name}**：偏差 — {actual}，预期：{expected}")
            else:
                st.markdown(f"ℹ️ **{check_name}**：无法校验（{actual}）")

    # AI 智能提取按钮
    if st.session_state.get('api_key'):
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            ai_clicked = st.button("🤖 AI 智能提取财务指标", type="primary")
        with col_btn2:
            if st.session_state.ai_extracted:
                st.success("✅ 已使用 AI 提取结果（点击按钮可重新提取）")
        if ai_clicked:
            with st.spinner("AI 正在分析文件内容，请稍候..."):
                try:
                    from modules.data_input import llm_extract_metrics
                    from utils.llm_helper import call_llm
                    ai_metrics = llm_extract_metrics(
                        st.session_state.raw_text,
                        st.session_state.df,
                        st.session_state.model,
                        st.session_state.api_key,
                        call_llm
                    )
                    if ai_metrics:
                        # 记录AI提取前的结果，用于对比
                        old_metrics = dict(st.session_state.metrics)
                        # v1.5.1 修复：清除指标输入框的旧缓存，否则 rerun 后输入框
                        # 仍显示 AI 提取前的值，并把旧值写回 metrics 覆盖 AI 结果
                        for k in list(ai_metrics.keys()):
                            st.session_state.pop(f"metric_{k}", None)
                        st.session_state.metrics = ai_metrics
                        st.session_state.ai_extracted = True
                        # v1.5：计算差异，适配含unit/page的指标格式
                        changes = []
                        for k in ai_metrics:
                            if k.startswith("__"):
                                continue  # 跳过特殊键
                            old_field = old_metrics.get(k, "")
                            new_field = ai_metrics.get(k, "")
                            # 兼容新旧格式
                            old_val = old_field.get("value", "") if isinstance(old_field, dict) else str(old_field)
                            new_val = new_field.get("value", "") if isinstance(new_field, dict) else str(new_field)
                            new_unit = new_field.get("unit", "") if isinstance(new_field, dict) else ""
                            new_page = new_field.get("page", "") if isinstance(new_field, dict) else ""
                            if old_val != new_val and new_val:
                                detail = f"{k}: {old_val or '(空)'} → {new_val}"
                                extra = []
                                if new_unit:
                                    extra.append(f"单位:{new_unit}")
                                if new_page:
                                    extra.append(f"第{new_page}页")
                                if extra:
                                    detail += f" ({', '.join(extra)})"
                                changes.append(detail)
                        if changes:
                            st.success(f"✅ AI 提取完成！更新了 {len(changes)} 个指标")
                            with st.expander("📋 查看AI提取的变更"):
                                for c in changes:
                                    st.markdown(f"- `{c}`")
                        else:
                            st.info("AI 提取完成，结果与自动提取一致。")
                        st.rerun()
                    else:
                        st.error("❌ AI 提取未能获得有效结果，请查看上方提示信息。")
                except Exception as e:
                    import traceback
                    st.error(f"AI 提取时发生异常：{type(e).__name__} – {e}")
                    with st.expander("查看详细错误跟踪"):
                        st.code(traceback.format_exc())
    else:
        st.info("💡 在侧边栏配置 API Key 后可使用 AI 智能提取财务指标。")

    # 原始数据预览（前10000字符 + 下载）
    with st.expander("查看原始解析数据"):
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df)
        if st.session_state.raw_text:
            text = st.session_state.raw_text
            st.text_area("提取的文本（前10000字符）", text[:10000], height=300)
            st.download_button(
                label="📥 下载完整原始文本",
                data=text,
                file_name="raw_text.txt",
                mime="text/plain"
            )

    # 补充经营信息
    st.subheader("📝 补充经营信息")
    col4, col5 = st.columns(2)
    with col4:
        operating_years = st.number_input("企业经营年限", min_value=0.0, step=0.5, value=3.0)
        customer_concentration = st.selectbox(
            "客户集中度",
            ["低（前五大客户占比<30%）", "中（30%~60%）", "高（>60%）"]
        )
    with col5:
        avg_interest_rate = st.number_input("现有融资平均年利率（%）", min_value=0.0, step=0.1, value=5.0)
        industry = st.text_input("所属行业（非必填）", value="制造业")

    st.markdown("**应收账款账龄结构**")
    c1, c2, c3 = st.columns(3)
    with c1:
        ar_less_3m = st.number_input("3个月内（%）", 0, 100, 60)
    with c2:
        ar_3_12m = st.number_input("3~12个月（%）", 0, 100, 30)
    with c3:
        ar_over_12m = st.number_input("超过12个月（%）", 0, 100, 10)

    # ===== v1.3 新增：补充信用与经营信息 =====
    st.subheader("📋 补充信用与经营信息")
    col_cr1, col_cr2, col_cr3 = st.columns(3)
    with col_cr1:
        tax_credit_rating = st.selectbox(
            "纳税信用评级",
            ["未评级", "A", "B", "M", "C", "D"]
        )
    with col_cr2:
        controller_credit = st.selectbox(
            "实控人征信状态",
            ["良好", "一般", "有逾期记录"]
        )
    with col_cr3:
        court_execution = st.selectbox(
            "是否有法院执行/诉讼记录",
            ["无", "有"]
        )

    col_cr4, col_cr5, col_cr6 = st.columns(3)
    with col_cr4:
        financing_institution_count = st.number_input(
            "融资机构数量",
            min_value=0, max_value=10, value=1, step=1
        )
    with col_cr5:
        revenue_growth_rate = st.number_input(
            "营收增长率（%）",
            min_value=-100.0, max_value=500.0, value=10.0, step=1.0
        )
    with col_cr6:
        profit_growth_rate = st.number_input(
            "净利润增长率（%）",
            min_value=-100.0, max_value=500.0, value=8.0, step=1.0
        )

    # v1.5.1 新增：抵押能力（影响需抵押物产品的匹配与差距分析）
    col_cr7, col_cr8 = st.columns(2)
    with col_cr7:
        can_collateral = st.selectbox(
            "是否可提供抵押/担保",
            ["可提供", "暂无法提供"],
            help="影响需要抵押物的产品匹配（如应收账款质押贷、政策性担保贷）"
        )
    with col_cr8:
        # v1.6 新增：行业周期信号（ML 隐藏因子的生产输入，演示阶段人工录入；
        # 正式版接行业景气度/宏观数据源）
        industry_cycle = st.slider(
            "行业周期信号（ML 外部输入）", -1.0, 1.0, 0.0, 0.1,
            help="-1=行业深度下行，0=平稳，1=高度景气。正式版将由行业景气度数据自动接入"
        )

    # ======================== 诊断按钮 ========================
    if st.button("🔍 开始金融健康诊断", type="primary"):
        from modules.diagnosis import diagnose
        from modules.product_matching import match_products
        from modules.report_generator import generate_pdf
        from utils.llm_helper import call_llm

        # v1.5：将嵌套格式指标展平为简单字符串，兼容diagnosis.py等下游模块
        flat_metrics = {}
        for k, v in st.session_state.metrics.items():
            if k.startswith("__"):
                continue  # 跳过特殊键
            if isinstance(v, dict):
                flat_metrics[k] = v.get("value", "")
            else:
                flat_metrics[k] = str(v)

        full_metrics = {
            **flat_metrics,
            "经营年限": operating_years,
            "客户集中度": customer_concentration,
            "平均融资利率": avg_interest_rate,
            "行业": industry,
            "应收账款_3月内占比": ar_less_3m,
            "应收账款_3_12月占比": ar_3_12m,
            "应收账款_超12月占比": ar_over_12m,
            # v1.3 新增字段
            "纳税信用评级": tax_credit_rating if tax_credit_rating != "未评级" else "",
            "实控人征信状态": controller_credit,
            "法院执行记录": court_execution,
            "融资机构数量": financing_institution_count,
            "营收增长率": revenue_growth_rate,
            "净利润增长率": profit_growth_rate,
            # v1.5.1 新增
            "可提供抵押": can_collateral == "可提供",
            # v1.6 新增
            "行业周期信号": industry_cycle,
        }

        result = diagnose(full_metrics)
        st.session_state.diagnosis_result = result

        st.markdown("---")
        st.header("🔎 诊断结果")

        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.metric("总体健康评分", f"{result['overall_score']} / 10")
            if result['overall_score'] >= 7:
                st.success("整体健康")
            elif result['overall_score'] >= 4:
                st.warning("中等水平")
            else:
                st.error("高风险")

        st.subheader("各维度评分（8维体系）")
        dims = result['dimension_scores']
        lights = result['traffic_lights']
        # 2行4列布局展示8个维度
        row1_cols = st.columns(4)
        row2_cols = st.columns(4)
        dim_names = list(dims.keys())
        for i, dim in enumerate(dim_names):
            if i < 4:
                with row1_cols[i]:
                    st.metric(dim, f"{dims[dim]}/10")
                    st.markdown(lights[dim])
            else:
                with row2_cols[i - 4]:
                    st.metric(dim, f"{dims[dim]}/10")
                    st.markdown(lights[dim])

        # ======================== v1.6 双轨对照（规则卡 × ML） ========================
        st.subheader("🧠 双轨对照：规则卡 × ML 违约模型")
        try:
            from modules.ml_model import (predict_default_proba, dual_track_conclusion,
                                          explain_statement)
            # 构造 ML 输入：数值字段清洗为 float，分类字段原样（隐藏因子=行业周期滑块）
            _CATEGORICAL = {"纳税信用评级", "实控人征信状态", "法院执行记录",
                            "客户集中度", "行业", "可提供抵押"}
            statement = {k: (v if k in _CATEGORICAL else _to_num(v))
                         for k, v in full_metrics.items()}
            proba = predict_default_proba(statement)
            conclusion, tag = dual_track_conclusion(result['overall_score'], proba)
            col_m1, col_m2 = st.columns([1, 3])
            with col_m1:
                st.metric("ML 违约概率", "不可用" if proba is None else f"{proba * 100:.1f}%")
            with col_m2:
                st.markdown(f"**双轨结论**：{conclusion}")
            contribs = explain_statement(statement)
            if contribs:
                with st.expander("SHAP 归因（各因子对违约概率的贡献方向与强度）"):
                    shap_rows = sorted(
                        ((k, v) for k, v in contribs.items() if k not in ("bias", "违约概率")),
                        key=lambda kv: -abs(kv[1]))
                    df_shap = pd.DataFrame({"因子": [k for k, _ in shap_rows],
                                            "贡献": [round(v, 3) for _, v in shap_rows]})
                    st.bar_chart(df_shap.set_index("因子"))
                    st.caption("正值推高违约概率、负值拉低。模型为合成数据方法论演示，仅供对照参考。")
        except Exception as e:
            st.warning(f"双轨模块不可用，已降级为单轨规则卡：{e}")

        # LLM 增强诊断文本
        llm_text = generate_diagnosis_text(
            full_metrics, dims, result['overall_score'],
            st.session_state.model,
            st.session_state.api_key
        )

        if llm_text:
            overall_match = re.search(r'【总体评价】\s*(.*?)\s*【风险点】', llm_text, re.DOTALL)
            risks_match = re.search(r'【风险点】\s*(.*?)\s*【改善建议】', llm_text, re.DOTALL)
            suggestions_match = re.search(r'【改善建议】\s*(.*)', llm_text, re.DOTALL)

            if overall_match:
                st.subheader("📊 AI 诊断总结")
                st.markdown(overall_match.group(1).strip())

            if risks_match:
                st.subheader("⚠️ 核心风险点")
                for line in risks_match.group(1).strip().split('\n'):
                    if line.startswith('-'):
                        st.markdown(line)
            else:
                st.subheader("⚠️ 核心风险点")
                st.markdown("(AI 未生成风险点，使用规则引擎结果)")
                for r in result['risks']:
                    st.markdown(f"- {r}")

            if suggestions_match:
                st.subheader("💡 改善行动建议")
                for line in suggestions_match.group(1).strip().split('\n'):
                    if line.startswith('-'):
                        st.markdown(line)
            else:
                st.subheader("💡 改善行动建议")
                st.markdown("(AI 未生成建议，使用规则引擎结果)")
                for s in result['suggestions']:
                    st.markdown(f"- {s}")
        else:
            st.subheader("⚠️ 核心风险点")
            if result['risks']:
                for risk in result['risks']:
                    st.markdown(f"- {risk}")
            else:
                st.success("未发现明显风险点。")

            st.subheader("💡 改善行动建议")
            for sug in result['suggestions']:
                st.markdown(f"- {sug}")

        st.caption("以上诊断基于规则引擎与AI生成内容，仅供参考，不构成金融建议。")

        # ======================== 产品匹配 ========================
        st.markdown("---")
        st.header("🏦 匹配银行信贷产品")

        product_path = "knowledge/bank_products/products.csv"
        ai_recommendation_text = ""

        try:
            df_products = pd.read_csv(product_path, encoding="utf-8")
            matches = match_products(full_metrics, df_products)

            if matches:
                df_matches = pd.DataFrame(matches)
                display_cols = ["匹配度", "产品名", "银行", "产品类型", "额度", "利率", "准入条件", "差距说明"]
                st.dataframe(df_matches[display_cols], width='stretch')
                st.caption("数据来源及采集日期见产品库明细。")

                with st.spinner("🤖 AI 正在分析最佳产品方案..."):
                    product_list = "\n".join([
                        f"- {m['匹配度']} {m['产品名']}（{m['银行']}），额度：{m['额度']}万元，利率：{m['利率']}%，差距：{m['差距说明']}"
                        for m in matches
                    ])
                    rec_prompt = f"""
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
                    rec_system = "你是资深小微企业信贷顾问，语言简洁专业，用'建议'而非'你应该'。"
                    ai_rec = call_llm(rec_system, rec_prompt,
                                     st.session_state.model,
                                     st.session_state.api_key,
                                     max_tokens=500)
                    if ai_rec:
                        ai_recommendation_text = ai_rec
                        st.subheader("🤖 AI 产品推荐说明")
                        st.markdown(ai_rec)
            else:
                st.warning("未找到匹配的信贷产品，建议改善财务状况后再查询。")
                with st.spinner("🤖 AI 正在生成改善建议..."):
                    advice_prompt = f"""
                    企业当前未匹配到任何信贷产品，请根据企业情况给出2-3条具体改善建议，
                    帮助其未来达到银行准入条件。企业情况：
                    总资产：{full_metrics.get('总资产', '未填写')}万元
                    营业收入：{full_metrics.get('营业收入', '未填写')}万元
                    经营年限：{full_metrics.get('经营年限', '未填写')}年
                    行业：{full_metrics.get('行业', '未填写')}
                    纳税信用评级：{full_metrics.get('纳税信用评级', '未填写')}
                    征信状态：{full_metrics.get('实控人征信状态', '未填写')}
                    """
                    advice_system = "你是小微企业融资改善顾问，给出可操作的建议。"
                    ai_advice = call_llm(advice_system, advice_prompt,
                                         st.session_state.model,
                                         st.session_state.api_key,
                                         max_tokens=400)
                    if ai_advice:
                        ai_recommendation_text = ai_advice
                        st.subheader("🤖 AI 改善建议")
                        st.markdown(ai_advice)
        except Exception as e:
            st.error(f"产品匹配出错：{e}")

        # ======================== v1.4 差距分析与行动方案 ========================
        st.markdown("---")
        st.header("📊 差距分析与行动方案")

        try:
            from modules.gap_analysis import analyze_gaps

            df_products = pd.read_csv(product_path, encoding="utf-8")
            gap_result = analyze_gaps(full_metrics, result['dimension_scores'], df_products)

            # 行动优先级表格
            action_plan = gap_result.get("action_plan", [])
            if action_plan:
                st.subheader("🎯 行动优先级（按性价比排序）")
                st.markdown(
                    "性价比 = 解锁产品数 ÷ 提升难度分，数值越高表示先做这件事的收益最大。"
                )
                action_rows = []
                for act in action_plan:
                    # v1.5.1：区分"补齐即可解锁"与"涉及但仍卡在其他条件"两种口径
                    mode = act.get("impact_mode", "unlock")
                    if mode == "unlock":
                        impact_label = str(act["impact"])  # 统一字符串类型，避免混合类型导致渲染失败
                        products_label = "、".join(act["impact_products"][:5]) if act["impact_products"] else "—"
                    else:
                        impact_label = f"{act['impact']}（尚有其他差距）"
                        products_label = "、".join(act["impact_products"][:5]) + "…" if act["impact_products"] else "—"
                    action_rows.append({
                        "优先级": act["priority"],
                        "行动": act["action"],
                        "当前值": act["current"],
                        "目标值": act["target"],
                        "难度": act["difficulty"],
                        "影响产品数": impact_label,
                        "相关产品": products_label,
                        "预计时间": act["estimated_time"],
                        "性价比": act["cost_efficiency"],
                    })
                df_actions = pd.DataFrame(action_rows)
                st.dataframe(df_actions, width='stretch', hide_index=True)
            else:
                st.success("✅ 所有匹配产品均无差距，无需额外行动！")

            # 未达标产品的详细差距
            gap_details = gap_result.get("product_gap_details", [])
            gap_products = [p for p in gap_details if p["match_status"] == "差距匹配"]
            if gap_products:
                st.subheader("📋 各产品差距明细")
                for gp in gap_products:
                    with st.expander(f"{gp['product']}（{gp['bank']}）— {gp['product_type']}"):
                        if gp["gaps"]:
                            gap_table = []
                            for g in gp["gaps"]:
                                gap_table.append({
                                    "差距项": g["item"],
                                    "当前值": g["current"],
                                    "准入要求": g["required"],
                                    "差距量": f"{g['gap_size']:.1f}" if isinstance(g['gap_size'], (int, float)) else g['gap_size'],
                                    "提升难度": g["difficulty"],
                                })
                            st.dataframe(pd.DataFrame(gap_table), hide_index=True)
                            st.caption(f"💡 最容易补齐：{gp['closest_to_qualify']}")
                        else:
                            st.info("该产品已达标")

            # 总结
            summary = gap_result.get("summary", "")
            if summary:
                st.subheader("📝 总结")
                st.markdown(summary)

        except Exception as e:
            st.error(f"差距分析出错：{e}")

        # ======================== 报告下载 ========================
        st.markdown("---")
        st.header("📄 下载诊断报告")

        try:
            df_all = pd.read_csv(product_path, encoding="utf-8")

            ai_summary_text = ""
            ai_risks_text = ""
            ai_suggestions_text = ""

            if llm_text:
                overall_match = re.search(r'【总体评价】\s*(.*?)\s*【风险点】', llm_text, re.DOTALL)
                risks_match = re.search(r'【风险点】\s*(.*?)\s*【改善建议】', llm_text, re.DOTALL)
                suggestions_match = re.search(r'【改善建议】\s*(.*)', llm_text, re.DOTALL)

                if overall_match:
                    ai_summary_text = overall_match.group(1).strip()
                if risks_match:
                    ai_risks_text = risks_match.group(1).strip()
                if suggestions_match:
                    ai_suggestions_text = suggestions_match.group(1).strip()

            # v1.6：RAG 政策依据（自动检索政策语料，随报告落库；条款摘编标注）
            rag_report_citations = []
            rag_report_asof = ""
            try:
                from utils.vector_store import load_index, retrieve
                _idx = load_index("knowledge/rag_corpus/bm25_index.json")
                _phits = retrieve(_idx, "小微企业融资政策支持", k=3, category="policy")
                rag_report_citations = [
                    f"{h['title']}（{h['source']}）{(' ' + h['clause']) if h.get('clause') else ''}（条款摘编）"
                    for h in _phits]
                import json as _json
                with open("knowledge/rag_corpus/bm25_index.json", encoding="utf-8") as _f:
                    rag_report_asof = _json.load(_f).get("meta", {}).get("asof", "")
            except Exception:
                pass

            pdf_buffer = generate_pdf(
                full_metrics, result, matches, df_all,
                ai_summary=ai_summary_text,
                ai_risks=ai_risks_text,
                ai_suggestions=ai_suggestions_text,
                ai_recommendation=ai_recommendation_text,
                rag_citations=rag_report_citations,
                rag_asof=rag_report_asof
            )
            st.download_button(
                label="📥 下载 PDF 报告",
                data=pdf_buffer,
                file_name="融资诊断报告.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"报告生成失败：{e}")

else:
    st.info("👆 请上传企业的财务报表、银行流水或应收账款明细。")

# ======================== 底部产品库展示 ========================
st.markdown("---")
st.header("📋 银行信贷产品库（所有产品）")
try:
    df_all = pd.read_csv("knowledge/bank_products/products.csv", encoding="utf-8")
    st.success(f"✅ 已加载 {len(df_all)} 条产品数据")
    st.dataframe(df_all, width='stretch')
except Exception as e:
    st.error(f"产品库加载失败：{e}")
st.caption("数据来源：各银行官网，采集日期见表中字段。")

# ======================== v1.6 政策/产品智能问答（RAG） ========================
st.markdown("---")
st.header("📚 政策/产品智能问答（可溯源）")
try:
    from utils.vector_store import load_index, retrieve, grounded_answer
    from utils.llm_helper import call_llm as rag_call_llm

    @st.cache_resource
    def _load_rag():
        import json
        idx = load_index("knowledge/rag_corpus/bm25_index.json")
        try:
            with open("knowledge/rag_corpus/bm25_index.json", encoding="utf-8") as f:
                meta = json.load(f).get("meta", {})
        except Exception:
            meta = {}
        return idx, meta

    rag_index, rag_meta = _load_rag()
    rcol1, rcol2 = st.columns([3, 1])
    with rcol1:
        rag_query = st.text_input("问题", key="rag_query",
                                  placeholder="例：科技型小微企业无抵押能贷多少？")
    with rcol2:
        rag_cat = st.selectbox("语料范围", ["全部", "政策", "产品", "研报"], key="rag_cat")
    cat_map = {"全部": None, "政策": "policy", "产品": "product", "研报": "research"}
    if st.button("🔎 检索回答", key="rag_ask") and rag_query.strip():
        hits = retrieve(rag_index, rag_query.strip(), k=5, category=cat_map[rag_cat])
        if not hits:
            st.info("该问题未被本模块语料收录（诚实返回空，不编造）。可改问信贷产品或普惠政策，或查看上方产品库。")
        else:
            use_mock = not st.session_state.get("api_key")

            def _rag_llm(prompt, temperature=0):
                return rag_call_llm("你是小微金融政策与银行产品问答助手，仅基于给定资料回答。",
                                    prompt, st.session_state.model, st.session_state.api_key,
                                    temperature=temperature, max_tokens=600)

            ans = grounded_answer(None if use_mock else _rag_llm, rag_query.strip(), hits,
                                  use_mock=use_mock)
            if use_mock:
                st.caption("未配置 API Key：当前展示检索片段（配置后由 DeepSeek 做溯源生成）")
            if ans.text:
                st.markdown(ans.text)
            with st.expander(f"📖 参考清单（{len(ans.citations)} 条；政策条目为条款摘编）", expanded=True):
                for c in ans.citations:
                    st.markdown(f"- {c}")
            if rag_meta.get("asof"):
                st.caption(f"检索截至 {rag_meta['asof']} ｜ 产品库指纹 {str(rag_meta.get('products_csv_sha256', ''))[:12]}…")
except Exception as e:
    st.warning(f"RAG 模块不可用：{e}")
