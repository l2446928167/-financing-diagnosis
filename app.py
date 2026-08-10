import streamlit as st
import pandas as pd
import re
import os
from dotenv import load_dotenv, set_key
from modules.data_input import parse_financial_data, auto_extract_metrics
from modules.diagnosis import diagnose
from modules.product_matching import match_products
from modules.report_generator import generate_pdf
from utils.llm_helper import call_llm

# 加载 .env 中的环境变量
load_dotenv()
saved_api_key = os.environ.get("DEEPSEEK_API_KEY", "")

st.set_page_config(page_title="小微企业融资诊断工具", layout="wide")

st.title("🏦 AI + 小微企业融资诊断系统")
st.markdown("上传企业财务数据，获取健康诊断与信贷产品匹配建议。")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 配置区")
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

# ---------- 初始化 session_state ----------
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "df" not in st.session_state:
    st.session_state.df = None
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None

# ---------- 辅助函数：用LLM生成诊断文本 ----------
def generate_diagnosis_text(full_metrics, dims, overall, model_choice, api_key):
    system_prompt = """
    你是一位资深的小微企业融资顾问。根据提供的企业财务指标和5维度健康评分，
    请输出以下内容（用中文）：
    1. 总体评价：100字以内的概括性评价。
    2. 风险点：不超过5条具体风险，每条以"- "开头，语气客观，使用"建议"而非"你应该"。
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

    各维度评分（满分10）：
    """
    for dim, score in dims.items():
        user_prompt += f"\n- {dim}：{score}"
    user_prompt += f"\n总体健康评分：{overall}/10"

    return call_llm(system_prompt, user_prompt, model_choice, api_key)

# ===================== 模块1：数据录入 =====================
st.header("📤 第一步：上传企业财务数据")
uploaded_file = st.file_uploader("支持 PDF / Excel / CSV 格式", type=["pdf", "xlsx", "xls", "csv"])

if uploaded_file is not None:
    with st.spinner("正在解析文件..."):
        try:
            raw_text, df = parse_financial_data(uploaded_file)
            st.session_state.raw_text = raw_text
            st.session_state.df = df
            extracted = auto_extract_metrics(raw_text, df)
            st.session_state.metrics = extracted
            st.success("文件解析完成！")
        except Exception as e:
            st.error(f"解析失败：{e}")
            st.stop()

    st.subheader("📊 自动提取的关键财务指标（可手动修正）")
    col1, col2, col3 = st.columns(3)
    updated_metrics = {}
    keys = list(st.session_state.metrics.keys())
    for i, key in enumerate(keys):
        with [col1, col2, col3][i % 3]:
            updated_metrics[key] = st.text_input(
                label=key,
                value=st.session_state.metrics[key],
                key=f"metric_{key}"
            )
    st.session_state.metrics = updated_metrics

    # 添加 AI 智能提取按钮（如果已配置 API Key）
    if st.session_state.get('api_key'):
        if st.button("🤖 AI 智能提取财务指标"):
            with st.spinner("AI 正在分析文件内容..."):
                from utils.llm_helper import call_llm
                # 导入我们刚写的函数
                from modules.data_input import llm_extract_metrics
                ai_metrics = llm_extract_metrics(
                    st.session_state.raw_text,
                    st.session_state.df,
                    st.session_state.get('model', ''),
                    st.session_state.api_key,
                    call_llm  # 传入函数本身
                )
                if ai_metrics:
                    # 更新 session_state 中的指标字典
                    st.session_state.metrics = ai_metrics
                    st.success("AI 提取完成！请检查下方指标并手动修正。")
                    # 刷新页面以更新输入框的值（通过 rerun）
                    st.rerun()
                else:
                    st.error("AI 提取失败，请检查 API Key 或文件内容。")
    else:
        st.info("💡 在侧边栏配置 API Key 后可使用 AI 智能提取财务指标。")

    with st.expander("查看原始解析数据"):
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df)
        if st.session_state.raw_text:
            text = st.session_state.raw_text
            st.text(text[:2000] + "..." if len(text) > 2000 else text)

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

    # ===================== 诊断按钮 =====================
    if st.button("🔍 开始金融健康诊断", type="primary"):
        full_metrics = {
            **st.session_state.metrics,
            "经营年限": operating_years,
            "客户集中度": customer_concentration,
            "平均融资利率": avg_interest_rate,
            "行业": industry,
            "应收账款_3月内占比": ar_less_3m,
            "应收账款_3_12月占比": ar_3_12m,
            "应收账款_超12月占比": ar_over_12m
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

        st.subheader("各维度评分")
        dims = result['dimension_scores']
        lights = result['traffic_lights']
        cols = st.columns(5)
        dim_names = list(dims.keys())
        for i, dim in enumerate(dim_names):
            with cols[i]:
                st.metric(dim, f"{dims[dim]}/10")
                st.markdown(lights[dim])

        # LLM 增强诊断文本
        llm_text = generate_diagnosis_text(
            full_metrics, dims, result['overall_score'],
            st.session_state.get('model', ''),
            st.session_state.get('api_key', '')
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

        # ===================== 产品匹配 =====================
        st.markdown("---")
        st.header("🏦 匹配银行信贷产品")

        product_path = "knowledge/bank_products/products.csv"
        ai_recommendation_text = ""

        try:
            df_products = pd.read_csv(product_path, encoding="utf-8")
            matches = match_products(full_metrics, df_products)

            if matches:
                df_matches = pd.DataFrame(matches)
                display_cols = ["匹配度", "产品名", "银行", "额度", "利率", "准入条件", "差距说明"]
                st.dataframe(df_matches[display_cols], use_container_width=True)
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

                    匹配产品列表：
                    {product_list}

                    请直接输出推荐内容，不要使用markdown标题。
                    """
                    rec_system = "你是资深小微企业信贷顾问，语言简洁专业，用'建议'而非'你应该'。"
                    ai_rec = call_llm(rec_system, rec_prompt,
                                     st.session_state.get('model', ''),
                                     st.session_state.get('api_key', ''),
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
                    """
                    advice_system = "你是小微企业融资改善顾问，给出可操作的建议。"
                    ai_advice = call_llm(advice_system, advice_prompt,
                                         st.session_state.get('model', ''),
                                         st.session_state.get('api_key', ''),
                                         max_tokens=400)
                    if ai_advice:
                        ai_recommendation_text = ai_advice
                        st.subheader("🤖 AI 改善建议")
                        st.markdown(ai_advice)
        except Exception as e:
            st.error(f"产品匹配出错：{e}")

        # ===================== 报告下载 =====================
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

            pdf_buffer = generate_pdf(
                full_metrics, result, matches, df_all,
                ai_summary=ai_summary_text,
                ai_risks=ai_risks_text,
                ai_suggestions=ai_suggestions_text,
                ai_recommendation=ai_recommendation_text
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

# ===================== 底部产品库展示 =====================
st.markdown("---")
st.header("📋 银行信贷产品库（所有产品）")
product_path = "knowledge/bank_products/products.csv"
try:
    df_all = pd.read_csv(product_path, encoding="utf-8")
    st.success(f"✅ 已加载 {len(df_all)} 条产品数据")
    st.dataframe(df_all, use_container_width=True)
except FileNotFoundError:
    st.error("❌ 未找到产品库文件，请检查 knowledge/bank_products/products.csv 是否存在")
except Exception as e:
    st.error(f"❌ 读取产品库出错：{e}")

st.caption("数据来源：各银行官网，采集日期见表中字段。")