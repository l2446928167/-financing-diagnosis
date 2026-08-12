"""
模块1：企业数据录入
支持上传 PDF / Excel / CSV 文件，提取关键财务指标。
包含规则提取和 LLM 智能提取（分块处理长文本、目录引导、详细错误提示）。
v1.3.1：修复长文本提取精准度问题：
      A) 分块prompt改为定向字段提取，明确16个字段名
      B) 最终提取prompt增加校验规则（空值/总负债/流动比率）
      C) 增加二次补提机制，空值字段在原始文本中正则定向搜索
      D) auto_extract_metrics正则匹配增加别名映射
v1.3：扩展指标字典，新增8维评分所需的经营现金流、存货、利息费用、营业成本、
      流动资产、流动负债、上年营业收入、上年净利润等字段。
"""
import pandas as pd
import PyPDF2
import io
import re
import json
import streamlit as st


# v1.3.1：字段别名映射，用于正则搜索时匹配多种财报表述（Fix D）
FIELD_ALIASES = {
    "总资产": ["资产总计", "资产合计"],
    "总负债": ["负债合计", "负债总计"],
    "营业收入": ["营业总收入", "营业收入合计"],
    "净利润": ["净利润合计", "归属于母公司所有者的净利润"],
    "应收账款": ["应收账款合计"],
    "短期借款": ["短期借款合计"],
    "流动资产": ["流动资产合计", "流动资产总计"],
    "流动负债": ["流动负债合计", "流动负债总计"],
    "经营活动现金流净额": ["经营活动产生的现金流量净额", "经营活动现金流量净额"],
    "存货": ["存货合计"],
    "利息费用": ["利息支出"],
    "营业成本": ["营业成本合计"],
    "上年营业收入": ["上年营业总收入"],
    "上年净利润": ["上年净利润合计"],
}

# v1.3.1：定向字段列表，用于分块提取prompt（Fix A）
FIELD_LIST_TEXT = (
    "1. 总资产（别名：资产总计、资产合计）\n"
    "2. 总负债（别名：负债合计、负债总计）\n"
    "3. 营业收入\n"
    "4. 净利润\n"
    "5. 应收账款\n"
    "6. 短期借款\n"
    "7. 流动比率\n"
    "8. 资产负债率\n"
    "9. 经营活动现金流净额（别名：经营活动产生的现金流量净额）\n"
    "10. 存货\n"
    "11. 利息费用\n"
    "12. 营业成本\n"
    "13. 流动资产（别名：流动资产合计）\n"
    "14. 流动负债（别名：流动负债合计）\n"
    "15. 上年营业收入\n"
    "16. 上年净利润"
)


def extract_from_csv(file):
    """从CSV文件读取数据框"""
    try:
        df = pd.read_csv(file, encoding="utf-8")
        return df
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="gbk")
        return df


def extract_from_excel(file):
    """从Excel文件读取第一个工作表"""
    df = pd.read_excel(file, engine="openpyxl")
    return df


def extract_from_pdf(file):
    """从PDF文件提取全部文本（所有页）"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def parse_financial_data(uploaded_file):
    """
    根据文件类型调用不同的解析函数。
    返回：
        - raw_text: 从PDF提取的原始文本（非PDF时为空）
        - df: 从CSV/Excel提取的数据框（非结构化文件时为空）
    """
    file_type = uploaded_file.name.split(".")[-1].lower()
    raw_text = ""
    df = None
    if file_type == "csv":
        df = extract_from_csv(uploaded_file)
    elif file_type in ["xls", "xlsx"]:
        df = extract_from_excel(uploaded_file)
    elif file_type == "pdf":
        raw_text = extract_from_pdf(uploaded_file)
    else:
        raise ValueError(f"不支持的文件格式：{file_type}")
    return raw_text, df


def auto_extract_metrics(raw_text, df):
    """
    从原始文本或DataFrame中自动提取常见财务指标。
    使用简单的关键词匹配，适合快速预览。
    v1.3.1：正则匹配增加别名映射（Fix D）。
    返回一个字典，包含指标名和提取到的值（字符串）。
    """
    metrics = {
        "总资产": "",
        "总负债": "",
        "营业收入": "",
        "净利润": "",
        "应收账款": "",
        "短期借款": "",
        "流动比率": "",
        "资产负债率": "",
        "经营活动现金流净额": "",
        "存货": "",
        "利息费用": "",
        "营业成本": "",
        "流动资产": "",
        "流动负债": "",
        "上年营业收入": "",
        "上年净利润": "",
    }
    # 如果DataFrame存在，尝试从列名匹配
    if df is not None:
        col_map = {col.lower(): col for col in df.columns}
        for key in metrics:
            possible_names = [key, key.replace("总", ""), key.replace("净", "")]
            for name in possible_names:
                if name in col_map:
                    val = df[col_map[name]].dropna().iloc[0] if not df[col_map[name]].dropna().empty else ""
                    metrics[key] = str(val)
                    break
    # 如果从PDF提取了文本，尝试用关键词+数字正则提取
    if raw_text:
        for key in metrics:
            if not metrics[key]:
                # v1.3.1：先搜索主名称
                pattern = rf"{key}.*?([0-9,]+\.?\d*)"
                match = re.search(pattern, raw_text)
                if match:
                    metrics[key] = match.group(1)
                    continue
                # v1.3.1：再搜索别名（Fix D）
                aliases = FIELD_ALIASES.get(key, [])
                for alias in aliases:
                    pattern = rf"{alias}.*?([0-9,]+\.?\d*)"
                    match = re.search(pattern, raw_text)
                    if match:
                        metrics[key] = match.group(1)
                        break
    return metrics


# v1.3：扩展的JSON模板，包含8维评分所需的所有指标
METRICS_JSON_TEMPLATE = (
    '{"总资产":"", "总负债":"", "营业收入":"", "净利润":"", '
    '"应收账款":"", "短期借款":"", "流动比率":"", "资产负债率":"", '
    '"经营活动现金流净额":"", "存货":"", "利息费用":"", "营业成本":"", '
    '"流动资产":"", "流动负债":"", "上年营业收入":"", "上年净利润":""}'
)


def parse_metrics_response(text):
    """辅助函数：从LLM返回的文本中解析出财务指标字典"""
    if not text:
        return None
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            metrics = json.loads(json_match.group())
        else:
            metrics = json.loads(text)
        # v1.3：扩展的expected_keys，包含8维评分所需的所有指标
        expected_keys = [
            "总资产", "总负债", "营业收入", "净利润", "应收账款", "短期借款",
            "流动比率", "资产负债率",
            "经营活动现金流净额", "存货", "利息费用", "营业成本",
            "流动资产", "流动负债", "上年营业收入", "上年净利润",
        ]
        for k in expected_keys:
            if k not in metrics:
                metrics[k] = ""
        return metrics
    except Exception as e:
        st.error(f"解析指标失败: {e}\n原始返回: {text[:300]}")
        return None


def _supplement_missing_metrics(result, raw_text):
    """
    v1.3.1：二次补提机制（Fix C）。
    检查result中的空值字段，在原始文本中用正则+别名映射定向搜索补提。
    对于"流动比率"，如果有流动资产和流动负债的值，自动计算。
    返回补提的字段数量。
    """
    if not raw_text or not result:
        return 0

    supplement_count = 0
    # 排除流动比率，最后单独处理（依赖流动资产和流动负债）
    empty_fields = [k for k, v in result.items()
                    if (not v or v == "未找到") and k != "流动比率"]

    for field in empty_fields:
        search_names = [field] + FIELD_ALIASES.get(field, [])
        for name in search_names:
            # 允许跨行匹配，限制标签与数值间距100字符以减少误匹配
            pattern = rf"{name}.{{0,100}}?([0-9,]+\.?\d*)"
            match = re.search(pattern, raw_text, re.DOTALL)
            if match and match.group(1):
                result[field] = match.group(1)
                supplement_count += 1
                break

    # 最后处理流动比率（依赖流动资产和流动负债的值）
    if not result.get("流动比率", ""):
        fa = result.get("流动资产", "")
        fl = result.get("流动负债", "")
        if fa and fl:
            try:
                fa_val = float(fa.replace(",", ""))
                fl_val = float(fl.replace(",", ""))
                if fl_val != 0:
                    ratio = fa_val / fl_val
                    result["流动比率"] = f"{ratio:.2f}(计算值)"
                    supplement_count += 1
            except (ValueError, ZeroDivisionError):
                pass
        else:
            # 流动资产或流动负债仍未找到，尝试在原始文本中直接搜索流动比率
            for name in ["流动比率"] + FIELD_ALIASES.get("流动比率", []):
                pattern = rf"{name}.{{0,100}}?([0-9,]+\.?\d*)"
                match = re.search(pattern, raw_text, re.DOTALL)
                if match and match.group(1):
                    result["流动比率"] = match.group(1)
                    supplement_count += 1
                    break

    return supplement_count


def llm_extract_metrics(raw_text, df, model_choice, api_key, call_llm_func):
    """
    使用大模型从原始文本或DataFrame中智能提取财务指标。
    v1.3.1：修复长文本提取精准度问题（Fix A/B/C）。
    支持长文本分块处理：
    1. 优先处理Excel/CSV表格
    2. 短文本直接提取
    3. 长文本分块提取摘要 → 合并浓缩 → 最终提取
    """
    if not api_key:
        st.warning("API Key 未填写，无法使用 AI 提取。")
        return None

    api_called = False  # 追踪是否实际调用了API

    # 优先处理表格数据
    if df is not None:
        st.info("📊 检测到表格数据，正在用 AI 分析表格...")
        csv_str = df.head(50).to_csv(index=False)
        prompt = f"""从以下表格中提取财务指标（单位：万元）。输出JSON：
{METRICS_JSON_TEMPLATE}
表格：
{csv_str}"""
        resp = call_llm_func(
            "你是财务数据提取器，仅输出JSON。",
            prompt, model_choice, api_key, temperature=0.1, max_tokens=800
        )
        api_called = True
        if resp:
            result = parse_metrics_response(resp)
            if result:
                st.success("✅ AI 从表格中提取成功！")
                return result
            else:
                st.warning("AI 返回了结果但解析失败，尝试其他方式...")
        else:
            st.warning("AI 表格提取未返回结果，尝试其他方式...")

    # 文本提取
    if raw_text and raw_text.strip():
        raw_text = raw_text.strip()
        text_len = len(raw_text)

        # 短文本直接提取
        if text_len <= 20000:
            st.info(f"📝 文本长度 {text_len} 字符，正在用 AI 直接提取...")
            prompt = f"从财报文本中提取财务指标（单位：万元）：\n{raw_text}\n输出JSON，格式如下：\n{METRICS_JSON_TEMPLATE}"
            resp = call_llm_func(
                "你是财务提取器，仅输出JSON。",
                prompt, model_choice, api_key, temperature=0.1, max_tokens=800
            )
            api_called = True
            if resp:
                result = parse_metrics_response(resp)
                if result:
                    # v1.3.1：二次补提（Fix C）
                    supp = _supplement_missing_metrics(result, raw_text)
                    if supp > 0:
                        st.info(f"🔧 已补提 {supp} 个字段")
                    st.success("✅ AI 从文本中提取成功！")
                    return result
                else:
                    st.warning("AI 返回了结果但解析失败。")
            else:
                st.warning("AI 文本提取未返回结果。")

        # 长文本分块处理
        else:
            CHUNK_SIZE = 18000
            OVERLAP = 500
            st.info(f"📄 文本较长（{text_len}字符），分块提取摘要后再整合，预计30-60秒...")
            chunks = []
            start = 0
            while start < text_len:
                end = min(start + CHUNK_SIZE, text_len)
                chunks.append(raw_text[start:end])
                if end >= text_len:
                    break
                start = end - OVERLAP

            total_chunks = len(chunks)
            st.write(f"共 {total_chunks} 块，正在逐块提取财务摘要...")
            progress_bar = st.progress(0)
            summaries = []

            for i, chunk in enumerate(chunks):
                # v1.3.1：chunk_prompt改为定向字段提取（Fix A）
                chunk_prompt = f"""从以下财报片段中，定向查找并提取以下16个财务字段的值。
如果某个字段在该片段中未出现，标注"未找到"。
字段列表：
{FIELD_LIST_TEXT}

输出格式：每行一个字段，格式为"字段名: 值"（如"总资产: 12,345,678.90"），未找到则写"字段名: 未找到"。
不要输出JSON，不要添加解释。
片段：
{chunk}"""
                summary = call_llm_func(
                    "你是财务摘要员，按指定字段列表逐一提取，用中文列出。",
                    chunk_prompt, model_choice, api_key, temperature=0.1, max_tokens=800
                )
                api_called = True
                if summary:
                    summaries.append(f"第{i+1}块摘要：\n{summary}")
                else:
                    summaries.append(f"第{i+1}块：提取失败")
                progress_bar.progress((i+1)/total_chunks)

            # 合并摘要
            condensed = "\n---\n".join(summaries)
            if len(condensed) > 30000:
                compress_prompt = f"将以下多段财务摘要整合为一份完整的财务事实清单，保留所有关键数字：\n{condensed}"
                compressed = call_llm_func(
                    "你是财务整合员，生成一份完整的财务事实清单。",
                    compress_prompt, model_choice, api_key, temperature=0.1, max_tokens=1000
                )
                api_called = True
                if compressed:
                    condensed = compressed

            # v1.3.1：最终提取prompt增加校验规则（Fix B）
            st.write("正在从浓缩信息中提取最终指标...")
            final_prompt = f"""根据以下浓缩财务信息，提取关键财务指标（单位：万元，百分比保留%）。

⚠️ 校验规则：
1. 如果某个字段在原始财报中确实为0或空（如"短期借款"行显示为空或无余额），则输出"0"，不要从上下文猜测或推断一个值。
2. "总负债"应优先查找"负债合计"关键字的值。
3. "流动比率"如果财报中没有直接给出，但有"流动资产合计"和"流动负债合计"的值，则计算 流动资产÷流动负债 并标注"(计算值)"；如果两者均无，则输出空字符串。

浓缩信息：
{condensed}
输出严格JSON：
{METRICS_JSON_TEMPLATE}"""
            final_resp = call_llm_func(
                "你是财务提取器，仅输出JSON。",
                final_prompt, model_choice, api_key, temperature=0.1, max_tokens=800
            )
            api_called = True
            if final_resp:
                result = parse_metrics_response(final_resp)
                if result:
                    # v1.3.1：二次补提（Fix C）
                    supp = _supplement_missing_metrics(result, raw_text)
                    if supp > 0:
                        st.info(f"🔧 已补提 {supp} 个字段")
                    st.success("✅ AI 长文本提取成功！")
                    return result

    # 如果既没有表格也没有文本
    if not api_called:
        st.error("⚠️ 无法调用 AI：文件中没有可提取的文本或表格数据。")
        st.info("💡 提示：如果是扫描件PDF，请先转换为可搜索PDF或Excel格式后再上传。")
    else:
        st.error("AI 提取未能获得有效结果，请检查上方的错误信息。")

    return None
