"""
模块1：企业数据录入
支持上传 PDF / Excel / CSV 文件，提取关键财务指标。
包含规则提取和 LLM 智能提取（支持长文本、目录引导、详细错误提示）。
"""
import pandas as pd
import PyPDF2
import io
import re
import json
import streamlit as st


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
        "资产负债率": ""
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

    # 如果从PDF提取了文本，尝试用关键词+数字正则提取（简化版）
    if raw_text:
        for key in metrics:
            if not metrics[key]:
                pattern = rf"{key}.*?([0-9,]+\.?\d*)"
                match = re.search(pattern, raw_text)
                if match:
                    metrics[key] = match.group(1)

    return metrics


def llm_extract_metrics(raw_text, df, model_choice, api_key, call_llm_func):
    """
    智能提取财务指标，支持长文本分块处理：
    1. 将超长文本分成多个块（每块约20000字符）
    2. 对每块生成一个“财务摘要”
    3. 将所有摘要合并成“浓缩全文”
    4. 从浓缩全文中提取最终指标
    """
    import streamlit as st
    import json, re

    if not api_key:
        st.warning("API Key 未填写，无法使用 AI 提取。")
        return None

    # 处理 DataFrame（Excel/CSV）
    if df is not None:
        # 如果同时有 raw_text 和 df，优先使用 df 直接提取（因为表格数据更好处理）
        csv_str = df.head(50).to_csv(index=False)
        # 直接尝试从表格提取
        prompt = f"""从以下表格数据中提取财务指标（单位：万元）：
{csv_str}
输出 JSON：{{"总资产":"", "总负债":"", "营业收入":"", "净利润":"", "应收账款":"", "短期借款":"", "流动比率":"", "资产负债率":""}}"""
        resp = call_llm_func(
            "你是一个财务数据提取器，仅输出JSON。",
            prompt,
            model_choice,
            api_key,
            temperature=0.1,
            max_tokens=800
        )
        if resp:
            return parse_metrics_response(resp)
        # 如果表格提取失败，继续利用 raw_text（如果有）

    if not raw_text:
        st.warning("没有可提取的文本内容。")
        return None

    # 分块参数
    CHUNK_SIZE = 20000  # 每块最大字符数（约5000-8000汉字，安全范围）
    OVERLAP = 1000  # 块之间重叠字符数，防止关键信息被截断

    raw_text = raw_text.strip()
    text_len = len(raw_text)

    # 如果文本较短，直接提取
    if text_len <= CHUNK_SIZE:
        st.info(f"文本长度 {text_len} 字符，正在直接提取...")
        prompt = f"从以下财报文本中提取关键财务指标（单位：万元）：\n{raw_text}\n输出JSON。"
        resp = call_llm_func(
            "你是一个财务数据提取器，仅输出JSON格式：{\"总资产\":\"\", \"总负债\":\"\", \"营业收入\":\"\", \"净利润\":\"\", \"应收账款\":\"\", \"短期借款\":\"\", \"流动比率\":\"\", \"资产负债率\":\"\"}",
            prompt,
            model_choice,
            api_key,
            temperature=0.1,
            max_tokens=800
        )
        return parse_metrics_response(resp)

    # --- 长文本分块处理 ---
    st.info(f"文本较长（{text_len}字符），将分块提取关键信息，共需3-5轮处理，请稍候...")

    # 第一步：分块提取摘要
    chunks = []
    start = 0
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        chunk = raw_text[start:end]
        chunks.append(chunk)
        start = end - OVERLAP if end < text_len else end
    total_chunks = len(chunks)
    st.write(f"已分为 {total_chunks} 个文本块，正在逐块提取财务摘要...")

    summaries = []
    progress_bar = st.progress(0)
    for idx, chunk in enumerate(chunks):
        # 对每个块提取财务摘要
        chunk_prompt = f"""请从以下财报文本片段中，提取所有出现的财务关键数字（如资产、负债、营收、净利润、应收账款等），
用简洁的文本列出，保留原始数值和单位。不要输出JSON，只需列出事实。
文本片段：
{chunk}
"""
        summary = call_llm_func(
            "你是一个财务数据摘要员，从财报文本中提取关键数字和指标，用简洁中文列出。",
            chunk_prompt,
            model_choice,
            api_key,
            temperature=0.1,
            max_tokens=600
        )
        if summary:
            summaries.append(summary)
        else:
            summaries.append(f"（第{idx + 1}块提取失败）")
        progress_bar.progress((idx + 1) / total_chunks)

    # 第二步：合并所有摘要
    condensed_text = "\n---\n".join([
        f"第{i + 1}部分财务摘要：\n{s}" for i, s in enumerate(summaries)
    ])

    # 如果合并后的文本仍然太长，可进一步压缩（但通常摘要很短）
    if len(condensed_text) > 30000:
        # 再压缩一次：让AI总结这些摘要
        compress_prompt = f"请将以下多段财务摘要整合为一份简洁的财务事实清单，保留所有关键数字：\n{condensed_text}"
        condensed_text = call_llm_func(
            "你是一个财务数据整合员，将多段摘要合并为一份完整的财务事实清单。",
            compress_prompt,
            model_choice,
            api_key,
            temperature=0.1,
            max_tokens=1000
        ) or condensed_text  # 如果压缩失败则用原始摘要
        st.write("已对摘要进行二次压缩。")

    # 第三步：从浓缩文本中提取最终指标
    st.write("正在从浓缩信息中提取最终财务指标...")
    final_prompt = f"""根据以下浓缩的财务摘要，提取关键财务指标（单位：万元，如为百分比则保留%）。
浓缩全文：
{condensed_text}

请严格按照以下JSON格式输出（找不到的留空）：
{{
    "总资产": "",
    "总负债": "",
    "营业收入": "",
    "净利润": "",
    "应收账款": "",
    "短期借款": "",
    "流动比率": "",
    "资产负债率": ""
}}"""
    final_resp = call_llm_func(
        "你是一个专业财务数据提取器，仅输出JSON，确保数值准确。",
        final_prompt,
        model_choice,
        api_key,
        temperature=0.1,
        max_tokens=800
    )
    if final_resp:
        return parse_metrics_response(final_resp)
    else:
        # 如果最终提取失败，尝试从摘要中直接用简单方式提取
        st.warning("最终提取失败，尝试从摘要直接解析...")
        return parse_metrics_response(condensed_text)  # 由parse处理文本


def parse_metrics_response(text):
    """辅助函数：解析AI返回的JSON或文本，提取指标字典"""
    import json, re
    import streamlit as st
    if not text:
        return None
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            metrics = json.loads(json_match.group())
        else:
            # 如果找不到JSON，尝试将整个文本作为JSON解析
            metrics = json.loads(text)
        expected_keys = ["总资产", "总负债", "营业收入", "净利润", "应收账款", "短期借款", "流动比率", "资产负债率"]
        for k in expected_keys:
            if k not in metrics:
                metrics[k] = ""
        return metrics
    except Exception as e:
        st.error(f"解析指标失败: {e}")
        return None