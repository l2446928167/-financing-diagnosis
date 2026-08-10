"""
模块1：企业数据录入
支持上传 PDF / Excel / CSV 文件，提取关键财务指标。
包含规则提取和 LLM 智能提取（分块处理长文本、目录引导、详细错误提示）。
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
        expected_keys = ["总资产", "总负债", "营业收入", "净利润", "应收账款", "短期借款", "流动比率", "资产负债率"]
        for k in expected_keys:
            if k not in metrics:
                metrics[k] = ""
        return metrics
    except Exception as e:
        st.error(f"解析指标失败: {e}\n原始返回: {text[:300]}")
        return None


def llm_extract_metrics(raw_text, df, model_choice, api_key, call_llm_func):
    """
    使用大模型从原始文本或DataFrame中智能提取财务指标。
    支持长文本分块处理：
    1. 优先处理Excel/CSV表格
    2. 短文本直接提取
    3. 长文本分块提取摘要 → 合并浓缩 → 最终提取
    """
    if not api_key:
        st.warning("API Key 未填写，无法使用 AI 提取。")
        return None

    # 优先处理表格数据
    if df is not None:
        csv_str = df.head(50).to_csv(index=False)
        prompt = f"""从以下表格中提取财务指标（单位：万元）。输出JSON：
{{"总资产":"", "总负债":"", "营业收入":"", "净利润":"", "应收账款":"", "短期借款":"", "流动比率":"", "资产负债率":""}}
表格：
{csv_str}"""
        resp = call_llm_func(
            "你是财务数据提取器，仅输出JSON。",
            prompt, model_choice, api_key, temperature=0.1, max_tokens=800
        )
        if resp:
            return parse_metrics_response(resp)

    if not raw_text:
        st.warning("没有可提取的文本内容。")
        return None

    raw_text = raw_text.strip()
    text_len = len(raw_text)

    # 短文本直接提取
    if text_len <= 20000:
        st.info(f"文本长度 {text_len} 字符，直接提取...")
        prompt = f"从财报文本中提取财务指标（单位：万元）：\n{raw_text}\n输出JSON。"
        resp = call_llm_func(
            "你是财务提取器，仅输出JSON。",
            prompt, model_choice, api_key, temperature=0.1, max_tokens=800
        )
        return parse_metrics_response(resp)

    # 长文本分块处理
    CHUNK_SIZE = 18000
    OVERLAP = 500
    st.info(f"文本较长（{text_len}字符），分块提取摘要后再整合，预计30-60秒...")

    # 分块
    chunks = []
    start = 0
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        chunks.append(raw_text[start:end])
        start = end - OVERLAP

    total_chunks = len(chunks)
    st.write(f"共 {total_chunks} 块，正在逐块提取财务摘要...")
    progress_bar = st.progress(0)

    summaries = []
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"""从以下财报片段中提取所有出现的财务关键数字（如总资产、负债、营收、净利润、应收账款等），
用简洁中文列出，保留原始数值和单位。不要输出JSON。
片段：
{chunk}"""
        summary = call_llm_func(
            "你是财务摘要员，提取关键数字，用中文列出。",
            chunk_prompt, model_choice, api_key, temperature=0.1, max_tokens=600
        )
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
        if compressed:
            condensed = compressed

    # 最终提取
    st.write("正在从浓缩信息中提取最终指标...")
    final_prompt = f"""根据以下浓缩财务信息，提取关键财务指标（单位：万元，百分比保留%）。
浓缩信息：
{condensed}

输出严格JSON：
{{"总资产":"", "总负债":"", "营业收入":"", "净利润":"", "应收账款":"", "短期借款":"", "流动比率":"", "资产负债率":""}}"""
    final_resp = call_llm_func(
        "你是财务提取器，仅输出JSON。",
        final_prompt, model_choice, api_key, temperature=0.1, max_tokens=800
    )

    if final_resp:
        return parse_metrics_response(final_resp)
    else:
        st.error("最终提取失败，大模型未返回有效结果。请检查上方是否有红色错误提示。")
        return None