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
    import streamlit as st
    if not api_key:
        st.warning("API Key 未填写，无法使用 AI 提取。")
        return None

    content = ""
    if raw_text:
        # 缩短截断长度，避免 token 超限（128K token ≈ 8-10万汉字，我们取保守值50000字符）
        max_chars = 50000
        if len(raw_text) > max_chars:
            raw_text = raw_text[:max_chars]
            st.warning(f"财报文本过长，已截取前 {max_chars} 字符用于提取，可能缺失末尾信息。")
        content += "以下是从企业PDF财报中提取的文本：\n" + raw_text.strip()
    if df is not None:
        csv_str = df.head(50).to_csv(index=False)
        content += "\n以下是从Excel/CSV中读取的表格数据：\n" + csv_str

    if not content.strip():
        st.warning("没有可提取的文本内容。")
        return None

    text_length = len(content)
    if text_length > 30000:
        st.warning(f"财报文本较长（{text_length}字符），AI 提取可能需要 10-30 秒，请耐心等待。")

    system_prompt = """
    你是一位专业的财务数据提取专家。请从提供的企业财务报表文本中，
    提取以下关键财务指标（以人民币万元为单位，如为百分比则保留原样）。
    如果找不到，则留空字符串 ""。

    请严格按照以下 JSON 格式输出，不要包含任何解释或markdown：
    {
        "总资产": "数值（万元）",
        "总负债": "数值（万元）",
        "营业收入": "数值（万元）",
        "净利润": "数值（万元）",
        "应收账款": "数值（万元）",
        "短期借款": "数值（万元）",
        "流动比率": "数值（如1.5）",
        "资产负债率": "数值（如65.2%）"
    }
    """
    user_prompt = content

    # 调用 LLM，并捕获可能的异常（虽然 call_llm 内部已处理，但这里再保险一次）
    try:
        llm_response = call_llm_func(system_prompt, user_prompt, model_choice, api_key, temperature=0.1, max_tokens=800)
    except Exception as e:
        st.error(f"调用 LLM 时发生异常：{e}")
        return None

    if not llm_response:
        st.error("LLM 未返回结果。可能原因：文本过长导致 token 超限，请尝试上传更小的文件；或 API 暂时不可用。")
        return None

    # 解析 JSON
    import json, re
    try:
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            metrics = json.loads(json_match.group())
        else:
            metrics = json.loads(llm_response)
        expected_keys = ["总资产", "总负债", "营业收入", "净利润", "应收账款", "短期借款", "流动比率", "资产负债率"]
        for k in expected_keys:
            if k not in metrics:
                metrics[k] = ""
        return metrics
    except Exception as e:
        st.error(f"解析 AI 返回结果失败：{e}\n\n原始返回内容：\n{llm_response[:500]}...")
        return None