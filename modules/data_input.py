"""
模块1：企业数据录入
支持上传 PDF / Excel / CSV 文件，提取关键财务指标。
包含规则提取和 LLM 智能提取（支持长文本、目录引导）。
"""
import pandas as pd
import PyPDF2
import io
import re
import json

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
    """从PDF文件提取文本（简单提取所有文字）"""
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
    用简单的关键词匹配，适合快速预览。
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
    使用大模型从原始文本或DataFrame中智能提取财务指标。
    支持长文本（如200页财报），利用大上下文一次性提取，
    并引导AI先看目录再定位章节。
    """
    content = ""
    if raw_text:
        # 完整传入所有文本，不截断（DeepSeek-V4 支持128K上下文）
        content += "以下是从企业PDF财报中提取的完整文本（可能很长）：\n" + raw_text.strip()
    if df is not None:
        csv_str = df.head(50).to_csv(index=False)
        content += "\n以下是从Excel/CSV中读取的表格数据：\n" + csv_str

    if not content.strip():
        return None

    # 长文本友好提示
    text_length = len(content)
    if text_length > 50000:
        import streamlit as st
        st.warning(f"财报文本较长（约{text_length}字符），AI 提取可能需要 10-30 秒，请耐心等待。")

    system_prompt = """
    你是一位专业的财务数据提取专家，擅长从上市公司年度报告中提取关键财务指标。
    你将收到一份完整的PDF财报文本，其中可能包含目录、管理层讨论、财务报表附注等大量内容。
    
    请按以下步骤完成数据提取任务：

    1. 【定位目录】：
       - 首先在文本中搜索“目录”、“索引”或类似的章节列表，快速了解整个财报的结构。
       - 找到“合并资产负债表”、“合并利润表”、“主要会计数据及财务指标”等关键章节的位置描述。

    2. 【定位关键章节】：
       - 根据目录指引，直接跳转到以下章节（通常在财报后三分之一处）：
         * “合并资产负债表”
         * “合并利润表”
         * “主要会计数据及财务指标”（或“近三年主要会计数据和财务指标”）
         * 如有必要，可参考“财务报表附注”中的相关说明。

    3. 【提取数据】：
       从上述章节中提取以下指标（均以人民币万元为单位，百分比除外）：
       - 总资产：取“合并资产负债表”中的“资产总计”（期末余额）
       - 总负债：取“合并资产负债表”中的“负债合计”（期末余额）
       - 营业收入：取“合并利润表”中的“营业总收入”（本期金额）
       - 净利润：取“合并利润表”中的“净利润”（归属于母公司股东的净利润，如无则取“净利润合计”）
       - 应收账款：取“合并资产负债表”中的“应收账款”（期末余额）
       - 短期借款：取“合并资产负债表”中的“短期借款”（期末余额）
       - 流动比率：如表中已给出则直接使用，否则用流动资产÷流动负债计算
       - 资产负债率：如表中已给出则直接使用，否则用总负债÷总资产×100% 计算

    4. 【输出格式】：
       请严格按照以下 JSON 格式输出，不要包含任何解释、说明或markdown标记，仅输出 JSON 对象：
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

    注意：
    - 数字中可能含有千分位逗号，请去除后填入。
    - 如果找不到某个指标，该字段值留空字符串 ""。
    - 确保提取的数据来源正确，不要被管理层讨论中的预估值或去年数据误导。
    """
    user_prompt = content

    llm_response = call_llm_func(system_prompt, user_prompt, model_choice, api_key, temperature=0.1, max_tokens=800)
    if not llm_response:
        return None

    try:
        # 提取 JSON 部分
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            metrics = json.loads(json_match.group())
        else:
            metrics = json.loads(llm_response)
        # 确保所有键都存在
        expected_keys = ["总资产", "总负债", "营业收入", "净利润", "应收账款", "短期借款", "流动比率", "资产负债率"]
        for k in expected_keys:
            if k not in metrics:
                metrics[k] = ""
        return metrics
    except Exception as e:
        print(f"解析LLM提取结果失败: {e}")
        return None