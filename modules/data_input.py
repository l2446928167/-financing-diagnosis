"""
模块1：企业数据录入
支持上传 PDF / Excel / CSV 文件，提取关键财务指标。
"""

import pandas as pd
import PyPDF2
import io

def extract_from_csv(file):
    """从CSV文件读取数据框"""
    try:
        df = pd.read_csv(file, encoding="utf-8")
        return df
    except UnicodeDecodeError:
        # 如果utf-8失败，尝试gbk编码
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
    目前用简单的关键词匹配（未来可接入大模型做更智能提取）。
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
        # 把列名统一转为小写方便匹配
        col_map = {col.lower(): col for col in df.columns}
        for key in metrics:
            # 简单匹配：比如"总资产"可能对应列名"总资产"或"资产总计"
            possible_names = [key, key.replace("总", ""), key.replace("净", "")]
            for name in possible_names:
                if name in col_map:
                    # 取该列第一个非空值
                    val = df[col_map[name]].dropna().iloc[0] if not df[col_map[name]].dropna().empty else ""
                    metrics[key] = str(val)
                    break

    # 如果从PDF提取了文本，尝试用关键词+数字正则提取（简化版，可后续用AI增强）
if raw_text:
    import re
    # 非常基础的提取：找"营业收入"后面的数字（仅示例，实际不准确）
    for key in metrics:
        if not metrics[key]:  # 如果DataFrame中没提取到
            pattern = rf"{key}.*?([0-9,]+\.?\d*)"
            match = re.search(pattern, raw_text)
            if match:
                metrics[key] = match.group(1)

def llm_extract_metrics(raw_text, df, model_choice, api_key, call_llm_func):
    """
    使用大模型从原始文本或DataFrame中智能提取财务指标。
    参数：
        raw_text: PDF提取的原始文本
        df: 从CSV/Excel读取的DataFrame
        model_choice, api_key: 模型和密钥
        call_llm_func: 调用LLM的函数（从外部传入，避免循环导入）
    返回：
        dict: 提取到的指标字典，格式与现有auto_extract_metrics相同
        如果失败返回None
    """
    # 准备要发送给LLM的内容
    content = ""
    if raw_text:
        # 限制长度，防止token超限（视模型上下文长度调整，这里取前4000字符）
        content += "以下是从企业财务报表PDF中提取的文本：\n" + raw_text[:4000]
    if df is not None:
        # 将DataFrame转为CSV字符串，限制前50行
        csv_str = df.head(50).to_csv(index=False)
        content += "\n以下是从Excel/CSV中读取的表格数据：\n" + csv_str

    if not content.strip():
        return None

    system_prompt = """
    你是一位专业的财务数据提取专家。请从提供的企业财务文本或表格中，
    提取以下关键财务指标（如果找不到，则留空字符串）。
    请严格按照JSON格式输出，不要包含任何其他文字。
    指标包括：
    {
        "总资产": "",
        "总负债": "",
        "营业收入": "",
        "净利润": "",
        "应收账款": "",
        "短期借款": "",
        "流动比率": "",
        "资产负债率": ""
    }
    注意：所有数值请保留原始单位（通常为万元），如果是百分比请保留原样。
    """

    user_prompt = content

    llm_response = call_llm_func(system_prompt, user_prompt, model_choice, api_key, temperature=0.1, max_tokens=500)
    if not llm_response:
        return None

    # 解析JSON
    import json
    try:
        # 尝试提取JSON部分（有时LLM会在前后加说明）
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            metrics = json.loads(json_match.group())
        else:
            metrics = json.loads(llm_response)
        # 确保所有键存在
        expected_keys = ["总资产", "总负债", "营业收入", "净利润", "应收账款", "短期借款", "流动比率", "资产负债率"]
        for k in expected_keys:
            if k not in metrics:
                metrics[k] = ""
        return metrics
    except Exception as e:
        print(f"解析LLM提取结果失败: {e}")
        return None
return metrics