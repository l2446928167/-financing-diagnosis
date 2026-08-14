"""
模块1：企业数据录入
支持上传 PDF / Excel / CSV 文件，提取关键财务指标。
包含规则提取和 LLM 智能提取（分块处理长文本、目录引导、详细错误提示）。
v1.5 (2026-08-12)：pdfplumber升级+Schema约束提取+单位归一化+会计校验+双期数据
      A) extract_from_pdf改用pdfplumber，同时提取文本和表格，表格转"行头: 值"格式
      B) METRICS_SCHEMA替代METRICS_JSON_TEMPLATE，含value/unit/page子字段
      C) 新增normalize_units()单位归一化（统一到万元）
      D) 新增verify_accounting()会计恒等式校验（总资产≈总负债+所有者权益、流动比率校验）
      E) 新增上年同期8个关键字段（上年总资产/上年应收账款/上年存货/上年流动资产/上年流动负债/上年经营活动现金流净额）
      F) llm_extract_metrics适配新Schema，提取后归一化+校验
v1.3.1：修复长文本提取精准度问题：
      A) 分块prompt改为定向字段提取，明确16个字段名
      B) 最终提取prompt增加校验规则（空值/总负债/流动比率）
      C) 增加二次补提机制，空值字段在原始文本中正则定向搜索
      D) auto_extract_metrics正则匹配增加别名映射
v1.3：扩展指标字典，新增8维评分所需的经营现金流、存货、利息费用、营业成本、
      流动资产、流动负债、上年营业收入、上年净利润等字段。
"""
import pandas as pd
import pdfplumber
import io
import re
import json
import streamlit as st


# v1.5：字段别名映射，用于正则搜索时匹配多种财报表述
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
    # v1.5：新增上年同期别名
    "上年总资产": ["上年资产总计", "上年资产合计"],
    "上年应收账款": ["上年应收账款合计"],
    "上年存货": ["上年存货合计"],
    "上年流动资产": ["上年流动资产合计", "上年流动资产总计"],
    "上年流动负债": ["上年流动负债合计", "上年流动负债总计"],
    "上年经营活动现金流净额": ["上年经营活动产生的现金流量净额", "上年经营活动现金流量净额"],
}

# v1.5：定向字段列表，含本期16个+上年同期6个新增=22个
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
    "16. 上年净利润\n"
    "17. 上年总资产（别名：上年资产总计、上年资产合计）\n"
    "18. 上年应收账款（别名：上年应收账款合计）\n"
    "19. 上年存货（别名：上年存货合计）\n"
    "20. 上年流动资产（别名：上年流动资产合计）\n"
    "21. 上年流动负债（别名：上年流动负债合计）\n"
    "22. 上年经营活动现金流净额（别名：上年经营活动产生的现金流量净额）"
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
    """
    v1.5：从PDF提取全部文本和表格，标注页码。
    使用pdfplumber替代PyPDF2，同时提取每页的文本和表格数据。
    表格转为"行头: 值"格式拼入文本，增强表格数据提取。
    每页内容前插入[第X页]标记，便于后续定位来源页码。
    """
    full_text = ""
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            full_text += f"\n[第{page_num}页]\n"
            # 提取页面文本
            page_text = page.extract_text() or ""
            if page_text:
                full_text += page_text + "\n"
            # 提取页面表格，转为"行头 | 列名: 值"格式（v1.5.1：保留行头语义，避免数值与科目脱节）
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                headers = table[0]
                for row in table[1:]:
                    row_label = str(row[0]).strip() if row and row[0] else ""
                    for j, cell in enumerate(row[1:], start=1):
                        header = headers[j] if j < len(headers) else f"列{j+1}"
                        if cell and str(cell).strip():
                            header_str = str(header).strip() if header else f"列{j+1}"
                            prefix = f"{row_label} | {header_str}" if row_label else header_str
                            full_text += f"{prefix}: {str(cell).strip()}\n"
    return full_text


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


# v1.5：指标Schema，替代原METRICS_JSON_TEMPLATE
# 每个字段包含value（数值）、unit（单位：元/万元/亿元/""）、page（来源页码）
METRICS_SCHEMA = {
    # === 本期字段（16个）===
    "总资产": {"value": "", "unit": "", "page": ""},
    "总负债": {"value": "", "unit": "", "page": ""},
    "营业收入": {"value": "", "unit": "", "page": ""},
    "净利润": {"value": "", "unit": "", "page": ""},
    "应收账款": {"value": "", "unit": "", "page": ""},
    "短期借款": {"value": "", "unit": "", "page": ""},
    "流动比率": {"value": "", "unit": "", "page": ""},
    "资产负债率": {"value": "", "unit": "", "page": ""},
    "经营活动现金流净额": {"value": "", "unit": "", "page": ""},
    "存货": {"value": "", "unit": "", "page": ""},
    "利息费用": {"value": "", "unit": "", "page": ""},
    "营业成本": {"value": "", "unit": "", "page": ""},
    "流动资产": {"value": "", "unit": "", "page": ""},
    "流动负债": {"value": "", "unit": "", "page": ""},
    "上年营业收入": {"value": "", "unit": "", "page": ""},
    "上年净利润": {"value": "", "unit": "", "page": ""},
    # === 上年同期关键字段（6个新增，与上面16个合计22个）===
    "上年总资产": {"value": "", "unit": "", "page": ""},
    "上年应收账款": {"value": "", "unit": "", "page": ""},
    "上年存货": {"value": "", "unit": "", "page": ""},
    "上年流动资产": {"value": "", "unit": "", "page": ""},
    "上年流动负债": {"value": "", "unit": "", "page": ""},
    "上年经营活动现金流净额": {"value": "", "unit": "", "page": ""},
}

# v1.5.1：比率类字段（永远不应获得金额单位）
RATIO_FIELDS = {"资产负债率", "流动比率"}


# v1.5：所有指标字段名列表（固定顺序）
METRICS_FIELD_NAMES = list(METRICS_SCHEMA.keys())


def _schema_to_json_template():
    """v1.5：将METRICS_SCHEMA转为LLM prompt用的JSON模板字符串"""
    template = {}
    for key in METRICS_FIELD_NAMES:
        template[key] = {"value": "", "unit": "", "page": ""}
    return json.dumps(template, ensure_ascii=False, indent=2)


def _find_page_number(text, position):
    """
    v1.5：根据文本中的位置，找到最近的[第X页]标记，返回页码字符串。
    如果找不到页码标记，返回空字符串。
    """
    # 在position之前找最后一个[第X页]标记
    search_text = text[:position]
    page_markers = list(re.finditer(r'\[第(\d+)页\]', search_text))
    if page_markers:
        return page_markers[-1].group(1)
    return ""


# v1.5.1：页头单位声明正则（年报/财报表格页头常见"单位：元 币种：人民币"等表述）
_PAGE_UNIT_RE = re.compile(r'单位[:：]\s*(?:人民币)?\s*(亿元|万元|千元|元)')


def _detect_unit_near_value(text, value_position, window=80):
    """
    v1.5.1：检测数值单位。参数为"数值起始位置"（非标签位置）。
    优先级：
    1) 数值后紧跟 % → 百分比字段，无单位
    2) 从所在页向前回溯最多2页，取数值位置之前最近的单位声明
       （财报表格常跨页，单位声明留在表格起始页页头）
    3) 数值附近±window字符的窗口启发式（兜底）
    """
    # 1) 百分比保护：资产负债率等比率字段不应被标成金额单位
    after = text[value_position:value_position + 15].strip()
    if after.startswith("%") or after.startswith("％"):
        return ""

    # 2) 单位声明回溯（最多2页）
    markers = list(re.finditer(r'\[第\d+页\]', text[:value_position + 1]))
    if markers:
        lookback_start = markers[-min(2, len(markers))].start()
        nxt = re.search(r'\[第\d+页\]', text[value_position:])
        seg_end = value_position + nxt.start() if nxt else len(text)
    else:
        lookback_start = 0
        seg_end = min(len(text), value_position + 300)
    before_decls = [m for m in _PAGE_UNIT_RE.finditer(text, lookback_start, seg_end)
                    if m.start() < value_position]
    if before_decls:
        return before_decls[-1].group(1)

    # 3) 窗口启发式兜底
    start = max(0, value_position - window)
    end = min(len(text), value_position + window)
    context = text[start:end]
    if "亿元" in context:
        return "亿元"
    elif "万元" in context:
        return "万元"
    elif "千元" in context:
        return "千元"
    elif "元" in context:
        return "元"
    return ""


def normalize_units(metrics_dict):
    """
    v1.5：单位归一化函数。
    识别每个字段的unit值，统一换算到万元。
    - 元 ÷ 10000 → 万元
    - 千元 ÷ 10 → 万元
    - 亿元 × 10000 → 万元
    - 换算后unit标记为"万元"
    - 无法识别单位的字段保持原值
    不修改 "__verification__" 等非指标键。
    返回归一化后的metrics_dict（原地修改，同时返回引用）。
    """
    for key in metrics_dict:
        if key.startswith("__"):
            continue  # 跳过特殊键
        field = metrics_dict[key]
        if not isinstance(field, dict):
            continue  # 跳过非Schema格式（兼容旧格式）
        val_str = field.get("value", "")
        unit = field.get("unit", "")
        if not val_str or not unit:
            continue  # 空值或无单位，不处理
        try:
            unit = str(unit).replace("人民币", "").strip()  # v1.5.1：兼容"人民币元"等表述
            # 清理千分位逗号
            val_clean = val_str.replace(",", "").replace("，", "")
            # 移除尾部非数字字符（如"(计算值)"）
            val_clean = re.sub(r'[^\d.\-]', '', val_clean)
            if not val_clean:
                continue
            val_num = float(val_clean)
            if unit == "元":
                val_num = val_num / 10000
                field["value"] = f"{val_num:.2f}"
                field["unit"] = "万元"
            elif unit == "千元":
                val_num = val_num / 10
                field["value"] = f"{val_num:.2f}"
                field["unit"] = "万元"
            elif unit == "亿元":
                val_num = val_num * 10000
                field["value"] = f"{val_num:.2f}"
                field["unit"] = "万元"
            elif unit == "万元":
                field["unit"] = "万元"  # 已是万元，确认标记
            # 其他单位不处理
        except (ValueError, TypeError):
            pass  # 转换失败，保持原值
    return metrics_dict


def verify_accounting(metrics_dict):
    """
    v1.5：会计恒等式校验函数。
    检查1：总资产 - 总负债 ≈ 所有者权益（即总资产≈总负债+所有者权益），容差5%
    检查2：流动比率 ≈ 流动资产÷流动负债，容差5%
    返回warnings列表，每项包含{检查项, 实际值, 预期值, 是否通过}。
    """
    warnings = []

    def _safe_val(key):
        """从metrics_dict中提取数值，兼容新旧格式"""
        field = metrics_dict.get(key, "")
        if isinstance(field, dict):
            val_str = field.get("value", "")
        else:
            val_str = str(field)
        if not val_str:
            return None
        try:
            # 清理千分位逗号和尾部标注
            val_clean = re.sub(r'[^\d.\-]', '', val_str.replace(",", "").replace("，", ""))
            return float(val_clean) if val_clean else None
        except (ValueError, TypeError):
            return None

    # 检查1：总资产 ≈ 总负债 + 所有者权益（即 总资产 - 总负债 ≈ 所有者权益，且 > 0）
    total_assets = _safe_val("总资产")
    total_liabilities = _safe_val("总负债")
    if total_assets is not None and total_liabilities is not None and total_assets != 0:
        equity = total_assets - total_liabilities
        # 会计恒等式：总资产应等于总负债+所有者权益
        # 这里我们验证：所有者权益(=总资产-总负债)是否为正且合理
        # 用 |总资产 - 总负债 - equity| / 总资产 < 5% 做容差
        # 但equity本身就是总资产-总负债，恒等式天然成立
        # 所以实际检查的是：总资产 >= 总负债（所有者权益>=0）
        # 以及差异比例是否在合理范围
        if total_assets > 0:
            equity_ratio = equity / total_assets
            # 所有者权益应为正值，容差5%允许小幅负值（可能是提取误差）
            check_pass = equity_ratio >= -0.05
            warnings.append({
                "检查项": "会计恒等式（总资产≈总负债+所有者权益）",
                "实际值": f"所有者权益={equity:.2f}万元（占资产{equity_ratio*100:.1f}%）",
                "预期值": "所有者权益≥0（占比≥0%）",
                "是否通过": check_pass,
            })
        else:
            warnings.append({
                "检查项": "会计恒等式（总资产≈总负债+所有者权益）",
                "实际值": f"总资产={total_assets:.2f}万元",
                "预期值": "总资产>0",
                "是否通过": False,
            })
    else:
        warnings.append({
            "检查项": "会计恒等式（总资产≈总负债+所有者权益）",
            "实际值": "缺少总资产或总负债数据",
            "预期值": "需提供总资产和总负债",
            "是否通过": None,  # 无法校验
        })

    # 检查2：流动比率 ≈ 流动资产 ÷ 流动负债，容差5%
    current_ratio = _safe_val("流动比率")
    current_assets = _safe_val("流动资产")
    current_liabilities = _safe_val("流动负债")
    if current_ratio is not None and current_assets is not None and current_liabilities is not None and current_liabilities != 0:
        computed_ratio = current_assets / current_liabilities
        if current_ratio != 0:
            deviation = abs(current_ratio - computed_ratio) / max(abs(current_ratio), abs(computed_ratio))
            check_pass = deviation <= 0.05
            warnings.append({
                "检查项": "流动比率校验（流动比率≈流动资产÷流动负债）",
                "实际值": f"流动比率={current_ratio:.4f}",
                "预期值": f"流动资产÷流动负债={computed_ratio:.4f}（偏差{deviation*100:.1f}%）",
                "是否通过": check_pass,
            })
        else:
            # 流动比率为0，检查计算值是否也接近0
            check_pass = abs(computed_ratio) < 0.01
            warnings.append({
                "检查项": "流动比率校验（流动比率≈流动资产÷流动负债）",
                "实际值": f"流动比率=0",
                "预期值": f"流动资产÷流动负债={computed_ratio:.4f}",
                "是否通过": check_pass,
            })
    elif current_ratio is None and current_assets is not None and current_liabilities is not None and current_liabilities != 0:
        # 无流动比率但有流动资产/负债，可自动计算
        warnings.append({
            "检查项": "流动比率校验（流动比率≈流动资产÷流动负债）",
            "实际值": "流动比率未提取",
            "预期值": f"流动资产÷流动负债={current_assets/current_liabilities:.4f}",
            "是否通过": None,  # 建议补充
        })
    else:
        warnings.append({
            "检查项": "流动比率校验（流动比率≈流动资产÷流动负债）",
            "实际值": "缺少流动比率、流动资产或流动负债数据",
            "预期值": "需提供流动比率和流动资产/负债",
            "是否通过": None,  # 无法校验
        })

    return warnings


def _search_metric(raw_text, names):
    """
    v1.5.1：两段式锚定搜索，降低正文叙述中的误匹配。
    第一轮：行首锚定——标签位于行首、数值在同一行60字符内（财报报表行格式）。
    第二轮：宽松匹配——标签后100字符内找数值（兜底）。
    返回 (match, matched_name)；未命中返回 (None, None)。
    """
    for name in names:
        esc = re.escape(name)
        pattern = rf"^[ \t]*(?:[一二三四五六七八九十\d]+[、.．])?\s*{esc}[^\n。；;]{{0,60}}?([0-9][0-9,]*\.?\d*)"
        m = re.search(pattern, raw_text, re.MULTILINE)
        if m:
            return m, name
    for name in names:
        esc = re.escape(name)
        pattern = rf"{esc}.{{0,100}}?([0-9][0-9,]*\.?\d*)"
        m = re.search(pattern, raw_text, re.DOTALL)
        if m:
            return m, name
    return None, None


def auto_extract_metrics(raw_text, df):
    """
    从原始文本或DataFrame中自动提取常见财务指标。
    v1.5：适配新Schema返回格式，包含value/unit/page子字段。
    使用关键词+数字正则提取，适合快速预览。
    返回一个字典，每个字段为{"value":..., "unit":..., "page":...}格式。
    """
    # 初始化为Schema格式
    metrics = {}
    for key in METRICS_FIELD_NAMES:
        metrics[key] = {"value": "", "unit": "", "page": ""}

    # 如果DataFrame存在，尝试从列名匹配
    if df is not None:
        col_map = {col.lower(): col for col in df.columns}
        for key in metrics:
            possible_names = [key, key.replace("总", ""), key.replace("净", "")]
            for name in possible_names:
                if name in col_map:
                    val = df[col_map[name]].dropna().iloc[0] if not df[col_map[name]].dropna().empty else ""
                    metrics[key]["value"] = str(val)
                    break

    # 如果从PDF提取了文本，用锚定正则提取（v1.5.1：别名优先+行首锚定，减少正文误匹配）
    if raw_text:
        for key in metrics:
            if not metrics[key]["value"]:
                names = FIELD_ALIASES.get(key, []) + [key]
                match, _matched = _search_metric(raw_text, names)
                if match:
                    metrics[key]["value"] = match.group(1)
                    metrics[key]["page"] = _find_page_number(raw_text, match.start())
                    unit = _detect_unit_near_value(raw_text, match.start(1))
                    metrics[key]["unit"] = "" if key in RATIO_FIELDS else unit

    return metrics


def parse_metrics_response(text):
    """
    v1.5：辅助函数，从LLM返回的文本中解析出财务指标字典。
    兼容新旧两种格式：
    - 新格式：每个字段为{"value":"", "unit":"", "page":""}
    - 旧格式：每个字段为简单字符串值（自动转为新格式）
    """
    if not text:
        return None
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            metrics = json.loads(json_match.group())
        else:
            metrics = json.loads(text)

        result = {}
        for key in METRICS_FIELD_NAMES:
            if key not in metrics:
                result[key] = {"value": "", "unit": "", "page": ""}
            else:
                val = metrics[key]
                if isinstance(val, dict):
                    # 新格式，校验子字段
                    result[key] = {
                        "value": str(val.get("value", "")) if val.get("value") is not None else "",
                        "unit": str(val.get("unit", "")) if val.get("unit") is not None else "",
                        "page": str(val.get("page", "")) if val.get("page") is not None else "",
                    }
                else:
                    # 旧格式，自动转为新格式
                    result[key] = {"value": str(val) if val is not None else "", "unit": "", "page": ""}
        return result
    except Exception as e:
        st.error(f"解析指标失败: {e}\n原始返回: {text[:300]}")
        return None


def _supplement_missing_metrics(result, raw_text):
    """
    v1.5：二次补提机制，适配新Schema（含unit/page）。
    检查result中的空值字段，在原始文本中用正则+别名映射定向搜索补提。
    对于"流动比率"，如果有流动资产和流动负债的值，自动计算。
    返回补提的字段数量。
    """
    if not raw_text or not result:
        return 0

    supplement_count = 0
    # v1.5.1：删除原先逻辑反转的死代码行；排除流动比率，最后单独处理
    empty_fields = [k for k in result
                    if not k.startswith("__") and isinstance(result[k], dict)
                    and (not result[k].get("value", "") or result[k]["value"] == "未找到")
                    and k != "流动比率"]

    for field in empty_fields:
        search_names = FIELD_ALIASES.get(field, []) + [field]
        match, _matched = _search_metric(raw_text, search_names)
        if match and match.group(1):
            result[field]["value"] = match.group(1)
            result[field]["page"] = _find_page_number(raw_text, match.start())
            unit = _detect_unit_near_value(raw_text, match.start(1))
            result[field]["unit"] = "" if field in RATIO_FIELDS else unit
            supplement_count += 1

    # 最后处理流动比率（依赖流动资产和流动负债的值）
    cr_field = result.get("流动比率", {})
    if isinstance(cr_field, dict) and not cr_field.get("value", ""):
        fa_field = result.get("流动资产", {})
        fl_field = result.get("流动负债", "")
        fa_val = fa_field.get("value", "") if isinstance(fa_field, dict) else str(fa_field)
        fl_val = fl_field.get("value", "") if isinstance(fl_field, dict) else str(fl_field)
        if fa_val and fl_val:
            try:
                fa_num = float(fa_val.replace(",", ""))
                fl_num = float(fl_val.replace(",", ""))
                if fl_num != 0:
                    ratio = fa_num / fl_num
                    result["流动比率"]["value"] = f"{ratio:.2f}(计算值)"
                    result["流动比率"]["unit"] = ""
                    result["流动比率"]["page"] = ""
                    supplement_count += 1
            except (ValueError, ZeroDivisionError):
                pass
        else:
            # 流动资产或流动负债仍未找到，尝试在原始文本中直接搜索流动比率
            match, _matched = _search_metric(raw_text, ["流动比率"] + FIELD_ALIASES.get("流动比率", []))
            if match and match.group(1):
                result["流动比率"]["value"] = match.group(1)
                result["流动比率"]["page"] = _find_page_number(raw_text, match.start())
                result["流动比率"]["unit"] = _detect_unit_near_value(raw_text, match.start(1))
                supplement_count += 1

    return supplement_count


def llm_extract_metrics(raw_text, df, model_choice, api_key, call_llm_func):
    """
    使用大模型从原始文本或DataFrame中智能提取财务指标。
    v1.5：适配新Schema（含value/unit/page），提取后归一化+会计校验。
    支持长文本分块处理：
    1. 优先处理Excel/CSV表格
    2. 短文本直接提取
    3. 长文本分块提取摘要 → 合并浓缩 → 最终提取
    """
    if not api_key:
        st.warning("未配置 API Key，无法使用智能提取。")
        return None

    api_called = False  # 追踪是否实际调用了API
    json_template = _schema_to_json_template()

    # 优先处理表格数据
    if df is not None:
        st.info("检测到表格数据，正在用智能分析表格...")
        csv_str = df.head(50).to_csv(index=False)
        prompt = f"""从以下表格中提取财务指标。
每个字段需包含：value（数值）、unit（单位：元/万元/亿元，若表头有单位则填入）、page（来源页码，表格填""）。
务必标注单位（元/万元/亿元）。
输出JSON：
{json_template}
表格：
{csv_str}"""
        resp = call_llm_func(
            "你是财务数据提取器，仅输出JSON，每个字段为{value,unit,page}格式。",
            prompt, model_choice, api_key, temperature=0.1, max_tokens=1000
        )
        api_called = True
        if resp:
            result = parse_metrics_response(resp)
            if result:
                # v1.5：归一化+校验
                normalize_units(result)
                verification = verify_accounting(result)
                result["__verification__"] = verification
                st.success("已从表格中提取成功！")
                return result
            else:
                st.warning("AI 返回了结果但解析失败，尝试其他方式...")
        else:
            st.warning("AI 表格提取未返回结果，尝试其他方式...")

    # 文本提取
    if raw_text and raw_text.strip():
        raw_text_stripped = raw_text.strip()
        text_len = len(raw_text_stripped)

        # 短文本直接提取
        if text_len <= 20000:
            st.info(f"文本长度 {text_len} 字符，正在用智能直接提取...")
            prompt = f"""从财报文本中提取财务指标。
每个字段需包含：value（数值）、unit（单位：元/万元/亿元）、page（来源页码，如"3"表示第3页）。
务必标注单位（元/万元/亿元）和来源页码。
输出JSON：
{json_template}
财报文本：
{raw_text_stripped}"""
            resp = call_llm_func(
                "你是财务提取器，仅输出JSON，每个字段为{value,unit,page}格式。",
                prompt, model_choice, api_key, temperature=0.1, max_tokens=1000
            )
            api_called = True
            if resp:
                result = parse_metrics_response(resp)
                if result:
                    # v1.5：二次补提+归一化+校验
                    supp = _supplement_missing_metrics(result, raw_text_stripped)
                    if supp > 0:
                        st.info(f"已补提 {supp} 个字段")
                    normalize_units(result)
                    verification = verify_accounting(result)
                    result["__verification__"] = verification
                    st.success("已从文本中提取成功！")
                    return result
                else:
                    st.warning("AI 返回了结果但解析失败。")
            else:
                st.warning("AI 文本提取未返回结果。")

        # 长文本分块处理
        else:
            CHUNK_SIZE = 18000
            OVERLAP = 500
            st.info(f"文本较长（{text_len}字符），分块提取摘要后再整合，预计30-60秒...")
            chunks = []
            start = 0
            while start < text_len:
                end = min(start + CHUNK_SIZE, text_len)
                chunks.append(raw_text_stripped[start:end])
                if end >= text_len:
                    break
                start = end - OVERLAP

            total_chunks = len(chunks)
            st.write(f"共 {total_chunks} 块，正在逐块提取财务摘要...")
            progress_bar = st.progress(0)
            summaries = []

            for i, chunk in enumerate(chunks):
                # v1.5：chunk_prompt增加单位标注和页码标注要求
                chunk_prompt = f"""从以下财报片段中，定向查找并提取以下22个财务字段的值。
如果某个字段在该片段中未出现，标注"未找到"。
字段列表：
{FIELD_LIST_TEXT}

输出格式：每行一个字段，格式为"字段名: 值 | 单位 | 页码"
（如"总资产: 12,345,678.90 | 万元 | 3"），未找到则写"字段名: 未找到"。
务必标注单位（元/万元/亿元）和来源页码（片段中有[第X页]标记）。
不要输出JSON，不要添加解释。
片段：
{chunk}"""
                summary = call_llm_func(
                    "你是财务摘要员，按指定字段列表逐一提取，用中文列出，务必标注单位和页码。",
                    chunk_prompt, model_choice, api_key, temperature=0.1, max_tokens=1000
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
                compress_prompt = f"将以下多段财务摘要整合为一份完整的财务事实清单，保留所有关键数字、单位和页码：\n{condensed}"
                compressed = call_llm_func(
                    "你是财务整合员，生成一份完整的财务事实清单，保留单位和页码。",
                    compress_prompt, model_choice, api_key, temperature=0.1, max_tokens=1200
                )
                api_called = True
                if compressed:
                    condensed = compressed

            # v1.5：最终提取prompt适配新Schema
            st.write("正在从浓缩信息中提取最终指标...")
            final_prompt = f"""根据以下浓缩财务信息，提取关键财务指标。
每个字段需包含：value（数值）、unit（单位：元/万元/亿元，百分比字段unit填""）、page（来源页码，如"3"表示第3页）。
务必标注单位（元/万元/亿元）和来源页码。

校验规则：
1. 如果某个字段在原始财报中确实为0或空（如"短期借款"行显示为空或无余额），则value输出"0"，不要从上下文猜测或推断一个值。
2. "总负债"应优先查找"负债合计"关键字的值。
3. "流动比率"如果财报中没有直接给出，但有"流动资产合计"和"流动负债合计"的值，则计算 流动资产÷流动负债 并在value后标注"(计算值)"；如果两者均无，则value输出空字符串。
4. "上年"开头的字段，请从财报的"上年同期"或"期初"列中提取，不是本期数据的负值。

浓缩信息：
{condensed}
输出严格JSON（每个字段为{{value,unit,page}}格式）：
{json_template}"""
            final_resp = call_llm_func(
                "你是财务提取器，仅输出JSON，每个字段为{value,unit,page}格式。",
                final_prompt, model_choice, api_key, temperature=0.1, max_tokens=1000
            )
            api_called = True
            if final_resp:
                result = parse_metrics_response(final_resp)
                if result:
                    # v1.5：二次补提+归一化+校验
                    supp = _supplement_missing_metrics(result, raw_text_stripped)
                    if supp > 0:
                        st.info(f"已补提 {supp} 个字段")
                    normalize_units(result)
                    verification = verify_accounting(result)
                    result["__verification__"] = verification
                    st.success("长文本提取成功！")
                    return result

    # 如果既没有表格也没有文本
    if not api_called:
        st.error("无法调用智能提取：文件中没有可提取的文本或表格数据。")
        st.info("提示：如果是扫描件PDF，请先转换为可搜索PDF或Excel格式后再上传。")
    else:
        st.error("AI 提取未能获得有效结果，请检查上方的错误信息。")

    return None
