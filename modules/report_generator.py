"""
模块4：诊断报告 PDF 生成（对话式 v3.0 重构版）

改进点（相对 v2.0 / 基线）：
  1. 字体自包含：优先使用 reportlab 内置 CJK 字体 STSong-Light，不再依赖
     外部 fonts/*.ttf（缺失时中文会乱码）；STSong-Light 在 Streamlit Cloud
     等标准 reportlab 安装下默认可用。
  2. 修复表格溢出 bug：所有表格单元格一律用 Paragraph 包裹，长文本（尤其
     「差距说明」「相关产品」「准入条件」）按列宽自动换行，彻底消除 A4 溢出。
  3. 内容增维（依用户勾选）：
     - ML 双轨对照节（违约概率 + 规则卡×ML 双轨结论）
     - SHAP 归因节（各因子对违约概率的贡献方向与强度）
     - 差距分析与行动方案表（按性价比排序）
     - 诊断指标明细表（全部指标取值，提升透明度）
  4. 规范化排版：统一字体、分节标题（主色 #2F54EB）、彩色表头、统一网格与内边距。
     （依用户选择：不做独立封面页、不加页码与页脚。）
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from xml.sax.saxutils import escape
import io, datetime

# ---------- 字体注册（自包含 CJK） ----------
try:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    BASE = "STSong-Light"          # 同时覆盖中文与拉丁字符，无需外部 ttf
except Exception:
    BASE = "Helvetica"             # 兜底（此时中文可能乱码，仅应急）

# ---------- 设计 token（与 UI v3.0 协调） ----------
C_PRIMARY = colors.HexColor("#2F54EB")   # 主色 / 表头
C_TEXT    = colors.HexColor("#1F2733")   # 正文深
C_SUB     = colors.HexColor("#5B6573")   # 次级文字
C_BORDER  = colors.HexColor("#E6E8EC")   # 网格线
C_GREEN   = colors.HexColor("#2BA471")   # 健康
C_AMBER   = colors.HexColor("#D98B1F")   # 关注
C_RED     = colors.HexColor("#E5484D")   # 高风险
C_HEADER_TXT = colors.white

# 供 <font color> 标签使用的 hex 字符串（reportlab 段落解析器要求 #RRGGBB）
HEX_GREEN = "#2BA471"
HEX_AMBER = "#D98B1F"
HEX_RED   = "#E5484D"


def _level_hex(text):
    if "健康" in text:
        return HEX_GREEN
    elif "关注" in text:
        return HEX_AMBER
    return HEX_RED


def _esc(text):
    """转义 HTML 特殊字符，保证单元格文本安全且可自动换行。"""
    if text is None:
        return ""
    return escape(str(text))


def _line_break(text):
    """纯文本转 <br/> 换行（先转义，再替换换行符）。"""
    return _esc(text).replace("\n", "<br/>")


def generate_pdf(metrics, diagnosis_result, matches, product_df,
                 ai_summary=None, ai_risks=None, ai_suggestions=None,
                 ai_recommendation=None, rag_citations=None, rag_asof=None,
                 ml_proba=None, ml_conclusion=None, shap_contribs=None,
                 gap_result=None, policy_result=None):
    """
    生成诊断报告 PDF（返回 BytesIO）。

    新增形参（与 app.py do_report() 调用一致）：
        ml_proba:        ML 违约概率（float 0~1）或 None
        ml_conclusion:   双轨组合结论文本 或 None
        shap_contribs:   SHAP 贡献 dict（含 'bias'/'违约概率' 等，将过滤）或 None
        gap_result:      gap_analysis.analyze_gaps() 返回值 或 None
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="小微企业融资诊断报告",
    )

    # ---- 样式 ----
    title_style = ParagraphStyle("Title", fontName=BASE, fontSize=18,
                                 leading=24, alignment=1, textColor=C_TEXT,
                                 spaceAfter=2)
    sub_style = ParagraphStyle("Sub", fontName=BASE, fontSize=9,
                               leading=13, alignment=1, textColor=C_SUB,
                               spaceAfter=4)
    h1_style = ParagraphStyle("H1", fontName=BASE, fontSize=12.5,
                              leading=16, textColor=C_PRIMARY,
                              spaceBefore=12, spaceAfter=5)
    body_style = ParagraphStyle("Body", fontName=BASE, fontSize=9.5,
                                leading=15, textColor=C_TEXT, spaceAfter=3)
    note_style = ParagraphStyle("Note", fontName=BASE, fontSize=7.5,
                                leading=10, textColor=C_SUB, spaceAfter=2)
    cell_style = ParagraphStyle("Cell", fontName=BASE, fontSize=8,
                                leading=10.5, textColor=C_TEXT)
    cell_hdr_style = ParagraphStyle("CellH", fontName=BASE, fontSize=8,
                                    leading=10.5, textColor=C_HEADER_TXT)
    bullet_style = ParagraphStyle("Bullet", fontName=BASE, fontSize=9.5,
                                  leading=15, textColor=C_TEXT,
                                  leftIndent=8, spaceAfter=2)
    warn_style = ParagraphStyle("Warn", fontName=BASE, fontSize=9,
                                leading=14, textColor=C_RED, spaceAfter=2)

    elements = []

    # ===== 顶部标题块（非独立封面页） =====
    elements.append(Paragraph("小微企业融资诊断报告", title_style))
    elements.append(Paragraph(
        f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
    elements.append(HRFlowable(width="100%", thickness=1.2, color=C_PRIMARY))
    elements.append(Spacer(1, 4 * mm))

    # ===== 一、企业概况 =====
    elements.append(Paragraph("企业概况", h1_style))
    overview_pairs = [
        ("总资产", metrics.get("总资产", "未填写")),
        ("营业收入", metrics.get("营业收入", "未填写")),
        ("净利润", metrics.get("净利润", "未填写")),
        ("经营年限", metrics.get("经营年限", "未填写")),
        ("所属行业", metrics.get("行业", "未填写")),
        ("平均融资利率", metrics.get("平均融资利率", "未填写")),
        ("纳税信用评级", metrics.get("纳税信用评级", "未填写")),
        ("实控人征信状态", metrics.get("实控人征信状态", "未填写")),
    ]
    ov_data = [[Paragraph("指标", cell_hdr_style), Paragraph("取值", cell_hdr_style)]]
    for k, v in overview_pairs:
        ov_data.append([Paragraph(_esc(k), cell_style),
                        Paragraph(_esc(v), cell_style)])
    ov_tbl = Table(ov_data, colWidths=[150, 240], repeatRows=1)
    ov_tbl.setStyle(_tbl_style())
    elements.append(ov_tbl)
    elements.append(Spacer(1, 3 * mm))

    # ===== 二、金融健康评分 =====
    elements.append(Paragraph("金融健康评分", h1_style))
    overall = diagnosis_result.get("overall_score", 0)
    lvl_txt, lvl_color = _score_level(overall)
    elements.append(Paragraph(
        f"总体健康评分：<b>{overall} / 10</b>　"
        f"<font color='{_level_hex(lvl_txt)}'><b>{lvl_txt}</b></font>", body_style))
    elements.append(Spacer(1, 2 * mm))

    # --- 诊断结论与风险概率 ---
    elements.append(Paragraph("诊断结论与风险概率", body_style))
    if ml_proba is not None:
        elements.append(Paragraph(
            f"· 违约风险概率：<b>{ml_proba * 100:.1f}%</b>", body_style))
        if ml_conclusion:
            elements.append(Paragraph(f"· 综合结论：{_line_break(ml_conclusion)}",
                                     body_style))
        else:
            elements.append(Paragraph(
                "· 综合结论：风险模型可用，但结论未生成（可配置 API Key 后重试）。",
                body_style))
    else:
        elements.append(Paragraph(
            "· 当前为<b>综合评分结论</b>：风险模型不可用（模型文件缺失或未配置），"
            "未提供违约概率。", warn_style))
    elements.append(Spacer(1, 3 * mm))

    # --- 8 维健康评分表 ---
    elements.append(Paragraph("八维健康评分", body_style))
    dims = diagnosis_result.get("dimension_scores", {})
    lights = diagnosis_result.get("traffic_lights", {})
    dim_data = [[Paragraph("维度", cell_hdr_style),
                 Paragraph("评分", cell_hdr_style),
                 Paragraph("等级", cell_hdr_style)]]
    level_cmds = []
    for i, (dim, sc) in enumerate(dims.items(), start=1):
        raw_light = lights.get(dim, "")
        ltext, lcolor = _light_level(raw_light)
        dim_data.append([
            Paragraph(_esc(dim), cell_style),
            Paragraph(f"{sc}/10", cell_style),
            Paragraph(ltext, ParagraphStyle(
                f"lvl{i}", parent=cell_style, textColor=colors.white)),
        ])
        level_cmds.append(("BACKGROUND", (2, i), (2, i), lcolor))
    dim_tbl = Table(dim_data, colWidths=[200, 90, 90], repeatRows=1)
    dim_tbl.setStyle(TableStyle(_tbl_style() + level_cmds))
    elements.append(dim_tbl)
    elements.append(Spacer(1, 3 * mm))

    # --- AI 诊断总结 ---
    if ai_summary:
        elements.append(Paragraph("智能诊断总结", body_style))
        elements.append(Paragraph(_line_break(ai_summary), body_style))
        elements.append(Spacer(1, 2 * mm))

    # ===== 三、核心风险点 =====
    elements.append(Paragraph("核心风险点", h1_style))
    _render_bullets(elements, ai_risks, diagnosis_result.get("risks", []),
                    bullet_style, body_style, "未发现明显风险点。")

    # ===== 四、SHAP 归因分析（用户勾选，若有） =====
    if shap_contribs:
        elements.append(Paragraph("SHAP 归因分析（各因子对违约概率的贡献）", h1_style))
        contrib_rows = _shap_rows(shap_contribs)
        shap_data = [[Paragraph("因子", cell_hdr_style),
                      Paragraph("贡献方向", cell_hdr_style),
                      Paragraph("贡献值", cell_hdr_style)]]
        shap_cmds = []
        for i, (factor, val, direction, color) in enumerate(contrib_rows, start=1):
            shap_data.append([
                Paragraph(_esc(factor), cell_style),
                Paragraph(direction, ParagraphStyle(
                    f"sd{i}", parent=cell_style, textColor=color)),
                Paragraph(f"{val:+.3f}", cell_style),
            ])
        shap_tbl = Table(shap_data, colWidths=[220, 120, 90], repeatRows=1)
        shap_tbl.setStyle(TableStyle(_tbl_style() + shap_cmds))
        elements.append(shap_tbl)
        elements.append(Paragraph(
            "说明：正值推高违约概率、负值拉低；模型为合成数据方法论演示，仅供对照参考，"
            "不代表真实风控结论。", note_style))
        elements.append(Spacer(1, 3 * mm))

    # ===== 五、信贷产品匹配结果 =====
    elements.append(Paragraph("信贷产品匹配结果", h1_style))
    if matches:
        prod_cols = ["匹配度", "产品名", "银行", "产品类型", "额度(万元)",
                     "利率(%)", "准入条件", "差距说明"]
        prod_keys = ["匹配度", "产品名", "银行", "产品类型", "额度", "利率",
                     "准入条件", "差距说明"]
        pdata = [[Paragraph(c, cell_hdr_style) for c in prod_cols]]
        for m in matches:
            pdata.append([Paragraph(_esc(m.get(k, "")), cell_style)
                          for k in prod_keys])
        ptbl = Table(pdata, colWidths=[42, 66, 50, 50, 42, 38, 78, 156],
                     repeatRows=1)
        ptbl.setStyle(_tbl_style())
        elements.append(ptbl)
    else:
        elements.append(Paragraph("未找到匹配的信贷产品，建议改善财务状况后再查询。",
                                  body_style))
    if ai_recommendation:
        elements.append(Spacer(1, 2 * mm))
        elements.append(Paragraph("智能产品推荐说明", body_style))
        elements.append(Paragraph(_line_break(ai_recommendation), body_style))
    if product_df is not None and not product_df.empty:
        try:
            elements.append(Spacer(1, 2 * mm))
            elements.append(Paragraph("数据来源：", note_style))
            for _, row in product_df[["产品名", "数据来源", "采集日期"]].drop_duplicates().iterrows():
                elements.append(Paragraph(
                    f"{row['产品名']} - {row['数据来源']}（{row['采集日期']}）",
                    note_style))
        except Exception:
            pass
    elements.append(Spacer(1, 3 * mm))

    # ===== 六、差距分析与行动方案（用户勾选，若有） =====
    if gap_result:
        elements.append(Paragraph("差距分析与行动方案", h1_style))
        summary = gap_result.get("summary")
        if summary:
            elements.append(Paragraph(f"分析总结：{_esc(summary)}", body_style))
            elements.append(Spacer(1, 2 * mm))
        action_plan = gap_result.get("action_plan", [])
        if action_plan:
            acols = ["优先级", "行动", "当前值", "目标值", "难度",
                     "影响产品数", "相关产品", "预计时间", "性价比"]
            adata = [[Paragraph(c, cell_hdr_style) for c in acols]]
            for a in action_plan:
                impact = a.get("impact", 0)
                mode = a.get("impact_mode", "unlock")
                impact_txt = f"{impact}（关联）" if mode != "unlock" else str(impact)
                rel = "、".join(a.get("impact_products", [])[:6]) or "—"
                if len(a.get("impact_products", [])) > 6:
                    rel += " 等"
                adata.append([
                    Paragraph(str(a.get("priority", "")), cell_style),
                    Paragraph(_esc(a.get("action", "")), cell_style),
                    Paragraph(_esc(a.get("current", "")), cell_style),
                    Paragraph(_esc(a.get("target", "")), cell_style),
                    Paragraph(_esc(a.get("difficulty", "")), cell_style),
                    Paragraph(impact_txt, cell_style),
                    Paragraph(_esc(rel), cell_style),
                    Paragraph(_esc(a.get("estimated_time", "")), cell_style),
                    Paragraph(str(a.get("cost_efficiency", "")), cell_style),
                ])
            atbl = Table(adata, colWidths=[30, 100, 60, 60, 40, 55, 78, 72, 42],
                         repeatRows=1)
            atbl.setStyle(_tbl_style())
            elements.append(atbl)
            elements.append(Paragraph(
                "性价比 = 可解锁/关联产品数 ÷ 提升难度分，数值越高越应优先。",
                note_style))
        elements.append(Spacer(1, 3 * mm))

    # ===== 七、诊断指标明细（用户勾选，全部指标） =====
    elements.append(Paragraph("诊断指标明细", h1_style))
    detail_rows = []
    for k, v in metrics.items():
        if k.startswith("__"):
            continue
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            v = "是" if v else "否"
        detail_rows.append((k, v))
    if detail_rows:
        ddata = [[Paragraph("指标", cell_hdr_style),
                  Paragraph("取值", cell_hdr_style)]]
        for k, v in detail_rows:
            ddata.append([Paragraph(_esc(k), cell_style),
                          Paragraph(_esc(v), cell_style)])
        dtbl = Table(ddata, colWidths=[220, 220], repeatRows=1)
        dtbl.setStyle(_tbl_style())
        elements.append(dtbl)
    else:
        elements.append(Paragraph("无可用指标明细。", body_style))
    elements.append(Spacer(1, 3 * mm))

    # ===== 八、改善行动建议 =====
    elements.append(Paragraph("改善行动建议", h1_style))
    _render_bullets(elements, ai_suggestions,
                    diagnosis_result.get("suggestions", []),
                    bullet_style, body_style, "暂无明显需改善项。")

    # ===== 行业政策环境（政策信号模型，融入整体分析） =====
    if policy_result:
        elements.append(Paragraph("行业政策环境", h1_style))
        elements.append(Paragraph(
            f"所属行业：{_esc(policy_result.get('industry', '通用'))}　"
            f"政策景气指数：<b>{policy_result.get('index')} / 100</b>"
            f"（{_esc(policy_result.get('level', ''))}）　趋势：{_esc(policy_result.get('trend', ''))}",
            body_style))
        elements.append(Paragraph(
            f"对经营稳定性的影响：{_esc(policy_result.get('effect', ''))}", body_style))
        recent = policy_result.get("recent") or []
        if recent:
            elements.append(Paragraph("近期政策摘编：", body_style))
            for ev in recent:
                elements.append(Paragraph(f"· {_esc(ev)}", note_style))
        elements.append(Spacer(1, 3 * mm))

    # ===== 附录：政策与产品依据 =====
    if rag_citations:
        elements.append(Paragraph("附录：政策与产品依据", h1_style))
        for c in rag_citations:
            elements.append(Paragraph(f"· {_esc(c)}", note_style))
        elements.append(Paragraph(
            "以上均为条款摘编，非法规全文镜像，请以官方原文为准。"
            + (f"检索截至 {rag_asof}。" if rag_asof else ""), note_style))
        elements.append(Spacer(1, 3 * mm))

    # ===== 免责声明（结尾段落，非页脚/页码） =====
    elements.append(HRFlowable(width="100%", thickness=0.8, color=C_BORDER))
    elements.append(Paragraph(
        "<i>免责声明：本报告由智能诊断工具生成，仅供企业融资决策参考，不构成任何金融建议。"
        "具体融资方案请咨询正规金融机构。</i>", note_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ======================== 辅助函数 ========================

def _tbl_style():
    """统一的表格基础样式（彩色表头 + 网格 + 内边距）。"""
    return [
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F6F8")]),
    ]


def _score_level(score):
    """总体评分 → (等级文本, 颜色)。"""
    if score >= 7:
        return ("健康", C_GREEN)
    elif score >= 4:
        return ("关注", C_AMBER)
    return ("高风险", C_RED)


def _light_level(raw):
    """traffic_lights 字符串（含 emoji）→ (纯中文等级, 颜色)。"""
    if "绿色" in raw:
        return ("绿色", C_GREEN)
    elif "黄色" in raw:
        return ("黄色", C_AMBER)
    elif "红色" in raw:
        return ("红色", C_RED)
    return ("—", C_SUB)


def _shap_rows(contribs):
    """SHAP 贡献 dict → [(因子, 值, 方向文本, 颜色), ...]，过滤 bias/违约概率。"""
    rows = []
    for k, v in contribs.items():
        if k in ("bias", "违约概率"):
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        if val >= 0:
            direction, color = "推高违约", C_RED
        else:
            direction, color = "拉低违约", C_GREEN
        rows.append((k, val, direction, color))
    rows.sort(key=lambda r: -abs(r[1]))
    return rows


def _render_bullets(elements, ai_text, fallback_list, b_style, body_style, empty_msg):
    """优先用 AI 多行文本（每行 '- ' 开头），否则回退规则引擎列表。"""
    if ai_text:
        for line in ai_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line = line[1:].strip() if line.startswith("-") else line
            if line:
                elements.append(Paragraph("• " + _esc(line), b_style))
    elif fallback_list:
        for item in fallback_list:
            elements.append(Paragraph("• " + _esc(item), b_style))
    else:
        elements.append(Paragraph(empty_msg, body_style))
