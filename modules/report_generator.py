"""
模块4：诊断报告PDF生成（支持AI增强内容）
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io, os, datetime

# ---------- 字体注册 ----------
FONT_DIR = "fonts"

def reg(name, filename):
    path = os.path.join(FONT_DIR, filename)
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(name, path))
        return name
    else:
        return 'Helvetica'

CN_FZXBS = reg('FZXBS', 'fzxbs.ttf') if os.path.exists(os.path.join(FONT_DIR, 'fzxbs.ttf')) else 'Helvetica'
CN_SIMHEI = reg('SimHei', 'simhei.ttf') if os.path.exists(os.path.join(FONT_DIR, 'simhei.ttf')) else 'Helvetica'
CN_KAITI = reg('KaiTi', 'kaiti.ttf') if os.path.exists(os.path.join(FONT_DIR, 'kaiti.ttf')) else 'Helvetica'
CN_FANGSONG = reg('FangSong', 'fangsong.ttf') if os.path.exists(os.path.join(FONT_DIR, 'fangsong.ttf')) else 'Helvetica'

# 如果字体不存在，使用备用（例如 Streamlit Cloud 上可能没有 fonts 文件夹）
# 我们让系统更稳健：如果注册失败，自动使用 Helvetica，但中文会乱码，所以尽量确保 fonts 上传
def generate_pdf(metrics, diagnosis_result, matches, product_df,
                 ai_summary=None, ai_risks=None, ai_suggestions=None,
                 ai_recommendation=None):
    """
    参数新增：
        ai_summary: AI 生成的总体评价文本
        ai_risks: AI 生成的风险点文本（多行，每行以"- "开头）
        ai_suggestions: AI 生成的改善建议文本（多行，每行以"- "开头）
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    # 样式定义（使用注册的字体，若失败则回退默认字体）
    title_style = ParagraphStyle('Title', fontName=CN_FZXBS, fontSize=20, leading=28, spaceAfter=10, alignment=1)
    h1_style = ParagraphStyle('H1', fontName=CN_SIMHEI, fontSize=14, leading=18, spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', fontName=CN_FANGSONG, fontSize=10, leading=16, spaceAfter=4)
    note_style = ParagraphStyle('Note', fontName=CN_KAITI, fontSize=7, leading=10, textColor=colors.gray)
    emphasis_style = ParagraphStyle('Emphasis', fontName=CN_SIMHEI, fontSize=10, leading=16, textColor=colors.red)

    elements = []

    # 标题
    elements.append(Paragraph("小微企业融资诊断报告", title_style))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", note_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    elements.append(Spacer(1, 5*mm))

    # 一、企业概况
    elements.append(Paragraph("一、企业概况", h1_style))
    asset = metrics.get("总资产", "未填写")
    revenue = metrics.get("营业收入", "未填写")
    profit = metrics.get("净利润", "未填写")
    years = metrics.get("经营年限", "未填写")
    industry = metrics.get("行业", "未填写")
    overview_text = (
        f"总资产：{asset} 万元<br/>"
        f"营业收入：{revenue} 万元<br/>"
        f"净利润：{profit} 万元<br/>"
        f"经营年限：{years} 年<br/>"
        f"所属行业：{industry}"
    )
    elements.append(Paragraph(overview_text, body_style))
    elements.append(Spacer(1, 3*mm))

    # 二、金融健康评分与AI总结
    elements.append(Paragraph("二、金融健康评分", h1_style))
    overall = diagnosis_result['overall_score']
    elements.append(Paragraph(f"总体健康评分：<b>{overall} / 10</b>", body_style))
    dims = diagnosis_result['dimension_scores']
    lights = diagnosis_result['traffic_lights']
    for dim, score in dims.items():
        elements.append(Paragraph(f"• {dim}：{score}/10 {lights[dim]}", body_style))
    elements.append(Spacer(1, 3*mm))

    # AI 诊断总结（若有）
    if ai_summary:
        elements.append(Paragraph("AI 诊断总结", h1_style))
        elements.append(Paragraph(ai_summary, body_style))
        elements.append(Spacer(1, 3*mm))

    # 三、核心风险点（AI优先）
    elements.append(Paragraph("三、核心风险点", h1_style))
    if ai_risks:
        # ai_risks 是多行文本，每行以 "- " 开头，直接展示
        for line in ai_risks.strip().split('\n'):
            line = line.strip()
            if line.startswith('-'):
                elements.append(Paragraph(line, emphasis_style))
    else:
        # 回退到规则引擎
        risks = diagnosis_result.get('risks', [])
        if risks:
            for risk in risks:
                elements.append(Paragraph(f"• {risk}", emphasis_style))
        else:
            elements.append(Paragraph("未发现明显风险点。", body_style))
    elements.append(Spacer(1, 3*mm))

    # 四、信贷产品匹配结果
    elements.append(Paragraph("四、信贷产品匹配结果", h1_style))
    if matches:
        table_data = [["匹配度", "产品名", "银行", "额度(万)", "利率(%)", "差距说明"]]
        for m in matches:
            table_data.append([
                m['匹配度'], m['产品名'], m['银行'], m['额度'], m['利率'], m['差距说明']
            ])
        col_widths = [55, 70, 55, 55, 55, 120]
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4F81BD")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,-1), CN_SIMHEI),
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("未找到匹配的信贷产品。", body_style))

    if product_df is not None and not product_df.empty:
        elements.append(Spacer(1, 3*mm))
        elements.append(Paragraph("数据来源：", note_style))
        for _, row in product_df[['产品名', '数据来源', '采集日期']].drop_duplicates().iterrows():
            elements.append(Paragraph(
                f"{row['产品名']} - {row['数据来源']}（{row['采集日期']}）", note_style
            ))

    elements.append(Spacer(1, 5*mm))


    # 新增：AI产品推荐说明（如果有）
    if ai_recommendation:
        elements.append(Paragraph("五、AI 产品推荐说明", h1_style))
        elements.append(Paragraph(ai_recommendation, body_style))
        elements.append(Spacer(1, 3*mm))
        # 后面的章节序号需要调整
        section_num = "六"  # 原来的"五"变成"六"
    else:
        section_num = "五"

    # 改善行动建议（原来的第五节）
    elements.append(Paragraph(f"{section_num}、改善行动建议", h1_style))

    if ai_suggestions:
        for line in ai_suggestions.strip().split('\n'):
            line = line.strip()
            if line.startswith('-'):
                elements.append(Paragraph(line, body_style))
    else:
        suggestions = diagnosis_result.get('suggestions', [])
        for sug in suggestions:
            elements.append(Paragraph(f"• {sug}", body_style))
    elements.append(Spacer(1, 8*mm))

    # 免责声明
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Paragraph(
        "<i>免责声明：本报告由AI工具生成，仅供企业决策参考，不构成任何金融建议。具体融资方案请咨询正规金融机构。</i>",
        note_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer