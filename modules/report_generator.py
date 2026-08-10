"""
模块4：诊断报告PDF生成（多字体层级，无需fonttools）
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

def reg(presumed_ps_name, filename):
    """尝试用预设的PostScript名称注册字体，如果失败则抛出"""
    path = os.path.join(FONT_DIR, filename)
    if not os.path.exists(path):
        print(f"警告：字体文件未找到 -> {path}")
        return 'Helvetica'
    try:
        pdfmetrics.registerFont(TTFont(presumed_ps_name, path))
        print(f"成功注册字体: {presumed_ps_name}")
        return presumed_ps_name
    except Exception as e:
        print(f"注册字体失败 ({filename})，错误信息: {e}")
        print("请检查字体内部PostScript名称，可能需要手动指定。")
        # 临时回退，但会乱码
        return 'Helvetica'

# 预设名称（根据常见情况）
CN_FZXBS = reg('FZXiaoBiaoSong-B05S', 'fzxbs.ttf')   # 方正小标宋
CN_SIMHEI = reg('SimHei', 'simhei.ttf')               # 黑体
CN_KAITI = reg('KaiTi', 'kaiti.ttf')                  # 楷体
CN_FANGSONG = reg('FangSong', 'fangsong.ttf')         # 仿宋
CN_HANYI = reg('HYDaSongJ', 'hanyi_ds.ttf')           # 汉仪大宋简（名称可能不同）
EN_FONT = 'Times-Roman'

def generate_pdf(metrics, diagnosis_result, matches, product_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    # ---------- 样式 ----------
    title_style = ParagraphStyle('T', fontName=CN_FZXBS, fontSize=20, leading=28, spaceAfter=10, alignment=1)
    h1_style = ParagraphStyle('H1', fontName=CN_SIMHEI, fontSize=14, leading=18, spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('B', fontName=CN_FANGSONG, fontSize=10, leading=16, spaceAfter=4)
    note_style = ParagraphStyle('N', fontName=CN_KAITI, fontSize=7, leading=10, textColor=colors.gray, spaceAfter=2)
    emphasis_style = ParagraphStyle('E', fontName=CN_SIMHEI, fontSize=10, leading=16, textColor=colors.red, spaceAfter=2)

    elements = []

    # 标题
    elements.append(Paragraph("小微企业融资诊断报告", title_style))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", note_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    elements.append(Spacer(1, 5*mm))

    # 企业概况
    elements.append(Paragraph("一、企业概况", h1_style))
    asset = metrics.get("总资产", "未填写")
    revenue = metrics.get("营业收入", "未填写")
    profit = metrics.get("净利润", "未填写")
    years = metrics.get("经营年限", "未填写")
    industry = metrics.get("行业", "未填写")
    elements.append(Paragraph(
        f"总资产：{asset} 万元<br/>营业收入：{revenue} 万元<br/>净利润：{profit} 万元<br/>经营年限：{years} 年<br/>所属行业：{industry}",
        body_style
    ))
    elements.append(Spacer(1, 3*mm))

    # 健康评分
    elements.append(Paragraph("二、金融健康评分", h1_style))
    overall = diagnosis_result['overall_score']
    elements.append(Paragraph(f"总体健康评分：<b>{overall} / 10</b>", body_style))
    for dim, score in diagnosis_result['dimension_scores'].items():
        elements.append(Paragraph(f"• {dim}：{score}/10 {diagnosis_result['traffic_lights'][dim]}", body_style))
    elements.append(Spacer(1, 3*mm))

    # 风险点
    elements.append(Paragraph("三、核心风险点", h1_style))
    risks = diagnosis_result.get('risks', [])
    if risks:
        for r in risks:
            elements.append(Paragraph(f"• {r}", emphasis_style))
    else:
        elements.append(Paragraph("未发现明显风险点。", body_style))
    elements.append(Spacer(1, 3*mm))

    # 产品匹配
    elements.append(Paragraph("四、信贷产品匹配结果", h1_style))
    if matches:
        table_data = [["匹配度", "产品名", "银行", "额度(万)", "利率(%)", "差距说明"]]
        for m in matches:
            table_data.append([m['匹配度'], m['产品名'], m['银行'], m['额度'], m['利率'], m['差距说明']])
        table = Table(table_data, colWidths=[55,70,55,55,55,120], repeatRows=1)
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
            elements.append(Paragraph(f"{row['产品名']} - {row['数据来源']}（{row['采集日期']}）", note_style))
    elements.append(Spacer(1, 5*mm))

    # 改善建议
    elements.append(Paragraph("五、改善行动建议", h1_style))
    for sug in diagnosis_result.get('suggestions', []):
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