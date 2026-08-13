"""
utils/ui_style.py — 全局视觉系统（v2.0 新增，纯展示层，不涉及任何算法逻辑）

设计令牌：
  主色   #1F4E79（金融深青蓝：标题/按钮/强调）
  语义色 绿 #16A34A / 黄 #D97706 / 红 #DC2626（仅用于红黄绿灯与双轨结论徽章）
  中性   页面底 #F7F8FA / 卡片 #FFFFFF / 边框 #E5E7EB / 正文 #374151 / 次要 #6B7280
配合 .streamlit/config.toml 使用；卡片容器用 st.container(border=True)。
"""
import streamlit as st

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1180px;}
h1, h2, h3 {color: #1F4E79;}
[data-testid="stMetricValue"] {color: #1F4E79;}
.fd-badge {display:inline-block; padding:2px 10px; border-radius:999px;
           font-size:12px; font-weight:500; line-height:1.6;}
.fd-green {background:#E8F6EE; color:#16A34A;}
.fd-amber {background:#FBF3E3; color:#D97706;}
.fd-red   {background:#FBEAEA; color:#DC2626;}
.fd-blue  {background:#E8F0F8; color:#1F4E79;}
.fd-hero  {text-align:center; padding:36px 0 20px;}
.fd-hero .t {font-size:26px; font-weight:500; color:#1F4E79; margin-bottom:8px;}
.fd-hero .s {color:#6B7280; font-size:14px; margin-bottom:18px;}
.fd-hero .steps {display:flex; justify-content:center; gap:32px;
                 color:#6B7280; font-size:13px;}
.fd-hero .steps b {color:#1F4E79; font-weight:500;}
.fd-empty {text-align:center; padding:48px 0; color:#6B7280;}
.fd-empty .t {font-size:15px; color:#374151; margin-bottom:6px;}
.fd-cite {font-size:12px; color:#6B7280;}
</style>
"""


def inject_css():
    """在 set_page_config 之后调用一次，注入全局样式。"""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text, level="blue"):
    """语义徽章 HTML。level ∈ green / amber / red / blue。"""
    return f'<span class="fd-badge fd-{level}">{text}</span>'


def score_level(score):
    """总体评分 → (徽章色, 文案)。阈值与规则卡一致：≥7 绿 / 4~7 黄 / <4 红。"""
    if score >= 7:
        return "green", "整体健康"
    if score >= 4:
        return "amber", "中等水平"
    return "red", "高风险"


def dim_level(score):
    """维度评分 → 徽章色。"""
    if score >= 7:
        return "green"
    if score >= 4:
        return "amber"
    return "red"


def hero():
    """Tab1 空状态：价值主张 + 三步引导。"""
    st.markdown(
        '<div class="fd-hero">'
        '<div class="t">上传一份财务数据，三步拿到融资诊断</div>'
        '<div class="s">规则引擎 × 违约 ML 双轨对照 · 8 维健康评分 · '
        '信贷产品匹配与差距分析 · 政策可溯源问答</div>'
        '<div class="steps"><span><b>1</b>　上传或手动录入数据</span>'
        '<span><b>2</b>　确认指标并开始诊断</span>'
        '<span><b>3</b>　查看报告与产品匹配</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def empty_state(title, hint):
    """Tab2/3 空状态：引导回数据录入页。"""
    st.markdown(
        f'<div class="fd-empty"><div class="t">{title}</div>'
        f'<div>{hint}</div></div>',
        unsafe_allow_html=True,
    )
