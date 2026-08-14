"""
utils/ui_style.py — 全局视觉系统（v3.0 对话式重做，纯展示层）

设计令牌（现代 fintech / AI 助手风）：
  主色    #2F54EB（geekblue：信任 + 科技感）
  强调弱  #EEF1FE
  背景    #F5F6F8（柔和中性灰）
  表面    #FFFFFF（卡片 / 助手气泡）
  文字    #1F2733（主） / #5B6573（次）
  边框    #E6E8EC
  语义    成功 #2BA471 / 警示 #D98B1F / 危险 #E5484D
字体：系统字体栈（覆盖 HarmonyOS / Windows / macOS / Linux 中文），不依赖外部字体，避免国内 / 离线环境中文渲染异常（乱码 / 豆腐块）。
配合 .streamlit/config.toml 使用；卡片用 st.container(border=True) / 对话气泡由 CSS 接管。
"""
import streamlit as st

_CSS = """
/* 不依赖外部字体（Google Fonts 在国内 / 离线环境常加载失败，会导致中文回退异常）；统一使用系统字体栈 */

:root{
  --fd-bg:#F5F6F8; --fd-surface:#FFFFFF; --fd-primary:#2F54EB;
  --fd-primary-weak:#EEF1FE; --fd-text:#1F2733; --fd-text-2:#5B6573;
  --fd-border:#E6E8EC; --fd-success:#2BA471; --fd-warning:#D98B1F; --fd-danger:#E5484D;
}

html, body, [class*="st-"]{
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Microsoft YaHei UI', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Source Han Sans CN', 'WenQuanYi Micro Hei', 'HarmonyOS Sans SC', sans-serif !important;
}
* { -webkit-font-smoothing:antialiased; }

.block-container{ max-width:880px; padding-top:1.1rem; padding-bottom:3.2rem; }
.main .block-container{ background:var(--fd-bg); }

/* 顶部品牌条 */
.fd-topbar{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.fd-topbar .dot{ width:30px; height:30px; border-radius:9px;
  background:linear-gradient(135deg,var(--fd-primary),#5B7CF0);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:15px; }
.fd-topbar .name{ font-size:17px; font-weight:600; color:var(--fd-text); letter-spacing:.3px; }
.fd-topbar .sub{ font-size:12px; color:var(--fd-text-2); margin-top:1px; }
.fd-divider{ height:1px; background:var(--fd-border); margin:8px 0 14px; }

/* 语义徽章 */
.fd-badge{ display:inline-block; padding:2px 11px; border-radius:999px;
  font-size:12px; font-weight:600; line-height:1.7; white-space:nowrap; }
.fd-green{ background:#E7F6EF; color:var(--fd-success); }
.fd-amber{ background:#FBF1E0; color:var(--fd-warning); }
.fd-red{ background:#FCEAEA; color:var(--fd-danger); }
.fd-blue{ background:var(--fd-primary-weak); color:var(--fd-primary); }

/* 对话气泡 */
[data-testid="chatAvatarIcon"]{ display:none !important; }
.stChatMessage{ padding:0.3rem 0; align-items:flex-end; }
[data-testid="stChatMessageContent"]{
  border-radius:16px; padding:10px 14px; line-height:1.62; font-size:15px;
  box-shadow:0 1px 2px rgba(20,30,60,.04); max-width:100%;
}
.stChatMessage--assistant [data-testid="stChatMessageContent"]{
  background:var(--fd-surface); color:var(--fd-text); border:1px solid var(--fd-border); }
.stChatMessage--user [data-testid="stChatMessageContent"]{
  background:var(--fd-primary); color:#fff; border:none; }
.stChatMessage--user [data-testid="stChatMessageContent"] *{ color:#fff; }
.stChatMessage--user [data-testid="stChatMessageContent"] a{ color:#DCE4FF; }

/* 输入框（对话式收件栏） */
[data-testid="stChatInput"]{
  background:var(--fd-surface); border:1px solid var(--fd-border);
  border-radius:18px; padding:6px 12px; box-shadow:0 4px 18px rgba(20,30,60,.07); }
[data-testid="stChatInput"] textarea{
  border-radius:12px !important; font-size:15px !important; }
[data-testid="stChatInput"] textarea::placeholder{ color:var(--fd-text-2); }

/* 按钮 / 表单 / 指标 */
.stButton>button{ border-radius:10px; font-weight:500; }
.stButton>button[kind="primary"]{ background:var(--fd-primary) !important;
  border-color:var(--fd-primary) !important; }
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div,
.stTextArea>div>div>textarea{
  border-radius:10px !important; border-color:var(--fd-border) !important; }
[data-testid="stMetricValue"]{ color:var(--fd-primary); font-weight:600; }
[data-testid="stExpander"]{ border-radius:12px !important; border-color:var(--fd-border) !important; }
.stDataFrame{ border-radius:12px; }

/* 侧边栏 */
[data-testid="stSidebar"]{ background:var(--fd-surface); border-right:1px solid var(--fd-border); }
[data-testid="stSidebar"] .fd-brand{ display:flex; align-items:center; gap:9px; margin-bottom:4px; }
[data-testid="stSidebar"] .fd-brand .dot{ width:26px; height:26px; border-radius:8px;
  background:linear-gradient(135deg,var(--fd-primary),#5B7CF0); color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center; font-size:13px; }
[data-testid="stSidebar"] .fd-brand .t{ font-size:15px; font-weight:600; color:var(--fd-text); }

/* 隐藏默认页脚，保持整洁 */
[data-testid="stFooter"]{ display:none !important; }
"""


def inject_css():
    """在 set_page_config 之后调用一次，注入全局样式。"""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


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
