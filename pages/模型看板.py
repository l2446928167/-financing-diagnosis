"""
pages/模型看板.py — 融资风险评分模型 · 训练与评估看板（创新大赛成果展示页）

与 app.py 主程序解耦：本页只读 models/real_metrics.json 与 models/real_charts/*.svg，
呈现训练/评估结果。SVG 通过 base64 data URI 内联渲染，无需 matplotlib/cairosvg。

数据源：Taiwan Economic Journal 企业破产预测（UCI/Kaggle 公开，6819 家真实上市公司）。
"""
import os
import sys
import json
import base64

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHART_DIR = os.path.join(ROOT, "models", "real_charts")
METRICS_PATH = os.path.join(ROOT, "models", "real_metrics.json")
META_PATH = os.path.join(ROOT, "models", "feature_meta.json")

st.set_page_config(page_title="模型看板 · 融资诊断", layout="wide")

_CSS = """
<style>
.big-metric { text-align:center; padding:14px 8px; border-radius:12px;
  background:#F4F7FF; border:1px solid #E2E9FB; }
.big-metric .v { font-size:30px; font-weight:700; color:#2F54EB; line-height:1.1; }
.big-metric .l { font-size:12px; color:#5A6478; margin-top:4px; }
.sec-title { font-size:18px; font-weight:700; margin:26px 0 10px;
  border-left:4px solid #2F54EB; padding-left:10px; color:#1F2733; }
.card-note { font-size:12px; color:#8A94A6; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def _show_svg(name, caption=None, width="100%"):
    p = os.path.join(CHART_DIR, name)
    if not os.path.exists(p):
        st.warning(f"图表缺失：{name}（请先运行 scripts/train_real.py 训练）")
        return
    b64 = base64.b64encode(open(p, "rb").read()).decode("ascii")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" style="width:{width};" />',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f'<div class="card-note">{caption}</div>', unsafe_allow_html=True)


def _metric_card(col, label, value, sub=""):
    col.markdown(
        f'<div class="big-metric"><div class="v">{value}</div>'
        f'<div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )
    if sub:
        col.markdown(f'<div class="card-note" style="text-align:center">{sub}</div>',
                     unsafe_allow_html=True)


def main():
    st.title("📊 融资风险评分模型 · 训练与评估看板")
    st.caption("真实公开数据训练 · 纯 Python 随机森林 · 创新大赛成果展示")

    if not os.path.exists(METRICS_PATH):
        st.error("尚未找到模型评估指标（models/real_metrics.json）。\n"
                 "请先运行 `python scripts/train_real.py` 完成训练与评估。")
        return

    with open(METRICS_PATH, encoding="utf-8") as f:
        m = json.load(f)
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

    rf = m.get("random_forest", {})
    ds = m.get("dataset", {})
    sp = m.get("split", {})

    # ---------------- 概览 ----------------
    st.markdown('<div class="sec-title">一、关键指标（测试集）</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    _metric_card(c1, "AUC", f"{rf.get('AUC',0):.3f}", "判别力")
    _metric_card(c2, "KS", f"{rf.get('KS',0):.3f}", "排序区分度")
    _metric_card(c3, "F1", f"{rf.get('F1',0):.3f}", "违约识别均衡")
    _metric_card(c4, "准确率", f"{rf.get('accuracy',0):.3f}", "")
    _metric_card(c5, "PR-AUC", f"{rf.get('PR_AUC',0):.3f}", "不平衡场景")
    _metric_card(c6, "Brier", f"{rf.get('Brier',0):.3f}", "概率校准")

    st.markdown('<div class="sec-title">二、数据基础（真实性）</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "样本量", f"{ds.get('n',0):,}", "真实企业")
    _metric_card(c2, "特征维度", f"{ds.get('n_features',0)}", "财务指标")
    _metric_card(c3, "破产率", f"{ds.get('bankrupt_rate',0)*100:.1f}%", "真实标签")
    _metric_card(c4, "训练/验证/测试", f"{sp.get('train',0)}/{sp.get('val',0)}/{sp.get('test',0)}",
                 "分层抽样")

    src = meta.get("data_source", "真实公开数据集（Taiwan Economic Journal 企业破产预测）")
    st.info(f"**数据来源**：{src}\n\n"
             f"**最佳超参**：{meta.get('best_params')} ｜ **判定阈值**：{meta.get('threshold')} ｜ "
             f"**模型版本**：{meta.get('version','2.0-real')}\n\n"
             f"{meta.get('note','')}")

    # ---------------- 图表 ----------------
    st.markdown('<div class="sec-title">三、训练收敛性</div>', unsafe_allow_html=True)
    _show_svg("training_curve.svg",
              "测试集 AUC 随树数量增长而收敛，约 100 棵树后趋稳，说明模型充分训练且不易欠拟合。")

    st.markdown('<div class="sec-title">四、特征重要性（Gini 不纯度下降）</div>',
                unsafe_allow_html=True)
    _show_svg("feature_importance.svg",
              "随机森林基于真实数据学习到各财务指标对违约风险的相对贡献，"
              "不再依赖人为设定的虚拟权重。")

    st.markdown('<div class="sec-title">五、特征对最终评分的边际影响（热力图）</div>',
                unsafe_allow_html=True)
    _show_svg("contrib_heatmap.svg",
              "红=推高违约风险，蓝=压低风险。每行代表一个测试样本，直观展示各指标如何共同决定最终得分。")

    st.markdown('<div class="sec-title">六、多模型性能对比</div>', unsafe_allow_html=True)
    _show_svg("model_comparison.svg",
              "随机森林（主模型）相较逻辑回归基线在 AUC / KS / F1 上均有显著提升，"
              "验证了非线性建模对真实财务数据的必要性。")

    st.markdown('<div class="sec-title">七、概率校准</div>', unsafe_allow_html=True)
    _show_svg("calibration.svg",
              "模型输出的违约概率与实际违约比例接近对角线，表明概率可解释为真实风险水平。")

    st.markdown('<div class="sec-title">八、混淆矩阵（测试集）</div>', unsafe_allow_html=True)
    _show_svg("confusion.svg",
              "在选定阈值下，模型以较高召回率捕捉违约企业，兼顾准确率，适合融资前置风控场景。")

    st.markdown("---")
    st.caption("本页所有指标与图表均由 scripts/train_real.py 在真实公开数据集上离线生成，"
               "运行时零第三方 ML 依赖。")


if __name__ == "__main__":
    main()
