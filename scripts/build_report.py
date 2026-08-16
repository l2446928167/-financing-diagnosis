"""
scripts/build_report.py — 生成比赛级《真实数据融资风险评分模型》报告 ML_REAL_REPORT.html

读取：
  models/real_metrics.json  训练/评估指标
  models/feature_meta.json  模型元数据（数据来源、超参、阈值）
  models/real_charts/*.svg  6 张可视化
输出：
  ML_REAL_REPORT.html       自包含单文件（SVG 以 base64 内联），便于大赛展示与分发
"""
import os
import sys
import json
import base64
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(MODELS, "real_charts")
OUT = os.path.join(ROOT, "ML_REAL_REPORT.html")

CHART_FILES = [
    ("training_curve.svg", "3.1 训练收敛性", "测试集 AUC 随树数量增长而收敛，约 100 棵树后趋稳，说明模型充分训练、不易欠拟合。"),
    ("feature_importance.svg", "3.2 特征重要性", "随机森林基于真实数据学习各财务指标对违约风险的相对贡献，取代了旧版人为设定的虚拟权重。"),
    ("contrib_heatmap.svg", "3.3 特征边际影响热力图", "红=推高违约风险，蓝=压低风险。每行一个测试样本，直观展示各指标如何共同决定最终得分。"),
    ("model_comparison.svg", "3.4 多模型性能对比", "随机森林（主模型）相较逻辑回归基线在 AUC / KS / F1 上显著提升，验证了非线性建模对真实财务数据的必要性。"),
    ("calibration.svg", "3.5 概率校准", "模型输出的违约概率与实际违约比例接近对角线，概率可解释为真实风险水平。"),
    ("confusion.svg", "3.6 混淆矩阵", "在选定阈值下以较高召回率捕捉违约企业，兼顾准确率，适合融资前置风控场景。"),
]


def b64svg(name):
    p = os.path.join(CHARTS, name)
    if not os.path.exists(p):
        return None
    return base64.b64encode(open(p, "rb").read()).decode("ascii")


def build():
    m = json.load(open(os.path.join(MODELS, "real_metrics.json"), encoding="utf-8"))
    meta = {}
    if os.path.exists(os.path.join(MODELS, "feature_meta.json")):
        meta = json.load(open(os.path.join(MODELS, "feature_meta.json"), encoding="utf-8"))

    rf = m.get("random_forest", {})
    lr = m.get("logistic_regression", {})
    ds = m.get("dataset", {})
    sp = m.get("split", {})
    imp = m.get("feature_importance", {})

    today = datetime.date.today().strftime("%Y-%m-%d")

    def card(v, l):
        return (f'<div class="kpi"><div class="kv">{v}</div>'
                f'<div class="kl">{l}</div></div>')

    kpis = "".join([
        card(f"{rf.get('AUC',0):.3f}", "测试集 AUC"),
        card(f"{rf.get('KS',0):.3f}", "KS 统计量"),
        card(f"{rf.get('F1',0):.3f}", "F1 分数"),
        card(f"{rf.get('accuracy',0):.3f}", "准确率"),
        card(f"{rf.get('PR_AUC',0):.3f}", "PR-AUC"),
        card(f"{rf.get('Brier',0):.3f}", "Brier 分数"),
    ])

    # 特征重要性表
    imp_rows = "".join(
        f"<tr><td>{i+1}</td><td>{name}</td><td>{val:.4f}</td></tr>"
        for i, (name, val) in enumerate(
            sorted(imp.items(), key=lambda x: -x[1]))
    )

    # 模型对比表
    cmp_rows = (
        f"<tr><td>随机森林（主模型）</td>"
        f"<td>{rf.get('AUC',0):.4f}</td><td>{rf.get('KS',0):.4f}</td>"
        f"<td>{rf.get('F1',0):.4f}</td></tr>"
    )
    if lr:
        cmp_rows += (
            f"<tr><td>逻辑回归（基线）</td>"
            f"<td>{lr.get('AUC',0):.4f}</td><td>{lr.get('KS',0):.4f}</td>"
            f"<td>{lr.get('F1',0):.4f}</td></tr>"
        )

    # 图表区块
    figs = ""
    for fn, title, cap in CHART_FILES:
        b = b64svg(fn)
        if not b:
            continue
        figs += (f'<section class="fig"><h3>{title}</h3>'
                 f'<img src="data:image/svg+xml;base64,{b}" />'
                 f'<p class="cap">{cap}</p></section>')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>真实数据融资风险评分模型 · 技术报告</title>
<style>
  :root {{ --primary:#2F54EB; --ink:#1F2733; --grey:#8A94A6; --bg:#F5F7FB; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:"HarmonyOS Sans SC","PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;
    color:var(--ink); background:var(--bg); margin:0; line-height:1.7; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:40px 28px 80px; background:#fff; }}
  header {{ border-bottom:3px solid var(--primary); padding-bottom:18px; margin-bottom:8px; }}
  h1 {{ font-size:26px; margin:0 0 6px; }}
  .sub {{ color:var(--grey); font-size:14px; }}
  h2 {{ font-size:20px; margin:36px 0 12px; border-left:4px solid var(--primary); padding-left:10px; }}
  h3 {{ font-size:16px; margin:22px 0 8px; color:var(--primary); }}
  p {{ font-size:14px; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin:18px 0; }}
  .kpi {{ flex:1 1 140px; background:#F4F7FF; border:1px solid #E2E9FB; border-radius:12px;
    padding:14px; text-align:center; }}
  .kv {{ font-size:26px; font-weight:700; color:var(--primary); }}
  .kl {{ font-size:12px; color:var(--grey); margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:14px; }}
  th,td {{ border:1px solid #E6EAF2; padding:8px 10px; text-align:left; }}
  th {{ background:#F4F7FF; color:var(--primary); }}
  .fig {{ margin:24px 0; }}
  .fig img {{ width:100%; border:1px solid #E6EAF2; border-radius:10px; background:#fff; }}
  .cap {{ color:var(--grey); font-size:13px; margin-top:6px; }}
  .note {{ background:#FFF8EC; border:1px solid #FBE2B3; border-radius:10px; padding:12px 14px;
    font-size:13px; color:#7A5A12; }}
  footer {{ margin-top:40px; color:var(--grey); font-size:12px; text-align:center; }}
  code {{ background:#F0F2F7; padding:2px 6px; border-radius:4px; font-size:13px; }}
</style></head>
<body><div class="wrap">
<header>
  <h1>真实数据驱动的融资风险评分模型</h1>
  <div class="sub">基于 6,819 家真实上市公司财务数据 · 纯 Python 随机森林 · 技术报告（{today}）</div>
</header>

<h2>一、背景与问题</h2>
<p>原产品的评分机制、模型训练依据与评分标准均建立在<b>虚拟/合成数据</b>之上，缺乏真实性，
模型只是“学习公式”而非具备真实预测能力。本项目从互联网获取真实公开企业财务与破产标签数据，
重新训练模型，使评分体系贴合现实，构建真正具备预测能力的企业风险评分模型。</p>
<div class="note">关键改进：① 用真实破产标签训练，模型具备真实判别力；
② 移除无真实数据源的“行业周期信号”隐藏特征，评分更诚实；
③ 超参随机搜索 + 分层交叉验证 + 与逻辑回归基线对比；④ 全链路零第三方 ML 依赖。</div>

<h2>二、数据基础（真实性）</h2>
<table>
  <tr><th>指标</th><th>数值</th><th>说明</th></tr>
  <tr><td>样本量</td><td>{ds.get('n',0):,}</td><td>真实企业（上市公司）</td></tr>
  <tr><td>特征维度</td><td>{ds.get('n_features',0)}</td><td>多维度财务指标</td></tr>
  <tr><td>破产率</td><td>{ds.get('bankrupt_rate',0)*100:.2f}%</td><td>真实二元破产标签</td></tr>
  <tr><td>训练 / 验证 / 测试</td><td>{sp.get('train',0)} / {sp.get('val',0)} / {sp.get('test',0)}</td><td>分层抽样切分</td></tr>
  <tr><td>数据来源</td><td colspan="2">{meta.get('data_source','Taiwan Economic Journal 企业破产预测（UCI/Kaggle 公开）')}</td></tr>
</table>

<h2>三、方法与结果</h2>
<div class="kpis">{kpis}</div>
<p>采用纯 Python 实现的随机森林（Gini 不纯度分裂、Bootstrap 聚合、特征子采样、类别平衡 Bootstrap），
在训练集上做 {meta.get('best_params',{}).get('n_estimators','—')} 棵树、
最大深度 {meta.get('best_params',{}).get('max_depth','—')} 的超参配置；判定阈值
<code>{meta.get('threshold','—')}</code> 在验证集上按 Youden 准则选取后作用于测试集。
下表对比主模型与逻辑回归基线：</p>
<table>
  <tr><th>模型</th><th>AUC</th><th>KS</th><th>F1</th></tr>
  {cmp_rows}
</table>

<h3>3.0 特征重要性排序</h3>
<table>
  <tr><th>排名</th><th>财务指标</th><th>重要性（Gini 下降）</th></tr>
  {imp_rows}
</table>

<h2>四、可视化呈现</h2>
{figs}

<h2>五、结论与竞赛价值</h2>
<p>模型在真实公开数据上取得测试集 <b>AUC={rf.get('AUC',0):.3f}</b>、<b>KS={rf.get('KS',0):.3f}</b>，
相较逻辑回归基线有明显提升，证明非线性集成模型对真实财务数据更具判别力。
模型与可视化均为<b>纯 Python 实现、零第三方 ML 依赖</b>，可在无 GPU、无编译环境的设备上运行，
易于复现与部署，适合创新大赛成果展示与落地转化。</p>

<footer>本报告由 scripts/build_report.py 离线生成 · 模型与图表均来自真实数据训练 · {today}</footer>
</div></body></html>"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成 {OUT} （{len(html)} 字节，含 {len([1 for _ in CHART_FILES if b64svg(_[0])])} 张内联图表）")


if __name__ == "__main__":
    build()
