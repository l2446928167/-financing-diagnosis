"""
modules/ml_model.py — 企业融资/违约评分模型运行时接口（v2，纯 Python 随机森林）

变更说明（相对旧版 XGBoost）：
- 模型文件改为 models/rf_model.json（纯 Python RandomForest 序列化，零第三方依赖）。
- 特征定义统一来自 modules.features.FEATURE_NAMES（单一事实源），杜绝顺序错位。
- 移除无真实数据源的“行业周期信号”隐藏特征——评分更诚实。
- 推理/归因均为纯 Python，应用运行时无需安装 numpy/sklearn/xgboost。

部署：训练在离线完成（scripts/train_real.py，使用真实公开数据集），运行期只 load。
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "rf_model.json")
META_PATH = os.path.join(ROOT, "models", "feature_meta.json")

from modules.features import FEATURE_NAMES, statement_to_features as _stmt_feats

_CACHE = {}


def load_model():
    """
    懒加载 + 兜底：首次调用才加载；文件缺失或异常时返回 (None, None)，
    调用方应降级为纯规则卡诊断。
    """
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["meta"]
    try:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            _CACHE.update(model=None, meta=None)
            return None, None
        from modules.rf_model import RandomForest
        rf = RandomForest.load(MODEL_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        _CACHE.update(model=rf, meta=meta)
        return rf, meta
    except Exception:
        _CACHE.update(model=None, meta=None)
        return None, None


def _clip_from_meta(meta):
    """可选：用训练集特征均值做轻量边界，使单条记录与训练分布对齐。"""
    return None  # 当前模型对轻微越界稳健，暂不裁剪；如需可在此加 clip


def predict_default_proba(statement):
    """返回违约概率；模型不可用时返回 None（调用方降级为纯规则卡）"""
    rf, _meta = load_model()
    if rf is None:
        return None
    feats = _stmt_feats(statement, clip=_clip_from_meta(_meta))
    return float(rf.predict_proba([feats])[0])


def explain_statement(statement):
    """
    单样本边际贡献（RF 扰动近似）。
    返回 {特征名: 贡献值, "bias": 0.0, "违约概率": p}。
    模型不可用返回 None。
    """
    rf, _meta = load_model()
    if rf is None:
        return None
    feats = _stmt_feats(statement)
    contrib = rf.contributions([feats])[0]
    out = {FEATURE_NAMES[i]: float(contrib[i]) for i in range(len(FEATURE_NAMES))}
    out["bias"] = 0.0
    out["违约概率"] = float(rf.predict_proba([feats])[0])
    return out


def dual_track_conclusion(rule_score, proba):
    """
    双轨对照组合结论（口径与 app.py 红黄绿灯一致）。
    proba 为 None 时返回规则卡单轨结论。
    """
    meta_green, meta_red = 7.0, 4.0
    ml_high, ml_low = 0.50, 0.35
    if proba is None:
        return "单轨（规则卡）：ML 模型不可用，仅按规则卡判断", "rule_only"
    if rule_score >= meta_green and proba < ml_low:
        return "双轨一致·低风险：规则卡与 ML 均判健康，可强推荐", "strong_accept"
    if rule_score < meta_red and proba > 0.60:
        return "双轨一致·高风险：规则卡与 ML 均判高危，建议强拒绝", "strong_reject"
    if rule_score >= meta_green and proba >= ml_high:
        return ("双轨分歧·ML 预警：规则卡判健康，但 ML 检出规则卡盲区风险"
                "，建议人工复核", "diverge_ml_warn")
    if rule_score < meta_red and proba <= ml_low:
        return ("双轨分歧·ML 乐观：规则卡判高危，但 ML 认为风险可控，"
                "建议人工复核", "diverge_ml_optimistic")
    return "双轨中间态：建议补充材料后再评估", "intermediate"


def selftest():
    """基础自检：模型可加载且对高杠杆样本给出更高违约概率。"""
    rf, _meta = load_model()
    if rf is None:
        return False, "模型文件缺失"
    healthy = {"总资产": 2000, "总负债": 800, "营业收入": 3000, "净利润": 300,
               "营业成本": 2400, "利息费用": 40, "应收账款": 300, "存货": 200,
               "流动资产": 1200, "流动负债": 600, "经营活动现金流净额": 400,
               "净利润增长率": 8.0}
    risky = {"总资产": 2000, "总负债": 1900, "营业收入": 3000, "净利润": -200,
             "营业成本": 3200, "利息费用": 200, "应收账款": 1500, "存货": 900,
             "流动资产": 700, "流动负债": 1800, "经营活动现金流净额": -300,
             "净利润增长率": -40.0}
    p_h = predict_default_proba(healthy)
    p_r = predict_default_proba(risky)
    ok = (p_h is not None and p_r is not None and p_r > p_h + 0.05)
    return ok, f"健康样本概率 {p_h:.3f} vs 高风险样本概率 {p_r:.3f}"
