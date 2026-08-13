"""
模块5：ML 违约模型运行时接口（双轨对照的 ML 轨道）
v0.1-ML：XGBoost 原生推理 + pred_contribs 归因，零 shap/numba 依赖。

审查追溯：
  P2 懒加载兜底：模型缺失时优雅降级为纯规则卡，不中断应用
  P1 归因：pred_contribs（sigmoid 自洽）
  隐藏因子：行业周期信号仅本模块可见，规则卡（diagnosis.py）不可见

部署说明：训练在离线完成（scripts/train_ml.py），运行期只 load。
模型文件：models/xgb_default.json（xgboost 3.4.0 为 py3-none 通用 wheel，
Python 3.14 无兼容风险，已核验——修正4）。
"""
import os
import math
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO, "models", "xgb_default.json")
META_PATH = os.path.join(REPO, "models", "feature_meta.json")

FEATURE_NAMES = [
    "资产负债率", "流动比率", "现金流覆盖", "净利率",
    "营收增长率", "净利润增长率", "应收3月内占比", "应收超12月占比",
    "经营年限ln", "纳税评级分", "融资机构数", "行业周期信号",
]
HIDDEN_FEATURE = "行业周期信号"

_MODEL_CACHE = {}


def _lazy_import_xgb():
    import xgboost as xgb
    return xgb


def load_model():
    """
    P2 懒加载 + 兜底：首次调用才加载；文件或缺依赖时返回 None，
    调用方应降级为纯规则卡诊断。
    """
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["meta"]
    try:
        import json
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            _MODEL_CACHE.update(model=None, meta=None)
            return None, None
        xgb = _lazy_import_xgb()
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        # 审查 P1-2：xgboost 主版本不一致时降级为纯规则卡（跨主版本 JSON 格式可能漂移）
        train_ver = str(meta.get("xgboost_version", ""))
        run_ver = getattr(xgb, "__version__", "")
        if train_ver.split(".")[0] != run_ver.split(".")[0]:
            print(f"[ml_model] 警告：模型由 xgboost {train_ver} 训练，当前运行 {run_ver}，"
                  "主版本不一致，降级为纯规则卡")
            _MODEL_CACHE.update(model=None, meta=None)
            return None, None
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        _MODEL_CACHE.update(model=booster, meta=meta)
        return booster, meta
    except Exception:
        _MODEL_CACHE.update(model=None, meta=None)
        return None, None


def statement_to_features(statement):
    """
    原始财务报表（金额单位万元）→ 12 维特征向量。
    与 scripts/train_ml.py 的 statement_to_ml_features 保持严格一致。
    statement 中必须含"行业周期信号"（生产环境接行业景气度数据源；
    演示环境可用人工录入的 -1~1 评分）。
    """
    A = float(statement.get("总资产", 0) or 0)
    L = float(statement.get("总负债", 0) or 0)
    R = float(statement.get("营业收入", 0) or 0)
    feats = {
        "资产负债率": L / A if A > 0 else 0.0,
        "流动比率": float(statement.get("流动资产", 0)) / max(float(statement.get("流动负债", 1)), 1e-6),
        "现金流覆盖": float(statement.get("经营活动现金流净额", 0)) / R if R > 0 else 0.0,
        "净利率": float(statement.get("净利润", 0)) / R if R > 0 else 0.0,
        "营收增长率": float(statement.get("营收增长率", 0)) / 100.0,
        "净利润增长率": float(np.clip(float(statement.get("净利润增长率", 0)) / 100.0, -1.0, 2.0)),
        "应收3月内占比": float(statement.get("应收账款_3月内占比", 0)) / 100.0,
        "应收超12月占比": float(statement.get("应收账款_超12月占比", 0)) / 100.0,
        "经营年限ln": math.log1p(float(statement.get("经营年限", 0))),
        "纳税评级分": {"A": 5, "B": 4, "M": 3, "C": 2, "D": 1}.get(
            str(statement.get("纳税信用评级", "")).strip(), 0) / 5.0,
        "融资机构数": float(statement.get("融资机构数量", 0)) / 7.0,
        HIDDEN_FEATURE: float(statement.get(HIDDEN_FEATURE, 0.0)),
    }
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=float)


def predict_default_proba(statement):
    """返回违约概率；模型不可用时返回 None（调用方降级为纯规则卡）"""
    booster, _meta = load_model()
    if booster is None:
        return None
    xgb = _lazy_import_xgb()
    X = statement_to_features(statement).reshape(1, -1)
    dm = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    return float(booster.predict(dm)[0])


def explain_statement(statement):
    """
    单样本 SHAP 归因（XGBoost 原生 pred_contribs）。
    返回 {特征名: 贡献值(logit空间), ..., "bias": ...}；模型不可用返回 None。
    """
    booster, _meta = load_model()
    if booster is None:
        return None
    xgb = _lazy_import_xgb()
    X = statement_to_features(statement).reshape(1, -1)
    dm = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    contribs = booster.predict(dm, pred_contribs=True)[0]
    # 自洽断言：sigmoid(贡献和) == 预测概率
    margin = float(contribs.sum())
    p = 1.0 / (1.0 + math.exp(-margin))
    p_direct = float(booster.predict(dm)[0])
    # 审查 P1-1：运行时不变量用显式 raise（assert 在 python -O 下会被剥离）
    if abs(p - p_direct) >= 1e-5:
        raise RuntimeError("SHAP 贡献求和与模型概率不一致")
    out = {name: float(contribs[i]) for i, name in enumerate(FEATURE_NAMES)}
    out["bias"] = float(contribs[-1])
    out["违约概率"] = p_direct
    return out


def dual_track_conclusion(rule_score, proba):
    """
    双轨对照组合结论（口径与 app.py 红黄绿灯一致）。
    proba 为 None 时返回规则卡单轨结论。
    """
    # 审查 P2-4 口径说明：本函数是三级结论分级（低/中/高危，0.60 用于强拒绝），
    # 与训练端 divergence_stats 的二分分歧口径（0.50/0.35）用途不同，非同一套阈值。
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
                "（行业周期/非线性交互），建议人工复核", "diverge_ml_warn")
    if rule_score < meta_red and proba <= ml_low:
        return ("双轨分歧·ML 乐观：规则卡判高危，但 ML 认为风险可控，"
                "建议人工复核", "diverge_ml_optimistic")
    return "双轨中间态：建议补充材料后再评估", "intermediate"


def selftest():
    """隐藏因子泄漏自检：扰动行业周期 → ML 概率必须变化"""
    booster, _meta = load_model()
    if booster is None:
        return False, "模型文件缺失"
    base = {"总资产": 2000, "总负债": 1500, "营业收入": 3000, "净利润": 100,
            "营业成本": 2400, "利息费用": 80, "应收账款": 400, "存货": 300,
            "短期借款": 500, "流动资产": 1000, "流动负债": 900,
            "经营活动现金流净额": 150, "应收账款_3月内占比": 60,
            "应收账款_超12月占比": 10, "营收增长率": -15.0, "净利润增长率": -30.0,
            "经营年限": 4, "纳税信用评级": "B", "融资机构数量": 3,
            "客户集中度": "中（30%~60%）", "平均融资利率": 6.0,
            "法院执行记录": "无", "实控人征信状态": "良好",
            HIDDEN_FEATURE: 0.8}
    p_up = predict_default_proba(dict(base))
    bad = dict(base); bad[HIDDEN_FEATURE] = -0.8
    p_down = predict_default_proba(bad)
    ok = abs(p_up - p_down) > 0.02
    return ok, f"行业上行 {p_up:.3f} vs 行业下行 {p_down:.3f}"
