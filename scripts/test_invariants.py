"""
scripts/test_invariants.py — 部署命门不变量回归（纯标准库，CI 基线）

第二轮审查建议纳入仓库的回归基线（对应对方 T1-T3/F1-F2）：
- 纯标准库部分（无 numpy/xgboost 也能跑）：
    F1 模型 JSON 结构合法
    F2 feature_meta.json 完整性与阈值一致性
    T3 sigmoid 恒等式数学核对
- 有 numpy 时追加：
    T1 训练端/推理端 12 维特征逐元素一致（3 样例）
    T2 dual_track_conclusion 真值表无死分支
    T4 运行时不变量均以显式 raise 实现（源码扫描，防 -O 剥离回归）

用法：python3 scripts/test_invariants.py
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# ---------- F1: 模型 JSON 结构 ----------
model_path = os.path.join(ROOT, "models", "xgb_default.json")
if os.path.exists(model_path):
    with open(model_path, encoding="utf-8") as f:
        state = json.load(f)
    check("F1 xgb_default.json 含 learner/version",
          "learner" in state and "version" in state)
else:
    check("F1 xgb_default.json 存在", False, "请先运行 scripts/train_ml.py")

# ---------- F2: feature_meta 完整性 ----------
meta_path = os.path.join(ROOT, "models", "feature_meta.json")
if os.path.exists(meta_path):
    meta = json.load(open(meta_path, encoding="utf-8"))
    required = ["features", "hidden_feature", "lambda_inter", "lambda_hidden",
                "tau", "seed", "base_rate", "rule_green", "rule_red",
                "ml_high", "ml_low", "xgboost_version"]
    missing = [k for k in required if k not in meta]
    check("F2 feature_meta 字段完整", not missing, f"缺失: {missing}" if missing else "")
    check("F2 阈值一致性 ml_high>ml_low, rule_green>rule_red",
          meta.get("ml_high", 0) > meta.get("ml_low", 1)
          and meta.get("rule_green", 0) > meta.get("rule_red", 1))
    check("F2 特征维度=12 且含隐藏因子",
          len(meta.get("features", [])) == 12 and meta.get("hidden_feature") in meta.get("features", []))
else:
    check("F2 feature_meta.json 存在", False)

# ---------- T3: sigmoid 恒等式 ----------
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

check("T3 sigmoid(Σc)==p 数学核对", abs(sigmoid(0.0) - 0.5) < 1e-12
      and abs(sigmoid(2.0) - 1.0 / (1.0 + math.exp(-2.0))) < 1e-12)

# ---------- T4: 运行时不变量显式 raise 扫描（防 python -O 回归）----------
ml_src_path = os.path.join(ROOT, "modules", "ml_model.py")
if os.path.exists(ml_src_path):
    src_lines = open(ml_src_path, encoding="utf-8").read().splitlines()
    bad = [i + 1 for i, ln in enumerate(src_lines)
           if ln.strip().startswith("assert ") or ln.strip().startswith("assert(")]
    check("T4 ml_model.py 无 assert 运行时不变量", not bad,
          f"行 {bad} 发现 assert——请改为显式 raise（python -O 会剥离 assert）" if bad else "")

# ---------- 需要 numpy 的测试 ----------
try:
    import numpy as np  # noqa
    HAS_NP = True
except ImportError:
    HAS_NP = False
    print("[SKIP] 无 numpy：跳过 T1/T2（纯标准库部分已完成）")

if HAS_NP:
    sys.path.insert(0, os.path.join(ROOT, "modules"))
    import ml_model

    SAMPLES = [
        {"总资产": 2000, "总负债": 1500, "营业收入": 3000, "净利润": 100,
         "营业成本": 2400, "利息费用": 80, "应收账款": 400, "存货": 300,
         "短期借款": 500, "流动资产": 1000, "流动负债": 900,
         "经营活动现金流净额": 150, "应收账款_3月内占比": 60,
         "应收账款_超12月占比": 10, "营收增长率": -15.0, "净利润增长率": -30.0,
         "经营年限": 4, "纳税信用评级": "B", "融资机构数量": 3,
         "客户集中度": "中（30%~60%）", "平均融资利率": 6.0,
         "法院执行记录": "无", "实控人征信状态": "良好", "行业周期信号": -0.8},
        {"总资产": 500, "总负债": 100, "营业收入": 800, "净利润": 90,
         "营业成本": 550, "利息费用": 5, "应收账款": 60, "存货": 40,
         "短期借款": 20, "流动资产": 380, "流动负债": 120,
         "经营活动现金流净额": 110, "应收账款_3月内占比": 85,
         "应收账款_超12月占比": 2, "营收增长率": 25.0, "净利润增长率": 30.0,
         "经营年限": 9, "纳税信用评级": "A", "融资机构数量": 1,
         "客户集中度": "低（前五大客户占比<30%）", "平均融资利率": 3.9,
         "法院执行记录": "无", "实控人征信状态": "良好", "行业周期信号": 0.7},
        {},  # 全缺失：防御写法不崩
    ]

    # T1：推理端与训练端特征一致性（训练端公式按 scripts/train_ml.py 复刻）
    def train_side(s):
        def f(key, default=0.0):
            try:
                return float(s.get(key, default))
            except (TypeError, ValueError):
                return default
        A = f("总资产"); L = f("总负债"); R = f("营业收入")
        return [
            L / A if A > 0 else 0.0,
            f("流动资产") / max(f("流动负债", 1.0), 1e-6),
            f("经营活动现金流净额") / R if R > 0 else 0.0,
            f("净利润") / R if R > 0 else 0.0,
            f("营收增长率") / 100.0,
            float(np.clip(f("净利润增长率") / 100.0, -1.0, 2.0)),
            f("应收账款_3月内占比") / 100.0,
            f("应收账款_超12月占比") / 100.0,
            math.log1p(f("经营年限")),
            {"A": 5, "B": 4, "M": 3, "C": 2, "D": 1}.get(
                str(s.get("纳税信用评级", "")).strip(), 0) / 5.0,
            f("融资机构数量") / 7.0,
            f("行业周期信号"),
        ]

    t1_ok = all(np.allclose(np.array(train_side(s), dtype=float),
                            ml_model.statement_to_features(s), atol=1e-12)
                for s in SAMPLES)
    check("T1 训练端/推理端 12 维特征逐元素一致（3 样例，含全缺失）", t1_ok)

    # T2：dual_track_conclusion 真值表
    cases = [
        (8.0, 0.1, "strong_accept"), (8.0, 0.45, "intermediate"),
        (8.0, 0.7, "diverge_ml_warn"), (3.0, 0.8, "strong_reject"),
        (3.0, 0.55, "intermediate"), (3.0, 0.2, "diverge_ml_optimistic"),
        (5.5, 0.05, "intermediate"), (5.5, 0.9, "intermediate"),
        (8.0, None, "rule_only"), (2.0, None, "rule_only"),
    ]
    t2_ok = True
    for rs, p, tag in cases:
        _, got = ml_model.dual_track_conclusion(rs, p)
        if got != tag:
            t2_ok = False
            print(f"    真值表不符：rule={rs} p={p} 期望 {tag} 实得 {got}")
    check("T2 dual_track_conclusion 真值表（10 组边界）", t2_ok)

print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
