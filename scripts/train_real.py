"""
scripts/train_real.py — 真实数据企业融资/破产评分模型训练管线（v2，纯 Python 实现）

数据来源（真实、公开）：
  Taiwan Economic Journal 企业破产预测数据集（UCI / Kaggle 镜像）
  - 6,819 家真实上市公司，95 个财务比率，二元破产标签（1999–2009）
  - 已下载至 data/real/company_bankruptcy.csv

相对旧版（ml_review 合成数据）的根本改进：
  1. 用真实企业财务 + 真实破产标签训练，模型具备真实预测能力（不再是“学公式”）。
  2. 移除无真实数据源的“行业周期信号”隐藏特征，评分更诚实。
  3. 模型对比：随机森林（主）vs 逻辑回归（基线）；含超参随机搜索 + 分层交叉验证。
  4. 可视化：训练曲线 / 特征重要性 / 特征影响热力图 / 模型对比 / 校准 / 混淆矩阵（SVG）。

运行：python scripts/train_real.py   （仅需 Python 标准库，无需 numpy/sklearn）
产物：models/rf_model.json, models/feature_meta.json, models/real_metrics.json,
      models/real_charts/*.svg, ML_REAL_REPORT 数据摘要（由调用方组装）。
"""
import os
import sys
import csv
import math
import json
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.features import FEATURE_NAMES, dataset_row_to_features
from modules.rf_model import RandomForest
import utils.svg_charts as svg

SEED = 42
random.seed(SEED)
DATA_PATH = os.path.join(ROOT, "data/real/company_bankruptcy.csv")
MODELS_DIR = os.path.join(ROOT, "models")
CHART_DIR = os.path.join(MODELS_DIR, "real_charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# 指标（纯 Python）
# ============================================================
def auc(y, p):
    n = len(y)
    order = sorted(range(n), key=lambda i: p[i])
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r + 1
    by_val = defaultdict(list)
    for i in order:
        by_val[p[i]].append(i)
    for v, idxs in by_val.items():
        avg = (min(rank[i] for i in idxs) + max(rank[i] for i in idxs)) / 2.0
        for i in idxs:
            rank[i] = avg
    pos = [i for i in range(n) if y[i] == 1]
    neg = [i for i in range(n) if y[i] == 0]
    if not pos or not neg:
        return float("nan")
    return (sum(rank[i] for i in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ks_stat(y, p):
    n = len(y)
    order = sorted(range(n), key=lambda i: p[i])
    pos = sum(y); neg = n - pos
    tp = fp = 0
    best = 0.0
    for i in order:
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        best = max(best, abs(tp / pos - fp / neg))
    return best


def pr_auc(y, p):
    n = len(y)
    order = sorted(range(n), key=lambda i: -p[i])
    pos = sum(y)
    if pos == 0:
        return float("nan")
    prec = 0.0; rec = 0.0; ap = 0.0; prev_rec = 0.0
    for i in order:
        if y[i] == 1:
            prec += 1; rec += 1
        else:
            prec += 1
        p_ = rec / prec if prec else 0
        r_ = rec / pos
        ap += (r_ - prev_rec) * (p_ if p_ else 0)
        prev_rec = r_
    return ap


def brier(y, p):
    return sum((p[i] - y[i]) ** 2 for i in range(len(y))) / len(y)


def confusion(y, pred, threshold):
    cm = [[0, 0], [0, 0]]
    for i in range(len(y)):
        a = 1 if y[i] >= 0.5 else 0
        b = 1 if pred[i] >= threshold else 0
        cm[a][b] += 1
    return cm


def threshold_youden(y, p):
    best_t, best_j = 0.5, -1
    for t in [i / 100 for i in range(1, 100)]:
        tp = fp = fn = tn = 0
        for i in range(len(y)):
            a = y[i] == 1; b = p[i] >= t
            if a and b: tp += 1
            elif a and not b: fn += 1
            elif (not a) and b: fp += 1
            else: tn += 1
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        if tpr - fpr > best_j:
            best_j = tpr - fpr; best_t = t
    return best_t


# ============================================================
# 逻辑回归基线（纯 Python，带标准化与 L2）
# ============================================================
class LogReg:
    def __init__(self, lr=0.3, iters=1500, C=1.0):
        self.lr = lr; self.iters = iters; self.C = C

    def fit(self, X, y):
        n = len(X); d = len(X[0])
        self.mu = [sum(r[f] for r in X) / n for f in range(d)]
        sd = [math.sqrt(sum((r[f] - self.mu[f]) ** 2 for r in X) / n) or 1.0 for f in range(d)]
        self.sd = sd
        Xs = [[(r[f] - self.mu[f]) / sd[f] for f in range(d)] for r in X]
        w = [0.0] * d; b = 0.0
        lam = 1.0 / (self.C * n)
        for _ in range(self.iters):
            gw = [0.0] * d; gb = 0.0
            for i in range(n):
                z = b + sum(w[f] * Xs[i][f] for f in range(d))
                pr = 1 / (1 + math.exp(-z))
                err = pr - y[i]
                gb += err
                for f in range(d):
                    gw[f] += err * Xs[i][f] + lam * w[f]
            b -= self.lr * gb / n
            for f in range(d):
                w[f] -= self.lr * gw[f] / n
        self.w = w; self.b = b
        return self

    def predict_proba(self, X):
        out = []
        for r in X:
            z = self.b + sum(self.w[f] * (r[f] - self.mu[f]) / self.sd[f] for f in range(len(r)))
            out.append(1 / (1 + math.exp(-z)))
        return out


# ============================================================
# 分层抽样 / 切分
# ============================================================
def stratified_split(y, frac, seed=SEED):
    rnd = random.Random(seed)
    pos = [i for i in range(len(y)) if y[i] == 1]
    neg = [i for i in range(len(y)) if y[i] == 0]
    rnd.shuffle(pos); rnd.shuffle(neg)
    kp = int(len(pos) * frac); kn = int(len(neg) * frac)
    train = pos[:kp] + neg[:kn]
    test = pos[kp:] + neg[kn:]
    rnd.shuffle(train); rnd.shuffle(test)
    return train, test


def stratified_kfold(y, k, seed=SEED):
    rnd = random.Random(seed)
    pos = [i for i in range(len(y)) if y[i] == 1]
    neg = [i for i in range(len(y)) if y[i] == 0]
    rnd.shuffle(pos); rnd.shuffle(neg)
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(pos):
        folds[i % k].append(idx)
    for i, idx in enumerate(neg):
        folds[i % k].append(idx)
    return folds


# ============================================================
# 主流程
# ============================================================
def main():
    print("[1] 加载真实数据集")
    rows = list(csv.DictReader(open(DATA_PATH, encoding="utf-8", errors="replace")))
    X = [dataset_row_to_features(r) for r in rows]
    y = [int(float(r["Bankrupt?"])) for r in rows]
    print(f"    样本={len(X)}  特征={len(X[0])}  破产率={sum(y)/len(y):.4f}")

    # train / val / test = 64% / 16% / 20%
    tr, rest = stratified_split(y, 0.80)
    va, te = stratified_split([y[i] for i in rest], 0.50,
                               seed=SEED + 1)
    va = [rest[i] for i in va]; te = [rest[i] for i in te]
    Xtr = [X[i] for i in tr]; ytr = [y[i] for i in tr]
    Xva = [X[i] for i in va]; yva = [y[i] for i in va]
    Xte = [X[i] for i in te]; yte = [y[i] for i in te]
    print(f"    train={len(tr)} val={len(va)} test={len(te)}")

    # ---------------------------------------------------------
    # [2] 超参随机搜索 + 分层 3-fold CV（在 train 上）
    # ---------------------------------------------------------
    print("[2] 超参随机搜索（3-fold CV）")
    folds = stratified_kfold(ytr, 3, seed=SEED)
    param_grid = [
        dict(n_estimators=150, max_depth=6, min_samples_leaf=40),
        dict(n_estimators=200, max_depth=8, min_samples_leaf=30),
        dict(n_estimators=250, max_depth=10, min_samples_leaf=25),
        dict(n_estimators=200, max_depth=8, min_samples_leaf=50),
    ]
    best = None
    for pg in param_grid:
        cv_aucs = []
        for fi in range(3):
            tridx = [i for k in range(3) if k != fi for i in folds[k]]
            teidx = folds[fi]
            rf = RandomForest(balanced_bootstrap=True, random_state=SEED,
                              max_features="sqrt", **pg)
            rf.fit([Xtr[i] for i in tridx], [ytr[i] for i in tridx])
            pa = rf.predict_proba([Xtr[i] for i in teidx])
            cv_aucs.append(auc([ytr[i] for i in teidx], pa))
        mean_auc = sum(cv_aucs) / 3
        print(f"    {pg} -> CV AUC={mean_auc:.4f} ({[round(a,3) for a in cv_aucs]})")
        if best is None or mean_auc > best[0]:
            best = (mean_auc, pg)
    best_pg = best[1]
    print(f"    >> 最佳参数 {best_pg}（CV AUC={best[0]:.4f}）")

    # ---------------------------------------------------------
    # [3] 最终模型（train+val 训练，test 评估）
    # ---------------------------------------------------------
    print("[3] 最终训练 + 测试评估")
    Xtrva = Xtr + Xva; ytrva = ytr + yva
    rf = RandomForest(balanced_bootstrap=True, random_state=SEED,
                      max_features="sqrt", **best_pg)
    rf.fit(Xtrva, ytrva)
    pte = rf.predict_proba(Xte)
    test_auc = auc(yte, pte)
    test_ks = ks_stat(yte, pte)
    test_pr = pr_auc(yte, pte)
    test_br = brier(yte, pte)
    # 阈值在 val 上选，再作用于 test
    pva = rf.predict_proba(Xva)
    thr = threshold_youden(yva, pva)
    pred_te = [1 if p >= thr else 0 for p in pte]
    tp = sum(1 for i in range(len(yte)) if yte[i] == 1 and pred_te[i] == 1)
    fp = sum(1 for i in range(len(yte)) if yte[i] == 0 and pred_te[i] == 1)
    fn = sum(1 for i in range(len(yte)) if yte[i] == 1 and pred_te[i] == 0)
    tn = sum(1 for i in range(len(yte)) if yte[i] == 0 and pred_te[i] == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(yte)
    cm = [[tn, fp], [fn, tp]]
    print(f"    TEST  AUC={test_auc:.4f} KS={test_ks:.4f} PR-AUC={test_pr:.4f} Brier={test_br:.4f}")
    print(f"    阈值={thr:.2f}  精确率={prec:.3f} 召回率={rec:.3f} F1={f1:.3f} 准确率={acc:.3f}")

    # ---------------------------------------------------------
    # [4] 逻辑回归基线（同切分，best-effort，失败不影响主产物）
    # ---------------------------------------------------------
    print("[4] 逻辑回归基线")
    lr_auc = lr_ks = lr_prec = lr_rec = lr_f1 = 0.0
    try:
        lr = LogReg(lr=0.3, iters=1500, C=1.0).fit(Xtrva, ytrva)
        plr = lr.predict_proba(Xte)
        lr_auc = auc(yte, plr); lr_ks = ks_stat(yte, plr)
        lpred = [1 if p >= 0.5 else 0 for p in plr]
        ltp = sum(1 for i in range(len(yte)) if yte[i] == 1 and lpred[i] == 1)
        lfp = sum(1 for i in range(len(yte)) if yte[i] == 0 and lpred[i] == 1)
        lfn = sum(1 for i in range(len(yte)) if yte[i] == 1 and lpred[i] == 0)
        lr_prec = ltp / (ltp + lfp) if (ltp + lfp) else 0
        lr_rec = ltp / (ltp + lfn) if (ltp + lfn) else 0
        lr_f1 = 2 * lr_prec * lr_rec / (lr_prec + lr_rec) if (lr_prec + lr_rec) else 0
        print(f"    LR    AUC={lr_auc:.4f} KS={lr_ks:.4f} F1={lr_f1:.3f}")
    except Exception as e:
        print(f"    [警告] 逻辑回归基线计算失败，跳过对比：{e}")

    # ---------------------------------------------------------
    # [5] 训练曲线：测试 AUC 随树数量变化（收敛性）
    # ---------------------------------------------------------
    print("[5] 生成可视化（SVG）")
    ns = [25, 50, 75, 100, 125, 150, 175, 200]
    curve = []
    for ntree in ns:
        m = RandomForest(balanced_bootstrap=True, random_state=SEED, max_features="sqrt",
                         n_estimators=ntree, **{k: v for k, v in best_pg.items() if k != "n_estimators"})
        m.fit(Xtrva, ytrva)
        curve.append(auc(yte, m.predict_proba(Xte)))
    with open(os.path.join(CHART_DIR, "training_curve.svg"), "w", encoding="utf-8") as f:
        f.write(svg.line_chart("随机森林训练曲线：测试集 AUC 随树数量变化",
                               ns, [("测试集 AUC", "#2F54EB", curve)],
                               xlabel="树数量", ylabel="AUC"))

    # 特征重要性
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1])
    svg.hbar_chart("特征重要性排序（Gini 不纯度下降）",
                   [n for n, _ in imp], [v for _, v in imp], color="#2F54EB")
    with open(os.path.join(CHART_DIR, "feature_importance.svg"), "w", encoding="utf-8") as f:
        f.write(svg.hbar_chart("特征重要性排序（Gini 不纯度下降）",
                               [n for n, _ in imp], [v for _, v in imp], color="#2F54EB"))

    # 特征影响热力图（扰动贡献，测试子集）
    n_sub = min(120, len(Xte))
    Xsub = Xte[:n_sub]
    contrib = rf.contributions(Xsub)
    # 按行最大绝对贡献排序，便于观察结构
    order = sorted(range(n_sub), key=lambda i: -max(abs(v) for v in contrib[i]))
    contrib_sorted = [contrib[i] for i in order]
    with open(os.path.join(CHART_DIR, "contrib_heatmap.svg"), "w", encoding="utf-8") as f:
        f.write(svg.heatmap("特征对最终评分的边际影响热力图（红=推高风险 / 蓝=压低风险）",
                            contrib_sorted, FEATURE_NAMES))

    # 模型对比
    with open(os.path.join(CHART_DIR, "model_comparison.svg"), "w", encoding="utf-8") as f:
        f.write(svg.grouped_bar("多模型预测性能对比（测试集）",
                                ["RF(主模型)", "LogReg(基线)"],
                                [("AUC", "#2F54EB", [test_auc, lr_auc]),
                                 ("KS", "#2BA471", [test_ks, lr_ks]),
                                 ("F1", "#D98B1F", [f1, lr_f1])],
                                ylabel="Score"))

    # 校准曲线（手动分箱，避免第三方依赖）
    bins = 10
    edges = [i / bins for i in range(bins + 1)]
    mp, fp2 = [], []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        seg = [i for i in range(len(yte)) if lo <= pte[i] < hi or (b == bins - 1 and pte[i] <= hi)]
        if seg:
            mp.append(sum(pte[i] for i in seg) / len(seg))
            fp2.append(sum(yte[i] for i in seg) / len(seg))
    with open(os.path.join(CHART_DIR, "calibration.svg"), "w", encoding="utf-8") as f:
        f.write(svg.calibration_curve("概率校准曲线（可靠性图）", mp, fp2))

    # 混淆矩阵
    with open(os.path.join(CHART_DIR, "confusion.svg"), "w", encoding="utf-8") as f:
        f.write(svg.confusion_matrix("混淆矩阵（测试集，阈值=%.2f）" % thr, cm,
                                     ["正常", "违约"]))

    # ---------------------------------------------------------
    # [6] 保存模型 + 元数据 + 指标
    # ---------------------------------------------------------
    print("[6] 保存模型与指标")
    rf.save(os.path.join(MODELS_DIR, "rf_model.json"))
    meta = {
        "model_type": "RandomForest",
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "base_rate": rf.base_rate_,
        "feature_means": rf.feature_means_,
        "best_params": best_pg,
        "threshold": thr,
        "data_source": "Taiwan Economic Journal 企业破产预测（UCI/Kaggle，6819 家真实上市公司，1999-2009）",
        "data_file": "data/real/company_bankruptcy.csv",
        "note": "纯 Python 实现，无第三方 ML 依赖；替代旧版合成数据 XGBoost 模型。",
        "version": "2.0-real",
    }
    with open(os.path.join(MODELS_DIR, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    metrics = {
        "dataset": {"n": len(X), "n_features": len(X[0]), "bankrupt_rate": sum(y) / len(y)},
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "best_params": best_pg,
        "random_forest": {
            "AUC": round(test_auc, 4), "KS": round(test_ks, 4),
            "PR_AUC": round(test_pr, 4), "Brier": round(test_br, 4),
            "threshold": round(thr, 3), "precision": round(prec, 4),
            "recall": round(rec, 4), "F1": round(f1, 4), "accuracy": round(acc, 4),
        },
        "logistic_regression": {
            "AUC": round(lr_auc, 4), "KS": round(lr_ks, 4), "F1": round(lr_f1, 4),
        },
        "feature_importance": {n: round(v, 4) for n, v in imp},
        "confusion_matrix": cm,
    }
    with open(os.path.join(MODELS_DIR, "real_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n==== 完成 ====")
    print(f"测试集 RF: AUC={test_auc:.4f} KS={test_ks:.4f} F1={f1:.4f}")
    print(f"测试集 LR: AUC={lr_auc:.4f} KS={lr_ks:.4f} F1={lr_f1:.4f}")
    print("产物：models/rf_model.json, models/feature_meta.json, models/real_metrics.json")
    print("图表：models/real_charts/*.svg (6 张)")
    return metrics


if __name__ == "__main__":
    main()
