"""
modules/rf_model.py — 纯 Python 随机森林（零第三方依赖，标准库实现）

为什么纯 Python：
- 目标部署/训练环境（HarmonyOS 沙箱）无法安装 numpy/sklearn/xgboost（musl libc 与
  manylinux 轮子不兼容、且无编译器），因此本模型用纯 Python 实现，可在任何有 Python
  解释器的环境直接运行，模型文件为可读 JSON，便于审计与移植。

能力：
- 分类随机森林：bootstrap 聚合 + 特征子采样 + Gini 分裂 + 深度/叶样本数约束。
- 类别不平衡：平衡自助采样（balanced_bootstrap）上采样少数类。
- 特征重要性：累计 Gini 不纯度下降（按节点样本权重）。
- 单样本边际贡献：扰动近似（将某特征置训练均值后观察概率变化），用于热力图。
- save/load：JSON 序列化，应用运行时直接加载推理，无需训练框架。
"""
import json
import math
import random


def _gini(pos, neg):
    """给定正/负样本权重和，返回 Gini 不纯度。"""
    tot = pos + neg
    if tot <= 0:
        return 0.0
    p = pos / tot
    return 1.0 - (p * p + (1.0 - p) * (1.0 - p))


class _Tree:
    def __init__(self, node):
        self.node = node  # dict: leaf->{'leaf':p} ; internal->{'f','t','l','r','imp','n'}

    @staticmethod
    def _predict(node, x):
        while "leaf" not in node:
            node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
        return node["leaf"]

    def predict(self, x):
        return _Tree._predict(self.node, x)


class RandomForest:
    def __init__(self, n_estimators=200, max_depth=8, min_samples_leaf=30,
                 max_features="sqrt", min_impurity_decrease=0.0,
                 balanced_bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.balanced_bootstrap = balanced_bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = []
        self.n_features_ = 0
        self.feature_means_ = []
        self.classes_ = [0, 1]
        self.base_rate_ = 0.0

    # -------------------------------------------------- fit
    def fit(self, X, y):
        random.seed(self.random_state)
        self.n_features_ = len(X[0]) if X else 0
        y = list(y)
        n = len(y)
        pos_idx = [i for i in range(n) if y[i] == 1]
        neg_idx = [i for i in range(n) if y[i] == 0]
        self.base_rate_ = len(pos_idx) / max(n, 1)
        # 训练集特征均值（用于扰动贡献）
        if n:
            self.feature_means_ = [
                sum(row[f] for row in X) / n for f in range(self.n_features_)
            ]
        else:
            self.feature_means_ = [0.0] * self.n_features_

        imp_acc = [0.0] * self.n_features_
        for _ in range(self.n_estimators):
            if self.balanced_bootstrap and pos_idx and neg_idx:
                # 平衡自助：每类各抽样 n/2（有放回）
                half = n // 2
                idx = [random.choice(pos_idx) for _ in range(half)] + \
                      [random.choice(neg_idx) for _ in range(n - half)]
            else:
                idx = [random.choice(range(n)) for _ in range(n)]
            Xb = [X[i] for i in idx]
            yb = [y[i] for i in idx]
            tree, imp = self._build(Xb, yb, 0)
            self.trees.append(_Tree(tree))
            for f in range(self.n_features_):
                imp_acc[f] += imp[f]
        tot = sum(imp_acc) or 1.0
        self.feature_importances_ = [v / tot for v in imp_acc]
        return self

    def _max_feats(self):
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(self.n_features_)))
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, self.n_features_))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features_))
        return self.n_features_

    def _build(self, X, y, depth):
        n = len(y)
        pos = sum(y)
        neg = n - pos
        node_imp = [0.0] * self.n_features_
        gini_node = _gini(pos, neg)
        if depth >= self.max_depth or n <= max(2, self.min_samples_leaf) or pos == 0 or neg == 0:
            return {"leaf": (pos / n if n else 0.0), "n": n}, node_imp

        k = self._max_feats()
        feats = random.sample(range(self.n_features_), k)
        best = None  # (gain, f, t, l_idx, r_idx)
        # 预按特征排序以加速阈值枚举
        for f in feats:
            # 收集该特征取值与标签，去重排序
            pairs = sorted(set((row[f], lab) for row, lab in zip(X, y)))
            if len(pairs) < 2:
                continue
            # 候选阈值：相邻不同取值的中点（若过多则抽样）
            cand = []
            for i in range(len(pairs) - 1):
                if pairs[i][1] != pairs[i + 1][1] or True:
                    cand.append((pairs[i][0] + pairs[i + 1][0]) / 2.0)
            if len(cand) > 50:
                step = len(cand) // 50
                cand = cand[::step]
            # 顺序扫描统计（按阈值排序，pairs 已按特征值排序）
            # 用前缀统计快速求左右正负
            # pairs 已按特征值排序（因 set 后 sorted 按 (val,lab)，val 为主键）
            vals = [p[0] for p in pairs]
            labs = [p[1] for p in pairs]
            # 累计：left 包含 val<=t 的样本
            # 因为同一 val 可能对应不同 lab，需按 val 分组
            # 直接对每个阈值扫描 O(m) 整体 O(m^2)，m 较小可接受
            # 优化：用字典按 val 聚合正负
            agg = {}
            for v, lb in zip(vals, labs):
                if v not in agg:
                    agg[v] = [0.0, 0.0]
                agg[v][lb] += 1.0
            sorted_vals = sorted(agg.keys())
            # 前缀累计
            pref_pos = [0.0]; pref_neg = [0.0]
            for v in sorted_vals:
                pref_pos.append(pref_pos[-1] + agg[v][1])
                pref_neg.append(pref_neg[-1] + agg[v][0])
            total_pos = pref_pos[-1]; total_neg = pref_neg[-1]
            cand_sorted = sorted(cand)
            j = 0
            for t in cand_sorted:
                # 滑动指针：cand 已递增，j 单调前进 -> O(m)
                while j < len(sorted_vals) and sorted_vals[j] <= t:
                    j += 1
                lp = pref_pos[j]; ln = pref_neg[j]
                rp = total_pos - lp; rn = total_neg - ln
                if lp + ln < self.min_samples_leaf or rp + rn < self.min_samples_leaf:
                    continue
                gl = _gini(lp, ln); gr = _gini(rp, rn)
                w_l = (lp + ln) / n; w_r = (rp + rn) / n
                gain = gini_node - (w_l * gl + w_r * gr)
                if gain > self.min_impurity_decrease and (best is None or gain > best[0]):
                    best = (gain, f, t)
        if best is None:
            return {"leaf": (pos / n if n else 0.0), "n": n}, node_imp
        gain, f, t = best
        left = [i for i in range(n) if X[i][f] <= t]
        right = [i for i in range(n) if X[i][f] > t]
        if not left or not right:
            return {"leaf": (pos / n if n else 0.0), "n": n}, node_imp
        l_node, l_imp = self._build([X[i] for i in left], [y[i] for i in left], depth + 1)
        r_node, r_imp = self._build([X[i] for i in right], [y[i] for i in right], depth + 1)
        for fi in range(self.n_features_):
            node_imp[fi] = l_imp[fi] + r_imp[fi]
        node_imp[f] += gain * n  # 不纯度下降按节点样本数加权
        return {"f": f, "t": t, "l": l_node, "r": r_node,
                "imp": gain, "n": n}, node_imp

    # -------------------------------------------------- predict
    def predict_proba(self, X):
        if not self.trees:
            return [0.5] * len(X)
        out = []
        for x in X:
            s = 0.0
            for tr in self.trees:
                s += tr.predict(x)
            out.append(s / len(self.trees))
        return out

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    # -------------------------------------------------- contributions (扰动近似)
    def contributions(self, X, means=None):
        """返回 (n,d) 边际贡献矩阵：contrib[i][f] = base_p[i] - p(将 f 置均值后)。
        正值表示该特征把样本推离均值时降低了违约概率（即该特征的“正常”值压低风险）。"""
        if means is None:
            means = self.feature_means_
        base = self.predict_proba(X)
        n = len(X); d = self.n_features_
        out = [[0.0] * d for _ in range(n)]
        for i in range(n):
            xi = list(X[i])
            save = xi[:]
            for f in range(d):
                xi[f] = means[f]
                p = 0.0
                for tr in self.trees:
                    p += tr.predict(xi)
                p /= len(self.trees)
                out[i][f] = base[i] - p
                xi[f] = save[f]
        return out

    # -------------------------------------------------- io
    def save(self, path):
        obj = {
            "type": "RandomForest",
            "params": {
                "n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "min_impurity_decrease": self.min_impurity_decrease,
                "balanced_bootstrap": self.balanced_bootstrap,
                "random_state": self.random_state,
            },
            "n_features": self.n_features_,
            "feature_means": self.feature_means_,
            "feature_importances": self.feature_importances_,
            "base_rate": self.base_rate_,
            "trees": [self._serialize(t.node) for t in self.trees],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @staticmethod
    def _serialize(node):
        if "leaf" in node:
            return {"leaf": node["leaf"]}
        return {"f": node["f"], "t": node["t"],
                "l": RandomForest._serialize(node["l"]),
                "r": RandomForest._serialize(node["r"])}

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        rf = cls(**obj["params"])
        rf.n_features_ = obj["n_features"]
        rf.feature_means_ = obj["feature_means"]
        rf.feature_importances_ = obj["feature_importances"]
        rf.base_rate_ = obj["base_rate"]
        rf.trees = [_Tree(node) for node in obj["trees"]]
        return rf
