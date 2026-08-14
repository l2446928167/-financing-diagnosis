"""
utils/persistence.py — 对话与跨期企业数据的本地持久化

能力（对应需求：对话记忆 + 企业数据保存 + 前后上传自动比对）：
  1) 会话（conversation）：每次进入可选择「从历史对话继续」或「新建对话」。
     会话以 JSON 落盘到 data_store/conversations/{conv_id}.json，
     包含消息列表、当前诊断状态与功能态，刷新 / 重启后可无缝恢复。
  2) 企业数据快照（enterprise snapshot）：每次完成诊断后，把核心财务指标
     写入 data_store/enterprise/{企业名}.json（按时间追加）。
  3) 跨期对比（compare）：同一企业再次上传时，自动与最近一次快照比较，
     输出关键指标变化与经营 / 资金健康度演变趋势，供智能体直接播报。

全部为纯文件 IO，无外部依赖；目录首次使用时自动创建。
"""
import os
import json
import uuid
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data_store")
CONV_DIR = os.path.join(DATA_DIR, "conversations")
ENT_DIR = os.path.join(DATA_DIR, "enterprise")


def _ensure_dirs():
    for d in (CONV_DIR, ENT_DIR):
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)


def _now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_conv_id():
    return uuid.uuid4().hex[:12]


# ======================== 会话 ========================
def list_conversations():
    """返回会话摘要列表，按更新时间倒序。"""
    _ensure_dirs()
    out = []
    if not os.path.isdir(CONV_DIR):
        return out
    for fn in os.listdir(CONV_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(CONV_DIR, fn), encoding="utf-8") as f:
                d = json.load(f)
            out.append({
                "id": d.get("id", fn[: -len(".json")]),
                "title": d.get("title", "对话"),
                "updated": d.get("updated", ""),
                "enterprise": d.get("enterprise", ""),
            })
        except Exception:
            continue
    out.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return out


def load_conversation(conv_id):
    """读取会话 JSON；不存在返回 None。"""
    _ensure_dirs()
    path = os.path.join(CONV_DIR, f"{conv_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_conversation(conv):
    """写入会话 JSON（conv 为完整字典）。"""
    _ensure_dirs()
    conv_id = conv.get("id")
    if not conv_id:
        return
    conv = dict(conv)
    conv["updated"] = _now_iso()
    path = os.path.join(CONV_DIR, f"{conv_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)


def delete_conversation(conv_id):
    path = os.path.join(CONV_DIR, f"{conv_id}.json")
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


# ======================== 企业数据快照 ========================
# 跨期跟踪的核心指标（单位已在 full_metrics 中统一为「万元 / %」）
_TRACK_KEYS = [
    ("营业收入", "万元"), ("净利润", "万元"), ("总资产", "万元"),
    ("总负债", "万元"), ("经营活动现金流净额", "万元"), ("应收账款", "万元"),
    ("存货", "万元"), ("流动比率", ""), ("资产负债率", "%"),
    ("营收增长率", "%"), ("净利润增长率", "%"), ("平均融资利率", "%"),
]


def build_snapshot(full_metrics, diagnosis_result, source_file, enterprise):
    """从诊断结果中抽取一份可跨期比较的快照。"""
    metrics = {}
    for k, unit in _TRACK_KEYS:
        v = full_metrics.get(k, "")
        if v is None or v == "":
            continue
        # 布尔（可提供抵押）等异常类型转文本
        if isinstance(v, bool):
            v = "是" if v else "否"
        metrics[k] = v
    return {
        "ts": _now_iso(),
        "source": source_file or "",
        "enterprise": enterprise or "默认企业",
        "overall_score": diagnosis_result.get("overall_score", 0),
        "dimension_scores": diagnosis_result.get("dimension_scores", {}),
        "metrics": metrics,
    }


def _ent_path(enterprise):
    _ensure_dirs()
    safe = "".join(c for c in (enterprise or "默认企业")
                   if c.isalnum() or c in ("_", "-", " ", "（", "）", "(", ")"))
    safe = safe.strip() or "默认企业"
    return os.path.join(ENT_DIR, f"{safe}.json")


def load_enterprise(enterprise):
    """返回该企业的快照列表（按时间正序）。"""
    path = _ent_path(enterprise)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_enterprise_snapshot(enterprise, snapshot):
    """
    追加保存一次企业快照，并与其上一次快照做对比。
    返回 compare 结果（含 has_prev）；首次上传时 has_prev=False。
    """
    enterprise = enterprise or "默认企业"
    path = _ent_path(enterprise)
    history = load_enterprise(enterprise)
    history.append(snapshot)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if len(history) >= 2:
        prev = history[-2]
        cur = history[-1]
        return compare_snapshots(prev, cur)
    return {"has_prev": False, "current": snapshot, "previous": None}


# ======================== 跨期对比 ========================
# 指标方向：True 表示「越大越好」，False 表示「越小越好」
_DIR = {
    "营业收入": True, "净利润": True, "总资产": True, "总负债": False,
    "经营活动现金流净额": True, "应收账款": False, "存货": False,
    "流动比率": True, "资产负债率": False, "营收增长率": True,
    "净利润增长率": True, "平均融资利率": False,
}


def _to_float(v):
    try:
        return float(str(v).replace("%", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def compare_snapshots(prev, cur):
    """比较两次快照，输出关键指标变化与演变趋势。"""
    rows = []
    better_count = 0
    worse_count = 0
    for k in _DIR:
        b = _to_float(prev.get("metrics", {}).get(k))
        a = _to_float(cur.get("metrics", {}).get(k))
        if b is None or a is None:
            continue
        delta = round(a - b, 3)
        denom = abs(b) if b != 0 else (abs(a) if a != 0 else 1)
        pct = round(delta / denom * 100, 1) if denom else 0.0
        up_is_good = _DIR[k]
        improved = (delta > 0) if up_is_good else (delta < 0)
        # 仅当变化显著（绝对值超过 0.5% 或绝对差异不为 0 的小额）计入好坏
        if abs(delta) > 1e-9:
            if improved:
                better_count += 1
            else:
                worse_count += 1
        rows.append({
            "name": k, "before": b, "after": a,
            "delta": delta, "pct": pct, "improved": improved,
        })

    score_delta = round(cur.get("overall_score", 0) - prev.get("overall_score", 0), 1)

    if score_delta > 0.3 or better_count > worse_count:
        verdict = "改善"
    elif score_delta < -0.3 or worse_count > better_count:
        verdict = "恶化"
    else:
        verdict = "持平"

    summary = (
        f"相较 {prev.get('ts', '')[:10]} 的上次数据，本次综合健康评分 "
        f"{'上升' if score_delta > 0 else '下降' if score_delta < 0 else '持平'} "
        f"{abs(score_delta)} 分，整体经营与资金健康度呈「{verdict}」态势"
        f"（{better_count} 项指标改善、{worse_count} 项走弱）。"
    )

    return {
        "has_prev": True,
        "previous": prev,
        "current": cur,
        "rows": rows,
        "score_delta": score_delta,
        "better_count": better_count,
        "worse_count": worse_count,
        "verdict": verdict,
        "summary": summary,
    }
