"""
scripts/smoke_rag.py — RAG 回归冒烟（建议项 7/8/11）

固定 8 组查询，按类别校验 top-5 命中（recall@5 级别）；
并跑 mock LLM 降级，验证无 key 时仍能返回可读结果。
每次重建索引后应跑一遍，防止分词/索引回归。

用法：python scripts/smoke_rag.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.vector_store import (  # noqa: E402
    load_index, retrieve, grounded_answer, verify_products_freshness,
)

INDEX_PATH = os.path.join(ROOT, "knowledge", "rag_corpus", "bm25_index.json")
PRODUCTS_CSV = os.path.join(ROOT, "knowledge", "bank_products", "products.csv")

# (查询, 期望命中的类别) —— 类别级召回校验，语料定稿后可收紧为具体 doc_id
CASES = [
    ("科技型小微企业无抵押能贷多少", ["product"]),
    ("小微企业授信监管评价办法看什么", ["policy"]),
    ("征信不良还能贷款吗", ["policy"]),
    ("普惠金融定向降准", ["policy"]),
    ("创业担保贷款怎么申请", ["policy"]),
    ("应收账款质押融资有哪些产品", ["product"]),
    ("保障中小企业款项支付条例", ["policy"]),
    ("行业下行期哪些小微企业风险高", ["policy", "research"]),
]


def main():
    if not os.path.exists(INDEX_PATH):
        print(f"[smoke] ERROR: 索引不存在 {INDEX_PATH}，请先运行 build_rag_index.py")
        sys.exit(1)
    idx = load_index(INDEX_PATH)

    all_pass = True
    for q, expect_cats in CASES:
        res = retrieve(idx, q, k=5)
        hit = any(r["category"] in expect_cats for r in res)
        top = res[0]["title"] if res else "(空)"
        top_cats = [r["category"] for r in res[:3]]
        status = "PASS" if hit else "FAIL"
        if not hit:
            all_pass = False
        print(f"[{status}] {q} -> top={top!r} cats={top_cats}")

    # mock 降级测试（无 key 也能跑）+ 结构化返回契约校验
    ans = grounded_answer(
        None, "无抵押信用贷有哪些", retrieve(idx, "无抵押信用贷", k=3), use_mock=True
    )
    assert hasattr(ans, "text") and hasattr(ans, "citations"), "grounded_answer 应返回 GroundedAnswer"
    print(f"[mock] text_len={len(ans.text)} citations={len(ans.citations)}")

    # 新鲜度校验（加载时比对 products.csv 指纹，集成阶段供 st.warning 使用）
    ok, msg = verify_products_freshness(PRODUCTS_CSV, idx.get("meta", {}).get("products_csv_sha256", ""))
    print(f"[freshness] ok={ok} :: {msg}")

    print("RESULT:", "ALL PASS" if all_pass else "SOME FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
