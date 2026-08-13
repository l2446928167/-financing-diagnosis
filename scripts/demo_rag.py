"""
scripts/demo_rag.py — 端到端演示（不改动 app.py，证明集成契约可用）

加载 bm25_index.json -> retrieve(top5) -> grounded_answer(mock)，
输出带 [n] 编号引用的可信回答 + 参考清单。

重点演示正式接入 app.py 时的契约：
  - grounded_answer 返回 GroundedAnswer(text, citations)（也兼容 body, refs 元组解包）
  - 政策类引用带「（条款摘编）」标记
  - 索引 meta.asof 可供报告标注"检索截至 YYYY-MM-DD"
  - verify_products_freshness 可在加载时比对语料新鲜度（不一致应 st.warning）

真实接入 app.py 时只需把 mock 换成真实 LLM callable：
    def llm_callable(prompt, temperature=0):
        # 调用你的 LLM（OpenAI / 通义 / 本地），返回字符串
        ...
    ans = grounded_answer(llm_callable, q, retrieved, use_mock=False, temperature=0)
    st.write(ans.text); st.caption("；".join(ans.citations))

用法：python scripts/demo_rag.py
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

DEMO_QUERIES = [
    "科技型小微企业无抵押能贷多少",
    "银税互动对小微企业融资有什么用",
    "创业担保贷款个人最高能贷多少、贴息多少",
    "征信有不良记录还能不能贷款",
    "客户应收账款多、账期长，融资上要注意什么",
]


def main():
    idx = load_index(INDEX_PATH)
    meta = idx.get("meta", {})
    print(f"[meta] 检索截至(asof)={meta.get('asof')}  docs={meta.get('n_docs')} chunks={meta.get('n_chunks')}")

    ok, msg = verify_products_freshness(PRODUCTS_CSV, meta.get("products_csv_sha256", ""))
    print(f"[freshness] ok={ok} :: {msg}")

    for q in DEMO_QUERIES:
        retrieved = retrieve(idx, q, k=5)
        ans = grounded_answer(None, q, retrieved, use_mock=True)
        print("=" * 64)
        print(f"问：{q}")
        print("-" * 64)
        print(ans.text)
        print("\n参考：")
        for r in ans.citations:
            print("  " + r)
        print()


if __name__ == "__main__":
    main()
