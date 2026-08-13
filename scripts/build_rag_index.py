"""
scripts/build_rag_index.py — 构建 RAG 检索引擎（采集结果已落在 corpus 文件后调用）

读取 knowledge/rag_corpus/{docs.jsonl, chunks.jsonl}
  -> 分词 + 自实现 BM25 建索引
  -> 写出 knowledge/rag_corpus/bm25_index.json
  -> 打印语料统计与 products.csv 新鲜度指纹（建议项 9）

用法：python scripts/build_rag_index.py
"""
import hashlib
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.vector_store import load_corpus, build_index  # noqa: E402

CORPUS_DIR = os.path.join(ROOT, "knowledge", "rag_corpus")
INDEX_PATH = os.path.join(CORPUS_DIR, "bm25_index.json")
PRODUCTS_CSV = os.path.join(ROOT, "knowledge", "bank_products", "products.csv")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8192), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    docs, chunks = load_corpus(CORPUS_DIR)
    print(f"[build] docs={len(docs)} chunks={len(chunks)}")
    if not chunks:
        print("[build] ERROR: chunks.jsonl 为空，请先采集语料")
        sys.exit(1)

    index = build_index(docs, chunks)
    # 索引元数据：构建日期 + 语料新鲜度指纹（建议项 1 + 建议项 2 升级）
    products_sha = sha256_of(PRODUCTS_CSV) if os.path.exists(PRODUCTS_CSV) else None
    meta = {
        "asof": date.today().isoformat(),            # 检索截至日期，便于报告标注"检索截至 YYYY-MM-DD"
        "products_csv_sha256": products_sha,         # 加载时与当前 products.csv 比对，不一致提示重建
        "n_docs": len(docs),
        "n_chunks": len(chunks),
    }
    state = {
        "bm25": index["bm25"].to_dict(),
        "chunks_meta": [
            {"doc_id": c.get("doc_id"), "clause": c.get("clause", ""), "text": c.get("text", "")}
            for c in index["chunks"]
        ],
        "doc_map": index["doc_map"],
        "meta": meta,
    }
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    print(f"[build] index -> {INDEX_PATH} (asof={meta['asof']})")

    # 新鲜度校验：products.csv 指纹（建议项 9）
    if products_sha:
        print(f"[freshness] products.csv sha256={products_sha[:16]}")

    # 类别分布
    from collections import Counter
    cat = Counter(d.get("category", "?") for d in docs)
    print(f"[build] category_dist={dict(cat)}")


if __name__ == "__main__":
    main()
