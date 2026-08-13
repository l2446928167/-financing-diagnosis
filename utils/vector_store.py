"""
utils/vector_store.py — 轻量 RAG 检索引擎（零外部依赖）

部署约束：Streamlit Community Cloud 1GB / 无 GPU / Python 3.14 / 仅 wheel / 禁 torch。
设计：自实现 Okapi BM25 + 自定义分词器(数字串保留 + 中文 bi-gram) + 两层语料(docs/chunks)。

审查采纳项：
- 分词：连续 [0-9A-Za-z.%]+ 保留为完整 token，中文相邻 2 字 bi-gram（建议项 3）
- BM25 自实现，零新增依赖，删除 rank_bm25（建议项 4）
- 两层结构：docs(文档元数据) + chunks(条款级)，检索返回 chunk、citation 回指 doc（必改项 1）
- retrieve 按 doc_id 去重（每 doc 至多 max_per_doc 条），采纳 doc_id 去重、暂缓模糊相似（建议项 10）
- grounded_answer：temperature=0 + [n] 编号引用 + 参考清单；无 key/异常降级纯检索；支持 mock（建议项 5/8/11）
"""
import hashlib
import json
import math
import os
import re
from collections import namedtuple

# ---------------- 结构化返回 ----------------
# grounded_answer 的返回值。向后兼容元组解包：``body, refs = grounded_answer(...)``
# 仍可用（此时 body == .text、refs == .citations），但推荐用 .text / .citations 访问。
GroundedAnswer = namedtuple("GroundedAnswer", ["text", "citations"])

# 政策类语料为对官方原文的浓缩转述（条款摘编），并非法规全文镜像；
# 在引用中显式标注「（条款摘编）」，避免被误读为逐字原文（审查必修 2）。
POLICY_EXCERPT_TAG = "（条款摘编）"

# ---------------- 分词 ----------------
_ALNUM = re.compile(r"[A-Za-z0-9.%]+")
_CJK = re.compile(r"[一-鿿]")  # CJK 统一表意文字


def _bigram(seg):
    """对一段文本抽取连续 CJK 字符，输出相邻 2 字 bi-gram。"""
    out = []
    buf = []
    for ch in seg:
        if _CJK.match(ch):
            buf.append(ch)
        else:
            if len(buf) >= 2:
                out.extend(buf[i] + buf[i + 1] for i in range(len(buf) - 1))
            buf = []
    if len(buf) >= 2:
        out.extend(buf[i] + buf[i + 1] for i in range(len(buf) - 1))
    return out


def tokenize(text):
    """数字/字母/百分号串保留为完整 token；其余中文走 bi-gram。"""
    if not text:
        return []
    tokens = []
    pos = 0
    for m in _ALNUM.finditer(text):
        tokens.extend(_bigram(text[pos:m.start()]))
        tokens.append(m.group(0).lower())
        pos = m.end()
    tokens.extend(_bigram(text[pos:]))
    return tokens


# ---------------- 自实现 Okapi BM25 ----------------
class BM25Okapi:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.avgdl = (sum(len(d) for d in corpus_tokens) / self.N) if self.N else 0.0
        df = {}
        for doc in corpus_tokens:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}
        self.corpus = corpus_tokens

    def get_scores(self, query_tokens):
        # 复杂度 ~ O(N * |query_tokens| * 平均文档长度)；当前语料（数十~数百块）完全够用。
        # 注意：此处为暴力扫描，未建倒排索引。若语料扩展到「千级 chunk」以上，
        # 应预建 term -> [doc_id,...] 的倒排表，只遍历命中文档以降复杂度（建议项 3 注释标记）。
        scores = [0.0] * self.N
        for q in query_tokens:
            qidf = self.idf.get(q)
            if qidf is None:
                continue
            for i, doc in enumerate(self.corpus):
                f = doc.count(q)
                if f == 0:
                    continue
                dl = len(doc)
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl) if self.avgdl else f
                scores[i] += qidf * f * (self.k1 + 1.0) / denom
        return scores

    def to_dict(self):
        return {
            "k1": self.k1,
            "b": self.b,
            "N": self.N,
            "avgdl": self.avgdl,
            "df": self.df,
            "corpus": self.corpus,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.k1 = d["k1"]
        obj.b = d["b"]
        obj.N = d["N"]
        obj.avgdl = d["avgdl"]
        obj.df = d["df"]
        obj.corpus = d["corpus"]
        obj.idf = {t: math.log((obj.N - f + 0.5) / (f + 0.5) + 1.0) for t, f in d["df"].items()}
        return obj


# ---------------- 语料加载 / 建索引 ----------------
def load_corpus(corpus_dir):
    """读取两层语料：docs.jsonl(文档元数据) + chunks.jsonl(条款级文本)。"""
    docs, chunks = [], []
    docs_path = os.path.join(corpus_dir, "docs.jsonl")
    chunks_path = os.path.join(corpus_dir, "chunks.jsonl")
    if os.path.exists(docs_path):
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    if os.path.exists(chunks_path):
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return docs, chunks


def build_index(docs, chunks):
    doc_map = {d["id"]: d for d in docs}
    chunk_tokens = [tokenize(c.get("text", "")) for c in chunks]
    bm25 = BM25Okapi(chunk_tokens)
    return {"bm25": bm25, "chunks": chunks, "doc_map": doc_map}


def load_index(index_path):
    with open(index_path, encoding="utf-8") as f:
        state = json.load(f)
    bm25 = BM25Okapi.from_dict(state["bm25"])
    chunks = state.get("chunks_meta", [])
    doc_map = state.get("doc_map", {})
    meta = state.get("meta", {})
    return {"bm25": bm25, "chunks": chunks, "doc_map": doc_map, "meta": meta}


def verify_products_freshness(products_csv, expected_sha256):
    """比对 products.csv 实际 sha256 与构建索引时记录的预期值（建议项 2 升级）。

    返回 (ok: bool, message: str)。集成进 Streamlit 时若 ok 为 False，
    应调用 ``st.warning(message)`` 提示语料可能过期、需重建索引。
    """
    try:
        data = open(products_csv, "rb").read()
    except OSError as e:
        return False, f"无法读取 products.csv：{e}"
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        return False, (
            f"products.csv 哈希不一致（当前 {actual[:10]}… ≠ 预期 {expected_sha256[:10]}…），"
            "语料可能已过期，请重新运行 scripts/build_rag_index.py 重建索引"
        )
    return True, "语料新鲜度校验通过"


# ---------------- 检索 ----------------
def retrieve(index, query, k=5, category=None, max_per_doc=2):
    """返回 top-k 结果（按 doc_id 去重）。每条含文档元数据 + 条款号 + 文本 + 分数。"""
    bm25 = index["bm25"]
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    per_doc = {}
    for i in ranked:
        if scores[i] <= 0:
            break
        chunk = index["chunks"][i]
        doc_id = chunk.get("doc_id")
        doc = index["doc_map"].get(doc_id, {})
        if category and doc.get("category") != category:
            continue
        if per_doc.get(doc_id, 0) >= max_per_doc:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        results.append({
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "url": doc.get("url", ""),
            "date": doc.get("date", ""),
            "category": doc.get("category", ""),
            "clause": chunk.get("clause", ""),
            "text": chunk.get("text", ""),
            "score": round(scores[i], 4),
        })
        if len(results) >= k:
            break
    return results


# ---------------- 溯源生成（grounded_answer） ----------------
def grounded_answer(llm_callable, query, retrieved, use_mock=False, temperature=0):
    """仅基于检索结果生成带 [n] 编号引用的回答，返回 ``GroundedAnswer(text, citations)``。

    返回结构向后兼容元组解包：``body, refs = grounded_answer(...)`` 仍可用，
    其中 ``body == .text``、``refs == .citations``。

    - use_mock=True 或 llm_callable=None 或调用异常 → 降级为纯检索展示（建议项 8/11）
    - 正常时调用 llm_callable(prompt, temperature=...) 并要求 [n] 引用（建议项 5）
    - 政策类结果在引用中标注「（条款摘编）」，表明来自官方出处的浓缩转述而非原文镜像（必修 2）
    """
    if not retrieved:
        return GroundedAnswer("(无相关语料)", [])
    refs = []
    ctx = []
    for i, r in enumerate(retrieved, 1):
        clause_tag = (" " + r["clause"]) if r.get("clause") else ""
        tag = f"[{i}] {r['title']}（{r['source']}）"
        if r.get("category") == "policy":
            tag += POLICY_EXCERPT_TAG
        if clause_tag:
            tag += clause_tag
        if r.get("url"):
            tag += f" — {r['url']}"
        refs.append(tag)
        ctx.append(f"[{i}] {r['title']}{clause_tag}\n{r['text']}")

    if use_mock or llm_callable is None:
        body = "\n\n".join(
            f"【{i + 1}】{r['title']}（{r['source']}）\n{r['text'][:300]}"
            for i, r in enumerate(retrieved)
        )
        return GroundedAnswer(body, refs)

    prompt = (
        "你是小微企业融资政策/产品问答助手。仅基于以下检索到的资料回答，"
        "每条结论用 [n] 标注来源编号，未知就说未知，禁止编造。\n\n"
        + "\n\n".join(ctx)
        + f"\n\n问题：{query}\n回答："
    )
    try:
        answer = llm_callable(prompt, temperature=temperature)
    except Exception:
        body = "\n\n".join(
            f"【{i + 1}】{r['title']}（{r['source']}）\n{r['text'][:300]}"
            for i, r in enumerate(retrieved)
        )
        return GroundedAnswer(body, refs)
    return GroundedAnswer(answer, refs)
