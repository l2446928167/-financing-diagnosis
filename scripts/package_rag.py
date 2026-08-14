#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RAG 模块全部交付物打包成一个 zip（零依赖，仅用标准库）。

用法: python3 scripts/package_rag.py
输出: <工作区>/rag_deliverable.zip
"""
import zipfile
from pathlib import Path

WS = Path("/storage/Users/currentUser/WorkBuddy/创")
PLAN = Path("/storage/Users/currentUser/.workbuddy/plans/quantum-pulse-tesla-yiPwn-R3.md")
OUT = WS / "rag_deliverable.zip"

# (源绝对路径, zip 内相对路径)
ENTRIES = [
    (WS / "utils/vector_store.py",
     "rag_deliverable/utils/vector_store.py"),
    (WS / "scripts/build_rag_index.py",
     "rag_deliverable/scripts/build_rag_index.py"),
    (WS / "scripts/smoke_rag.py",
     "rag_deliverable/scripts/smoke_rag.py"),
    (WS / "scripts/convert_products.py",
     "rag_deliverable/scripts/convert_products.py"),
    (WS / "scripts/build_policy_corpus.py",
     "rag_deliverable/scripts/build_policy_corpus.py"),
    (WS / "scripts/build_research_corpus.py",
     "rag_deliverable/scripts/build_research_corpus.py"),
    (WS / "scripts/demo_rag.py",
     "rag_deliverable/scripts/demo_rag.py"),
    (WS / "knowledge/bank_products/products.csv",
     "rag_deliverable/knowledge/bank_products/products.csv"),
    (WS / "knowledge/rag_corpus/docs.jsonl",
     "rag_deliverable/knowledge/rag_corpus/docs.jsonl"),
    (WS / "knowledge/rag_corpus/chunks.jsonl",
     "rag_deliverable/knowledge/rag_corpus/chunks.jsonl"),
    (WS / "knowledge/rag_corpus/bm25_index.json",
     "rag_deliverable/knowledge/rag_corpus/bm25_index.json"),
    (WS / "RAG方案审查回应.md",
     "rag_deliverable/docs/RAG方案审查回应.md"),
    (WS / "RAG模块整合修改说明.md",
     "rag_deliverable/docs/RAG模块整合修改说明.md"),
    (WS / "requirements.txt",
     "rag_deliverable/requirements.txt"),
    (PLAN, "rag_deliverable/docs/RAG执行计划.md"),
]

README = """# 融资诊断 RAG 模块交付物（v0.2 整合版）

为「AI + 小微企业融资诊断」Streamlit 应用提供**零依赖、可部署于 Streamlit Community Cloud**
的检索增强生成（RAG）能力。本版本已落实协作方《RAG 交付审查：整合前必修清单》的 **3 项必修 + 5 条建议**。

## 本次修改要点（相对 v0.1）
- **集成契约结构化（必修 1）**：`grounded_answer()` 返回 `GroundedAnswer(text, citations)`，
  同时向后兼容旧式 `body, refs = grounded_answer(...)` 元组解包。README 示例已对齐真实签名。
- **政策语料诚实标注（必修 2）**：政策类引用统一加「（条款摘编）」标记；`docs.jsonl` 政策条目含
  `body` 声明"非法规全文镜像"。关键数字已逐条核对官方原文（详见 docs/RAG模块整合修改说明.md）。
- **语料 URL 修正（必修 3）**：银税互动换为税务总局政策法规库页面
  `fgk.chinatax.gov.cn/.../c5248694/content.html`；两篇研报 url 置空并标注"（摘要，无公开链接）"。
- **索引元数据 + 新鲜度（建议 1/2）**：`bm25_index.json` 写入 `meta.asof`（检索截至日期）与
  `meta.products_csv_sha256`（加载时比对，不一致应提示重建索引）。
- **BM25 倒排表注释（建议 3）**：`get_scores` 标注千级语料需改倒排表。
- **BOM 统一（建议 4）**：`products.csv` 去除 BOM，避免合并时无意义 diff。
- **OOV 话术（建议 5）**：README「诚实性声明」给出超纲问题应答口径。

## 设计要点
- 自实现 BM25 Okapi，**零重型依赖**（无 torch / sentence-transformers / FAISS），规避 1GB 内存 OOM。
- 两层语料：`docs.jsonl`（文档元数据）+ `chunks.jsonl`（条款级文本）；检索返回 chunk，引用指向 doc。
- `grounded_answer`：带 `[n]` 编号引用、`temperature=0`、无 API Key 时 `mock` 降级，绝不编造。
- 中文分词：英文/数字/百分号作为完整 token，中文按二元组（bi-gram）。
- 事实性来源：cbirc→nfra；金规〔2024〕18号、税总纳服发〔2026〕19号、国务院令631/802号、央行令〔2021〕4号 等官方口径。

## 诚实性声明（评委问答口径）
- **政策语料为条款摘编，非法规全文镜像**：chunk 是对官方原文的浓缩转述，引用均标注「（条款摘编）」，
  请以官网原文为准。关键数字已逐条核对官方来源，但实际作答仍建议回链官方页面。
- **OOV（超出语料范围）是正确行为**：问到语料未收录的内容时，检索为空、`grounded_answer` 返回
  "（无相关语料）"，这是诚实行为而非缺陷。应答话术：引导到产品查询，或坦承"该问题暂未收录于本模块语料"；
  请勿为此类情形编造引用。

## 目录结构
```
rag_deliverable/
├── README.md
├── requirements.txt             # 零运行时依赖声明（仅 Python 3.8+ 标准库）
├── docs/
│   ├── RAG执行计划.md
│   ├── RAG方案审查回应.md
│   └── RAG模块整合修改说明.md   # 协作方反馈逐条评估与本次修改记录
├── utils/
│   └── vector_store.py          # 核心：tokenize / BM25Okapi / retrieve / grounded_answer
├── scripts/
│   ├── build_rag_index.py       # 读语料 → 构建并写出 bm25_index.json（含 meta）
│   ├── smoke_rag.py             # 8 个固定回归用例（recall@5 + mock + 新鲜度）
│   ├── convert_products.py      # products.csv → 产品语料
│   ├── build_policy_corpus.py   # 7 部政策 → 条款级 chunk（含摘编声明）
│   ├── build_research_corpus.py # 6 条研报摘要
│   └── demo_rag.py              # 端到端演示（load → retrieve → grounded_answer）
└── knowledge/
    ├── bank_products/products.csv   # 已去除 BOM
    └── rag_corpus/
        ├── docs.jsonl           # 33 篇（20 产品 + 7 政策 + 6 研报）
        ├── chunks.jsonl         # 66 块（政策按条/款拆分）
        └── bm25_index.json      # 重建的索引（含 meta.asof / products_csv_sha256）
```

## 快速验证
```bash
cd rag_deliverable
python3 scripts/smoke_rag.py    # 期望 8/8 PASS（含 mock 降级 + 新鲜度校验）
python3 scripts/demo_rag.py     # 端到端可信回答（带 [n] 引用，政策标「条款摘编」）
```

## 重建索引（语料更新后）
```bash
python3 scripts/build_rag_index.py
```

## 集成契约（正式接入 app.py 时）
```python
import sys, os
sys.path.insert(0, "utils")
from vector_store import load_index, retrieve, grounded_answer, verify_products_freshness

index = load_index("knowledge/rag_corpus/bm25_index.json")
meta = index["meta"]
print("检索截至", meta["asof"])  # 报告可标注"检索截至 YYYY-MM-DD"

# 加载时新鲜度校验：不一致应 st.warning(...) 提示重建索引
ok, msg = verify_products_freshness("knowledge/bank_products/products.csv", meta["products_csv_sha256"])
if not ok:
    st.warning(msg)

def my_llm(prompt, temperature=0):     # 替换为真实 LLM callable（OpenAI/通义/本地）
    return "<你的模型返回>"

hits = retrieve(index, "科技型小微企业无抵押能贷多少")
ans = grounded_answer(my_llm, "科技型小微企业无抵押能贷多少", hits)
st.write(ans.text)                      # 含 [n] 引用的回答
st.caption("；".join(ans.citations))    # 引用清单（政策类带「条款摘编」）
```
> 当前 app.py 处于冻结期（审查 Item14），本交付为"模块态"——仅新增文件、不改现有 app.py；
> 与 ML 双轨模块的正式接线留待统一接线阶段一次完成，避免交叉覆盖。
"""

missing = []
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("rag_deliverable/README.md", README)
    for src, arc in ENTRIES:
        if src.exists():
            z.write(src, arc)
        else:
            missing.append(str(src))

print(f"WROTE  : {OUT}")
print(f"SIZE   : {OUT.stat().st_size:,} bytes")
print(f"ENTRIES: {len(z.namelist())}")
if missing:
    print("MISSING (not packed):")
    for m in missing:
        print("  -", m)
else:
    print("ALL ENTRIES PACKED OK")
