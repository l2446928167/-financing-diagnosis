# 执行方案：RAG 语料收集 + BM25 索引（融资诊断项目）

## 0. 背景与目标
- 现状：`utils/vector_store.py` 是空占位文件；原方案的 FAISS + bge-small-zh + sentence-transformers 在 **Streamlit Community Cloud（1GB 内存 / 无 GPU / Python 3.14 / 仅 wheel）** 上会拉 torch（500MB+），叠加年报解析 642MB 峰值 → 必 OOM（即"OH NO"死法）。
- 目标：用**纯 Python 的 BM25（rank_bm25）+ LLM 溯源生成**替代重向量方案，把 RAG 真正落地到 `vector_store.py`，让"产品推荐/政策依据"每条可溯源，堵住幻觉质疑，并作为"真 AI"叙事的一环。
- 范围（已与用户确认）：聚焦 MVP 语料 ~40 篇；混合收集——核心国家政策抓官方全文，行业研报存摘要+出处。

## 1. 部署约束（硬性，不可违背）
- 运行期**禁止** torch / sentence-transformers / transformers / faiss-gpu。
- 允许：rank_bm25（纯 Python，可源码安装无编译）、可选 jieba（仅作增强，非必需）。
- 内存预算：RAG 部分峰值 < 50MB；分词走零依赖方案，规避 jieba 在 Py3.14 的 wheel 风险。
- 索引随仓库或构建期生成，运行期只读 load，不在云上训练/嵌入。

## 2. 语料设计（~40 篇，三类）
统一落库为 JSONL：`knowledge/rag_corpus/corpus.jsonl`，单条结构：
```
{ "id", "category": "policy|product|research", "title", "body",
  "source", "url"(可选), "date"(可选), "tags":[], "meta":{} }
```

### 2.1 监管政策（~10–12 篇，抓官方全文）
候选来源（务必用官方域：pbc.gov.cn / cbirc.gov.cn / gov.cn / samr.gov.cn）：
- 《商业银行小企业授信工作指引（试行）》
- 《银行业金融机构小微企业金融服务监管评价办法（试行）》
- 《征信业管理条例》（国务院令第631号）
- 《保障中小企业款项支付条例》（国务院令第728号）
- 普惠小微贷款支持工具 / 支小再贷款相关通知
- 银税互动相关文件
- 关于深入开展中小微企业金融服务能力提升工程的通知
- 创业担保贷款相关政策
> 策略：每项存 `title + 全文正文 + 发文机构 + 官方 url + 发文日期`，便于逐句溯源。

### 2.2 银行产品库（20 篇，由现有表生成）
- 源：`knowledge/bank_products/products.csv`（已有 20 条：建行/工行/农行/招行/平安/中行…）。
- 转换：每条 → 一篇 doc，`meta` 含 `{bank, 额度, 利率, 期限, 担保方式, 准入条件, 适合客群, 材料清单}`；`body` 为可读产品说明文本。
- 这是 RAG 中**最实用、可被产品匹配模块直接引用**的部分。

### 2.3 行业研报（5–8 篇，摘要+出处）
- 主题：小微企业融资现状、制造业/零售业/科创小微细分、区域普惠金融。
- 策略：**只存标题 + 要点摘要 + 来源机构 + 原文链接**，不抓全文 → 体积小、合规风险低，溯源指向原文。

## 3. 分词方案（零依赖优先）
- **默认：字符 bi-gram 分词**（相邻 2 字切分），纯 Python、无 jieba 依赖，规避 Py3.14 wheel 风险；对中文 BM25 召回足够。
- **增强（可选）**：若运行期 `import jieba` 成功则用 jieba.cut，否则回退 bi-gram。jieba 列为可选依赖，不阻塞部署。
- 查询与语料用同一分词器，保证一致性。

## 4. BM25 索引实现
- 库：`rank_bm25.BM25Okapi`（纯 Python）。
- 构建：`scripts/build_rag_index.py` 一次性读取 corpus.jsonl → 分词 → 训练 BM25 → 将 `(tokenized_docs, docs_meta, bm25)` pickle 到 `knowledge/rag_corpus/bm25_index.pkl`。
- 运行期：`vector_store.py` 启动时 load pickle（~40 篇 < 100ms）；提供函数：
  - `load_index()` → 载入索引与元数据
  - `tokenize(text)` → bi-gram（+可选 jieba）
  - `retrieve(query, k=5, category=None)` → 返回 `[(doc, score)]`，含 citation 字段（title/source/url）
  - `grounded_answer(llm, query, docs)` → 调 DeepSeek（复用 `utils/llm_helper.py`）做**溯源生成**
- **兜底**：若 rank_bm25 在 Py3.14 无可用 wheel，改用自实现 BM25（≤40 行纯 Python），零依赖——写入 `vector_store.py` 备选分支。

## 5. 检索 + 溯源生成（堵幻觉的关键）
- `retrieve` 取 top-k（默认 5），可按 `category` 过滤（如"只查产品"）。
- `grounded_answer` prompt 约束：① 仅基于检索段落作答；② 每条结论标注来源（doc.title / source / url）；③ 未知即说未知，禁止编造。
- 价值：to G / to B 场景直接加分，且和既有"可解释性"主线吻合。

## 6. 集成改动点（文件级）
- `utils/vector_store.py`：**实现**（当前空），填入上述 API。
- `app.py`：新增「政策/产品智能问答」区块，调用 `retrieve` + `grounded_answer`；产品推荐理由可引用检索到的产品 doc。
- `report_generator.py`：报告末尾加「政策与产品依据」小节，列出被引来源（title + source + url + 检索截至日期）。
- `requirements.txt`：加 `rank_bm25`；`jieba` 标为可选（extras / 注释）。

## 7. 执行步骤（审批通过后）
1. 建目录 `knowledge/rag_corpus/` 与 `scripts/build_rag_index.py`（采集+构建二合一）。
2. **收政策**：WebFetch 官方源 → 存全文 JSONL（核对域名为官方）。
3. **转产品**：读 `products.csv` → 生成 20 篇产品 doc。
4. **收研报**：WebSearch/WebFetch 摘要+出处 → JSONL。
5. 跑 `build_rag_index.py` → 生成 `bm25_index.pkl`（本机用可用 Python 3.12.9 运行时执行）。
6. 检索冒烟测试：如"科技型小微无抵押能贷多少"应回产品 doc；"小微授信监管评价看什么"应回政策 doc。
7. 接入 `app.py` + `report_generator.py`，本地 `streamlit run` 验证。

## 8. 风险与缓解
- rank_bm25 wheel 风险 → 已有纯 Python 自实现兜底。
- jieba wheel 风险 → 默认不依赖，bi-gram 兜底。
- 政策时效性 → 存 `date + url`，报告标注"检索截至 YYYY-MM-DD"。
- 收集准确性 → 只用官方域，存 url 供复核；研报不抓全文降合规面。
- 体积 → 全文政策仅限"关键少数"，研报仅摘要，整体 < 几 MB。

## 9. 完成标准（Definition of Done）
- [ ] `vector_store.py` 非空且实现 retrieve + grounded_answer。
- [ ] corpus.jsonl 含 ~40 篇（policy/product/research 三类齐）。
- [ ] BM25 索引构建成功，冒烟查询命中相关文档。
- [ ] `app.py` 有可用 RAG 问答区；报告能列出被引来源。
- [ ] Community Cloud 部署无 torch，RAG 内存 < ~50MB。

## 10. 后续衔接
- 本方案与 `rule-app-ai-credibility` skill 的 RAG 段一致（BM25 + 零依赖分词 + 溯源生成）。
- 双轨 ML 模型（ml_model.py）与 SHAP 归因（pred_contribs）另立任务，本方案不耦合其训练，仅共享"可解释/可溯源"主线。
