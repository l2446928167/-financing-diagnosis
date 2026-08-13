# RAG 模块整合修改说明（v0.2）

**背景**：协作方对 v0.1 RAG 交付物给出「整合前必修清单（3 项）+ 建议项（5 条）」。
本文档逐条评估合理性、记录实际改动与验证方式，并说明未采纳项及原因。

> 范围说明：本交付为**模块态**。按既定纪律（审查 Item14：app.py 冻结、只新增文件不改现有代码），
> 以及协作方「统一接线阶段一次完成」的建议，**app.py 正式接线不在本次范围**，RAG 模块已做到 drop-in 可用。
> v0.1 中已逐项核对、落实的 14 条审查意见（含 cbirc→nfra 等事实性修正）不在本文重复。

---

## 一、必修项处理

### 必修 1：README 集成契约与实际返回不符
- **合理性**：✅ 真实。`grounded_answer()` 原返回 `(body, refs)` 元组，而 README 写 `answer.text` / `answer.citations`，照抄会 `AttributeError`（demo/smoke 当时用的是元组解包，故能跑，但契约文档误导）。
- **处理（采纳推荐方案 a）**：返回值改为 `namedtuple GroundedAnswer(text, citations)`。
  - 向后兼容：`body, refs = grounded_answer(...)` 仍可用（`body==.text`、`refs==.citations`）。
  - `demo_rag.py` / `smoke_rag.py` 改为示范 `.text` / `.citations` 访问；`smoke_rag.py` 增加结构化返回断言。
  - README 集成契约示例同步修正为真实签名 `llm_callable(prompt, temperature=0)` 与 `.text` / `.citations`。
- **验证**：`smoke_rag.py` 通过 `assert hasattr(ans,"text") and hasattr(ans,"citations")`；`demo_rag.py` 输出正常。

### 必修 2：政策语料是条款摘编，引用须标注且数字须抽查
- **合理性**：✅ 真实且必要。chunk 为"零网络"下对官方原文的浓缩转述，引用格式却像逐字原文，有误导风险；且含具体数字，错一处比没有更被动。
- **处理（全部采纳）**：
  1. 引用格式加摘编标记：`grounded_answer` 对 `category=="policy"` 的引用追加「（条款摘编）」。demo 实测输出如
     `[3] …（国家金融监督管理总局）（条款摘编） 评价指标表-信贷投放 — https://…`。
  2. `docs.jsonl` 政策条目加 `body` 字段声明"政策类语料为条款摘编，非法规全文镜像"；README/本文档均明示。
  3. **关键数字逐条抽查原文**（见下表），结论：协作方点名的数字在 v0.1 中已全部正确，无需改动；本次仅补强声明。
  4. 答辩/说明材料声明"语料为条款摘编+官方出处，非全文镜像"（README「诚实性声明」段）。
- **数字抽查结果（均核对官方来源，与原文一致）**：

  | 政策 | 抽查点 | 结论 |
  |---|---|---|
  | 金规〔2024〕18号 | 六要素构成；常规指标满分100分；≥90 一级 / [75,90) 二级 / [60,75) 三级；普惠型小微贷款 **15分**、占比 **8分**、户数 **4分**；贷款成本5分/资产质量5分 | ✅ 与金融监管总局官网《评价指标表》一致 |
  | 银税互动 税总纳服发〔2026〕19号 | 发文日期 **2026-03-27** | ✅ 税务总局政策法规库、多地税务官网一致 |
  | 保障中小企业款项支付条例 | **国务院令802号**、自 **2025-06-01** 施行 | ✅ 中国政府网（国令第802号）一致 |
  | 支小再贷款 | 2024年4月设科技创新和技术改造再贷款、**额度5000亿元**、年末签约**超9000亿元** | ✅ 央行/货币政策执行报告一致 |

### 必修 3：三处 URL 修正
- **合理性**：✅ 真实。
- **处理（全部采纳）**：
  1. 银税互动通知：原 `shanghai.chinatax.gov.cn/gate/big5/...`（繁体网关，易死链）→ 换为
     `https://fgk.chinatax.gov.cn/zcfgk/c102424/c5248694/content.html`。
     **已实测**：该 URL 返回完整原文（标题、税总纳服发〔2026〕19号、成文日期 2026-03-27 齐全）。
  2. 《中国普惠金融发展报告》（research_005）：原 `source="行业研究机构综述"` 但 `url=人民银行官网` →
     来源与链接不匹配。改为 `url=""` + `source="（行业综述，摘要无公开链接）"`。
  3. 小微企业融资风险与成因研究（research_006）：原 `url=https://www.gov.cn`（泛化首页，不可作引用）→
     改为 `url=""` + `source="（综合研究，摘要无公开链接）"`。
- **说明**：research_001/004（来源"中国人民银行"、url=pbc.gov.cn）来源与链接一致，保留；research_002/003 同理保留。

---

## 二、建议项处理

| # | 建议 | 合理性 | 处理 |
|---|---|---|---|
| 1 | `bm25_index.json` 加 `meta.asof`（索引构建日期） | ✅ | 采纳。`build_rag_index.py` 写入 `meta.asof`（运行日 ISO 日期）+ `n_docs`/`n_chunks`；`load_index` 透传 `meta`。报告可标注"检索截至 YYYY-MM-DD"。 |
| 2 | products.csv 新鲜度哈希从"build 时 print"升级为"加载时比对" | ✅ | 采纳。新增 `verify_products_freshness(products_csv, expected_sha256)`；索引存 `meta.products_csv_sha256`；demo/smoke 均已演示，集成时不一致应 `st.warning`。 |
| 3 | `BM25Okapi.get_scores` 暴力扫描在千级语料换倒排表 | ✅ | 采纳（注释标记）。当前语料数十~数百块，暴力扫描足够；已在 `get_scores` 加复杂度说明与倒排表改造提示。 |
| 4 | 包内 products.csv 带 BOM、仓库版没有，合并时统一 | ✅ | 采纳。实测 `products.csv` 确带 `ef bb bf` BOM，已去除；`convert_products.py` 用 `utf-8-sig` 读取，BOM 有无均兼容，避免无意义 diff。 |
| 5 | 准备"评委问到语料外问题"的口径 | ✅ | 采纳。README 新增「诚实性声明」段：OOV 时检索为空、返回"（无相关语料）"是正确行为；话术引导到产品查询或坦承未收录，禁止编造。 |

---

## 三、未采纳项及原因

无。3 项必修 + 5 条建议全部合理并予以采纳。
唯一**有意推迟**的动作是 **app.py 正式接线**：按审查 Item14 冻结纪律与协作方"统一接线阶段一次完成"的建议，
本次仅交付集成就绪的 RAG 模块（含契约示例与新鲜度校验接口），与 ML 双轨模块的接线合并到下一阶段，避免对 app.py 反复交叉覆盖。

---

## 四、本次交付物清单（rag_deliverable.zip）

- 代码：`utils/vector_store.py`（结构化返回+摘编标记+新鲜度校验）、`scripts/*`（build/convert/smoke/demo 全部更新）
- 语料：`knowledge/rag_corpus/{docs.jsonl(33), chunks.jsonl(66), bm25_index.json(含 meta)}`
- 产品：`knowledge/bank_products/products.csv`（已去 BOM）
- 文档：`README.md`（v0.2）、`docs/RAG模块整合修改说明.md`（本文）、`docs/RAG执行计划.md`、`docs/RAG方案审查回应.md`
- 依赖：`requirements.txt`（零运行时依赖声明）

**验证状态**：`smoke_rag.py` 8/8 PASS（含 mock 降级 + 新鲜度校验）；`demo_rag.py` 端到端可信回答带 `[n]` 引用、政策标「条款摘编」；`bm25_index.json` 含 `meta.asof=2026-08-13` 与 `products_csv_sha256`。

---

## 五、下一步

1. 与 ML 双轨模块（modules/ml_model.py、scripts/train_ml.py）进入**统一接线阶段**：
   app.py 问答区 + ML 双轨展示区 + report_generator「政策与产品依据」小节，一次完成。
2. 正式接入时把 `grounded_answer` 的 `use_mock=True` 换为真实 LLM callable，并接 `verify_products_freshness` 到 `st.warning`。
3. 参赛前如需更强召回，可按建议项 3 将 `get_scores` 升级为倒排表（语料上规模时）。
