"""
scripts/build_research_corpus.py — 行业研究/公开报告摘要语料（零网络）

写入 6 条与小微企业融资高度相关的研究/报告摘要（每条 1 doc + 1 chunk，abstract+url）。
说明（诚实数据策略）：摘要为对公开信息的综合归纳，url 指向出版方官网（非深链）以避免失效；
正式参赛前可替换为具体报告深链。类别为 research，用于作答“行业/风险/趋势”类问题。

幂等：已存在的 doc_id 跳过。

用法：python scripts/build_research_corpus.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "knowledge", "rag_corpus", "docs.jsonl")
CHUNKS = os.path.join(ROOT, "knowledge", "rag_corpus", "chunks.jsonl")

RESEARCH = [
    {
        "id": "research_001",
        "title": "《中国小微企业金融服务报告》（中国人民银行）摘要",
        "source": "中国人民银行",
        "url": "http://www.pbc.gov.cn",
        "date": "2024-01-01",
        "tags": ["普惠小微", "首贷户", "信用贷"],
        "meta": {"类型": "监管报告摘要", "主题": "普惠小微贷款总量、结构与首贷/信用贷进展"},
        "text": (
            "人民银行小微企业金融服务报告持续跟踪普惠小微金融服务成效：普惠型小微企业贷款余额保持较快增长，"
            "贷款户数持续扩大；首贷户、信用贷款、中长期贷款、个体工商户贷款等结构性指标被纳入监管评价，"
            "引导银行“敢贷愿贷能贷会贷”。重点包括扩大首贷覆盖、提升信用贷占比、规范续贷（无还本续贷）服务，"
            "以及运用金融科技降低信息不对称、提升审批效率。"),
    },
    {
        "id": "research_002",
        "title": "世界银行《中小微企业融资缺口》（SME Finance Gap）摘要",
        "source": "World Bank",
        "url": "https://www.worldbank.org",
        "date": "2018-01-01",
        "tags": ["融资缺口", "新兴市场", "SME"],
        "meta": {"类型": "国际机构研究", "主题": "全球中小微企业融资不足规模估计"},
        "text": (
            "世界银行《中小微企业融资缺口》研究估计，新兴市场和低收入国家的中小微企业存在巨大的未满足融资需求，"
            "融资缺口以万亿美元计，女性创办企业、微型企业缺口尤为突出。研究指出，缺乏抵押品、有限信用历史、"
            "高交易成本与信息不对称是主要制约；呼吁发展数字金融、信用担保与征信基础设施以弥合缺口。"),
    },
    {
        "id": "research_003",
        "title": "北京大学数字普惠金融指数（PKU-DFI）摘要",
        "source": "北京大学数字金融研究中心",
        "url": "http://idf.pku.edu.cn",
        "date": "2023-01-01",
        "tags": ["数字普惠", "金融科技", "覆盖广度"],
        "meta": {"类型": "学术指数", "主题": "数字普惠金融的覆盖广度、使用深度与数字化程度"},
        "text": (
            "北京大学数字普惠金融指数从覆盖广度、使用深度、数字化程度三个维度测度中国数字普惠金融发展水平。"
            "研究显示，移动支付、线上信贷、互联网征信显著提升了小微与“长尾”客群的金融可及性，"
            "区域间数字鸿沟逐步收敛；数字风控（替代数据、机器学习）在扩大小微授信覆盖中作用关键，但也带来数据治理与算法公平议题。"),
    },
    {
        "id": "research_004",
        "title": "《中国区域金融运行报告》（中国人民银行）摘要",
        "source": "中国人民银行",
        "url": "http://www.pbc.gov.cn",
        "date": "2024-01-01",
        "tags": ["区域", "普惠小微", "区域差异"],
        "meta": {"类型": "监管报告摘要", "主题": "各地区普惠小微金融服务差异"},
        "text": (
            "人民银行区域金融运行报告显示，各地区普惠小微贷款增速普遍高于各项贷款平均增速，但区域间在金融基础设施、"
            "担保体系、信用环境上存在差异：东部数字普惠与直接融资更成熟，中西部更依赖银行信贷与政策性工具。"
            "报告建议结合区域产业特征，差异化配置再贷款、担保与风险补偿机制。"),
    },
    {
        "id": "research_005",
        "title": "《中国普惠金融发展报告》（行业综述）摘要",
        "source": "（行业综述，摘要无公开链接）",
        "url": "",
        "date": "2024-01-01",
        "tags": ["普惠金融", "成效", "挑战"],
        "meta": {"类型": "行业综述", "主题": "普惠金融成效与剩余挑战"},
        "text": (
            "普惠金融发展报告综述指出，我国普惠小微金融服务在覆盖面、可得性、融资成本上取得显著进展，"
            "但仍有挑战：小微企业生命周期短、财务报表不规范导致授信难；信用信息共享不充分；"
            "银行风险定价与不良容忍机制有待完善；疫情与行业周期波动下，现金流薄弱企业更易出现断贷、抽贷。"),
    },
    {
        "id": "research_006",
        "title": "小微企业融资风险与成因研究（综合）摘要",
        "source": "（综合研究，摘要无公开链接）",
        "url": "",
        "date": "2024-01-01",
        "tags": ["风险", "行业周期", "信息不对称"],
        "meta": {"类型": "风险研究", "主题": "小微企业违约/断贷风险的行业与规模异质性"},
        "text": (
            "小微企业融资风险具有显著行业周期与规模异质性。行业下行期，重资产、长账期、低毛利、强周期行业"
            "（如建材、批发零售、住宿餐饮、低端制造）现金流承压更大；叠加抵押物不足、财务不规范与银企信息不对称，"
            "违约与断贷风险上升。诊断应关注营收稳定性、经营现金流、应收账款周转、资产负债率与行业景气，"
            "对强周期、高应收、低流动性的企业提示更高风险并建议缩短授信期限、增加担保或政策工具对冲。"),
    },
]


def main():
    existing_ids = set()
    if os.path.exists(DOCS):
        with open(DOCS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line).get("id"))

    new_docs, new_chunks = [], []
    for r in RESEARCH:
        if r["id"] in existing_ids:
            print(f"[skip] {r['id']} 已存在")
            continue
        new_docs.append({
            "id": r["id"],
            "category": "research",
            "title": r["title"],
            "source": r["source"],
            "url": r["url"],
            "date": r["date"],
            "tags": r["tags"],
            "meta": r["meta"],
        })
        new_chunks.append({"doc_id": r["id"], "clause": "", "text": r["text"]})

    with open(DOCS, "a", encoding="utf-8") as f:
        for d in new_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(CHUNKS, "a", encoding="utf-8") as f:
        for c in new_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[research] appended docs={len(new_docs)} chunks={len(new_chunks)}")


if __name__ == "__main__":
    main()
