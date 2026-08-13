"""
scripts/convert_products.py — 把 products.csv 转为两层 RAG 语料（产品优先，零网络）

读取 knowledge/bank_products/products.csv
  -> 写出 knowledge/rag_corpus/docs.jsonl + chunks.jsonl（"w" 覆盖，作为语料起点）
每款产品生成 1 个 doc + 1 个 chunk（产品短文本，clause 留空）。

用法：python scripts/convert_products.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "knowledge", "bank_products", "products.csv")
DOCS = os.path.join(ROOT, "knowledge", "rag_corpus", "docs.jsonl")
CHUNKS = os.path.join(ROOT, "knowledge", "rag_corpus", "chunks.jsonl")


def fmt(v):
    return (v or "").strip()


def main():
    rows = []
    with open(SRC, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    docs, chunks = [], []
    for i, r in enumerate(rows, 1):
        pid = f"product_{i:03d}"
        name = fmt(r["产品名"])
        bank = fmt(r["银行"])
        ptype = fmt(r["产品类型"])
        quota = f"{fmt(r['额度下限'])}-{fmt(r['额度上限'])}万"
        rate = f"{fmt(r['利率下限'])}-{fmt(r['利率上限'])}%"
        term = f"{fmt(r['期限下限'])}-{fmt(r['期限上限'])}个月"
        amt_th = fmt(r["营收门槛"])
        age_th = fmt(r["成立年限门槛"])
        collateral = fmt(r["抵押要求"])
        tax = fmt(r["纳税评级要求"])
        fin_up = fmt(r["融资机构数上限"])
        debt_up = fmt(r["资产负债率上限"])
        cr_up = fmt(r["流动比率下限"])
        industry = fmt(r["行业限制"])
        src = fmt(r["数据来源"])
        date = fmt(r["采集日期"])

        text = (
            f"{name} 是{bank}面向小微企业的{ptype}产品。"
            f"授信额度约{quota}；年化利率约{rate}；期限{term}。"
            f"营收门槛约{amt_th}万元；成立年限门槛约{age_th}年；抵押要求：{collateral}。"
        )
        if tax:
            text += f"纳税评级要求：{tax}及以上。"
        if fin_up and fin_up != "0":
            text += f"融资机构数不超过{fin_up}家。"
        if debt_up and debt_up != "0":
            text += f"资产负债率不高于{debt_up}%。"
        if cr_up and cr_up != "0":
            text += f"流动比率不低于{cr_up}。"
        if industry:
            text += f"优先支持{industry}行业。"
        text += f"数据来源：{src}（采集日期 {date}）。"

        docs.append({
            "id": pid,
            "category": "product",
            "title": f"{name}（{bank}）",
            "source": bank,
            "url": "",
            "date": date,
            "tags": [ptype] + ([industry] if industry else []),
            "meta": {
                "银行": bank, "产品类型": ptype, "额度": quota, "利率": rate,
                "期限": term, "营收门槛": amt_th, "成立年限门槛": age_th,
                "抵押要求": collateral, "纳税评级要求": tax,
                "融资机构数上限": fin_up, "行业限制": industry,
                "数据来源": src, "采集日期": date,
            },
        })
        chunks.append({"doc_id": pid, "clause": "", "text": text})

    with open(DOCS, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[convert_products] wrote {len(docs)} product docs/chunks")


if __name__ == "__main__":
    main()
