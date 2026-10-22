# -*- coding: utf-8 -*-
"""检索冒烟测试：验证 BGE-M3 + Milvus 混合检索 + 元数据过滤在项目 venv 中可用。"""
import sys
import time

from rag_qa.core.vector_store import VectorStore

QUERIES = [
    ("贵州茅台2025年营业收入和毛利率是多少", {"source_filter": "贵州茅台", "year_filter": 2025}),
    ("五粮液2025年经营情况回顾", {"source_filter": "五粮液", "year_filter": 2025}),
    ("白酒行业2025年渠道库存和动销情况", {"year_filter": 2025}),
]


def main() -> int:
    print("loading BGE-M3 ...", flush=True)
    t0 = time.time()
    vs = VectorStore()
    print(f"model ready in {time.time() - t0:.1f}s, count={vs.count()}", flush=True)

    for q, flt in QUERIES:
        t0 = time.time()
        docs = vs.hybrid_search_with_rerank(q, k=5, **flt)
        dt = time.time() - t0
        print(f"\n=== {q}   ({dt:.2f}s, {len(docs)} parent docs)", flush=True)
        for d in docs:
            m = d.metadata
            snippet = d.page_content[:70].replace("\n", " ").replace("\r", " ")
            print(f"  [{m.get('company_name')} {m.get('year')} {m.get('section')} p{m.get('page')}] {snippet}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
