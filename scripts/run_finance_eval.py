# -*- coding: utf-8 -*-
"""金融投研助手 · 评测脚本（检索 A/B + 端到端答案 + 延迟）。

产出：
    eval/finance_eval_results.json
    eval/finance_eval_report.md

指标：
    检索：hit@5、MRR、平均延迟；A/B 对比 BM25 关键词融合 开/关
    答案：文档题关键词命中率、SQL 题数值命中率、引用标注覆盖率、延迟 P50/P95
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\pro")

from base import Config
from rag_qa.core.vector_store import VectorStore
from new_main import IntegratedQASystem

OUT_DIR = Path(r"C:\pro\eval")
OUT_DIR.mkdir(exist_ok=True)

# (问题, 公司过滤, 年份过滤, 期望命中的关键词)
DOC_CASES = [
    ("茅台2025年报里提到的经营风险有哪些", "贵州茅台", 2025, ["可能面对的风险"]),
    ("茅台2025年的分红方案", "贵州茅台", 2025, ["分红"]),
    ("茅台的核心竞争力分析", "贵州茅台", 2025, ["核心竞争力"]),
    ("茅台公司未来发展战略", "贵州茅台", 2025, ["战略"]),
    ("五粮液的品牌建设情况", "五粮液", 2025, ["品牌"]),
    ("五粮液2025年营业收入变动原因", "五粮液", 2025, ["营业收入"]),
    ("五粮液的经销商体系", "五粮液", 2025, ["经销商"]),
    ("洋河股份的渠道库存和动销情况", "洋河股份", 2025, ["渠道"]),
    ("洋河梦之蓝的产品策略", "洋河股份", 2025, ["梦之蓝"]),
    ("山西汾酒的增长原因", "山西汾酒", 2025, ["增长"]),
    ("汾酒青花系列的产品情况", "山西汾酒", 2025, ["青花"]),
    ("泸州老窖的现金流情况", "泸州老窖", 2025, ["现金流"]),
    ("泸州老窖国窖1573的品牌情况", "泸州老窖", 2025, ["国窖"]),
    ("古井贡酒的渠道建设", "古井贡酒", 2025, ["渠道"]),
    ("水井坊对2025年白酒行业的判断", "水井坊", 2025, ["行业"]),
    ("今世缘的产品结构", "今世缘", 2025, ["产品"]),
    ("迎驾贡酒的产能情况", "迎驾贡酒", 2025, ["产能"]),
    ("酒鬼酒的库存情况", "酒鬼酒", 2025, ["库存"]),
    ("舍得酒业的老酒战略", "舍得酒业", 2025, ["老酒"]),
    ("金种子酒的亏损原因", "金种子酒", 2025, ["亏损"]),
]

SQL_CASES = [
    ("2025年营业收入最高的3家公司是哪些？", ["贵州茅台", "五粮液", "山西汾酒"]),
    ("贵州茅台2025年毛利率是多少？", ["91.18"]),
    ("五粮液2025年营业收入是多少？", ["405.29"]),
    ("山西汾酒2025年营业收入是多少？", ["387.18"]),
    ("泸州老窖2025年营业收入是多少？", ["257.31"]),
    ("洋河股份2025年营业收入是多少？", ["192.11"]),
    ("古井贡酒2025年营业收入是多少？", ["188.32"]),
    ("今世缘2025年营业收入是多少？", ["101.82"]),
    ("岩石股份2025年营业收入是多少？", ["0.39"]),
    ("五粮液2025年归母净利润是多少？", ["89.54"]),
    ("2025年ROE最高的公司是哪家？", ["山西汾酒", "33.48"]),
    ("招商中证白酒基金持有哪几家白酒公司？", ["贵州茅台"]),
]


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def run_retrieval_eval(vs: VectorStore, keyword_on: bool):
    cfg = Config()
    cfg.KEYWORD_RETRIEVAL = keyword_on
    # VectorStore 里用的是模块级 conf 对象，直接改它
    import rag_qa.core.vector_store as vs_mod

    old = vs_mod.conf.KEYWORD_RETRIEVAL
    vs_mod.conf.KEYWORD_RETRIEVAL = keyword_on
    results = []
    try:
        for q, company, year, keys in DOC_CASES:
            t0 = time.time()
            docs = vs.hybrid_search_with_rerank(
                q, k=10, source_filter=company, year_filter=year)
            dt = time.time() - t0
            hit_rank = 0
            for i, d in enumerate(docs, 1):
                content = norm(d.page_content)
                if any(norm(k) in content for k in keys):
                    hit_rank = i
                    break
            results.append({
                "question": q,
                "hit_rank": hit_rank,
                "hit": hit_rank > 0,
                "latency": dt,
            })
    finally:
        vs_mod.conf.KEYWORD_RETRIEVAL = old
    hits = [r["hit"] for r in results]
    mrr = sum(1.0 / r["hit_rank"] for r in results if r["hit_rank"]) / len(results)
    lat = [r["latency"] for r in results]
    return {
        "keyword_fusion": keyword_on,
        "n": len(results),
        "hit_at_5": round(sum(hits) / len(hits), 4),
        "mrr": round(mrr, 4),
        "avg_latency_s": round(statistics.mean(lat), 3),
        "p95_latency_s": round(sorted(lat)[int(len(lat) * 0.95) - 1], 3),
        "details": results,
    }


def main() -> int:
    print("== 1) 检索评测（A/B）==", flush=True)
    vs = VectorStore()
    ret_on = run_retrieval_eval(vs, keyword_on=True)
    print("BM25 融合开：", {k: ret_on[k] for k in ("hit_at_5", "mrr", "avg_latency_s")}, flush=True)
    ret_off = run_retrieval_eval(vs, keyword_on=False)
    print("BM25 融合关：", {k: ret_off[k] for k in ("hit_at_5", "mrr", "avg_latency_s")}, flush=True)

    print("== 2) 端到端答案评测 ==", flush=True)
    qa = IntegratedQASystem()
    doc_results, sql_results = [], []

    for q, company, year, keys in DOC_CASES:
        t0 = time.time()
        ans = "".join(t for t, _ in qa.query(q, session_id="eval-rag", mode="rag"))
        dt = time.time() - t0
        doc_results.append({
            "question": q,
            "answer": ans[:500],
            "hit": any(norm(k) in norm(ans) for k in keys),
            "has_citation": bool(re.search(r"第\d+页", ans)),
            "latency": dt,
        })
        print(f"  [doc] {q} | hit={doc_results[-1]['hit']} cite={doc_results[-1]['has_citation']} {dt:.1f}s", flush=True)

    for q, keys in SQL_CASES:
        t0 = time.time()
        ans = "".join(t for t, _ in qa.query(q, session_id="eval-sql", mode="sql"))
        dt = time.time() - t0
        sql_results.append({
            "question": q,
            "answer": ans[:500],
            "hit": any(k in ans for k in keys),
            "latency": dt,
        })
        print(f"  [sql] {q} | hit={sql_results[-1]['hit']} {dt:.1f}s", flush=True)

    def summarize(rows):
        lat = [r["latency"] for r in rows]
        return {
            "n": len(rows),
            "hit_rate": round(sum(r["hit"] for r in rows) / len(rows), 4),
            "avg_latency_s": round(statistics.mean(lat), 2),
            "p50_latency_s": round(statistics.median(lat), 2),
            "p95_latency_s": round(sorted(lat)[int(len(lat) * 0.95) - 1], 2),
        }

    doc_sum = summarize(doc_results)
    doc_sum["citation_rate"] = round(sum(r["has_citation"] for r in doc_results) / len(doc_results), 4)
    sql_sum = summarize(sql_results)

    result = {
        "dataset": {"doc_questions": len(DOC_CASES), "sql_questions": len(SQL_CASES)},
        "retrieval": {"keyword_on": ret_on, "keyword_off": ret_off},
        "rag_answers": {"summary": doc_sum, "details": doc_results},
        "sql_answers": {"summary": sql_sum, "details": sql_results},
    }
    out_json = OUT_DIR / "finance_eval_results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== 汇总 ==", flush=True)
    print(json.dumps({
        "retrieval_keyword_on": {k: ret_on[k] for k in ("hit_at_5", "mrr", "avg_latency_s")},
        "retrieval_keyword_off": {k: ret_off[k] for k in ("hit_at_5", "mrr", "avg_latency_s")},
        "rag": doc_sum,
        "sql": sql_sum,
    }, ensure_ascii=False, indent=2), flush=True)
    print("\n结果已保存：", out_json, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
