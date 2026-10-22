# -*- coding: utf-8 -*-
"""校验 Milvus 中年报 chunk 的覆盖情况（按公司/年份/章节统计）。"""
import sys
from collections import Counter

from pymilvus import MilvusClient

COL = "baijiu_annual_report_1006"
DB = "finance"


def main() -> int:
    c = MilvusClient(uri="http://localhost:19530")
    c.use_database(DB)
    try:
        c.flush(COL)
    except Exception as e:  # noqa: BLE001
        print("flush-warn:", type(e).__name__, e)
    try:
        print("stats:", c.get_collection_stats(COL))
    except Exception as e:  # noqa: BLE001
        print("stats-warn:", type(e).__name__, e)

    rows = c.query(
        COL,
        filter="",
        output_fields=["id", "company_name", "year", "section", "page"],
        limit=16384,
    )
    print("total rows:", len(rows))
    print("by year:", dict(Counter(r["year"] for r in rows)))

    comp = Counter((r["company_name"], r["year"]) for r in rows)
    print("by company/year:")
    for (name, year), n in sorted(comp.items(), key=lambda x: (x[0][1], -x[1])):
        print(f"  {year}  {name:<10s} {n:5d}")

    ids = [r["id"] for r in rows]
    print("distinct ids:", len(set(ids)), "| duplicate ids:", len(ids) - len(set(ids)))

    # 每家公司章节目录是否齐全（11 个章节）
    print("section count per company:")
    for name in sorted({r["company_name"] for r in rows}):
        secs = sorted({r["section"] for r in rows if r["company_name"] == name})
        print(f"  {name:<10s} {len(secs):2d} sections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
