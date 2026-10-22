# -*- coding: utf-8 -*-
"""把年报子块补进入 Milvus（支持断点续传 / 按公司 / 跳过财务报表章节）。

为什么需要：init_data.py --milvus 是全量重跑，中断后只能从头再来；
本脚本先查 Milvus 里已入库的 chunk id，只补缺失的部分，可反复执行。

用法：
  python scripts/embed_annual_reports.py                      # 全部公司，续传
  python scripts/embed_annual_reports.py --codes 600519,000858
  python scripts/embed_annual_reports.py --skip-financial      # 跳过财务报表章节
  python scripts/embed_annual_reports.py --dry-run             # 只统计缺口，不入库
"""
import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base import Config, logger  # noqa: E402


def load_parents(processed_dir: str) -> dict:
    parents = {}
    path = os.path.join(processed_dir, "parents.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            parents[obj["id"]] = obj["text"][:60000]
    return parents


# 年报章节的投研价值排序：管理层讨论与分析最有用，财务报告已结构化进 MySQL、价值最低
SECTION_PRIORITY = [
    "第三节",   # 管理层讨论与分析：业务、渠道、产能、风险
    "第二节",   # 公司简介和主要财务指标
    "第六节",   # 重要事项
    "第七节",   # 股份变动及股东情况
    "第五节",   # 环境和社会责任
    "第四节",   # 公司治理
    "第一节",   # 重要提示
    "第八节",   # 优先股
    "第九节",   # 债券
    "前置",     # 封面/目录
    "第十节",   # 财务报告
]


def section_rank(section: str) -> int:
    sec = str(section or "")
    for idx, key in enumerate(SECTION_PRIORITY):
        if sec.startswith(key):
            return idx
    return len(SECTION_PRIORITY)


def existing_stats(client, collection: str, codes: set[str]) -> tuple[set[str], dict[str, int]]:
    """查这些公司已入库的 chunk id 与数量（每公司一次 query）。"""
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for code in codes:
        try:
            rows = client.query(
                collection_name=collection,
                filter=f'company_code == "{code}"',
                output_fields=["id"],
                limit=16384,
            )
            ids.update(str(r["id"]) for r in rows)
            counts[code] = len(rows)
            logger.info(f"  已入库 {code}: {len(rows)} 条")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  查询 {code} 已入库数量失败: {e}")
            counts[code] = 0
    return ids, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="增量补全 Milvus 年报子块")
    parser.add_argument("--codes", default="", help="逗号分隔代码，默认全部 20 家")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-company", type=int, default=0,
                        help="每家目标总条数（按章节优先级取核心章节），0 表示不限")
    parser.add_argument("--skip-financial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conf = Config()
    codes = {c.strip() for c in args.codes.split(",") if c.strip()}
    if not codes:
        # 默认取 companies.csv 里的全部公司
        import csv as _csv
        with open(conf.COMPANIES_FILE, encoding="utf-8-sig") as fh:
            codes = {row["code"].strip() for row in _csv.DictReader(fh) if row.get("code")}

    logger.info(f"目标公司 {len(codes)} 家：{sorted(codes)}")

    from pymilvus import MilvusClient

    client = MilvusClient(uri=f"http://{conf.MILVUS_HOST}:{conf.MILVUS_PORT}",
                          db_name=conf.MILVUS_DATABASE_NAME)
    if not client.has_collection(conf.MILVUS_COLLECTION_NAME):
        logger.error("集合不存在，请先运行 init_data.py --milvus")
        return 2

    logger.info("查询已入库 chunk ...")
    done, counts = existing_stats(client, conf.MILVUS_COLLECTION_NAME, codes)
    logger.info(f"已入库合计：{len(done)} 条")

    parents = load_parents(conf.PROCESSED_DIR)
    logger.info(f"父块：{len(parents)} 条")

    # 按公司收集候选（排除已入库的 id）
    by_company: dict[str, list] = {}
    skipped_financial = 0
    skipped_no_parent = 0
    with open(os.path.join(conf.PROCESSED_DIR, "children.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            code = str(meta.get("company_code") or "")
            if code not in codes:
                continue
            if str(obj["id"]) in done:
                continue
            if args.skip_financial and str(meta.get("section", "")).endswith("财务报告"):
                skipped_financial += 1
                continue
            if meta.get("parent_id") not in parents:
                skipped_no_parent += 1
                continue
            by_company.setdefault(code, []).append(obj)

    pending = []
    for code in sorted(by_company):
        items = by_company[code]
        # 同一公司内按「章节价值 → 章节序号 → 页码」排序，优先补管理层讨论
        items.sort(key=lambda o: (
            section_rank(o["metadata"].get("section")),
            int(o["metadata"].get("section_no") or 0),
            int(o["metadata"].get("page") or 0),
        ))
        quota = len(items)
        if args.per_company:
            quota = max(0, args.per_company - counts.get(code, 0))
        if quota:
            pending.extend(items[:quota])

    logger.info(f"待入库 {len(pending)} 条（跳过财务报表 {skipped_financial}，缺父块 {skipped_no_parent}）")
    if args.per_company:
        logger.info(f"每家目标 {args.per_company} 条，按章节优先级取核心章节")
    if args.limit:
        pending = pending[: args.limit]
        logger.info(f"本次限 {args.limit} 条")
    if args.dry_run or not pending:
        logger.info("dry-run 结束，未写入 Milvus")
        return 0

    from langchain_core.documents import Document
    from rag_qa import VectorStore

    vector_store = VectorStore()
    logger.info(f"入库前集合总量：{vector_store.count()}")

    total = 0
    started = time.time()
    batch = []
    for obj in pending:
        meta = obj.get("metadata", {})
        meta["parent_content"] = parents[meta["parent_id"]]
        meta["id"] = obj["id"]
        batch.append(Document(page_content=obj["text"], metadata=meta))
        if len(batch) >= args.batch_size:
            vector_store.add_documents(batch)
            total += len(batch)
            batch = []
            cost = time.time() - started
            logger.info(f"已入库 {total}/{len(pending)} 条（{cost:.0f}s，{total / max(cost, 1e-6):.2f} 条/秒）")

    if batch:
        vector_store.add_documents(batch)
        total += len(batch)

    logger.info(f"完成：本次 {total} 条，集合总量 {vector_store.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
