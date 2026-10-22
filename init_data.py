# -*- coding: utf-8 -*-
"""FinSight 一键初始化：MySQL 建表 + 导入结构化数据；Milvus 入库年报子块；Neo4j（可选）。

用法：
  python init_data.py --all
  python init_data.py --mysql
  python init_data.py --milvus --limit 100        # 先小批量验证
  python init_data.py --milvus --batch-size 200   # 全量入库
  python init_data.py --milvus --recreate-milvus  # 删除集合并重建
  python init_data.py --neo4j                     # 需要先安装/启动 Neo4j，并把 config.ini 的 graph.enabled 改为 true
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from base import Config, logger  # noqa: E402

conf = Config()


# ============================ MySQL ============================

DDL = [
    """CREATE TABLE IF NOT EXISTS companies (
        code VARCHAR(10) PRIMARY KEY,
        name VARCHAR(50) NOT NULL,
        industry VARCHAR(50) DEFAULT '白酒',
        first_holder VARCHAR(200) DEFAULT '',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS financial_indicators (
        code VARCHAR(10) NOT NULL,
        report_year INT NOT NULL,
        report_date DATE,
        report_type VARCHAR(20) DEFAULT '年报',
        notice_date DATE,
        total_operate_income DOUBLE,
        total_operate_income_yoy DOUBLE,
        parent_netprofit DOUBLE,
        parent_netprofit_yoy DOUBLE,
        roe_weighted DOUBLE,
        roe_deducted DOUBLE,
        gross_margin DOUBLE,
        net_margin DOUBLE,
        debt_ratio DOUBLE,
        eps_basic DOUBLE,
        bps DOUBLE,
        op_cashflow_per_share DOUBLE,
        roic DOUBLE,
        PRIMARY KEY (code, report_year)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS holders (
        code VARCHAR(10) NOT NULL,
        end_date DATE NOT NULL,
        holder_rank INT NOT NULL,
        holder_name VARCHAR(200) NOT NULL,
        hold_num BIGINT,
        hold_ratio DOUBLE,
        hold_change VARCHAR(50),
        holder_code VARCHAR(50),
        PRIMARY KEY (code, end_date, holder_rank)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS documents (
        code VARCHAR(10) NOT NULL,
        report_year INT NOT NULL,
        title VARCHAR(300),
        file_path VARCHAR(500),
        pages INT,
        PRIMARY KEY (code, report_year)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS conversations (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(36) NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        INDEX idx_session_id (session_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


def _clean(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value != "" else None


def _to_float(value):
    value = _clean(value)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(value):
    value = _clean(value)
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _date(value):
    value = _clean(value)
    return value[:10] if value else None


def init_mysql(recreate=False):
    import pymysql

    logger.info("开始初始化 MySQL ...")
    conn = pymysql.connect(host=conf.MYSQL_HOST, port=3306, user=conf.MYSQL_USER,
                           password=conf.MYSQL_PASSWORD, charset="utf8mb4", autocommit=False)
    cur = conn.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS `%s` DEFAULT CHARSET utf8mb4" % conf.MYSQL_DATABASE)
    cur.execute("USE `%s`" % conf.MYSQL_DATABASE)
    if recreate:
        for table in ["conversations", "documents", "holders", "financial_indicators", "companies"]:
            cur.execute("DROP TABLE IF EXISTS `%s`" % table)
    for sql in DDL:
        cur.execute(sql)

    # 1) companies + first_holder
    first_holder = {}
    rows = []
    if os.path.exists(conf.COMPANIES_FILE):
        with open(conf.COMPANIES_FILE, "r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                rows.append((row.get("code", "").strip(), row.get("name", "").strip()))
    holder_rows = []
    if os.path.exists(conf.HOLDERS_CSV):
        with open(conf.HOLDERS_CSV, "r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                holder_rows.append(row)
                if str(row.get("rank")).strip() == "1":
                    first_holder[row.get("code", "").strip()] = (
                        "%s（%s%%）" % (row.get("holder_name", ""), row.get("hold_ratio", "")))
    cur.executemany(
        "REPLACE INTO companies (code, name, industry, first_holder) VALUES (%s, %s, %s, %s)",
        [(c, n, "白酒", first_holder.get(c, "")) for c, n in rows])
    logger.info("companies: %d 条" % len(rows))

    # 2) financial_indicators
    fin_rows = []
    if os.path.exists(conf.FINANCIAL_CSV):
        with open(conf.FINANCIAL_CSV, "r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                fin_rows.append((
                    row.get("code", "").strip(), _to_int(row.get("report_year")),
                    _date(row.get("report_date")), _clean(row.get("report_type")) or "年报",
                    _date(row.get("notice_date")),
                    _to_float(row.get("total_operate_income")),
                    _to_float(row.get("total_operate_income_yoy")),
                    _to_float(row.get("parent_netprofit")),
                    _to_float(row.get("parent_netprofit_yoy")),
                    _to_float(row.get("roe_weighted")), _to_float(row.get("roe_deducted")),
                    _to_float(row.get("gross_margin")), _to_float(row.get("net_margin")),
                    _to_float(row.get("debt_ratio")), _to_float(row.get("eps_basic")),
                    _to_float(row.get("bps")), _to_float(row.get("op_cashflow_per_share")),
                    _to_float(row.get("roic")),
                ))
    cur.executemany(
        """REPLACE INTO financial_indicators
           (code, report_year, report_date, report_type, notice_date, total_operate_income,
            total_operate_income_yoy, parent_netprofit, parent_netprofit_yoy, roe_weighted,
            roe_deducted, gross_margin, net_margin, debt_ratio, eps_basic, bps,
            op_cashflow_per_share, roic)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", fin_rows)
    logger.info("financial_indicators: %d 条" % len(fin_rows))

    # 3) holders
    holder_values = []
    for row in holder_rows:
        rank = _to_int(row.get("rank"))
        if rank is None:
            continue
        holder_values.append((
            row.get("code", "").strip(), _date(row.get("end_date")), rank,
            row.get("holder_name", "").strip(), _to_int(row.get("hold_num")),
            _to_float(row.get("hold_ratio")), _clean(row.get("hold_change")),
            _clean(row.get("holder_code")),
        ))
    cur.executemany(
        """REPLACE INTO holders
           (code, end_date, holder_rank, holder_name, hold_num, hold_ratio, hold_change, holder_code)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""", holder_values)
    logger.info("holders: %d 条" % len(holder_values))

    # 4) documents
    doc_values = []
    doc_jsonl = os.path.join(conf.PROCESSED_DIR, "documents.jsonl")
    if os.path.exists(doc_jsonl):
        with open(doc_jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                doc_values.append((obj.get("code", ""), _to_int(obj.get("year")),
                                   obj.get("title", ""), obj.get("file_path", ""),
                                   _to_int(obj.get("pages"))))
    cur.executemany(
        "REPLACE INTO documents (code, report_year, title, file_path, pages) VALUES (%s,%s,%s,%s,%s)",
        doc_values)
    logger.info("documents: %d 条" % len(doc_values))

    conn.commit()

    # 校验
    for table in ["companies", "financial_indicators", "holders", "documents"]:
        cur.execute("SELECT COUNT(*) FROM `%s`" % table)
        logger.info("MySQL 表 %-24s 行数=%s" % (table, cur.fetchone()[0]))
    cur.close()
    conn.close()
    logger.info("MySQL 初始化完成：%s" % conf.MYSQL_DATABASE)


# ============================ Milvus ============================

def load_parents():
    """parent_id -> parent_content（用于检索时回填父块上下文）"""
    path = os.path.join(conf.PROCESSED_DIR, "parents.jsonl")
    parents = {}
    if not os.path.exists(path):
        raise FileNotFoundError("缺少 %s，请先运行 scripts/parse_annual_reports.py" % path)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            parents[obj["id"]] = obj["text"][:60000]
    return parents


def init_milvus(limit=0, batch_size=200, recreate=False, years=None, skip_financial=False):
    from pymilvus import MilvusClient
    from langchain_core.documents import Document

    logger.info("开始初始化 Milvus ...")
    if years:
        logger.info("只入库年份：%s；跳过财务报告章节：%s" % (years, skip_financial))
    client = MilvusClient(uri="http://%s:%s" % (conf.MILVUS_HOST, conf.MILVUS_PORT))
    databases = client.list_databases()
    if conf.MILVUS_DATABASE_NAME not in databases:
        client.create_database(conf.MILVUS_DATABASE_NAME)
        logger.info("已创建 Milvus 数据库：%s" % conf.MILVUS_DATABASE_NAME)
    if recreate:
        db_client = MilvusClient(uri="http://%s:%s" % (conf.MILVUS_HOST, conf.MILVUS_PORT),
                                 db_name=conf.MILVUS_DATABASE_NAME)
        if db_client.has_collection(conf.MILVUS_COLLECTION_NAME):
            db_client.drop_collection(conf.MILVUS_COLLECTION_NAME)
            logger.info("已删除集合：%s" % conf.MILVUS_COLLECTION_NAME)

    from rag_qa import VectorStore
    vector_store = VectorStore()
    logger.info("当前集合实体数：%s" % vector_store.count())

    parents = load_parents()
    children_path = os.path.join(conf.PROCESSED_DIR, "children.jsonl")
    if not os.path.exists(children_path):
        raise FileNotFoundError("缺少 %s，请先运行 scripts/parse_annual_reports.py" % children_path)

    docs, total, skipped, start = [], 0, 0, time.time()
    with open(children_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            if years and int(meta.get("year") or 0) not in years:
                continue
            if skip_financial and str(meta.get("section", "")).endswith("财务报告"):
                continue
            parent_id = meta.get("parent_id")
            parent_content = parents.get(parent_id, "")
            if not parent_content:
                skipped += 1
                continue
            meta["parent_content"] = parent_content
            meta["id"] = obj["id"]
            docs.append(Document(page_content=obj["text"], metadata=meta))
            if len(docs) >= batch_size:
                vector_store.add_documents(docs)
                total += len(docs)
                docs = []
                cost = time.time() - start
                logger.info("Milvus 已入库 %d 条（%.1fs，%.1f 条/秒）" % (
                    total, cost, total / max(cost, 1e-6)))
                if limit and total >= limit:
                    break
    if docs and not (limit and total >= limit):
        vector_store.add_documents(docs)
        total += len(docs)
    logger.info("Milvus 入库完成：本次 %d 条，跳过 %d 条（无父块），集合总量 %s" % (
        total, skipped, vector_store.count()))


# ============================ Neo4j（可选） ============================

def init_neo4j():
    if not conf.GRAPH_ENABLED:
        logger.info("config.ini [graph] enabled=false，跳过 Neo4j（需要时安装 Neo4j 并改为 true）")
        return
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("未安装 neo4j 依赖，跳过：pip install neo4j")
        return
    driver = GraphDatabase.driver(conf.NEO4J_URI, auth=(conf.NEO4J_USER, conf.NEO4J_PASSWORD))
    rows = []
    with open(conf.HOLDERS_CSV, "r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    with driver.session(database=conf.NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE CONSTRAINT company_code IF NOT EXISTS FOR (c:Company) REQUIRE c.code IS UNIQUE")
        session.run("CREATE CONSTRAINT holder_name IF NOT EXISTS FOR (h:Holder) REQUIRE h.name IS UNIQUE")
        # 公司节点
        companies = {}
        for row in rows:
            companies[row["code"]] = row["name"]
        for code, name in companies.items():
            session.run("MERGE (c:Company {code: $code}) SET c.name = $name, c.industry = '白酒'",
                        code=code, name=name)
        # 股东节点 + 持有关系
        for row in rows:
            session.run(
                """MATCH (c:Company {code: $code})
                   MERGE (h:Holder {name: $holder_name})
                   SET h.holder_code = $holder_code
                   MERGE (h)-[r:HOLDS {end_date: $end_date}]->(c)
                   SET r.rank = $rank, r.ratio = $ratio, r.shares = $shares""",
                code=row["code"], holder_name=row["holder_name"],
                holder_code=row.get("holder_code", ""), end_date=row.get("end_date", ""),
                rank=_to_int(row.get("rank")), ratio=_to_float(row.get("hold_ratio")),
                shares=_to_int(row.get("hold_num")))
    driver.close()
    logger.info("Neo4j 初始化完成：%d 家公司 / %d 条持股关系" % (len(companies), len(rows)))


# ============================ main ============================

def main():
    parser = argparse.ArgumentParser(description="FinSight 数据初始化")
    parser.add_argument("--all", action="store_true", help="MySQL + Milvus（Neo4j 视配置）")
    parser.add_argument("--mysql", action="store_true")
    parser.add_argument("--milvus", action="store_true")
    parser.add_argument("--neo4j", action="store_true")
    parser.add_argument("--recreate", action="store_true", help="重建 MySQL 表")
    parser.add_argument("--recreate-milvus", action="store_true", help="删除并重建 Milvus 集合")
    parser.add_argument("--limit", type=int, default=0, help="Milvus 只入库前 N 条（调试）")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--years", default="", help="只入库指定年份，逗号分隔，如 2025 或 2024,2025")
    parser.add_argument("--skip-financial-report", action="store_true",
                        help="跳过第十节 财务报告（数字类问题走 MySQL，可大幅减少入库量）")
    args = parser.parse_args()

    if not any([args.all, args.mysql, args.milvus, args.neo4j]):
        args.all = True

    years = sorted({int(y.strip()) for y in args.years.split(",") if y.strip()})
    if args.all or args.mysql:
        init_mysql(recreate=args.recreate)
    if args.all or args.milvus:
        init_milvus(limit=args.limit, batch_size=args.batch_size,
                    recreate=args.recreate_milvus, years=years,
                    skip_financial=args.skip_financial_report)
    if args.all or args.neo4j:
        init_neo4j()


if __name__ == "__main__":
    main()
