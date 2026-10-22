# -*- coding: utf-8 -*-
"""股东关系图谱问答（P3）。

两条后端：
  - Neo4j（config.ini [graph] enabled=true 且服务可达）：公司-股东二部图，Cypher 查询；
  - MySQL holders 表（默认兜底）：没有装 Neo4j 也能回答同样的关系问题。

支持三类问题：
  1. 谁持有 X / X 的股东结构          -> 十大股东列表
  2. X 和 Y 的共同股东 / 同时持有      -> 股东名取交集
  3. 某基金/机构持有哪几家白酒公司      -> 按股东名反查持仓
"""
import re
from typing import Any, Dict, List, Optional

import pymysql

from base import Config, logger
from agent.router import detect_companies


class GraphQA:
    def __init__(self, conf: Optional[Config] = None):
        self.conf = conf or Config()
        self.backend = "mysql"
        self.driver = None
        self._init_neo4j()

    # ---------- 初始化 ----------
    def _init_neo4j(self):
        """只有在配置启用且服务可达时才用 Neo4j，否则静默降级到 MySQL。"""
        if not getattr(self.conf, "GRAPH_ENABLED", False):
            logger.info("Neo4j 图谱未启用（config.ini [graph] enabled=false），使用 MySQL holders 兜底")
            return
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.conf.NEO4J_URI,
                auth=(self.conf.NEO4J_USER, self.conf.NEO4J_PASSWORD),
                connection_timeout=3,
            )
            driver.verify_connectivity()
            self.driver = driver
            self.backend = "neo4j"
            logger.info(f"Neo4j 已连接：{self.conf.NEO4J_URI}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Neo4j 不可用（{exc}），使用 MySQL holders 兜底")
            self.driver = None
            self.backend = "mysql"

    def _mysql(self):
        return pymysql.connect(
            host=self.conf.MYSQL_HOST, port=3306, user=self.conf.MYSQL_USER,
            password=self.conf.MYSQL_PASSWORD, database=self.conf.MYSQL_DATABASE,
            charset="utf8mb4", cursorclass=pymysql.cursors.Cursor,
            connect_timeout=10, read_timeout=30,
        )

    def _query(self, sql, params=()):
        conn = self._mysql()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                columns = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall() if columns else []
        finally:
            conn.close()
        return columns, [list(r) for r in rows]

    # ---------- 意图判断 ----------
    @staticmethod
    def can_answer(question: str) -> bool:
        text = question or ""
        relation = any(k in text for k in (
            "同时持有", "共同持有", "都持有", "共同股东", "股东", "持股", "持仓",
            "重仓", "持有哪", "持有哪些", "股权", "参股", "控股", "关联方", "实际控制人",
        ))
        return relation and bool(re.search(r"(持有|股东|持股|持仓|重仓|股权)", text))

    # ---------- 三类查询 ----------
    def _holders_of(self, company: str) -> List[Dict[str, Any]]:
        if self.backend == "neo4j":
            cypher = """
            MATCH (h:Holder)-[r:HOLDS]->(c:Company {name: $name})
            RETURN h.name AS holder_name, r.ratio AS hold_ratio,
                   r.rank AS holder_rank, r.change AS hold_change
            ORDER BY r.rank ASC LIMIT 10
            """
            with self.driver.session(database=self.conf.NEO4J_DATABASE) as sess:
                return [dict(rec) for rec in sess.run(cypher, name=company)]
        columns, rows = self._query(
            "SELECT holder_name, hold_ratio, holder_rank, hold_change "
            "FROM holders WHERE name = %s ORDER BY holder_rank ASC LIMIT 10",
            (company,),
        )
        return [dict(zip(columns, row)) for row in rows]

    def _common_holders(self, company_a: str, company_b: str) -> List[Dict[str, Any]]:
        if self.backend == "neo4j":
            cypher = """
            MATCH (h:Holder)-[r1:HOLDS]->(a:Company {name: $a})
            MATCH (h)-[r2:HOLDS]->(b:Company {name: $b})
            RETURN h.name AS holder_name, r1.ratio AS ratio_a, r2.ratio AS ratio_b
            ORDER BY (r1.ratio + r2.ratio) DESC LIMIT 20
            """
            with self.driver.session(database=self.conf.NEO4J_DATABASE) as sess:
                return [dict(rec) for rec in sess.run(cypher, a=company_a, b=company_b)]
        columns, rows = self._query(
            "SELECT h1.holder_name, h1.hold_ratio AS ratio_a, h2.hold_ratio AS ratio_b "
            "FROM holders h1 JOIN holders h2 ON h1.holder_name = h2.holder_name "
            "WHERE h1.name = %s AND h2.name = %s "
            "ORDER BY (h1.hold_ratio + h2.hold_ratio) DESC LIMIT 20",
            (company_a, company_b),
        )
        return [dict(zip(columns, row)) for row in rows]

    def _portfolio_of(self, holder_keyword: str) -> List[Dict[str, Any]]:
        if self.backend == "neo4j":
            cypher = """
            MATCH (h:Holder)-[r:HOLDS]->(c:Company)
            WHERE h.name CONTAINS $kw
            RETURN c.name AS company, r.ratio AS hold_ratio
            ORDER BY r.ratio DESC LIMIT 20
            """
            with self.driver.session(database=self.conf.NEO4J_DATABASE) as sess:
                return [dict(rec) for rec in sess.run(cypher, kw=holder_keyword)]
        columns, rows = self._query(
            "SELECT name AS company, hold_ratio FROM holders "
            "WHERE holder_name LIKE %s ORDER BY hold_ratio DESC LIMIT 20",
            (f"%{holder_keyword}%",),
        )
        return [dict(zip(columns, row)) for row in rows]

    # ---------- 统一入口 ----------
    def ask(self, question: str) -> Dict[str, Any]:
        """返回 dict：answer / columns / rows / chart_kind / error。"""
        text = question or ""
        companies = detect_companies(text)

        # 1) 两家公司 -> 共同股东
        if len(companies) >= 2 and re.search(r"(同时持有|共同持有|都持有|共同股东|共同|交集)", text):
            a, b = companies[0], companies[1]
            rows = self._common_holders(a, b)
            if not rows:
                return self._result(
                    f"十大股东名单里没有同时持有 **{a}** 和 **{b}** 的机构（两家公司的股东名单无交集）。",
                    ["holder_name", "ratio_a", "ratio_b"], [], "bar",
                )
            names = "、".join(str(r["holder_name"]) for r in rows[:5])
            answer = (
                f"同时持有 **{a}** 和 **{b}** 的股东共 **{len(rows)}** 家，前几位是：{names}。"
                f"其中最高的是 {rows[0]['holder_name']}（{a} {rows[0]['ratio_a']}%、{b} {rows[0]['ratio_b']}%）。"
            )
            return self._result(answer, ["holder_name", "ratio_a", "ratio_b"],
                                [[r["holder_name"], r["ratio_a"], r["ratio_b"]] for r in rows], "bar")

        # 2) 某股东持有哪些公司
        holder_match = re.search(
            r"([\u4e00-\u9fa5A-Za-z0-9（）()·]{2,30}?(?:基金|资产管理|投资|保险|证券|银行|公司|集团))"
            r"[^，。？?]{0,8}(?:持有|重仓|持仓)", text)
        if holder_match and not companies:
            keyword = holder_match.group(1)
            rows = self._portfolio_of(keyword)
            if not rows:
                return self._result(f"十大股东名单里没有找到「{keyword}」的持仓记录。",
                                    ["company", "hold_ratio"], [], "bar")
            detail = "、".join(f"{r['company']}（{r['hold_ratio']}%）" for r in rows)
            return self._result(
                f"**{keyword}** 出现在 {len(rows)} 家白酒公司的十大股东名单中：{detail}。",
                ["company", "hold_ratio"],
                [[r["company"], r["hold_ratio"]] for r in rows], "bar")

        # 3) 单家公司 -> 股东结构
        if len(companies) == 1:
            company = companies[0]
            rows = self._holders_of(company)
            if not rows:
                return self._result(f"没有查到 **{company}** 的股东数据。",
                                    ["holder_name", "hold_ratio"], [], "pie")
            top = rows[0]
            summary = "、".join(f"{r['holder_name']}（{r['hold_ratio']}%）" for r in rows[:3])
            return self._result(
                f"**{company}** 前三大股东：{summary}。第一大股东 {top['holder_name']} 持股 {top['hold_ratio']}%，"
                f"前十大股东合计约 {round(sum(float(r['hold_ratio'] or 0) for r in rows), 2)}%。",
                ["holder_name", "hold_ratio"],
                [[r["holder_name"], r["hold_ratio"]] for r in rows], "pie")

        return self._result(
            "请说明要查哪家公司的股东，或哪两家公司的共同股东（例如「茅台和五粮液的共同股东」）。",
            ["holder_name", "hold_ratio"], [], None)

    @staticmethod
    def _result(answer, columns, rows, chart_kind) -> Dict[str, Any]:
        return {
            "answer": answer,
            "columns": columns,
            "rows": rows,
            "chart_kind": chart_kind,
            "backend": "graph",
            "error": None,
        }

    def close(self):
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    qa = GraphQA()
    for q in [
        "贵州茅台的前三大股东是谁",
        "哪些基金同时持有贵州茅台和五粮液",
        "招商中证白酒基金持有哪几家公司",
    ]:
        print("Q:", q)
        r = qa.ask(q)
        print("A:", r["answer"])
        print("-" * 60)
