# -*- coding: utf-8 -*-
"""SQL 安全校验与执行（只允许单条 SELECT，表白名单，强制 LIMIT）。"""
import datetime
import decimal
import re
import sys

import pymysql
import sqlglot
from sqlglot import exp

sys.path.insert(0, r"C:\pro")
from base import Config  # noqa: E402
from .schema import ALLOWED_TABLES  # noqa: E402

FORBIDDEN_TOKENS = [
    "insert", "update", "delete", "drop", "alter", "truncate", "create", "grant",
    "revoke", "into outfile", "load_file", "sleep(", "benchmark(", "information_schema",
    "mysql.", "@@", "procedure", "call ",
]


class SQLValidationError(Exception):
    pass


class SQLExecutor:
    def __init__(self, max_rows=200, conf=None):
        self.conf = conf or Config()
        self.max_rows = max_rows

    # ---------- 校验 + 强制 LIMIT ----------
    def validate(self, sql, max_rows=None):
        max_rows = max_rows or self.max_rows
        raw = (sql or "").strip()
        raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw).strip().rstrip(";").strip()
        if not raw:
            raise SQLValidationError("SQL 为空")
        lowered = raw.lower()
        for token in FORBIDDEN_TOKENS:
            if token in lowered:
                raise SQLValidationError("包含禁止的关键字：%s" % token)
        try:
            parsed = sqlglot.parse(raw, read="mysql")
        except Exception as exc:  # noqa: BLE001
            raise SQLValidationError("SQL 语法错误：%s" % exc)
        if len(parsed) != 1:
            raise SQLValidationError("只允许一条 SQL 语句")
        root = parsed[0]
        if not isinstance(root, exp.Select):
            raise SQLValidationError("只允许 SELECT 查询")
        cte_names = {str(cte.alias_or_name).lower() for cte in root.find_all(exp.CTE)}
        tables = {str(t.name).lower() for t in root.find_all(exp.Table)}
        unknown = tables - ALLOWED_TABLES - cte_names
        if unknown:
            raise SQLValidationError("查询了不允许的表：%s" % ", ".join(sorted(unknown)))
        limit = root.args.get("limit")
        if limit is None:
            root = root.limit(max_rows)
        else:
            try:
                num = int(limit.expression.this)
                if num > max_rows:
                    root = root.limit(max_rows)
            except Exception:  # noqa: BLE001
                root = root.limit(max_rows)
        return root.sql(dialect="mysql")

    # ---------- 执行 ----------
    def execute(self, sql, max_rows=None):
        max_rows = max_rows or self.max_rows
        conn = pymysql.connect(
            host=self.conf.MYSQL_HOST, port=3306, user=self.conf.MYSQL_USER,
            password=self.conf.MYSQL_PASSWORD, database=self.conf.MYSQL_DATABASE,
            charset="utf8mb4", cursorclass=pymysql.cursors.Cursor,
            connect_timeout=10, read_timeout=30)
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                columns = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchmany(max_rows) if columns else []
        finally:
            conn.close()
        rows = [[self._jsonable(v) for v in row] for row in rows]
        return columns, rows

    @staticmethod
    def _jsonable(value):
        if value is None:
            return None
        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, (datetime.date, datetime.datetime)):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    # ---------- 展示 ----------
    def to_markdown(self, columns, rows, max_display=50):
        if not columns:
            return ""
        shown = rows[:max_display]
        lines = ["| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in shown:
            lines.append("| " + " | ".join(self._fmt(v) for v in row) + " |")
        if len(rows) > max_display:
            lines.append("")
            lines.append("> 共 %d 行，仅展示前 %d 行" % (len(rows), max_display))
        return "\n".join(lines)

    @staticmethod
    def _fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            if abs(value) >= 100:
                return "%.2f" % value
            return ("%.4f" % value).rstrip("0").rstrip(".")
        if isinstance(value, int):
            return str(value)
        return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    ex = SQLExecutor()
    sql = ex.validate("SELECT c.name, f.total_operate_income/1e8 AS revenue_yi "
                      "FROM financial_indicators f JOIN companies c ON c.code=f.code "
                      "WHERE f.report_year=2025 ORDER BY f.total_operate_income DESC")
    print("safe sql:", sql)
    cols, rows = ex.execute(sql)
    print(cols)
    print(rows[:3])
    print(ex.to_markdown(cols, rows))
