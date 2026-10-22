# -*- coding: utf-8 -*-
"""板块聚合服务：中证白酒指数 + 涨跌宽度 + 财务聚合 + 估值中位数。

数据来源：
- 指数行情：腾讯财经（与 market_service 同一数据源）
- 个股涨跌 / PE / PB：market_service.get_quotes
- 财务聚合：MySQL financial_indicators（20 家公司 × 年报）
"""
from __future__ import annotations

import statistics

from base import Config, logger
from services import market_service

INDEX_CODE = "399997"      # 中证白酒指数（深交所）
INDEX_NAME = "中证白酒"


def _query(sql: str) -> tuple[list, list]:
    from sql_qa.sql_executor import SQLExecutor

    return SQLExecutor(max_rows=200).execute(sql)


def _safe_median(values: list[float], digits: int) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.median(clean), digits)


def _finance_summary(year: int) -> dict | None:
    """板块财务汇总：总量 + 加权盈利能力 + 加权增速。

    盈利指标一律按营业收入加权（整体法），不用简单平均：
    20 家公司里有营收 0.4 亿、亏损 2.7 亿的极端值（岩石股份），
    简单 AVG 会被它拉到 -28% 净利率，完全没有统计意义。
    """
    try:
        _, rows = _query(
            "SELECT COUNT(*),"
            " SUM(total_operate_income) / 1e8,"
            " SUM(parent_netprofit) / 1e8,"
            " SUM(gross_margin * total_operate_income) / SUM(total_operate_income),"
            " SUM(parent_netprofit) / SUM(total_operate_income) * 100,"
            " SUM(roe_weighted * total_operate_income) / SUM(total_operate_income),"
            " SUM(debt_ratio * total_operate_income) / SUM(total_operate_income),"
            " SUM(total_operate_income_yoy * total_operate_income) / SUM(total_operate_income),"
            " SUM(parent_netprofit_yoy * parent_netprofit) / SUM(parent_netprofit)"
            f" FROM financial_indicators WHERE report_year = {int(year)}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"板块财务汇总失败 {year}: {e}")
        return None
    if not rows or not rows[0] or not rows[0][0]:
        return None

    r = rows[0]
    return {
        "year": int(year),
        "count": int(r[0]),
        "revenue_total": round(float(r[1] or 0), 1),
        "netprofit_total": round(float(r[2] or 0), 1),
        "gross_margin": round(float(r[3] or 0), 1),
        "net_margin": round(float(r[4] or 0), 1),
        "roe": round(float(r[5] or 0), 1),
        "debt_ratio": round(float(r[6] or 0), 1),
        "revenue_yoy": round(float(r[7] or 0), 1),
        "netprofit_yoy": round(float(r[8] or 0), 1),
        "basis": "营收加权",
    }


def _yearly_trend() -> list[dict]:
    """全板块营收 / 净利三年趋势。"""
    try:
        _, rows = _query(
            "SELECT report_year, SUM(total_operate_income) / 1e8, SUM(parent_netprofit) / 1e8"
            " FROM financial_indicators GROUP BY report_year ORDER BY report_year"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"板块趋势查询失败: {e}")
        return []
    return [
        {"year": int(r[0]), "revenue": round(float(r[1] or 0), 1), "netprofit": round(float(r[2] or 0), 1)}
        for r in rows
    ]


def _ranking(year: int) -> list[dict]:
    """公司排行（按营收降序）。"""
    try:
        _, rows = _query(
            "SELECT c.name, c.code, f.total_operate_income / 1e8, f.parent_netprofit / 1e8,"
            " f.gross_margin, f.roe_weighted, f.total_operate_income_yoy"
            " FROM financial_indicators f JOIN companies c ON c.code = f.code"
            f" WHERE f.report_year = {int(year)}"
            " ORDER BY f.total_operate_income DESC"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"板块排行查询失败 {year}: {e}")
        return []
    return [
        {
            "name": r[0],
            "code": r[1],
            "revenue": round(float(r[2] or 0), 1),
            "netprofit": round(float(r[3] or 0), 1),
            "gross_margin": round(float(r[4] or 0), 1),
            "roe": round(float(r[5] or 0), 1),
            "revenue_yoy": round(float(r[6] or 0), 1),
        }
        for r in rows
    ]


def overview(year: int = 2025) -> dict:
    """板块总览：指数 + 宽度 + 估值 + 财务 + 排行 + 趋势。"""
    conf = Config()
    names = list(conf.VALID_SOURCES or [])
    code_of = {name: market_service.code_of(name) for name in names}
    codes = [c for c in code_of.values() if c]

    quotes = market_service.get_quotes(codes)
    index_quote = market_service.get_quote(INDEX_CODE)

    companies = []
    for name in names:
        code = code_of.get(name)
        q = quotes.get(code) if code else None
        companies.append({
            "name": name,
            "code": code,
            "price": round(q["price"], 2) if q else None,
            "pct": round(q["pct"], 2) if q else None,
            "pe": round(q["pe"], 1) if q and q.get("pe") else None,
            "pb": round(q["pb"], 2) if q and q.get("pb") else None,
        })

    pcts = [c["pct"] for c in companies if c["pct"] is not None]
    pes = [q["pe"] for q in quotes.values() if q.get("pe") and q["pe"] > 0]
    pbs = [q["pb"] for q in quotes.values() if q.get("pb") and q["pb"] > 0]

    ranking = _ranking(year)
    # 把实时涨跌并进排行，方便前端一张表看完
    pct_of = {c["name"]: c["pct"] for c in companies}
    price_of = {c["name"]: c["price"] for c in companies}
    for row in ranking:
        row["pct"] = pct_of.get(row["name"])
        row["price"] = price_of.get(row["name"])

    return {
        "name": "白酒板块",
        "index": {
            "code": INDEX_CODE,
            "name": INDEX_NAME,
            "price": round(index_quote["price"], 2) if index_quote else None,
            "pct": round(index_quote["pct"], 2) if index_quote else None,
            "change": round(index_quote["change"], 2) if index_quote else None,
            "time": index_quote["time"] if index_quote else None,
        },
        "breadth": {
            "up": sum(1 for p in pcts if p > 0),
            "down": sum(1 for p in pcts if p < 0),
            "flat": sum(1 for p in pcts if p == 0),
            "median_pct": _safe_median(pcts, 2),
            "quoted": len(pcts),
        },
        "valuation": {
            "pe_median": _safe_median(pes, 1),
            "pb_median": _safe_median(pbs, 2),
            "pe_min": round(min(pes), 1) if pes else None,
            "pe_max": round(max(pes), 1) if pes else None,
        },
        "finance": _finance_summary(year),
        "trend": _yearly_trend(),
        "ranking": ranking,
        "companies": companies,
    }
