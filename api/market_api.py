# -*- coding: utf-8 -*-
"""行情数据 API。"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from base import Config
from services import market_service

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/watchlist")
def watchlist():
    """自选公司：20 家白酒公司 + 实时涨跌，一次批量请求取回。

    行情源不可用时仍然返回 200，price/pct 为 null，前端降级成“—”。
    """
    conf = Config()
    names = list(conf.VALID_SOURCES or [])
    codes = {name: market_service.code_of(name) for name in names}
    quotes = market_service.get_quotes([c for c in codes.values() if c])

    companies = []
    for name in names:
        code = codes.get(name)
        q = quotes.get(code) if code else None
        companies.append({
            "name": name,
            "code": code,
            "price": round(q["price"], 2) if q else None,
            "pct": round(q["pct"], 2) if q else None,
            "change": round(q["change"], 2) if q else None,
            "time": q["time"] if q else None,
        })
    return {
        "companies": companies,
        "quoted": sum(1 for c in companies if c["price"] is not None),
        "updated_at": datetime.now().strftime("%H:%M:%S"),
    }


@router.get("/quote")
def quote(code: str = Query(..., description="股票代码，如 600519")):
    """实时行情（腾讯数据源，缓存 30 秒）。"""
    data = market_service.get_quote(code)
    if not data:
        raise HTTPException(status_code=502, detail="行情数据源暂时不可用")
    return {"quote": data}


@router.get("/history")
def history(code: str = Query(..., description="股票代码"), days: int = Query(120, ge=5, le=500)):
    """日 K 线（东方财富数据源，缓存 1 小时）。"""
    rows = market_service.get_history(code, days)
    if not rows:
        raise HTTPException(status_code=502, detail="历史行情数据源暂时不可用")
    return {"code": code, "name": market_service.name_of(code), "rows": rows}
