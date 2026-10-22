# -*- coding: utf-8 -*-
"""板块数据 API。"""
from fastapi import APIRouter, Query

from services import sector_service

router = APIRouter(prefix="/api/sector", tags=["sector"])


@router.get("/overview")
def overview(year: int = Query(2025, ge=2020, le=2030)):
    """白酒板块总览：指数行情、涨跌家数、估值中位数、财务聚合、公司排行。"""
    return sector_service.overview(year)
