# -*- coding: utf-8 -*-
"""图表数据 API：直接返回 ECharts option，前端 setOption 即可。"""
from fastapi import APIRouter, Query

from services import chart_service

router = APIRouter(prefix="/api/charts", tags=["charts"])


@router.get("/trend")
def trend(company: str = Query(...), metric: str = Query("revenue")):
    """单公司趋势图 option。"""
    option = chart_service.trend_option(company, metric)
    if not option:
        return {"charts": []}
    return {"charts": [option]}


@router.get("/trends")
def trends(company: str = Query(...)):
    """公司速览：营收/净利/毛利率/ROE 四张图。"""
    return {"charts": chart_service.trend_options(company)}


@router.get("/compare")
def compare(year: int = Query(2025), metrics: str = Query("revenue,netprofit")):
    """全行业对比图 option 列表。"""
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    return {"charts": chart_service.compare_options(year, metric_list)}
