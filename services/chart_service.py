# -*- coding: utf-8 -*-
"""图表数据服务：从 MySQL 读财务指标，生成 ECharts option（前端直接 setOption）。

对外方法：
    trend_option(company, metric)         -> dict（单公司 2023-2025 趋势）
    compare_options(year, metrics)        -> list[dict]（全行业对比，按指标各一张图）
    trend_options(company)                -> list[dict]（营收+净利+毛利率+ROE 四张）
"""
from __future__ import annotations

from base import logger
from services.market_service import code_of

# 指标元数据：key -> (中文名, 单位, 字段, 缩放)
METRICS = {
    "revenue": ("营业收入", "亿元", "total_operate_income", 1e-8),
    "netprofit": ("归母净利润", "亿元", "parent_netprofit", 1e-8),
    "gross_margin": ("销售毛利率", "%", "gross_margin", 1.0),
    "net_margin": ("销售净利率", "%", "net_margin", 1.0),
    "roe": ("加权ROE", "%", "roe_weighted", 1.0),
    "debt_ratio": ("资产负债率", "%", "debt_ratio", 1.0),
}

_COLORS = ["#2f5d9e", "#c8a24a", "#4a7fc4", "#8fb3e0", "#a8842f",
           "#5b8fd0", "#d8b96a", "#7aa3d8", "#b89a4a", "#9cbfe6"]

_BASE = {
    "backgroundColor": "transparent",
    "animationDuration": 600,
    "grid": {"left": 48, "right": 24, "top": 48, "bottom": 36},
    "tooltip": {"trigger": "axis"},
    "legend": {"top": 8, "textStyle": {"fontSize": 12}},
}


def _query(sql: str) -> tuple[list, list]:
    from sql_qa.sql_executor import SQLExecutor

    return SQLExecutor(max_rows=500).execute(sql)


def _fmt(v: float, scale: float) -> float:
    return round(float(v or 0) * scale, 2)


def trend_option(company: str, metric: str = "revenue") -> dict | None:
    """单公司某指标 2023-2025 趋势（柱状图）。"""
    if metric not in METRICS:
        return None
    code = code_of(company)
    if not code:
        return None
    label, unit, col, scale = METRICS[metric]
    try:
        _, rows = _query(
            f"SELECT report_year, {col} FROM financial_indicators"
            f" WHERE code = '{code}' ORDER BY report_year")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"趋势图查询失败 {company}/{metric}: {e}")
        return None
    if not rows:
        return None
    years = [str(int(r[0])) for r in rows]
    values = [_fmt(r[1], scale) for r in rows]
    option = dict(_BASE)
    option.update({
        "title": {"text": f"{company} · {label}（{unit}）", "left": 12, "top": 10,
                  "textStyle": {"fontSize": 14, "fontWeight": 600}},
        "legend": {"show": False},
        "xAxis": {"type": "category", "data": years,
                  "axisLine": {"lineStyle": {"color": "#c9d2df"}}},
        "yAxis": {"type": "value", "name": unit, "splitLine": {"lineStyle": {"color": "#eef2f8"}}},
        "series": [{
            "name": label, "type": "bar", "data": values, "barWidth": "42%",
            "itemStyle": {"color": "#2f5d9e", "borderRadius": [6, 6, 0, 0]},
            "label": {"show": True, "position": "top", "fontSize": 11},
        }],
    })
    return option


def compare_options(year: int = 2025, metrics: list[str] | None = None) -> list[dict]:
    """全行业对比：每个指标一张柱状图（按高到低）。"""
    metrics = metrics or ["revenue", "netprofit"]
    metrics = [m for m in metrics if m in METRICS][:4]
    if not metrics:
        return []
    try:
        cols, rows = _query(
            "SELECT c.name, f." + ", f.".join(METRICS[m][2] for m in metrics)
            + " FROM financial_indicators f JOIN companies c ON c.code = f.code"
            f" WHERE f.report_year = {int(year)}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"对比图查询失败 {year}: {e}")
        return []
    if not rows:
        return []
    data = {r[0]: r[1:] for r in rows}
    charts = []
    for idx, metric in enumerate(metrics):
        label, unit, _col, scale = METRICS[metric]
        pairs = sorted(((name, _fmt(vals[idx], scale)) for name, vals in data.items()),
                       key=lambda x: x[1], reverse=True)
        names = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        option = dict(_BASE)
        option.update({
            "title": {"text": f"{year} 年白酒行业 · {label}（{unit}）", "left": 12, "top": 10,
                      "textStyle": {"fontSize": 14, "fontWeight": 600}},
            "legend": {"show": False},
            "grid": {"left": 52, "right": 24, "top": 48, "bottom": 64},
            "xAxis": {"type": "category", "data": names,
                      "axisLabel": {"rotate": 30, "fontSize": 11},
                      "axisLine": {"lineStyle": {"color": "#c9d2df"}}},
            "yAxis": {"type": "value", "name": unit,
                      "splitLine": {"lineStyle": {"color": "#eef2f8"}}},
            "series": [{
                "name": label, "type": "bar", "data": values, "barWidth": "52%",
                "itemStyle": {"color": _COLORS[idx % len(_COLORS)],
                              "borderRadius": [6, 6, 0, 0]},
                "label": {"show": True, "position": "top", "fontSize": 10},
            }],
        })
        charts.append(option)
    return charts


def trend_options(company: str) -> list[dict]:
    """公司速览：营收、净利、毛利率、ROE 四张图。"""
    return [opt for opt in (trend_option(company, m)
                            for m in ("revenue", "netprofit", "gross_margin", "roe")) if opt]


if __name__ == "__main__":
    import json

    print(json.dumps(trend_option("贵州茅台", "revenue"), ensure_ascii=False)[:300])
    print("compare charts:", len(compare_options(2025, ["revenue", "netprofit"])))
