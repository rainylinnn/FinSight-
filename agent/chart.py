# -*- coding: utf-8 -*-
"""把 SQL / 图谱返回的二维表转成 ECharts 图表配置。

设计原则：不猜测业务语义，只做「列类型 + 行数」的形状判断：
  - 有年份列 + 数值列            -> 折线（多公司自动成多系列）
  - 一个类别列 + 数值列          -> 柱状；占比类问题 -> 环形图
  - 其余形状                     -> 不出图（前端只展示表格）

图表配置是通用的，不含 ECharts 具体实现，前端负责组装 option。
"""
import re
from typing import Any, Dict, List, Optional, Sequence

# 年份列 / 类别列 / 不应作为指标的数值列
YEAR_FIELDS = {"report_year", "year", "年份", "年度", "report_date", "notice_date"}
CATEGORY_FIELDS = {
    "name", "company_name", "公司", "公司简称", "公司名称", "code", "股票代码",
    "holder_name", "股东名称", "股东", "section", "章节",
}
IGNORED_NUMERIC = {"rank", "holder_rank", "id", "hold_num", "code", "pages"}

# 列名 -> (展示名, 单位)
COLUMN_LABELS = {
    "revenue_yi": ("营业总收入", "亿元"),
    "total_operate_income_yi": ("营业总收入", "亿元"),
    "income_yi": ("营业总收入", "亿元"),
    "netprofit_yi": ("归母净利润", "亿元"),
    "parent_netprofit_yi": ("归母净利润", "亿元"),
    "profit_yi": ("归母净利润", "亿元"),
    "total_operate_income": ("营业总收入", "元"),
    "parent_netprofit": ("归母净利润", "元"),
    "total_operate_income_yoy": ("营收同比", "%"),
    "parent_netprofit_yoy": ("净利同比", "%"),
    "roe_weighted": ("加权ROE", "%"),
    "roe_deducted": ("扣非ROE", "%"),
    "gross_margin": ("毛利率", "%"),
    "net_margin": ("净利率", "%"),
    "debt_ratio": ("资产负债率", "%"),
    "roic": ("投入资本回报率", "%"),
    "eps_basic": ("基本每股收益", "元"),
    "bps": ("每股净资产", "元"),
    "op_cashflow_per_share": ("每股经营现金流", "元"),
    "hold_ratio": ("持股比例", "%"),
    "hold_num": ("持股数量", "股"),
    "yoy": ("同比", "%"),
}

PERCENT_HINTS = ("ratio", "yoy", "margin", "roe", "roic", "rate", "pct", "占比", "比例", "率")
PIE_HINTS = ("占比", "比例", "结构", "分布", "持股", "股东", "份额")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _column_values(rows: Sequence[Sequence[Any]], idx: int) -> List[Any]:
    return [row[idx] if idx < len(row) else None for row in rows]


def _numeric_columns(columns: List[str], rows: List[List[Any]]) -> List[int]:
    out = []
    for idx, col in enumerate(columns):
        key = str(col).lower()
        if key in YEAR_FIELDS or key in IGNORED_NUMERIC:
            continue
        values = [v for v in _column_values(rows, idx) if v is not None]
        if not values:
            continue
        nums = [v for v in values if _is_number(v)]
        # 至少 80% 是数字才算数值列（NULL 容忍）
        if len(nums) >= max(2, int(len(values) * 0.8)):
            # 全是 1990-2100 之间的整数 -> 当成年份，不当指标
            if all(float(v).is_integer() and 1990 <= float(v) <= 2100 for v in nums):
                continue
            out.append(idx)
    return out


def _find_column(columns: List[str], candidates: set) -> Optional[int]:
    for idx, col in enumerate(columns):
        if str(col).lower() in candidates:
            return idx
    return None


def _label_of(column: str) -> str:
    key = str(column).lower()
    if key in COLUMN_LABELS:
        return COLUMN_LABELS[key][0]
    # 去掉聚合/换算后缀：netprofit_yi -> netprofit
    base = re.sub(r"_(yi|yuan|pct)$", "", key)
    if base in COLUMN_LABELS:
        return COLUMN_LABELS[base][0]
    return str(column)


def _unit_of(columns: List[str], numeric_idx: List[int]) -> str:
    units = [COLUMN_LABELS.get(str(columns[i]).lower(), ("", ""))[1] for i in numeric_idx]
    units = [u for u in units if u]
    if not units:
        return ""
    return units[0] if len(set(units)) == 1 else ""


def _is_percent(column: str) -> bool:
    key = str(column).lower()
    if key in COLUMN_LABELS:
        return COLUMN_LABELS[key][1] == "%"
    return any(h in key for h in PERCENT_HINTS)


def _clean_title(question: str) -> str:
    text = re.sub(r"[？?。！!]+$", "", (question or "").strip())
    text = re.sub(r"\s+", " ", text)
    return text[:26] + ("…" if len(text) > 26 else "")


def build_chart(columns: List[str], rows: List[List[Any]], question: str = "") -> Optional[Dict[str, Any]]:
    """把查询结果转成图表配置；形状不合适时返回 None。"""
    if not columns or not rows or len(rows) < 2 or len(columns) < 2:
        return None
    if len(rows) > 60:  # 太长的明细不适合画图
        return None

    numeric_idx = _numeric_columns(columns, rows)
    if not numeric_idx:
        return None

    year_idx = _find_column(columns, YEAR_FIELDS)
    # 年份列如果被判断成数值列，从指标里剔除
    if year_idx is not None:
        numeric_idx = [i for i in numeric_idx if i != year_idx]
    if not numeric_idx:
        return None

    cat_idx = _find_column(columns, CATEGORY_FIELDS)
    # 兜底：第一个非数值、非年份列当作类别列
    if cat_idx is None:
        for idx in range(len(columns)):
            if idx not in numeric_idx and idx != year_idx:
                cat_idx = idx
                break

    title = _clean_title(question)
    unit = _unit_of(columns, numeric_idx)

    # ---------- 1) 年份趋势：折线 ----------
    if year_idx is not None and len({row[year_idx] for row in rows}) >= 2:
        years = sorted({str(row[year_idx]) for row in rows if row[year_idx] is not None})
        series = []

        if cat_idx is not None and cat_idx != year_idx:
            categories = []
            for row in rows:
                name = str(row[cat_idx])
                if name not in categories:
                    categories.append(name)
        else:
            categories = [""]

        for cat in categories:
            for n_idx in numeric_idx[:1]:  # 一张图只画一个指标，避免量纲混在一起
                data = []
                for year in years:
                    value = None
                    for row in rows:
                        if str(row[year_idx]) == year and (not cat or str(row[cat_idx]) == cat):
                            value = row[n_idx]
                            break
                    data.append(value)
                series.append({
                    "name": (f"{cat} · " if cat else "") + _label_of(columns[n_idx]),
                    "data": data,
                })
        if not series:
            return None
        return {
            "type": "line",
            "title": title,
            "unit": unit,
            "xAxis": years,
            "series": series[:6],
        }

    # ---------- 2) 单类别 + 数值：柱状 / 环形 ----------
    if cat_idx is None:
        return None

    categories = [str(row[cat_idx]) for row in rows]
    # 类别太多时只保留数值最大的前 12 个
    if len(categories) > 12:
        order = sorted(range(len(rows)), key=lambda i: -(rows[i][numeric_idx[0]] or 0))[:12]
        rows = [rows[i] for i in order]
        categories = [categories[i] for i in order]

    text = question or ""
    wants_pie = (kind == "pie") or (
        kind is None and any(h in text for h in PIE_HINTS) and 2 <= len(rows) <= 12
    )
    if kind == "bar":
        wants_pie = False
    first_is_percent = _is_percent(columns[numeric_idx[0]])

    if wants_pie and first_is_percent:
        return {
            "type": "pie",
            "title": title,
            "unit": "%" if first_is_percent else unit,
            "series": [{
                "name": _label_of(columns[numeric_idx[0]]),
                "data": [{"name": c, "value": row[numeric_idx[0]]}
                         for c, row in zip(categories, rows)],
            }],
        }

    series = []
    for n_idx in numeric_idx[:3]:
        series.append({
            "name": _label_of(columns[n_idx]),
            "data": [row[n_idx] for row in rows],
        })
    return {
        "type": "bar",
        "title": title,
        "unit": unit,
        "xAxis": categories,
        "series": series,
    }
