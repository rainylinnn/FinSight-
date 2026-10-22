# -*- coding: utf-8 -*-
"""行情数据服务（免费数据源，无需 API Key）。

- 实时行情：腾讯财经 qt.gtimg.cn
- 历史 K 线：东方财富 push2his.eastmoney.com
- 数据只用于投研辅助展示，不构成投资建议

对外方法：
    get_quote(code)            -> dict | None
    get_history(code, days)    -> list[dict]
    answer(question)           -> str（自然语言回答，带行情 + 年报指标）
"""
from __future__ import annotations

import csv
import time
from datetime import datetime

import requests

from base import Config, logger

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://gu.qq.com/",
}

_QUOTE_TTL = 30      # 实时行情缓存 30 秒
_HIST_TTL = 3600     # 历史 K 线缓存 1 小时
_cache: dict = {}

_comp_conf = Config()
_company_by_name: dict | None = None
_company_by_code: dict | None = None


def _load_companies() -> None:
    """从 data/companies.csv 读取 公司名 <-> 代码 映射。"""
    global _company_by_name, _company_by_code
    if _company_by_name is not None:
        return
    by_name, by_code = {}, {}
    try:
        with open(_comp_conf.COMPANIES_FILE, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                code = str(row.get("code", "")).strip()
                name = str(row.get("name", "")).strip()
                if code and name:
                    by_name[name] = code
                    by_code[code] = name
    except Exception as e:  # noqa: BLE001
        logger.warning(f"读取 companies.csv 失败: {e}")
    _company_by_name, _company_by_code = by_name, by_code


def _market_prefix(code: str) -> str:
    """6 开头是上交所，其余是深交所。"""
    return "sh" if code.startswith(("6", "9")) else "sz"


def _eastmoney_secid(code: str) -> str:
    return f"1.{code}" if code.startswith(("6", "9")) else f"0.{code}"


def code_of(company: str) -> str | None:
    """公司名 -> 股票代码；也接受直接传代码。"""
    _load_companies()
    if company in _company_by_code:
        return company
    return _company_by_name.get(company)


def name_of(code: str) -> str:
    _load_companies()
    return _company_by_code.get(code, code)


def _cache_get(key: str, ttl: int):
    item = _cache.get(key)
    if item and time.time() - item[0] < ttl:
        return item[1]
    return None


def _cache_set(key: str, value) -> None:
    _cache[key] = (time.time(), value)


def _parse_quote(raw: str) -> dict | None:
    """解析腾讯行情接口里一行 ~ 分隔的字段；字段不足时返回 None。"""
    f = raw.split("~")
    if len(f) < 46:
        return None
    try:
        return {
            "code": f[2],
            "name": f[1],
            "price": float(f[3] or 0),
            "prev_close": float(f[4] or 0),
            "open": float(f[5] or 0),
            "change": float(f[31] or 0),
            "pct": float(f[32] or 0),
            "high": float(f[33] or 0),
            "low": float(f[34] or 0),
            "volume_hand": float(f[36] or 0),          # 手
            "amount_wan": float(f[37] or 0),           # 万元
            "turnover": float(f[38] or 0),             # %
            "pe": float(f[39] or 0),                   # 市盈率 TTM
            "pb": float(f[46] or 0),                   # 市净率
            "float_cap_yi": float(f[44] or 0),         # 流通市值（亿）
            "market_cap_yi": float(f[45] or 0),        # 总市值（亿）
            "time": f[30],
        }
    except (TypeError, ValueError) as e:  # noqa: BLE001
        logger.warning(f"行情字段解析失败: {e}")
        return None


def get_quote(code: str) -> dict | None:
    """实时行情（腾讯接口）。返回 dict，失败 None。"""
    cached = _cache_get(f"quote:{code}", _QUOTE_TTL)
    if cached is not None:
        return cached
    url = f"https://qt.gtimg.cn/q={_market_prefix(code)}{code}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        text = resp.content.decode("gbk", errors="replace")
        quote = _parse_quote(text.split('="', 1)[1].rstrip('";\n '))
        if quote is None:
            raise ValueError("行情字段不足")
        quote["code"] = code
        _cache_set(f"quote:{code}", quote)
        return quote
    except Exception as e:  # noqa: BLE001
        logger.warning(f"获取行情失败 {code}: {e}")
        return None


def get_quotes(codes: list[str]) -> dict[str, dict]:
    """批量实时行情：一次 HTTP 请求拿多只，返回 {code: quote}。

    单只失败不会拖垮整批——取不到的 code 直接不出现在返回值里，调用方按需降级。
    """
    result: dict[str, dict] = {}
    pending: list[str] = []
    for code in codes:
        if not code:
            continue
        cached = _cache_get(f"quote:{code}", _QUOTE_TTL)
        if cached is not None:
            result[code] = cached
        else:
            pending.append(code)
    if not pending:
        return result

    url = "https://qt.gtimg.cn/q=" + ",".join(_market_prefix(c) + c for c in pending)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        text = resp.content.decode("gbk", errors="replace")
        for line in text.splitlines():
            if '="' not in line:
                continue
            quote = _parse_quote(line.split('="', 1)[1].rstrip('";\n '))
            if quote is None:
                continue
            _cache_set(f"quote:{quote['code']}", quote)
            result[quote["code"]] = quote
    except Exception as e:  # noqa: BLE001
        logger.warning(f"批量获取行情失败: {e}")
    return result


def get_history(code: str, days: int = 120) -> list[dict]:
    """日 K 线（东方财富），返回 [{date,open,close,high,low,volume,amount,pct}]。"""
    cached = _cache_get(f"hist:{code}:{days}", _HIST_TTL)
    if cached is not None:
        return cached
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": _eastmoney_secid(code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "klt": "101",
        "fqt": "1",
        "lmt": str(max(5, min(days, 500))),
        "end": "20500101",
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        data = resp.json().get("data") or {}
        rows = []
        for line in data.get("klines") or []:
            p = line.split(",")
            if len(p) < 8:
                continue
            rows.append({
                "date": p[0], "open": float(p[1]), "close": float(p[2]),
                "high": float(p[3]), "low": float(p[4]),
                "volume": float(p[5]), "amount": float(p[6]), "pct": float(p[7]),
            })
        _cache_set(f"hist:{code}:{days}", rows)
        return rows
    except Exception as e:  # noqa: BLE001
        logger.warning(f"获取历史行情失败 {code}: {e}")
        return []


def _latest_metrics(code: str) -> dict | None:
    """从 MySQL 取最近一年财务指标。"""
    try:
        from sql_qa.sql_executor import SQLExecutor

        ex = SQLExecutor(max_rows=5)
        cols, rows = ex.execute(
            "SELECT report_year, total_operate_income, parent_netprofit, gross_margin,"
            f" roe_weighted FROM financial_indicators WHERE code = '{code}'"
            " ORDER BY report_year DESC LIMIT 1")
        if not rows:
            return None
        return dict(zip(cols, rows[0]))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"读取财务指标失败 {code}: {e}")
        return None


def answer(question: str) -> str:
    """把行情问题转成有数据支撑的自然语言回答。"""
    _load_companies()
    from agent.router import detect_companies

    code = None
    hits = detect_companies(question or "")
    if hits:
        code = _company_by_name.get(hits[0])
    if code is None:
        for name, c in _company_by_name.items():
            if name in (question or ""):
                code = c
                break
    if code is None:
        return ("未能识别公司。行情服务支持 20 家白酒公司，例如：\"茅台今日股价\"、"
                "\"五粮液最近90天走势\"。")

    q = get_quote(code)
    if not q:
        return (f"{name_of(code)}（{code}）实时行情暂时获取失败（数据源超时），"
                "请稍后重试；也可以查看行情软件或交易所公告。")

    lines = [
        f"【{q['name']} {code} 行情】{q['time']}",
        f"最新价 {q['price']:.2f} 元，涨跌 {q['change']:+.2f} 元（{q['pct']:+.2f}%）；"
        f"今开 {q['open']:.2f}，最高 {q['high']:.2f}，最低 {q['low']:.2f}。",
        f"成交额 {q['amount_wan'] / 1e4:.2f} 亿元，换手率 {q['turnover']:.2f}%；"
        f"总市值 {q['market_cap_yi']:.2f} 亿元，PE(TTM) {q['pe']:.2f}，PB {q['pb']:.2f}。",
    ]

    m = _latest_metrics(code)
    if m:
        year = m.get("report_year")
        lines.append(
            f"【{year} 年报基本面】营收 {float(m['total_operate_income']) / 1e8:.2f} 亿元，"
            f"归母净利 {float(m['parent_netprofit']) / 1e8:.2f} 亿元，"
            f"毛利率 {float(m['gross_margin']):.2f}%，加权 ROE {float(m['roe_weighted']):.2f}%。")

    lines.append("注：行情为盘后/延时数据，仅供研究参考，不构成投资建议；"
                 "\"是否值得买\"需结合估值、持仓与风险偏好自行判断。")
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_quote("600519"))
    print(get_history("600519", 5)[-2:])
    print("\n" + answer("茅台今日股价和市值"))
