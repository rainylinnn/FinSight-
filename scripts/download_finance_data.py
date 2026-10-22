# -*- coding: utf-8 -*-
"""
金融 RAG 项目数据下载器（白酒行业 20 家公司）
=================================================
数据来源（全部走 HTTP，本环境 HTTPS 被限制）：
  1) 巨潮资讯网  http://www.cninfo.com.cn   —— 年报 PDF（文档知识库）
  2) 东方财富    http://datacenter-web.eastmoney.com —— 财务指标、十大股东（结构化/图谱）

用法：
  python download_finance_data.py --out C:/pro/data
  python download_finance_data.py --out ./test_data --companies 600519,000858 --years 2025
  python download_finance_data.py --out C:/pro/data --skip-financials --skip-holders

特性：
  - 仅用标准库（urllib），无需 pip 安装
  - 断点续传（已存在且 >50KB 的 PDF 会跳过）
  - 失败重试 + 限速，单公司失败不影响其他公司
  - 自动生成 index/annual_report_index.csv、manifest.json、README.md
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from datetime import datetime

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# 白酒行业 20 家（A 股主要上市酒企，可自行增删）
DEFAULT_COMPANIES = [
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("000568", "泸州老窖"),
    ("600809", "山西汾酒"),
    ("002304", "洋河股份"),
    ("000596", "古井贡酒"),
    ("603369", "今世缘"),
    ("603198", "迎驾贡酒"),
    ("000799", "酒鬼酒"),
    ("600779", "水井坊"),
    # 2026-10 扩展：其余 A 股白酒标的
    ("600702", "舍得酒业"),
    ("603589", "口子窖"),
    ("600559", "老白干酒"),
    ("600197", "伊力特"),
    ("603919", "金徽酒"),
    ("000860", "顺鑫农业"),
    ("002646", "天佑德酒"),
    ("600199", "金种子酒"),
    ("000995", "皇台酒业"),
    ("600696", "岩石股份"),
]

CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STOCK_LIST = "http://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_PDF_BASE = "http://static.cninfo.com.cn/"
EM_FINANCE = "http://datacenter-web.eastmoney.com/api/data/v1/get"
EM_HOLDERS = "http://datacenter-web.eastmoney.com/api/data/v1/get"

ANNUAL_EXCLUDE = re.compile(r"摘要|英文|English|更正|取消|补充|说明|已取消|修订|意见|问询|公告|提示")
ANNUAL_PATTERN = re.compile(r"(\d{4})\s*年?年度报告")


def log(*args):
    print(*args, flush=True)


def http_request(url, data=None, headers=None, timeout=60, retries=4):
    """GET/POST 请求，返回 bytes；失败重试。"""
    hdr = {"User-Agent": UA}
    if headers:
        hdr.update(headers)
    body = None
    if isinstance(data, dict):
        body = urllib.parse.urlencode(data).encode("utf-8")
    elif isinstance(data, bytes):
        body = data
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log("  [retry %d/%d] %s -> %s" % (attempt, retries, url[:110], exc))
            time.sleep(1.2 * attempt + random.random())
    raise RuntimeError("request failed: %s (%s)" % (url, last_err))


def sanitize(name):
    return re.sub(r'[\\/:*?"<>|\r\n]+', "_", str(name)).strip(" ._") or "unnamed"


def fetch_stock_list():
    """巨潮全 A 股列表：code -> {orgId, name}"""
    raw = http_request(CNINFO_STOCK_LIST, timeout=90)
    obj = json.loads(raw.decode("utf-8", errors="ignore"))
    mapping = {}
    for item in obj.get("stockList", []) or []:
        mapping[str(item.get("code"))] = item
    return mapping


def query_announcements(code, org_id, start_date, end_date, max_pages=40, sleep=0.35):
    """按公司分页查询公告（seDate 过滤日期范围），返回公告列表。"""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": "http://www.cninfo.com.cn/new/disclosure/stock?stockCode=%s&orgId=%s" % (code, org_id),
    }
    results = []
    for page in range(1, max_pages + 1):
        form = {
            "pageNum": str(page),
            "pageSize": "30",
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "%s,%s" % (code, org_id),
            "searchkey": "",
            "seDate": "%s~%s" % (start_date, end_date),
            "tdate": "",
            "isHLtitle": "true",
        }
        raw = http_request(CNINFO_QUERY, data=form, headers=headers)
        obj = json.loads(raw.decode("utf-8", errors="ignore"))
        items = obj.get("announcements") or []
        if not items:
            break
        results.extend(items)
        if len(results) >= int(obj.get("totalAnnouncement") or 0):
            break
        time.sleep(sleep)
    return results


def pick_annual_reports(announcements, years):
    """从公告里挑各年份年度报告（排除摘要/英文/更正等）。"""
    picked = {}
    for ann in announcements:
        title = (ann.get("announcementTitle") or "").strip()
        if not title or ANNUAL_EXCLUDE.search(title):
            continue
        match = ANNUAL_PATTERN.search(title)
        if not match:
            continue
        year = int(match.group(1))
        if year not in years:
            continue
        # 同一年份出现多份时，优先标题完整度高的（取更长标题）
        old = picked.get(year)
        if old is None or len(title) > len(old.get("announcementTitle") or ""):
            picked[year] = ann
    return [picked[y] for y in sorted(picked)]


def download_pdf(adjunct_url, dest_path):
    """下载 PDF 到 dest_path；已存在且大于 50KB 则跳过。"""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 50 * 1024:
        log("    skip existing: %s (%.0f KB)" % (os.path.basename(dest_path), os.path.getsize(dest_path) / 1024))
        return os.path.getsize(dest_path)
    url = CNINFO_PDF_BASE + str(adjunct_url).lstrip("/")
    raw = http_request(url, timeout=180)
    if not raw.startswith(b"%PDF"):
        raise RuntimeError("downloaded content is not a PDF: %s (head=%r)" % (url, raw[:16]))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(raw)
    os.replace(tmp, dest_path)
    return len(raw)


def stock_suffix(code):
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    return "SZ"


def em_query(report_name, secucode, sort_column, page_size=50, extra_filter=""):
    flt = '(SECUCODE="%s")' % secucode
    if extra_filter:
        flt += extra_filter
    url = (EM_FINANCE + "?" + urllib.parse.urlencode({
        "reportName": report_name,
        "columns": "ALL",
        "filter": flt,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "sortColumns": sort_column,
        "sortTypes": "-1",
    }))
    raw = http_request(url, timeout=60)
    return json.loads(raw.decode("utf-8", errors="ignore"))


FIN_FIELDS = [
    ("REPORT_DATE", "report_date"),
    ("REPORT_YEAR", "report_year"),
    ("REPORT_TYPE", "report_type"),
    ("REPORT_DATE_NAME", "report_date_name"),
    ("NOTICE_DATE", "notice_date"),
    ("TOTALOPERATEREVE", "total_operate_income"),
    ("TOTALOPERATEREVETZ", "total_operate_income_yoy"),
    ("PARENTNETPROFIT", "parent_netprofit"),
    ("PARENTNETPROFITTZ", "parent_netprofit_yoy"),
    ("ROEJQ", "roe_weighted"),
    ("ROEKCJQ", "roe_deducted"),
    ("XSMLL", "gross_margin"),
    ("XSJLL", "net_margin"),
    ("ZCFZL", "debt_ratio"),
    ("EPSJB", "eps_basic"),
    ("BPS", "bps"),
    ("MGJYXJJE", "op_cashflow_per_share"),
    ("ROIC", "roic"),
]


def fetch_financials(code, name, years, out_root, sleep=0.4):
    secucode = "%s.%s" % (code, stock_suffix(code))
    raw_obj = em_query("RPT_F10_FINANCE_MAINFINADATA", secucode, "REPORT_DATE", page_size=60)
    os.makedirs(os.path.join(out_root, "structured", "financial_indicators"), exist_ok=True)
    raw_path = os.path.join(out_root, "structured", "financial_indicators", "%s_%s.json" % (code, sanitize(name)))
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(raw_obj, fh, ensure_ascii=False, indent=1)

    rows = []
    data = (raw_obj.get("result") or {}).get("data") or []
    for item in data:
        if item.get("REPORT_TYPE") != "年报":
            continue
        try:
            year = int(item.get("REPORT_YEAR") or 0)
        except (TypeError, ValueError):
            continue
        if years and year not in years:
            continue
        row = {"code": code, "name": name}
        for src, dst in FIN_FIELDS:
            row[dst] = item.get(src)
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("report_date") or ""), reverse=True)
    time.sleep(sleep)
    return rows, raw_path


HOLDER_FIELDS = [
    ("END_DATE", "end_date"),
    ("HOLDER_RANK", "rank"),
    ("HOLDER_NAME", "holder_name"),
    ("HOLD_NUM", "hold_num"),
    ("HOLD_NUM_RATIO", "hold_ratio"),
    ("HOLD_NUM_CHANGE", "hold_change"),
    ("HOLDER_CODE", "holder_code"),
]


def fetch_holders(code, name, out_root, sleep=0.4):
    secucode = "%s.%s" % (code, stock_suffix(code))
    raw_obj = em_query("RPT_F10_EH_HOLDERS", secucode, "END_DATE", page_size=100)
    os.makedirs(os.path.join(out_root, "structured", "holders"), exist_ok=True)
    raw_path = os.path.join(out_root, "structured", "holders", "%s_%s.json" % (code, sanitize(name)))
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(raw_obj, fh, ensure_ascii=False, indent=1)

    data = (raw_obj.get("result") or {}).get("data") or []
    # 按报告期分组：优先取"股东数 >= 5 的最近一期"（有些公司最新一期只更新了前两大股东）
    by_date = {}
    for item in data:
        date_key = str(item.get("END_DATE"))
        if not date_key or date_key == "None":
            continue
        by_date.setdefault(date_key, []).append(item)
    latest = None
    for date_key in sorted(by_date.keys(), reverse=True):
        if len(by_date[date_key]) >= 5:
            latest = date_key
            break
    if latest is None and by_date:
        latest = sorted(by_date.keys(), reverse=True)[0]
    rows = []
    for item in by_date.get(latest, []):
        row = {"code": code, "name": name}
        for src, dst in HOLDER_FIELDS:
            row[dst] = item.get(src)
        rows.append(row)
    def _rank(r):
        try:
            return int(r.get("rank") or 999)
        except (TypeError, ValueError):
            return 999
    rows.sort(key=_rank)
    rows = rows[:10]
    time.sleep(sleep)
    return rows, raw_path


def write_csv(path, rows, fields):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_readme(out_root, meta):
    path = os.path.join(out_root, "README.md")
    lines = [
        "# 金融 RAG 项目数据集（白酒行业）",
        "",
        "> 自动生成于 %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "## 目录结构",
        "```",
        "data/",
        "├── raw/annual_reports/<代码>_<简称>/      # 巨潮年报 PDF（知识库原始文档）",
        "├── structured/financial_indicators/      # 东财财务指标原始 JSON",
        "├── structured/holders/                   # 东财十大股东原始 JSON",
        "├── structured/financial_indicators.csv   # 汇总财务指标表（MySQL 导入用）",
        "├── structured/holders.csv                # 汇总十大股东表（图谱/关系用）",
        "├── index/annual_report_index.csv         # 年报清单（代码/年份/标题/URL/本地路径）",
        "├── companies.csv                         # 公司清单",
        "├── manifest.json                         # 下载清单与统计",
        "└── README.md",
        "```",
        "",
        "## 数据来源",
        "- 年报 PDF：巨潮资讯网 http://www.cninfo.com.cn （A 股法定信息披露平台）",
        "- 财务指标/十大股东：东方财富数据中心 http://datacenter-web.eastmoney.com",
        "",
        "## 规模统计",
        "- 公司数：%d" % meta["company_count"],
        "- 年报 PDF 数：%d（总 %.1f MB）" % (meta["pdf_count"], meta["pdf_mb"]),
        "- 财务指标年度记录：%d" % meta["financial_rows"],
        "- 十大股东记录：%d" % meta["holder_rows"],
        "- 年份范围：%s" % meta["years"],
        "",
        "## 使用说明",
        "1. 年报 PDF 可直接作为 RAG 知识库原始文档（PDF 解析 → 父子块切分 → 向量化）；",
        "2. financial_indicators.csv 导入 MySQL（如 finance_db.financial_indicators）做 NL2SQL；",
        "3. holders.csv 用于构建 Neo4j 股权关系图：`(股东)-[:持有]->(上市公司)`；",
        "4. annual_report_index.csv 存 PDF 与元数据映射，便于引用溯源到公司/年份/文件。",
        "",
        "## 合规说明",
        "- 年报为上市公司法定公开披露信息，仅用于学习/研究/演示，请注明来源，不再分发；",
        "- 东财数据为公开行情/财务数据，仅用于学习研究；",
        "- 抓取已限速（每次请求间隔 ~0.4-1s），请勿大规模爬取。",
        "",
        "## 复现 / 扩展",
        "```bash",
        "python scripts/download_finance_data.py --out C:/pro/data",
        "python scripts/download_finance_data.py --out C:/pro/data --years 2022,2023,2024,2025",
        "python scripts/download_finance_data.py --out C:/pro/data --companies 600519,000858",
        "```",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="下载金融 RAG 项目数据（白酒行业）")
    parser.add_argument("--out", default="C:/pro/data", help="输出数据目录")
    parser.add_argument("--companies", default="", help="公司代码逗号分隔，默认 20 家白酒公司")
    parser.add_argument("--years", default="2023,2024,2025", help="年报年份，逗号分隔")
    parser.add_argument("--start-date", default="2024-01-01", help="公告查询起始日期")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-12-31"), help="公告查询截止日期")
    parser.add_argument("--skip-financials", action="store_true")
    parser.add_argument("--skip-holders", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.6, help="PDF 下载间隔（秒）")
    args = parser.parse_args()

    out_root = os.path.abspath(args.out)
    years = sorted({int(y.strip()) for y in args.years.split(",") if y.strip()})
    if args.companies.strip():
        codes = [c.strip() for c in args.companies.split(",") if c.strip()]
        name_map = dict(DEFAULT_COMPANIES)
        companies = [(c, name_map.get(c, c)) for c in codes]
    else:
        companies = list(DEFAULT_COMPANIES)

    log("=" * 70)
    log("输出目录：%s" % out_root)
    log("公司数：%d；年报年份：%s" % (len(companies), years))
    log("=" * 70)
    os.makedirs(out_root, exist_ok=True)

    # 0) 巨潮股票列表（取 orgId）
    try:
        stock_map = fetch_stock_list()
        log("巨潮股票列表加载成功：%d 只" % len(stock_map))
    except Exception as exc:  # noqa: BLE001
        log("[WARN] 股票列表获取失败，将使用内置 orgId 猜测：%s" % exc)
        stock_map = {}

    annual_index = []          # 年报清单
    fin_rows = []
    holder_rows = []
    stats = {"companies_ok": 0, "companies_fail": 0, "pdf_bytes": 0, "pdf_count": 0}

    for code, name in companies:
        log("\n[%s %s] ====================" % (code, name))
        org_id = (stock_map.get(code) or {}).get("orgId") or ""
        # 兜底 orgId（深市 gssz+code，沪市 gssh+code；个别公司不适用时会被公告查询返回空）
        if not org_id:
            org_id = ("gssh" if stock_suffix(code) == "SH" else "gssz") + code.zfill(6)
        company_ok = True

        # 1) 年报公告 -> PDF
        try:
            anns = query_announcements(code, org_id, args.start_date, args.end_date)
            log("  公告数：%d" % len(anns))
            reports = pick_annual_reports(anns, years)
            log("  命中年度报告：%s" % [(int(ANNUAL_PATTERN.search(a["announcementTitle"]).group(1)),
                                        a["announcementTitle"]) for a in reports])
            for ann in reports:
                title = ann.get("announcementTitle") or ""
                year = int(ANNUAL_PATTERN.search(title).group(1))
                ann_id = ann.get("announcementId") or "unknown"
                folder = os.path.join(out_root, "raw", "annual_reports", "%s_%s" % (code, sanitize(name)))
                filename = "%d_%s_%s_%s.pdf" % (year, code, sanitize(name), sanitize(ann_id))
                dest = os.path.join(folder, filename)
                size = download_pdf(ann.get("adjunctUrl") or "", dest)
                stats["pdf_count"] += 1
                stats["pdf_bytes"] += size
                ts = ann.get("announcementTime")
                date_str = ""
                if ts:
                    try:
                        date_str = datetime.fromtimestamp(int(ts) / 1000).strftime("%Y-%m-%d")
                    except Exception:  # noqa: BLE001
                        date_str = str(ts)
                annual_index.append({
                    "code": code, "name": name, "year": year, "title": title,
                    "announcement_date": date_str, "announcement_id": ann_id,
                    "pdf_url": CNINFO_PDF_BASE + str(ann.get("adjunctUrl") or "").lstrip("/"),
                    "local_path": os.path.relpath(dest, out_root).replace("\\", "/"),
                    "bytes": size,
                })
                log("    [OK] %s (%.0f KB)" % (filename, size / 1024))
                time.sleep(args.sleep + random.random() * 0.3)
        except Exception as exc:  # noqa: BLE001
            company_ok = False
            log("  [ERROR] 年报下载失败：%s" % exc)

        # 2) 财务指标
        if not args.skip_financials:
            try:
                rows, raw_path = fetch_financials(code, name, years, out_root)
                fin_rows.extend(rows)
                log("  [OK] 财务指标 %d 条（原始 JSON：%s）" % (len(rows), os.path.basename(raw_path)))
            except Exception as exc:  # noqa: BLE001
                company_ok = False
                log("  [ERROR] 财务指标失败：%s" % exc)

        # 3) 十大股东
        if not args.skip_holders:
            try:
                rows, raw_path = fetch_holders(code, name, out_root)
                holder_rows.extend(rows)
                log("  [OK] 十大股东 %d 条（原始 JSON：%s）" % (len(rows), os.path.basename(raw_path)))
            except Exception as exc:  # noqa: BLE001
                company_ok = False
                log("  [ERROR] 十大股东失败：%s" % exc)

        if company_ok:
            stats["companies_ok"] += 1
        else:
            stats["companies_fail"] += 1

    # 汇总 CSV
    write_csv(os.path.join(out_root, "index", "annual_report_index.csv"),
              annual_index,
              ["code", "name", "year", "title", "announcement_date", "announcement_id",
               "pdf_url", "local_path", "bytes"])
    write_csv(os.path.join(out_root, "structured", "financial_indicators.csv"),
              fin_rows,
              ["code", "name"] + [dst for _, dst in FIN_FIELDS])
    write_csv(os.path.join(out_root, "structured", "holders.csv"),
              holder_rows,
              ["code", "name"] + [dst for _, dst in HOLDER_FIELDS])
    write_csv(os.path.join(out_root, "companies.csv"),
              [{"code": c, "name": n} for c, n in companies],
              ["code", "name"])

    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": out_root,
        "companies": [{"code": c, "name": n} for c, n in companies],
        "years": years,
        "stats": stats,
        "annual_reports": annual_index,
        "financial_rows": len(fin_rows),
        "holder_rows": len(holder_rows),
        "sources": {
            "annual_reports": "http://www.cninfo.com.cn (巨潮资讯网)",
            "financial_indicators": "http://datacenter-web.eastmoney.com (东方财富)",
            "holders": "http://datacenter-web.eastmoney.com (东方财富)",
        },
        "notes": [
            "仅用 HTTP 接口下载；HTTPS 在下载环境被限制。",
            "PDF 已做 %PDF 头校验；重复运行会跳过已下载文件（断点续传）。",
            "数据仅用于学习/研究/演示，请勿再分发。",
        ],
    }
    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    write_readme(out_root, {
        "company_count": len(companies),
        "pdf_count": stats["pdf_count"],
        "pdf_mb": stats["pdf_bytes"] / 1024 / 1024,
        "financial_rows": len(fin_rows),
        "holder_rows": len(holder_rows),
        "years": ",".join(str(y) for y in years),
    })

    log("\n" + "=" * 70)
    log("完成：成功 %d 家，失败 %d 家；PDF %d 份（%.1f MB）；财务 %d 行；股东 %d 行"
        % (stats["companies_ok"], stats["companies_fail"], stats["pdf_count"],
           stats["pdf_bytes"] / 1024 / 1024, len(fin_rows), len(holder_rows)))
    log("输出目录：%s" % out_root)
    log("=" * 70)
    return 0 if stats["companies_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
