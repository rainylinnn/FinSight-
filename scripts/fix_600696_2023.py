# -*- coding: utf-8 -*-
"""修复 600696 岩石股份 2023 年报抓取错误。

原脚本的正则 "2023年年度报告" 命中了《...关于公司2023年年度报告的信息披露监管工作函的回复》，
下载到的是一份 6 页的函件。这里用更严格的筛选重新定位并替换。

用法：
  python scripts/fix_600696_2023.py            # 只打印候选
  python scripts/fix_600696_2023.py --apply    # 下载并更新索引
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from download_finance_data import (  # noqa: E402
    CNINFO_PDF_BASE, download_pdf, fetch_stock_list, pick_annual_reports, query_announcements,
)

CODE, NAME, YEAR = "600696", "岩石股份", 2023
DATA = Path(r"C:\pro\data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    stocks = fetch_stock_list()
    org_id = (stocks.get(CODE) or {}).get("orgId") or ("gssh" + CODE)
    print(f"orgId={org_id}")

    anns = query_announcements(CODE, org_id, "2024-01-01", "2025-12-31")
    print(f"公告数={len(anns)}")

    cands = []
    for a in anns:
        title = (a.get("announcementTitle") or "").strip()
        if "2023" not in title or "年度报告" not in title:
            continue
        bad = any(k in title for k in ("回复", "函", "摘要", "英文", "更正", "意见", "问询", "公告", "提示", "取消", "补充", "说明"))
        print(("  跳过 " if bad else "  候选 ") + title)
        if not bad:
            cands.append(a)

    if not cands:
        print("没有找到干净的候选，可能只有这一份（需人工处理）")
        return 1
    if not args.apply:
        print("dry-run：加 --apply 执行替换")
        return 0

    best = cands[0]
    title = best.get("announcementTitle") or ""
    ann_id = best.get("announcementId")
    folder = DATA / "raw" / "annual_reports" / f"{CODE}_{NAME}"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{YEAR}_{CODE}_{NAME}_{ann_id}.pdf"
    size = download_pdf(best.get("adjunctUrl") or "", str(dest))
    print(f"已下载 {dest.name} ({size/1024:.0f} KB)")

    # 更新索引 CSV：删掉旧的 600696/2023 行，插入新行
    idx = DATA / "index" / "annual_report_index.csv"
    with open(idx, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    old = [r for r in rows if r["code"] == CODE and str(r["year"]) == str(YEAR)]
    rows = [r for r in rows if not (r["code"] == CODE and str(r["year"]) == str(YEAR))]
    for r in old:
        p = DATA / r["local_path"]
        if p.exists() and p.name != dest.name:
            p.unlink()
            print(f"已删除旧文件 {p.name}")
    rows.append({
        "code": CODE, "name": NAME, "year": YEAR, "title": title,
        "announcement_date": "", "announcement_id": ann_id,
        "pdf_url": CNINFO_PDF_BASE + str(best.get("adjunctUrl") or "").lstrip("/"),
        "local_path": str(dest.relative_to(DATA)).replace("\\", "/"),
        "bytes": size,
    })
    with open(idx, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"索引已更新：{len(rows)} 行")
    return 0


if __name__ == "__main__":
    sys.stdout = open(r"C:\pro\.workbuddy\_fix600696.txt", "w", encoding="utf-8")
    sys.exit(main())
