# -*- coding: utf-8 -*-
"""FinSight 年报解析管道：PDF → 父子块 JSONL（带公司/年份/章节/页码元数据）。

输出（默认 data/processed/）：
  children.jsonl   子块（入库 Milvus 的检索单元，300 字左右）
  parents.jsonl    父块（章节级窗口，1200 字左右，用于上下文回填）
  documents.jsonl  文档级清单
  parse_stats.json 解析统计

用法：
  python scripts/parse_annual_reports.py
  python scripts/parse_annual_reports.py --limit 3 --out .tmp_parse_test
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import fitz  # PyMuPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 年报章节标题：第X节 xxx（允许后面带内容）
SECT_RE = re.compile(r"^[ \t]*(第[一二三四五六七八九十]+节[^\n]{0,50})", re.M)
# 章节标题可能不在行首（PDF 抽取时混在页眉后），用关键词做二次兜底
SECTION_HINT_RE = re.compile(
    r"第[一二三四五六七八九十]+节\s*(?:重要提示|公司简介|管理层讨论|经营情况讨论|公司治理|"
    r"环境和社会|重要事项|股份变动|股东情况|优先股|债券|财务会计报告|财务报告|审计报告|备查文件)")
# 纯页码行（1 / - 12 - / — 3 —）
PAGE_NO_RE = re.compile(r"^\s*[-—\s]*\d{1,4}[-—\s]*\s*$")
# 目录名：000858_五粮液
COMPANY_DIR_RE = re.compile(r"^(\d{6})_(.+)$")
# 文件名前缀：2025_...
YEAR_RE = re.compile(r"^(\d{4})_")


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def norm_line(line):
    """归一化：数字→#、去空白，用于统计页眉页脚。"""
    s = re.sub(r"\d+", "#", line.strip())
    return re.sub(r"\s+", "", s)


def collect_line_freq(pages):
    freq = Counter()
    for text in pages:
        seen = set()
        for line in text.splitlines():
            key = norm_line(line)
            if key and len(key) <= 100 and key not in seen:
                seen.add(key)
                freq[key] += 1
    return freq


def clean_page(text, freq, npages):
    """去页码 + 去重复页眉页脚。"""
    out = []
    threshold = max(3, int(npages * 0.3))
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if PAGE_NO_RE.match(raw):
            continue
        key = norm_line(raw)
        if key and freq.get(key, 0) >= threshold and len(key) <= 100:
            continue
        out.append(raw)
    return "\n".join(out)


def is_toc_page(text):
    """判断目录页。

    注意：有些年报会把"优先股/债券/财务报告"几个短章节连着排在同一页，
    一页只出现 3 个标题并不一定是目录，所以这里要求更强证据。
    """
    count = len(SECT_RE.findall(text))
    if count >= 5:
        return True
    if count >= 3 and ("目录" in text or text.count("......") >= 3):
        return True
    return False


def detect_section(text):
    """返回本页第一个章节标题（没有则 None）。

    有些年报的标题会排版成两行（"第八节" + 下一行"财务报告"），
    这里对只有"第X节"的标题做一次向下看两行的补全。
    """
    match = SECT_RE.search(text)
    if not match:
        # 标题不在行首时（混在页眉/正文里），用关键词兜底
        hint = SECTION_HINT_RE.search(text)
        return hint.group(0).strip() if hint else None
    raw = re.sub(r"\s+", " ", match.group(1)).strip()
    if re.match(r"^第[一二三四五六七八九十]+节$", raw):
        tail = text[match.end():].splitlines()
        for line in tail[:4]:
            line = line.strip()
            if line and len(line) <= 40 and not PAGE_NO_RE.match(line):
                return "%s %s" % (raw, line)
    return raw


# A 股年报标准章节（用于把"第十节 / 第十节 财务报告 / 第八节财务报告"统一成同一名称）
CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}
CANONICAL_SECTIONS = {
    1: "第一节 重要提示、目录和释义",
    2: "第二节 公司简介和主要财务指标",
    3: "第三节 管理层讨论与分析",
    4: "第四节 公司治理",
    5: "第五节 环境和社会责任",
    6: "第六节 重要事项",
    7: "第七节 股份变动及股东情况",
    8: "第八节 优先股相关情况",
    9: "第九节 债券相关情况",
    10: "第十节 财务报告",
    11: "第十一节 备查文件目录",
}


def cn2int(text):
    """中文数字转 int（支持 一~九十九）。"""
    text = text.strip()
    if not text:
        return 0
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = CN_DIGITS.get(left, 0) * 10 if left else 10
        ones = CN_DIGITS.get(right, 0)
        return tens + ones
    return CN_DIGITS.get(text, 0)


# 关键词优先（有的公司没有优先股/债券章节，会导致章节号整体前移，不能只靠序号判断）
SECTION_KEYWORDS = [
    ("重要提示", 1), ("公司简介", 2), ("管理层讨论", 3), ("经营情况讨论", 3),
    ("公司治理", 4), ("环境和社会", 5), ("社会责任", 5), ("重要事项", 6),
    ("股份变动", 7), ("股东情况", 7), ("优先股", 8), ("债券", 9),
    ("财务会计报告", 10), ("财务报告", 10), ("审计报告", 10), ("备查文件", 11),
]


def canonical_section(raw):
    """把各种章节写法归一化为标准章节名（关键词优先，序号兜底）。"""
    compact = re.sub(r"\s+", "", raw)
    for keyword, num in SECTION_KEYWORDS:
        if keyword in compact:
            return CANONICAL_SECTIONS[num]
    m = re.match(r"第([一二三四五六七八九十]+)节", compact)
    if m:
        num = cn2int(m.group(1))
        if num in CANONICAL_SECTIONS:
            return CANONICAL_SECTIONS[num]
    return re.sub(r"\s+", " ", raw).strip()


def section_no_of(section):
    """章节名 -> 章节序号（0 表示前置/无法识别）。"""
    m = re.match(r"第([一二三四五六七八九十]+)节", str(section))
    return cn2int(m.group(1)) if m else 0


def _simple_split(text, size, overlap):
    """兜底切分：按句号/换行切，合并到 size 左右。"""
    sentences = re.split(r"(?<=[。！？；\n])", text)
    chunks, buf = [], ""
    for sent in sentences:
        if not sent:
            continue
        if len(buf) + len(sent) <= size:
            buf += sent
        else:
            if buf:
                chunks.append(buf)
            if len(sent) <= size:
                buf = sent
            else:
                step = max(1, size - overlap)
                for i in range(0, len(sent), step):
                    chunks.append(sent[i:i + size])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def split_children(text, size, overlap):
    """优先用项目里的中文递归切分器，失败则用兜底实现。"""
    try:
        from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter
        splitter = ChineseRecursiveTextSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
    except Exception:
        chunks = _simple_split(text, size, overlap)
    return [c.strip() for c in chunks if len(c.strip()) >= 20]


def doc_info(pdf_path):
    """从路径解析 代码/公司/年份。"""
    folder = os.path.basename(os.path.dirname(pdf_path))
    match = COMPANY_DIR_RE.match(folder)
    code, name = (match.group(1), match.group(2)) if match else ("", "unknown")
    year_match = YEAR_RE.match(os.path.basename(pdf_path))
    year = int(year_match.group(1)) if year_match else 0
    return code, name, year


def load_index(index_csv):
    """(code, year) -> 年报标题/公告日期。"""
    result = {}
    if not os.path.exists(index_csv):
        return result
    import csv
    with open(index_csv, "r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            result[(str(row.get("code", "")).strip(), str(row.get("year", "")).strip())] = {
                "title": row.get("title", ""),
                "announcement_date": row.get("announcement_date", ""),
            }
    return result


def split_page_sections(text):
    """把一页按章节标题切成若干段。

    年报里常见"上一节结尾 + 下一节开头"在同一页（例如"第七节 债券相关情况"
    只占几行，紧接着就是"第八节 财务报告"），如果按整页只取第一个标题，
    后面几百页财务报告都会被标成债券。
    """
    matches = list(SECT_RE.finditer(text))
    if not matches:
        hint = SECTION_HINT_RE.search(text)
        if hint:
            matches = [hint]
    if not matches:
        return [(text, None)]
    segments = []
    if matches[0].start() > 60:
        segments.append((text[:matches[0].start()], None))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[match.start():end].strip()
        if seg:
            segments.append((seg, detect_section(seg)))
    return segments


def parse_pdf(path, code, name, year, title, parent_size, child_size, overlap):
    doc = fitz.open(path)
    raw_pages = [page.get_text() or "" for page in doc]
    npages = len(raw_pages)
    doc.close()

    freq = collect_line_freq(raw_pages)
    pages = []
    current_section = "前置（封面/目录）"
    for pno, raw in enumerate(raw_pages, 1):
        if is_toc_page(raw):
            continue
        text = clean_page(raw, freq, npages)
        if len(text.strip()) < 30:
            continue
        for seg_text, heading in split_page_sections(text):
            if heading:
                current_section = canonical_section(heading)
            seg_text = seg_text.strip()
            if len(seg_text) < 20:
                continue
            pages.append({"page": pno, "section": current_section,
                          "section_no": section_no_of(current_section), "text": seg_text})

    # 父块：章节内连续页窗口（>= parent_size 左右）
    parents, page_to_parent = [], {}
    buf, start_page, cur_section = [], None, None

    def flush():
        nonlocal buf, start_page, cur_section
        if not buf:
            return
        parent_id = md5("%s|%s|%s|%s" % (code, year, cur_section, start_page))
        parent_text = "\n".join("[P%d]\n%s" % (pno, t) for pno, t in buf)
        parents.append({
            "id": parent_id,
            "text": parent_text,
            "metadata": {
                "source": name, "company_code": code, "company_name": name,
                "year": year, "section": cur_section, "section_no": section_no_of(cur_section),
                "start_page": start_page, "end_page": buf[-1][0],
                "parent_id": parent_id, "doc_title": title, "file_path": path,
            },
        })
        for pno, _ in buf:
            page_to_parent[pno] = parent_id
        buf, start_page, cur_section = [], None, None

    for item in pages:
        if buf and (item["section"] != cur_section
                    or sum(len(x[1]) for x in buf) + len(item["text"]) > parent_size * 1.3):
            flush()
        if not buf:
            start_page = item["page"]
            cur_section = item["section"]
        buf.append((item["page"], item["text"]))
    flush()

    # 子块：按页切分，页码精确到页
    children = []
    for item in pages:
        parent_id = page_to_parent.get(item["page"])
        if not parent_id:
            continue
        for chunk in split_children(item["text"], child_size, overlap):
            child_id = md5("%s|%s|%s" % (parent_id, item["page"], chunk))
            children.append({
                "id": child_id,
                "text": chunk,
                "metadata": {
                    "source": name, "company_code": code, "company_name": name,
                    "year": year, "section": item["section"],
                    "section_no": item.get("section_no", 0), "page": item["page"],
                    "parent_id": parent_id, "doc_title": title, "file_path": path,
                },
            })

    document = {
        "code": code, "name": name, "year": year, "title": title,
        "file_path": path, "pages": npages,
    }
    return document, parents, children


def main():
    parser = argparse.ArgumentParser(description="年报 PDF 解析为父子块")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "processed"))
    parser.add_argument("--limit", type=int, default=0, help="只解析前 N 份（调试用）")
    parser.add_argument("--parent-size", type=int, default=1200)
    parser.add_argument("--child-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    index = load_index(os.path.join(args.data_dir, "index", "annual_report_index.csv"))
    pdfs = sorted(glob.glob(os.path.join(args.data_dir, "raw", "annual_reports", "**", "*.pdf"),
                            recursive=True))
    if args.limit:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        print("[ERROR] 没有找到 PDF，请检查 --data-dir：", args.data_dir)
        return 1

    os.makedirs(args.out, exist_ok=True)
    documents, all_parents, all_children = [], [], []
    per_company = defaultdict(lambda: {"docs": 0, "pages": 0, "parents": 0, "children": 0})

    for i, pdf in enumerate(pdfs, 1):
        code, name, year = doc_info(pdf)
        meta = index.get((code, str(year)), {})
        title = meta.get("title") or os.path.basename(pdf)
        try:
            document, parents, children = parse_pdf(
                pdf, code, name, year, title,
                args.parent_size, args.child_size, args.overlap)
        except Exception as exc:  # noqa: BLE001
            print("[ERROR] %s -> %s" % (pdf, exc))
            continue
        documents.append(document)
        all_parents.extend(parents)
        all_children.extend(children)
        stat = per_company[name]
        stat["docs"] += 1
        stat["pages"] += document["pages"]
        stat["parents"] += len(parents)
        stat["children"] += len(children)
        print("[%02d/%02d] %s %s %s -> %d 页 / %d 父块 / %d 子块" % (
            i, len(pdfs), code, name, year, document["pages"], len(parents), len(children)))

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump(os.path.join(args.out, "children.jsonl"), all_children)
    dump(os.path.join(args.out, "parents.jsonl"), all_parents)
    dump(os.path.join(args.out, "documents.jsonl"), documents)

    stats = {
        "pdf_count": len(documents),
        "pages": sum(d["pages"] for d in documents),
        "parents": len(all_parents),
        "children": len(all_children),
        "avg_child_chars": round(
            sum(len(c["text"]) for c in all_children) / max(len(all_children), 1), 1),
        "avg_parent_chars": round(
            sum(len(p["text"]) for p in all_parents) / max(len(all_parents), 1), 1),
        "sections": len({p["metadata"]["section"] for p in all_parents}),
        "per_company": dict(sorted(per_company.items())),
    }
    with open(os.path.join(args.out, "parse_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print("\n完成：%d 份 PDF / %d 页 / %d 父块 / %d 子块" % (
        stats["pdf_count"], stats["pages"], stats["parents"], stats["children"]))
    print("输出目录：", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
