# -*- coding: utf-8 -*-
"""校验本地化的前端资源：文件齐全 + CSS 里的 webfont 引用都能落到实际文件上。"""
import os
import re

ROOT = r"C:\pro\static\vendor"
CSS = os.path.join(ROOT, "fontawesome", "css", "all.min.css")

log = []

needed = [
    "marked/marked.min.js",
    "echarts/echarts.min.js",
    "fontawesome/css/all.min.css",
]
for rel in needed:
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    size = os.path.getsize(p) if os.path.exists(p) else -1
    log.append("%-46s %s" % (rel, ("OK  %8.1f KB" % (size / 1024)) if size >= 0 else "缺失"))

if os.path.exists(CSS):
    with open(CSS, encoding="utf-8") as f:
        css = f.read()
    refs = set(re.findall(r'url\(\.\./webfonts/([^)"\']+)\)', css))
    log.append("")
    log.append("CSS 里引用的 webfont：%d 个" % len(refs))
    for r in sorted(refs):
        p = os.path.join(ROOT, "fontawesome", "webfonts", r)
        ok = os.path.exists(p)
        log.append("  %-32s %s" % (r, "OK" if ok else "缺失（图标会退化成方块）"))

# index.html 里不该再有外链 CDN
html = open(r"C:\pro\static\index.html", encoding="utf-8").read()
log.append("")
ext = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
log.append("index.html 里的外链资源：%d 个 %s" % (len(ext), ext if ext else "（已完全本地化）"))

with open(r"C:\pro\_vendor_check.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
print("done")
