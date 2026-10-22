# -*- coding: utf-8 -*-
"""探测 ECharts 的可用下载源（jsdelivr 在本机不通，多试几个镜像）。"""
import requests

CANDIDATES = [
    "https://cdnjs.cloudflare.com/ajax/libs/echarts/5.3.3/echarts.min.js",
    "https://unpkg.com/echarts@5.5.1/dist/echarts.min.js",
    "https://registry.npmmirror.com/echarts/5.5.1/files/dist/echarts.min.js",
    "https://cdn.bootcdn.net/ajax/libs/echarts/5.5.1/echarts.min.js",
    "https://npm.elemecdn.com/echarts@5.5.1/dist/echarts.min.js",
    "https://lf9-cdn-tos.bytecdntp.com/cdn/expire-1-M/echarts/5.5.1/echarts.min.js",
]

session = requests.Session()
session.trust_env = False
session.headers.update({"User-Agent": "Mozilla/5.0"})

log = []
for url in CANDIDATES:
    try:
        r = session.get(url, timeout=45, stream=True)
        size = r.raw.read(200000, decode_content=True) and len(r.content)
        log.append("%-3s %8.1f KB  %s" % (r.status_code, size / 1024, url))
    except Exception as exc:  # noqa: BLE001
        log.append("ERR  %s: %s  %s" % (type(exc).__name__, str(exc)[:60], url))

with open(r"C:\pro\_echarts_probe.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
print("done")
