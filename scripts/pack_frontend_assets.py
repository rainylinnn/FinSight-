# -*- coding: utf-8 -*-
"""把前端依赖的 CDN 资源下载到本地 static/vendor/，实现离线可用。

背景：前端原来直接引 CDN（marked.js / ECharts / Font Awesome），
答辩现场断网或 CDN 被墙时页面就白板。下载到本地后完全离线可用。
"""
import os
import sys

import requests

OUT = r"C:\pro\static\vendor"
FILES = [
    # (url, 本地相对路径)
    # 优先用 cdnjs（jsdelivr 在本机网络下连不上，cdnjs 实测可达）
    ("https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js",
     "marked/marked.min.js"),
    ("https://registry.npmmirror.com/echarts/5.5.1/files/dist/echarts.min.js",
     "echarts/echarts.min.js"),
    ("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css",
     "fontawesome/css/all.min.css"),
    ("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/webfonts/fa-solid-900.woff2",
     "fontawesome/webfonts/fa-solid-900.woff2"),
    ("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/webfonts/fa-regular-400.woff2",
     "fontawesome/webfonts/fa-regular-400.woff2"),
    ("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/webfonts/fa-brands-400.woff2",
     "fontawesome/webfonts/fa-brands-400.woff2"),
]

log = []


def main():
    session = requests.Session()
    # 绕过 Windows 系统代理（本机代理地址经常失效，直连更稳）
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0 FinSight-offline-packager"})

    for url, rel in FILES:
        dest = os.path.join(OUT, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code != 200:
                log.append("FAIL %s HTTP %s" % (rel, resp.status_code))
                continue
            with open(dest, "wb") as f:
                f.write(resp.content)
            log.append("OK   %-46s %8.1f KB" % (rel, len(resp.content) / 1024))
        except Exception as exc:  # noqa: BLE001
            log.append("FAIL %s -> %s: %s" % (rel, type(exc).__name__, exc))

    with open(r"C:\pro\_vendor.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("done")


if __name__ == "__main__":
    main()
