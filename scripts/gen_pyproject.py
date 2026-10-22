# -*- coding: utf-8 -*-
"""根据 requirements.txt 生成 uv 用的 pyproject.toml。"""
import json
from pathlib import Path

REQ = Path(r"C:\pro\requirements.txt")
OUT = Path(r"C:\pro\pyproject.toml")

deps = []
for line in REQ.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    deps.append(line)

lines = [
    "[project]",
    'name = "finsight-baijiu"',
    'version = "0.1.0"',
    'description = "FinSight 酒研助手 · 白酒行业智能投研助手（年报 RAG + 财务指标 NL2SQL + 股东图谱）"',
    'requires-python = ">=3.12,<3.13"',
    "dependencies = [",
]
for d in deps:
    lines.append(f"    {json.dumps(d, ensure_ascii=False)},")
lines += [
    "]",
    "",
    "[tool.uv]",
    "# 这是应用项目（不是要发布的库），uv 不需要构建/打包本项目",
    "package = false",
    "",
    "[[tool.uv.index]]",
    'name = "tuna"',
    'url = "https://pypi.tuna.tsinghua.edu.cn/simple"',
    "default = true",
    "",
]
OUT.write_text("\n".join(lines), encoding="utf-8")
print("pyproject.toml written:", OUT, "| deps:", len(deps))
