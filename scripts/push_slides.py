# -*- coding: utf-8 -*-
"""批量提交页面：逐页 lint → upsert。用法：python _push.py 02 03 04"""
import json
import os
import subprocess
import sys
from pathlib import Path

NODE = r"C:\Users\黄瑞霖\.workbuddy\binaries\node\versions\22.22.2-3\node.exe"
ENTRY = r"C:\Users\黄瑞霖\.workbuddy\binaries\node\versions\22.22.2-3\node_modules\@tencent\slidep\dist\index.js"
PROJ = Path(r"C:\pro\docs\FinSight_项目汇报")
PPTX = str(PROJ / "FinSight_项目汇报.pptx")
LOG = Path(r"C:\pro\.workbuddy\_push.txt")


def run(args):
    r = subprocess.run(
        [NODE, ENTRY] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJ),
        timeout=180,
    )
    return (r.stdout or "") + (r.stderr or "")


def pick_json(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


lines = []
codes = sys.argv[1:] or ["01"]
for n in codes:
    dsl = f"slides/{n}.slide"
    if not (PROJ / dsl).exists():
        lines.append(f"{n}: MISSING FILE")
        continue

    lint_out = run(["lint", PPTX, "--dsl-file", dsl])
    lint_json = pick_json(lint_out)
    if not lint_json or not lint_json.get("ok"):
        diag = (lint_json or {}).get("diagnostics", "no-json")
        lines.append(f"{n}: LINT FAIL {str(diag)[:400]}")
        continue

    up_out = run(["upsert-dsl", PPTX, "--dsl-file", dsl])
    up_json = pick_json(up_out)
    if up_json and up_json.get("ok"):
        lines.append(f"{n}: OK page={up_json.get('page_index')} total={up_json.get('slide_count')}")
    else:
        lines.append(f"{n}: UPSERT FAIL {str(up_json)[:400]}")

LOG.write_text("\n".join(lines), encoding="utf-8")
print("done")
