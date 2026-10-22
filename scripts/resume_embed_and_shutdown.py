# -*- coding: utf-8 -*-
"""等 Milvus 就绪 → 跑完年报入库 → 自动关机。

Docker Desktop 停机会把 Milvus 容器一起带走，入库脚本会直接退出。
所以先确认 19530 端口通了，再开跑。

用法：
    .venv\\Scripts\\python.exe scripts\\resume_embed_and_shutdown.py
    .venv\\Scripts\\python.exe scripts\\resume_embed_and_shutdown.py --no-shutdown
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def milvus_ready(host: str = "127.0.0.1", port: int = 19530, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="续跑 Milvus 年报入库，完成后关机")
    parser.add_argument("--no-shutdown", action="store_true", help="跑完不关机")
    parser.add_argument("--wait", type=int, default=180, help="等待 Milvus 的最长秒数")
    parser.add_argument("--delay", type=int, default=120, help="关机前留的秒数")
    args = parser.parse_args()

    print(f"等待 Milvus（最多 {args.wait} 秒）...")
    waited = 0
    while waited < args.wait:
        if milvus_ready():
            print(f"Milvus 已就绪（等待 {waited} 秒）")
            break
        time.sleep(3)
        waited += 3
    else:
        print("Milvus 仍未就绪。请先启动 Docker Desktop，等鲸鱼图标变绿后重跑本脚本。")
        return 2

    print("开始入库（已入库的子块会自动跳过）...")
    code = subprocess.call([str(PY), str(ROOT / "scripts" / "embed_annual_reports.py")])
    print(f"入库脚本退出码：{code}")
    if code != 0:
        print("入库未成功，不执行关机。")
        return code

    if args.no_shutdown:
        print("入库完成，已按 --no-shutdown 跳过关机。")
        return 0

    print(f"入库完成，{args.delay} 秒后关机（取消：shutdown /a）")
    os.system(f"shutdown /s /t {args.delay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
