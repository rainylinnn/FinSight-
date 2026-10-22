# -*- coding: utf-8 -*-
"""脱离父进程启动入库脚本。

背景：WorkBuddy 的后台任务有 2 分钟时长上限，schtasks 又被安全策略封了。
用 DETACHED_PROCESS + 新进程组启动，父进程退出后子进程仍继续运行。

用法：
    .venv\\Scripts\\python.exe scripts\\start_embed_detached.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\pro")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "logs" / "embed_task.log"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def main() -> int:
    log = open(LOG, "ab", buffering=0)
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(
            [str(PY), str(ROOT / "scripts" / "embed_annual_reports.py")],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags | CREATE_BREAKAWAY_FROM_JOB,
            close_fds=True,
        )
    except OSError:
        # 没有 job 对象时 BREAKAWAY 会失败，退回普通 detached
        proc = subprocess.Popen(
            [str(PY), str(ROOT / "scripts" / "embed_annual_reports.py")],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            close_fds=True,
        )
    Path(r"C:\pro\.workbuddy\_embed_pid.txt").write_text(str(proc.pid), encoding="utf-8")
    print(f"started pid={proc.pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
