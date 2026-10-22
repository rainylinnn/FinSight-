# -*- coding: utf-8 -*-
"""提交最后的清理并推送。"""
import os
import subprocess

env = dict(os.environ)
env["GIT_TERMINAL_PROMPT"] = "0"

steps = [
    ["git", "add", "-A"],
    ["git", "commit", "-q", "-m", "chore: 移除开发期临时检查脚本"],
    ["git", "-c", "credential.username=rainylinnn", "push", "origin", "main"],
]
out = []
for cmd in steps:
    r = subprocess.run(cmd, cwd=r"C:\pro", capture_output=True, env=env, timeout=300)
    out.append("### %s -> rc=%d" % (" ".join(cmd), r.returncode))
    so = r.stdout.decode("utf-8", "replace").strip()
    se = r.stderr.decode("utf-8", "replace").strip()
    if so:
        out.append(so)
    if se:
        out.append("[stderr] " + se)

r = subprocess.run(["git", "status", "-sb"], cwd=r"C:\pro", capture_output=True)
out.append("### status")
out.append(r.stdout.decode("utf-8", "replace").rstrip())

with open(r"C:\pro\_final_push.txt", "wb") as f:
    f.write("\n".join(out).encode("utf-8"))
print("ok")
