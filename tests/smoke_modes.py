# -*- coding: utf-8 -*-
"""功能模式冒烟：sql / compare / rag / graph / auto。"""
import os
import sys
import time

from new_main import IntegratedQASystem

CASES = [
    ("sql", "茅台2025年营收和净利率是多少？"),
    ("compare", "茅台、五粮液、汾酒2025年营收和净利对比"),
    ("rag", "茅台2025年报里提到了哪些经营风险？"),
    ("graph", "招商中证白酒基金持有哪几家白酒公司？"),
    ("auto", "2025年营业收入最高的3家公司是哪些？"),
]


def main() -> int:
    if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("QIANWEN_API_KEY")):
        print("[ERROR] 未设置 DEEPSEEK_API_KEY", flush=True)
        return 2
    qa = IntegratedQASystem()
    session = os.environ.get("SMOKE_SESSION", "mode-test")
    for mode, q in CASES:
        print(f"\n===== [{mode}] {q}", flush=True)
        t0 = time.time()
        chunks = list(qa.query(q, session_id=session, mode=mode))
        dt = time.time() - t0
        ans = "".join(t for t, _ in chunks if isinstance(t, str))
        print(f"[{dt:.1f}s] {ans}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
