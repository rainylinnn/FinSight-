# -*- coding: utf-8 -*-
"""端到端冒烟：IntegratedQASystem 全链路（路由 -> BM25/缓存 -> RAG/SQL -> 千问）。"""
import os
import sys
import time

from new_main import IntegratedQASystem

QUESTIONS = [
    "贵州茅台2025年营业收入和毛利率是多少？",
    "五粮液2025年营业收入是多少？",
    "2025年营业收入最高的3家公司是哪些？",
    "茅台2025年年报的管理层讨论与分析中，提到了哪些经营风险？",
    "洋河股份2025年年报里渠道库存和动销是怎么描述的？",
]


def main() -> int:
    if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("QIANWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")):
        print("[ERROR] 未设置 DEEPSEEK_API_KEY / QIANWEN_API_KEY", flush=True)
        return 2
    questions = sys.argv[1:] or QUESTIONS
    session_id = os.environ.get("SMOKE_SESSION", "smoke-e2e")
    t0 = time.time()
    qa = IntegratedQASystem()
    print(f"system ready in {time.time() - t0:.1f}s | session={session_id}", flush=True)

    for i, q in enumerate(questions, 1):
        print(f"\n===== Q{i}: {q}", flush=True)
        t0 = time.time()
        try:
            chunks = list(qa.query(q, session_id=session_id))
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {type(e).__name__}: {e}", flush=True)
            continue
        dt = time.time() - t0
        ans = "".join(t for t, _done in chunks if isinstance(t, str))
        print(f"[{dt:.2f}s] {ans}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
