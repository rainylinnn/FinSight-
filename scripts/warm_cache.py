# -*- coding: utf-8 -*-
"""预热语义缓存：把答辩/演示最常问的问题提前跑一遍灌进缓存。

演示效果：同一个问题第二次问，直接秒回（跳过向量检索 + LLM 生成）。

运行（需要 MySQL / Milvus / LLM 都在线）：
    uv run python scripts/warm_cache.py
    uv run python scripts/warm_cache.py --list     # 只看问题清单
"""
import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 答辩高频问题：指标类 + 年报原文类
QUESTIONS = [
    "2025年营业收入最高的公司是哪些",
    "贵州茅台2025年营业收入和归母净利润分别是多少",
    "贵州茅台2023到2025年的毛利率和ROE变化趋势",
    "哪些公司ROE连续三年下降",
    "2025年营收同比正增长的公司有哪些",
    "五粮液2025年的经营情况怎么样",
    "茅台年报里提到了哪些经营风险",
    "2025年白酒行业整体情况如何",
    "五粮液年报里的渠道和经销商情况",
    "泸州老窖2025年的经营计划是什么",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="只打印问题清单，不跑")
    parser.add_argument("--repeat", action="store_true", help="再问一遍，对比命中耗时")
    args = parser.parse_args()

    if args.list:
        for i, q in enumerate(QUESTIONS, 1):
            print("%2d. %s" % (i, q))
        return

    from new_main import IntegratedQASystem

    t0 = time.time()
    qa = IntegratedQASystem()
    print("QA 系统初始化耗时 %.1f 秒" % (time.time() - t0))

    session_id = str(uuid.uuid4())
    cache = getattr(qa, "semantic_cache", None)
    if cache is None:
        print("语义缓存未启用（config.ini [cache] semantic_enabled=false）")
        return
    print("缓存后端: %s" % cache.backend)

    total_first = 0.0
    for i, q in enumerate(QUESTIONS, 1):
        t = time.time()
        answer = "".join(tok for tok, _ in qa.query(q, session_id=session_id))
        cost = time.time() - t
        total_first += cost
        status = "已缓存" if cache.stats()["writes"] else "未写入"
        print("[%2d/%d] %5.1fs  %s  %s" % (i, len(QUESTIONS), cost, status, q))
        if not answer:
            print("      （无回答）")

    print("\n首轮总耗时 %.1f 秒，平均 %.1f 秒/题" % (total_first, total_first / len(QUESTIONS)))
    print("缓存状态: %s" % cache.stats())

    if args.repeat:
        print("\n=== 第二轮（应当全部命中缓存）===")
        t = time.time()
        hit = 0
        for q in QUESTIONS:
            start = time.time()
            qa.query(q, session_id=session_id)
            cost = time.time() - start
            hit += 1
            print("%5.1fs  %s" % (cost, q))
        print("第二轮总耗时 %.1f 秒" % (time.time() - t))


if __name__ == "__main__":
    main()
