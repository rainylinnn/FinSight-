# -*- coding: utf-8 -*-
"""用真实 BGE-M3 校准语义缓存阈值。

为什么要校准：相似度阈值定高了缓存形同虚设，定低了会把「营收最高」当成
「净利最高」直接复用错答案。这里实测三组问题的相似度：

  A 组 同义改写   -> 期望命中（相似度应该够高）
  B 组 不同指标   -> 期望不命中（相似度偏高，必须靠词面/指标门槛拦）
  C 组 无关/跨年  -> 期望不命中

跑完看输出里的建议阈值，再回填 config.ini 的 semantic_threshold。
运行：uv run python scripts/calibrate_cache_threshold.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Config  # noqa: E402

conf = Config()

# (问题 A, 问题 B, 分组, 期望)
PAIRS = [
    # A 组：同义改写，应该命中
    ("2025年营业收入最高的公司", "哪家公司2025年营收最高", "A-同义", "hit"),
    ("贵州茅台2025年营业收入是多少", "贵州茅台2025年营收多少", "A-同义", "hit"),
    ("茅台2025年毛利率", "贵州茅台2025年的毛利率是多少", "A-同义", "hit"),
    ("2025年ROE最高的前三家公司", "哪些公司2025年ROE排名前三", "A-同义", "hit"),
    ("五粮液2025年归母净利润", "五粮液2025年净利润是多少", "A-同义", "hit"),
    # B 组：不同指标，向量很接近但不能命中
    ("2025年营业收入最高的公司", "2025年归母净利润最高的公司", "B-异指标", "miss"),
    ("茅台2025年毛利率", "茅台2025年资产负债率", "B-异指标", "miss"),
    ("2025年营收TOP3", "2025年ROE TOP3", "B-异指标", "miss"),
    ("2025年营业收入最高的3家公司", "2025年净利润最高的3家公司", "B-异指标", "miss"),
    # C 组：无关 / 跨年，不该命中
    ("2025年营业收入最高的公司", "茅台年报里的经营风险有哪些", "C-无关", "miss"),
    ("贵州茅台2025年营业收入", "贵州茅台2023年营业收入", "C-跨年", "miss"),
    ("2025年营收排名", "白酒行业的竞争格局如何", "C-无关", "miss"),
]


def main():
    from milvus_model.hybrid import BGEM3EmbeddingFunction

    model_path = os.path.join(conf.MODEL_ROOT, "bge-m3")
    lines = ["模型目录: %s" % model_path]
    if not os.path.isdir(model_path):
        lines.append("!! 找不到 bge-m3 模型目录，跳过")
        _write(lines)
        return

    lines.append("正在加载 BGE-M3（CPU，较慢请等待）…")
    _write(lines)
    ef = BGEM3EmbeddingFunction(model_name_or_path=model_path, use_fp16=False, device="cpu")

    texts = []
    for a, b, _g, _e in PAIRS:
        texts.append(a)
        texts.append(b)
    lines.append("编码 %d 条文本…" % len(texts))
    _write(lines)

    dense = np.asarray(ef(texts)["dense"], dtype=np.float32)
    dense = dense / np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-9)

    # ---------- 第一部分：纯向量相似度 ----------
    hits, misses = [], []
    lines.append("=" * 68)
    lines.append("第一部分：纯向量余弦相似度")
    lines.append("=" * 68)
    for i, (a, b, group, expect) in enumerate(PAIRS):
        sim = float(np.dot(dense[2 * i], dense[2 * i + 1]))
        lines.append("[%s] cos=%.4f  期望=%s" % (group, sim, expect))
        lines.append("        A: %s" % a)
        lines.append("        B: %s" % b)
        (hits if expect == "hit" else misses).append(sim)

    if hits and misses:
        lines.append("")
        lines.append("同义组最低相似度 = %.4f" % min(hits))
        lines.append("异指标/无关组最高相似度 = %.4f" % max(misses))
        gap = min(hits) - max(misses)
        lines.append("区分度 gap = %.4f" % gap)
        if gap > 0:
            lines.append("→ 向量阈值可用，取中点 %.3f" % ((min(hits) + max(misses)) / 2))
        else:
            lines.append("→ gap <= 0：纯向量阈值不可用（异指标组比同义组还像），"
                         "必须依赖词面/指标/年份三道门槛兜底")

    # ---------- 第二部分：跑完整判定逻辑 ----------
    lines.append("")
    lines.append("=" * 68)
    lines.append("第二部分：用真实 BGE-M3 跑完整缓存判定（阈值 %.2f / Jaccard %.2f）"
                 % (conf.SEMANTIC_THRESHOLD, conf.SEMANTIC_MIN_JACCARD))
    lines.append("=" * 68)

    from services.semantic_cache import SemanticCache

    cache = SemanticCache(
        lambda ts: ef(ts)["dense"], conf=conf, redis_client=None)
    cache.backend = "memory"          # 只测判定逻辑，不落 Redis

    ok_count = 0
    for a, b, group, expect in PAIRS:
        cache.clear()
        cache.set(a, "占位答案", mode="auto", route="sql")
        got = cache.get(b, mode="auto")
        actual = "hit" if got else "miss"
        flag = "OK " if actual == expect else "!! "
        if actual == expect:
            ok_count += 1
        lines.append("%s[%s] A -> B  期望=%s 实际=%s%s"
                     % (flag, group, expect, actual,
                        ("  相似度=%.4f" % got["similarity"]) if got else ""))
        lines.append("        A: %s" % a)
        lines.append("        B: %s" % b)

    lines.append("")
    lines.append("判定正确：%d / %d" % (ok_count, len(PAIRS)))
    if ok_count < len(PAIRS):
        lines.append("（未命中的同义组属于漏命中：宁可多算一次，也不能给错答案）")
    _write(lines)


def _write(lines):
    with open(r"C:\pro\_calib.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
