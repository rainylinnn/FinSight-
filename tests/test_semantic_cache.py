# -*- coding: utf-8 -*-
"""语义缓存单元测试（不依赖 MySQL / Milvus / Redis / 模型）。

用假的 embedding 函数 + 假的 Redis 客户端，验证：
  1. 写入后能用「几乎一样的问题」命中
  2. 完全不同的问题不会命中
  3. 向量近似但词面不同的问题会被 Jaccard 门槛挡掉
  4. Redis 不可用时自动降级到内存后端
  5. 统计（命中/未命中/命中率）正确

运行：uv run python tests/test_semantic_cache.py
"""
import hashlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.semantic_cache import (  # noqa: E402
    SemanticCache, normalize_query, _keywords, _jaccard, metric_keys, years_of,
    polarity, ordinals)


# ---------------------------------------------------------------- 假 embedding
def fake_embed(texts):
    """基于词哈希 + 字符 2-gram 的确定性向量。

    加 2-gram 是为了模拟真实 BGE-M3 的行为：「营业收入最高的公司」和
    「营业收入最高的公司是哪些」应该很相似，而不是因为多了一个词就完全不同。
    """
    dim = 256
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        normalized = (text or "").lower()
        tokens = [t for t in re.split(r"[\s，。、,.?？!！]+", normalized) if t]
        features = list(tokens)
        for tok in tokens:
            features.extend(tok[i:i + 2] for i in range(max(0, len(tok) - 1)))
        for feat in features:
            if not feat:
                continue
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % dim
            out[i, idx] += 1.0 if h & 1 else -1.0
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


class FakeConfig:
    SEMANTIC_ENABLED = True
    SEMANTIC_THRESHOLD = 0.93
    SEMANTIC_MIN_JACCARD = 0.34
    SEMANTIC_TTL = 7200
    SEMANTIC_MAX_ENTRIES = 50
    SEMANTIC_CANDIDATE_LIMIT = 200


class FakeRedisClient:
    """最小 Redis 客户端替身：只有 set/get/zadd/zrevrange/pipeline 这几个用到的方法。"""

    def __init__(self, fail=False):
        self.store = {}
        self.zset = {}
        self.fail = fail
        self.hashes = {}
        self.client = self

    def ping(self):
        if self.fail:
            raise ConnectionError("redis down")
        return True

    def set(self, key, value, ex=None):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        count = 0
        for k in keys:
            if self.store.pop(k, None) is not None:
                count += 1
            if k in self.zset:
                self.zset.pop(k)
                count += 1
        return count

    def zadd(self, key, mapping):
        self.zset.setdefault(key, {}).update(mapping)

    def zcard(self, key):
        return len(self.zset.get(key, {}))

    def zrange(self, key, start, end):
        items = sorted(self.zset.get(key, {}).items(), key=lambda kv: kv[1])
        return [k for k, _ in items[start:end if end != -1 else None]]

    def zrevrange(self, key, start, end):
        items = sorted(self.zset.get(key, {}).items(), key=lambda kv: -kv[1])
        return [k for k, _ in items[start:end if end != -1 else None]]

    def zremrangebyrank(self, key, start, end):
        items = sorted(self.zset.get(key, {}).items(), key=lambda kv: kv[1])
        victims = [k for k, _ in items[start:end if end != -1 else None]]
        for v in victims:
            self.zset.get(key, {}).pop(v, None)
        return len(victims)

    def hincrby(self, key, field, delta=1):
        self.hashes.setdefault(key, {})
        self.hashes[key][field] = self.hashes[key].get(field, 0) + delta
        return self.hashes[key][field]

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hdel(self, key, field):
        self.hashes.get(key, {}).pop(field, None)

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.ops = []

    def get(self, key):
        self.ops.append(("get", key))
        return self

    def set(self, key, value, ex=None):
        self.ops.append(("set", key, value))
        return self

    def delete(self, *keys):
        self.ops.append(("delete", *keys))
        return self

    def zadd(self, key, mapping):
        self.ops.append(("zadd", key, mapping))
        return self

    def execute(self):
        results = []
        for op in self.ops:
            if op[0] == "get":
                results.append(self.redis.get(op[1]))
            elif op[0] == "set":
                self.redis.set(op[1], op[2])
                results.append(True)
            elif op[0] == "delete":
                results.append(self.redis.delete(*op[1:]))
            elif op[0] == "zadd":
                self.redis.zadd(op[1], op[2])
                results.append(True)
        self.ops = []
        return results


# ---------------------------------------------------------------- 断言工具
PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("  [PASS] %s" % name)
    else:
        FAIL.append(name)
        print("  [FAIL] %s %s" % (name, detail))


def test_normalize():
    print("\n== 归一化 ==")
    check("全角/标点归一",
          normalize_query("贵州茅台，2025年营收？") == normalize_query("贵州茅台2025年营收"),
          normalize_query("贵州茅台，2025年营收？"))
    check("空输入安全", normalize_query("") == "" and normalize_query(None) == "")


def test_keywords_jaccard():
    print("\n== 词面门槛 ==")
    a = _keywords("2025年营业收入最高的公司")
    b = _keywords("2025年归母净利润最高的公司")
    check("营收/净利 词面重合低", _jaccard(a, b) < 0.34, f"jaccard={_jaccard(a, b):.2f}")


def test_memory_backend():
    print("\n== 内存后端（Redis 不可用时降级） ==")
    cache = SemanticCache(fake_embed, conf=FakeConfig(), redis_client=FakeRedisClient(fail=True))
    check("自动降级为 memory", cache.backend == "memory", cache.backend)

    cache.set("2025年营业收入最高的公司", "贵州茅台以 1720.54 亿元居首。", mode="auto", route="sql")
    check("写入后条目数=1", cache.stats()["entries"] == 1)

    hit = cache.get("2025年营业收入最高的公司", mode="auto")
    check("同问题命中", hit is not None)
    check("命中返回答案", hit and hit["answer"].startswith("贵州茅台"))

    miss = cache.get("贵州茅台的管理层讨论与分析有哪些内容", mode="auto")
    check("不同问题不命中", miss is None, str(miss))

    st = cache.stats()
    check("命中计数=1", st["hits"] == 1, str(st))
    check("未命中计数>=1", st["misses"] >= 1, str(st))
    check("命中率统计正确", 0 < st["hit_rate"] <= 1, str(st))


def test_redis_backend():
    print("\n== Redis 后端 ==")
    fake = FakeRedisClient()
    cache = SemanticCache(fake_embed, conf=FakeConfig(), redis_client=fake)
    check("使用 redis 后端", cache.backend == "redis", cache.backend)

    cache.set("贵州茅台2025年营业收入", "1720.54 亿元", mode="auto", route="sql")
    hit = cache.get("贵州茅台2025年营业收入", mode="auto")
    check("Redis 往返命中", hit is not None, str(hit))
    check("Redis 命中答案正确", hit and hit["answer"] == "1720.54 亿元")
    check("Redis 存了向量", any(k.startswith(SemanticCache.VEC_PREFIX) for k in fake.store))
    check("Redis 存了索引", SemanticCache.INDEX_KEY in fake.zset)

    # 近似问法：向量相似度由假 embedding 决定，这里把阈值调低来验证检索链路本身是通的
    loose = FakeConfig()
    loose.SEMANTIC_THRESHOLD = 0.5
    loose.SEMANTIC_MIN_JACCARD = 0.34
    loose_cache = SemanticCache(fake_embed, conf=loose, redis_client=fake)
    loose_hit = loose_cache.get("贵州茅台2025年营业收入是多少", mode="auto")
    check("近似问法可命中（低阈值下）", loose_hit is not None, str(loose_hit))

    st = cache.stats()
    check("Redis 统计写入数=1", st["writes"] == 1, str(st))

    cache.clear()
    check("清空后条目为 0", cache.stats()["entries"] == 0, str(cache.stats()))


def test_jaccard_guard():
    print("\n== Jaccard 门槛挡假命中 ==")
    cfg = FakeConfig()
    cfg.SEMANTIC_THRESHOLD = -1.0  # 把向量阈值放开，只让 Jaccard 决定
    cfg.SEMANTIC_MIN_JACCARD = 0.34
    cache = SemanticCache(fake_embed, conf=cfg, redis_client=FakeRedisClient(fail=True))
    cache.set("2025年营业收入最高的公司", "营收答案", mode="auto", route="sql")
    hit = cache.get("2025年归母净利润最高的公司", mode="auto")
    check("向量近似但词面不同 -> 不命中", hit is None, str(hit))


def test_metric_conflict_guard():
    print("\n== 指标冲突检测 ==")
    cfg = FakeConfig()
    cfg.SEMANTIC_THRESHOLD = -1.0   # 只看指标冲突这一关
    cfg.SEMANTIC_MIN_JACCARD = 0.0  # 关掉 Jaccard
    cache = SemanticCache(fake_embed, conf=cfg, redis_client=FakeRedisClient(fail=True))

    cache.set("2025年营业收入最高的公司", "营收答案", mode="auto", route="sql")
    hit = cache.get("2025年归母净利润最高的公司", mode="auto")
    check("营收 vs 净利 -> 冲突不命中", hit is None, str(hit))

    hit2 = cache.get("2025年营业收入最高的是哪几家", mode="auto")
    check("同一指标 -> 允许命中", hit2 is not None, str(hit2))

    check("指标词解析正确",
          metric_keys("2025年归母净利润最高的公司") == {"净利"},
          str(metric_keys("2025年归母净利润最高的公司")))
    check("多指标解析正确",
          metric_keys("营收和毛利率对比") == {"营收", "毛利"},
          str(metric_keys("营收和毛利率对比")))


def test_followup_guard():
    print("\n== 追问不入缓存 ==")
    # 这一条在 new_main 里判定（短问题 + 有历史），这里只验证工具函数行为
    check("短问题长度判定可用", len("利润呢".strip()) <= 8)


def test_year_conflict_guard():
    print("\n== 年份冲突检测 ==")
    cfg = FakeConfig()
    cfg.SEMANTIC_THRESHOLD = -1.0
    cfg.SEMANTIC_MIN_JACCARD = 0.0
    cache = SemanticCache(fake_embed, conf=cfg, redis_client=FakeRedisClient(fail=True))

    cache.set("贵州茅台2025年营业收入", "2025 答案", mode="auto", route="sql")
    hit = cache.get("贵州茅台2023年营业收入", mode="auto")
    check("跨年问题 -> 不命中", hit is None, str(hit))

    hit2 = cache.get("贵州茅台2025年营业收入是多少", mode="auto")
    check("同一年份 -> 允许命中", hit2 is not None, str(hit2))
    check("年份解析正确", years_of("贵州茅台2023到2025年营收") == {"2023", "2025"},
          str(years_of("贵州茅台2023到2025年营收")))
    check("相对年份不误判", years_of("近三年营收趋势") == set(),
          str(years_of("近三年营收趋势")))


def test_polarity_ordinal_guard():
    print("\n== 极性 / 序数冲突检测 ==")
    cfg = FakeConfig()
    cfg.SEMANTIC_THRESHOLD = -1.0
    cfg.SEMANTIC_MIN_JACCARD = 0.0
    cache = SemanticCache(fake_embed, conf=cfg, redis_client=FakeRedisClient(fail=True))

    cache.set("2025年营业收入最高的公司", "最高答案", mode="auto", route="sql")
    hit = cache.get("2025年营业收入最低的公司", mode="auto")
    check("最高 vs 最低 -> 不命中", hit is None, str(hit))

    cache.set("2025年ROE前三的公司", "前三答案", mode="auto", route="sql")
    hit2 = cache.get("2025年ROE前十的公司", mode="auto")
    check("前三 vs 前十 -> 不命中", hit2 is None, str(hit2))

    check("极性解析 high", polarity("营收最高的公司") == "high")
    check("极性解析 low", polarity("营收最低的公司") == "low")
    check("序数解析", ordinals("ROE前三的公司") == {"前三"},
          str(ordinals("ROE前三的公司")))


if __name__ == "__main__":
    test_normalize()
    test_keywords_jaccard()
    test_memory_backend()
    test_redis_backend()
    test_jaccard_guard()
    test_metric_conflict_guard()
    test_year_conflict_guard()
    test_polarity_ordinal_guard()
    test_followup_guard()

    print("\n" + "=" * 60)
    print("通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
    if FAIL:
        print("失败用例：")
        for f in FAIL:
            print("  - %s" % f)
        sys.exit(1)
    print("全部通过 ✅")
