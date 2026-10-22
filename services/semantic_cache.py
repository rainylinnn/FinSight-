# -*- coding: utf-8 -*-
"""语义缓存：用 BGE-M3 向量把「问过的问题 → 答案」缓存起来。

背景：RAG 类问题一次要 10-15 秒（检索 + LLM 生成），答辩/演示时同一个问题
反复问很浪费时间。语义缓存把问题编码成向量，新问题与历史问题相似度超过阈值
就直接复用答案，整条检索 + 生成链路都跳过。

两种后端（自动切换，不用手工配）：
  - redis：跨进程共享、可持久化，优先使用；
  - memory：Redis 连不上时的降级方案，进程内 LRU + numpy 暴力检索，
            重启失效，但演示效果一致（问题来了照样秒回）。

相似度判定除了向量余弦，还加一道「实词 Jaccard」门槛：
「2025年营收最高的公司」和「2025年净利最高的公司」向量很接近，但答案完全不同，
Jaccard 门槛能把这类假命中挡掉。
"""
import hashlib
import json
import re
import time
import unicodedata

import numpy as np

from base import logger

# 停用词（分词后过滤，只保留实词做 Jaccard 判定）
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "对", "把", "被", "着", "过", "吗", "呢",
    "什么", "怎么", "如何", "多少", "哪些", "哪个", "公司", "请问", "请", "一下",
    "这个", "那个", "我们", "你们", "他们", "可以", "需要", "想要", "帮我", "查一下",
    "年", "份", "情况", "相关", "信息", "数据", "报告", "年报",
}

_PUNCT_RE = re.compile(r"[\s，。、；：？！“”‘’（）《》【】\[\]{}()\"'`,.;:?!~`@#$%^&*_+=|\\/<>-]+")

# 指标词表：用于「指标冲突检测」。
# 向量上「2025年营收最高的公司」和「2025年归母净利润最高的公司」非常接近（只差两个字），
# 只靠余弦 + Jaccard 分不开，必须显式比对两边关心的指标是否一致。
METRIC_TERMS = {
    "营收": ["营收", "营业收入", "收入", "revenue"],
    "净利": ["净利", "净利润", "归母", "利润", "profit"],
    "毛利": ["毛利"],
    "净利率": ["净利率"],
    "roe": ["roe", "净资产收益率", "回报率"],
    "roa": ["roa", "总资产报酬"],
    "资产负债": ["资产负债", "负债率"],
    "每股": ["每股", "eps", "净资产"],
    "现金流": ["现金流"],
    "同比": ["同比", "增长", "下滑", "降幅", "增速"],
    "股东": ["股东", "持股", "持仓", "基金", "股权", "重仓"],
    "风险": ["风险"],
    "战略": ["战略", "规划", "经营计划", "目标"],
    "渠道": ["渠道", "经销商"],
    "分红": ["分红", "派息", "利润分配"],
    "产能": ["产能", "产量", "基酒", "酒库"],
    "价格": ["价格", "出厂价", "提价", "涨价"],
    "行业": ["行业", "板块", "市场", "竞争"],
}


def metric_keys(text):
    """取出问题关心的「指标类别」集合（用于判断两个问题是不是在问同一件事）。"""
    low = (text or "").lower()
    return {key for key, words in METRIC_TERMS.items() if any(w in low for w in words)}


# 年份：跨年问题必须分开缓存（“茅台2025年营收”与“茅台2023年营收”答案完全不同）
_YEAR_RE = re.compile(r"(20\d{2})\s*年?")


def years_of(text):
    """取出问题里出现的年份集合（相对表述如“近三年”不含数字，这里取不到，属正常）。"""
    return set(_YEAR_RE.findall(text or ""))


# 极性（最高/最低）：方向相反的问题答案完全相反，必须挡住
POLARITY_HIGH = ["最高", "最大", "最多", "第一", "榜首", "领先", "居首", "领跑", "最好"]
POLARITY_LOW = ["最低", "最小", "最少", "最后", "垫底", "最差", "最弱"]

# 序数（前三/前十）：口径不同答案不同
ORDINAL_TERMS = ["前三", "前3", "top3", "前五", "前5", "top5", "前十", "前10", "top10",
                 "前二", "前两", "第二", "第三", "第五", "第十", "top20", "前二十"]


def polarity(text):
    """返回 'high' / 'low' / ''（方向词）。"""
    low = (text or "").lower()
    hi = any(w in low for w in POLARITY_HIGH)
    lo = any(w in low for w in POLARITY_LOW)
    if hi and not lo:
        return "high"
    if lo and not hi:
        return "low"
    return ""


def ordinals(text):
    """返回问题里的序数标记集合（前三 / 前十 …）。"""
    low = (text or "").lower()
    return {o for o in ORDINAL_TERMS if o in low}



def normalize_query(text):
    """归一化问题：全角转半角、去标点、压空白、小写。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = _PUNCT_RE.sub("", text)
    return text


def _keywords(text):
    """用 jieba 搜索引擎模式切词，过滤停用词和单字，得到实词集合。"""
    try:
        import jieba
        words = jieba.cut_for_search(text or "")
    except Exception:  # noqa: BLE001
        words = re.split(r"\s+", text or "")
    return {w.strip() for w in words if len(w.strip()) >= 2 and w.strip() not in STOPWORDS}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _pack_vec(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()


def _unpack_vec(raw):
    return np.frombuffer(raw, dtype=np.float32)


class SemanticCache:
    """向量语义缓存。embed_fn(texts: list[str]) -> np.ndarray [n, dim]，已归一化。"""

    NAMESPACE = "finsight:sem"
    VEC_PREFIX = "finsight:sem:v:"
    ITEM_PREFIX = "finsight:sem:e:"
    INDEX_KEY = "finsight:sem:index"
    STATS_KEY = "finsight:sem:stats"
    HITS_KEY = "finsight:sem:hits"

    def __init__(self, embed_fn, conf=None, redis_client=None):
        self.logger = logger
        self.embed_fn = embed_fn
        self.conf = conf
        self.redis = redis_client
        self.backend = "memory"
        self._memory = {}          # key -> {"vec": np.float32, "item": dict}
        self._memory_order = []    # LRU 顺序
        self._disabled = False

        # 阈值允许取 0（= 全放开），不能用 `or` 兜底，否则 0 会被当成缺省值
        # 0.85 这个值是实测出来的：同义改写最低 0.9068，异指标最高 0.9302（两者交叉，
        # 纯向量阈值不可靠），所以向量只做粗筛，真正的判定交给下面三道精确门槛。
        threshold = getattr(conf, "SEMANTIC_THRESHOLD", 0.85)
        self.threshold = 0.85 if threshold is None else float(threshold)
        jaccard = getattr(conf, "SEMANTIC_MIN_JACCARD", 0.2)
        self.min_jaccard = 0.2 if jaccard is None else float(jaccard)
        self.ttl = int(getattr(conf, "SEMANTIC_TTL", 7200) or 7200)
        self.max_entries = int(getattr(conf, "SEMANTIC_MAX_ENTRIES", 500) or 500)
        self.candidate_limit = int(getattr(conf, "SEMANTIC_CANDIDATE_LIMIT", 200) or 200)
        self.enabled = bool(getattr(conf, "SEMANTIC_ENABLED", True))

        if not self.enabled:
            self._disabled = True
            self.logger.info("语义缓存已关闭（config.ini [cache] semantic_enabled=false）")
            return

        # Redis 优先，连不上就退到内存后端（不能让缓存拖垮整个服务）
        if self.redis is not None:
            if self._probe_redis():
                self.backend = "redis"
                self.logger.info(f"语义缓存启用，后端=Redis（阈值 {self.threshold}）")
                return
            self.logger.warning("Redis 不可用，语义缓存降级为进程内内存模式（重启失效）")
        else:
            self.logger.warning("未提供 Redis 客户端，语义缓存降级为进程内内存模式")
        self.backend = "memory"

    # ------------------------------------------------------------------ 基础设施
    def _probe_redis(self):
        try:
            self.redis.client.ping()
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Redis 连接失败（{type(e).__name__}），语义缓存走内存后端")
            return False

    def _embed(self, text):
        """编码单条问题，返回归一化后的 float32 向量。"""
        try:
            dense = self.embed_fn([text])
            arr = np.asarray(dense, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[0]
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            return arr
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"语义缓存编码失败（忽略缓存）: {e}")
            return None

    @staticmethod
    def _key(text):
        return hashlib.sha1(normalize_query(text).encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ 统计
    def _bump(self, field, delta=1):
        if self.backend != "redis":
            return
        try:
            self.redis.client.hincrby(self.STATS_KEY, field, delta)
        except Exception:  # noqa: BLE001
            pass

    def stats(self):
        """命中率统计（答辩可展示）。"""
        hits = misses = writes = 0
        if self.backend == "redis":
            try:
                raw = self.redis.client.hgetall(self.STATS_KEY) or {}
                hits = int(raw.get("hits", 0) or 0)
                misses = int(raw.get("misses", 0) or 0)
                writes = int(raw.get("writes", 0) or 0)
                entries = int(self.redis.client.zcard(self.INDEX_KEY))
            except Exception:  # noqa: BLE001
                entries = 0
        else:
            entries = len(self._memory)
            hits = int(self._memory_stats.get("hits", 0))
            misses = int(self._memory_stats.get("misses", 0))
            writes = int(self._memory_stats.get("writes", 0))
        total = hits + misses
        return {
            "enabled": self.enabled and not self._disabled,
            "backend": self.backend,
            "threshold": self.threshold,
            "min_jaccard": self.min_jaccard,
            "entries": entries,
            "hits": hits,
            "misses": misses,
            "writes": writes,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }

    _memory_stats = {"hits": 0, "misses": 0, "writes": 0}

    # ------------------------------------------------------------------ 读
    def _candidates(self, limit):
        """取候选条目：[(key, vec, item), ...]"""
        if self.backend == "redis":
            try:
                keys = self.redis.client.zrevrange(self.INDEX_KEY, 0, max(0, limit - 1))
                if not keys:
                    return []
                pipe = self.redis.client.pipeline()
                for k in keys:
                    pipe.get(self.VEC_PREFIX + k)
                    pipe.get(self.ITEM_PREFIX + k)
                raw = pipe.execute()
                out = []
                for i, k in enumerate(keys):
                    vec_raw = raw[2 * i]
                    item_raw = raw[2 * i + 1]
                    if not vec_raw or not item_raw:
                        continue
                    try:
                        item = json.loads(item_raw)
                    except Exception:  # noqa: BLE001
                        continue
                    out.append((k, _unpack_vec(vec_raw), item))
                return out
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"语义缓存读取失败（忽略）: {e}")
                return []
        # 内存后端
        out = []
        for k in list(self._memory_order)[-limit:]:
            entry = self._memory.get(k)
            if entry:
                out.append((k, entry["vec"], entry["item"]))
        return out

    def get(self, query, mode="auto"):
        """查缓存。命中返回 dict，未命中/出错返回 None。"""
        if self._disabled or not self.enabled:
            return None
        if not query or not str(query).strip():
            return None

        vec = self._embed(query)
        if vec is None:
            return None

        candidates = self._candidates(self.candidate_limit)
        if not candidates:
            self._count_miss()
            return None

        best = None
        best_sim = -1.0
        for key, cvec, item in candidates:
            if cvec.shape[0] != vec.shape[0]:
                continue
            sim = float(np.dot(vec, cvec))
            if sim > best_sim:
                best_sim = sim
                best = (key, item)

        if best is None or best_sim < self.threshold:
            self._count_miss()
            return None

        key, item = best
        # 词面门槛（弱）：只挡极端词面无关的情况。
        # 实测发现它不能当硬门槛——「营收」和「营业收入」分词后不重合，
        # Jaccard 只有 0.33，会把最该命中的同义改写挡掉。
        if self.min_jaccard > 0:
            sim_words = _jaccard(_keywords(query), _keywords(item.get("query", "")))
            if sim_words < self.min_jaccard:
                self.logger.info(
                    f"语义缓存相似度 {best_sim:.3f} 达标但词面重合 {sim_words:.2f} 不足，放弃命中")
                self._count_miss()
                return None
        else:
            sim_words = 1.0

        # 指标冲突检测：问的是不同指标（营收 vs 净利）时，哪怕向量很像也不能复用答案
        q_metrics = metric_keys(query)
        c_metrics = metric_keys(item.get("query", ""))
        if q_metrics and c_metrics and not (q_metrics & c_metrics):
            self.logger.info(
                f"语义缓存指标冲突（{sorted(q_metrics)} vs {sorted(c_metrics)}），放弃命中")
            self._count_miss()
            return None

        # 年份冲突检测：跨年问题必须分开（2025 年营收 ≠ 2023 年营收）
        q_years = years_of(query)
        c_years = years_of(item.get("query", ""))
        if q_years and c_years and not (q_years & c_years):
            self.logger.info(
                f"语义缓存年份冲突（{sorted(q_years)} vs {sorted(c_years)}），放弃命中")
            self._count_miss()
            return None

        # 方向冲突：最高 vs 最低，答案完全相反
        q_pol = polarity(query)
        c_pol = polarity(item.get("query", ""))
        if q_pol and c_pol and q_pol != c_pol:
            self.logger.info(f"语义缓存方向冲突（{q_pol} vs {c_pol}），放弃命中")
            self._count_miss()
            return None

        # 序数口径冲突：前三 vs 前十，答案不同
        q_ord = ordinals(query)
        c_ord = ordinals(item.get("query", ""))
        if q_ord and c_ord and not (q_ord & c_ord):
            self.logger.info(
                f"语义缓存序数冲突（{sorted(q_ord)} vs {sorted(c_ord)}），放弃命中")
            self._count_miss()
            return None

        if self.backend == "redis":
            try:
                # 单条命中次数放在独立 hash 里（条目本身是 JSON 字符串，不能 hincrby）
                self.redis.client.hincrby(self.HITS_KEY, key, 1)
            except Exception:  # noqa: BLE001
                pass
        else:
            self._memory[key]["item"]["hits"] = self._memory[key]["item"].get("hits", 0) + 1
        self._count_hit()
        self.logger.info(
            f"语义缓存命中：'{query[:30]}' ~ '{item.get('query', '')[:30]}' "
            f"cos={best_sim:.3f} jaccard={sim_words:.2f}")
        return {
            "answer": item.get("answer", ""),
            "query": item.get("query", ""),
            "mode": item.get("mode", "auto"),
            "route": item.get("route", ""),
            "similarity": round(best_sim, 4),
            "jaccard": round(sim_words, 4),
            "key": key,
        }

    def _count_hit(self):
        if self.backend == "redis":
            self._bump("hits")
        else:
            self._memory_stats["hits"] += 1

    def _count_miss(self):
        if self.backend == "redis":
            self._bump("misses")
        else:
            self._memory_stats["misses"] += 1

    # ------------------------------------------------------------------ 写
    def set(self, query, answer, mode="auto", route="", meta=None):
        """写入缓存。answer 为空或查询太短时直接忽略。"""
        if self._disabled or not self.enabled:
            return False
        text = (query or "").strip()
        if not text or not answer or len(text) < 4:
            return False

        vec = self._embed(text)
        if vec is None:
            return False

        key = self._key(text)
        item = {
            "key": key,
            "query": text,
            "answer": answer,
            "mode": mode,
            "route": route,
            "created_at": int(time.time()),
            "hits": 0,
            "meta": meta or {},
        }

        if self.backend == "redis":
            try:
                pipe = self.redis.client.pipeline()
                pipe.set(self.VEC_PREFIX + key, _pack_vec(vec), ex=self.ttl)
                pipe.set(self.ITEM_PREFIX + key, json.dumps(item, ensure_ascii=False), ex=self.ttl)
                pipe.zadd(self.INDEX_KEY, {key: item["created_at"]})
                pipe.execute()
                # 索引里可能有已过期的 key，顺手清一下并限制总量
                if self.redis.client.zcard(self.INDEX_KEY) > self.max_entries:
                    self.redis.client.zremrangebyrank(self.INDEX_KEY, 0, -self.max_entries - 1)
                self._bump("writes")
                self.logger.info(f"语义缓存写入：'{text[:30]}' (route={route})")
                return True
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"语义缓存写入失败（忽略）: {e}")
                return False

        # 内存后端
        self._memory[key] = {"vec": vec, "item": item}
        if key in self._memory_order:
            self._memory_order.remove(key)
        self._memory_order.append(key)
        while len(self._memory_order) > self.max_entries:
            old = self._memory_order.pop(0)
            self._memory.pop(old, None)
        self._memory_stats["writes"] += 1
        self.logger.info(f"语义缓存写入（内存）：'{text[:30]}' (route={route})")
        return True

    def clear(self):
        """清空缓存（Redis 与内存都清）。"""
        if self.backend == "redis":
            try:
                keys = self.redis.client.zrange(self.INDEX_KEY, 0, -1)
                pipe = self.redis.client.pipeline()
                for k in keys:
                    pipe.delete(self.VEC_PREFIX + k)
                    pipe.delete(self.ITEM_PREFIX + k)
                pipe.delete(self.INDEX_KEY)
                pipe.delete(self.STATS_KEY)
                pipe.delete(self.HITS_KEY)
                pipe.execute()
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"清空 Redis 缓存失败: {e}")
                return False
        self._memory.clear()
        self._memory_order.clear()
        self._memory_stats = {"hits": 0, "misses": 0, "writes": 0}
        return True
