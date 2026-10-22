# -*- coding: utf-8 -*-
"""轻量意图路由：把问题分到 sql / graph / rag 三条链路。

P2 先用关键词规则（快、可解释、不用额外 LLM 调用）；
P3 可替换为 LLM 路由或小分类模型。
"""
import re

# 指标/数值类关键词 → SQL（MySQL 财务指标、股东表）
METRIC_KEYWORDS = [
    "营收", "营业收入", "收入", "净利", "净利润", "利润", "盈利", "毛利", "毛利率", "归母",
    "净利率", "roe", "roa",
    "回报率", "资产负债", "每股", "eps", "现金流", "同比", "增长", "下降", "下滑", "排名",
    "最高", "最低", "前三", "top", "对比", "比较", "cagr", "复合增长", "连续", "多少",
    "市值", "股东", "持股", "重仓", "基金", "机构",
]

# 文档/解释类关键词 → RAG（年报原文）
DOC_KEYWORDS = [
    "年报", "原文", "怎么解释", "为什么", "原因", "战略", "规划", "渠道", "产能", "经销商",
    "管理层", "讨论", "经营计划", "风险", "分红", "募集", "项目", "行业情况", "竞争格局",
    "怎么说", "提到", "披露", "说明",
]

# 关系类关键词（图谱链路：Neo4j 可用用 Cypher，否则用 MySQL holders 表兜底）
RELATION_KEYWORDS = [
    "什么关系", "共同股东", "共同持有", "同时持有", "都持有", "持股关系", "股权",
    "参股", "控股", "关联方", "实际控制人", "持有哪", "持有哪些", "重仓", "股东结构",
    "前三大股东", "十大股东",
]

# 知识库外的问题（行情/估值/预测/荐股）→ 明确拒答，不硬凑
# 行情/估值类：有实时数据源（services/market_service.py），走 market 分支
MARKET_KEYWORDS = [
    "股价", "股票", "行情", "涨跌", "涨幅", "跌幅", "市值", "市盈率", "pe", "pb",
    "估值", "走势", "成交量", "成交额", "换手率", "涨停", "跌停", "开盘", "收盘",
    "k线", "股价表现", "值得买", "能不能买", "能买吗", "北向", "资金流向", "股票怎么样",
]

UNSUPPORTED_KEYWORDS = [
    "股价", "行情", "涨跌", "涨幅", "跌幅", "价格", "市值", "市盈率", "pe", "pb",
    "估值", "预测", "会涨", "能买", "推荐", "目标价", "买入", "卖出", "抄底", "涨停",
]

_YEAR_RE = re.compile(r"(20(2[0-9]))\s*年?")

# 公司简称/别名 → 年报 metadata 里的 source（公司全称）
COMPANY_ALIASES = {
    "贵州茅台": ["贵州茅台", "茅台"],
    "五粮液": ["五粮液"],
    "泸州老窖": ["泸州老窖", "老窖", "国窖"],
    "洋河股份": ["洋河股份", "洋河", "梦之蓝", "海之蓝"],
    "山西汾酒": ["山西汾酒", "汾酒", "青花汾"],
    "古井贡酒": ["古井贡酒", "古井贡", "古井"],
    "今世缘": ["今世缘", "国缘"],
    "水井坊": ["水井坊"],
    "酒鬼酒": ["酒鬼酒"],
    "迎驾贡酒": ["迎驾贡酒", "迎驾"],
}

# 按别名长度倒序匹配，避免“古井”先命中“古井贡酒”之外的歧义
_ALIAS_INDEX = sorted(
    ((alias, full) for full, aliases in COMPANY_ALIASES.items() for alias in aliases),
    key=lambda x: len(x[0]),
    reverse=True,
)


def detect_companies(query):
    """从问题中识别提到的所有公司（按出现位置排序，去重）。"""
    text = query or ""
    hits = []
    for alias, full in _ALIAS_INDEX:
        pos = text.find(alias)
        if pos >= 0 and full not in [h[0] for h in hits]:
            hits.append((full, pos))
    # 按在句子里出现的先后排序，保证「A 和 B」的 A、B 顺序正确
    hits.sort(key=lambda x: x[1])
    return [full for full, _ in hits]


def detect_company(query):
    """从问题中识别唯一提到的公司，返回公司全称；没有或多于一家时返回 None。"""
    hits = detect_companies(query)
    return hits[0] if len(hits) == 1 else None


def has_domain_signal(query):
    """问题里是否含有金融领域信号（年份/指标/文档/关系类关键词）。

    有信号 → 按“专业咨询”处理（SQL / RAG）；没有信号且不是定义类问题时，
    才交给意图分类器判断是否“通用知识”。
    """
    text = (query or "").lower()
    if _YEAR_RE.search(text):
        return True
    keywords = METRIC_KEYWORDS + DOC_KEYWORDS + RELATION_KEYWORDS
    return any(k in text for k in keywords)


def route_question(query, graph_enabled=False):
    """返回 'market' / 'sql' / 'graph' / 'rag' / 'unsupported'。"""
    text = (query or "").strip().lower()
    if not text:
        return "rag"

    # 0) 行情/估值类：走行情数据服务（实时数据 + 年报基本面）
    # 单公司才走行情服务；多公司/对比类交给 SQL（避免“茅台五粮液估值对比”被行情接口截断）
    if any(k in text for k in MARKET_KEYWORDS) and len(detect_companies(text)) <= 1:
        return "market"

    # 0.1) 其他知识库外问题：交给 LLM 通用回答
    if any(k in text for k in UNSUPPORTED_KEYWORDS):
        return "unsupported"

    # 1) 关系类：走股东图谱（Neo4j 不可用时 GraphQA 内部用 MySQL 兜底）
    is_relation = any(k in text for k in RELATION_KEYWORDS)
    is_doc = any(k in text for k in DOC_KEYWORDS)
    if is_relation and not is_doc:
        return "graph"

    # 2) 明确的文档解释类优先（"为什么下滑/年报怎么解释" 即使带指标词也走原文）
    if is_doc:
        return "rag"

    # 3) 关系 + 文档词混在一起时，仍交给图谱（"股东结构"多半问的是名单）
    if is_relation:
        return "graph"

    # 3) 指标/数值类
    if any(k in text for k in METRIC_KEYWORDS):
        return "sql"

    # 4) 带年份 + 数字类动词
    if _YEAR_RE.search(text) and any(k in text for k in ["多少", "几", "变化", "相比", "比"]):
        return "sql"

    return "rag"


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    tests = [
        "2025年营业收入最高的3家公司",
        "贵州茅台2023到2025年的毛利率和ROE",
        "五粮液年报里怎么解释营收下滑",
        "招商中证白酒基金持有哪几家公司",
        "哪些公司ROE连续三年下降",
        "茅台和五粮液是什么关系",
        "茅台今天股价多少",
    ]
    for q in tests:
        print("%-30s -> %s" % (q, route_question(q)))
