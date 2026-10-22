# -*- coding: utf-8 -*-
"""生成金融二分类意图识别数据集（v2：模板槽位 + LLM 扩样，稳定不重复）。

标签（与旧项目 query_classifier 的 label_map 一致）：
    专业咨询 = 1  —— 与项目知识库/服务有关的专业知识
                     （白酒公司、财务指标、年报原文、股东、行业对比、估值行情、公司研究）
    通用知识 = 0  —— 与项目无关的通用问题
                     （问候闲聊、身份能力、金融常识、其他领域常识、生活百科）

输出：
    rag_qa/classify_data/training_data_finance_<N>.jsonl
        每行：{"query": "...", "label": "专业咨询"|"通用知识", "category": "..."}
    rag_qa/classify_data/dataset_stats.json

用法：
    uv run python scripts/gen_intent_dataset.py --per-class 600
    uv run python scripts/gen_intent_dataset.py --per-class 600 --no-llm   # 纯模板
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, r"C:\pro")

OUT_DIR = os.path.join(r"C:\pro", "rag_qa", "classify_data")

COMPANIES = ["贵州茅台", "五粮液", "泸州老窖", "山西汾酒", "洋河股份",
             "古井贡酒", "今世缘", "迎驾贡酒", "酒鬼酒", "水井坊"]
YEARS = ["2023", "2024", "2025"]

# ---------- 专业咨询：模板 ----------
PRO_TEMPLATES = {
    "财务指标": [
        "{c}{y}年营业收入是多少？", "{c}{y}年归母净利润是多少？",
        "{c}{y}年的毛利率和净利率分别是多少？", "{c}{y}年ROE是多少？",
        "{c}{y}年资产负债率是多少？", "{c}近三年营收增速怎么样？",
        "{c}{y}年每股收益和每股净资产是多少？", "{c}{y}年经营现金流怎么样？",
        "{c}和{c2}{y}年营业收入对比", "{c}和{c2}的净利润谁更高？",
        "{y}年营收最高的三家公司是哪些？", "{y}年ROE最高的公司是哪家？",
        "{c}{y}年营收同比增速是多少？", "{c}连续三年营收下滑了吗？",
        "{c}{y}年的分红金额是多少？", "{c}的营收和净利润同比变化",
        "{c}和{c2}的毛利率差距有多大？",
    ],
    "年报原文": [
        "{c}{y}年年报里提到的经营风险有哪些？", "{c}{y}年报的管理层讨论与分析说了什么？",
        "{c}年报里怎么解释业绩下滑？", "{c}{y}年报提到的经营计划是什么？",
        "{c}的渠道库存和动销情况如何？", "{c}年报里关于产能和募集项目的描述",
        "{c}年报披露的分红方案是什么？", "{c}{y}年报的公司治理情况怎么样？",
        "{c}年报原文关于行业情况的描述", "{c}年报里提到了哪些竞争格局变化？",
        "{c}年报中的发展战略和规划", "{c}年报关于经销商和渠道的披露",
    ],
    "股东机构": [
        "{c}的十大股东有哪些？", "{c}的第一大股东是谁？",
        "香港中央结算持有{c}多少股份？", "招商中证白酒基金持有哪几家白酒公司？",
        "哪些基金重仓了{c}？", "{c}和{c2}的共同股东是谁？",
        "{c}的股东增减持情况怎么样？", "{c}的实际控制人是谁？",
        "{c}前十大股东持股比例", "证金公司持有{c}多少股份？",
    ],
    "行业对比": [
        "{y}年白酒行业整体情况如何？", "{c}和{c2}的毛利率对比",
        "{y}年白酒行业产量和集中度怎么样？", "{c}在白酒行业中的地位如何？",
        "高端白酒和次高端白酒有什么区别？", "{y}年白酒行业排名前十的公司",
        "{c}和{c2}的ROE对比", "{c}和{c2}的营收增速哪个更快？",
        "白酒行业深度调整期的情况", "{c}所在价格带的竞争格局",
    ],
    "估值行情": [
        "{c}今日股价是多少？", "{c}的市盈率和市净率是多少？",
        "{c}的总市值是多少？", "{c}最近走势怎么样？",
        "{c}股票值得买吗？", "{c}的估值水平贵不贵？",
        "{c}今天的涨跌幅和成交额", "{c}最近90天的股价表现",
        "{c}和{c2}的市值对比", "{c}的PE历史分位数大概多少？",
    ],
    "公司研究": [
        "{c}的主营业务是什么？", "{c}的核心竞争力有哪些？",
        "{c}的品牌和产品结构怎么样？", "{c}的渠道布局怎么样？",
        "{c}未来的增长点在哪里？", "{c}是国企吗？",
        "{c}和{c2}哪家更强？", "{c}的主力产品是什么？",
        "{c}的产能和扩产计划", "{c}的国际化进展如何？",
    ],
}

# ---------- 通用知识：模板 ----------
GEN_TEMPLATES = {
    "问候闲聊": [
        "你好", "在吗", "早上好", "晚上好", "最近怎么样", "谢谢你", "再见",
        "能陪我聊会天吗", "我今天心情不好", "给我讲个笑话", "今天有点无聊",
        "你吃饭了吗", "周末愉快", "你累不累", "讲个睡前故事吧",
    ],
    "身份能力": [
        "你是谁", "你叫什么名字", "你是什么模型", "你能做什么", "你会什么",
        "你的知识截止到什么时候", "你和ChatGPT有什么区别", "怎么使用你",
        "你是AI吗", "你能帮我写代码吗",
    ],
    "金融常识": [
        "股票是什么", "基金怎么买", "什么是市盈率", "A股开户流程是什么",
        "什么是牛市和熊市", "什么是价值投资", "ETF是什么", "打新股怎么操作",
        "可转债是什么", "分红除权是什么意思", "什么是熔断机制",
        "港股和A股有什么区别", "什么是量化投资", "国债怎么买", "什么是复利",
    ],
    "其他领域常识": [
        "帮我写一段Python代码", "英语翻译一下这句话", "北京有什么好玩的",
        "今天天气怎么样", "推荐一部电影", "红烧肉怎么做", "怎么减肥",
        "如何提高睡眠质量", "汽车保养要注意什么", "装修要注意什么",
        "怎么养猫", "历史上有哪些朝代", "地球到月亮有多远", "怎么学英语",
        "帮我写一份简历", "Excel怎么用VLOOKUP", "手机内存不够怎么办",
        "推荐几本书", "怎么提高工作效率", "什么是区块链", "怎么练腹肌",
        "杭州有哪些景点", "怎么煮咖啡", "什么是人工智能", "如何学好数学",
        "帮我总结一下这篇文章", "怎么写好一份PPT", "最近有什么好听的歌",
        "养狗需要注意什么", "感冒了怎么办",
    ],
    "生活百科": [
        "怎么快速入睡", "空调不制冷怎么办", "洗衣机怎么清洗",
        "怎么选净水器", "儿童发烧怎么处理", "怎么存钱理财", "如何整理房间",
        "怎样去除冰箱异味", "厨房油污怎么清理", "怎么挑选西瓜",
        "水费电费怎么交", "房子首付一般多少", "怎么办理护照",
        "社保和医保有什么区别", "怎么缓解颈椎痛",
    ],
}

LLM_TOPICS = {
    "问候闲聊": "日常问候、寒暄、闲聊、情绪表达、告别，口语化，像朋友聊天",
    "身份能力": "询问助手身份、能力边界、使用方法、模型信息",
    "金融常识": "与具体公司无关的金融/投资概念科普、交易规则、理财常识",
    "其他领域常识": "编程、英语、历史、地理、科技、教育、医疗、法律、写作等通用知识提问",
    "生活百科": "健康、饮食、运动、家居、宠物、出行、消费等生活问题",
}

PROMPT = """你是一个中文问题生成器。请围绕主题「{topic}」，生成 {n} 条**互不相同**、自然口语化的中文提问。

要求：
1. 每条 5~25 个汉字，像真实用户会问的话；不要编号、不要解释；
2. 不要出现以下已生成过的内容：{avoid}
3. 只输出一个 JSON 字符串数组，例如 ["问题1", "问题2"]。

主题说明：{desc}

输出："""


def clean_query(text: str) -> str:
    text = (text or "").strip().strip('"“”')
    text = re.sub(r"^\d+[.、]\s*", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def parse_queries(raw: str) -> list[str]:
    raw = (raw or "").strip()
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [clean_query(x) for x in arr if isinstance(x, str)]
        except json.JSONDecodeError:
            pass
    lines = [clean_query(x) for x in raw.splitlines() if x.strip()]
    return [x for x in lines if x and not x.startswith("[")]


def build_pro(per_class: int) -> list[dict]:
    """模板 + 槽位生成专业咨询问题。"""
    per_cat = per_class // len(PRO_TEMPLATES)
    rows: list[dict] = []
    seen: set[str] = set()
    for cat, templates in PRO_TEMPLATES.items():
        pool: list[str] = []
        for t in templates:
            for c in COMPANIES:
                for c2 in COMPANIES:
                    if c2 == c:
                        continue
                    for y in YEARS:
                        pool.append(t.format(c=c, c2=c2, y=y))
        random.shuffle(pool)
        for q in pool:
            if q in seen:
                continue
            seen.add(q)
            rows.append({"query": q, "label": "专业咨询", "category": cat})
            if sum(1 for r in rows if r["category"] == cat) >= per_cat:
                break
    return rows


def build_general(per_class: int, use_llm: bool) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    per_cat = per_class // len(GEN_TEMPLATES)
    for cat, templates in GEN_TEMPLATES.items():
        for q in templates:
            if q not in seen:
                seen.add(q)
                rows.append({"query": q, "label": "通用知识", "category": cat})
    # LLM 扩样，补到 per_class
    if use_llm:
        from sql_qa.llm_client import LLMClient

        llm = LLMClient()
        cats = list(GEN_TEMPLATES.keys())
        idx = 0
        failures = 0
        while len(rows) < per_class and failures < 12:
            cat = cats[idx % len(cats)]
            idx += 1
            need = min(30, per_class - len(rows))
            recent = [r["query"] for r in rows[-40:]]
            prompt = PROMPT.format(
                topic=cat, n=need, desc=LLM_TOPICS[cat],
                avoid="、".join(recent[-20:]) or "无")
            try:
                raw = llm.chat(prompt, temperature=0.9, max_tokens=1400,
                               system="你是严谨的中文数据生成器，只输出 JSON 数组。")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {cat}: {e}", flush=True)
                failures += 1
                time.sleep(2)
                continue
            added = 0
            for q in parse_queries(raw):
                if not (4 <= len(q) <= 40) or q in seen:
                    continue
                seen.add(q)
                rows.append({"query": q, "label": "通用知识", "category": cat})
                added += 1
                if len(rows) >= per_class:
                    break
            failures = 0 if added else failures + 1
            print(f"  通用知识: {len(rows)}/{per_class} (+{added})", flush=True)
            time.sleep(0.2)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("== 生成 专业咨询（模板槽位）==", flush=True)
    pro = build_pro(args.per_class)
    print(f"  专业咨询: {len(pro)}", flush=True)

    print("== 生成 通用知识（模板 + LLM）==", flush=True)
    gen = build_general(args.per_class, use_llm=not args.no_llm)
    print(f"  通用知识: {len(gen)}", flush=True)

    # 打散 + 跨类去重
    rows = pro + gen
    seen, unique = set(), []
    for r in rows:
        if r["query"] in seen:
            continue
        seen.add(r["query"])
        unique.append(r)
    random.shuffle(unique)

    out_path = os.path.join(OUT_DIR, f"training_data_finance_{len(unique)}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total": len(unique),
        "label_counts": {
            "专业咨询": sum(1 for r in unique if r["label"] == "专业咨询"),
            "通用知识": sum(1 for r in unique if r["label"] == "通用知识"),
        },
        "category_counts": {},
        "file": out_path,
    }
    for r in unique:
        stats["category_counts"][r["category"]] = stats["category_counts"].get(r["category"], 0) + 1
    with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n== 完成 ==", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
