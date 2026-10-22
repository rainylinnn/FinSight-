# -*- coding: utf-8 -*-
"""RAGAS 评测（金融投研助手）：Faithfulness / Answer Relevancy / Context Precision / Context Recall。

流程：
    1) 对每道题用项目检索链路取上下文（Milvus 混合检索 + BM25 RRF）
    2) 用项目 Prompt + DeepSeek 生成答案
    3) 用 RAGAS 四个指标打分（LLM 裁判 = DeepSeek，embedding = 本地 BGE-M3）

产出：
    eval/ragas_results.json

运行：
    uv run python scripts/run_ragas_eval.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\pro")

# 本机系统代理会让 api.deepseek.com TLS 握手失败，RAGAS 走 httpx 时需要绕过
os.environ["NO_PROXY"] = "api.deepseek.com,localhost,127.0.0.1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]

from base import Config  # noqa: E402

conf = Config()
os.environ.setdefault("OPENAI_API_KEY", conf.LLM_API_KEY or "")

from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import BaseRagasEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from rag_qa.core.prompts import RAGPrompts  # noqa: E402
from rag_qa.core.vector_store import VectorStore  # noqa: E402

OUT = Path(r"C:\pro\eval")
OUT.mkdir(exist_ok=True)

# (问题, 公司过滤, 年份过滤, 参考答案)
CASES = [
    ("茅台2025年报提到的可能面对的风险有哪些？", "贵州茅台", 2025,
     "可能面对的风险包括宏观经济风险、安全风险、舆情风险、环境保护风险。"),
    ("茅台的核心竞争力有哪些？", "贵州茅台", 2025,
     "公司核心竞争力包括品质、品牌、工艺、环境、文化五个方面。"),
    ("茅台2024年度分红方案是什么？", "贵州茅台", 2024,
     "2024年度利润分配预案为每10股派发现金红利276.24元（含税）。"),
    ("五粮液2025年营业收入下降的原因？", "五粮液", 2025,
     "受白酒行业深度调整、消费需求变化影响，公司2025年营业收入同比下降54.55%。"),
    ("洋河股份的渠道库存和动销策略？", "洋河股份", 2025,
     "公司以促动销、去库存、控发货为核心，严格调控发货节奏，推动市场库存去化。"),
    ("山西汾酒2025年增长的主要原因？", "山西汾酒", 2025,
     "公司青花系列产品增长、全国化市场拓展带动营业收入同比增长7.52%。"),
    ("泸州老窖国窖1573的品牌定位？", "泸州老窖", 2025,
     "国窖1573是公司高端核心品牌，定位中国白酒鉴赏标准级酒品。"),
    ("水井坊如何判断2025年白酒行业形势？", "水井坊", 2025,
     "行业处于三期叠加的深度调整期，存量竞争、头部集中与结构分化并存。"),
    ("古井贡酒的渠道建设思路？", "古井贡酒", 2025,
     "公司推进全国化、次高端战略，织密渠道网络、控库存稳价格，并推进数字化。"),
    ("舍得酒业的老酒战略是什么？", "舍得酒业", 2025,
     "公司坚持老酒战略，以老酒品质和年份为差异化竞争点。"),
    ("今世缘的主要产品结构？", "今世缘", 2025,
     "公司以国缘、今世缘系列产品为主，聚焦次高端价格带。"),
]


class LocalEmbeddings(BaseRagasEmbeddings):
    """把本地 BGE-M3 包装成 RAGAS embeddings。"""

    def __init__(self, model_path: str):
        super().__init__(cache=None)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path)

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str):
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts):
        return await asyncio.to_thread(self.embed_documents, texts)


def main() -> int:
    vs = VectorStore()
    client = OpenAI(api_key=conf.LLM_API_KEY, base_url=conf.LLM_BASE_URL,
                    http_client=httpx.Client(trust_env=False))
    prompt_tpl = RAGPrompts.rag_prompt()

    samples, rows = [], []
    for q, company, year, ref in CASES:
        t0 = time.time()
        docs = vs.hybrid_search_with_rerank(
            q, k=10, source_filter=company, year_filter=year)
        contexts = [d.page_content for d in docs]
        context_text = "\n\n".join(
            f"【{d.metadata.get('company_name')} {d.metadata.get('year')} 第{d.metadata.get('page')}页】\n{d.page_content}"
            for d in docs)
        prompt = prompt_tpl.format(context=context_text, history="", question=q, phone="")
        try:
            resp = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=2000, timeout=90,
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            answer = f"回答生成失败：{e}"
        dt = time.time() - t0
        rows.append({"question": q, "answer": answer, "contexts": contexts,
                     "reference": ref, "latency": round(dt, 2)})
        samples.append(SingleTurnSample(
            user_input=q, response=answer,
            retrieved_contexts=contexts, reference=ref))
        print(f"[{len(rows)}/{len(CASES)}] {q} | {dt:.1f}s | ans={answer[:60]}", flush=True)

    print("\n开始 RAGAS 打分（DeepSeek 裁判 + 本地 BGE-M3）...", flush=True)
    llm = llm_factory(conf.LLM_MODEL, base_url=conf.LLM_BASE_URL)
    emb = LocalEmbeddings(os.path.join(conf.MODEL_ROOT, "bge-m3"))
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset, metrics=metrics, raise_exceptions=False, show_progress=True)

    df = result.to_pandas()
    summary = {}
    for col in ["faithfulness", "answer_relevancy",
                "llm_context_precision_with_reference", "context_recall"]:
        if col in df.columns:
            vals = [float(v) for v in df[col].tolist() if v == v]  # 过滤 NaN
            summary[col] = round(sum(vals) / len(vals), 4) if vals else None

    out = {"n": len(CASES), "summary": summary, "details": rows,
           "per_sample": df.to_dict(orient="records")}
    (OUT / "ragas_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nRAGAS 汇总:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("结果已保存:", OUT / "ragas_results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
