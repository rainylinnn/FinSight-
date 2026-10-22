# -*- coding: utf-8 -*-
"""意图模型分布外（OOD）小样本评估：用真实风格、人工标注的问题检验泛化能力。

用法：
    uv run python scripts/eval_intent_ood.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, r"C:\pro")

import numpy as np

from rag_qa.core.query_classifier import QueryClassifier

# 人工构造的真实风格问题（与训练集模板不同），label: 1=专业咨询, 0=通用知识
CASES = [
    ("茅台今天股价多少", 1),
    ("帮我看看五粮液近几年的营收", 1),
    ("汾酒2025年净利润是多少", 1),
    ("哪些基金持有泸州老窖", 1),
    ("茅台年报里怎么解释业绩下滑", 1),
    ("今世缘和洋河哪个毛利率高", 1),
    ("白酒行业现在处于什么周期", 1),
    ("古井贡酒的渠道库存怎么样", 1),
    ("迎驾贡酒的大股东是谁", 1),
    ("水井坊的ROE大概多少", 1),
    ("酒鬼酒今年分红多少", 1),
    ("茅台和五粮液的市值对比一下", 1),
    ("老窖近三年现金流怎么样", 1),
    ("汾酒的收入增速如何", 1),
    ("帮我对比一下这几家白酒公司的估值", 1),
    ("茅台值得长期持有吗", 1),
    ("五粮液最近走势如何", 1),
    ("有哪些机构重仓白酒", 1),
    ("你好啊", 0),
    ("你能做什么", 0),
    ("什么是市盈率", 0),
    ("基金怎么买", 0),
    ("今天天气怎么样", 0),
    ("帮我写个Python脚本", 0),
    ("推荐几本投资类的书", 0),
    ("怎么减肥", 0),
    ("股票和债券有什么区别", 0),
    ("什么是ROE", 0),
    ("谢谢你", 0),
    ("北京有什么好玩的", 0),
    ("美股怎么开户", 0),
    ("什么是牛市", 0),
    ("红烧肉怎么做", 0),
    ("最近有什么好看的电影", 0),
]


def rule_predict(q: str) -> int:
    import re

    from agent.router import detect_company, has_domain_signal

    text = (q or "").strip()
    if re.search(r"(什么是|是什么意思|怎么理解|如何理解|有什么区别|解释一下)", text) and not detect_company(text):
        return 0
    if detect_company(text) or has_domain_signal(text):
        return 1
    return 0


def main() -> int:
    model_path = os.path.join(r"C:\pro", "rag_qa", "core", "bert_query_classifier")
    clf = QueryClassifier(model_path)
    y_true, y_pred = [], []
    print(f"{'真实':<6}{'BERT':<6}{'规则':<6}问题")
    for q, label in CASES:
        pred = clf.predict_category(q)
        bert = 1 if pred == "专业咨询" else 0
        rule = rule_predict(q)
        y_true.append(label)
        y_pred.append(bert)
        flag = "" if bert == label else "  <-- 误判"
        print(f"{label:<6}{bert:<6}{rule:<6}{q}{flag}")

    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score)

    rule_preds = [rule_predict(q) for q, _ in CASES]
    for name, pred in (("BERT（新训练）", y_pred), ("规则基线", rule_preds)):
        print(f"\n{name}: acc={accuracy_score(y_true, pred):.3f} "
              f"precision={precision_score(y_true, pred, zero_division=0):.3f} "
              f"recall={recall_score(y_true, pred, zero_division=0):.3f} "
              f"f1={f1_score(y_true, pred, zero_division=0):.3f}")
        print("confusion [[TN,FP],[FN,TP]]:", confusion_matrix(y_true, pred, labels=[0, 1]).tolist())

    result = {"n": len(CASES), "bert": {}, "rule": {}, "misclassified": []}
    for name, pred in (("bert", y_pred), ("rule", rule_preds)):
        result[name] = {
            "accuracy": round(accuracy_score(y_true, pred), 3),
            "precision": round(precision_score(y_true, pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, pred, zero_division=0), 3),
        }
    for (q, label), p in zip(CASES, y_pred):
        if p != label:
            result["misclassified"].append({"query": q, "gold": label, "pred": p})
    with open(os.path.join(r"C:\pro", "rag_qa", "classify_data", "ood_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nOOD 评估结果已保存: rag_qa/classify_data/ood_eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
