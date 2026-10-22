# -*- coding: utf-8 -*-
"""金融意图识别 BERT 二分类：训练 + 模型评估。

数据格式（JSONL，每行一条）：
    {"query": "茅台2025年营收是多少", "label": "专业咨询", "category": "财务指标"}

标签：专业咨询 = 1，通用知识 = 0

划分：
    全量数据按 9:1 分层切分 → 训练集 / 测试集；
    训练集内部再按 9:1 切出验证集（只用于训练期早停/选模型，不参与最终评估）。

评估：
    - 测试集 Accuracy / Precision / Recall / F1
    - 每类 P/R/F1（classification_report）
    - 混淆矩阵
    - 误判样本清单（CSV）
    - 与规则路由基线对比（agent/router.py 的规则）
    - 推理耗时（单条平均毫秒）

用法：
    uv run python scripts/train_intent_bert.py \
        --data rag_qa/classify_data/training_data_finance_1200.jsonl \
        --epochs 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, r"C:\pro")

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from transformers import (BertForSequenceClassification, BertTokenizer,
                          Trainer, TrainingArguments)

LEGACY_MODEL = (r"C:\Users\黄瑞霖\PyCharmMiscProject\venv\project\Itcast_qa_system"
                r"\rag_qa\core\bert_query_classifier")
LABEL_MAP = {"通用知识": 0, "专业咨询": 1}
ID2LABEL = {0: "通用知识", 1: "专业咨询"}
OUT_DIR = r"C:\pro\rag_qa\core\bert_query_classifier"
REPORT_DIR = r"C:\pro\rag_qa\classify_data"


def load_data(path: str):
    texts, labels, cats = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = (d.get("query") or "").strip()
            lab = d.get("label")
            if not q or lab not in LABEL_MAP:
                continue
            texts.append(q)
            labels.append(LABEL_MAP[lab])
            cats.append(d.get("category", ""))
    return texts, labels, cats


# ---------------- 规则基线（与 agent/router.py 思路一致） ----------------
def rule_predict(query: str) -> int:
    """规则版意图：有公司/年份/指标/文档信号 → 专业咨询；定义类 → 通用。"""
    import re

    from agent.router import detect_company, has_domain_signal

    text = (query or "").strip()
    if re.search(r"(什么是|是什么意思|怎么理解|如何理解|有什么区别|解释一下)", text) and not detect_company(text):
        return 0
    if detect_company(text) or has_domain_signal(text):
        return 1
    return 0


def compute_metrics(eval_pred):
    logits, gold = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "precision": float(precision_score(gold, pred, zero_division=0)),
        "recall": float(recall_score(gold, pred, zero_division=0)),
        "f1": float(f1_score(gold, pred, zero_division=0)),
    }


def evaluate_model(model, tokenizer, test_x, test_y, device, max_len=64):
    """逐条推理，返回预测列表与单条平均耗时。"""
    import torch

    model.eval()
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(test_x), 32):
            batch = test_x[i:i + 32]
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    elapsed = time.time() - t0
    avg_ms = elapsed / max(1, len(test_x)) * 1000
    return preds, avg_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPORT_DIR, "training_data_finance_1200.jsonl"))
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--base-model", default=LEGACY_MODEL)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--test-size", type=float, default=0.1)
    args = ap.parse_args()

    texts, labels, cats = load_data(args.data)
    print(f"样本数: {len(texts)} | 专业咨询: {sum(labels)} | 通用知识: {len(labels) - sum(labels)}", flush=True)
    if len(texts) < 100:
        print("样本太少，退出", flush=True)
        return 1

    # 9:1 训练 / 测试
    train_x, test_x, train_y, test_y, train_c, test_c = train_test_split(
        texts, labels, cats, test_size=args.test_size, random_state=42, stratify=labels)
    # 训练集内部再留 10% 做验证（不碰测试集）
    tr_x, val_x, tr_y, val_y = train_test_split(
        train_x, train_y, test_size=0.1, random_state=42, stratify=train_y)
    print(f"训练集: {len(tr_x)} | 验证集: {len(val_x)} | 测试集: {len(test_x)}", flush=True)

    from sklearn.metrics import classification_report as _cr  # noqa: F401
    import torch

    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    model = BertForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL_MAP)
    device = torch.device("cpu")
    model.to(device)

    def encode(batch_texts, batch_labels):
        enc = tokenizer(batch_texts, truncation=True, padding=True,
                        max_length=args.max_len, return_tensors="pt")
        enc["labels"] = np.array(batch_labels)
        return enc

    train_enc = encode(tr_x, tr_y)
    val_enc = encode(val_x, val_y)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc

        def __len__(self):
            return len(self.enc["labels"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()}

    train_ds, val_ds = Dataset(train_enc), Dataset(val_enc)

    train_out = os.path.join(REPORT_DIR, "bert_train_out")
    os.makedirs(train_out, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=train_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        fp16=False,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    print("开始训练...", flush=True)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    print(f"训练完成，耗时 {train_seconds / 60:.1f} 分钟", flush=True)

    preds, avg_ms = evaluate_model(model, tokenizer, test_x, test_y, device, args.max_len)
    rule_preds = [rule_predict(q) for q in test_x]

    def metrics_of(gold, pred):
        return {
            "accuracy": float(accuracy_score(gold, pred)),
            "precision": float(precision_score(gold, pred, zero_division=0)),
            "recall": float(recall_score(gold, pred, zero_division=0)),
            "f1": float(f1_score(gold, pred, zero_division=0)),
        }

    bert_metrics = metrics_of(test_y, preds)
    rule_metrics = metrics_of(test_y, rule_preds)
    cm = confusion_matrix(test_y, preds, labels=[0, 1]).tolist()
    report = classification_report(
        test_y, preds, labels=[0, 1],
        target_names=[ID2LABEL[0], ID2LABEL[1]], digits=3, zero_division=0)

    print("\n== 测试集评估（BERT）==", flush=True)
    print(json.dumps(bert_metrics, ensure_ascii=False, indent=2), flush=True)
    print(report, flush=True)
    print("混淆矩阵 [[TN, FP], [FN, TP]]:", cm, flush=True)
    print(f"单条推理耗时: {avg_ms:.2f} ms | 规则基线:", json.dumps(rule_metrics, ensure_ascii=False), flush=True)

    # 误判样本
    mis = [{"query": q, "gold": ID2LABEL[g], "pred": ID2LABEL[p], "category": c}
           for q, g, p, c in zip(test_x, test_y, preds, test_c) if g != p]

    # 保存模型
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    # 保存评估报告
    eval_result = {
        "data_file": args.data,
        "train_size": len(tr_x),
        "val_size": len(val_x),
        "test_size": len(test_x),
        "epochs": args.epochs,
        "train_minutes": round(train_seconds / 60, 1),
        "bert": bert_metrics,
        "rule_baseline": rule_metrics,
        "confusion_matrix": {"labels": [ID2LABEL[0], ID2LABEL[1]], "matrix": cm},
        "per_class_report": report,
        "avg_inference_ms": round(avg_ms, 2),
        "misclassified_count": len(mis),
        "misclassified": mis[:100],
    }
    eval_path = os.path.join(args.out, "intent_eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(REPORT_DIR, "intent_eval_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 金融意图识别 BERT · 训练与评估报告\n\n")
        f.write(f"- 数据：`{args.data}`（训练 {len(tr_x)} / 验证 {len(val_x)} / 测试 {len(test_x)}）\n")
        f.write(f"- 训练：{args.epochs} epochs，batch {args.batch_size}，lr {args.lr}，CPU 耗时 {train_seconds / 60:.1f} 分钟\n")
        f.write(f"- 单条推理：{avg_ms:.2f} ms\n\n")
        f.write("## 测试集指标\n\n| 模型 | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|\n")
        f.write(f"| BERT（新训练） | {bert_metrics['accuracy']:.3f} | {bert_metrics['precision']:.3f} | "
                f"{bert_metrics['recall']:.3f} | {bert_metrics['f1']:.3f} |\n")
        f.write(f"| 规则基线 | {rule_metrics['accuracy']:.3f} | {rule_metrics['precision']:.3f} | "
                f"{rule_metrics['recall']:.3f} | {rule_metrics['f1']:.3f} |\n\n")
        f.write("## 分类报告\n\n```\n" + report + "\n```\n\n")
        f.write(f"## 混淆矩阵\n\n[[TN, FP], [FN, TP]] = `{cm}`\n\n")
        f.write(f"## 误判样本（{len(mis)} 条，展示前 20 条）\n\n")
        for m in mis[:20]:
            f.write(f"- `{m['query']}`：真实={m['gold']}，预测={m['pred']}（{m['category']}）\n")

    print(f"\n模型保存: {args.out}", flush=True)
    print(f"评估 JSON: {eval_path}", flush=True)
    print(f"评估报告: {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
