# 金融意图识别 BERT · 训练与评估报告

- 数据：`C:\pro\rag_qa\classify_data\training_data_finance_1200.jsonl`（训练 972 / 验证 108 / 测试 120）
- 训练：3.0 epochs，batch 16，lr 2e-05，CPU 耗时 8.3 分钟
- 单条推理：54.60 ms

## 测试集指标

| 模型 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| BERT（新训练） | 1.000 | 1.000 | 1.000 | 1.000 |
| 规则基线 | 0.958 | 0.951 | 0.967 | 0.959 |

## 分类报告

```
              precision    recall  f1-score   support

        通用知识      1.000     1.000     1.000        60
        专业咨询      1.000     1.000     1.000        60

    accuracy                          1.000       120
   macro avg      1.000     1.000     1.000       120
weighted avg      1.000     1.000     1.000       120

```

## 混淆矩阵

[[TN, FP], [FN, TP]] = `[[60, 0], [0, 60]]`

## 误判样本（0 条，展示前 20 条）

