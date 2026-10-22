# 金融 RAG 项目数据集（白酒行业）

> 自动生成于 2026-10-19 17:42:01

## 目录结构
```
data/
├── raw/annual_reports/<代码>_<简称>/      # 巨潮年报 PDF（知识库原始文档）
├── structured/financial_indicators/      # 东财财务指标原始 JSON
├── structured/holders/                   # 东财十大股东原始 JSON
├── structured/financial_indicators.csv   # 汇总财务指标表（MySQL 导入用）
├── structured/holders.csv                # 汇总十大股东表（图谱/关系用）
├── index/annual_report_index.csv         # 年报清单（代码/年份/标题/URL/本地路径）
├── companies.csv                         # 公司清单
├── manifest.json                         # 下载清单与统计
└── README.md
```

## 数据来源
- 年报 PDF：巨潮资讯网 http://www.cninfo.com.cn （A 股法定信息披露平台）
- 财务指标/十大股东：东方财富数据中心 http://datacenter-web.eastmoney.com

## 规模统计
- 公司数：20
- 年报 PDF 数：60（总 166.8 MB）
- 财务指标年度记录：60
- 十大股东记录：190
- 年份范围：2023,2024,2025

## 使用说明
1. 年报 PDF 可直接作为 RAG 知识库原始文档（PDF 解析 → 父子块切分 → 向量化）；
2. financial_indicators.csv 导入 MySQL（如 finance_db.financial_indicators）做 NL2SQL；
3. holders.csv 用于构建 Neo4j 股权关系图：`(股东)-[:持有]->(上市公司)`；
4. annual_report_index.csv 存 PDF 与元数据映射，便于引用溯源到公司/年份/文件。

## 合规说明
- 年报为上市公司法定公开披露信息，仅用于学习/研究/演示，请注明来源，不再分发；
- 东财数据为公开行情/财务数据，仅用于学习研究；
- 抓取已限速（每次请求间隔 ~0.4-1s），请勿大规模爬取。

## 复现 / 扩展
```bash
python scripts/download_finance_data.py --out C:/pro/data
python scripts/download_finance_data.py --out C:/pro/data --years 2022,2023,2024,2025
python scripts/download_finance_data.py --out C:/pro/data --companies 600519,000858
```
