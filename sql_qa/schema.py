# -*- coding: utf-8 -*-
"""财务数据表结构说明（用于 NL2SQL 提示词与 SQL 校验白名单）。"""

DB_NAME = "finance_qa"

ALLOWED_TABLES = {"companies", "financial_indicators", "holders", "documents"}

SCHEMA_TEXT = """
数据库：MySQL 8，库名 finance_qa，只允许查询以下 4 张表（只能 SELECT）：

1) companies 公司主数据
   code VARCHAR(10) 股票代码
   name VARCHAR(50) 公司简称（如 贵州茅台、五粮液）
   industry VARCHAR(50) 行业（均为 白酒）
   first_holder VARCHAR(200) 第一大股东（含持股比例）

2) financial_indicators 年度财务指标（每家公司每年一行，2023-2025 年）
   code VARCHAR(10) 股票代码
   report_year INT 报告年份（2023/2024/2025）
   report_date DATE 报告期
   report_type VARCHAR(20) 报告类型（年报）
   notice_date DATE 公告日期
   total_operate_income DOUBLE 营业总收入（单位：元）
   total_operate_income_yoy DOUBLE 营业总收入同比（单位：%）
   parent_netprofit DOUBLE 归母净利润（单位：元）
   parent_netprofit_yoy DOUBLE 归母净利润同比（单位：%）
   roe_weighted DOUBLE 加权平均ROE（单位：%）
   roe_deducted DOUBLE 扣非加权ROE（单位：%）
   gross_margin DOUBLE 销售毛利率（单位：%）
   net_margin DOUBLE 销售净利率（单位：%）
   debt_ratio DOUBLE 资产负债率（单位：%）
   eps_basic DOUBLE 基本每股收益（元）
   bps DOUBLE 每股净资产（元）
   op_cashflow_per_share DOUBLE 每股经营活动现金流（元）
   roic DOUBLE 投入资本回报率（单位：%）

3) holders 十大股东（每家公司最新一期，end_date=2026-06-30）
   code VARCHAR(10) 股票代码
   end_date DATE 报告期
   holder_rank INT 排名（1-10）
   holder_name VARCHAR(200) 股东名称
   hold_num BIGINT 持股数量（股）
   hold_ratio DOUBLE 持股比例（单位：%）
   hold_change VARCHAR(50) 较上期变化（如 不变/新进/-2937625）
   holder_code VARCHAR(50) 股东代码

4) documents 年报文档清单
   code VARCHAR(10) 股票代码
   report_year INT 报告年份
   title VARCHAR(300) 年报标题
   file_path VARCHAR(500) PDF 本地路径
   pages INT 页数
"""

COLUMN_NOTES = {
    "total_operate_income": "营业总收入，单位元；展示时建议 /1e8 转亿元",
    "parent_netprofit": "归母净利润，单位元；展示时建议 /1e8 转亿元",
    "total_operate_income_yoy": "营收同比，本身是百分数（7.52 表示 7.52%）",
    "parent_netprofit_yoy": "净利同比，本身是百分数",
    "roe_weighted": "加权ROE，百分数",
    "gross_margin": "毛利率，百分数",
    "net_margin": "净利率，百分数",
    "debt_ratio": "资产负债率，百分数",
    "hold_ratio": "持股比例，百分数",
}


def schema_text(table=None):
    """返回完整 schema 或指定表说明（预留）。"""
    return SCHEMA_TEXT
