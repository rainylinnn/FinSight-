# -*- coding: utf-8 -*-
"""FinSight SQL QA：自然语言查询财务指标 / 股东数据（NL2SQL）。

用法：
    python -m sql_qa "2025年营业收入最高的3家公司"
    python -m sql_qa --sql "SELECT ..."          # 直接执行 SQL（不调 LLM）
    python -m sql_qa "..." --no-summary          # 只出 SQL + 表格，不生成自然语言总结
"""
from .schema import SCHEMA_TEXT, ALLOWED_TABLES
from .llm_client import LLMClient
from .sql_generator import SQLGenerator, NoSQLAnswer
from .sql_executor import SQLExecutor, SQLValidationError
from .sql_qa import SQLQA

__all__ = [
    "SCHEMA_TEXT", "ALLOWED_TABLES", "LLMClient",
    "SQLGenerator", "NoSQLAnswer", "SQLExecutor", "SQLValidationError", "SQLQA",
]
