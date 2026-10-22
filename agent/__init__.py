# -*- coding: utf-8 -*-
"""FinSight Agent：意图路由 + 查询上下文 + 图表构建。"""
from .context import QueryContext
from .router import (
    route_question,
    detect_company,
    detect_companies,
    METRIC_KEYWORDS,
    DOC_KEYWORDS,
    RELATION_KEYWORDS,
)

__all__ = [
    "QueryContext",
    "route_question",
    "detect_company",
    "detect_companies",
    "METRIC_KEYWORDS",
    "DOC_KEYWORDS",
    "RELATION_KEYWORDS",
]
