# -*- coding: utf-8 -*-
"""一次问答请求的结构化产物。

流式接口只负责吐 token，但前端还需要「来源卡片」「图表」「数据表」这些
非文本产物。用 QueryContext 作为一次请求的旁路容器：问答链路往里写，
app.py 在流结束时统一打包成一个结构化事件推给前端。
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class QueryContext:
    # 实际走的路由：sql / rag / graph / general / greeting
    route: str = ""
    # RAG 检索到的原文出处（公司 / 年份 / 页码 / 章节 / 片段）
    sources: List[Dict[str, Any]] = field(default_factory=list)
    # ECharts 图表配置（由 agent.chart 生成）
    chart: Optional[Dict[str, Any]] = None
    # SQL / 图谱查询结果表
    table: Optional[Dict[str, Any]] = None
    # 实际执行的 SQL（答辩演示时可展开）
    sql: Optional[str] = None
    # 路由补充说明（例如 graph 后端是 neo4j 还是 mysql）
    backend: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        """转成前端可用的 JSON（去掉空字段，避免前端判断一堆 null）。"""
        raw = asdict(self)
        return {k: v for k, v in raw.items() if v not in (None, "", [], {})}
