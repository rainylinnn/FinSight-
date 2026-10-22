# -*- coding: utf-8 -*-
"""报告导出 API（Word / Markdown）。"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from services import report_service

router = APIRouter(prefix="/api/report", tags=["report"])


class ReportRequest(BaseModel):
    session_id: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    title: Optional[str] = None
    mode: Optional[str] = "auto"
    format: Optional[str] = "docx"   # docx | md


@router.post("/export")
def export_report(req: ReportRequest):
    """导出报告：带 session_id 时导出整段会话，否则导出单条问答。"""
    payload = {
        "title": req.title or "FinSight 酒研助手 · 问答报告",
        "mode": req.mode or "auto",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if req.session_id and not req.question:
        try:
            from app import qa_system
            history = qa_system.get_session_history(req.session_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"读取会话历史失败: {e}")
        if not history:
            raise HTTPException(status_code=400, detail="当前会话没有可导出的对话")
        payload["history"] = history
    else:
        if not req.question or not req.answer:
            raise HTTPException(status_code=400, detail="需要提供 question 和 answer（或 session_id）")
        payload["question"] = req.question
        payload["answer"] = req.answer

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if (req.format or "docx").lower() == "md":
        content = report_service.build_markdown(payload).encode("utf-8")
        return Response(
            content=content,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="finsight_report_{stamp}.md"'},
        )

    content = report_service.build_docx(payload)
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="finsight_report_{stamp}.docx"'},
    )
