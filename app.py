from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool
from starlette.websockets import WebSocketDisconnect, WebSocketState
import anyio
import contextlib
import functools
import os
from pydantic import BaseModel
import json
import uuid
from typing import Optional, AsyncGenerator, Generator, Tuple
import time
import re

# 导入现有的系统
from new_main import IntegratedQASystem
# 导入日志
from base import logger
# 扩展路由：行情 / 图表 / 报告导出（实现在 api/ 与 services/）
from api.chart_api import router as chart_router
from api.market_api import router as market_router
from api.report_api import router as report_router
from api.sector_api import router as sector_router

# 创建应用实例
app = FastAPI(title="FinSight 酒研助手 API", description="白酒行业智能投研助手（年报 RAG + 财务指标 + 股东图谱）")

app.include_router(chart_router)
app.include_router(market_router)
app.include_router(report_router)
app.include_router(sector_router)

# 配置CORS，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
os.makedirs("static", exist_ok=True)

# 创建全局QA系统实例
qa_system = IntegratedQASystem()

# 定义日常问候用语模式和回复
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好！我是 FinSight 酒研助手，可以帮你查白酒公司年报、对比财务指标、查看股东关系。"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是 FinSight 酒研助手，基于 %d 家白酒公司 2023–2025 年报构建的投研问答助手。"
                    % len(qa_system.config.VALID_SOURCES)
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",
        "response": "我在！随时可以帮你查年报原文、财务指标和股东持仓。"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我正在待命，随时回答白酒行业投研问题。"
    }
]


# 定义请求模型
class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None
    # 功能模式：auto(智能问答) / sql(财务指标) / rag(年报原文) / compare(多公司对比) / graph(股东图谱)
    mode: Optional[str] = None


# 定义响应模型
class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float
    # 语义缓存命中标记（命中时 from_cache=true，similarity 为向量相似度）
    from_cache: bool = False
    similarity: Optional[float] = None


# 添加静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


# 根路径重定向到index.html
@app.get("/")
async def read_root():
    # 禁用缓存：改了前端后用户刷新即生效，不用手动清缓存
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# 创建新会话
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


# 登录（演示用固定账号：账号与密码均为 12345）
@app.post("/api/login")
def login(payload: dict):
    """校验账号密码并下发一个前端用的 token。

    说明：这是界面层的门禁，不是安全边界——所有业务接口仍然无鉴权，
    仅用于演示「登录后才能进入主界面」的交互。
    """
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="请输入账号和密码")
    if username != "12345" or password != "12345":
        raise HTTPException(status_code=401, detail="账号或密码错误")

    return {
        "ok": True,
        "token": uuid.uuid4().hex,
        "username": username,
        "expires_in": 8 * 3600,
    }


# 查询历史消息
# 注意：这里用同步 def 而不是 async def，FastAPI 会自动把它丢到线程池执行，
# 避免 MySQL 的阻塞 IO 卡住事件循环（卡住事件循环 = 所有人的流式输出一起卡住）
@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    try:
        history = qa_system.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


# 清除历史消息
@app.delete("/api/history/{session_id}")
def clear_history(session_id: str):
    success = qa_system.clear_session_history(session_id)
    if success:
        return {"status": "success", "message": "历史记录已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除历史记录失败")


# ==================== 引用原文（点击【公司 年份 第X页】看原页） ====================
_source_client = None


def _get_source_client():
    """懒加载只读 Milvus 客户端，避免每次请求重建连接。"""
    global _source_client
    if _source_client is None:
        from pymilvus import MilvusClient

        conf = qa_system.config
        _source_client = MilvusClient(
            uri=f"http://{conf.MILVUS_HOST}:{conf.MILVUS_PORT}",
            db_name=conf.MILVUS_DATABASE_NAME,
            timeout=10,
        )
    return _source_client


@app.get("/api/source")
def get_source(company: str, year: int, page: int):
    """按【公司 / 年份 / 页码】取回年报原文，供前端点击引用时展示。"""
    company = (company or "").strip()
    if not company:
        raise HTTPException(status_code=400, detail="缺少公司名")

    try:
        conf = qa_system.config
        client = _get_source_client()
        safe_company = company.replace('"', '\\"')
        expr = f'company_name == "{safe_company}" and year == {int(year)} and page == {int(page)}'
        rows = client.query(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            filter=expr,
            output_fields=[
                "text", "section", "section_no", "page",
                "company_name", "year", "parent_content", "doc_title",
            ],
            limit=60,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取原文失败：{e}")

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="这一页没有收录进向量库（财务报告章节不入库，指标类问题请用「财务指标」模式）",
        )

    rows.sort(key=lambda r: (r.get("section_no") or 0))
    seen = set()
    chunks = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        chunks.append({
            "text": text,
            "section": r.get("section") or "",
            "parent": (r.get("parent_content") or "").strip()[:600],
        })

    sections = [c["section"] for c in chunks if c["section"]]
    return {
        "company": rows[0].get("company_name") or company,
        "year": int(year),
        "page": int(page),
        "section": sections[0] if sections else "",
        "doc_title": rows[0].get("doc_title") or "",
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


# 历史会话列表：按最近活跃排序，取每个会话的首问当标题
@app.get("/api/sessions")
def list_sessions(limit: int = 50, include_tests: bool = False):
    """列出全部历史会话。

    默认只返回标准 uuid 会话——开发期的临时 session（smoke-xxx / mode-http 等）
    混在列表里会把真实对话冲掉，需要时可传 include_tests=true 看全部。
    """
    try:
        where = "" if include_tests else (
            " WHERE c.session_id REGEXP '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-'"
        )
        sql = (
            "SELECT c.session_id, LEFT(c.question, 60) AS first_q, c.timestamp AS started_at,"
            "       t.turns, t.last_at"
            " FROM conversations c"
            " JOIN (SELECT session_id, COUNT(*) AS turns, MAX(timestamp) AS last_at, MIN(id) AS min_id"
            "       FROM conversations GROUP BY session_id) t"
            "   ON t.session_id = c.session_id AND t.min_id = c.id"
            + where
            + " ORDER BY t.last_at DESC LIMIT %s"
        )
        cur = qa_system.mysql_client.cursor
        cur.execute(sql, (int(limit),))
        rows = cur.fetchall()
        return {
            "sessions": [
                {
                    "session_id": r[0],
                    "title": r[1] or "未命名会话",
                    "started_at": r[2].strftime("%Y-%m-%d %H:%M") if r[2] else None,
                    "turns": int(r[3] or 0),
                    "last_at": r[4].strftime("%Y-%m-%d %H:%M") if r[4] else None,
                }
                for r in rows
            ]
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"获取会话列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {e}")


# ==================== 语义缓存（统计 / 清空） ====================

@app.get("/api/cache/stats")
def cache_stats():
    """语义缓存命中率统计（答辩可用来展示缓存效果）。"""
    cache = getattr(qa_system, "semantic_cache", None)
    if cache is None:
        return {"enabled": False, "backend": "none", "hits": 0, "misses": 0,
                "writes": 0, "entries": 0, "hit_rate": 0.0}
    return cache.stats()


@app.delete("/api/cache")
def cache_clear():
    """清空语义缓存。"""
    cache = getattr(qa_system, "semantic_cache", None)
    if cache is None:
        raise HTTPException(status_code=400, detail="语义缓存未启用")
    ok = cache.clear()
    return {"status": "success" if ok else "failed"}


# 检查是否为日常问候用语并返回模板回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


# ==================== 流式输出公共逻辑 ====================

async def iter_qa_tokens(query: str, source_filter: Optional[str],
                         session_id: str, mode: Optional[str] = None,
                         event_sink: Optional[list] = None) -> AsyncGenerator[Tuple[str, bool], None]:
    """把同步生成器 qa_system.query(...) 变成异步可迭代对象，逐块产出 (token, is_complete)。

    qa_system.query 是 yield 出来的同步生成器，内部是阻塞 IO（MySQL/Redis/LLM 流式请求）。
    如果在 async 函数里直接 `for token, is_complete in qa_system.query(...)`，
    事件循环会被这段同步代码占住，其他用户的流式输出也会一起“卡住不吐字”。
    所以用 iterate_in_threadpool 把每次 next() 放到线程池里跑，事件循环始终空闲。

    生成器协议（来自 IntegratedQASystem.query）：
      (token, False) -> 中间分片，直接推给前端
      (token, True)  -> 一次性完整答案（如 MySQL/BM25 命中的答案）
      ("", True)     -> 流结束标志
    """
    gen: Generator[Tuple[str, bool], None, None] = qa_system.query(
        query, source_filter=source_filter, session_id=session_id, mode=mode,
        event_sink=event_sink,
    )
    try:
        async for token, is_complete in iterate_in_threadpool(gen):
            yield token, is_complete
            if is_complete:
                break
    finally:
        # 客户端中途断开（SSE/WS 关闭）时显式关闭生成器，释放 LLM 连接等资源。
        # close() 是同步且廉价的（只是往暂停点抛 GeneratorExit），不会被取消打断。
        close = getattr(gen, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()


def _sse_event(payload: dict) -> str:
    """按 SSE 规范拼一条事件：event 行给 EventSource 用，data 行是 JSON 正文"""
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {payload['type']}\ndata: {data}\n\n"


# 入参 出参
# 非流式查询接口
@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    # 快路径（Redis/MySQL FAQ）可选：金融项目默认关闭，直接走完整问答链路
    if qa_system.bm25_search is None:
        event_sink = []

        def _run_full_query():
            return "".join(
                token for token, _done in qa_system.query(
                    request.query, session_id=session_id, mode=request.mode,
                    event_sink=event_sink)
            )

        answer = await anyio.to_thread.run_sync(_run_full_query)
        cache_hit = next((e for e in event_sink if e.get("type") == "cache"), None)
        return {
            "answer": answer,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time,
            "from_cache": bool(cache_hit),
            "similarity": (cache_hit or {}).get("similarity"),
        }

    # 判断是否需要流式处理（基于 need_rag）
    # BM25 检索里有 Redis/MySQL 阻塞 IO，同样丢到线程池执行
    answer, need_rag = await anyio.to_thread.run_sync(
        functools.partial(qa_system.bm25_search.search, request.query, threshold=0.85)
    )
    if need_rag:
        # 需要 RAG 和 LLM 处理，返回流式响应提示
        return {
            "answer": "请使用流式接口获取流式响应：WebSocket /api/stream 或 SSE /api/query/stream",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    # 非流式查询，直接返回 BM25 检索的答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }


# ==================== 流式接口 1：SSE (HTTP) ====================

@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """SSE 流式查询接口：一次 HTTP 请求，服务端用 yield 逐块把答案推回来。

    适合浏览器 fetch + ReadableStream、curl、Postman、以及任何 HTTP 客户端，
    不需要像 WebSocket 那样先建长连接、也不需要先调 /api/query 探测是否要流式。

    事件格式（与 WebSocket 完全一致，方便前端复用同一套解析逻辑）：
      event: start / token / end / error
      data:  {"type": "...", ...}
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def event_stream() -> AsyncGenerator[str, None]:
        start_time = time.time()
        # 先发 start：既是协议的一部分，也相当于立刻给客户端一个心跳
        yield _sse_event({"type": "start", "session_id": session_id})

        # 日常问候语不用走 LLM，直接一次性返回
        greeting_response = check_greeting(request.query)
        if greeting_response:
            yield _sse_event({"type": "token", "token": greeting_response, "session_id": session_id})
            yield _sse_event({
                "type": "end",
                "session_id": session_id,
                "is_complete": True,
                "processing_time": time.time() - start_time,
            })
            return

        try:
            event_sink = []
            async for token, is_complete in iter_qa_tokens(
                    request.query, request.source_filter, session_id, request.mode,
                    event_sink=event_sink):
                if token:
                    yield _sse_event({"type": "token", "token": token, "session_id": session_id})
                if is_complete:
                    break
            # 结构化事件（如「语义缓存命中」）：在正文之后、结束之前推给前端
            for ev in event_sink:
                yield _sse_event(dict(ev, session_id=session_id))
            yield _sse_event({
                "type": "end",
                "session_id": session_id,
                "is_complete": True,
                "processing_time": time.time() - start_time,
            })
        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield _sse_event({"type": "error", "error": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",          # 不要缓存流式响应
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",            # 关掉 nginx 缓冲，否则会被攒成一大块
        },
    )


# ==================== 流式接口 2：WebSocket ====================

async def _ws_send(websocket: WebSocket, payload: dict) -> bool:
    """安全发送：连接已关闭/正在关闭时返回 False，调用方据此结束本次处理"""
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.warning(f"WebSocket 发送失败（客户端可能已断开）: {e}")
        return False


# 流式查询WebSocket接口
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 接收消息（一个连接可以连续问多个问题）
            data = await websocket.receive_text()

            try:
                request_data = json.loads(data)
            except json.JSONDecodeError:
                await _ws_send(websocket, {"type": "error", "error": "请求格式错误，需要 JSON 字符串"})
                continue

            query = (request_data.get("query") or "").strip()
            if not query:
                await _ws_send(websocket, {"type": "error", "error": "query 不能为空"})
                continue

            source_filter = request_data.get("source_filter")
            mode = request_data.get("mode")
            session_id = request_data.get("session_id") or str(uuid.uuid4())

            start_time = time.time()

            # 发送开始标志
            if not await _ws_send(websocket, {"type": "start", "session_id": session_id}):
                break

            # 检查是否为日常问候。注意这里必须是 continue 而不是 break：
            # break 会跳出 while True，问一句“你好”就把整个连接关掉了。
            greeting_response = check_greeting(query)
            if greeting_response:
                await _ws_send(websocket, {"type": "token", "token": greeting_response, "session_id": session_id})
                await _ws_send(websocket, {
                    "type": "end",
                    "session_id": session_id,
                    "is_complete": True,
                    "processing_time": time.time() - start_time,
                })
                continue

            # 调用QA系统进行查询，流式返回结果
            try:
                event_sink = []
                async for token, is_complete in iter_qa_tokens(
                        query, source_filter, session_id, mode, event_sink=event_sink):
                    if token:
                        if not await _ws_send(websocket, {
                            "type": "token",
                            "token": token,
                            "session_id": session_id,
                        }):
                            return
                    if is_complete:
                        break
                # 结构化事件（如「语义缓存命中」）
                for ev in event_sink:
                    if not await _ws_send(websocket, dict(ev, session_id=session_id)):
                        return
            except Exception as e:
                logger.error(f"流式查询失败: {str(e)}")
                await _ws_send(websocket, {"type": "error", "error": str(e)})
                continue

            # 发送结束标志
            if not await _ws_send(websocket, {
                "type": "end",
                "session_id": session_id,
                "is_complete": True,
                "processing_time": time.time() - start_time,
            }):
                break

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await _ws_send(websocket, {"type": "error", "error": str(e)})
    finally:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {str(e)}")


# 健康检查端点 用户运维监控服务存活状态
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 获取有效的学科类别
@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}


if __name__ == "__main__":
    import uvicorn

    # 直接传 app 实例：用 "app:app" 字符串会让 uvicorn 再次 import app 模块，
    # 把 IntegratedQASystem（向量库 + BERT 分类器）重复初始化一遍，启动慢一倍、内存翻倍。
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
