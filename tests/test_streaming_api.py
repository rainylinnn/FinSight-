"""流式输出（yield）离线验证 —— 不依赖 MySQL / Redis / Milvus / DashScope。

思路：把 new_main 换成假的模块塞进 sys.modules，再 import app，
这样 app.qa_system 就是我们控制的“假 QA 系统”，可以精确控制
每个 token 的产出时间、是否抛异常、是否被提前关闭。

运行（在项目根目录）：
    .venv\\Scripts\\python.exe -m unittest tests.test_streaming_api -v
"""
import asyncio
import json
import os
import sys
import time
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)  # app.py 里的 StaticFiles(directory="static") 依赖工作目录


# ---------------------------------------------------------------- 打桩
class FakeBM25:
    """按需返回 (answer, need_rag)"""

    def __init__(self, answer=None, need_rag=False):
        self.answer = answer
        self.need_rag = need_rag
        self.calls = []

    def search(self, query, threshold=0.85):
        self.calls.append(query)
        return self.answer, self.need_rag


class FakeConfig:
    VALID_SOURCES = ["ai", "java", "bigdata"]


class _Raise:
    def __init__(self, exc):
        self.exc = exc


class FakeQASystem:
    """模拟 IntegratedQASystem.query 这个“同步生成器”

    script 里的每一项：
      ("文本", False)  -> 中间分片
      ("文本", True)   -> 一次性完整答案，流应当到此结束
      _Raise(exc)      -> 在流中间抛异常
    delay 用来模拟 LLM 逐字返回的耗时（同步阻塞，正是需要被丢进线程池的那种）。
    """

    def __init__(self, script, delay=0.0, bm25_answer=None, bm25_need_rag=False):
        self.script = script
        self.delay = delay
        self.config = FakeConfig()
        self.bm25_search = FakeBM25(bm25_answer, bm25_need_rag)
        self.consumed = 0
        self.completed = False
        self.closed = False
        self.queries = []

    def query(self, query, source_filter=None, session_id=None):
        self.queries.append((query, source_filter, session_id))
        try:
            for item in self.script:
                if self.delay:
                    time.sleep(self.delay)  # 注意：故意阻塞，用来验证事件循环没被卡住
                if isinstance(item, _Raise):
                    raise item.exc
                self.consumed += 1
                yield item
            self.completed = True
        finally:
            self.closed = True

    # 兼容 app.py 里其它入口
    def get_session_history(self, session_id):
        return []

    def clear_session_history(self, session_id):
        return True


def _install_fake_new_main():
    module = types.ModuleType("new_main")

    class _PlaceholderIntegratedQASystem:
        def __init__(self):
            self.config = FakeConfig()
            self.bm25_search = FakeBM25()

    module.IntegratedQASystem = _PlaceholderIntegratedQASystem
    sys.modules["new_main"] = module


_install_fake_new_main()

from fastapi.testclient import TestClient  # noqa: E402
from base import logger  # noqa: E402

import app as app_module  # noqa: E402


# ---------------------------------------------------------------- 工具
def parse_sse(text):
    """把 SSE 文本解析成 [(event, payload_dict), ...]"""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("event:"):
            events.append([line[len("event:"):].strip(), None])
        elif line.startswith("data:"):
            payload = json.loads(line[len("data:"):].strip())
            if events and events[-1][1] is None:
                events[-1][1] = payload
            else:
                events.append([payload.get("type"), payload])
    return [(name, payload) for name, payload in events]


def collect_ws(ws, expect_end=True):
    """收一条完整回答，返回消息列表"""
    messages = []
    while True:
        message = ws.receive_json()
        messages.append(message)
        if message["type"] in ("end", "error") and (expect_end or message["type"] == "error"):
            return messages


def call_asgi_post(path, payload):
    """直接按 ASGI 协议调用 app，返回 (status, headers, 起始时间, [(时刻, body分片), ...])

    绕开 TestClient 的响应体缓冲，能观察到服务端真实的“分片刷出”时机。
    """
    body = json.dumps(payload).encode("utf-8")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    started = {}
    chunks = []
    t0 = time.monotonic()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            started["status"] = message["status"]
            started["headers"] = message["headers"]
        elif message["type"] == "http.response.body":
            chunks.append((time.monotonic(), message.get("body", b"")))

    async def main():
        await app_module.app(scope, receive, send)

    asyncio.run(main())
    return started["status"], started["headers"], t0, chunks


# ---------------------------------------------------------------- 生成器本身
class TestIterQaTokens(unittest.TestCase):
    def setUp(self):
        self._saved = app_module.qa_system

    def tearDown(self):
        app_module.qa_system = self._saved

    def test_tokens_pass_through_in_order_and_stop_at_terminator(self):
        fake = FakeQASystem([("你", False), ("好", False), ("呀", False), ("", True)])
        app_module.qa_system = fake

        async def main():
            return [item async for item in app_module.iter_qa_tokens("q", None, "s")]

        got = asyncio.run(main())
        self.assertEqual(got, [("你", False), ("好", False), ("呀", False), ("", True)])
        self.assertEqual(fake.queries, [("q", None, "s")])

    def test_single_shot_answer_terminates_stream(self):
        """MySQL/BM25 命中时是一次性答案 (token, True)，后面的分片不应继续消费"""
        fake = FakeQASystem([("MySQL里的答案", True), ("不该出现", False)])
        app_module.qa_system = fake

        async def main():
            return [item async for item in app_module.iter_qa_tokens("q", None, "s")]

        got = asyncio.run(main())
        self.assertEqual(got, [("MySQL里的答案", True)])
        self.assertEqual(fake.consumed, 1, "is_complete=True 之后不应继续迭代生成器")

    def test_tokens_arrive_incrementally(self):
        """每个 token 都要立刻交给调用方，而不是等 LLM 全部生成完才一次性返回"""
        fake = FakeQASystem([(f"t{i}", False) for i in range(4)] + [("", True)], delay=0.15)
        app_module.qa_system = fake

        async def main():
            stamps = []
            async for _token, is_complete in app_module.iter_qa_tokens("q", None, "s"):
                stamps.append(time.monotonic())
                if is_complete:
                    break
            return stamps

        stamps = asyncio.run(main())
        self.assertEqual(len(stamps), 5)
        # 4 个分片各 sleep 0.15s，首尾至少应差 0.45s；若被攒成一坨则差值接近 0
        self.assertGreater(stamps[-1] - stamps[0], 0.3,
                           "token 是攒在一起返回的，说明流式输出没有真正逐块 yield")

    def test_blocking_generator_does_not_block_event_loop(self):
        """同步生成器必须在工作线程里迭代：否则整个事件循环被卡住，其他用户的流也一起卡"""
        fake = FakeQASystem([(f"t{i}", False) for i in range(5)] + [("", True)], delay=0.1)
        app_module.qa_system = fake

        async def main():
            ticks = 0
            stop = asyncio.Event()

            async def heartbeat():
                nonlocal ticks
                while not stop.is_set():
                    ticks += 1
                    await asyncio.sleep(0.02)

            beat = asyncio.create_task(heartbeat())
            async for _token, is_complete in app_module.iter_qa_tokens("q", None, "s"):
                if is_complete:
                    break
            stop.set()
            await beat
            return ticks

        ticks = asyncio.run(main())
        # 生成器总共阻塞 0.5s，心跳每 20ms 一次 → 正常应有 ~25 次
        self.assertGreaterEqual(ticks, 5,
                                f"事件循环被阻塞了（心跳只跳了 {ticks} 次），需要 iterate_in_threadpool")

    def test_early_close_releases_underlying_generator(self):
        """客户端中途断开时，底层的同步生成器要被 close 掉（释放 LLM 连接）"""
        fake = FakeQASystem([(f"t{i}", False) for i in range(5)] + [("", True)])
        app_module.qa_system = fake

        async def main():
            agen = app_module.iter_qa_tokens("q", None, "s")
            first = await agen.__anext__()
            await agen.aclose()  # 等价于 Starlette 在客户端断开时对生成器做的事
            return first

        first = asyncio.run(main())
        self.assertEqual(first, ("t0", False))
        self.assertTrue(fake.closed, "底层生成器没有被 close")
        self.assertFalse(fake.completed, "生成器不应该被跑完")


# ---------------------------------------------------------------- SSE 接口
class TestSSEEndpoint(unittest.TestCase):
    def setUp(self):
        self._saved = app_module.qa_system
        self.client = TestClient(app_module.app)

    def tearDown(self):
        app_module.qa_system = self._saved
        self.client.close()

    def _post_stream(self, payload, fake):
        app_module.qa_system = fake
        response = self.client.post("/api/query/stream", json=payload)
        return response

    def test_sse_event_sequence_and_payloads(self):
        fake = FakeQASystem([("装饰器", False), ("是", False), ("语法糖", False), ("", True)])
        response = self._post_stream({"query": "什么是装饰器", "session_id": "sess-1"}, fake)

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertEqual(response.headers.get("x-accel-buffering"), "no")
        self.assertEqual(response.headers.get("cache-control"), "no-cache")

        events = parse_sse(response.text)
        self.assertEqual([name for name, _ in events], ["start", "token", "token", "token", "end"])

        self.assertEqual(events[0][1], {"type": "start", "session_id": "sess-1"})
        self.assertEqual("".join(p["token"] for name, p in events if name == "token"), "装饰器是语法糖")
        end = events[-1][1]
        self.assertTrue(end["is_complete"])
        self.assertEqual(end["session_id"], "sess-1")
        self.assertIsInstance(end["processing_time"], float)

    def test_sse_body_is_flushed_chunk_by_chunk(self):
        """确认服务端是“一个一个 body 分片刷出去”的，而不是攒完一次性返回。

        这里不能走 TestClient —— 它会把整个响应体缓冲下来再交给测试代码，
        时间戳自然全为 0。直接按 ASGI 协议调用 app，观察 http.response.body
        消息的产生时刻，这才是 uvicorn 真正写进 socket 的粒度。
        """
        fake = FakeQASystem([(f"第{i}段", False) for i in range(4)] + [("", True)], delay=0.15)
        app_module.qa_system = fake

        status, headers, t0, body_chunks = call_asgi_post(
            "/api/query/stream", {"query": "流式测试"}
        )

        self.assertEqual(status, 200)
        header_map = {k.decode(): v.decode() for k, v in headers}
        self.assertIn("text/event-stream", header_map["content-type"])
        self.assertEqual(header_map.get("x-accel-buffering"), "no")

        payloads = [json.loads(chunk.decode("utf-8").split("data:", 1)[1])
                    for _t, chunk in body_chunks]
        kinds = [p["type"] for p in payloads]
        self.assertEqual(kinds, ["start", "token", "token", "token", "token", "end"])

        # 第一个分片必须立刻发出（不能等 LLM 生成完）
        self.assertLess(body_chunks[0][0] - t0, 0.1,
                        "start 事件被拖延了，说明响应不是流式的")

        token_times = [t for (t, chunk), kind in zip(body_chunks, kinds) if kind == "token"]
        self.assertGreater(token_times[-1] - token_times[0], 0.3,
                           "token 被攒成一坨才刷出去，不是逐块流式输出")

    def test_sse_greeting_bypasses_llm(self):
        fake = FakeQASystem([("不该被调用", False)], delay=0)
        response = self._post_stream({"query": "你好"}, fake)

        events = parse_sse(response.text)
        self.assertEqual([name for name, _ in events], ["start", "token", "end"])
        self.assertIn("黑马程序员", events[1][1]["token"])
        self.assertEqual(fake.consumed, 0, "问候语不应触发 QA 系统/LLM")

    def test_sse_reports_error_without_swallowing_it(self):
        fake = FakeQASystem([("部分内容", False), _Raise(RuntimeError("LLM 挂了"))])
        response = self._post_stream({"query": "会失败的问题"}, fake)

        events = parse_sse(response.text)
        self.assertEqual([name for name, _ in events], ["start", "token", "error"])
        self.assertIn("LLM 挂了", events[-1][1]["error"])

    def test_sse_generates_session_id_when_missing(self):
        fake = FakeQASystem([("答案", True)])
        response = self._post_stream({"query": "没有 session_id"}, fake)

        events = parse_sse(response.text)
        session_id = events[0][1]["session_id"]
        self.assertTrue(session_id)
        self.assertEqual(events[1][1]["session_id"], session_id)


# ---------------------------------------------------------------- WebSocket 接口
class TestWebSocketEndpoint(unittest.TestCase):
    def setUp(self):
        self._saved = app_module.qa_system

    def tearDown(self):
        app_module.qa_system = self._saved

    def test_websocket_stream_protocol(self):
        app_module.qa_system = FakeQASystem([("A", False), ("B", False), ("", True)])

        with TestClient(app_module.app) as client:
            with client.websocket_connect("/api/stream") as ws:
                ws.send_json({"query": "问题", "session_id": "ws-1"})
                messages = collect_ws(ws)

        self.assertEqual([m["type"] for m in messages], ["start", "token", "token", "end"])
        self.assertEqual("".join(m["token"] for m in messages if m["type"] == "token"), "AB")
        self.assertTrue(messages[-1]["is_complete"])
        self.assertEqual(messages[0]["session_id"], "ws-1")

    def test_connection_survives_greeting_and_serves_next_question(self):
        """回归测试：以前问候语分支用 break 跳出 while，问一句“你好”连接就被关掉了"""
        app_module.qa_system = FakeQASystem([("正式回答", False), ("", True)])

        with TestClient(app_module.app) as client:
            with client.websocket_connect("/api/stream") as ws:
                # 第 1 轮：问候语
                ws.send_json({"query": "你好", "session_id": "ws-2"})
                greeting = collect_ws(ws)
                self.assertEqual([m["type"] for m in greeting], ["start", "token", "end"])
                self.assertIn("黑马程序员", greeting[1]["token"])

                # 第 2 轮：同一个连接继续问真正的问题（连接若已关闭，这里会抛异常）
                ws.send_json({"query": "什么是装饰器", "session_id": "ws-2"})
                answer = collect_ws(ws)
                self.assertEqual([m["type"] for m in answer], ["start", "token", "end"])
                self.assertEqual(answer[1]["token"], "正式回答")

                # 第 3 轮：再来一次，确认可以持续复用
                ws.send_json({"query": "再问一次", "session_id": "ws-2"})
                again = collect_ws(ws)
                self.assertEqual(again[-1]["type"], "end")

    def test_websocket_error_keeps_connection_usable(self):
        app_module.qa_system = FakeQASystem([_Raise(RuntimeError("检索崩了"))])

        with TestClient(app_module.app) as client:
            with client.websocket_connect("/api/stream") as ws:
                ws.send_json({"query": "会失败的问题"})
                first = collect_ws(ws)
                self.assertEqual([m["type"] for m in first], ["start", "error"])
                self.assertIn("检索崩了", first[-1]["error"])

                # 错误之后连接仍然可用
                app_module.qa_system = FakeQASystem([("恢复后的回答", True)])
                ws.send_json({"query": "再来一个"})
                second = collect_ws(ws)
                self.assertEqual([m["type"] for m in second], ["start", "token", "end"])
                self.assertEqual(second[1]["token"], "恢复后的回答")

    def test_websocket_rejects_bad_payload(self):
        app_module.qa_system = FakeQASystem([("不该被调用", False)])

        with TestClient(app_module.app) as client:
            with client.websocket_connect("/api/stream") as ws:
                ws.send_text("这不是 JSON")
                message = ws.receive_json()
                self.assertEqual(message["type"], "error")

                ws.send_json({"query": "   "})
                message = ws.receive_json()
                self.assertEqual(message["type"], "error")

                # 连接仍然可用
                ws.send_json({"query": "正常问题"})
                messages = collect_ws(ws)
                self.assertEqual(messages[0]["type"], "start")


# ---------------------------------------------------------------- 非流式接口
class TestQueryEndpoint(unittest.TestCase):
    def setUp(self):
        self._saved = app_module.qa_system
        self.client = TestClient(app_module.app)

    def tearDown(self):
        app_module.qa_system = self._saved
        self.client.close()

    def test_bm25_hit_returns_answer(self):
        app_module.qa_system = FakeQASystem([], bm25_answer="BM25 直接命中", bm25_need_rag=False)
        body = self.client.post("/api/query", json={"query": "学费多少"}).json()

        self.assertEqual(body["answer"], "BM25 直接命中")
        self.assertFalse(body["is_streaming"])
        self.assertIn("processing_time", body)

    def test_need_rag_points_to_streaming_interfaces(self):
        app_module.qa_system = FakeQASystem([], bm25_answer=None, bm25_need_rag=True)
        body = self.client.post("/api/query", json={"query": "复杂问题"}).json()

        self.assertTrue(body["is_streaming"])
        self.assertIn("/api/query/stream", body["answer"])
        self.assertIn("/api/stream", body["answer"])

    def test_greeting_short_circuit(self):
        app_module.qa_system = FakeQASystem([], bm25_answer="不该被使用")
        body = self.client.post("/api/query", json={"query": "你是谁"}).json()

        self.assertFalse(body["is_streaming"])
        self.assertIn("黑马程序员", body["answer"])


class TestStreamingResponseHeaderContract(unittest.TestCase):
    """接口文档里承诺的东西，别在重构中丢掉"""

    def test_documented_routes_exist(self):
        paths = {route.path for route in app_module.app.routes}
        for path in ("/health", "/api/create_session", "/api/history/{session_id}",
                     "/api/query", "/api/query/stream", "/api/stream", "/api/sources"):
            self.assertIn(path, paths)

    def test_only_one_check_greeting_definition(self):
        """回归测试：app.py 里曾经重复定义了两份 check_greeting，后者静默覆盖前者"""
        with open(os.path.join(ROOT, "app.py"), encoding="utf-8") as f:
            source = f.read()
        self.assertEqual(source.count("def check_greeting("), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
