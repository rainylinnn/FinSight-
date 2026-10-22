"""验证真实 new_main.IntegratedQASystem.call_dashscope 的生成器语义。

重点回归：生成器里写 `return "错误：..."` 只会进 StopIteration.value，
调用方 `for token in self.llm(...)` 永远拿不到 → 前端表现为“回答是空的”。
这里用假的 OpenAI 客户端把两条路径都钉死。

注意：import new_main 会连带 import torch / langchain / pymilvus 等重依赖，
第一次跑大约 2~3 分钟，属正常现象（不需要 MySQL/Redis/Milvus 真的在跑）。

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_call_dashscope_stream -v
"""
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


class _CreateFails:
    """让 chat.completions.create(...) 本身抛错（建连/鉴权失败）"""

    def __init__(self, exc):
        self.exc = exc


class _Chunk:
    """构造一个 OpenAI 流式 chunk"""

    def __init__(self, content=None, empty_choices=False, no_delta_content=False):
        if empty_choices:
            self.choices = []
        elif no_delta_content:
            self.choices = [types.SimpleNamespace(delta=types.SimpleNamespace(content=None))]
        else:
            self.choices = [types.SimpleNamespace(delta=types.SimpleNamespace(content=content))]


class _MidStreamFails:
    """流式读取到一半时抛错"""

    def __init__(self, exc):
        self.exc = exc


def make_fake_openai(items):
    """items: _CreateFails / _Chunk / _MidStreamFails / str 的序列"""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        first = items[0]

        def chunk_iter():
            for item in items:
                if isinstance(item, _CreateFails):
                    raise item.exc
                if isinstance(item, _MidStreamFails):
                    raise item.exc
                yield item

        if isinstance(first, _CreateFails):
            raise first.exc
        return chunk_iter()

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    return client, calls


def make_system(items):
    """不跑 __init__（那会连 MySQL/Redis/Milvus），只注入 call_dashscope 需要的属性"""
    from new_main import IntegratedQASystem
    from base import logger

    qa = IntegratedQASystem.__new__(IntegratedQASystem)
    qa.logger = logger
    qa.config = types.SimpleNamespace(LLM_MODEL="qwen3-max")
    client, calls = make_fake_openai(items)
    qa.client = client
    return qa, calls


class TestCallDashscopeStreaming(unittest.TestCase):
    def test_yields_chunks_in_order(self):
        qa, calls = make_system([_Chunk("流"), _Chunk("式"), _Chunk("输出")])
        self.assertEqual(list(qa.call_dashscope("hi")), ["流", "式", "输出"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["stream"], "必须带 stream=True 才是流式调用")
        self.assertEqual(calls[0]["model"], "qwen3-max")
        self.assertEqual(calls[0]["messages"][-1]["content"], "hi")

    def test_skips_useless_chunks(self):
        qa, _ = make_system([
            _Chunk(empty_choices=True),
            _Chunk(no_delta_content=True),
            _Chunk("有用"),
        ])
        self.assertEqual(list(qa.call_dashscope("hi")), ["有用"])

    def test_generator_is_lazy(self):
        """生成器函数：不迭代就不发请求"""
        qa, calls = make_system([_Chunk("x")])
        gen = qa.call_dashscope("hi")
        self.assertEqual(calls, [], "调用生成器函数时不应该立刻发请求")
        next(gen)
        self.assertEqual(len(calls), 1)

    def test_create_failure_propagates_instead_of_silently_ending(self):
        """回归：以前这里 return 错误文本 → 调用方一个字都收不到，还以为是“空回答”"""
        qa, _ = make_system([_CreateFails(RuntimeError("鉴权失败/网络不通"))])
        with self.assertRaises(RuntimeError) as ctx:
            list(qa.call_dashscope("hi"))
        self.assertIn("鉴权失败", str(ctx.exception))

    def test_mid_stream_failure_keeps_earlier_tokens_and_propagates(self):
        qa, _ = make_system([_Chunk("已经"), _Chunk("发出去的"), _MidStreamFails(RuntimeError("连接中断"))])
        received = []
        with self.assertRaises(RuntimeError):
            for token in qa.call_dashscope("hi"):
                received.append(token)
        self.assertEqual(received, ["已经", "发出去的"], "出错前已经 yield 的内容不应丢失")

    def test_generate_answer_turns_llm_failure_into_visible_token(self):
        """LLM 挂掉时，RAGSystem.generate_answer 要 yield 出人类可读的兜底文案，而不是空流"""
        from rag_qa.core.new_rag_system import RAGSystem

        qa, _ = make_system([_CreateFails(RuntimeError("余额不足"))])

        rag = RAGSystem.__new__(RAGSystem)
        rag.llm = qa.call_dashscope
        rag.max_prompt_length = 4096 * 3

        class _Classifier:
            def predict_category(self, query):
                return "通用知识"  # 跳过检索，直接走 LLM

        class _Prompt:
            def format(self, **kwargs):
                return "prompt"

        rag.query_classifier = _Classifier()
        rag.rag_prompt = _Prompt()
        rag.strategy_selector = types.SimpleNamespace(select_strategy=lambda q: "直接检索")

        tokens = list(rag.generate_answer("任意问题"))
        self.assertTrue(tokens, "失败时不能是空流")
        text = "".join(tokens)
        self.assertIn("余额不足", text)
        self.assertIn("人工客服", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
