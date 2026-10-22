# 导入MySQL和Redis客户端，管理数据库和缓存
from mysql_qa import MySQLClient, RedisClient, BM25Search
# 导入RAG系统组件
from rag_qa import VectorStore, RAGSystem
from rag_qa.core.prompts import RAGPrompts
# 导入配置和日志
from base import logger, Config
# 导入OpenAI客户端（OpenAI 兼容接口，当前接 DeepSeek）
from openai import OpenAI
import httpx
# 导入时间库，记录处理时间
import time
# 导入正则（意图识别用）
import re
# 导入UUID生成唯一会话ID
import uuid
# 导入pymysql错误处理
import pymysql
# 导入意图路由与 SQL 问答（P2）
from agent import route_question
from agent.router import detect_company, detect_companies, has_domain_signal
from sql_qa import SQLQA
from services import market_service
from services.semantic_cache import SemanticCache


class IntegratedQASystem:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        # 初始化配置
        self.config = Config()
        # 初始化MySQL客户端
        self.mysql_client = MySQLClient()
        # 初始化Redis客户端
        self.redis_client = RedisClient()
        # 初始化BM25搜索（旧项目的 jpkb FAQ 表在金融项目里不存在：
        # 且共享 Redis 里可能残留旧项目的 qa_original_questions 题库，
        # 因此金融项目默认 enable_fast_path=false，直接跳过快路径）
        if getattr(self.config, "ENABLE_FAST_PATH", False):
            try:
                self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
                self.has_fast_path = True
            except Exception as e:
                self.logger.warning(f"BM25 快路径不可用（将直接走 RAG）: {e}")
                self.bm25_search = None
                self.has_fast_path = False
        else:
            self.bm25_search = None
            self.has_fast_path = False
            self.logger.info("FAQ 快路径已关闭（enable_fast_path=false），问题直接走 SQL/RAG")
        try:
            # 初始化 OpenAI 兼容客户端（当前接 DeepSeek：base_url + api_key 来自环境变量）
            # trust_env=False：绕过 Windows 系统代理（实测系统代理会让 api.deepseek.com 的 TLS 握手失败）
            self.client = OpenAI(api_key=self.config.LLM_API_KEY,
                                 base_url=self.config.LLM_BASE_URL,
                                 http_client=httpx.Client(trust_env=False))
        except Exception as e:
            self.logger.error(f"LLM 客户端初始化失败: {e}")
            raise
        # 初始化向量存储
        self.vector_store = VectorStore()
        # 初始化RAG系统
        self.rag_system = RAGSystem(self.vector_store, self.call_llm)
        # 语义缓存：复用 VectorStore 已加载的 BGE-M3 编码器（不额外占一份显存/内存）
        self.semantic_cache = None
        if getattr(self.config, "SEMANTIC_ENABLED", False):
            try:
                def _embed(texts):
                    return self.vector_store.embedding_function(texts)["dense"]

                self.semantic_cache = SemanticCache(
                    _embed, conf=self.config, redis_client=self.redis_client)
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"语义缓存初始化失败（已关闭，不影响问答）: {e}")
        # 知识库外问题（行情/估值/预测等）的通用回答模板
        self.general_prompt = RAGPrompts.general_prompt()
        # 初始化 SQL 问答（NL2SQL）；没有 API Key / 依赖缺失时降级为不可用
        try:
            self.sql_qa = SQLQA()
        except Exception as e:
            self.logger.warning(f"SQL QA 初始化失败（指标类问题将回退 RAG）: {e}")
            self.sql_qa = None
        # 初始化对话历史表
        self.init_conversation_table()

    def init_conversation_table(self):
        """初始化MySQL中的conversations表，用于存储对话历史"""
        try:
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                )
            """)
            self.mysql_client.connection.commit()
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"初始化对话历史表失败: {e}")
            raise

    # 调用模型
    def call_llm(self, prompt):
        """调用 LLM（OpenAI 兼容接口，当前为 DeepSeek）生成答案（生成器：每收到一个分片就 yield 给上层）

        注意：这是生成器函数，调用它不会立刻发请求，只有被迭代（for/next）时才执行。
        所以出错时不能写 `return "错误：..."`——生成器里的 return 只会把值塞进
        StopIteration.value，调用方 `for token in self.llm(...)` 永远看不到，
        表现就是“LLM 调用失败 → 前端一个字都不显示”。这里改为 raise，交给
        RAGSystem.generate_answer 的 except 分支给出可见的兜底回答。
        """
        # 第一阶段：建连 / 发起请求，出错直接抛出
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True  # 启用流式输出
            )
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise

        # 第二阶段：逐块读取，出错同样抛出（已 yield 出去的内容无法撤回）
        try:
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    # 每读到一个分片就 yield，上游一路透传到前端
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"LLM流式输出中断: {e}")
            raise

    def _fetch_recent_history(self, session_id: str) -> list:
        """获取最近5轮对话历史"""
        try:
            self.mysql_client.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (session_id, 5))
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except pymysql.MySQLError as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []

    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """更新会话历史到MySQL，保留最近5轮对话"""
        try:
            self.mysql_client.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))
            history = self._fetch_recent_history(session_id)
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS sub
                )
            """, (session_id, session_id, 5))
            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史更新成功")
            return history
        except pymysql.MySQLError as e:
            self.logger.error(f"更新会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            raise
        except Exception as e:
            self.logger.error(f"更新会话历史意外错误: {e}")
            self.mysql_client.connection.rollback()
            raise

    def get_session_history(self, session_id: str) -> list:
        """从MySQL获取会话历史"""
        return self._fetch_recent_history(session_id)

    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        try:
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史已清除")
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"清除会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            return False

    def _rewrite_followup(self, query, history):
        """把“利润呢”“那营收呢”这类短追问，结合上一轮问题改写成完整问题。

        否则追问会因为缺少公司/年份/指标而走错链路（例如“利润呢”没有指标关键词，
        会被当成文档问题去检索年报原文，于是答“未找到”）。
        """
        text = (query or "").strip()
        if not history or len(text) > 8:
            return query
        recent = history[-2:]
        ctx = "\n".join(
            f"Q: {h['question']}\nA: {h['answer'][:200]}" for h in recent
        )
        prompt = (
            "你是投研问答系统的查询改写器。根据最近对话，把用户当前的追问改写成一个"
            "可以独立理解、信息完整的中文问题：补全省略的公司、年份、指标"
            "（例如把“利润”补成“归母净利润”），并保留用户的原始意图。\n"
            "只输出改写后的问题本身（一行），不要回答问题，不要解释。\n\n"
            f"最近对话:\n{ctx}\n\n用户追问: {text}\n改写:"
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                timeout=20,
            )
            content = (completion.choices[0].message.content or "").strip()
            # 去掉模型可能加的“改写：”前缀 / 引号 / 换行（有时会折行导致被截断）
            content = re.sub(r"^(改写|问题)[:：]\s*", "", content)
            rewritten = " ".join(line.strip() for line in content.splitlines() if line.strip())
            rewritten = rewritten.strip().strip('"“”\'')
            if 3 <= len(rewritten) <= 80:
                return rewritten
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"追问改写失败（忽略，用原问题）: {e}")
        return query

    def _classify_intent(self, query):
        """意图识别（在追问改写之前执行）：通用知识 → LLM；专业咨询 → SQL/RAG。

        规则优先、BERT 兜底：金融项目里“公司/年份/指标”这些专业信号必须走数据库，
        不能被教育场景训练出来的 BERT 分类器误判为“通用知识”。
        """
        text = (query or "").strip()
        route = route_question(text, graph_enabled=False)
        if route == "market":
            return "market"
        if route == "unsupported":
            return "unsupported"
        # 定义/解释类问题（如“什么是ROE”），没提到具体公司时直接问 LLM
        is_definition = re.search(r"(什么是|是什么意思|怎么理解|如何理解|有什么区别|解释一下)", text)
        if is_definition and not detect_company(text):
            return "general"
        # 有公司 / 年份 / 指标 / 文档类信号 → 专业咨询
        if detect_company(text) or has_domain_signal(text):
            return "professional"
        # 其余（纯聊天/常识问题）交给 BERT 意图分类器判断
        try:
            category = self.rag_system.query_classifier.predict_category(text)
            if category == "通用知识":
                return "general"
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"意图分类失败，按专业咨询处理: {e}")
        return "professional"

    @staticmethod
    def _looks_like_miss(text):
        """判断 RAG 回答是不是“知识库里没找到”。"""
        t = (text or "").strip()
        if not t:
            return False
        patterns = ("未找到相关", "未找到", "没有找到相关", "未检索到", "未提及",
                    "未收录", "无法从年报", "年报中没有", "年报中未", "查不到",
                    "不包含相关", "该问题无法", "无法用财务指标表回答", "查询失败",
                    "生成失败", "回退 RAG")
        return any(p in t for p in patterns)

    # ==================== 语义缓存 ====================
    def _is_followup(self, query, history):
        """短问题 + 已有历史 = 追问（答案依赖上下文，不能进缓存）。"""
        text = (query or "").strip()
        return bool(history) and len(text) <= 8

    def _cache_lookup(self, query, mode, session_id, history, event_sink):
        """查语义缓存，命中直接返回答案文本（跳过检索 + LLM 生成）。"""
        if not self.semantic_cache:
            return None
        # 行情/闲聊类问题必须实时，不走缓存
        if mode in ("market", "general"):
            return None
        if self._is_followup(query, history):
            return None
        try:
            hit = self.semantic_cache.get(query, mode=mode)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"语义缓存查询失败（忽略缓存）: {e}")
            return None
        if not hit:
            return None

        if session_id:
            try:
                self.update_session_history(session_id, query, hit["answer"])
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"会话历史写入失败（不影响本次回答）: {e}")
        if event_sink is not None:
            event_sink.append({
                "type": "cache",
                "hit": True,
                "similarity": hit.get("similarity"),
                "matched_query": hit.get("query"),
                "backend": self.semantic_cache.backend,
            })
        self.logger.info(
            f"语义缓存命中（相似度 {hit.get('similarity')}），跳过检索与生成")
        return hit["answer"]

    def _cache_store(self, cache_query, answer, mode, route):
        """把答案写进语义缓存。

        只缓存有确定答案的链路：财务指标（sql）与年报原文（rag）。
        行情（实时数据）与 LLM 通用回答（含时效性/风险提示）一律不缓存。
        """
        if not self.semantic_cache or not answer:
            return
        if route not in ("sql", "rag"):
            return
        if self._looks_like_miss(answer):
            return
        try:
            self.semantic_cache.set(cache_query, answer, mode=mode, route=route)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"语义缓存写入失败（忽略）: {e}")

    def _rag_answer_with_fallback(self, route_query, source_filter, history,
                                  session_id, original_query, mode="auto"):
        """先流式跑 RAG；若开头就明确“未找到”，自动改用 LLM 通用回答兜底。"""
        gen = self.rag_system.generate_answer(
            route_query, source_filter=source_filter, history=history)
        buf = ""
        miss = False
        try:
            for token in gen:
                buf += token
                if len(buf) >= 200:
                    break
                if self._looks_like_miss(buf):
                    miss = True
                    break
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"RAG 流式生成失败: {e}")
            fallback = f"抱歉，生成回答时出错：{e}"
            yield fallback, False
            if session_id:
                self.update_session_history(session_id, original_query, fallback)
            yield "", True
            return

        if miss and getattr(self.config, "RAG_LLM_FALLBACK", True):
            self.logger.info(f"RAG 明确未找到，转 LLM 通用回答: '{route_query}'")
            prefix = ("> 年报知识库中没有检索到相关信息，以下为通用知识回答，"
                      "仅供参考、不构成投资建议：\n\n")
            text = prefix
            yield prefix, False
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            ) if history else ""
            prompt = self.general_prompt.format(history=history_context, question=route_query)
            try:
                for token in self.call_llm(prompt):
                    text += token
                    yield token, False
            except Exception as e:  # noqa: BLE001
                err = f"\n\n（通用回答失败：{type(e).__name__}）"
                text += err
                yield err, False
            if session_id and text:
                try:
                    self.update_session_history(session_id, original_query, text)
                except Exception as e:  # noqa: BLE001
                    self.logger.error(f"会话历史写入失败（不影响本次回答）: {e}")
            yield "", True
            return

        # 正常 RAG 流式：先吐已缓冲部分，再继续
        collected = buf
        if buf:
            yield buf, False
        try:
            for token in gen:
                collected += token
                yield token, False
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"RAG 流式输出中断: {e}")
            fallback = f"\n\n抱歉，生成回答时出错：{e}"
            collected += fallback
            yield fallback, False
        if session_id and collected:
            try:
                self.update_session_history(session_id, original_query, collected)
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"会话历史写入失败（不影响本次回答）: {e}")
        # 年报原文类答案进语义缓存（下一轮问同样的问题直接秒回）
        self._cache_store(route_query, collected, mode, "rag")
        yield "", True

    def _market_answer(self, query, history, session_id):
        """行情/估值类问题：用行情数据服务回答（实时行情 + 年报基本面）。"""
        self.logger.info("意图=market，调用行情数据服务")
        try:
            text = market_service.answer(query)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"行情服务失败，回退 LLM: {e}")
            yield from self._general_answer(query, history, session_id, reason="market_fallback")
            return
        if (not text) or text.startswith("未能识别公司") or "实时行情暂时获取失败" in text:
            self.logger.info("行情服务未命中，回退 LLM 通用回答")
            yield from self._general_answer(query, history, session_id, reason="market_fallback")
            return
        if session_id and text:
            try:
                self.update_session_history(session_id, query, text)
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"会话历史写入失败（不影响本次回答）: {e}")
        yield text, True

    def _general_answer(self, query, history, session_id, reason="general"):
        """通用知识 / 知识库外问题：直接调用 LLM 回答（不查库、不检索）。"""
        self.logger.info(f"意图={reason}，直接调用 LLM 通用回答")
        history_context = "\n".join(
            [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
        ) if history else ""
        prompt = self.general_prompt.format(history=history_context, question=query)
        collected = ""
        try:
            for token in self.call_llm(prompt):
                collected += token
                yield token, False
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"通用问答 LLM 调用失败: {e}")
            fallback = (f"抱歉，调用大模型失败（{type(e).__name__}）。"
                        "请稍后重试，或者换成与公司年报/财务指标相关的问题。")
            if session_id:
                self.update_session_history(session_id, query, fallback)
            yield fallback, True
            return
        if session_id and collected:
            try:
                self.update_session_history(session_id, query, collected)
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"会话历史写入失败（不影响本次回答）: {e}")
        yield "", True

    def _sql_lookup(self, question):
        """跑一次 NL2SQL，返回 (answer, table_markdown, safe_sql)；失败返回 (None, None, None)。"""
        if not (self.config.SQL_ENABLED and self.sql_qa is not None):
            return None, None, None
        try:
            res = self.sql_qa.ask(question)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"SQL 查询失败: {e}")
            return None, None, None
        if res.get("error"):
            self.logger.warning(f"SQL 未得到有效结果: {res['error']}")
            return None, None, None
        self.logger.info("SQL: %s | rows=%d | sql=%s",
                         question, len(res.get("rows") or []), res.get("safe_sql"))
        return res["answer"], res.get("table_markdown"), res.get("safe_sql")

    def _answer_by_mode(self, mode, query, route_query, source_filter, history, session_id):
        """显式功能模式（前端功能选择）：财务指标 / 年报原文 / 多公司对比 / 股东图谱。"""
        answer = None
        cache_key = route_query
        if mode == "sql":
            answer, _table, _sql = self._sql_lookup(route_query)
            if not answer:
                answer = ("当前模式是「财务指标」：该问题无法用财务指标表回答。"
                          "可切换到「智能问答」或「年报原文」再试。")
        elif mode == "compare":
            companies = detect_companies(route_query)
            question = route_query
            # 没点名公司时，默认做全行业 2025 年关键指标对比
            if not companies:
                question = "2025年20家白酒公司的营业收入、归母净利润、毛利率、ROE 从高到低对比"
            cache_key = question
            answer, table, _sql = self._sql_lookup(question)
            if not answer:
                answer = "多公司对比查询失败，请换个问法，例如：茅台、五粮液、汾酒2025年营收和净利对比。"
            elif table:
                answer = answer + "\n\n" + table
        elif mode == "rag":
            if source_filter is None:
                source_filter = detect_company(route_query)
            yield from self._rag_answer_with_fallback(
                route_query, source_filter, history, session_id, query, mode=mode)
            return
        elif mode == "market":
            yield from self._market_answer(query, history, session_id)
            return
        else:  # graph：Neo4j 图谱未实现，先用十大股东数据兜底
            if any(k in route_query for k in ("股东", "基金", "持股", "重仓", "持仓")):
                answer, _table, _sql = self._sql_lookup(route_query)
                if answer:
                    answer = "（股东图谱开发中，先展示十大股东数据）\n" + answer
            if not answer:
                answer = ("「股东图谱」功能开发中（Neo4j）。当前可查十大股东数据，"
                          "例如：招商中证白酒基金持有哪几家白酒公司？")

        if session_id and answer:
            try:
                self.update_session_history(session_id, query, answer)
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"会话历史写入失败（不影响本次回答）: {e}")
        if answer:
            self._cache_store(cache_key, answer, mode, "sql")
        yield answer, True

    # 修改 main.py 中的 query 方法
    def query(self, query, source_filter=None, session_id=None, mode="auto", event_sink=None):
        """查询集成系统，支持对话历史和流式输出。

        mode: auto(智能问答) / sql(财务指标) / rag(年报原文) / market(行情) /
              compare(多公司对比) / graph(股东图谱) / general(闲聊)
        event_sink: 可选 list，用来往外抛结构化事件（如「语义缓存命中」），
                    由 API 层转成 SSE 事件推给前端。
        """
        start_time = time.time()
        mode = (mode or "auto").strip().lower()
        if mode not in ("auto", "sql", "rag", "compare", "graph", "general", "market"):
            mode = "auto"
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id}, 模式: {mode})")
        history = self.get_session_history(session_id) if session_id else []

        # 0) 语义缓存：完整问题先查缓存，命中直接返回（跳过检索 + LLM 生成）
        cached = self._cache_lookup(query, mode, session_id, history, event_sink)
        if cached is not None:
            yield cached, True
            return

        # 1) 指定功能模式：跳过自动意图识别，直接走对应链路
        if mode != "auto":
            if mode == "general":
                yield from self._general_answer(query, history, session_id, reason="general")
                return
            route_query = self._rewrite_followup(query, history)
            if route_query != query:
                self.logger.info(f"追问改写: '{query}' -> '{route_query}'")
            yield from self._answer_by_mode(mode, query, route_query, source_filter, history, session_id)
            return

        # 2) 意图识别：通用知识/知识库外 → LLM 直答；专业咨询 → SQL/RAG
        intent = self._classify_intent(query)
        self.logger.info(f"意图识别: {intent} (查询: '{query}')")
        if intent == "market":
            yield from self._market_answer(query, history, session_id)
            return
        if intent in ("general", "unsupported"):
            yield from self._general_answer(query, history, session_id, reason=intent)
            return

        # 3) 专业咨询：追问改写补全省略信息，再路由到 SQL / RAG
        route_query = self._rewrite_followup(query, history)
        if route_query != query:
            self.logger.info(f"追问改写: '{query}' -> '{route_query}'")

        # 意图路由（P2）：指标/关系类问题优先走 SQL，文档解释类走 RAG
        route = route_question(route_query, graph_enabled=self.config.GRAPH_ENABLED)
        if route == "sql" and self.config.SQL_ENABLED and self.sql_qa is not None:
            try:
                self.logger.info(f"路由=sql，执行 NL2SQL: {route_query}")
                sql_result = self.sql_qa.ask(route_query)
                if sql_result.get("error"):
                    self.logger.warning(f"SQL QA 未得到有效结果，回退 RAG: {sql_result['error']}")
                else:
                    # 前端只要“关键答案”：只返回自然语言结论，表格/SQL 只写日志，不塞进回答
                    answer_text = sql_result["answer"]
                    self.logger.info(
                        "SQL: %s | rows=%d | sql=%s",
                        query, len(sql_result.get("rows") or []), sql_result.get("safe_sql"))
                    if session_id:
                        self.update_session_history(session_id, query, answer_text)
                    self._cache_store(route_query, answer_text, mode, "sql")
                    processing_time = time.time() - start_time
                    self.logger.info(f"SQL 问答完成，耗时 {processing_time:.2f}秒")
                    yield answer_text, True
                    return
            except Exception as e:
                self.logger.warning(f"SQL QA 失败，回退 RAG: {e}")
        elif route == "graph":
            self.logger.info("路由=graph（P3 尚未实现，先回退 RAG）")

        # 问题里只提到一家公司时，自动把该公司作为 RAG 的 source_filter，
        # 避免跨公司检索把别家年报的段落混进上下文
        if source_filter is None:
            detected = detect_company(route_query)
            if detected:
                source_filter = detected
                self.logger.info(f"识别到公司实体，RAG 限定 source={source_filter}")

        # 首先尝试 MySQL/BM25 快路径（金融项目暂无 FAQ 表时自动跳过）
        answer, need_rag = None, True
        if self.bm25_search is not None:
            try:
                answer, need_rag = self.bm25_search.search(route_query, threshold=0.85)
            except Exception as e:
                self.logger.warning(f"BM25 快路径失败，回退 RAG: {e}")
                answer, need_rag = None, True

        if answer:
            # MySQL找到精确答案，直接返回
            self.logger.info(f"MySQL答案: {answer}")
            if session_id:
                self.update_session_history(session_id, query, answer)  # 把BM25的QA更新到会话历史
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 对于MySQL答案，一次性返回
            yield answer, True  # 第二个参数表示这是完整答案
        elif need_rag:
            # 需要使用RAG系统；若明确“未找到”会自动转 LLM 通用回答兜底
            self.logger.info("无可靠MySQL答案，回退到RAG")
            yield from self._rag_answer_with_fallback(
                route_query, source_filter, history, session_id, query, mode=mode)
        else:
            # 未找到答案
            self.logger.info("未找到答案")
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "未找到答案", True  # 一次性返回


def _to_plain_text(markdown_text):
    """CLI 展示用：把 LLM 返回的 Markdown 转成终端里好读的纯文本。

    Web 前端由 marked.js 渲染 Markdown；命令行没有渲染器，直接打印会出现
    `**加粗**`、`| 表格 |`、`- 列表` 这类原始符号，所以这里做一层轻量转换。
    """
    import re

    lines = []
    for line in (markdown_text or "").splitlines():
        stripped = line.strip()
        # 表格分隔行（|---|---|）直接丢掉
        if "-" in stripped and re.fullmatch(r"\|?[\s:|-]+\|?", stripped):
            continue
        # 表格行：| a | b | -> a  b
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            line = "  ".join(cells)
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)      # 标题
        line = re.sub(r"^\s*[-*+]\s+", "· ", line)          # 无序列表
        line = re.sub(r"^\s*>\s?", "  ", line)              # 引用
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)        # 加粗
        line = re.sub(r"`([^`]*)`", r"\1", line)            # 行内代码
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\n欢迎使用集成问答系统！")
    print(f"会话ID: {session_id}")
    print(f"支持的公司：{qa_system.config.VALID_SOURCES}")
    print(f"当前模型：{qa_system.config.LLM_MODEL}（{qa_system.config.LLM_BASE_URL}）")
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        while True:
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见！")
                break
            source_filter = input(
                f"请输入公司名称 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                source_filter = None

            print("\n答案: ", end="", flush=True)  # 开始打印答案
            answer = ""  # 用于累积完整答案
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:  # 累积答案（CLI 下等生成完再统一转纯文本打印）
                    answer += token
                if is_complete:  # 如果是完整答案或流结束，退出循环
                    break
            print(_to_plain_text(answer))

            # 打印对话历史（同样去掉 Markdown 记号）
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. 问: {entry['question']}\n   答: {_to_plain_text(entry['answer'])}")
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        qa_system.mysql_client.close()


if __name__ == "__main__":
    main()
