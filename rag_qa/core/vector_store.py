# -*- coding:utf-8 -*-
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
# from langchain.docstore.document import Document
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import pickle
import sys, os

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

# from base import logger, Config
from rag_qa.core.document_processor import *

conf = Config()


# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        # 检查CUDA是否可用；显存不足 4GB 时强制 CPU，避免 BGE-M3 + reranker 同时加载 OOM
        env_device = os.getenv("FIN_DEVICE", "").strip()
        if env_device in ("cpu", "cuda"):
            self.device = env_device
        elif torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            self.device = 'cuda' if total_mem >= 4 * 1024 ** 3 else 'cpu'
        else:
            self.device = 'cpu'
        # 日志提醒使用的是什么设备
        self.logger.info(f"使用设置：{self.device}")
        # 是否启用重排：CPU 机器上 CrossEncoder 很慢，可在 config.ini [retrieval] 关闭
        self.use_reranker = conf.USE_RERANKER
        if self.use_reranker:
            # 初始化 BGE-Reranker 模型，用于重排序检索结果（模型目录优先 config.MODEL_ROOT）
            reranker_path = os.path.join(conf.MODEL_ROOT, 'bge-reranker-v2-m3')
            # print(f'reranker_path--》{reranker_path}')
            self.reranker = CrossEncoder(reranker_path, device=self.device)
        else:
            self.reranker = None
            self.logger.info("已关闭 reranker（config.ini [retrieval] use_reranker=false）")
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        m3_path = os.path.join(conf.MODEL_ROOT, 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(model_name_or_path=m3_path, use_fp16=(self.device == 'cuda'),
                                                         device=self.device)
        # 获取稠密向量的维度# 1024
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 关键词召回（BM25）索引，首次使用时懒加载
        self._bm25 = None
        self._bm25_corpus = None
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    # 类私有化方法
    def _create_or_load_collection(self):
        # 检查指定集合是否已经存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
            # ===== 金融投研扩展字段（公司/年份/章节/页码，用于 metadata 过滤与引用溯源）=====
            schema.add_field(field_name="company_code", datatype=DataType.VARCHAR, max_length=20)
            schema.add_field(field_name="company_name", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="year", datatype=DataType.INT32)
            schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="section_no", datatype=DataType.INT32)
            schema.add_field(field_name="page", datatype=DataType.INT32)
            schema.add_field(field_name="doc_title", datatype=DataType.VARCHAR, max_length=300)
            schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=500)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # 定义方法，向向量存储添加文档
    def add_documents(self, documents):
        # print(f'documents--》{documents[0]}')
        # 提取所有文档的内容列表
        texts = [doc.page_content for doc in documents]

        # 使用 BGE-M3 嵌入函数生成文档的嵌入
        embeddings = self.embedding_function(texts)
        # print(f'embeddings--》{embeddings}')
        # print(f'embeddings--》{embeddings.keys()}')
        # 初始化空列表，存储插入的数据
        data = []
        # 遍历每个文档，带上索引i
        for i, doc in enumerate(documents):
            # 生成文档内容的哈希值作为唯一的ID
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # print(f'text_hash--》{text_hash}')
            # print(f'text_hash--》{type(text_hash)}')
            # 初始化一个稀疏向量的字典（Milvus要求存储稀疏向量的格式）
            sparse_vector = {}

            try:
                # 新版本 milvus-model 使用 coo_array 格式
                row = embeddings["sparse"][i]
                if hasattr(row, 'col'):  # coo_array 格式
                    indices = row.col
                    values = row.data
                else:  # csr_matrix 格式
                    indices = row.indices
                    values = row.data
            except Exception as e:
                # 兼容旧版本 milvus-model
                row = embeddings["sparse"].getrow(i)
                indices = row.indices
                values = row.data

            # 获取第i行对应的稀疏向量数据[0.4, 0.2, 0, 0, 0.1]
            # row = embeddings["sparse"].getrow(i)
            # row = embeddings["sparse"][i]  # 新版本milvus-model，支持这种获取稀疏向量的形式
            # indices = row.row
            # print(f'row--》{row}')
            # print(f'row--》{row.shape}')
            # 获取稀疏向量的非零值的索引
            # indices = row.col
            # print(f'indics--》{indics}')
            # 获取稀疏向量的非零值
            # values = row.data

            # 将索引和值进行配对，存储到字典中
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
            # print(f'sparse_vector--》{sparse_vector}')
            # print(f'sparse_vector--》{len(sparse_vector)}')
            # print(embeddings["dense"][i])
            # print(embeddings["dense"][i].shape)
            # 创建数据字典，包含所有字段
            data.append({
                "id": doc.metadata.get("id") or text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata.get("parent_id", ""),
                "parent_content": doc.metadata.get("parent_content", ""),
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown"),
                "company_code": doc.metadata.get("company_code", ""),
                "company_name": doc.metadata.get("company_name", ""),
                "year": int(doc.metadata.get("year") or 0),
                "section": doc.metadata.get("section", ""),
                "section_no": int(doc.metadata.get("section_no") or 0),
                "page": int(doc.metadata.get("page") or 0),
                "doc_title": doc.metadata.get("doc_title", ""),
                "file_path": doc.metadata.get("file_path", ""),
            })
        # 检查是否有数据需要插入
        if data:
            # 使用 upsert 操作插入数据，覆盖重复 ID
            self.client.upsert(collection_name=self.collection_name, data=data)
            # 记录插入或更新的文档数量日志
            logger.info(f"已插入或更新 {len(data)} 个文档")

    def count(self):
        """集合中实体数量（用于入库进度检查）。"""
        try:
            result = self.client.query(collection_name=self.collection_name, filter="",
                                       output_fields=["count(*)"])
            return int(result[0]["count(*)"]) if result else 0
        except Exception as e:
            logger.warning(f"统计集合数量失败: {e}")
            return -1

    # ---------- 关键词召回（BM25） ----------
    # 查询里的“问句/文档结构”词，会稀释 BM25 对真正主题词的权重，先过滤掉
    KEYWORD_STOPWORDS = {
        "的", "了", "中", "，", "？", "。", "哪些", "什么", "怎么", "如何", "提到", "提到哪些",
        "描述", "介绍", "说明", "情况", "多少", "是", "与", "和", "及", "请", "问", "一下",
        "年报", "年度报告", "报告", "管理层", "讨论", "分析", "具体", "主要", "相关", "内容",
        "列举", "概括", "总结", "吗", "呢", "在", "里", "中，", "以及", "他们", "公司", "报告中",
    }

    def _ensure_keyword_index(self):
        """从 Milvus 拉取全部 chunk 建 BM25 索引（只建一次）。

        稠密/稀疏向量对“经营风险”“渠道库存”这类关键词问题召回不稳定，
        用 BM25 做一路关键词召回再融合，能显著提升原文段落命中率。
        """
        if self._bm25 is not None:
            return
        import jieba
        from rank_bm25 import BM25Okapi

        # 用 query_iterator 全量拉取（Milvus 单次 query 最多 16384 条，数据扩容后会截断）
        fields = ["text", "parent_id", "parent_content", "source", "company_code",
                  "company_name", "year", "section", "section_no", "page", "doc_title"]
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=2000,
            filter="",
            output_fields=fields,
        )
        rows = []
        while True:
            batch = iterator.next()
            if not batch:
                iterator.close()
                break
            rows.extend(batch)
        self._bm25_corpus = rows

        # 分词结果落盘缓存：2.4 万 chunk 首次分词要 ~100s，之后直接加载
        cache_dir = getattr(conf, "PROCESSED_DIR", None) or os.path.join(project_root, "data", "processed")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            sample = hashlib.md5(
                "".join((r.get("text") or "")[:20] for r in rows[:50]).encode("utf-8")
            ).hexdigest()[:8]
            cache_path = os.path.join(
                cache_dir, f"bm25_tokens_{self.collection_name}_{len(rows)}_{sample}.pkl")
        except Exception:
            cache_path = None

        tokenized = None
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    tokenized = pickle.load(fh)
                logger.info(f"BM25 分词缓存命中：{cache_path}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"BM25 缓存读取失败，重新分词: {e}")

        if tokenized is None:
            # 用搜索引擎模式分词：把“经营风险”拆成“经营/风险”，和正文里的“风险”能对上
            tokenized = [list(jieba.cut_for_search(r.get("text") or "")) for r in rows]
            if cache_path:
                try:
                    with open(cache_path, "wb") as fh:
                        pickle.dump(tokenized, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(f"BM25 分词缓存已保存：{cache_path}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"BM25 缓存写入失败: {e}")

        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 关键词索引构建完成：{len(rows)} 个 chunk")

    def keyword_search(self, query, k=None, source_filter=None, year_filter=None,
                       section_filter=None):
        """BM25 关键词召回，返回父文档级 Document 列表。"""
        k = k or conf.RETRIEVAL_K
        self._ensure_keyword_index()
        import jieba
        import numpy as np

        raw_tokens = list(jieba.cut_for_search(query or ""))
        tokens = [t for t in raw_tokens
                  if len(t.strip()) >= 2
                  and t not in self.KEYWORD_STOPWORDS
                  and not t.isdigit()  # 年份/数字会命中大量“2025 年”模板句，稀释权重
                  and not (source_filter and t in str(source_filter))]  # 公司名已由字段过滤
        # 去掉问句词后如果太短（比如只剩一个词），退回全量分词
        if len(tokens) < 2:
            tokens = raw_tokens
        scores = self._bm25.get_scores(tokens)
        docs = []
        for idx in np.argsort(scores)[::-1]:
            row = self._bm25_corpus[int(idx)]
            if scores[idx] <= 0:
                break
            if source_filter and row.get("source") != source_filter:
                continue
            if year_filter and row.get("year") != year_filter:
                continue
            if section_filter and row.get("section") != section_filter:
                continue
            doc = self._doc_from_hit(row)
            doc = Document(page_content=self._pick_content(doc), metadata=doc.metadata)
            docs.append(doc)
            if len(docs) >= k:
                break
        return docs

    @staticmethod
    def _merge_rrf(dense_docs, keyword_docs, k=60, dense_weight=1.0, keyword_weight=1.5):
        """Reciprocal Rank Fusion：把两路召回结果按排名融合。

        关键词一路权重更高：向量检索在“经营风险/渠道库存”这类专有表达上召回不稳定，
        BM25 命中时通常更准，给 1.5 倍权重可把原文段落顶进上下文。
        """
        scores = {}
        by_key = {}
        for docs, weight in ((dense_docs, dense_weight), (keyword_docs, keyword_weight)):
            for rank, doc in enumerate(docs, 1):
                key = doc.metadata.get("parent_id") or (doc.page_content or "")[:80]
                scores[key] = scores.get(key, 0.0) + weight / (k + rank)
                by_key.setdefault(key, doc)
        ordered = sorted(scores.items(), key=lambda x: -x[1])
        return [by_key[key] for key, _ in ordered]

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None,
                                  year_filter=None, section_filter=None, section_no_filter=None):
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([query])
        # 获取查询的稠密向量
        dense_query_vector = query_embeddings["dense"][0]
        # print(f'dense_query_vector--》{dense_query_vector.shape}')
        # 初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # ====================
        # 获取查询稀疏向量的第 0 行数据
        # row = query_embeddings["sparse"][0]
        # # 获取稀疏向量的非零值索引
        # indices = row.indices
        # # 获取稀疏向量的非零值
        # values = row.data
        # ====================
        try:
            # 新版本 milvus-model 使用 coo_array 格式
            row = query_embeddings["sparse"][0]
            if hasattr(row, 'col'):  # coo_array 格式
                indices = row.col
                values = row.data
            else:  # csr_matrix 格式
                indices = row.indices
                values = row.data
        except Exception as e:
            # 兼容旧版本 milvus-model
            row = query_embeddings["sparse"].getrow(0)
            indices = row.indices
            values = row.data

        # 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value
        # print(f'sparse_query_vector-->{sparse_query_vector}')
        # 初始化过滤表达式，默认不过滤
        # 组合过滤条件：公司/年份/章节/章节序号（对应 Milvus schema 的显式字段）
        expr_parts = []
        if source_filter:
            expr_parts.append("source == '%s'" % str(source_filter).replace("'", "\\'"))
        if year_filter:
            expr_parts.append("year == %d" % int(year_filter))
        if section_filter:
            expr_parts.append("section == '%s'" % str(section_filter).replace("'", "\\'"))
        if section_no_filter:
            expr_parts.append("section_no == %d" % int(section_no_filter))
        filter_expr = " and ".join(expr_parts)
        # print(f'filter_expr--》{filter_expr}')
        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        # 创建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.3，稠密向量权重 0.7
        ranker = WeightedRanker(0.7, 0.3)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp",
                           "company_code", "company_name", "year", "section", "section_no",
                           "page", "doc_title", "file_path"]
        )[0]
        # print(f'results--》{results}')
        # print(f'results--》{type(results)}')
        # print(f'results--》{len(results)}')
        # 将上述搜索到的结果进行Document对象封装，便于查询使用
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        # print(f'sub_chunks--》{len(sub_chunks)}')
        # 从子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # print(f'parent_docs--》{parent_docs}')
        # print(f'parent_docs--》{len(parent_docs)}')

        # 关键词召回融合（BM25）：解决“经营风险/渠道库存”等关键词问题的召回盲区
        if getattr(conf, "KEYWORD_RETRIEVAL", True):
            try:
                keyword_docs = self.keyword_search(
                    query, k=conf.CANDIDATE_M, source_filter=source_filter,
                    year_filter=year_filter, section_filter=section_filter,
                )
                if keyword_docs:
                    parent_docs = self._merge_rrf(parent_docs, keyword_docs)
                    logger.info(
                        f"BM25 关键词召回 {len(keyword_docs)} 个父文档，RRF 融合后 {len(parent_docs)} 个")
            except Exception as e:
                logger.warning(f"BM25 关键词召回失败（忽略，仅用向量结果）: {e}")

        # # 如果只有1个文档或者没有，直接返回跳过重排序
        if len(parent_docs) < 2 or not self.use_reranker:
            return parent_docs[:conf.CANDIDATE_M]
            # 如果有父文档，进行重排序
        if parent_docs:
            # 创建查询与文档内容的配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用 BGE-Reranker 计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # print(f'scores--》{scores}')
            # 根据得分从高到低排序文档
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        # 如果没有父文档，返回空列表
        # 如果没有父文档，返回空列表
        else:
            ranked_parent_docs = []

        # 返回前 m 个重排序后的文档
        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取父块内容（并校验父子是否对得上）
            content = self._pick_content(chunk)
            # 检查内容是否非空且未重复
            if content and content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=content, metadata=chunk.metadata))
                # 将内容添加到去重集合
                parent_contents.add(content)
            # 返回去重后的父文档列表
        return unique_docs

    @staticmethod
    def _normalize_for_match(text):
        """去掉空白/换行，用于判断子块文本是否真的在父块里。"""
        return "".join((text or "").split())

    def _pick_content(self, chunk):
        """选择真正要放进上下文的文本。

        年报解析时“一页拆多节”的页可能让父窗口落在相邻章节上（父子内容对不上），
        这时父块文本是错的，直接用子块文本，避免上下文出现无关章节。
        """
        parent_content = chunk.metadata.get("parent_content") or ""
        child_text = chunk.metadata.get("child_text") or chunk.page_content or ""
        if not parent_content:
            return child_text
        if not child_text:
            return parent_content
        needle = self._normalize_for_match(child_text)[:40]
        if needle and needle in self._normalize_for_match(parent_content):
            return parent_content
        # 父块对不上时用子块文本，并把去重键改成子块级，避免和同页的“错误父块”合并后丢失内容
        parent_id = chunk.metadata.get("parent_id") or ""
        chunk.metadata["parent_id"] = (
            f"{parent_id}#{hashlib.md5(child_text.encode('utf-8')).hexdigest()[:8]}")
        return child_text

    # 定义类似私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "child_text": hit.get("text"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp"),
                "company_code": hit.get("company_code"),
                "company_name": hit.get("company_name"),
                "year": hit.get("year"),
                "section": hit.get("section"),
                "section_no": hit.get("section_no"),
                "page": hit.get("page"),
                "doc_title": hit.get("doc_title"),
                "file_path": hit.get("file_path"),
            }
        )


if __name__ == "__main__":
    vector_store = VectorStore()  # 建表
    directory_path = r'C:\Users\黄瑞霖\PyCharmMiscProject\venv\project\Itcast_qa_system\rag_qa\data\ai_data'
    print(f"embedding_function.dim--》{vector_store.embedding_function.dim}")
    documents = process_documents(directory_path)  # 文档读取与切块
    vector_store.add_documents(documents)  # 子块编码并且写入Milvus

    # query = "AI学科学费是多少？"
    # results = vector_store.hybrid_search_with_rerank(query, source_filter='ai')
    # print(f'results-->{results}')
    # print(f'results-->{len(results)}')
