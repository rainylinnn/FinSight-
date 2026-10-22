# -*- coding: utf-8 -*-
"""FinSight 配置中心：读取项目根目录 config.ini，暴露配置属性。

与旧项目 Itcast_qa_system 的 Config 保持同名属性，方便复用 base/mysql_qa/rag_qa。
新增：SQL / 图谱 / 金融数据目录 / 模型目录回退逻辑。
"""
import ast
import configparser
import os

# 项目根目录（base/ 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 旧项目路径：模型权重太大不复制，找不到新模型目录时回退到这里
LEGACY_PROJECT = r"C:\Users\黄瑞霖\PyCharmMiscProject\venv\project\Itcast_qa_system"

# .env 文件：存在就加载（不覆盖已存在的系统环境变量）
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")


def load_env_file(path=ENV_FILE, override=False):
    """加载 .env 到 os.environ。

    之前 README 写着「支持 .env」，但代码里从没调用过 load_dotenv，
    实际只认系统环境变量。现在真正接上：
      - 优先读系统环境变量（override=False，不会被 .env 里的旧值盖掉）
      - 没装 python-dotenv 时退化为自带的极简解析器（KEY=VALUE / # 注释 / 可选引号）
    """
    if not os.path.isfile(path):
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=override)
        return True
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (override or not os.getenv(key)):
                    os.environ[key] = value
        return True
    except OSError:
        return False


# 加载 .env（必须在读环境变量之前）
load_env_file()

API_KEY_ENV_NAMES = ("DEEPSEEK_API_KEY", "DEEPSEEK_API", "QIANWEN_API_KEY", "DASHSCOPE_API_KEY", "API_KEY")


def resolve_api_key_with_source(fallback=None):
    """按环境变量优先级解析 LLM API Key（DeepSeek 优先，兼容旧的通义/DashScope）。"""
    for name in API_KEY_ENV_NAMES:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), name
    if fallback and str(fallback).strip():
        return str(fallback).strip(), "config.ini"
    return None, None


def resolve_api_key(fallback=None):
    return resolve_api_key_with_source(fallback)[0]


class Config:
    def __init__(self, config_file=None):
        self.project_root = PROJECT_ROOT
        if config_file is None:
            config_file = os.path.join(PROJECT_ROOT, "config.ini")
        self.config_file = config_file

        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding="utf-8")

        # ---------- MySQL ----------
        self.MYSQL_HOST = self.config.get("mysql", "host", fallback="localhost")
        self.MYSQL_USER = self.config.get("mysql", "user", fallback="root")
        # 密码：优先读环境变量 MYSQL_PASSWORD，配置文件里留空（仓库不含明文密码）
        self.MYSQL_PASSWORD = (
            os.getenv("MYSQL_PASSWORD")
            or self.config.get("mysql", "password", fallback="")
        ).strip()
        self.MYSQL_DATABASE = self.config.get("mysql", "database", fallback="finance_qa")

        # ---------- Redis ----------
        self.REDIS_HOST = self.config.get("redis", "host", fallback="localhost")
        self.REDIS_PORT = self.config.getint("redis", "port", fallback=6379)
        # 密码：优先读环境变量 REDIS_PASSWORD；留空表示不需要密码
        self.REDIS_PASSWORD = (
            os.getenv("REDIS_PASSWORD")
            or self.config.get("redis", "password", fallback="")
        ).strip()
        self.REDIS_DB = self.config.getint("redis", "db", fallback=0)

        # ---------- Milvus ----------
        self.MILVUS_HOST = self.config.get("milvus", "host", fallback="localhost")
        self.MILVUS_PORT = self.config.get("milvus", "port", fallback="19530")
        self.MILVUS_DATABASE_NAME = self.config.get("milvus", "database_name", fallback="finance")
        self.MILVUS_COLLECTION_NAME = self.config.get(
            "milvus", "collection_name", fallback="baijiu_annual_report_1006")

        # ---------- LLM ----------
        # 当前默认走 DeepSeek：DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL 由系统环境变量提供
        self.LLM_MODEL = self.config.get("llm", "model", fallback="deepseek-flash")
        self.DASHSCOPE_API_KEY, self.DASHSCOPE_API_KEY_SOURCE = resolve_api_key_with_source(
            self.config.get("llm", "api_key", fallback=None)
            or self.config.get("llm", "dashscope_api_key", fallback=None)
        )
        # 新命名：LLM_API_KEY / LLM_BASE_URL（优先读环境变量 DEEPSEEK_BASE_URL）
        self.LLM_API_KEY = self.DASHSCOPE_API_KEY
        self.LLM_API_KEY_SOURCE = self.DASHSCOPE_API_KEY_SOURCE
        self.LLM_BASE_URL = (
            os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or self.config.get("llm", "base_url", fallback=None)
            or self.config.get(
                "llm", "dashscope_base_url",
                fallback="https://api.deepseek.com")
        )
        # 兼容旧命名（复用模块里仍用 DASHSCOPE_* 的地方自动跟随 DeepSeek 配置）
        self.DASHSCOPE_BASE_URL = self.LLM_BASE_URL

        # ---------- 检索参数 ----------
        self.PARENT_CHUNK_SIZE = self.config.getint("retrieval", "parent_chunk_size", fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint("retrieval", "child_chunk_size", fallback=300)
        self.CHUNK_OVERLAP = self.config.getint("retrieval", "chunk_overlap", fallback=50)
        self.RETRIEVAL_K = self.config.getint("retrieval", "retrieval_k", fallback=5)
        self.CANDIDATE_M = self.config.getint("retrieval", "candidate_m", fallback=5)
        self.USE_RERANKER = self.config.getboolean("retrieval", "use_reranker", fallback=True)
        # 金融领域项目默认强制检索：跳过 BERT 意图分类的“通用知识”分支，
        # 避免文档类问题绕过 RAG 直接问 LLM（会编造/拒答）
        self.ALWAYS_RETRIEVE = self.config.getboolean("retrieval", "always_retrieve", fallback=True)
        # FAQ 快路径（Redis 题库 + BM25）：金融项目没有 FAQ 表，默认关闭
        self.ENABLE_FAST_PATH = self.config.getboolean("retrieval", "enable_fast_path", fallback=False)
        # BM25 关键词召回与向量召回 RRF 融合：补“经营风险/渠道库存”等关键词问题的召回
        self.KEYWORD_RETRIEVAL = self.config.getboolean("retrieval", "keyword_retrieval", fallback=True)
        # RAG 明确“未找到”时是否转 LLM 通用回答兜底
        self.RAG_LLM_FALLBACK = self.config.getboolean("retrieval", "llm_fallback_on_miss", fallback=True)

        # ---------- SQL（NL2SQL） ----------
        self.SQL_ENABLED = self.config.getboolean("sql", "enabled", fallback=True)
        self.SQL_MAX_ROWS = self.config.getint("sql", "max_rows", fallback=200)

        # ---------- 语义缓存（BGE-M3 向量缓存） ----------
        self.SEMANTIC_ENABLED = self.config.getboolean("cache", "semantic_enabled", fallback=True)
        self.SEMANTIC_THRESHOLD = self.config.getfloat("cache", "semantic_threshold", fallback=0.85)
        self.SEMANTIC_MIN_JACCARD = self.config.getfloat("cache", "semantic_min_jaccard", fallback=0.2)
        self.SEMANTIC_TTL = self.config.getint("cache", "semantic_ttl", fallback=7200)
        self.SEMANTIC_MAX_ENTRIES = self.config.getint("cache", "semantic_max_entries", fallback=500)
        self.SEMANTIC_CANDIDATE_LIMIT = self.config.getint("cache", "semantic_candidate_limit", fallback=200)

        # ---------- 图谱（Neo4j，可选） ----------
        self.GRAPH_ENABLED = self.config.getboolean("graph", "enabled", fallback=False)
        self.GRAPH_BACKEND = self.config.get("graph", "backend", fallback="neo4j")
        self.NEO4J_URI = self.config.get("graph", "neo4j_uri", fallback="bolt://localhost:7687")
        self.NEO4J_USER = self.config.get("graph", "neo4j_user", fallback="neo4j")
        self.NEO4J_PASSWORD = (
            os.getenv("NEO4J_PASSWORD")
            or self.config.get("graph", "neo4j_password", fallback="")
        ).strip()
        self.NEO4J_DATABASE = self.config.get("graph", "neo4j_database", fallback="neo4j")

        # ---------- 金融数据目录 ----------
        self.DATA_DIR = self._path("finance", "data_dir", "data")
        self.ANNUAL_REPORTS_DIR = self._path("finance", "annual_reports_dir", "data/raw/annual_reports")
        self.PROCESSED_DIR = self._path("finance", "processed_dir", "data/processed")
        self.COMPANIES_FILE = self._path("finance", "companies_file", "data/companies.csv")
        self.FINANCIAL_CSV = self._path("finance", "financial_csv",
                                        "data/structured/financial_indicators.csv")
        self.HOLDERS_CSV = self._path("finance", "holders_csv", "data/structured/holders.csv")
        self.INDEX_CSV = self._path("finance", "index_csv", "data/index/annual_report_index.csv")

        # ---------- 应用 ----------
        raw_sources = self.config.get("app", "valid_sources", fallback='["贵州茅台"]')
        try:
            self.VALID_SOURCES = ast.literal_eval(raw_sources)
            if not isinstance(self.VALID_SOURCES, list):
                raise ValueError
        except Exception:
            self.VALID_SOURCES = [s.strip() for s in raw_sources.split(",") if s.strip()]
        self.CUSTOMER_SERVICE_PHONE = self.config.get("app", "customer_service_phone", fallback="")
        self.LOG_FILE = self.config.get("logger", "log_file", fallback="logs/app.log")
        if not os.path.isabs(self.LOG_FILE):
            self.LOG_FILE = os.path.join(PROJECT_ROOT, self.LOG_FILE)

        # ---------- 模型目录 ----------
        # 优先 FIN_MODEL_ROOT 环境变量 / 项目下 models；都没有时回退到旧项目模型目录
        self.MODEL_ROOT = os.getenv("FIN_MODEL_ROOT") or os.path.join(PROJECT_ROOT, "models")
        legacy_models = os.path.join(LEGACY_PROJECT, "rag_qa", "models")
        if not os.path.isdir(os.path.join(self.MODEL_ROOT, "bge-m3")) and os.path.isdir(
                os.path.join(legacy_models, "bge-m3")):
            self.MODEL_ROOT = legacy_models
        self.nlp_bert_doc_seg = os.path.join(
            self.MODEL_ROOT, "nlp_bert_document-segmentation_chinese-base")

        # BERT 意图分类模型：新项目里没有就回退旧项目训练产物
        self.bert_intent_cls = os.path.join(PROJECT_ROOT, "rag_qa", "core", "bert_query_classifier")
        if not os.path.isdir(self.bert_intent_cls):
            self.bert_intent_cls = os.path.join(
                LEGACY_PROJECT, "rag_qa", "core", "bert_query_classifier")

    def _path(self, section, key, fallback):
        """读取相对项目根目录的路径并转成绝对路径。"""
        value = self.config.get(section, key, fallback=fallback)
        return value if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value)


if __name__ == "__main__":
    conf = Config()
    print("config:", conf.config_file)
    print("mysql:", conf.MYSQL_HOST, conf.MYSQL_DATABASE)
    print("milvus:", conf.MILVUS_DATABASE_NAME, conf.MILVUS_COLLECTION_NAME)
    print("sources:", conf.VALID_SOURCES)
    print("models:", conf.MODEL_ROOT)
    print("data:", conf.DATA_DIR)
