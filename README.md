# FinSight 酒研助手 · 白酒行业智能投研助手

[![CI](https://github.com/rainylinnn/FinSight-/actions/workflows/ci.yml/badge.svg)](https://github.com/rainylinnn/FinSight-/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![RAG](https://img.shields.io/badge/RAG-BGE--M3%20%2B%20RRF%20%2B%20RAGAS-2f5d9e)](docs/语义缓存-设计与实测.md)

> 面向券商/基金研究员的白酒行业 RAG 投研工作台：**财报问答（带页码引用）+ 财务指标 NL2SQL + 股东关系图谱（可选）+ Agent 工具路由**。
> 工程结构复用课程项目 Itcast_qa_system（FastAPI + MySQL + Redis + Milvus + BGE-M3 + bge-reranker + DeepSeek）。

## 0. 快速开始（从 GitHub 克隆）

```bash
git clone https://github.com/rainylinnn/FinSight-.git
cd FinSight-

# 1) 装依赖（uv 管理，不需要 pip）
uv sync

# 2) 配环境变量：复制模板并填写
cp .env.example .env        # Windows: copy .env.example .env
#    至少要填 DEEPSEEK_API_KEY 和 MYSQL_PASSWORD

# 3) 跑起来
./start.sh                  # Windows 双击 start.bat
#    → http://127.0.0.1:8003
```

需要本机已装好 MySQL / Redis / Milvus 三件套（见第 2 节）。
**Redis 不是必需品**：语义缓存在 Redis 不可用时会自动降级为进程内内存模式，不影响问答。

> 仓库里**不含**年报 PDF（版权原因，169 MB）与训练好的 BERT 权重（390 MB，超过 GitHub 单文件上限）。
> 两者都可以按下面的说明在本机重新生成，其余代码克隆下来即可运行。

## 1. 数据

| 数据 | 位置 | 规模 |
|---|---|---|
| 年报 PDF | `data/raw/annual_reports/<代码>_<公司>/` | 20 家公司 × 2023–2025 = 60 份 / 11,001 页 |
| 财务指标 | `data/structured/financial_indicators.csv` | 30 行（13 项指标） |
| 十大股东 | `data/structured/holders.csv` | 100 行 |
| 年报索引 | `data/index/annual_report_index.csv` | 30 行（含 PDF URL / 公告日期） |
| 解析产物 | `data/processed/` | children.jsonl / parents.jsonl / documents.jsonl |

数据来源：巨潮资讯网（年报 PDF）、东方财富（财务指标/十大股东），仅用于学习研究，不再分发。

## 2. 依赖服务（不使用 Docker）

项目代码不包含任何 Docker 文件，直接连接本机已有服务：

| 服务 | 默认地址 | 用途 |
|---|---|---|
| MySQL | `localhost:3306`，库 `finance_qa` | 用户/会话/财务指标 |
| Redis | `localhost:6379` | 语义缓存（**可选**，连不上自动降级为内存模式） |
| Milvus | `localhost:19530`，库 `finance` | 年报向量库 |
| Neo4j（可选） | `localhost:7687` | 股东关系图；未安装时 `config.ini` 里 `[graph] enabled = false` |

> 说明：这些服务可以复用你旧项目里已经在跑的实例；本项目只负责连接，不负责编排。
> 如果某个服务没装（比如 Neo4j），保持 `[graph] enabled = false` 即可，核心问答功能不受影响。
> 密码统一写在 `.env`（`MYSQL_PASSWORD` / `REDIS_PASSWORD` / `NEO4J_PASSWORD`），仓库里不含明文。

## 3. 环境（uv 管理，Python 3.12）

本项目用 **uv** 管理依赖（和旧项目 Itcast_qa_system 一致）：

- `pyproject.toml`：依赖声明（由 `requirements.txt` 迁移，261 个包，Python 3.12）
- `uv.lock`：锁定的精确版本
- `.venv`：由 `uv venv` 创建 —— **不含 pip**，用 `uv add/uv sync/uv run` 管理

```bash
# 首次安装 / 同步依赖（读 uv.lock）
uv sync

# 需要重建环境时（uv 会挑本机 Python 3.12）
uv venv --python 3.12
uv sync

# 运行脚本（自动用 .venv，不需要 activate，也不需要 pip）
uv run python app.py
```

> `requirements.txt` 只作历史参考（pip 时代的冻结清单），**不要再用 pip 安装**。

LLM 配置（当前使用 **DeepSeek / deepseek-flash**）：

```bash
# 方式一：系统环境变量（Windows 用户变量设置一次即可）
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 方式二：写进 .env（复制 .env.example 修改即可，代码里已真正接上 load_dotenv）
# 优先级：系统环境变量 > .env > config.ini
```

> 数据库密码同理：`MYSQL_PASSWORD` / `REDIS_PASSWORD` / `NEO4J_PASSWORD` 走 `.env`，
> `config.ini` 里只留非敏感配置。仓库里的 `config.ini` 已经是「无密码版」，可直接用；
> 想改成别的值就复制 `config.example.ini` 为 `config.ini` 自行修改。

模型权重（BGE-M3 / bge-reranker / 文档分割模型）：
- 默认从 `models/` 读取；
- 找不到时自动回退到你旧项目的 `Itcast_qa_system/rag_qa/models`；
- 也可用环境变量 `FIN_MODEL_ROOT` 指定目录。
- 模型权重本身不在仓库里（体积大），需自行下载或用 `FIN_MODEL_ROOT` 指向已有目录。

## 4. 数据管道

```bash
# 1) 下载/补数据（已有 data 可跳过）
uv run python scripts/download_finance_data.py --out C:/pro/data

# 2) 年报解析：PDF -> 父子块 + 元数据（公司/年份/章节/页码）
uv run python scripts/parse_annual_reports.py

# 3) 入库（已完成）
#    MySQL：公司/财务指标/十大股东/文档/会话表
#    Milvus：2023–2025 年报“非财务报告”章节 children.jsonl -> BGE-M3 稀疏+稠密向量
#    Neo4j（可选）：holders.csv -> 公司-股东图
uv run python init_data.py --all --recreate            # 全量重建（含 60 份年报全部年份，耗时数小时）
uv run python init_data.py --milvus --years 2025 --skip-financial-report --batch-size 16
```

**当前已入库状态（2026-10）**

| 层 | 内容 | 数量 |
|---|---|---|
| MySQL | companies / financial_indicators / holders / documents | 20 / 60 / 186 / 60 |
| Milvus | 2023–2025 年报非财务章节子块（20 家公司，含公司/年份/章节/页码 metadata） | **24,235 条** |
| Neo4j | 股东图谱（P3，未启用） | — |

> 财务报告章节（约 10,337 个子块）**故意不入向量库**，指标类问题交给 MySQL + NL2SQL；
> 向量库只负责“年报原文解释类”问题，避免把报表数字切碎后检索不准。

**仓库里带了什么、没带什么**

| 内容 | 是否入库 | 说明 |
|---|---|---|
| `data/structured/*.csv`（财务指标 / 十大股东 / 索引） | ✅ | 几十 KB，别人克隆即可复现 MySQL |
| `rag_qa/classify_data/training_data_finance_1200.jsonl` | ✅ | BERT 意图分类器训练集 |
| `eval/`（评测结果与 RAGAS 明细） | ✅ | 答辩用的评测证据 |
| `docs/`（方案文档 + 汇报 PPT） | ✅ | — |
| `data/raw/annual_reports/`（60 份 PDF，169 MB） | ❌ | 版权原因，改用 `scripts/download_finance_data.py` 重新下载 |
| `data/processed/`（切块产物，86 MB） | ❌ | 由 `scripts/parse_annual_reports.py` 重新生成 |
| `rag_qa/core/bert_query_classifier/`（390 MB） | ❌ | 超过 GitHub 单文件上限，用 `scripts/train_intent_bert.py` 复现 |
| `.env` | ❌ | 含密码与 Key，仓库只留 `.env.example` |

## 5. 启动

```bash
# 建议先在系统环境变量里设置（否则 LLM 客户端初始化会失败）
#   DEEPSEEK_API_KEY=sk-xxxx
#   DEEPSEEK_BASE_URL=https://api.deepseek.com
# 也可以写进 .env（复制 .env.example 修改）

uv run python app.py      # FastAPI + SSE + WebSocket，端口 8003
# 或直接跑脚本：Windows 双击 start.bat / Linux 用 ./start.sh
```

打开 `http://127.0.0.1:8003/`（单文件前端，无构建步骤，静态资源已本地化，断网也能演示）。

**页面功能模式**（输入框上方可切换）：

| 模式 | 说明 | 实现 |
|---|---|---|
| 智能问答 | 自动意图识别 + 路由（默认） | `mode=auto` |
| 财务指标 | 强制走 MySQL NL2SQL | `mode=sql` |
| 年报原文 | 强制走 Milvus RAG，带页码引用 | `mode=rag` |
| 多公司对比 | 自动生成对比表（默认全行业 2025） | `mode=compare` |
| 股东图谱 | Neo4j 开发中，先用十大股东数据回答 | `mode=graph` |

> 接口层：`/api/query`、`/api/query/stream`、WebSocket 都支持传 `mode` 字段，不传即 `auto`。

> 首次查询会懒加载 BM25 关键词索引（24,235 个 chunk，约 15-20 秒），之后查询 1-3 秒检索 + LLM 生成。

## 6. 工程结构（与 Itcast_qa_system 对齐）

```
FinSight-/
├── app.py                    # FastAPI 入口（HTTP/SSE/WS/会话/引用原文）
├── new_main.py               # 集成问答主逻辑（意图识别 → 追问改写 → SQL/RAG）
├── config.ini                # 配置中心（仓库版本无密码，密码走 .env）
├── config.example.ini        # 配置模板
├── init_data.py              # 一键初始化 MySQL / Milvus
├── start.bat / start.sh      # 一键启动脚本
├── base/                     # 配置 + 日志（轮转）
├── mysql_qa/                 # MySQL 客户端 + Redis 客户端 + BM25 快路径
├── rag_qa/
│   ├── core/                 # 向量库 / 检索策略 / 提示词 / 意图分类
│   ├── edu_document_loaders/ # PDF/Word/PPT/图片 加载器
│   └── edu_text_spliter/     # 中文递归切分 / 语义切分
├── sql_qa/                   # NL2SQL（财务指标）
├── graph_qa/                 # Neo4j 股东图谱（可选）
├── agent/                    # 意图路由 + 图表构建
├── services/                 # 行情 / 图表 / 报告导出 / 板块 / 语义缓存
├── api/                      # chart / market / report / sector 路由
├── scripts/                  # 下载 / 解析 / 入库 / 训练 / 评测 / 校准脚本
├── static/                   # 前端单页（含本地化的 marked / ECharts / Font Awesome）
├── tests/                    # 单元测试与冒烟脚本
├── eval/                     # 评测集与评测结果
├── data/                     # 数据（structured/index 入库，raw/processed 不入库）
└── docs/                     # 项目方案 / 背景 / 汇报 PPT / 专题文档
```

## 7. 改造进度

- [x] P0：骨架 + 复用模块 + 新配置 + 年报解析管道（60 份 → 11,001 页 → 34,572 子块）
- [x] P1：数据层完成
  - [x] 章节归一化（10 类标准章节 + 前置，0 异常文档）
  - [x] MySQL 建表 + 导入（companies 10 / financial_indicators 30 / holders 100 / documents 30）
  - [x] Milvus schema 扩展（公司/年份/章节/页码等 metadata）+ 按公司/年份过滤
  - [x] 金融提示词（要求【公司 年份 第X页】引用）
  - [x] **Milvus 年报非财务章节入库完成：24,235 条，20 家公司 × 2023–2025 全覆盖**
  - [x] 清理早期测试残留（2023 泸州老窖 100 条），集合总量 = 24,235
- [x] P1.5：RAG 检索质量修复（实测踩出来的坑）
  - [x] 强制检索：新增 `[retrieval] always_retrieve=true`，跳过旧教育 BERT 分类器的“通用知识→直接问 LLM”分支（否则文档题永远不检索）
  - [x] 关闭 FAQ 快路径：`enable_fast_path=false`。共享 Redis 里残留旧项目教育客服题库（`qa_original_questions`），不关会拿“CSS 样式修改”来答白酒问题
  - [x] 公司实体识别：问题只提到一家公司时自动加 `source=` 过滤（茅台→贵州茅台、汾酒→山西汾酒…）
  - [x] 上下文带出处：检索块格式化为【公司 年份 第X页 章节】再进 Prompt，回答可标注页码
  - [x] **BM25 关键词召回 + 向量召回 RRF 融合**：修复“经营风险/渠道库存”这类关键词问题向量召回不到的盲区；`jieba.cut_for_search` 分词 + 问句词过滤
  - [x] 父子块错位兜底：年报“一页拆多节”会导致父窗口落到相邻章节，命中时自动回退用子块文本
  - [x] app.py 无 BM25 时 `/api/query` 500 修复（快路径关闭后走完整问答链路）
- [x] P2：`sql_qa`（NL2SQL）+ 意图路由（规则版）+ 接入 `new_main`
  - [x] `sql_qa/schema.py` 表结构/字段说明（4 张表）
  - [x] `sql_qa/sql_generator.py` LLM 生成 SQL + 结果自然语言总结
  - [x] `sql_qa/sql_executor.py` sqlglot 校验（单条 SELECT / 表白名单 / 禁用关键字 / 强制 LIMIT）+ MySQL 执行 + Markdown 表格
  - [x] `sql_qa/sql_qa.py` + CLI：`python -m sql_qa "问题"`
  - [x] `agent/router.py` 规则路由：sql / rag / graph / unsupported（行情/估值/荐股等知识库外问题）
  - [x] `new_main.py` 接入：指标题走 SQL，文档题走 RAG，**知识库外问题直接交给 LLM 通用回答**（不硬拒答）
  - [ ] 图表数据接口（P4 前端一起做）
- [ ] P3：`graph_qa`（Neo4j 股东图）+ Agent 工具调用升级
- [x] P4-1：前端美化 + 回答精简 + 功能模式
  - [x] `static/index.html` 全新 UI：深蓝+金色金融风、品牌头部、聊天气泡 + 头像、Markdown 表格/代码样式、响应式、滚动条/加载动效
  - [x] 输入框上方 5 个功能模式切换：智能问答 / 财务指标 / 年报原文 / 多公司对比 / 股东图谱
  - [x] 回答只保留**关键结论**：SQL 题不再附表格和 SQL 原文（只写日志），RAG/通用回答 Prompt 要求先给结论、控制在 250 字内
  - [x] **金融意图识别 BERT 训练 + 评估**：1200 条二分类数据（9:1 训练/测试），CPU 训练 8.3 分钟；测试集 acc 1.0，34 条人工 OOD 样本 acc 1.0（规则基线 0.941）
- [ ] P4-2：Vue 前端、测试、评测、README/答辩材料

## 7.3 端到端验证结果（2026-10-19，`C:\pro\.venv`）

| 测试 | 结果 |
|---|---|
| MySQL 指标查询 | 2025 营收 TOP3：茅台 1720.54 亿 / 五粮液 405.29 亿 / 汾酒 387.18 亿 ✅ |
| NL2SQL 安全 | `DELETE FROM companies`、`SELECT * FROM mysql.user` 均被拒绝 ✅ |
| RAG 原文问答 | “茅台 2025 经营风险” → 答出宏观经济/安全/舆情/环保四项，标注【贵州茅台 2025 第23页】✅ |
| RAG 多渠道融合 | “2025 白酒行业整体情况” → 跨 5 家公司原文对比 + 逐条页码引用 ✅ |
| 知识库外问题 | “茅台今日股价” → 不再拒答，LLM 通用回答 + 数据时效说明 + 风险提示 + 替代问法 ✅ |
| 意图识别 | 问答前先做意图识别：通用知识/闲聊/定义类 → LLM 直答；专业咨询 → SQL/RAG；行情/估值 → LLM 通用回答 ✅ |
| 功能模式 | 页面顶栏可切换 5 种功能：智能问答 / 财务指标 / 年报原文 / 多公司对比 / 股东图谱（`mode` 参数，兼容 `/api/query`、SSE、WebSocket）✅ |
| 追问改写 | “利润呢”“那现金流呢” → 结合上一轮补全为完整问题再查 SQL（实测“利润呢”→ 贵州茅台归母净利润 823.20 亿）✅ |
| LLM 切换 | 全部链路改走 **DeepSeek（deepseek-flash）**，Key/Base URL 取系统环境变量 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` ✅ |
| 服务接口 | `/health` 200、`/` 200（FinSight 页面）、`/api/sources` 20 家公司、`/api/query` 非流式、`/api/query/stream` SSE 均正常 ✅ |
| 前端 UI | 新品牌主题（深蓝+金色）、气泡+头像、Markdown 表格样式、响应式，`static/index.html` 全量重写 ✅ |
| 回答精简 | SQL 题只返回关键结论（无表格/SQL，约 138 字）；RAG 答案要求先结论、≤250 字 ✅ |
| 行情路由 | “茅台的股票怎么样” → 走行情数据服务（实时价/涨跌/市值/PE/PB + 2025 年报基本面）✅ |
| RAG 未命中兜底 | “茅台年报里有没有提到无人机” → RAG 明确未找到后自动转 LLM 通用回答，并标注“知识库未检索到” ✅ |
| 测试脚本 | `tests/verify_milvus.py`、`tests/smoke_retrieval.py`、`tests/smoke_e2e.py` |

> 单次问答耗时（DeepSeek 实测）：SQL 类约 **4-6 秒**（生成 SQL + 总结 2 次调用）；RAG 类约 **10-15 秒**（策略选择 + 检索 + 生成）；知识库外问题约 **2 秒**。
> CPU 机器上属于正常范围；有 GPU 后可把 `use_reranker=true` 进一步提精度。

**环境验证（uv）**：`uv sync` 成功（281 个解析包 / 267 个安装包），`pyproject.toml` + `uv.lock` 已生成；
`.venv` 内**已无 pip**（`python -m pip` 报 `No module named pip`），全部测试改用 `uv run` 重跑通过：
Milvus 24,235 条可读、检索冒烟、`/health`、`/`、`/api/query`、`/api/query/stream` 均正常。

## 7.4 金融意图识别 BERT（训练 / 评估）

| 项目 | 内容 |
|---|---|
| 训练数据 | `rag_qa/classify_data/training_data_finance_1200.jsonl`（申请 1200 条：专业咨询 600 / 通用知识 600，11 个子类） |
| 生成脚本 | `scripts/gen_intent_dataset.py`（模板槽位 + DeepSeek 扩样，可 `--per-class` 调整） |
| 训练脚本 | `scripts/train_intent_bert.py`（9:1 训练/测试，训练集内再留 10% 验证；3 epochs；CPU 8.3 分钟） |
| 模型产物 | `rag_qa/core/bert_query_classifier/`（390 MB，**不在 Git 仓库里**，见下方复现说明） |
| 评估脚本 | `scripts/eval_intent_ood.py`（人工构造 34 条真实风格问题做分布外测试） |
| 评估报告 | `rag_qa/classify_data/intent_eval_report.md`、`intent_eval.json`、`ood_eval.json` |

> **为什么模型不入库**：单个文件 390 MB，超过 GitHub 单文件 100 MB 硬限制。
> 仓库里保留了训练数据与训练脚本，克隆后 8 分钟即可复现出同样的权重：
>
> ```bash
> uv run python scripts/gen_intent_dataset.py      # 可选：重新生成训练集
> uv run python scripts/train_intent_bert.py       # 训练，产物写入 rag_qa/core/bert_query_classifier/
> uv run python scripts/eval_intent_ood.py         # 分布外评估
> ```
>
> 没有这个模型目录时系统也能跑：意图识别会退回规则版（`agent/router.py`），
> 准确率略低（见上表规则基线），但问答链路完全不受影响。

**评估结果**

| 数据集 | 模型 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 合成测试集（120 条，9:1 切分） | BERT | 1.000 | 1.000 | 1.000 | 1.000 |
| 合成测试集（120 条） | 规则基线 | 0.958 | 0.951 | 0.967 | 0.959 |
| 人工 OOD（34 条真实风格问题） | BERT | 1.000 | 1.000 | 1.000 | 1.000 |
| 人工 OOD（34 条） | 规则基线 | 0.941 | 0.944 | 0.944 | 0.944 |

> 说明：合成测试集与训练集同分布（同模板不同公司/年份），1.0 会偏乐观；人工 OOD 小样本更接近真实用户问法，但样本量仍小。
> 后续应补充 100–200 条真实 query 人工标注，并按"模板/问题簇"分组切分再评估泛化能力。
> 单条推理 CPU 约 55 ms；模型加载约 0.5 s。


## 7.5 评测结果（自建评测集，2026-10-20）

脚本：`scripts/run_finance_eval.py`、`scripts/run_ragas_eval.py`（RAGAS）；报告：[finance_eval_report.md](eval/finance_eval_report.md)；原始数据：`eval/finance_eval_results.json`
评测集：**20 道文档题 + 12 道指标题**（覆盖 20 家公司，自建小样本）。

| 评测项 | 结果 |
|---|---|
| 检索 hit@5（BM25+RRF 融合） | **95.0%**（关闭 BM25 为 90.0%） |
| 检索 MRR（融合 / 仅向量） | **0.885 / 0.767** |
| 检索平均延迟（融合） | 1.44 s（仅向量 0.53 s） |
| RAG 答案关键词命中率 | **95.0%（19/20）** |
| RAG 引用标注覆盖率 | **85.0%（17/20）** |
| RAG 平均 / P50 / P95 延迟 | 8.47 s / 7.74 s / 15.29 s |
| SQL 数值命中率 | **100%（12/12）** |
| SQL 平均 / P50 / P95 延迟 | 1.86 s / 1.73 s / 2.63 s |
| RAGAS Faithfulness / Answer Relevancy | **0.878 / 0.928**（11 题，DeepSeek 裁判 + 本地 BGE-M3） |
| RAGAS Context Precision / Recall | **0.755 / 0.636**（检索层是当前瓶颈） |

> 局限：自建小样本（32 题）、单请求串行测量、未开 reranker（CPU 单次 30–60s）；下一步扩到 100+ 题、做章节偏好检索、GPU 下 reranker A/B 与并发压测。

## 7.1 Milvus 入库速度与常用命令

```bash
# 先验证 100 条（约 3-5 分钟，含模型加载）
uv run python init_data.py --milvus --limit 100 --batch-size 16

# 只入库 2025 年报、跳过财务报告章节（约 8,174 子块，约 4-6 小时）
uv run python init_data.py --milvus --years 2025 --skip-financial-report --batch-size 16

# 全量（34,572 子块，CPU 约 10-12 小时，建议挂后台）
uv run python init_data.py --milvus --batch-size 16
```

> CPU 上 BGE-M3 编码慢是本机硬件限制（torch 为 CPU 版）；装 CUDA 版 torch 后可用 GPU。
> `config.ini [retrieval] use_reranker=false` 时单次检索约 2-5 秒；开启重排会慢 30-60 秒。

## 7.2 NL2SQL 用法（P2）

```bash
# 自然语言 → SQL → 执行 → 中文总结（需要 DEEPSEEK_API_KEY）
uv run python -m sql_qa "2025年营业收入最高的3家公司"
uv run python -m sql_qa "贵州茅台2023到2025年的毛利率和ROE分别是多少"
uv run python -m sql_qa "招商中证白酒基金持有哪几家白酒公司"

# 只生成 SQL + 表格，不生成自然语言总结
uv run python -m sql_qa "哪些公司ROE连续三年下降" --no-summary

# 直接执行 SQL（不调 LLM，用于调试）
uv run python -m sql_qa --sql "SELECT c.name, f.total_operate_income FROM financial_indicators f JOIN companies c ON c.code=f.code WHERE f.report_year=2025 LIMIT 3"
```

安全限制：只允许单条 SELECT、只允许 companies/financial_indicators/holders/documents 四张表、
禁止 DDL/DML/系统表、自动补 LIMIT 200。

## 7.6 语义缓存（BGE-M3 向量缓存）

同一个问题问第二遍，不再走「向量检索 + LLM 生成」整条链路，直接复用上次的答案。
RAG 类问题原本要 10–15 秒，命中缓存后基本是秒回。

### 怎么工作的

```
用户问题
   │
   ├─ 归一化（全角→半角、去标点、压空白）
   ├─ BGE-M3 编码成 1024 维向量（复用 VectorStore 已加载的模型，不额外占内存）
   ├─ 与缓存里的历史向量做余弦相似度比对
   └─ 六道判定全过才命中：
        ① 向量余弦 ≥ 0.85        （粗筛，挡掉明显无关的问题）
        ② 实词 Jaccard ≥ 0.2     （弱门槛，只挡极端词面无关的情况）
        ③ 指标类别不冲突          （营收 vs 净利 属于不同问题）
        ④ 年份不冲突              （2025 年营收 ≠ 2023 年营收）
        ⑤ 方向不冲突              （营收最高 ≠ 营收最低）
        ⑥ 序数口径不冲突          （ROE 前三 ≠ ROE 前十）
```

### 为什么阈值是 0.85（实测，不是拍脑袋）

`scripts/calibrate_cache_threshold.py` 用**真实 BGE-M3** 跑了 12 组问题：

| 分组 | 例子 | 相似度 |
|---|---|---|
| 同义改写（应命中） | 「2025年营业收入最高的公司」/「哪家公司2025年营收最高」 | 0.9068 ~ 0.9767 |
| 异指标（应拒绝） | 「2025年营业收入最高的3家公司」/「2025年净利润最高的3家公司」 | **0.9302** |
| 跨年（应拒绝） | 「贵州茅台2025年营业收入」/「贵州茅台2023年营业收入」 | 0.8605 |
| 无关（应拒绝） | 「2025年营收排名」/「白酒行业竞争格局」 | 0.44 ~ 0.47 |

**同义组最低 0.9068 < 异指标组最高 0.9302，两者交叉 —— 纯向量阈值无论怎么调都不行**
（调高会漏掉同义改写，调低会把「营收TOP3」当成「净利TOP3」）。

还有一条实测教训：最初把「实词 Jaccard ≥ 0.45」当硬门槛，结果**同义改写反而被挡掉**——
「营收」和「营业收入」分词后是 `营收` vs `营业+收入`，Jaccard 只有 0.33，
「ROE最高的前三家公司」vs「ROE排名前三」也只有 0.40。
所以 Jaccard 被降级为弱门槛（0.2），正确性改由 ③④⑤⑥ 四道精确判定保证。

这就是为什么要六道门槛，而不是简单地「相似度超过 0.9 就复用」——
每道门槛的取值都来自真实模型实测，且每道都对应一类**答错就是硬伤**的情况。

### 缓存范围（有讲究）

| 链路 | 是否缓存 | 原因 |
|---|---|---|
| 财务指标（SQL） | ✅ | 数据来自年报，答案确定 |
| 年报原文（RAG） | ✅ | 知识库内容固定 |
| 行情（market） | ❌ | 实时数据，缓存就是错的 |
| LLM 通用回答 | ❌ | 含时效性说明与风险提示，缓存会过期 |
| 追问（「利润呢」） | ❌ | 答案依赖上下文，且短问题容易误命中 |

### Redis 优先，自动降级

- **Redis 后端**（默认）：`finsight:sem:v:*` 存 float32 向量、`finsight:sem:e:*` 存条目 JSON、
  `finsight:sem:index` 是 ZSET 索引，带 TTL（默认 2 小时），最多 500 条。
- **内存后端**：Redis 连不上时自动切换（进程内 LRU，重启失效）。
  本机默认没装 Redis，演示时走的就是这条路径，日志里会写明「降级为进程内内存模式」。

### 用法

```bash
# 预热：把答辩高频问题先跑一遍灌进缓存
uv run python scripts/warm_cache.py            # 首轮跑一遍
uv run python scripts/warm_cache.py --repeat   # 再问一遍看命中耗时

# 校准阈值（真实 BGE-M3，约 3 分钟）
uv run python scripts/calibrate_cache_threshold.py

# 单元测试（不需要任何服务）
uv run python tests/test_semantic_cache.py
```

接口：`GET /api/cache/stats` 看命中率，`DELETE /api/cache` 清空。
前端侧栏「语义缓存」实时显示后端 / 条目数 / 命中率，回答下方会挂一个
「⚡ 语义缓存命中 · 相似度 xx%」的徽标。

## 8. 合规

- 仅供信息检索与研究辅助，**不构成投资建议**；
- 年报/财务数据仅用于学习演示，注明来源、不再分发；
- 知识库外问题（行情/预测/荐股）：**不硬拒答，交给 LLM 通用能力回答**，但要求它
  （1）说明无法访问实时数据、建议查看行情软件；（2）不编造实时价格/市值；
  （3）提示"不构成投资建议"；（4）给出可用年报数据回答的替代问法。

