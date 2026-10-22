# STORY.md — FinSight 酒研助手 项目汇报

## ① 用户意图对齐

- **目标受众**：技术面试官 / 毕设答辩老师 / 技术分享会同行。他们不熟悉白酒行业，但懂 RAG、懂工程。
- **核心目标**：让听众相信「这是一个跑通了的行业 RAG 系统，不是 demo」——记住三个数字：**24,235 条向量**、**hit@5 95%**、**SQL 数值 100%**；并认可「知道自己哪里不行」的工程判断力。
- **PPT 长度**：12 页。10 页档的 Hero 配额是 2–3 页，本稿 3 页（25%）。
- **视觉调性**：专业、数据驱动、深蓝金融、克制、有纵深。
- **内容边界**：
  - 必讲：数据规模与处理链路、三条能力线（原文 RAG / NL2SQL / 意图路由）、量化评测、真实踩坑。
  - 不讲：依赖版本、代码结构、部署细节。
  - 禁碰：任何投资建议、收益预测。

## ② 页面布局骨架

**12 页 / 2 章**，目录声明 2 章 → 全篇有且仅有 2 个 `type: section` 扉页（P03 = 第 1 章，P07 = 第 2 章），编号 01、02 连续，标题与目录逐字一致。

| 章节 | 页码区间 | 扉页位置 |
|---|---|---|
| 01 一个入口，三条链路 | P03–P06 | P03 |
| 02 让它准起来，也让它说实话 | P07–P11 | P07 |

**Hero 页定位**：P01 封面、P06 数据规模、P12 结束页 = 3 页（25%）；两两之间有 supporting 页间隔（P06 与 P12 之间隔 5 页）。

**rhythm 曲线**：
P01 peak → P02 valley → P03 transition → P04 valley → P05 valley → **P06 peak** → P07 transition → P08 valley → P09 valley → P10 valley → P11 valley → **P12 peak**

> P08–P11 连续 4 页 valley，靠 P10（评测表格）+ P11（卡片组）的版式差异与色彩节奏区分，避免沉闷；P12 以结束页收束成 peak。

**非对称版式预算**：P01、P02、P04、P05、P06、P08、P09、P12 共 8 页（67%）≥ 40% ✅
**对称版式预算**：仅内容页 P10、P11 使用（全篇 2 页，达标）；P03/P07 章节扉页按规范用「全屏视觉+大标题」，不计入内容页预算。

## ③ 页面大纲

### P01 封面 · hero · peak
- layout：`全幅图+骑线文字`
- visual：L1: `cover_bg.png`（满版，占 100%），文字从左侧骑入占 42%
- visual_role：atmosphere
- density：字数约 28 / 图片 1 张 / 留白约 42%
- anti_pattern：禁止标题居中对称排版；禁止在封面塞能力清单；禁止图片压暗到看不清
- description：主标题「FinSight 酒研助手」，副标「白酒行业智能投研工作台 · 年报原文 RAG + 财务指标 NL2SQL」；一行数据锚点「20 家公司 · 60 份年报 · 24,235 条向量」。

### P02 目录 · supporting · valley
- layout：`左标题+右内容`
- visual：L3 角标（无 L1）
- visual_role：evidence
- density：字数约 110 / 图片 0 张 / 留白约 30%
- anti_pattern：禁止四个等宽方块罗列；禁止目录项只有名词没有判断
- description：左侧竖排「目录 / CONTENTS」大字，右侧两章条目，每条带一句实质说明——01「一个入口，三条链路：把散落的年报变成可问答的知识」；02「让它准起来，也让它说实话：检索、生成、评测、返工」。

### P03 章节扉页 01 · transition
- layout：`全屏视觉+大标题`（对称，章节页豁免）
- visual：L1: `section01_bg.png`（满版低对比质感，opacity 0.35）
- visual_role：atmosphere
- density：字数约 34 / 图片 1 张 / 留白约 55%
- anti_pattern：禁止章节页塞内容预告卡片；禁止放大段正文
- description：大字「01 / 一个入口，三条链路」，一句小标「研究员的同一个问题，系统自己判断该查哪里」。

### P04 痛点 · supporting · valley
- layout：`非对称双栏`（65:35）
- visual：L1: `pain_flow.png`（左 63%，流程图式对比：现在 vs 系统）
- visual_role：evidence
- density：字数约 210 / 图片 1 张 / 留白约 22%
- anti_pattern：禁止把痛点写成三行 bullet；禁止用通用「效率低」这种空话
- description：数据+判断——「一份年报 180 页，五个关键指标散在三个章节，人工对 20 家公司做横比要 2 天 → 真正的瓶颈不是找不到文件，是**找不到那句话在第几页**」。右侧窄栏列出三类高频问题：原文解释、指标计算、跨公司横比。

### P05 系统全景 · supporting · valley
- layout：`上大图+下方卡片`
- visual：L1: SVG 架构图（上 58%）：数据层 → 检索层 → 路由层 → 生成层 → 前端
- visual_role：anchor
- density：字数约 230 / 图示 1 张 / 留白约 20%
- anti_pattern：禁止把架构图画成等宽方块流水线；禁止卡片里再放小架构图
- description：「一次提问，五层协作」——下方三张不等宽卡片对应三条链路：自动（智能问答）/ 强制 SQL（财务指标）/ 强制 RAG（年报原文），并注明「路由错一次，答案就答非所问，所以先分类再检索」。

### P06 数据规模 · hero · peak
- layout：`巨型数字+洞察`
- visual：L1 主视觉为 4 个巨型数字（≥ 72px），无图片
- visual_role：anchor
- density：字数约 130 / 图片 0 张 / 留白约 38%
- anti_pattern：禁止把四个数字做成等宽四卡；禁止数字小于 56px；禁止只报数不给判断
- description：「24,235」为主锚点（Milvus 向量条数），辅以 60 份年报 / 11,001 页 / 34,572 子块。判断：**财务报告章节的 10,337 个子块被故意排除在向量库外**——报表数字切碎后检索反而更不准，这类问题交给 MySQL。

### P07 章节扉页 02 · transition
- layout：`全屏视觉+大标题`（对称，章节页豁免）
- visual：L1: `section02_bg.png`（满版质感，opacity 0.35）
- visual_role：atmosphere
- density：字数约 32 / 图片 1 张 / 留白约 55%
- anti_pattern：禁止预告后续四页标题；禁止装饰性图表堆叠
- description：大字「02 / 让它准起来，也让它说实话」，小标「检索、生成、评测，以及四次返工」。

### P08 RAG 双路召回 · supporting · valley
- layout：`左大图+右侧文字`
- visual：L1: SVG 检索链路图（左 58%）：问句 → BM25 关键词召回 ‖ 向量召回 → RRF 融合 → 父子块回填 → Prompt
- visual_role：anchor
- density：字数约 240 / 图示 1 张 / 留白约 24%
- anti_pattern：禁止用三行文字描述流程；禁止把「RRF」写成公式堆砌而不解释收益
- description：数据+判断——「纯向量 hit@5 只有 90%，补上 BM25 后到 **95%**，MRR 从 0.767 提到 **0.885**，代价是延迟从 0.53s 涨到 1.44s → **多花 1 秒换回 5 个百分点的召回，值**」。右侧说明为什么必须补关键词：『渠道库存』『经营风险』这类词向量抓不住。

### P09 NL2SQL 与意图路由 · supporting · valley
- layout：`左标题+右内容`
- visual：右侧两层结构：上层 BERT 分类（1200 条训练 / OOD acc 1.0 / 55ms），下层 SQL 安全闸（单条 SELECT、四表白名单、强制 LIMIT 200）
- visual_role：evidence
- density：字数约 250 / 图片 0 张 / 留白约 25%
- anti_pattern：禁止把安全校验写成「做了很多校验」；禁止不提规则基线的对比数字
- description：数据+判断——「12 道指标题数值命中 **100%**，平均 1.86 秒；`DELETE FROM companies` 被拒 → **让 LLM 写 SQL，危险的不是它写错，是它写对了一句你不想要的话**」。BERT 分类器 OOD acc 1.0 vs 规则 0.941，但小样本，须说明局限。

### P10 评测结果 · supporting · valley
- layout：`图表+洞察`（对称）
- visual：主图 = 横向条形对比（RAG 三指标 / SQL 两指标）+ 右侧 RAGAS 四分数
- visual_role：anchor
- density：字数约 200 / 图表 1 张 / 留白约 18%
- anti_pattern：禁止只列一堆数字不排序；禁止隐藏短板
- description：数据+判断——「检索 hit@5 95%、答案关键词命中 95%、SQL 100%、Faithfulness 0.878 → 生成层已经够用，**Context Recall 只有 0.636，瓶颈在检索没捞全，不在模型**」。下方一行小字如实标注：32 题自建小样本、单请求串行、未开 reranker。

### P11 四次返工 · supporting · valley
- layout：`N卡片横排`（4 卡，对称，全篇唯一一次）
- visual：每卡顶部一行「症状 → 根因 → 改法」三行结构
- visual_role：evidence
- density：字数约 320 / 图片 0 张 / 留白约 16%
- anti_pattern：禁止写成「遇到的问题与解决方案」八股标题；禁止只写症状不写根因
- description：四张卡对应四次真实故障：① 用「CSS 样式修改」回答白酒问题（共享 Redis 残留旧项目 FAQ）② 文档题永不检索（旧 BERT 判为通用知识）③ 关键词召回不到（补 BM25+RRF）④ 父窗口落到相邻章节（子块回退）。判断收束：**这些 bug 都不是模型能力问题，是「复用」带来的上下文污染**。

### P12 结束 · hero · peak
- layout：`居中金句/巨型数字`（对称）
- visual：L1: `ending_bg.png`（满版极简，opacity 0.25）
- visual_role：atmosphere
- density：字数约 40 / 图片 1 张 / 留白约 50%
- anti_pattern：禁止「谢谢观看」四个字占满一页；禁止堆联系方式卡片
- description：金句「**让模型少说一点，让数据多说一点**」，下方三行落点：一个入口三条链路 / 24,235 条向量 / hit@5 95%。落款 FinSight · 2026.10。
