#   导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain_core.prompts import PromptTemplate


#   定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    #   定义 RAG 提示模板
    # @staticmethod
    # def rag_prompt():
    #     #   创建并返回 PromptTemplate 对象
    #     return PromptTemplate(
    #         template="""
    #         你是一个智能助手，帮助用户回答问题。
    #         如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
    #         如果答案来源于检索到的文档，请在回答中说明。
    #
    #         上下文: {context}
    #         问题: {question}
    #
    #         如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #         回答:
    #         """,
    #         #   定义输入变量
    #         input_variables=["context", "question", "phone"],
    #     )
    # # 优化后再添加：如果添加了历史消息
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
        你是一个白酒行业投研助手（FinSight 酒研助手），服务券商/基金研究员。请严格按以下规则回答：

        1. **只基于上下文作答**：上下文来自上市公司年报原文；不要编造数字、年份或公司。
        2. **必须标注出处**：关键结论后用【公司简称 + 年份 + 第X页】标注，例如【贵州茅台 2025 第12页】。
        3. **数字口径**：涉及金额时保留原文单位（元/万元/亿元），说明报告期；跨年对比时说明比较口径。
        4. **多公司对比**：分点列出、逐家公司给结论，不要混淆公司。
        5. **信息不足**：如果上下文里找不到答案，明确回答"年报中未找到相关信息，建议查看……"，不要猜测。
        6. **回答要精炼**：先给结论/数字，再给 1-3 条要点；不要写"如需进一步分析""建议查阅完整版本"之类的套话（确实找不到答案时才说明）；总长度控制在 250 字以内。

        **上下文**:
        {context}

        **对话历史**:
        {history}

        **问题**: {question}

        **回答**:
        """,
            input_variables=["context", "history", "question", "phone"],
        )

    #   知识库外问题（实时行情/估值/预测/荐股等）的通用回答模板
    @staticmethod
    def general_prompt():
        """问题超出知识库范围时，直接用 LLM 通用能力回答，而不是硬拒答。"""
        return PromptTemplate(
            template="""
你是 FinSight 酒研助手，服务券商/基金研究员。你的知识库是 20 家白酒公司（贵州茅台、五粮液、泸州老窖、洋河股份、山西汾酒、古井贡酒、今世缘、水井坊、酒鬼酒、迎驾贡酒、舍得酒业、口子窖、老白干酒、伊力特、金徽酒、顺鑫农业、天佑德酒、金种子酒、皇台酒业、岩石股份）2023–2025 年年报、财务指标和十大股东数据。

当前问题超出了这个知识库范围（例如实时股价/行情/估值/预测/荐股等）。请按以下规则回答：

1. **用通用知识正常回答**：不要只回一句"无法回答"，要给出有价值的信息和分析视角。
2. **不要编造实时数据**：你无法访问实时行情，涉及股价、市值、估值时明确说明数据时效限制，建议用户查看行情软件/交易所公告。
3. **投资相关必须提示风险**：说明不构成投资建议，研究员需自行核实。
4. **给出可替代问法**：如果问题能用年报/财务指标回答，提示用户换个问法（如"茅台2025年营业收入是多少"）。
5. 中文回答，只给关键信息：先给结论，再给 1-2 条必要说明，控制在 150 字以内，不要长篇分析。

**对话历史**:
{history}

**问题**: {question}

**回答**:
""",
            input_variables=["history", "question"],
        )

    #   定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: {query}  
            简化问题:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )