# -*- coding: utf-8 -*-
"""自然语言 → SQL（LLM，当前为 DeepSeek）。"""
import re

from .llm_client import LLMClient
from .schema import SCHEMA_TEXT


class NoSQLAnswer(Exception):
    """问题无法用现有表回答。"""


PROMPT_TEMPLATE = """你是 MySQL 专家。请根据下面的表结构，把用户的中文问题转换成**一条** MySQL SELECT 语句。

{schema}

## 硬性规则
1. 只能查询 companies / financial_indicators / holders / documents 这 4 张表；
2. 只能生成一条 SELECT；禁止 INSERT、UPDATE、DELETE、DROP、ALTER、TRUNCATE 等任何写操作；
3. 金额字段单位是"元"，展示时请 `ROUND(x/1e8, 2) AS xxx_yi` 转成亿元；
4. 同比/ROE/毛利率/净利率/资产负债率/持股比例等字段**本身就是百分数**（如 7.52 表示 7.52%），不要再乘 100；
5. 公司名用 companies.name 关联展示，年份用 financial_indicators.report_year；
6. 必须带 LIMIT（最多 200 行）；
7. 如果问题无法用这 4 张表回答，只输出：NO_SQL
8. 只输出 SQL 本身，不要解释，不要用 ``` 代码块。

## 参考示例
问题：2025 年营业收入最高的 3 家公司
SQL：SELECT c.name, ROUND(f.total_operate_income/1e8, 2) AS revenue_yi FROM financial_indicators f JOIN companies c ON c.code = f.code WHERE f.report_year = 2025 ORDER BY f.total_operate_income DESC LIMIT 3

问题：贵州茅台 2023 到 2025 年的毛利率和 ROE
SQL：SELECT c.name, f.report_year, f.gross_margin, f.roe_weighted FROM financial_indicators f JOIN companies c ON c.code = f.code WHERE c.name = '贵州茅台' AND f.report_year BETWEEN 2023 AND 2025 ORDER BY f.report_year

问题：2025 年营收同比正增长的公司有哪些
SQL：SELECT c.name, f.total_operate_income_yoy FROM financial_indicators f JOIN companies c ON c.code = f.code WHERE f.report_year = 2025 AND f.total_operate_income_yoy > 0 ORDER BY f.total_operate_income_yoy DESC LIMIT 200

问题：招商中证白酒基金持有哪几家白酒公司，比例多少
SQL：SELECT c.name, h.hold_ratio FROM holders h JOIN companies c ON c.code = h.code WHERE h.holder_name LIKE '%招商中证白酒%' ORDER BY h.hold_ratio DESC LIMIT 200

问题：哪家公司 ROE 连续三年下降
SQL：SELECT c.name FROM financial_indicators f23 JOIN companies c ON c.code = f23.code JOIN financial_indicators f24 ON f24.code = f23.code AND f24.report_year = 2024 JOIN financial_indicators f25 ON f25.code = f23.code AND f25.report_year = 2025 WHERE f23.report_year = 2023 AND f23.roe_weighted > f24.roe_weighted AND f24.roe_weighted > f25.roe_weighted LIMIT 200

## 用户问题
{question}
{feedback}
## SQL
"""


def _extract_sql(text):
    text = (text or "").strip()
    if not text:
        return ""
    if text.upper().startswith("NO_SQL"):
        raise NoSQLAnswer("模型判断现有数据表无法回答该问题")
    # 去掉 markdown 代码块
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    # 只取第一条 SELECT
    m = re.search(r"(SELECT\b[\s\S]+)", text, flags=re.I)
    if m:
        text = m.group(1)
    return text.strip().rstrip(";").strip()


class SQLGenerator:
    def __init__(self, llm=None):
        self.llm = llm or LLMClient()

    def generate(self, question, error_feedback=None):
        feedback = ""
        if error_feedback:
            feedback = ("\n## 上一次生成的 SQL 有问题，请修正\n错误信息：%s\n"
                        "请重新生成一条合法的 SELECT。\n" % error_feedback)
        prompt = PROMPT_TEMPLATE.format(schema=SCHEMA_TEXT, question=question, feedback=feedback)
        raw = self.llm.chat(prompt, temperature=0.0, max_tokens=800)
        return _extract_sql(raw)

    def summarize(self, question, sql, table_markdown):
        """把查询结果转成自然语言回答。"""
        if not table_markdown:
            return "未查询到符合条件的数据。"
        prompt = (
            "请根据下面的问题和 SQL 查询结果，用中文直接给出**关键答案**（1-2 句，最多 3 句），"
            "把核心数字说清楚（单位换算成亿元/百分比时保留 2 位小数），"
            "只使用查询结果里的数据，不要编造；不要给投资建议；不要输出表格或 SQL。\n\n"
            "问题：%s\nSQL：%s\n查询结果（Markdown 表格）：\n%s\n\n回答：" % (
                question, sql, table_markdown[:6000])
        )
        try:
            answer = self.llm.chat(prompt, temperature=0.2, max_tokens=800)
            if answer:
                return answer
            # 模型偶尔会返回空内容（例如推理 token 占满预算），用确定性兜底
            return self._fallback_answer(table_markdown) or "查询完成，但未生成自然语言总结。"
        except Exception as exc:  # noqa: BLE001
            return "查询完成，但生成自然语言总结失败（%s）。" % exc

    @staticmethod
    def _fallback_answer(table_markdown):
        """从 Markdown 表格里提取前几行，拼一句确定性回答。"""
        lines = [ln.strip() for ln in (table_markdown or "").splitlines() if ln.strip().startswith("|")]
        if len(lines) < 3:
            return ""
        header = [c.strip() for c in lines[0].strip("|").split("|")]
        rows = []
        for line in lines[2:5]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == len(header):
                rows.append("，".join(f"{h}={v}" for h, v in zip(header, cells)))
        if not rows:
            return ""
        return "；".join(rows) + "。"


if __name__ == "__main__":
    gen = SQLGenerator()
    print(gen.generate("2025年营业收入最高的3家公司"))
