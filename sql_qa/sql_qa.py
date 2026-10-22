# -*- coding: utf-8 -*-
"""SQL QA 编排：问题 → SQL → 校验 → 执行 → 自然语言总结。"""
from .sql_generator import NoSQLAnswer, SQLGenerator
from .sql_executor import SQLExecutor, SQLValidationError


class SQLQA:
    def __init__(self, generator=None, executor=None, max_rows=200):
        self.executor = executor or SQLExecutor(max_rows=max_rows)
        self.generator = generator or SQLGenerator()

    def ask(self, question, with_summary=True, max_retries=1):
        """返回 dict：question / sql / columns / rows / table_markdown / answer / error"""
        sql, safe_sql, columns, rows, table = None, None, [], [], ""
        error = None
        for _ in range(max_retries + 1):
            try:
                sql = self.generator.generate(question, error_feedback=error)
            except NoSQLAnswer as exc:
                return self._result(question, None, None, [], [], "",
                                    "这个问题无法用现有财务数据表回答：%s" % exc)
            except Exception as exc:  # noqa: BLE001
                return self._result(question, None, None, [], [], "",
                                    "生成 SQL 失败：%s" % exc)
            try:
                safe_sql = self.executor.validate(sql)
                columns, rows = self.executor.execute(safe_sql)
                table = self.executor.to_markdown(columns, rows)
                break
            except (SQLValidationError, Exception) as exc:  # noqa: BLE001
                error = str(exc)
                safe_sql = None
                continue

        if safe_sql is None:
            return self._result(question, sql, None, [], [], "",
                                "SQL 校验/执行失败：%s" % error)

        if with_summary:
            answer = self.generator.summarize(question, safe_sql, table)
        else:
            answer = "查询完成，共返回 %d 行。" % len(rows)
        return self._result(question, sql, safe_sql, columns, rows, table, None, answer)

    @staticmethod
    def _result(question, sql, safe_sql, columns, rows, table, error, answer=None):
        return {
            "question": question,
            "sql": sql,
            "safe_sql": safe_sql,
            "columns": columns,
            "rows": rows,
            "table_markdown": table,
            "answer": answer if answer is not None else error,
            "error": error,
        }

    def answer_text(self, question):
        """给对话/接口用的文本：自然语言回答 + 表格。"""
        res = self.ask(question)
        text = res["answer"]
        if res.get("table_markdown"):
            text = "%s\n\n%s" % (text, res["table_markdown"])
        if res.get("safe_sql"):
            text += "\n\n> SQL: `%s`" % res["safe_sql"]
        return text


if __name__ == "__main__":
    qa = SQLQA()
    for q in ["2025年营业收入最高的3家公司",
              "贵州茅台2023到2025年的毛利率和ROE",
              "2025年营收同比正增长的公司有哪些"]:
        print("\n" + "=" * 80)
        print("Q:", q)
        r = qa.ask(q)
        print("SQL:", r["safe_sql"])
        print("ERR:", r["error"])
        print(r["answer"])
