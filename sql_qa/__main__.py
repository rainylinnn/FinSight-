# -*- coding: utf-8 -*-
"""SQL QA 命令行入口。"""
import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, r"C:\pro")
from sql_qa import SQLExecutor, SQLQA  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="FinSight NL2SQL 命令行")
    parser.add_argument("question", nargs="?", default="", help="中文问题")
    parser.add_argument("--sql", default="", help="直接执行一条 SQL（不调用 LLM）")
    parser.add_argument("--no-summary", action="store_true", help="不生成自然语言总结")
    args = parser.parse_args()

    if args.sql:
        ex = SQLExecutor()
        try:
            safe = ex.validate(args.sql)
            columns, rows = ex.execute(safe)
        except Exception as exc:  # noqa: BLE001
            print("SQL 被拒绝或执行失败:", exc)
            return 2
        print("SQL:", safe)
        print("行数:", len(rows))
        print(ex.to_markdown(columns, rows))
        return 0

    if not args.question:
        parser.print_help()
        return 1

    qa = SQLQA()
    result = qa.ask(args.question, with_summary=not args.no_summary)
    print("=" * 80)
    print("问题:", result["question"])
    print("SQL :", result["safe_sql"] or result["sql"])
    if result["error"]:
        print("错误:", result["error"])
    print("-" * 80)
    print(result["answer"])
    if result["table_markdown"]:
        print()
        print(result["table_markdown"])
    return 0 if not result["error"] else 2


if __name__ == "__main__":
    sys.exit(main())
