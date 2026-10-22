# -*- coding: utf-8 -*-
"""快速验证：新进程里 Config 解析 key + OpenAI 客户端初始化。"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

out = open(r"C:\pro\.llm_init_result.txt", "w", encoding="utf-8")
try:
    from base import Config
    from openai import OpenAI

    conf = Config()
    out.write("LLM_API_KEY resolved: %s len: %s source: %s\n" % (
        conf.LLM_API_KEY is not None, len(conf.LLM_API_KEY or ""), conf.LLM_API_KEY_SOURCE))
    out.write("LLM_BASE_URL: %s\n" % conf.LLM_BASE_URL)
    client = OpenAI(api_key=conf.LLM_API_KEY, base_url=conf.LLM_BASE_URL)
    out.write("OpenAI client init OK\n")
except Exception:
    out.write(traceback.format_exc())
finally:
    out.close()
