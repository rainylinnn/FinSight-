# -*- coding: utf-8 -*-
"""统一的非流式 LLM 调用封装（OpenAI 兼容接口，当前为 DeepSeek）。"""
import sys

from openai import OpenAI
import httpx

sys.path.insert(0, r"C:\pro")
from base import Config, logger  # noqa: E402


class LLMClient:
    def __init__(self, model=None):
        conf = Config()
        if not conf.LLM_API_KEY:
            raise RuntimeError(
                "未配置 LLM API Key：请设置系统环境变量 DEEPSEEK_API_KEY，"
                "或在 C:\\pro\\config.ini 的 [llm] api_key 填写")
        self.model = model or conf.LLM_MODEL
        # trust_env=False：绕过 Windows 系统代理，避免 TLS 握手失败
        self.client = OpenAI(api_key=conf.LLM_API_KEY,
                             base_url=conf.LLM_BASE_URL,
                             http_client=httpx.Client(trust_env=False))

    def chat(self, prompt, temperature=0.0, max_tokens=1024, system="你是一个严谨的金融数据助手。"):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60,
        )
        if not completion.choices:
            return ""
        return (completion.choices[0].message.content or "").strip()


if __name__ == "__main__":
    print(LLMClient().chat("只回复两个字：成功"))
