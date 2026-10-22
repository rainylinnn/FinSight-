import os
k = os.getenv("DEEPSEEK_API_KEY", "")
print("resolved_len:", len(k.strip()))
print("base_url:", os.getenv("DEEPSEEK_BASE_URL", ""))
