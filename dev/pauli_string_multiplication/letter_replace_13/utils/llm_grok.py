import os
import requests
from typing import Tuple, Dict, Optional

# Minimal Grok (xAI) client using OpenAI-compatible Chat Completions API
# Docs: https://docs.x.ai/

API_BASE = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")


def query_llm(
    prompt: str,
    model: str,
    api_key: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens

    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    # Extract text
    content = ""
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except Exception:
        content = ""

    # Token usage (OpenAI-compatible fields)
    usage = data.get("usage") or {}
    token_metadata = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

    return content, token_metadata 