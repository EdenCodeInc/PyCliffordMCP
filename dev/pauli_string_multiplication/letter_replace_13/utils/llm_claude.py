import anthropic
from typing import Tuple, Dict, Optional

# Model-specific max output token limits (examples; keep in sync with docs)
MODEL_MAX_TOKENS = {
    "claude-opus-4-20250514": 32000,
    "claude-sonnet-4-20250514": 32000,
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
}

def query_llm(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
    thinking_budget_tokens: Optional[int] = 8192,  # ← set e.g. 8192 to enable extended thinking
    stream: bool = False,
) -> Tuple[str, Dict]:
    """
    Query a Claude model with optional extended thinking.

    Args:
        thinking_budget_tokens: if set >= 1024, enables extended thinking and
            allocates this many tokens for internal reasoning (counts against max_tokens).
    """
    client = anthropic.Anthropic(api_key=api_key)

    # pick model max if not provided
    model_cap = MODEL_MAX_TOKENS.get(model, 4096)
    max_tokens = max_output_tokens or model_cap

    # If using thinking, ensure budget < max_tokens
    thinking = None
    if thinking_budget_tokens is not None:
        if thinking_budget_tokens < 1024:
            raise ValueError("thinking_budget_tokens must be at least 1024.")
        if thinking_budget_tokens >= max_tokens:
            # leave room for the final answer
            max_tokens = thinking_budget_tokens + 1024
            # but never exceed model cap
            max_tokens = min(max_tokens, model_cap)
            if thinking_budget_tokens >= max_tokens:
                raise ValueError("thinking_budget_tokens must be smaller than max_tokens/model cap.")
        thinking = {"type": "enabled", "budget_tokens": thinking_budget_tokens}

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,            # ← enables extended thinking when not None
        stream=stream,                # you can stream `thinking_delta` and `text_delta`
        messages=[{"role": "user", "content": prompt}],
    )

    # token usage
    usage = getattr(resp, "usage", None)
    token_metadata = {
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "total_tokens": None,
    }
    if token_metadata["input_tokens"] is not None and token_metadata["output_tokens"] is not None:
        token_metadata["total_tokens"] = token_metadata["input_tokens"] + token_metadata["output_tokens"]

    # For non-streaming, text lives in the content blocks. You may also see 'thinking' blocks.
    text_parts = [blk.text for blk in resp.content if getattr(blk, "type", "") == "text"]
    return ("\n".join(text_parts) if text_parts else ""), token_metadata