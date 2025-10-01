import anthropic
from typing import Tuple, Dict, Optional

# Model-specific max output token limits (examples; keep in sync with docs)
MODEL_MAX_TOKENS = {
    "claude-opus-4-1-20250805": 32000,
    "claude-opus-4-20250514": 32000,
    "claude-sonnet-4-20250514": 32000,
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
}

# Reserve tokens for final answer when extended thinking is enabled
RESERVED_OUTPUT_TOKENS = 10000

def query_llm(
    prompt: str,
    model: str,
    api_key: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    thinking_budget_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    """
    Minimal Claude caller (non-streaming) for consistent behavior:
    - No thinking blocks are sent
    - Temperature omitted unless provided
    - Moderate default cap if not specified
    - Extracts all text blocks from the response
    """
    client = anthropic.Anthropic(api_key=api_key)

    # cap selection
    model_cap = MODEL_MAX_TOKENS.get(model, 32000)
    max_tokens = max_output_tokens or 10240
    max_tokens = min(max_tokens, model_cap)

    # build request (omit temperature if None; never send thinking)
    req: Dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        req["temperature"] = temperature

    # non-streaming call
    resp = client.messages.create(**req)

    # token usage
    usage = getattr(resp, "usage", None)
    token_metadata: Dict[str, Optional[int]] = {
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "total_tokens": None,
    }
    if token_metadata["input_tokens"] is not None and token_metadata["output_tokens"] is not None:
        token_metadata["total_tokens"] = token_metadata["input_tokens"] + token_metadata["output_tokens"]

    # collect all text blocks (Claude 4.x may include non-text blocks first)
    text_parts = [blk.text for blk in resp.content if getattr(blk, "type", "") == "text"]
    return ("\n".join(text_parts) if text_parts else ""), token_metadata