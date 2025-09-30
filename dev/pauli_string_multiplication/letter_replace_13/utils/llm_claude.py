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
    temperature: Optional[float] = 0.0,
    max_output_tokens: Optional[int] = None,
    thinking_budget_tokens: Optional[int] = None,  # set (>=1024) only if you explicitly want extended thinking
) -> Tuple[str, Dict]:
    """
    Query a Claude model in streaming-only mode (max reliability for long generations).
    - High cap by default via MODEL_MAX_TOKENS; pass max_output_tokens to override.
    - 'thinking' omitted unless thinking_budget_tokens is provided (>=1024).
    """
    client = anthropic.Anthropic(api_key=api_key)

    # pick model max if not provided
    model_cap = MODEL_MAX_TOKENS.get(model, 32000)
    max_tokens = max_output_tokens or model_cap

    # Optional extended thinking (only if explicitly requested)
    thinking = None
    if thinking_budget_tokens is not None:
        if thinking_budget_tokens < 1024:
            raise ValueError("thinking_budget_tokens must be at least 1024.")
        # Ensure at least RESERVED_OUTPUT_TOKENS remain for the final answer when possible
        min_needed = thinking_budget_tokens + RESERVED_OUTPUT_TOKENS
        if min_needed <= model_cap:
            # Raise max_tokens to satisfy the reserve if we can
            if max_tokens < min_needed:
                max_tokens = min_needed
        else:
            # Can't satisfy full reserve; cap at model_cap
            max_tokens = model_cap
        if thinking_budget_tokens >= max_tokens:
            raise ValueError("thinking_budget_tokens must be smaller than max_tokens/model cap.")
        thinking = {"type": "enabled", "budget_tokens": thinking_budget_tokens}

    # Build request params; only include fields that should be sent
    req: Dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        req["temperature"] = temperature
    if thinking is not None:
        req["thinking"] = thinking

    # Streaming-only: accumulate text deltas until completion
    text_out: list[str] = []
    token_metadata: Dict[str, Optional[int]] = {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    stream_iter = client.with_options(timeout=1800.0).messages.create(stream=True, **req)
    try:
        for event in stream_iter:
            etype = getattr(event, "type", None)
            if etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta is not None and getattr(delta, "type", None) == "text_delta":
                    txt = getattr(delta, "text", "")
                    if txt:
                        text_out.append(txt)
            elif etype == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    token_metadata["input_tokens"] = getattr(usage, "input_tokens", token_metadata["input_tokens"])  # type: ignore
                    token_metadata["output_tokens"] = getattr(usage, "output_tokens", token_metadata["output_tokens"])  # type: ignore
    finally:
        try:
            close = getattr(stream_iter, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    if token_metadata.get("input_tokens") is not None and token_metadata.get("output_tokens") is not None:
        token_metadata["total_tokens"] = token_metadata["input_tokens"] + token_metadata["output_tokens"]  # type: ignore

    return ("".join(text_out), token_metadata)