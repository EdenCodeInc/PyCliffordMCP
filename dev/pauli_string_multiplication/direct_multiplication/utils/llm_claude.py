import anthropic
from typing import Tuple, Dict

def query_llm(prompt: str, model: str, api_key: str, temperature: float = 0.0, max_output_tokens: int = None) -> Tuple[str, Dict]:
    """
    Query the Claude model with configurable parameters.
    
    Args:
        prompt (str): The input prompt
        model (str): The model name (e.g., 'claude-3-sonnet-20240229')
        api_key (str): Your Anthropic API key
        temperature (float): Controls randomness (0.0 to 1.0)
        max_output_tokens (int): Maximum output length (None for default)
    
    Returns:
        Tuple of (response_text, token_metadata_dict)
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # Set max_tokens with fallback
    max_tokens = max_output_tokens if max_output_tokens is not None else 10240
    
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract token usage metadata
    token_metadata = {
        'input_tokens': None,
        'output_tokens': None,
        'total_tokens': None,
    }
    
    # Claude provides token usage in response.usage
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        token_metadata['input_tokens'] = getattr(usage, 'input_tokens', None)
        token_metadata['output_tokens'] = getattr(usage, 'output_tokens', None)
        # Calculate total if both are available
        if token_metadata['input_tokens'] and token_metadata['output_tokens']:
            token_metadata['total_tokens'] = token_metadata['input_tokens'] + token_metadata['output_tokens']
    
    # For Claude 3, the response is in response.content[0].text
    return response.content[0].text, token_metadata 