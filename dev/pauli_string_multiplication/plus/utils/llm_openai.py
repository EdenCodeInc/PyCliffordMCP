import openai
from typing import Tuple, Dict

def query_llm(prompt: str, model: str, api_key: str, temperature: float = None, max_output_tokens: int = None) -> Tuple[str, Dict]:
    """
    Query the OpenAI model with configurable parameters.
    
    Args:
        prompt (str): The input prompt
        model (str): The model name (e.g., 'gpt-5')
        api_key (str): Your OpenAI API key
        temperature (float): Optional randomness control; omit to use model default
        max_output_tokens (int): Optional max completion tokens (if supported by model)
    
    Returns:
        Tuple of (response_text, token_metadata_dict)
    """
    client = openai.OpenAI(api_key=api_key)
    
    # Build request kwargs; use max_completion_tokens if provided
    req_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        req_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        req_kwargs["max_completion_tokens"] = max_output_tokens
    
    response = client.chat.completions.create(**req_kwargs)
    
    # Extract token usage metadata
    token_metadata = {
        'input_tokens': None,
        'output_tokens': None,
        'total_tokens': None,
    }
    
    # OpenAI provides token usage in response.usage
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        token_metadata['input_tokens'] = getattr(usage, 'prompt_tokens', None)
        token_metadata['output_tokens'] = getattr(usage, 'completion_tokens', None) 
        token_metadata['total_tokens'] = getattr(usage, 'total_tokens', None)
    
    return response.choices[0].message.content, token_metadata 