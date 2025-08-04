import openai
from typing import Tuple, Dict

def query_llm(prompt: str, model: str, api_key: str, temperature: float = 0.0, max_output_tokens: int = None) -> Tuple[str, Dict]:
    """
    Query the OpenAI model with configurable parameters.
    
    Args:
        prompt (str): The input prompt
        model (str): The model name (e.g., 'gpt-4')
        api_key (str): Your OpenAI API key
        temperature (float): Controls randomness (0.0 to 1.0)
        max_output_tokens (int): Maximum output length (None for model's default)
    
    Returns:
        Tuple of (response_text, token_metadata_dict)
    """
    client = openai.OpenAI(api_key=api_key)
    
    # Set max_tokens with fallback
    max_tokens = max_output_tokens if max_output_tokens is not None else 4096
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
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