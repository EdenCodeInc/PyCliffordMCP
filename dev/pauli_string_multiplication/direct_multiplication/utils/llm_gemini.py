import google.generativeai as genai
from typing import Tuple, Optional, Dict

def query_llm(prompt: str, model: str, api_key: str, temperature: float = 0.0, max_output_tokens: int = None) -> Tuple[str, Dict]:
    """
    Query the Gemini model with configurable parameters.
    
    Args:
        prompt (str): The input prompt
        model (str): The model name (e.g., 'gemini-pro')
        api_key (str): Your Google API key
        temperature (float): Controls randomness (0.0 to 1.0)
        max_output_tokens (int): Maximum output length (None for model's maximum)
    
    Returns:
        Tuple of (response_text, token_metadata_dict)
    """
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    
    generation_config = {
        "temperature": temperature,
    }
    
    # Handle max_output_tokens
    if max_output_tokens is not None:
        # Auto-detect thinking models and set appropriate limits
        if "thinking" in model.lower():
            # Thinking models support up to 65K tokens
            generation_config["max_output_tokens"] = min(max_output_tokens, 65536)
        elif "2.5" in model and ("pro" in model.lower() or "exp" in model.lower()):
            # Gemini 2.5 Pro models support up to 65K tokens
            generation_config["max_output_tokens"] = min(max_output_tokens, 65536)
        else:
            # Regular models support up to 8K tokens
            generation_config["max_output_tokens"] = min(max_output_tokens, 8192)
    # If max_output_tokens is None, don't set any limit (use model's maximum)
    
    response = model_obj.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    # Extract token usage metadata
    token_metadata = {
        'input_tokens': None,
        'output_tokens': None,
        'total_tokens': None,
    }
    
    # Try to get token usage from response
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        token_metadata['input_tokens'] = getattr(usage, 'prompt_token_count', None)
        token_metadata['output_tokens'] = getattr(usage, 'candidates_token_count', None) 
        token_metadata['total_tokens'] = getattr(usage, 'total_token_count', None)
    
    return response.text, token_metadata

 