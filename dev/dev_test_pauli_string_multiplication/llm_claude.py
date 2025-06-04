import anthropic

def query_llm(prompt: str, model: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=10240,
        messages=[{"role": "user", "content": prompt}]
    )
    # For Claude 3, the response is in response.content[0].text
    return response.content[0].text 