import openai

def query_llm(prompt: str, model: str, api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
    )
    return response.choices[0].message.content 