import google.generativeai as genai

def query_llm(prompt: str, model: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)
    return response.text 