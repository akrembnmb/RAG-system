from openai import OpenAI
from config.settings import Setting
from models.model_loader import TEMPERATURE,MAX_TOKENS,PROMPT,MODEL

client = OpenAI(api_key=Setting.OPENAI_API_KEY)
def create_completion(input : str):
    response = client.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        prompt=PROMPT+input
    )
    return response.choices[0].text
