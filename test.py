from openai import OpenAI

client = OpenAI(api_key="ta_clé_openai")


response = client.completions.create(engine="gpt-4",  # ou "gpt-3.5-turbo"
prompt="Bonjour, comment ça va ?",
max_tokens=150)

print(response.choices[0].text)