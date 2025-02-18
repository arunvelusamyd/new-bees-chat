import os
from openai import OpenAI

client = OpenAI(
  api_key=os.getenv("OPEN_AI_API_KEY")
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "Hello, chatbot"}
  ]
)

print(completion.choices[0].message)
