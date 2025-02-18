import os
import openai

openai.api_key = os.getenv("OPEN_AI_API_KEY")

def chat_with_gpt(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}]
    )
    return response["choices"][0]["message"]["content"]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chat_with_gpt(user_input)
    print("Chatbot:", response)
