import os
from transformers import pipeline
from huggingface_hub import login

# Set your Hugging Face token as an environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face API Token is missing. Set HF_TOKEN as an environment variable.")

login(token=HF_TOKEN)

chatbot = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", token=HF_TOKEN)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chatbot(user_input, max_length=100, do_sample=True)[0]["generated_text"]
    print("Chatbot:", response)
