import os
from huggingface_hub import login
from huggingface_hub import whoami

# Replace with your actual Hugging Face token
login(os.getenv("HF_TOKEN"))
print(whoami())
