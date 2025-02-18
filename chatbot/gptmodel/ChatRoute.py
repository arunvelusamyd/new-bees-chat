import os
from flask import Flask, request, jsonify
import openai

openai.api_key = os.getenv("OPEN_AI_API_KEY")
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    client = openai.OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))  # Create a client object

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}]
    )
    return jsonify(response.choices[0].message.to_dict())

if __name__ == "__main__":
    app.run(port=5000)
