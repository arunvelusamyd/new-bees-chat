from flask import Flask, request, jsonify, render_template
import pickle
import random
import json

app = Flask(__name__)

# Load model and intents
with open('intents/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('intents/intents.json') as f:
    intents = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def chatbot_response():
    user_message = request.json.get('message')
    tag = model.predict([user_message])[0]

    # Fetch response
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return jsonify({"response": response})

    return jsonify({"response": "Sorry, I didn't understand that."})

if __name__ == '__main__':
    app.run(debug=True)
