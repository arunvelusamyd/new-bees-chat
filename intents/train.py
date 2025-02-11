import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load intents
with open('intents/intents.json') as file:
    data = json.load(file)

# Prepare training data
sentences = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        tags.append(intent['tag'])

# Train a model
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(sentences, tags)

# Save the model
with open('intents/model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
