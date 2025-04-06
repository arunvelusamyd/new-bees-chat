from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load an open-source embedding model
embedding_model = SentenceTransformer("BAAI/bge-base-en")

# Sample internal documents
documents = {
    "deployment_strategy": "Our deployment uses Kubernetes with GitLab CI/CD for automated builds.",
    "codebase_structure": "The internal codebase follows a microservices architecture with Spring Boot and FastAPI.",
    "branch_details": "We have three main branches: dev, staging, and production. Merges follow GitFlow strategy.",
    "project_requirements": "Each project requires an architecture review and a security audit before production release."
}

# Convert document texts into embeddings
doc_texts = list(documents.values())
doc_embeddings = embedding_model.encode(doc_texts, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # L2 distance for similarity search
index.add(doc_embeddings)  # Add embeddings to the FAISS index

# Store doc_texts for reference
doc_id_map = {i: doc_texts[i] for i in range(len(doc_texts))}

# Request model
class QueryRequest(BaseModel):
    question: str

# Function to retrieve relevant document
def retrieve_document(query):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, 1)  # Retrieve top 1 similar doc
    return doc_id_map[I[0][0]]

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = retrieve_document(request.question)
    return {"answer": answer}

