import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunking PDF text into smaller pieces
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Create embeddings and store in FAISS
def create_and_store_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

# Retrieve relevant chunks for a question
def retrieve_relevant_chunks(question, index, chunks, top_k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    return [chunks[i] for i in indices[0]]
