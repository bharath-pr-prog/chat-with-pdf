import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, dim=768, index_path="vector_index.faiss"):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)

        # Load existing FAISS index if it exists
        if os.path.exists(index_path):
            print("✅ Loading existing FAISS index...")
            self.load_index()

    def add_to_index(self, vectors, metadata):
        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)

        # Save metadata alongside the index
        with open(self.index_path + ".pkl", "wb") as f:
            pickle.dump(metadata, f)

        self.save_index()

    def search(self, query_vector, top_k=5):
        query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)

        with open(self.index_path + ".pkl", "rb") as f:
            metadata = pickle.load(f)

        results = [metadata[i] for i in indices[0]]
        return results

    def save_index(self):
        faiss.write_index(self.index, self.index_path)

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
