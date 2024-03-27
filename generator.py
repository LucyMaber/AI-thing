import faiss
import json
import numpy as np

class FaissDatabase:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.urls = []

    def add_embeddings(self, embeddings, urls):
        if len(embeddings) != len(urls):
            raise ValueError("Number of embeddings must be equal to the number of URLs")
        
        # Convert embeddings to a numpy array
        embeddings_array = np.array(embeddings)

        # Add embeddings to Faiss index
        self.index.add(embeddings_array)
        self.urls.extend(urls)

    def search(self, query_embedding, k=5):
        _, result_indices = self.index.search(np.array([query_embedding]), k)
        return [self.urls[idx] for idx in result_indices[0]]

    def remove_url(self, url):
        if url in self.urls:
            idx = self.urls.index(url)
            self.urls.pop(idx)
            self.index.remove_ids(np.array([idx]))
        else:
            print(f"URL '{url}' not found in the database.")

    def save_to_json(self, filename):
        data = {"embedding_dim": self.embedding_dim, "urls": self.urls}
        faiss.write_index(self.index, filename + ".faiss")
        with open(filename + ".json", "w") as json_file:
            json.dump(data, json_file)

    @classmethod
    def load_from_json(cls, filename):
        with open(filename + ".json", "r") as json_file:
            data = json.load(json_file)
        instance = cls(data["embedding_dim"])
        instance.urls = data["urls"]
        instance.index = faiss.read_index(filename + ".faiss")
        return instance

fff = FaissDatabase(512)
print(fff)