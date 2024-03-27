import requests
import numpy as np
from typing import List
from transformers import pipeline
from transformers import RagRetriever, RagTokenizer, RagConfig
import os

class CustomWebRagRetriever(RagRetriever):
    def __init__(self, config: RagConfig, urls: List[str] = None, index_path: str = None):
        super().__init__(config, question_encoder_tokenizer=None, generator_tokenizer=None, index=None)
        self.urls = urls or []
        self.pipeline = pipeline("feature-extraction", model="facebook/bart-large-cnn")
        self.index_path = index_path

    def add_url(self, url: str):
        self.urls.append(url)

    def retrieve_web_documents(self, question: str, n_docs: int) -> List[dict]:
        web_documents = []
        for url in self.urls:
            response = requests.get(url)
            if response.status_code == 200:
                html_text = response.text
                extracted_text = self.pipeline(html_text)
                web_documents.append({"title": url, "text": extracted_text[0]})
        return web_documents

    def init_retrieval(self):
        if not self.index:
            self.index = self.build_index()

    def build_index(self):
        embeddings = []
        for url in self.urls:
            response = requests.get(url)
            if response.status_code == 200:
                html_text = response.text
                extracted_text = self.pipeline(html_text)
                embeddings.append(np.mean(extracted_text, axis=0))

        embeddings = np.array(embeddings)
        index = self.build_faiss_index(embeddings)
        return index

    def build_faiss_index(self, embeddings):
        import faiss
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        return index

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)
        retriever_state = {
            "urls": self.urls,
            "index_path": self.index_path,
        }
        np.savez(os.path.join(save_directory, "retriever_state.npz"), **retriever_state)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        # RagConfig.from_pretrained(model_name_or_path, **kwargs)
        urls = kwargs.pop("urls", [])
        index_path = kwargs.pop("index_path", None)
        retriever = cls(config=config, urls=urls, index_path=index_path)
        retriever.init_retrieval()
        return retriever

    def load_retriever_state(self, state_path):
        retriever_state = np.load(state_path)
        self.urls = retriever_state["urls"]
        self.index_path = retriever_state["index_path"]

# Example usage:
retriever = CustomWebRagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", urls=["https://example.com"])
retriever.add_url("https://anotherexample.com")
retriever.init_retrieval()
retriever.save_pretrained("custom_web_retriever")
loaded_retriever = CustomWebRagRetriever.from_pretrained("custom_web_retriever")
