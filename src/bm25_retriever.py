import json
from rank_bm25 import BM25Okapi
import re

class BM25Retriever:
    def __init__(self):
        with open("db/chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.documents = [chunk['text'] for chunk in self.chunks]
        self.tokenized_docs = [re.findall(r'\w+', doc.lower()) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query, top_k=50):
        scores = self.bm25.get_scores(re.findall(r'\w+', query.lower()))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]