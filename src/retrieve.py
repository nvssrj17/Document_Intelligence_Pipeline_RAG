import sys
from pathlib import Path

# Ensure both the repo root and src/ are importable when this file is run directly or imported as src.retrieve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embeddings import EmbeddingGenerator
from vectorstore import VectorStore
from bm25_retriever import BM25Retriever
from reranker import DocumentReranker

VECTOR_K = 50
BM25_K = 50
FINAL_K = 10


class Retriever:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self.reranker = DocumentReranker()

    def retrieve(self, query):
        query_embedding = self.embedder.generate_embeddings(query)

        vector_results = self.vector_store.collection.query(query_embedding, n_results=VECTOR_K)
        vector_docs = []
        docs = vector_results['documents'][0]
        metas = vector_results['metadatas'][0]
        for doc, meta in zip(docs, metas):
            vector_docs.append({
                'text': doc,
                'page': meta.get('page'),
                'chunk_id': meta.get('chunk_id'),
            })

        bm25_results = self.bm25.search(query, top_k=BM25_K)
        bm25_docs = []
        for chunk, score in bm25_results:
            bm25_docs.append({
                'text': chunk['text'],
                'page': chunk.get('page'),
                'chunk_id': chunk.get('id'),
            })

        combined = vector_docs + bm25_docs
        unique_docs = {}
        for doc in combined:
            key = f"{doc.get('page')}_{doc.get('chunk_id')}"
            unique_docs[key] = doc

        combined_docs = list(unique_docs.values())
        reranked = self.reranker.rerank(query, combined_docs, top_k=FINAL_K)

        return reranked