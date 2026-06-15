from bm25_retriever import BM25Retriever
from vectorstore import VectorStore

class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.vector_store = VectorStore()

    def search(self, query, bm25_k=20, vector_k=20):
        bm25_results = self.bm25.search(query, top_k=bm25_k)
        vector_results = self.vector_store.search(query, top_k=vector_k)
        combined = {}
        for doc, score in vector_results:
            chunk_id = doc['id']
            combined[chunk_id] = {'doc': doc, 'bm25_score': score, 'vector_score': 0}

        for doc, score in bm25_results:
            chunk_id = doc['chunk_id']
            if chunk_id not in combined:
                combined[chunk_id] = {'doc': doc, 'bm25_score': 0, 'vector_score': score}
            else:
                combined[chunk_id]['vector_score'] = score
        return list(combined.values())