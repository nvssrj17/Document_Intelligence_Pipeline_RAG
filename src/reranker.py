from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self, model_name='BAAI/bge-reranker-base'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=15):
        # Create pairs of (query, document) for scoring
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get relevance scores for each pair
        scores = self.model.predict(pairs)
        
        # Combine documents with their scores and sort by score
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        # Return the sorted documents
        return ranked[:top_k]