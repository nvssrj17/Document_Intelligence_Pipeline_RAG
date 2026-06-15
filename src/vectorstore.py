import sys
from pathlib import Path

# When running this file directly, ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from config import DB_DIR, COLLECTION_NAME
from tqdm import tqdm

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def add_documents(self, chunks, embeddings):
        total = len(chunks)
        for i in tqdm(range(0, total), desc="Indexing Chunks"):
            batch_chunks = chunks[i:i+1]
            batch_embeddings = embeddings[i:i+1]
            ids = [chunk['id'] for chunk in batch_chunks]
            docs = [chunk['text'] for chunk in batch_chunks]
            metadata = [{'page': chunk['page'], 'chunk_id': chunk['id']} for chunk in batch_chunks]
            self.collection.add(ids=ids, documents=docs, metadatas=metadata, embeddings=batch_embeddings)

    def similarity_search(self, query_embedding, k=50):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return results