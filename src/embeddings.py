import sys
from pathlib import Path

# When running this file directly, ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def generate_embeddings(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    def generate_batch_embeddings(self, texts, batch_size=64):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True).tolist()