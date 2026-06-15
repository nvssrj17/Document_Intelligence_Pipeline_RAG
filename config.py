from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf"
OUTPUT_DIR = BASE_DIR / "output"
DB_DIR = BASE_DIR / "db"
LOG_DIR = BASE_DIR / "logs"

#Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

#Embedding Model
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

#Retrieval
TOP_K_RETRIEVAL = 50
TOP_K_RERANK = 10

#Ollama
OLLAMA_MODEL = "gemma3:4b"

#Chroma
COLLECTION_NAME = "documents"
