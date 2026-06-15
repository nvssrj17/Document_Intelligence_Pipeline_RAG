from parser import PDFParser
from chunker import DocumentChunker
from embeddings import EmbeddingGenerator
from vectorstore import VectorStore

PDF_PATH = '/Users/nvssrj/Documents/Document_Intelligence_Pipeline_RAG/Data/MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf'

EMBEDDING_BATCH_SIZE = 2000

print("Loading PDF...")
pages = PDFParser(PDF_PATH).parse()

print("Chunking document...")
chunks = DocumentChunker().create_chunks(pages)

print(f"Created {len(chunks)} chunks.")
with open("db/chunks.json", "w", encoding="utf-8") as f:
    import json
    json.dump(chunks, f, ensure_ascii=False)
    print("Chunks saved to db/chunks.json")
texts = [chunk['text'] for chunk in chunks]

print("Generating embeddings...")
embedder = EmbeddingGenerator()
vector_store = VectorStore()
total_chunks = len(chunks)
for start_idx in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
    end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, total_chunks)
    batch_chunks = chunks[start_idx:end_idx]
    texts = [chunk['text'] for chunk in batch_chunks]
    print(f"Embedding chunks {start_idx} to {end_idx}...")
    embeddings = embedder.generate_batch_embeddings(texts)
    vector_store.add_documents(batch_chunks, embeddings)

print("Indexing Complete")



