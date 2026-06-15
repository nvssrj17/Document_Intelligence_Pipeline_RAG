from src.parser import PDFParser
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingGenerator

pdf = PDFParser("data/MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf")
pages = pdf.parse()
print(f"Extracted {len(pages)} pages.")

chunker = DocumentChunker()
chunks = chunker.create_chunks(pages)
print(f"Created {len(chunks)} chunks.")

embedder = EmbeddingGenerator()
vector = embedder.generate_embeddings(chunks[0]["text"])
print(f"Generated embedding of length {len(vector)} for the first chunk.")