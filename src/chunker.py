import sys
from pathlib import Path

# When running this file directly, ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentChunker:
    def __init__(self):

        self.splitter = (RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            separators = ["\n\n", "\n", ". ", ""]))

    def create_chunks(self, pages):
        chunks = []
        chunk_id = 0
        for page in pages:
            text = page["text"]
            split_chunks = self.splitter.split_text(text)
            for chunk in split_chunks:
                chunks.append(
                    {
                        "id": str(chunk_id),
                        "page": page["page"],
                        "text": chunk
                    }
                )
                chunk_id += 1
        return chunks