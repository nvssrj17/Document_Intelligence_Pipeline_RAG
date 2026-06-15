# Document Intelligence Pipeline with RAG (PDF to Structured Data)

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that processes large PDF documents, indexes their contents using semantic embeddings, retrieves relevant context using hybrid search, and generates accurate answers using a local Large Language Model (LLM).

This project demonstrates how to build a scalable document intelligence system capable of handling large document corpora (6,000+ pages) with semantic search, reranking, evaluation, and structured data extraction.

---

## Project Overview

Organizations often store critical information inside lengthy PDF documents, making information retrieval time-consuming and inefficient.

This project addresses that challenge by:

- Parsing and processing large PDF documents
- Chunking documents into semantically meaningful segments
- Generating embeddings using transformer models
- Indexing embeddings in ChromaDB
- Performing Hybrid Retrieval (BM25 + Vector Search)
- Reranking retrieved results using a Cross-Encoder
- Generating answers with Gemma 3 4B via Ollama
- Evaluating answer quality using semantic similarity metrics

---

## Architecture

```text
PDF Document
      │
      ▼
Document Parser
      │
      ▼
Chunking
(512 tokens + overlap)
      │
      ▼
Embedding Generation
(BGE Base v1.5)
      │
      ▼
ChromaDB Vector Store
      │
      ▼
 ┌───────────────┐
 │ Hybrid Search │
 └───────────────┘
      │
      ├── BM25 Retrieval
      │
      └── Vector Retrieval
              │
              ▼
      Cross Encoder Reranker
      (BGE Reranker Base)
              │
              ▼
      Top-K Context
              │
              ▼
       Gemma 3 4B
         (Ollama)
              │
              ▼
      Generated Answer
```

---

## Features

### Document Processing

- Large PDF ingestion
- Metadata extraction
- Intelligent chunking
- Page-level traceability

### Semantic Search

- Transformer-based embeddings
- ChromaDB vector indexing
- Fast retrieval over 37,000+ chunks

### Hybrid Retrieval

Combines:

- BM25 keyword search
- Dense vector similarity search

Improves retrieval accuracy compared to using either method alone.

### Cross-Encoder Reranking

Uses:

```python
BAAI/bge-reranker-base
```

to rerank retrieved passages and improve context quality before generation.

### Local LLM Generation

Uses:

```python
Gemma 3 4B
```

running locally through:

```python
Ollama
```

No paid API required.

### Evaluation Framework

Evaluates:

- Semantic similarity
- Response latency
- Retrieval effectiveness

---

## Tech Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| LLM | Gemma 3 4B |
| LLM Runtime | Ollama |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| PDF Processing | PyMuPDF |
| Evaluation | Sentence Transformers + Cosine Similarity |

---

## Dataset

### Document

Mahabharata (English Translation by K. M. Ganguli)

### Corpus Statistics

| Metric | Value |
|----------|----------|
| Pages | ~6,000 |
| Chunks Indexed | ~37,771 |
| Chunk Size | 512 |
| Chunk Overlap | 128 |
| Embedding Dimensions | 768 |

---

## Project Structure

```text
Document_Intelligence_Pipeline_RAG/
│
├── data/
│   └── Mahabharata.pdf
│
├── db/
│   ├── chroma_db/
│   └── chunks.json
│
├── evaluation/
│   ├── questions.json
│   ├── results.json
│   └── evaluate_answers.py
│
├── output/
│
├── src/
│   ├── config.py
│   ├── document_parser.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── bm25_retriever.py
│   ├── retrieve.py
│   ├── reranker.py
│   ├── rag_pipeline.py
│   ├── index_documents.py
│   └── test_retrieval.py
│
├── requirements.txt
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/document-intelligence-rag.git

cd document-intelligence-rag
```

### Create Virtual Environment

```bash
python -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Install Ollama:

### macOS

```bash
brew install ollama
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull Gemma 3:

```bash
ollama pull gemma3:4b
```

Verify:

```bash
ollama run gemma3:4b
```

---

## Index Documents

Place PDF inside:

```text
data/
```

Run:

```bash
python src/index_documents.py
```

Pipeline:

```text
PDF
→ Chunking
→ Embeddings
→ ChromaDB Index
```

---

## Query the System

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

response = rag.answer_query(
    "Why did Surya warn Karna?"
)

print(response["answer"])
```

---

## Run Evaluation

```bash
python evaluation/evaluate_answers.py
```

Evaluation output:

```json
{
  "average_similarity": 0.71,
  "average_generation_time": 2.1
}
```

---

## Evaluation Results

### Metrics

| Metric | Value |
|----------|---------|
| Evaluation Questions | 25 |
| Average Semantic Similarity | 0.71 |
| Retrieval Strategy | Hybrid Search |
| Embedding Model | BGE Base v1.5 |
| Reranker | BGE Reranker Base |
| LLM | Gemma 3 4B |

### Evaluation Methodology

Generated answers are compared against reference answers using:

```python
Cosine Similarity
```

over embeddings produced by:

```python
BAAI/bge-base-en-v1.5
```

---

## Example Query

### Question

```text
Why did Surya advise Karna not to give away his celestial earrings?
```

### Retrieved Context

Relevant passages retrieved through:

- BM25 Search
- Vector Search
- Cross-Encoder Reranking

### Generated Answer

```text
Surya warned Karna that giving away his celestial
earrings would remove his divine protection and
make him vulnerable in future battles, particularly
against Arjuna.
```

---

## Future Improvements

- Streamlit Frontend
- Multi-PDF Support
- Query Expansion
- Context Compression
- Citation Generation
- Knowledge Graph Integration
- Agentic RAG
- Evaluation using LLM-as-a-Judge

---

## Resume Highlights

- Built an end-to-end RAG pipeline for large-scale PDF document intelligence.
- Indexed 37,000+ document chunks using transformer embeddings and ChromaDB.
- Implemented Hybrid Retrieval (BM25 + Vector Search) with Cross-Encoder reranking.
- Integrated Gemma 3 4B via Ollama for fully local answer generation.
- Developed an automated evaluation framework achieving 0.71 semantic similarity on a 6,000-page corpus.

---

## Author

Venkata Sesha Sai Raj Nanduri

M.S. Artificial Intelligence  
University of Michigan – Dearborn

