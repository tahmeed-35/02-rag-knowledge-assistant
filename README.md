# RAG-Based Technical Knowledge Assistant

A production-grade Retrieval-Augmented Generation (RAG) assistant designed for technical documentation. Ingests PDF and Markdown documents, stores semantic chunks in ChromaDB with cosine similarity, and serves answers via FastAPI with source citations and strict refusal thresholds for out-of-domain queries.

## Key Features

- **Multi-Format Ingestion**: Ingests Markdown, plain text, and PDF files with header-aware parsing and configurable chunk size/overlap.
- **ChromaDB Vector Store**: Persistent local vector database with cosine metric for semantic retrieval.
- **FastAPI Endpoints**:
  - `GET /health`: Health checks and total indexed chunk count.
  - `POST /ingest`: Ingest individual files or trigger batch indexing.
  - `POST /ask`: Query knowledge base with inline citations and similarity scoring.
- **Strict Confidence Threshold Refusal**: Automatically refuses queries with a structured explanation if confidence falls below the configured threshold (default `0.60`).
- **Retrieval Evaluation Benchmark**: Included evaluation script measuring Hit@K, MRR, and out-of-domain refusal accuracy.
- **CI / CD Pipeline**: GitHub Actions workflow for automated testing.

## Directory Structure

```
02-rag-knowledge-assistant/
├── .github/workflows/
│   └── ci.yml               # Automated CI test pipeline configuration
├── data/
│   └── docs/                # Knowledge base documents (Markdown/PDF)
│       └── architecture_guide.md
├── eval/
│   └── eval_retrieval.py    # Retrieval evaluation benchmark script
├── src/
│   ├── __init__.py
│   ├── app.py               # FastAPI application
│   ├── config.py            # Pydantic environment configurations
│   ├── ingest.py            # Document chunking & ChromaDB ingestion engine
│   ├── retriever.py         # Semantic search, scoring & refusal logic
│   └── schemas.py           # API request and response schemas
├── tests/
│   ├── test_api.py          # API route & schema tests
│   └── test_retriever.py    # Chunking & retrieval unit tests
├── .env.example             # Environment variable template
├── .gitignore
├── requirements.txt         # Project dependencies
└── README.md
```

## Quick Start

### 1. Installation

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the API Server

```bash
uvicorn src.app:app --reload --port 8000
```
Interactive Swagger API documentation is available at: `http://localhost:8000/docs`

### 3. Ingest Documentation

```bash
curl -X POST "http://localhost:8000/ingest" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "data/docs/architecture_guide.md"}'
```

### 4. Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What algorithm is used for rate limiting?"}'
```

### 5. Run the Retrieval Evaluation

```bash
python eval/eval_retrieval.py
```
