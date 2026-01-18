# Documentation RAG System - Quick Start Guide

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized setup)
- Ollama (for local LLM execution)

## Installation

### Option 1: Local Development

1. **Clone the repository:**
```bash
git clone <repository-url>
cd LLM-Knowledge-Base
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Start Qdrant:**
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant
```

6. **Pull Ollama models:**
```bash
ollama pull deepseek-r1:14b
ollama pull nomic-embed-text:v1.5
```

7. **Run the application:**
```bash
python -m uvicorn src.api.app:app --reload
```

### Option 2: Docker Compose

1. **Clone the repository:**
```bash
git clone <repository-url>
cd LLM-Knowledge-Base
```

2. **Configure environment:**
```bash
cp .env.example .env
```

3. **Start all services:**
```bash
docker-compose up -d
```

4. **Pull Ollama models (in the container):**
```bash
docker exec -it ollama ollama pull deepseek-r1:14b
docker exec -it ollama ollama pull nomic-embed-text:v1.5
```

## Usage

### API Endpoints

The API will be available at `http://localhost:8000`. Access the interactive documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### CLI Tool

Install CLI dependencies:
```bash
pip install typer rich
```

**Check service health:**
```bash
python cli.py health
```

**Ingest a document:**
```bash
python cli.py ingest-file /path/to/document.pdf
```

**Ingest from URL:**
```bash
python cli.py ingest-url https://example.com/article
```

**Query the knowledge base:**
```bash
python cli.py query "What is the main topic of the documents?"
```

**Summarize text:**
```bash
python cli.py summarize --file-path document.txt --format paragraph --length medium
```

**Extract key points:**
```bash
python cli.py key-points --file-path document.txt --num-points 5
```

**View statistics:**
```bash
python cli.py stats
```

### Python API Examples

```python
import httpx
import asyncio

async def example():
    async with httpx.AsyncClient() as client:
        # Ingest text
        response = await client.post(
            "http://localhost:8000/api/v1/ingest/text",
            json={
                "text": "Your document content here...",
                "metadata": {"source": "example"}
            }
        )
        print(response.json())
        
        # Query
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={
                "question": "What is this about?",
                "k": 5
            }
        )
        result = response.json()
        print(result["answer"])

asyncio.run(example())
```

## Testing

Run tests with pytest:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   FastAPI                       │
│              (API Gateway)                      │
└─────────────────┬───────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼──────────┐
│   Document     │  │  RAG Service    │
│   Processor    │  │  (LangChain)    │
└────────┬───────┘  └─────────┬───────┘
         │                    │
         │        ┌───────────▼────────┐
         │        │  Summarization     │
         │        │  Service           │
         │        └────────────────────┘
         │
    ┌────▼──────┐
    │  Vector   │
    │  Store    │
    │ (Qdrant)  │
    └───────────┘
```

## Key Features

✅ **Document Processing**
- Multi-format support (PDF, DOCX, TXT, MD, HTML)
- URL content extraction
- Intelligent chunking with overlap

✅ **RAG Pipeline**
- Semantic search with Qdrant
- Context-aware answer generation
- Multi-query retrieval for better coverage
- Source citation

✅ **Summarization**
- Multiple formats (bullets, paragraphs, executive)
- Configurable length
- Key point extraction
- Document comparison

✅ **Production Ready**
- Docker support
- Health checks
- Async operations
- Error handling
- API documentation

## Configuration

Key environment variables in `.env`:

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:14b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=documents

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.7
```

## Troubleshooting

**Issue: Connection to Ollama fails**
- Ensure Ollama is running: `ollama serve`
- Check the URL in `.env` matches your Ollama instance

**Issue: Qdrant connection fails**
- Verify Qdrant is running: `docker ps | grep qdrant`
- Check the port 6333 is not already in use

**Issue: Out of memory errors**
- Reduce `CHUNK_SIZE` in `.env`
- Reduce `RAG_TOP_K` for fewer retrieved documents
- Use a smaller model

## Next Steps

- Add authentication middleware
- Implement Redis-based job queue for long-running tasks
- Add support for more document formats
- Implement caching layer
- Add monitoring with Prometheus
- Deploy to production (K8s, Cloud Run, etc.)
