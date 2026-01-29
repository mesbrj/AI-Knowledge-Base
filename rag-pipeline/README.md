System combining document Q&A (RAG) and content summarization capabilities.

**Architecture:**
- Document Ingestion Service (parsing, chunking, embedding)
- Vector Storage (Qdrant)
- Query Service (semantic search + answer generation)
- Summarization Service (content extraction + summarization)
- API Gateway (REST API + CLI)

**Core Features:**

*Document Processing:*
- Multi-format support (PDF, TXT, Markdown, DOCX)
- URL content extraction and cleaning
- Intelligent text chunking (1000 tokens, 200 overlap)
- Metadata extraction and indexing

*RAG Pipeline:*
- Embedding generation using `nomic-embed-text` (768 dimensions)
- Qdrant vector storage with HNSW indexing
- Hybrid search (semantic + keyword)
- Context-aware answer generation with citations
- Multi-query retrieval for better coverage

*Summarization:*
- Multiple formats (bullet points, paragraphs, executive summary)
- Configurable length (short/medium/long)
- Batch processing with concurrent workers
- Job queue with Go channels

*API & Interface:*
- REST API (ingestion, query, summarization endpoints)
- CLI tool for local operations
- WebSocket for streaming responses
- Health checks and metrics

MVP Technical Stack:
- **Language:** Python 3.11+
- **LLM Framework:** LangChain with Ollama (`deepseek-r1:14b` for generation, `nomic-embed-text` for embeddings)
- **Vector DB (semantic search):** Qdrant (via langchain-qdrant)
- **API Framework:** FastAPI (Auto-generated OpenAPI 3.0, async support)
- **Testing:** pytest + pytest-asyncio

[*Quick Start*](/rag-pipeline/docs/QUICK_START.md)
>
***improvements needed***: advanced chunking (use semantic + structure-aware) / token management (replace Character counting to token counting)