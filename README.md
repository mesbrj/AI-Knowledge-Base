# LLM-Knowledge-Base

# Locally LLMs with Ollama

Windows 11 host with Nvidia RTX 4070TI GPU (12GB VRAM) through WSL2

**NVIDIA Driver installed only on Windows host**

**General GPU Access in WSL2**
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2) 
- [CUDA Toolkit (WSL-Ubuntu)](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

**GPU Access in WSL2 containers**
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Ollama and LLMs**
- [Install Ollama and pull models](https://docs.ollama.com/)
    - [deepseek-r1:14b Q4_K_M (size ~9.0GB)](https://ollama.com/library/deepseek-r1:14b)
    - [gemma3:12b Q4_K_M (size ~8.1GB)](https://ollama.com/library/gemma3:12b)
    - [llama3.1:8b Q8_0 (size ~8.5GB)](https://ollama.com/library/llama3.1:8b-instruct-q8_0)
    - [nomic-embed-text:v1.5 F16 (size ~274MB)](https://ollama.com/library/nomic-embed-text:v1.5)


# Implementation System Example

## *Documentation System (RAG + Summarization)*

A microservices-based system combining document Q&A (RAG) and content summarization capabilities.

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
- Job queue with Redis/in-memory queue

*API & Interface:*
- REST API (ingestion, query, summarization endpoints)
- CLI tool for local operations
- WebSocket for streaming responses
- Health checks and metrics

MVP Technical Stack: Monolith (1 Service):
- **Language:** Go 1.24+
- **LLM:** Ollama (`deepseek-r1:14b` for generation, `nomic-embed-text` for embeddings)
- **Vector DB:** Qdrant 
- **Queue:** Go channels
- **Storage:** S3/local filesystem for raw documents
- **API Framework:** Fuego (Auto-generated OpenAPI 3.0, type-safe Go generics)
- **Testing:** Go testing + testify
