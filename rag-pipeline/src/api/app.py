"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config import get_settings
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_service import RAGService
from src.summarization_service import SummarizationService
from src.api.models import (
    DocumentUploadRequest,
    URLIngestionRequest,
    TextIngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    SummarizeTextRequest,
    SummarizeResponse,
    ExtractKeyPointsRequest,
    KeyPointsResponse,
    CompareDocumentsRequest,
    CompareDocumentsResponse,
    HealthCheckResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
doc_processor: DocumentProcessor | None = None
vector_store: VectorStore | None = None
rag_service: RAGService | None = None
summarization_service: SummarizationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global doc_processor, vector_store, rag_service, summarization_service

    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Initialize services
    try:
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        rag_service = RAGService(vector_store)
        summarization_service = SummarizationService()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    yield
    # Cleanup
    logger.info("Shutting down services")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Documentation RAG and Summarization System",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error", detail=str(exc)).model_dump(),
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Check service health."""
    try:
        vector_stats = await vector_store.get_collection_stats() if vector_store else None
        return HealthCheckResponse(
            status="healthy",
            version=settings.app_version,
            vector_store_status=vector_stats,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Document Ingestion Endpoints
@app.post(f"{settings.api_prefix}/ingest/file", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
) -> IngestionResponse:
    """Ingest a document file."""
    if not doc_processor or not vector_store:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        # Save uploaded file temporarily
        import tempfile
        import shutil

        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        # Process document
        chunks = await doc_processor.process_file(tmp_path)

        # Add to vector store
        doc_ids = await vector_store.add_documents(chunks)

        # Cleanup
        import os
        os.unlink(tmp_path)

        return IngestionResponse(
            document_ids=doc_ids,
            chunk_count=len(chunks),
            status="success",
        )
    except Exception as e:
        logger.error(f"Error ingesting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/ingest/url", response_model=IngestionResponse)
async def ingest_url(request: URLIngestionRequest) -> IngestionResponse:
    """Ingest content from a URL."""
    if not doc_processor or not vector_store:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        chunks = await doc_processor.process_url(request.url, request.metadata)
        doc_ids = await vector_store.add_documents(chunks)

        return IngestionResponse(
            document_ids=doc_ids,
            chunk_count=len(chunks),
            status="success",
        )
    except Exception as e:
        logger.error(f"Error ingesting URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/ingest/text", response_model=IngestionResponse)
async def ingest_text(request: TextIngestionRequest) -> IngestionResponse:
    """Ingest raw text content."""
    if not doc_processor or not vector_store:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        chunks = await doc_processor.process_text(request.text, request.metadata)
        doc_ids = await vector_store.add_documents(chunks)

        return IngestionResponse(
            document_ids=doc_ids,
            chunk_count=len(chunks),
            status="success",
        )
    except Exception as e:
        logger.error(f"Error ingesting text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# RAG Query Endpoint
@app.post(f"{settings.api_prefix}/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Query the knowledge base."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        if request.use_multi_query:
            result = await rag_service.multi_query(
                question=request.question,
                k=request.k,
                filter=request.filter,
            )
        else:
            result = await rag_service.query(
                question=request.question,
                k=request.k,
                filter=request.filter,
                include_sources=request.include_sources,
            )

        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Summarization Endpoints
@app.post(f"{settings.api_prefix}/summarize/text", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeTextRequest) -> SummarizeResponse:
    """Summarize text content."""
    if not summarization_service:
        raise HTTPException(status_code=503, detail="Summarization service not initialized")

    try:
        summary = await summarization_service.summarize_text(
            text=request.text,
            format=request.format,
            length=request.length,
            custom_instructions=request.custom_instructions,
        )

        return SummarizeResponse(
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
        )
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/summarize/key-points", response_model=KeyPointsResponse)
async def extract_key_points(request: ExtractKeyPointsRequest) -> KeyPointsResponse:
    """Extract key points from text."""
    if not summarization_service:
        raise HTTPException(status_code=503, detail="Summarization service not initialized")

    try:
        key_points = await summarization_service.extract_key_points(
            text=request.text,
            num_points=request.num_points,
        )
        
        return KeyPointsResponse(
            key_points=key_points,
            count=len(key_points),
        )
    except Exception as e:
        logger.error(f"Error extracting key points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/summarize/compare", response_model=CompareDocumentsResponse)
async def compare_documents(request: CompareDocumentsRequest) -> CompareDocumentsResponse:
    """Compare two documents."""
    if not summarization_service:
        raise HTTPException(status_code=503, detail="Summarization service not initialized")

    try:
        result = await summarization_service.compare_documents(
            doc1=request.document1,
            doc2=request.document2,
        )

        return CompareDocumentsResponse(**result)
    except Exception as e:
        logger.error(f"Error comparing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Collection Statistics Endpoint
@app.get(f"{settings.api_prefix}/stats")
async def get_stats() -> dict[str, Any]:
    """Get vector store statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        stats = await vector_store.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    """Run the application."""
    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
