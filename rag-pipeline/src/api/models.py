"""API models for request and response validation."""
from typing import Any
from pydantic import BaseModel, Field

from src.summarization_service import SummaryFormat, SummaryLength


# Document Ingestion Models
class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""

    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata for the document")


class URLIngestionRequest(BaseModel):
    """Request model for URL ingestion."""

    url: str = Field(..., description="URL to ingest content from")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata")


class TextIngestionRequest(BaseModel):
    """Request model for text ingestion."""

    text: str = Field(..., description="Text content to ingest")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata")


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""

    document_ids: list[str] = Field(..., description="List of ingested document IDs")
    chunk_count: int = Field(..., description="Number of chunks created")
    status: str = Field(default="success", description="Ingestion status")


# RAG Query Models
class QueryRequest(BaseModel):
    """Request model for RAG query."""

    question: str = Field(..., description="Question to ask")
    k: int | None = Field(default=None, description="Number of documents to retrieve")
    filter: dict[str, Any] | None = Field(default=None, description="Metadata filter")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    use_multi_query: bool = Field(default=False, description="Use multi-query retrieval")


class SourceInfo(BaseModel):
    """Source document information."""

    content: str = Field(..., description="Document content snippet")
    metadata: dict[str, Any] = Field(..., description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    answer: str = Field(..., description="Generated answer")
    question: str = Field(..., description="Original question")
    sources: list[SourceInfo] | None = Field(default=None, description="Source documents")
    source_count: int | None = Field(default=None, description="Number of sources used")
    queries_used: list[str] | None = Field(default=None, description="Queries used (for multi-query)")


# Summarization Models
class SummarizeTextRequest(BaseModel):
    """Request model for text summarization."""

    text: str = Field(..., description="Text to summarize")
    format: SummaryFormat = Field(default=SummaryFormat.PARAGRAPH, description="Summary format")
    length: SummaryLength = Field(default=SummaryLength.MEDIUM, description="Summary length")
    custom_instructions: str | None = Field(default=None, description="Custom instructions")


class SummarizeResponse(BaseModel):
    """Response model for summarization."""

    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text length")
    summary_length: int = Field(..., description="Summary length")


class ExtractKeyPointsRequest(BaseModel):
    """Request model for key point extraction."""

    text: str = Field(..., description="Text to extract key points from")
    num_points: int = Field(default=5, description="Number of key points to extract")


class KeyPointsResponse(BaseModel):
    """Response model for key points."""

    key_points: list[str] = Field(..., description="Extracted key points")
    count: int = Field(..., description="Number of key points")


class CompareDocumentsRequest(BaseModel):
    """Request model for document comparison."""

    document1: str = Field(..., description="First document text")
    document2: str = Field(..., description="Second document text")


class CompareDocumentsResponse(BaseModel):
    """Response model for document comparison."""

    comparison: str = Field(..., description="Comparison result")
    doc1_length: int = Field(..., description="First document length")
    doc2_length: int = Field(..., description="Second document length")


# Health Check Models
class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    vector_store_status: dict[str, Any] | None = Field(default=None, description="Vector store stats")


# Error Response
class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
