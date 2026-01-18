"""Tests for document processor."""
import pytest
from pathlib import Path

from src.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    """Create a document processor instance."""
    return DocumentProcessor()


@pytest.mark.asyncio
async def test_process_text(processor):
    """Test text processing."""
    text = "This is a test document. " * 100
    chunks = await processor.process_text(text)
    
    assert len(chunks) > 0
    assert all(hasattr(chunk, "page_content") for chunk in chunks)
    assert all(hasattr(chunk, "metadata") for chunk in chunks)


@pytest.mark.asyncio
async def test_compute_file_hash(processor, tmp_path):
    """Test file hash computation."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    hash1 = await processor._compute_file_hash(str(test_file))
    hash2 = await processor._compute_file_hash(str(test_file))
    
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex length
