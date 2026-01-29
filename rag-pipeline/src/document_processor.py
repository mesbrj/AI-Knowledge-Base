"""Document processing service for parsing, chunking, and embedding documents."""
import asyncio
import hashlib
import mimetypes
from pathlib import Path
from typing import Any

import aiofiles
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import httpx
import html2text

from src.config import get_settings


class DocumentProcessor:
    """Handles document parsing, chunking, and preprocessing."""

    def __init__(self) -> None:
        """Initialize the document processor."""
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def process_file(self, file_path: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """
        Process a file and return chunked documents.

        Args:
            file_path: Path to the file to process
            metadata: Optional metadata to attach to documents

        Returns:
            List of chunked Document objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(file_path)

        # Load document based on type
        if mime_type == "application/pdf" or file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif mime_type == "text/plain" or file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_path.endswith(".md") or file_path.endswith(".markdown"):
            loader = UnstructuredMarkdownLoader(file_path)
        elif mime_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ] or file_path.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
        else:
            # Default to text loader
            loader = TextLoader(file_path, encoding="utf-8")

        # Load and split documents
        documents = await asyncio.to_thread(loader.load)
        chunks = self.text_splitter.split_documents(documents)

        # Add metadata
        file_metadata = {
            "source": file_path,
            "filename": path.name,
            "file_type": mime_type or "unknown",
            "file_hash": await self._compute_file_hash(file_path),
            **(metadata or {}),
        }

        for chunk in chunks:
            chunk.metadata.update(file_metadata)

        return chunks

    async def process_url(self, url: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """
        Fetch and process content from a URL.

        Args:
            url: URL to fetch content from
            metadata: Optional metadata to attach to documents

        Returns:
            List of chunked Document objects
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.text

        # Clean HTML content
        soup = BeautifulSoup(content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Convert to markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        text = h.handle(str(soup))

        # Create document
        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "source_type": "url",
                "content_type": response.headers.get("content-type", "unknown"),
                **(metadata or {}),
            },
        )

        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        return chunks

    async def process_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """
        Process raw text content.

        Args:
            text: Text content to process
            metadata: Optional metadata to attach to documents

        Returns:
            List of chunked Document objects
        """
        doc = Document(
            page_content=text,
            metadata={"source": "raw_text", "source_type": "text", **(metadata or {})},
        )

        chunks = self.text_splitter.split_documents([doc])
        return chunks

    async def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            # Read in 8kb chunks
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def batch_process_files(
        self, file_paths: list[str], metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        """
        Process multiple files concurrently.

        Args:
            file_paths: List of file paths to process
            metadata: Optional metadata to attach to all documents

        Returns:
            List of all chunked Document objects
        """
        tasks = [self.process_file(fp, metadata) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_chunks: list[Document] = []
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue processing other files
                print(f"Error processing file: {result}")
                continue
            all_chunks.extend(result)

        return all_chunks
