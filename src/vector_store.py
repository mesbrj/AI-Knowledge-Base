"""Vector store management with Qdrant."""
from typing import Any

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import get_settings


class VectorStore:
    """Manages vector storage and retrieval with Qdrant."""

    def __init__(self) -> None:
        """Initialize the vector store."""
        self.settings = get_settings()

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_embedding_model,
        )

        # Ensure collection exists
        self._ensure_collection()

        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.qdrant_collection_name,
            embedding=self.embeddings,
        )

    def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.settings.qdrant_collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=self.settings.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        ids = await self.vector_store.aadd_documents(documents)
        return ids

    async def similarity_search(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Perform similarity search.

        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score
            filter: Optional metadata filter

        Returns:
            List of relevant documents
        """
        k = k or self.settings.rag_top_k

        if score_threshold is None:
            results = await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                filter=filter,
            )
        else:
            results = await self.vector_store.asimilarity_search_with_relevance_scores(
                query=query,
                k=k,
                score_threshold=score_threshold,
                filter=filter,
            )
            # Extract documents from (doc, score) tuples
            if results and isinstance(results[0], tuple):
                results = [doc for doc, _ in results]

        return results

    async def delete_by_metadata(self, filter: dict[str, Any]) -> None:
        """
        Delete documents by metadata filter.

        Args:
            filter: Metadata filter to match documents for deletion
        """
        # This requires custom implementation with Qdrant client
        # as LangChain's interface doesn't directly support filter-based deletion
        pass

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection."""
        collection_info = self.client.get_collection(self.settings.qdrant_collection_name)
        return {
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": collection_info.status,
        }
