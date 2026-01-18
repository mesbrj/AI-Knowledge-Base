"""RAG service for question answering with document retrieval."""
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.llms import Ollama

from src.config import get_settings
from src.vector_store import VectorStore


class RAGService:
    """Handles RAG (Retrieval Augmented Generation) operations."""

    def __init__(self, vector_store: VectorStore) -> None:
        """Initialize the RAG service."""
        self.settings = get_settings()
        self.vector_store = vector_store
        
        # Initialize LLM
        self.llm = Ollama(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            temperature=self.settings.ollama_temperature,
            timeout=self.settings.ollama_request_timeout,
        )

    async def query(
        self,
        question: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """
        Query the knowledge base and generate an answer.

        Args:
            question: User's question
            k: Number of documents to retrieve
            filter: Optional metadata filter
            include_sources: Whether to include source documents in response

        Returns:
            Dictionary with answer and optionally source documents
        """
        # Create retriever
        retriever = self.vector_store.vector_store.as_retriever(
            search_kwargs={
                "k": k or self.settings.rag_top_k,
                "score_threshold": self.settings.rag_score_threshold,
                "filter": filter,
            }
        )

        # Create custom prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context.
            
Use the following pieces of context to answer the question. If you cannot find the answer in the context, say so.
Always cite the sources when providing information from the context.

Context:
{context}"""),
            ("human", "{input}"),
        ])

        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create LCEL chain using RunnableParallel for retrieval
        retrieval_chain = RunnableParallel(
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
        )
        
        # Create the full chain
        chain = retrieval_chain | prompt | self.llm | StrOutputParser()

        # Execute query - retrieve context first for sources
        retrieved_docs = await retriever.ainvoke(question)
        
        # Execute the full chain
        answer = await chain.ainvoke(question)

        # Format response
        response: dict[str, Any] = {
            "answer": answer,
            "question": question,
        }

        if include_sources:
            sources = []
            for doc in retrieved_docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                }
                sources.append(source_info)
            response["sources"] = sources
            response["source_count"] = len(sources)

        return response

    async def multi_query(
        self,
        question: str,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Perform multi-query retrieval for better coverage.

        Args:
            question: User's question
            k: Number of documents to retrieve per query
            filter: Optional metadata filter

        Returns:
            Dictionary with answer and source documents
        """
        # Generate query variations
        query_generation_prompt = f"""You are an AI assistant helping to generate multiple search queries.
        
For the following question, generate 3 different ways to phrase it or related questions that might help find relevant information:

Question: {question}

Provide only the alternative questions, one per line, without numbering or explanations."""

        variations_response = await self.llm.ainvoke(query_generation_prompt)
        
        # Parse variations
        variations = [q.strip() for q in variations_response.split("\n") if q.strip()]
        variations = [question] + variations[:3]  # Include original + up to 3 variations

        # Retrieve documents for each variation
        all_docs = []
        seen_contents = set()
        
        for query in variations:
            docs = await self.vector_store.similarity_search(
                query=query,
                k=k or self.settings.rag_top_k,
                filter=filter,
            )
            
            # Deduplicate by content
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)

        # Limit total context
        all_docs = all_docs[: (k or self.settings.rag_top_k) * 2]

        # Generate answer with combined context
        context = "\n\n".join([f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(all_docs)])
        
        answer_prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information, say so.

Context:
{context[:self.settings.rag_max_context_length]}

Question: {question}

Answer:"""

        answer = await self.llm.ainvoke(answer_prompt)

        return {
            "answer": answer,
            "question": question,
            "queries_used": variations,
            "sources": [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in all_docs
            ],
            "source_count": len(all_docs),
        }
