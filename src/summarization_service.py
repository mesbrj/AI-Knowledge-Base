"""Summarization service for document content."""
import asyncio
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from src.config import get_settings


class SummaryFormat(str, Enum):
    """Supported summary formats."""

    BULLET_POINTS = "bullet_points"
    PARAGRAPH = "paragraph"
    EXECUTIVE = "executive"


class SummaryLength(str, Enum):
    """Supported summary lengths."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SummarizationService:
    """Handles document summarization operations."""

    def __init__(self) -> None:
        """Initialize the summarization service."""
        self.settings = get_settings()

        # Initialize LLM
        self.llm = Ollama(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            temperature=0.3,  # Lower temperature for more consistent summaries
            timeout=self.settings.ollama_request_timeout,
        )

    async def summarize_text(
        self,
        text: str,
        format: SummaryFormat = SummaryFormat.PARAGRAPH,
        length: SummaryLength = SummaryLength.MEDIUM,
        custom_instructions: str | None = None,
    ) -> str:
        """
        Summarize text content.

        Args:
            text: Text to summarize
            format: Summary format (bullet_points, paragraph, executive)
            length: Summary length (short, medium, long)
            custom_instructions: Optional custom instructions for the summary

        Returns:
            Summary text
        """
        # Create document
        doc = Document(page_content=text)
        return await self.summarize_documents([doc], format, length, custom_instructions)

    async def summarize_documents(
        self,
        documents: list[Document],
        format: SummaryFormat = SummaryFormat.PARAGRAPH,
        length: SummaryLength = SummaryLength.MEDIUM,
        custom_instructions: str | None = None,
    ) -> str:
        """
        Summarize multiple documents.

        Args:
            documents: List of documents to summarize
            format: Summary format
            length: Summary length
            custom_instructions: Optional custom instructions

        Returns:
            Combined summary
        """
        # Build prompt based on format and length
        prompt_template = self._build_prompt(format, length, custom_instructions)
        
        # Use map-reduce for long documents, stuff for shorter ones
        total_length = sum(len(doc.page_content) for doc in documents)

        if total_length > 4000:
            # Map-reduce approach for long content
            # First, summarize each document individually
            map_prompt = PromptTemplate(
                template="""Summarize the following text:

{text}

Summary:""",
                input_variables=["text"],
            )
            map_chain = map_prompt | self.llm | StrOutputParser()

            # Generate individual summaries
            individual_summaries = []
            for doc in documents:
                summary = await map_chain.ainvoke({"text": doc.page_content})
                individual_summaries.append(summary)

            # Combine summaries
            combined_text = "\n\n".join(individual_summaries)
            reduce_chain = prompt_template | self.llm | StrOutputParser()
            result = await reduce_chain.ainvoke({"text": combined_text})
        else:
            # Stuff approach for shorter content
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            chain = prompt_template | self.llm | StrOutputParser()
            result = await chain.ainvoke({"text": combined_text})

        return result

    async def batch_summarize(
        self,
        texts: list[str],
        format: SummaryFormat = SummaryFormat.PARAGRAPH,
        length: SummaryLength = SummaryLength.MEDIUM,
    ) -> list[str]:
        """
        Summarize multiple texts concurrently.

        Args:
            texts: List of texts to summarize
            format: Summary format
            length: Summary length

        Returns:
            List of summaries
        """
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_workers)

        async def summarize_with_limit(text: str) -> str:
            async with semaphore:
                return await self.summarize_text(text, format, length)

        tasks = [summarize_with_limit(text) for text in texts]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        return [s if not isinstance(s, Exception) else f"Error: {str(s)}" for s in summaries]

    def _build_prompt(
        self,
        format: SummaryFormat,
        length: SummaryLength,
        custom_instructions: str | None = None,
    ) -> PromptTemplate:
        """Build the summarization prompt based on parameters."""
        # Length specifications
        length_specs = {
            SummaryLength.SHORT: "concise (2-3 sentences)",
            SummaryLength.MEDIUM: "moderate length (1-2 paragraphs)",
            SummaryLength.LONG: "comprehensive (3-4 paragraphs)",
        }

        # Format specifications
        format_specs = {
            SummaryFormat.BULLET_POINTS: "Format the summary as bullet points, highlighting key points.",
            SummaryFormat.PARAGRAPH: "Format the summary as flowing paragraphs.",
            SummaryFormat.EXECUTIVE: "Format as an executive summary with an overview and key takeaways.",
        }

        base_instruction = f"""Write a {length_specs[length]} summary of the following text.
{format_specs[format]}

Focus on the main ideas and key information."""

        if custom_instructions:
            base_instruction += f"\n\nAdditional instructions: {custom_instructions}"

        template = f"""{base_instruction}

Text:
{{text}}

Summary:"""

        return PromptTemplate(template=template, input_variables=["text"])

    async def extract_key_points(self, text: str, num_points: int = 5) -> list[str]:
        """
        Extract key points from text.

        Args:
            text: Text to extract key points from
            num_points: Number of key points to extract

        Returns:
            List of key points
        """
        prompt = f"""Extract the {num_points} most important key points from the following text.
Provide each point as a single, clear sentence.

Text:
{text}

Key Points (one per line):"""

        response = await self.llm.ainvoke(prompt)

        # Parse key points
        points = [line.strip() for line in response.split("\n") if line.strip()]
        # Remove numbering if present
        points = [p.lstrip("0123456789.-) ") for p in points]
        
        return points[:num_points]

    async def compare_documents(self, doc1: str, doc2: str) -> dict[str, Any]:
        """
        Compare two documents and summarize similarities and differences.

        Args:
            doc1: First document text
            doc2: Second document text

        Returns:
            Dictionary with comparison results
        """
        prompt = f"""Compare the following two documents and provide:
1. Main similarities
2. Key differences
3. Overall comparison summary

Document 1:
{doc1[:2000]}

Document 2:
{doc2[:2000]}

Comparison:"""

        response = await self.llm.ainvoke(prompt)

        return {
            "comparison": response,
            "doc1_length": len(doc1),
            "doc2_length": len(doc2),
        }
