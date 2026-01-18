#!/usr/bin/env python3
"""CLI tool for interacting with the documentation RAG system."""
import asyncio
import sys
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

app = typer.Typer(help="Documentation RAG System CLI")
console = Console()

# Default API URL
API_URL = "http://localhost:8000/api/v1"


@app.command()
def ingest_file(
    file_path: str = typer.Argument(..., help="Path to the file to ingest"),
    api_url: str = typer.Option(API_URL, help="API base URL"),
) -> None:
    """Ingest a document file into the knowledge base."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    async def _ingest() -> None:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (path.name, f, "application/octet-stream")}
                response = await client.post(f"{api_url}/ingest/file", files=files)
                response.raise_for_status()
                result = response.json()
            
            console.print(Panel(
                f"[green]Successfully ingested {path.name}[/green]\n"
                f"Document IDs: {len(result['document_ids'])}\n"
                f"Chunks created: {result['chunk_count']}",
                title="Ingestion Complete"
            ))
    
    asyncio.run(_ingest())


@app.command()
def ingest_url(
    url: str = typer.Argument(..., help="URL to ingest"),
    api_url: str = typer.Option(API_URL, help="API base URL"),
) -> None:
    """Ingest content from a URL."""
    async def _ingest() -> None:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{api_url}/ingest/url",
                json={"url": url}
            )
            response.raise_for_status()
            result = response.json()
            
            console.print(Panel(
                f"[green]Successfully ingested URL[/green]\n"
                f"URL: {url}\n"
                f"Chunks created: {result['chunk_count']}",
                title="Ingestion Complete"
            ))
    
    asyncio.run(_ingest())


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    k: Optional[int] = typer.Option(None, help="Number of documents to retrieve"),
    multi_query: bool = typer.Option(False, help="Use multi-query retrieval"),
    api_url: str = typer.Option(API_URL, help="API base URL"),
) -> None:
    """Query the knowledge base."""
    async def _query() -> None:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "question": question,
                "use_multi_query": multi_query,
                "include_sources": True,
            }
            if k:
                payload["k"] = k
            
            response = await client.post(f"{api_url}/query", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Display answer
            console.print(Panel(
                Markdown(result["answer"]),
                title=f"[bold cyan]Answer[/bold cyan]",
                border_style="cyan"
            ))
            
            # Display sources
            if result.get("sources"):
                console.print("\n[bold yellow]Sources:[/bold yellow]")
                for i, source in enumerate(result["sources"], 1):
                    console.print(f"\n[dim]{i}. {source['metadata'].get('source', 'Unknown')}[/dim]")
                    console.print(f"[dim]{source['content']}[/dim]")
    
    asyncio.run(_query())


@app.command()
def summarize(
    text: Optional[str] = typer.Option(None, help="Text to summarize"),
    file_path: Optional[str] = typer.Option(None, help="File containing text to summarize"),
    format: str = typer.Option("paragraph", help="Summary format (bullet_points, paragraph, executive)"),
    length: str = typer.Option("medium", help="Summary length (short, medium, long)"),
    api_url: str = typer.Option(API_URL, help="API base URL"),
) -> None:
    """Summarize text content."""
    if not text and not file_path:
        console.print("[red]Error: Either --text or --file-path must be provided[/red]")
        raise typer.Exit(1)
    
    if file_path:
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
        text = path.read_text()
    
    async def _summarize() -> None:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{api_url}/summarize/text",
                json={"text": text, "format": format, "length": length}
            )
            response.raise_for_status()
            result = response.json()
            
            console.print(Panel(
                Markdown(result["summary"]),
                title=f"[bold green]Summary ({format}, {length})[/bold green]",
                border_style="green"
            ))
            console.print(f"\n[dim]Original: {result['original_length']} chars | "
                         f"Summary: {result['summary_length']} chars[/dim]")
    
    asyncio.run(_summarize())


@app.command()
def key_points(
    text: Optional[str] = typer.Option(None, help="Text to extract key points from"),
    file_path: Optional[str] = typer.Option(None, help="File containing text"),
    num_points: int = typer.Option(5, help="Number of key points to extract"),
    api_url: str = typer.Option(API_URL, help="API base URL"),
) -> None:
    """Extract key points from text."""
    if not text and not file_path:
        console.print("[red]Error: Either --text or --file-path must be provided[/red]")
        raise typer.Exit(1)
    
    if file_path:
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
        text = path.read_text()
    
    async def _extract() -> None:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{api_url}/summarize/key-points",
                json={"text": text, "num_points": num_points}
            )
            response.raise_for_status()
            result = response.json()
            
            console.print(Panel(
                "\n".join(f"{i}. {point}" for i, point in enumerate(result["key_points"], 1)),
                title=f"[bold magenta]Key Points ({result['count']})[/bold magenta]",
                border_style="magenta"
            ))
    
    asyncio.run(_extract())


@app.command()
def stats(api_url: str = typer.Option(API_URL, help="API base URL")) -> None:
    """Get vector store statistics."""
    async def _stats() -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/stats")
            response.raise_for_status()
            result = response.json()
            
            table = Table(title="Vector Store Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in result.items():
                table.add_row(key.replace("_", " ").title(), str(value))
            
            console.print(table)
    
    asyncio.run(_stats())


@app.command()
def health(api_url: str = typer.Option(API_URL, help="API base URL")) -> None:
    """Check service health."""
    async def _health() -> None:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_url.replace('/api/v1', '')}/health")
                response.raise_for_status()
                result = response.json()
                
                console.print(Panel(
                    f"[green]Status: {result['status']}[/green]\n"
                    f"Version: {result['version']}\n"
                    f"Vector Store: {result.get('vector_store_status', {}).get('status', 'N/A')}",
                    title="Service Health"
                ))
            except Exception as e:
                console.print(f"[red]Error: Service unavailable - {str(e)}[/red]")
                raise typer.Exit(1)
    
    asyncio.run(_health())


if __name__ == "__main__":
    app()
