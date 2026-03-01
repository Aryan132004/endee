from __future__ import annotations
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from utils.embedder import embed_one
from utils.vector_store import search
from utils.llm import answer
import config

console = Console()

def ask(question: str) -> str:
    with console.status("Thinking …"):
        q_vector = embed_one(question)
        chunks = search(q_vector, top_k=config.TOP_K)
        if not chunks:
            return "No relevant documents found. Did you ingest any files yet?"
        response = answer(question, chunks)
    console.print(Panel(response, title="[bold cyan]Answer[/]", border_style="cyan"))
    table = Table(title="Retrieved Context", show_lines=True, style="dim")
    table.add_column("#", style="bold", width=3)
    table.add_column("Source", style="yellow")
    table.add_column("Similarity", style="green", justify="right")
    table.add_column("Snippet", no_wrap=False)
    for i, chunk in enumerate(chunks, 1):
        table.add_row(str(i), chunk["source"], str(chunk["similarity"]), chunk["text"][:120].replace("\n", " ") + "…")
    console.print(table)
    return response

def main() -> None:
    if len(sys.argv) >= 2:
        ask(" ".join(sys.argv[1:]))
    else:
        console.print(Panel("[bold]Personal Knowledge Assistant[/]\nType your question. Type exit to quit.", border_style="blue"))
        while True:
            try:
                question = console.input("\n[bold blue]You:[/] ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not question or question.lower() in ("exit", "quit"):
                break
            ask(question)

if __name__ == "__main__":
    main()
