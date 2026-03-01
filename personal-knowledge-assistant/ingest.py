from __future__ import annotations
import sys, glob, time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from utils.pdf_parser import file_to_chunks
from utils.embedder import embed
from utils.vector_store import upsert_chunks

console = Console()

def ingest_file(path: str) -> None:
    console.print(f"\n[bold cyan]📄 Ingesting:[/] {path}")
    with console.status("Parsing document …"):
        chunks = file_to_chunks(path)
    console.print(f"   ✔ Split into [bold]{len(chunks)}[/] chunks")
    if not chunks:
        console.print("[yellow]   ⚠ No text extracted – skipping.[/]")
        return
    texts = [c["text"] for c in chunks]
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console, transient=True) as progress:
        task = progress.add_task("Generating embeddings …", total=len(texts))
        vectors = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i+64]
            vectors.extend(embed(batch))
            progress.advance(task, len(batch))
    console.print(f"   ✔ Embeddings generated ({len(vectors)} vectors)")
    with console.status("Uploading to Endee …"):
        count = upsert_chunks(chunks, vectors)
    console.print(f"   [bold green]✔ Stored {count} chunks in Endee[/]")

def main() -> None:
    if len(sys.argv) < 2:
        console.print("[red]Usage: python ingest.py <file1> [file2 ...][/]")
        sys.exit(1)
    paths = []
    for arg in sys.argv[1:]:
        expanded = glob.glob(arg)
        paths.extend(expanded if expanded else [arg])
    start = time.time()
    errors = []
    for path in paths:
        try:
            ingest_file(path)
        except Exception as exc:
            errors.append(path)
            console.print(f"   [red]✖ Error:[/] {exc}")
    console.print(f"\n[bold green]Done![/] {len(paths)-len(errors)}/{len(paths)} files in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
