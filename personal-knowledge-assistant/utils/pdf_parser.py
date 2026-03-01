from __future__ import annotations
import os
from typing import List, Dict, Any
import pdfplumber
import config

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def parse_pdf(path: str) -> str:
    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)

def parse_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(path)
    elif ext in (".txt", ".md"):
        return parse_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}")

def file_to_chunks(path: str) -> List[Dict[str, Any]]:
    filename = os.path.basename(path)
    raw_text = parse_file(path)
    raw_chunks = chunk_text(raw_text)
    return [
        {"id": f"{filename}::chunk{i}", "text": chunk, "source": filename, "chunk": i}
        for i, chunk in enumerate(raw_chunks)
    ]
