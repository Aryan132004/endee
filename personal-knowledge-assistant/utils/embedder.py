from __future__ import annotations
from typing import List
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import config

@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    print(f"[embedder] Loading model '{config.EMBEDDING_MODEL}' …")
    return SentenceTransformer(config.EMBEDDING_MODEL)

def embed(texts: List[str]) -> List[List[float]]:
    model = _load_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]

def embed_one(text: str) -> List[float]:
    return embed([text])[0]
