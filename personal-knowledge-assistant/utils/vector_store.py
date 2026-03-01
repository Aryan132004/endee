from __future__ import annotations
from typing import List, Dict, Any
from endee import Endee
import config

_client = None

def get_client() -> Endee:
    global _client
    if _client is None:
        token = config.ENDEE_AUTH_TOKEN or ""
        _client = Endee(token) if token else Endee()
        _client.set_base_url(config.ENDEE_BASE_URL)
    return _client

# def ensure_index(name: str = config.INDEX_NAME):
#     client = get_client()
#     existing = client.list_indexes()
#     if name not in existing:
#         print(f"[vector_store] Creating index '{name}' …")
#         client.create_index(name=name, dimension=config.EMBEDDING_DIM, space_type="cosine", precision="float32")
#     return client.get_index(name=name)
def ensure_index(name: str = config.INDEX_NAME):
    client = get_client()
    try:
        client.create_index(name=name, dimension=config.EMBEDDING_DIM, space_type="cosine", precision="float32")
        print(f"[vector_store] Created index '{name}'")
    except Exception:
        pass  # Index already exists, that's fine
    return client.get_index(name=name)

def upsert_chunks(chunks: List[Dict[str, Any]], vectors: List[List[float]], index_name: str = config.INDEX_NAME) -> int:
    index = ensure_index(index_name)
    items = [
        {"id": chunk["id"], "vector": vector, "meta": {"text": chunk["text"], "source": chunk["source"], "chunk": chunk["chunk"]}}
        for chunk, vector in zip(chunks, vectors)
    ]
    index.upsert(items)
    return len(items)

def search(query_vector: List[float], top_k: int = config.TOP_K, index_name: str = config.INDEX_NAME) -> List[Dict[str, Any]]:
    index = ensure_index(index_name)
    raw_results = index.query(vector=query_vector, top_k=top_k)
    results = []
    for r in raw_results:
        # Endee returns either dicts or objects depending on version — handle both
        if isinstance(r, dict):
            meta = r.get("meta") or {}
            results.append({
                "id":         r.get("id", ""),
                "similarity": round(r.get("similarity", 0), 4),
                "text":       meta.get("text", ""),
                "source":     meta.get("source", "unknown"),
                "chunk":      meta.get("chunk", -1),
            })
        else:
            meta = r.meta or {}
            results.append({
                "id":         r.id,
                "similarity": round(r.similarity, 4),
                "text":       meta.get("text", ""),
                "source":     meta.get("source", "unknown"),
                "chunk":      meta.get("chunk", -1),
            })
    return results
