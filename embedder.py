"""Local embedding — all-MiniLM-L6-v2 via sentence-transformers. No network calls after first download."""

from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    return _model(model_name).encode(texts, show_progress_bar=False).tolist()


def embed_one(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    return _model(model_name).encode([text]).tolist()[0]
