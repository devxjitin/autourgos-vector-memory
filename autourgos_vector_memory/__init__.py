"""
autourgos-vector-memory — Local, provider-agnostic embedding (vector) memory
for Autourgos agents.

You supply the embedding function (local model, cloud API, anything with
``fn(text: str) -> Sequence[float]``); this package only stores vectors in
SQLite and ranks them by cosine similarity — no embedding-provider
dependency of its own.

    from autourgos_vector_memory import VectorMemory

    def embed(text: str) -> list[float]:
        ...  # call your local model or a cloud embeddings API

    memory = VectorMemory(embed_fn=embed, db_path="memory.db")
"""
from .memory import VectorMemory, VectorMemoryError, VectorRetriever

from autourgos_core import package_version

__version__ = package_version("autourgos-vector-memory", fallback="1.0.4")

__all__ = ["VectorMemory", "VectorRetriever", "VectorMemoryError"]
