"""
memory.py — Local, provider-agnostic embedding (vector) memory.

VectorRetriever never computes embeddings itself. The caller supplies an
``embed_fn: Callable[[str], Sequence[float]]`` — a local model, a cloud API
call, anything with that shape — so this package stays a thin, zero-ML-
dependency storage + similarity layer, matching the rest of the
autourgos-memory family's "bring your own everything" design.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from autourgos_buffer_memory import RuntimeShortTermMemory
from autourgos_core import open_sqlite, row_cap_evict
from autourgos_memory import BaseMemory, MemoryMessage

from .base import BaseRetriever, Document


class VectorMemoryError(Exception):
    """Raised for misuse of VectorRetriever/VectorMemory (bad embed_fn output,
    dimension mismatch, etc.)."""


def _validate_vector(vec: object, expected_dim: Optional[int] = None) -> List[float]:
    if isinstance(vec, (str, bytes)) or not isinstance(vec, Sequence):
        raise VectorMemoryError(
            f"embed_fn must return a sequence of numbers, got {type(vec).__name__!r}."
        )
    values = list(vec)
    if not values:
        raise VectorMemoryError("embed_fn returned an empty vector.")
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise VectorMemoryError(f"embed_fn returned non-numeric values: {exc}") from exc
    if expected_dim is not None and len(values) != expected_dim:
        raise VectorMemoryError(
            f"embed_fn returned a {len(values)}-dimensional vector, but this store "
            f"was created with {expected_dim}-dimensional vectors. Use a fresh "
            f"db_path if you're switching embedding models."
        )
    return values


class VectorRetriever(BaseRetriever):
    """SQLite-persisted retriever over embeddings the caller supplies.

    Parameters
    ----------
    embed_fn : callable
        ``fn(text: str) -> Sequence[float]``. Called once to embed each
        document on ``add_document()`` and once per ``retrieve()`` call to
        embed the query. Can wrap a local model or a cloud API — this class
        does not care.
    db_path : str
        SQLite file path. ``":memory:"`` (default) keeps everything in RAM
        and discards it when the process exits; pass a real file path for
        persistence across restarts.
    max_documents : int, optional
        Oldest documents are dropped once this count is exceeded. ``None``
        (default) keeps everything.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Sequence[float]],
        db_path: str = ":memory:",
        max_documents: Optional[int] = None,
    ) -> None:
        if not callable(embed_fn):
            raise VectorMemoryError("embed_fn must be callable.")
        if max_documents is not None and (not isinstance(max_documents, int) or max_documents < 1):
            raise ValueError("max_documents must be an integer >= 1 or None")

        self.embed_fn = embed_fn
        self.max_documents = max_documents
        self._lock = threading.RLock()
        self._dim: Optional[int] = None

        self._conn = open_sqlite(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                vector TEXT NOT NULL,
                dim INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

        row = self._conn.execute("SELECT dim FROM documents LIMIT 1").fetchone()
        if row is not None:
            self._dim = row[0]

    def add_document(self, document: Document) -> None:
        with self._lock:
            # Re-derive the dimension from the table itself rather than
            # trusting the in-memory `self._dim` cache alone: if another
            # process (or another VectorRetriever instance) inserted rows
            # into this same db_path after this instance was constructed,
            # `self._dim` here would still be stale (or None), letting a
            # differently-sized vector through and corrupting the table
            # with mixed dimensions -- which later crashes retrieve()'s
            # `np.array([...])` with a ragged-sequence ValueError.
            row = self._conn.execute("SELECT dim FROM documents LIMIT 1").fetchone()
            authoritative_dim = row[0] if row is not None else self._dim
            vector = _validate_vector(self.embed_fn(document.content), authoritative_dim)
            self._dim = len(vector)
            self._conn.execute(
                "INSERT INTO documents (content, metadata, vector, dim, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    document.content,
                    json.dumps(document.metadata or {}),
                    json.dumps(vector),
                    len(vector),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
            if self.max_documents is not None:
                row_cap_evict(self._conn, "documents", "id", self.max_documents)
                self._conn.commit()

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            self.add_document(doc)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT content, metadata, vector FROM documents"
            ).fetchall()
            if not rows:
                return []
            query_vec = np.array(
                _validate_vector(self.embed_fn(query), self._dim), dtype=np.float64
            )
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return []

            vectors = [json.loads(r[2]) for r in rows]
            expected_dim = len(query_vec)
            # Defense in depth against a table that ended up with mixed
            # vector dimensions (e.g. a stale `self._dim` cache let a
            # mismatched embed_fn through, or two processes raced on an
            # empty db_path -- see add_document()). Rather than letting
            # `np.array(vectors)` raise a ragged-sequence ValueError and
            # take retrieve() down entirely, silently drop any row whose
            # stored vector doesn't match the query's dimension so the
            # rest of the store stays queryable.
            usable = [(i, v) for i, v in enumerate(vectors) if len(v) == expected_dim]
            if not usable:
                return []
            usable_idx = [i for i, _ in usable]
            matrix = np.array([v for _, v in usable], dtype=np.float64)
            rows = [rows[i] for i in usable_idx]
            norms = np.linalg.norm(matrix, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                scores = (matrix @ query_vec) / (norms * query_norm)
            scores = np.nan_to_num(scores, nan=0.0)

            top_k = max(0, top_k)
            if top_k == 0:
                return []
            order = np.argsort(-scores)[:top_k]
            return [
                Document(
                    content=rows[i][0],
                    metadata=json.loads(rows[i][1]),
                    score=float(scores[i]),
                )
                for i in order
            ]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM documents")
            self._conn.commit()
            self._dim = None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "VectorRetriever":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class VectorMemory(BaseMemory):
    """Dual-store: sliding short-term buffer + persisted vector retrieval.

    Every message is added to a short-term buffer (recent turns, always
    included) and indexed into a VectorRetriever (older, relevant turns,
    surfaced only when semantically similar to the current query). Mirrors
    autourgos-semantic-memory's KeywordMemory, swapping TF-IDF for
    caller-supplied embeddings.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
        short_term: Optional[BaseMemory] = None,
        retriever: Optional[VectorRetriever] = None,
        db_path: str = ":memory:",
        top_k: int = 3,
        max_documents: Optional[int] = None,
    ) -> None:
        if retriever is None and embed_fn is None:
            raise VectorMemoryError(
                "Pass either embed_fn= (VectorMemory will build its own "
                "VectorRetriever) or retriever= (a pre-built VectorRetriever)."
            )
        self.short_term = short_term or RuntimeShortTermMemory(max_messages=10, name="vector")
        self.retriever = retriever or VectorRetriever(
            embed_fn=embed_fn,  # type: ignore[arg-type]
            db_path=db_path,
            max_documents=max_documents,
        )
        self.top_k = top_k

    def _index(self, content: str, role: str, ts: datetime) -> None:
        self.retriever.add_document(Document(
            content=content,
            metadata={"role": role, "timestamp": ts.astimezone(timezone.utc).isoformat()},
        ))

    def add_user_message(self, content: str) -> MemoryMessage:
        msg = self.short_term.add_user_message(content)
        self._index(content, "user", msg.timestamp)
        return msg

    def add_agent_message(self, content: str) -> MemoryMessage:
        msg = self.short_term.add_agent_message(content)
        self._index(content, "agent", msg.timestamp)
        return msg

    def add_tool_message(self, tool_name: str, result: str) -> MemoryMessage:
        msg = self.short_term.add_tool_message(tool_name, result)
        self._index(msg.content, "tool", msg.timestamp)
        return msg

    def format_for_llm(self, query: Optional[str] = None) -> str:
        st_context = self.short_term.format_for_llm()
        if not query:
            return st_context
        recent: set = set()
        get_msgs = getattr(self.short_term, "get_messages", None)
        if callable(get_msgs):
            recent = {
                m.content if hasattr(m, "content") else m.get("content", "")
                for m in get_msgs()
            }
        relevant = [d for d in self.retriever.retrieve(query, top_k=self.top_k) if d.content not in recent]
        if not relevant:
            return st_context
        past = "\n--- Relevant Past Context ---\n"
        for doc in relevant:
            prefix = f"[{doc.metadata['role']}]: " if "role" in doc.metadata else ""
            past += f"{prefix}{doc.content}\n"
        past += "-----------------------------\n\n"
        return past + st_context

    def clear(self) -> None:
        self.short_term.clear()
        self.retriever.clear()
