# autourgos-vector-memory — Features

Local, persisted, provider-agnostic embedding (vector) memory for Autourgos agents. The caller supplies the embedding function (`fn(text: str) -> Sequence[float]` — a local model, a cloud API, anything); this package only stores vectors in SQLite and ranks them by cosine similarity. No embedding-provider dependency, no vector-database server. Documented as the upgrade path from `autourgos-semantic-memory`'s TF-IDF keyword matching when queries are related in meaning but share no keywords.

## Full Feature List

### Core memory
- **`VectorMemory`** — chat-buffer wrapper: short-term recent-turns buffer (always included) + vector retrieval over the full history
- **`VectorRetriever`** — storage-only, usable standalone as a `BaseRetriever` without the chat-buffer wrapper
- `top_k` control over how many relevant past documents are surfaced per `format_for_llm(query=...)` call

### Provider-agnostic embeddings
- `embed_fn` is fully caller-supplied — works with `sentence-transformers` (fully offline), an OpenAI/Azure/local-server embeddings endpoint, or a hand-written function
- The package itself never imports an embedding library, so there's no forced dependency on any particular embedding provider or model

### Storage & persistence
- SQLite-backed storage — `db_path=":memory:"` (default) for in-process/no-persistence use, or a real file path for recall across restarts
- Depends only on `autourgos-memory`, `autourgos-buffer-memory`, and `numpy` — no embedding-model or embedding-API package is pulled in

### Data integrity
- Dimension checking: `add_document()` checks the new vector's dimension against the table's existing dimension (re-read from the DB each call, not cached), raising `VectorMemoryError` on mismatch — protects against silently mixing outputs from two different embedding models in the same store
- Documented multi-process race handling: under concurrent writes from separate processes racing on a brand-new empty `db_path`, `retrieve()` defends by silently skipping any stored vector whose dimension doesn't match the query's, rather than crashing — explicitly scoped as a narrow multi-process edge case, not a single-process concern

---

## Competitor Comparison

This is a "bring your own embeddings" local vector store for agent memory, so the natural comparison set spans embedded/local vector databases (Chroma, FAISS, LanceDB) and other agent-framework vector memories.

| Capability | **autourgos-vector-memory** | Chroma (embedded mode) | FAISS | LanceDB | LangChain `VectorStoreRetrieverMemory` |
|---|---|---|---|---|---|
| Embedding provider | Bring-your-own (`embed_fn`), any shape | Built-in default embedder, or bring-your-own | None — you supply vectors directly | Bring-your-own, or built-in embedding functions | Depends on the wrapped LangChain vector store's embedding class |
| Storage backend | SQLite | Its own storage layer (in-memory, SQLite, or client-server) | None — pure in-memory index library, no persistence of its own | Columnar lakehouse format (local disk or object storage) | Whatever backing vector store is configured (Chroma, FAISS, Pinecone, etc.) |
| Persistence | Yes, via `db_path` file (or `:memory:` to opt out) | Yes, built-in | No — must be paired with your own save/load of the index | Yes, built-in, scales to large datasets | Depends on backing store |
| Metadata filtering / CRUD API | Basic (documents + metadata via `Document`) | Yes, full CRUD + metadata filtering | No — index only, no metadata layer | Yes, full CRUD + metadata filtering | Depends on backing store |
| Runs fully offline / in-process | Yes | Yes (embedded mode) | Yes | Yes | Depends on backing store |
| External service required | No | No (embedded) / optional (client-server mode) | No | No | Depends on backing store |
| Index types / ANN scaling | Simple cosine similarity scan (no ANN index) | HNSW-based ANN indexing | 10+ index types (Flat, IVF, HNSW, PQ, GPU-accelerated) — the reference library for raw speed/scale | Scales to billions of vectors via lakehouse format | Depends on backing store |
| Framework lock-in | None — plugs into any Autourgos `Agent`/`BaseRetriever` | None — usable standalone | None — usable standalone | None — usable standalone | Tied to LangChain's memory interfaces (also on a deprecation path) |
| Setup complexity | `pip install`, supply `embed_fn` | `pip install` | `pip install`, requires you to manage persistence and metadata yourself | `pip install` | `pip install langchain` + a vector store + an embeddings class |

### How to read this

- **vs. Chroma**: Chroma is a fuller embedded vector database — CRUD API, metadata filtering, ANN indexing (HNSW) — described in comparisons as "SQLite for embeddings." This package is intentionally narrower: a similarity-scan-over-SQLite store with no ANN index, trading scale/performance ceiling for a smaller footprint and a stricter "we don't touch your embedding stack" boundary (Chroma ships a default embedder; this package never does).
- **vs. FAISS**: FAISS is a raw similarity-search *library*, not a database — no persistence, no metadata, no server, and it wins decisively on raw speed and index flexibility (IVF/HNSW/PQ/GPU) at scale. This package instead bundles the missing pieces FAISS deliberately leaves out (SQLite persistence, document/metadata storage, a retriever interface) at the cost of a much simpler, non-ANN cosine scan that won't scale as far.
- **vs. LanceDB**: LanceDB is the closer "embedded database" analog to what this package is going for (local-first, no server), but built on a columnar lakehouse format explicitly designed to scale to billions of vectors — a much heavier and more scalable engine than a cosine scan over a SQLite table.
- **vs. LangChain's `VectorStoreRetrieverMemory`**: that's a thin memory adapter over whichever vector store/embeddings class you configure — flexible, but pulls in LangChain's abstractions (and inherits their deprecation churn toward LangGraph). This package has no such framework coupling and no forced embedding dependency at all.
- **When this package is the right choice**: small-to-medium agent memory stores (hundreds to low tens-of-thousands of documents) where a full vector database is overkill, the team wants to freely choose/swap the embedding function without dependency creep, and SQLite-file persistence is sufficient. For very large corpora or high-QPS ANN search, Chroma, FAISS, or LanceDB are the better fit — this package doesn't build or maintain an ANN index.

Sources:
- [Best Vector Databases 2026: Pinecone, Chroma, Qdrant & More | DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases)
- [Chroma vs FAISS — Application Database or Raw Speed Library? (2026) | MyEngineeringPath](https://myengineeringpath.dev/tools/chroma-vs-faiss/)
- [FAISS vs Chroma? Let's Settle the Vector Database Debate!](https://www.capellasolutions.com/blog/faiss-vs-chroma-lets-settle-the-vector-database-debate)
- [Best Chroma Alternatives (2026): Lightweight Vector Database](https://www.buildmvpfast.com/alternatives/chroma)
- [Vector Store Memory in LangChain - GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/vector-store-memory-in-langchain/)
