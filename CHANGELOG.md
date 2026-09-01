# Changelog

## [0.2.0] - 2026-09-01

- Fixed: `add_document()` now re-derives the table's dimension from the
  database itself under the lock, instead of trusting only the in-memory
  `_dim` cache — closes a window where two processes racing on an empty
  `db_path` could each insert a differently-sized vector before either
  committed.
- Fixed: `retrieve()` now filters out any stored vector whose length
  doesn't match the query's dimension before building the similarity
  matrix, instead of crashing with a ragged-sequence `ValueError` if a
  mixed-dimension table ever occurred.
- Added: `VectorRetriever` supports the context-manager protocol
  (`with VectorRetriever(...) as r:`), closing its connection automatically.

## [0.1.0] - 2026-08-31

- Initial release: `VectorRetriever` (SQLite-persisted, cosine-similarity `BaseRetriever` over
  caller-supplied embeddings) and `VectorMemory` (short-term buffer + `VectorRetriever`,
  `BaseMemory`-compatible, mirrors `autourgos-semantic-memory`'s `KeywordMemory`).
- Provider-agnostic by design: the package never computes embeddings itself, only stores and
  ranks vectors produced by a caller-supplied `embed_fn`.
