# Changelog

## [0.1.0] - 2026-08-31

- Initial release: `VectorRetriever` (SQLite-persisted, cosine-similarity `BaseRetriever` over
  caller-supplied embeddings) and `VectorMemory` (short-term buffer + `VectorRetriever`,
  `BaseMemory`-compatible, mirrors `autourgos-semantic-memory`'s `KeywordMemory`).
- Provider-agnostic by design: the package never computes embeddings itself, only stores and
  ranks vectors produced by a caller-supplied `embed_fn`.
