# Changelog

## [1.0.1] - 2026-09-01

- Metadata: added `maintainers` (Sonia, Vishwanil Suman) to `pyproject.toml`,
  and added Contributor badges for both to the README (Sonia's linked to
  her GitHub profile, https://github.com/dahiyasonia). No code changes.

## [1.0.0] - 2026-09-01

- Promoted to stable (`Development Status :: 5 - Production/Stable`, up from
  `4 - Beta`). No code changes since 0.2.0 -- the API (`VectorMemory`,
  `VectorRetriever`) has held steady across both prior releases, the
  concurrency and mixed-dimension bugs found in 0.2.0 are fixed and
  covered by regression tests, and the full test suite (16/16) passes.
  This release is a version/classifier bump only, marking the package
  ready for production use rather than early iteration.

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
