# autourgos-vector-memory

[![Framework: Autourgos](https://img.shields.io/badge/Framework-Autourgos-orange.svg)](https://github.com/devxjitin)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/autourgos-vector-memory/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/devxjitin/autourgos-vector-memory/blob/main/LICENSE)
[![Author](https://img.shields.io/badge/Author-Jitin%20Kumar%20Sengar-blue.svg)](https://github.com/devxjitin)

Local, persisted, **provider-agnostic** embedding (vector) memory for [Autourgos](https://github.com/devxjitin)
agents. You bring the embedding function — a local model, a cloud API, anything shaped
`fn(text: str) -> Sequence[float]` — this package only stores vectors in SQLite and ranks
them by cosine similarity. No embedding-provider dependency, no vector-database server.

Upgrade path from `autourgos-semantic-memory`'s TF-IDF keyword matching when you need real
semantic recall (queries that are related in meaning but share no keywords).

```python
from autourgos_vector_memory import VectorMemory

def embed(text: str) -> list[float]:
    # call your local model, or a cloud embeddings API — your choice
    ...

memory = VectorMemory(embed_fn=embed, db_path="agent_memory.db", top_k=3)

memory.add_user_message("My favorite color is blue.")
memory.add_agent_message("Got it, blue it is.")

# ... much later, possibly a fresh process (db_path persists) ...
print(memory.format_for_llm(query="what color do I like?"))
```

---

## Table of Contents

- [Install](#install)
- [Why provider-agnostic](#why-provider-agnostic)
- [Quick Start](#quick-start)
- [VectorRetriever (storage-only)](#vectorretriever-storage-only)
- [Constructor Reference](#constructor-reference)
- [License](#license)

---

## Install

```bash
pip install autourgos-vector-memory
```

Depends on `autourgos-memory`, `autourgos-buffer-memory`, and `numpy`. No embedding-model
or embedding-API package is pulled in — you supply `embed_fn`.

---

## Why provider-agnostic

Every other piece of this framework works the same way — `Agent(llm=...)` accepts any
object with `.invoke()`, not a fixed provider. `VectorMemory`/`VectorRetriever` follow the
same rule for embeddings: `embed_fn` can wrap `sentence-transformers` running fully
offline, an OpenAI/Azure/local-server embeddings endpoint, or a hand-written function —
this package never imports an embedding library itself.

```python
# Local, offline (requires sentence-transformers installed separately)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: model.encode(text).tolist()

# Or a cloud API (requires autourgos-openaichat or the openai SDK installed separately)
from openai import OpenAI
client = OpenAI()
embed_fn = lambda text: client.embeddings.create(
    model="text-embedding-3-small", input=text
).data[0].embedding
```

---

## Quick Start

```python
from autourgos_agent import Agent
from autourgos_vector_memory import VectorMemory

memory = VectorMemory(embed_fn=embed_fn, db_path="agent_memory.db")
agent = Agent(llm=llm, memory=memory)

agent.invoke("Remember that my deploy target is us-east-1.")
# ... many turns and tool calls later ...
agent.invoke("What region do I deploy to?")  # recalled via similarity, not exact keywords
```

`db_path=":memory:"` (the default) keeps everything in RAM for the process lifetime. Pass a
real file path for recall across restarts.

---

## VectorRetriever (storage-only)

If you don't need the chat-buffer wrapper, use `VectorRetriever` directly as a
`BaseRetriever`:

```python
from autourgos_vector_memory import VectorRetriever
from autourgos_memory import Document

retriever = VectorRetriever(embed_fn=embed_fn, db_path="notes.db")
retriever.add_document(Document(content="The deploy target is us-east-1.", metadata={"source": "config"}))

results = retriever.retrieve("which AWS region do we use?", top_k=3)
for doc in results:
    print(doc.score, doc.content)
```

All documents in a given `db_path` must embed to the same dimension. `add_document()` checks
the dimension of every new vector against the table's existing dimension (re-read from the
database each call, not just cached in memory) and raises `VectorMemoryError` on a mismatch.
Use a fresh `db_path` when you switch embedding models.

Under true concurrent writes from separate processes racing on a brand-new, empty `db_path`,
it's possible for two differently-sized vectors to both be inserted before either write is
visible to the other. `retrieve()` defends against this: it silently skips any stored vector
whose dimension doesn't match the query's, instead of crashing, so the rest of the store stays
queryable even if this happens. This is a narrow multi-process edge case, not something a
single-process application needs to worry about.

---

## Constructor Reference

### `VectorMemory`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embed_fn` | `callable` | required (unless `retriever=` given) | `fn(text: str) -> Sequence[float]` |
| `short_term` | `BaseMemory` | `RuntimeShortTermMemory(max_messages=10)` | Recent-turns buffer, always included |
| `retriever` | `VectorRetriever` | built from `embed_fn`/`db_path`/`max_documents` | Pass a pre-built retriever instead |
| `db_path` | `str` | `":memory:"` | SQLite file path, or `:memory:` for no persistence |
| `top_k` | `int` | `3` | Relevant past documents surfaced per `format_for_llm(query=...)` call |
| `max_documents` | `int`, optional | `None` | Oldest documents dropped once this count is exceeded |

### `VectorRetriever`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embed_fn` | `callable` | required | `fn(text: str) -> Sequence[float]` |
| `db_path` | `str` | `":memory:"` | SQLite file path, or `:memory:` for no persistence |
| `max_documents` | `int`, optional | `None` | Oldest documents dropped once this count is exceeded |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
