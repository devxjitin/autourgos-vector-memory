"""Tests for VectorRetriever and VectorMemory."""
import os

import pytest

from autourgos_memory import Document
from autourgos_vector_memory import VectorMemory, VectorMemoryError, VectorRetriever

# Deterministic 4-dim "embedding": counts of a few marker words. Similar
# texts (sharing marker words) get high cosine similarity; unrelated texts
# don't -- good enough to exercise ranking without a real embedding model.
_MARKERS = ["color", "blue", "deploy", "region"]


def fake_embed(text: str) -> list:
    lowered = text.lower()
    return [float(lowered.count(m)) + 0.01 for m in _MARKERS]


def bad_embed_empty(text: str) -> list:
    return []


def bad_embed_wrong_type(text: str) -> str:
    return "not-a-vector"


# ── VectorRetriever ─────────────────────────────────────────────────────────

def test_retrieve_ranks_by_similarity():
    r = VectorRetriever(embed_fn=fake_embed)
    r.add_document(Document(content="My favorite color is blue."))
    r.add_document(Document(content="The deploy target is us-east-1 region."))
    r.add_document(Document(content="Completely unrelated sentence."))

    results = r.retrieve("what color do I like?", top_k=1)
    assert len(results) == 1
    assert "blue" in results[0].content.lower()


def test_retrieve_empty_store_returns_empty_list():
    r = VectorRetriever(embed_fn=fake_embed)
    assert r.retrieve("anything", top_k=5) == []


def test_max_documents_evicts_oldest():
    r = VectorRetriever(embed_fn=fake_embed, max_documents=2)
    r.add_document(Document(content="first"))
    r.add_document(Document(content="second"))
    r.add_document(Document(content="third"))
    results = r.retrieve("first second third", top_k=10)
    contents = {d.content for d in results}
    assert "first" not in contents
    assert contents == {"second", "third"}


def test_dimension_mismatch_raises():
    r = VectorRetriever(embed_fn=fake_embed)
    r.add_document(Document(content="seed"))

    def wrong_dim_embed(text: str) -> list:
        return [1.0, 2.0]  # only 2 dims, store is locked to 4

    r.embed_fn = wrong_dim_embed
    with pytest.raises(VectorMemoryError):
        r.add_document(Document(content="second"))


def test_invalid_embed_fn_output_raises():
    r = VectorRetriever(embed_fn=bad_embed_empty)
    with pytest.raises(VectorMemoryError):
        r.add_document(Document(content="x"))

    r2 = VectorRetriever(embed_fn=bad_embed_wrong_type)
    with pytest.raises(VectorMemoryError):
        r2.add_document(Document(content="x"))


def test_non_callable_embed_fn_rejected():
    with pytest.raises(VectorMemoryError):
        VectorRetriever(embed_fn="not-callable")  # type: ignore[arg-type]


def test_persistence_across_reopen(tmp_path):
    db_path = str(tmp_path / "vec.db")
    r = VectorRetriever(embed_fn=fake_embed, db_path=db_path)
    r.add_document(Document(content="My favorite color is blue."))
    r.close()

    r2 = VectorRetriever(embed_fn=fake_embed, db_path=db_path)
    results = r2.retrieve("what color?", top_k=1)
    assert len(results) == 1
    assert "blue" in results[0].content.lower()


def test_clear_resets_dimension_lock():
    r = VectorRetriever(embed_fn=fake_embed)
    r.add_document(Document(content="seed"))
    r.clear()
    assert r.retrieve("seed", top_k=5) == []

    def other_dim_embed(text: str) -> list:
        return [1.0, 2.0, 3.0]

    r.embed_fn = other_dim_embed
    r.add_document(Document(content="ok now"))  # should not raise post-clear


# ── VectorMemory ─────────────────────────────────────────────────────────────

def test_vector_memory_requires_embed_fn_or_retriever():
    with pytest.raises(VectorMemoryError):
        VectorMemory()


def test_vector_memory_format_for_llm_includes_relevant_past_context():
    mem = VectorMemory(embed_fn=fake_embed, top_k=1)
    mem.add_user_message("My favorite color is blue.")
    mem.add_agent_message("Got it, blue it is.")
    for i in range(15):
        mem.add_user_message(f"filler message {i}")  # push the color turn out of the buffer

    formatted = mem.format_for_llm(query="what color do I like?")
    assert "blue" in formatted.lower()


def test_vector_memory_format_for_llm_without_query_returns_buffer_only():
    mem = VectorMemory(embed_fn=fake_embed)
    mem.add_user_message("hello")
    formatted = mem.format_for_llm()
    assert "hello" in formatted


def test_vector_memory_clear_empties_both_stores():
    mem = VectorMemory(embed_fn=fake_embed, top_k=5)
    mem.add_user_message("My favorite color is blue.")
    mem.clear()
    assert mem.retriever.retrieve("blue", top_k=5) == []
    assert mem.short_term.get_messages() == []


def test_vector_memory_with_prebuilt_retriever():
    retriever = VectorRetriever(embed_fn=fake_embed)
    mem = VectorMemory(retriever=retriever)
    mem.add_user_message("The deploy target is the region us-east-1.")
    results = mem.retriever.retrieve("which region?", top_k=1)
    assert len(results) == 1
