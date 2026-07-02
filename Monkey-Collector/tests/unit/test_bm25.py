"""Tests for the pure-Python Okapi BM25 index (top-K retrieval, no numpy)."""

from monkey_collector.pipeline.screen_matching.bm25 import Bm25Index


def test_top_k_orders_by_overlap():
    idx = Bm25Index()
    idx.add_document("0", ["a", "b", "c"])
    idx.add_document("1", ["a", "b", "z"])
    idx.add_document("2", ["x", "y", "w"])
    keys = [pk for pk, _ in idx.top_k(["a", "b", "c"], 3)]
    assert keys[0] == "0"   # shares all three query lines → highest score
    assert keys[1] == "1"   # shares a, b
    assert keys[2] == "2"   # shares none → score 0, ranked last


def test_top_k_respects_k():
    idx = Bm25Index()
    for i in range(5):
        idx.add_document(str(i), ["a"])
    assert len(idx.top_k(["a"], 2)) == 2


def test_top_k_fewer_than_k_when_corpus_small():
    idx = Bm25Index()
    idx.add_document("0", ["a"])
    assert len(idx.top_k(["a"], 5)) == 1


def test_tie_break_page_key_ascending():
    idx = Bm25Index()
    idx.add_document("3", ["a"])
    idx.add_document("1", ["a"])
    idx.add_document("2", ["a"])
    # All equal score → deterministic ascending numeric page_key.
    assert [pk for pk, _ in idx.top_k(["a"], 3)] == ["1", "2", "3"]


def test_single_doc_idf_non_negative():
    idx = Bm25Index()
    idx.add_document("0", ["a", "b"])
    pk, score = idx.top_k(["a"], 1)[0]
    assert pk == "0"
    assert score >= 0.0  # +1 IDF variant keeps a 1-doc corpus non-negative


def test_all_empty_docs_no_zero_division():
    idx = Bm25Index()
    idx.add_document("0", [])   # avgdl == 0 guard exercised
    idx.add_document("1", [])
    res = idx.top_k(["a"], 2)   # must not raise
    assert len(res) == 2
    assert all(score == 0.0 for _, score in res)


def test_empty_corpus_and_nonpositive_k():
    idx = Bm25Index()
    assert idx.top_k(["a"], 5) == []
    idx.add_document("0", ["a"])
    assert idx.top_k(["a"], 0) == []


def test_clear_resets_corpus():
    idx = Bm25Index()
    idx.add_document("0", ["a"])
    assert len(idx) == 1
    idx.clear()
    assert len(idx) == 0
    assert idx.top_k(["a"], 1) == []
