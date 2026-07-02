"""Okapi BM25 index over per-page element-line documents (Mobile3M port).

Ports the reference ``BM25`` class (Mobile3M ``corpus/googleCreatDataset``,
``k1=1.5, b=0.75``) with two deliberate changes:

* **top-K, not top-1.** The reference ``get_max`` returns a single ``np.argmax``;
  the paper describes retrieving the top-5 candidates for precise verification,
  so :meth:`top_k` returns the K highest-scoring pages (score-desc). No numpy —
  a plain sort replaces ``np.argmax``.
* **full IDF recompute on every add.** The reference ``appendItem`` mutates only
  ``idf`` using a stale ``len(self.docs)`` (it never appends to ``docs`` /
  ``doc_freqs`` / ``doc_len``), which desyncs df/avgdl. Page counts per session
  are small (tens–low-hundreds), so :meth:`add_document` rebuilds df/idf/avgdl
  from scratch — cheap and correct.

A "term" is a whole element-line string (the reference tokenizes on
``readlines()`` of the per-page HTML, i.e. one line == one token); we keep that
granularity. The ``+1`` IDF variant keeps IDF ≥ 0 even for a single-document
corpus, and :meth:`_score` guards ``avgdl == 0`` (all-empty docs) against a
ZeroDivisionError.
"""

from __future__ import annotations

import math
from collections import Counter


def _sort_key(page_key: str) -> tuple[int, object]:
    """Deterministic tie-break: numeric page_keys ascending, then strings.

    Matcher page_keys are bare integers (``"0"``, ``"1"``, …); order them
    numerically. Any non-numeric key sorts after all numeric ones, by string.
    """
    return (0, int(page_key)) if page_key.isdigit() else (1, page_key)


class Bm25Index:
    """In-memory Okapi BM25 index keyed by ``page_key``.

    One document per page (its element-line list, frozen at page creation). The
    corpus is a live mirror of the matcher's registry: :meth:`add_document` per
    minted page, :meth:`clear` on session reset, full re-add on resume.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._page_keys: list[str] = []
        self._freqs: list[Counter] = []
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._page_keys)

    def clear(self) -> None:
        """Drop the whole corpus (called on session reset)."""
        self._page_keys.clear()
        self._freqs.clear()
        self._doc_len.clear()
        self._idf.clear()
        self._avgdl = 0.0

    def add_document(self, page_key: str, lines: list[str]) -> None:
        """Add one page's element-line document and rebuild df/idf/avgdl."""
        self._page_keys.append(page_key)
        self._freqs.append(Counter(lines))
        self._doc_len.append(len(lines))
        self._recompute()

    def _recompute(self) -> None:
        n = len(self._freqs)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        df: Counter = Counter()
        for freq in self._freqs:
            df.update(freq.keys())  # each term counted once per document
        # +1 variant → IDF ≥ 0 even when df == n (single-doc corpus).
        self._idf = {
            term: math.log((n - d + 0.5) / (d + 0.5) + 1) for term, d in df.items()
        }

    def _score(self, doc_idx: int, query: list[str]) -> float:
        freq = self._freqs[doc_idx]
        # Length normalization; guard avgdl == 0 (all documents empty).
        norm = (self._doc_len[doc_idx] / self._avgdl) if self._avgdl else 0.0
        denom_len = self.k1 * (1 - self.b + self.b * norm)
        score = 0.0
        for term in query:
            f = freq.get(term, 0)
            if f == 0:
                continue
            idf = self._idf.get(term, 0.0)
            score += idf * f * (self.k1 + 1) / (f + denom_len)
        return score

    def top_k(self, query: list[str], k: int) -> list[tuple[str, float]]:
        """The ``k`` highest-scoring pages for *query*, ``(page_key, score)``.

        Sorted by score descending, ties broken by ascending page_key
        (deterministic). Returns fewer than ``k`` when the corpus is smaller;
        an empty corpus or ``k <= 0`` yields ``[]``. Zero-score candidates are
        included (as in the reference ``argmax``) — the caller's element/pixel
        gate is the real filter.
        """
        if not self._page_keys or k <= 0:
            return []
        scored = [
            (self._page_keys[i], self._score(i, query))
            for i in range(len(self._page_keys))
        ]
        scored.sort(key=lambda t: (-t[1], _sort_key(t[0])))
        return scored[:k]
