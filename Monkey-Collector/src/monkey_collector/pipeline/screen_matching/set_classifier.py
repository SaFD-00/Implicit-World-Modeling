"""Cluster-assignment set classifier.

Ports MobileGPT-V2 Node-Clustering's ``find_best_match`` step-3 classification
(``subtask_trigger_ui_strategy.py:194-261``). Given A = the element-name set
recognized on the current screen (supported ∪ newly-extracted) and B = a stored
page's element-name set, decide the match type:

    A == B                       -> EQSET           (merge into B)
    A ⊋ B (both non-empty)       -> SUPERSET_MERGE   (containment → always merge)
    A ⊊ B (both non-empty)       -> SUBSET_MERGE     (containment → always merge)
    partial overlap              -> OVERLAP_MERGE iff two-sided tolerance band
                                    passes, else OVERLAP_NEW
    A ∩ B = ∅, or B = ∅          -> DISJOINT         (new page)

Containment always merges: a SUPERSET/SUBSET match is the same screen at a
different scroll/render state (collapses the scroll-reveal over-split). BOTH
containment branches require a non-empty B: an empty stored page (B=∅) is never
a merge target, so an accidentally-registered empty page can never become a sink
that swallows every later screen (the blackhole). The two-sided tolerance band
is retained ONLY for OVERLAP — the case where a rich new screen would wrongly
collapse into a small stored page on an incidental shared element (e.g. a
persistent bottom bar).
"""

from __future__ import annotations

from dataclasses import dataclass

MERGE_LABELS = frozenset({"EQSET", "SUPERSET_MERGE", "SUBSET_MERGE", "OVERLAP_MERGE"})


@dataclass(frozen=True)
class Classification:
    match_type: str
    is_merge: bool


def classify(a_names: set[str], b_names: set[str], tolerance: float) -> Classification:
    """Classify A vs B into a cluster-assignment match type.

    ``tolerance`` (``a``) gates only OVERLAP: with N=|B|, M=|A|, the band on
    |A∩B| is ``[round(N-aN), round(N+aN)]`` for B and ``[round(M-aM),
    round(M+aM)]`` for A; OVERLAP merges only when |A∩B| falls in BOTH bands.
    """
    a = a_names
    b = b_names
    inter = a & b
    inter_count = len(inter)

    if a == b:
        match_type = "EQSET"
    elif a and b and a > b:
        # b non-empty guard: an empty stored page (B=∅) must NOT swallow this
        # screen as a superset — that is the blackhole. With B=∅ this falls
        # through to DISJOINT, so any accidentally-registered empty page is
        # harmless (the next real screen forks its own node) rather than a sink.
        match_type = "SUPERSET_MERGE"  # containment → always merge (scroll-reveal)
    elif a and b and a < b:
        # A must be non-empty: empty A means we recognized nothing → DISJOINT.
        match_type = "SUBSET_MERGE"  # containment → always merge (partial render)
    elif inter:
        n = len(b)
        m = len(a)
        low = round(n - tolerance * n)
        high = round(n + tolerance * n)
        in_band = low <= inter_count <= high
        low_a = round(m - tolerance * m)
        high_a = round(m + tolerance * m)
        in_band_a = low_a <= inter_count <= high_a
        match_type = "OVERLAP_MERGE" if (in_band and in_band_a) else "OVERLAP_NEW"
    else:
        match_type = "DISJOINT"

    return Classification(match_type=match_type, is_merge=match_type in MERGE_LABELS)
