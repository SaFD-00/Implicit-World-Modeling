"""BM25 unique-page screen matching (Mobile3M mechanism).

Serialize the encoded XML to an element-line document, retrieve BM25 top-K
candidate pages, and confirm the first candidate that passes a conjunctive gate
(element criterion AND pixel gate). This decides page identity — LLM-free — for
both the page graph and exploration.
"""

from __future__ import annotations

from monkey_collector.pipeline.screen_matching.page_knowledge import (
    KnowledgeRegistry,
    PageKnowledge,
)
from monkey_collector.pipeline.screen_matching.rehydrate import (
    rehydrate_screen_matcher,
)
from monkey_collector.pipeline.screen_matching.screen_matcher import (
    ScreenMatch,
    ScreenMatcher,
)

__all__ = [
    "ScreenMatcher",
    "ScreenMatch",
    "PageKnowledge",
    "KnowledgeRegistry",
    "create_screen_matcher",
    "rehydrate_screen_matcher",
]


def create_screen_matcher(
    luminance_prefilter: bool = False,
    luminance_threshold: int = 10,
    screenshot_diff_threshold: float = 0.02,
    luminance_low_res_width: int = 100,
    persist_filtered: bool = False,
    bm25_top_k: int = 5,
    element_criterion: str = "diff",
    element_diff_max: int = 5,
    element_jaccard_min: float = 0.5,
    page_pixel_diff_threshold: float = 0.3,
    canvas_merge: bool = True,
    canvas_min_area_frac: float = 0.7,
    package_guard: bool = True,
) -> ScreenMatcher | None:
    """Build a :class:`ScreenMatcher`, or ``None`` when nothing distinguishes pages.

    The BM25 matcher decides page identity without the LLM (element-line
    document + element/pixel verification). ``None`` is returned only when the
    luminance prefilter is off — the historical opt-out that routes page
    identity to the structural ``page_graph.get_or_create_page`` fallback.
    (BM25 needs no dependency, so this gating is conservative, not required.)
    """
    if not luminance_prefilter:
        return None
    return ScreenMatcher(
        luminance_prefilter=luminance_prefilter,
        luminance_threshold=luminance_threshold,
        screenshot_diff_threshold=screenshot_diff_threshold,
        luminance_low_res_width=luminance_low_res_width,
        persist_filtered=persist_filtered,
        bm25_top_k=bm25_top_k,
        element_criterion=element_criterion,
        element_diff_max=element_diff_max,
        element_jaccard_min=element_jaccard_min,
        page_pixel_diff_threshold=page_pixel_diff_threshold,
        canvas_merge=canvas_merge,
        canvas_min_area_frac=canvas_min_area_frac,
        package_guard=package_guard,
    )
