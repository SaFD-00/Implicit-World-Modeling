"""Element-set screen matching (MobileGPT-V2 Node-Clustering port).

Extract screen elements (same-function family + representative anchor) in one
LLM call, match anchor fingerprints against the live screen, expand over the
leftover UI, and set-assign the result to an existing or new page. This element
set drives page identity for both the page graph and exploration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monkey_collector.pipeline.screen_matching.page_knowledge import (
    KnowledgeRegistry,
    PageKnowledge,
)
from monkey_collector.pipeline.screen_matching.rehydrate import (
    rehydrate_screen_matcher,
)
from monkey_collector.pipeline.screen_matching.screen_matcher import (
    ElementFamily,
    ScreenMatch,
    ScreenMatcher,
)
from monkey_collector.pipeline.screen_matching.ui_attributes import MatchResult, UIAttributes

if TYPE_CHECKING:
    from monkey_collector.llm.element_extractor import ElementExtractor

__all__ = [
    "ScreenMatcher",
    "ScreenMatch",
    "ElementFamily",
    "PageKnowledge",
    "KnowledgeRegistry",
    "UIAttributes",
    "MatchResult",
    "create_screen_matcher",
    "rehydrate_screen_matcher",
]


def create_screen_matcher(
    extractor: ElementExtractor | None,
    cluster_merge_tolerance: float = 0.2,
    max_expand_iters: int = 3,
    luminance_prefilter: bool = False,
    luminance_threshold: int = 10,
    screenshot_diff_threshold: float = 0.02,
    luminance_low_res_width: int = 100,
    persist_filtered: bool = False,
) -> ScreenMatcher | None:
    """Build a :class:`ScreenMatcher`, or ``None`` when neither backend is active.

    With an *extractor* the matcher runs full element-set page matching. Without
    one it still runs a **prefilter-only** mode (structural fingerprint +
    luminance prefilter + observation dedup, no LLM call) as long as
    *luminance_prefilter* is on — this is what keeps page/observation dedup
    working when LLM element extraction is disabled. Only when there is neither
    an extractor nor the luminance prefilter is there nothing for the matcher to
    do, and ``None`` is returned (the loop then falls back to structural-only
    page identity via ``page_graph.get_or_create_page``).
    """
    if extractor is None and not luminance_prefilter:
        return None
    return ScreenMatcher(
        extractor,
        cluster_merge_tolerance=cluster_merge_tolerance,
        max_expand_iters=max_expand_iters,
        luminance_prefilter=luminance_prefilter,
        luminance_threshold=luminance_threshold,
        screenshot_diff_threshold=screenshot_diff_threshold,
        luminance_low_res_width=luminance_low_res_width,
        persist_filtered=persist_filtered,
    )
