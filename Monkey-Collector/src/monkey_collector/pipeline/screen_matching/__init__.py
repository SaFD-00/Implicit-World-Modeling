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
]


def create_screen_matcher(
    extractor: ElementExtractor | None,
    enabled: bool = True,
    cluster_merge_tolerance: float = 0.2,
    max_expand_iters: int = 3,
) -> ScreenMatcher | None:
    """Build a :class:`ScreenMatcher`, or ``None`` when disabled / no extractor."""
    if not enabled or extractor is None:
        return None
    return ScreenMatcher(
        extractor,
        cluster_merge_tolerance=cluster_merge_tolerance,
        max_expand_iters=max_expand_iters,
    )
