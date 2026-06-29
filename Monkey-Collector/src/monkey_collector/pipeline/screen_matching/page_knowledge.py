"""Per-page element knowledge and an in-session registry.

A :class:`PageKnowledge` records, for one logical page, the extracted elements
(name + family + anchor indices), the anchor fingerprints used to re-match the
page on later visits (``key_elements``), and the leftover-UI fingerprints
(``extra_uis``) so unaccounted interactables don't spuriously fork a page.
Mirrors MobileGPT-V2 ``models.PageKnowledge`` + ``KnowledgeRegistry`` with
Monkey-Collector vocabulary (element / key_element instead of subtask /
trigger_ui).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from monkey_collector.llm.element_extractor import ExtractedElement
from monkey_collector.pipeline.screen_matching.ui_attributes import UIAttributes


@dataclass
class PageKnowledge:
    """Stored knowledge for one logical page (keyed by ``page_key``)."""

    page_key: str
    elements: list[ExtractedElement] = field(default_factory=list)
    # element name -> anchor fingerprints (the re-match key for this page).
    key_elements: dict[str, list[UIAttributes]] = field(default_factory=dict)
    # leftover interactable fingerprints not owned by any element's anchors.
    extra_uis: list[UIAttributes] = field(default_factory=list)

    @property
    def element_names(self) -> set[str]:
        return {e.name for e in self.elements}


class KnowledgeRegistry:
    """In-memory store of page knowledge for one collection session."""

    def __init__(self) -> None:
        self._pages: dict[str, PageKnowledge] = {}

    def add(self, page: PageKnowledge) -> None:
        self._pages[page.page_key] = page

    def get(self, page_key: str) -> PageKnowledge | None:
        return self._pages.get(page_key)

    def all_page_keys(self) -> list[str]:
        return list(self._pages.keys())

    def __len__(self) -> int:
        return len(self._pages)
