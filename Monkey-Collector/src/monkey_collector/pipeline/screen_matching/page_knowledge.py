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
    # Stage-0 luminance prefilter observations: (observation_num, resized
    # BT.601 L-mode PIL image) pairs, one per sighting that became a new
    # observation (a reused observation is never appended again). The number
    # travels with the fingerprint so FIFO eviction of the cap never desyncs
    # the in-memory index from the durable on-disk observation directory it
    # was derived from. In-memory only — session-scoped like the rest of the
    # registry, never serialized directly (``to_dict`` excludes it; a later
    # persistence stage re-derives it from each observation's saved
    # screenshot). Typed ``list`` to avoid importing PIL here; capped by
    # ScreenMatcher._MAX_LUMINANCE_OBS.
    luminance_features: list = field(default_factory=list)
    # Next observation number to allocate for this page. In-memory working
    # counter for a live session; a later persistence stage recomputes it from
    # on-disk observation directories on resume, so it is never serialized.
    # 0 for a freshly-created page — its first observation (obs 0) is
    # allocated by the same call that registers the page.
    next_observation_num: int = 0

    @property
    def element_names(self) -> set[str]:
        return {e.name for e in self.elements}

    def to_dict(self) -> dict:
        """Serialize page identity for ``page.json`` (a page's frozen anchors).

        Deliberately excludes ``luminance_features``/``next_observation_num``
        — both are re-derived (from each observation's saved screenshot / the
        on-disk observation directory listing), never persisted directly, so a
        stale in-memory image or counter can never desync from disk.
        """
        return {
            "page_key": self.page_key,
            "elements": [
                {
                    "name": e.name,
                    "description": e.description,
                    "parameters": dict(e.parameters),
                    "element_index": list(e.element_index),
                    "key_element_index": list(e.key_element_index),
                }
                for e in self.elements
            ],
            "key_elements": {
                name: [ui.to_dict() for ui in ui_list]
                for name, ui_list in self.key_elements.items()
            },
            "extra_uis": [ui.to_dict() for ui in self.extra_uis],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PageKnowledge:
        """Inverse of :meth:`to_dict`. ``luminance_features``/
        ``next_observation_num`` are left at their defaults — the caller
        (rehydration) populates them separately from the on-disk observations.
        """
        return cls(
            page_key=d["page_key"],
            elements=[ExtractedElement(**e) for e in d.get("elements", [])],
            key_elements={
                name: [UIAttributes.from_attrib_dict(u) for u in ui_list]
                for name, ui_list in d.get("key_elements", {}).items()
            },
            extra_uis=[UIAttributes.from_attrib_dict(u) for u in d.get("extra_uis", [])],
        )


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
