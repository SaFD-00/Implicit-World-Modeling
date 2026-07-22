"""Per-page identity knowledge and an in-session registry.

A :class:`PageKnowledge` records, for one logical page, the element-line
document that is its BM25 identity and the activity it was minted under (the
merge guard's package source). Mirrors MobileGPT-V2 ``models.PageKnowledge`` +
``KnowledgeRegistry`` in shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageKnowledge:
    """Stored knowledge for one logical page (keyed by ``page_key``)."""

    page_key: str
    # BM25 document for this page: the normalized element-line list serialized
    # from the encoded XML at page creation (see element_lines.py). Frozen at
    # first sighting — the page's identity for BM25 retrieval + element-diff /
    # Jaccard verification. Serialized to page.json (additive/back-compat).
    element_lines: list[str] = field(default_factory=list)
    # Activity label (``package/window.Class``) this page was minted under. Only
    # its PACKAGE part is load-bearing: it is what the BM25 merge guard compares
    # against the current screen's, so a launcher/home frame can never merge into
    # an app's page. Serialized to page.json (additive/back-compat); a legacy file
    # without it is refilled on resume from the first observation's meta.
    first_activity: str = ""
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

    def to_dict(self) -> dict:
        """Serialize page identity for ``page.json``.

        Deliberately excludes ``luminance_features``/``next_observation_num``
        — both are re-derived (from each observation's saved screenshot / the
        on-disk observation directory listing), never persisted directly, so a
        stale in-memory image or counter can never desync from disk.
        """
        return {
            "page_key": self.page_key,
            "element_lines": list(self.element_lines),
            "first_activity": self.first_activity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PageKnowledge:
        """Inverse of :meth:`to_dict`. ``luminance_features``/
        ``next_observation_num`` are left at their defaults — the caller
        (rehydration) populates them separately from the on-disk observations.
        Legacy keys (``elements``/``key_elements``/``extra_uis``) are ignored.
        """
        return cls(
            page_key=d["page_key"],
            element_lines=list(d.get("element_lines", [])),
            first_activity=str(d.get("first_activity", "")),
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
