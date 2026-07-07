"""Single-call screen element extractor.

Replaces the old ``ScreenGrouper``. One LLM call per screen emits, for each
element, BOTH the full same-function family (``element_index``) and the
representative anchor(s) (``key_element_index``) — collapsing MobileGPT-V2's
two-agent (SubtaskExtractor + TriggerUI) pipeline into one. Reuses the shared
:class:`~monkey_collector.llm.client.LLMClient`; never raises (returns ``[]`` on
any failure) so the collection loop is unaffected. Cost rows are attributed to
the ``element_extractor`` agent label.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.llm.prompts import element_extractor_prompt

if TYPE_CHECKING:
    from monkey_collector.llm.client import LLMClient

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# In-loop element extraction runs on the matcher's hot path, so it is bounded
# to keep a runaway/hung LLM call from stalling the whole collection loop.
# Normal screens emit ~1000-4600 output tokens; a step-30 field incident spiked
# to 13151 tokens and blocked the main loop for ~4 minutes. 6000 caps runaway
# generation without truncating a normal screen; the timeout bounds a hung
# provider connection; max_retries=1 keeps a tight timeout from tripling the
# wall-clock across the SDK's default retries. Truncation/timeout degrade
# gracefully — `extract()` catches everything and returns [].
_EXTRACT_MAX_TOKENS = 6000
_EXTRACT_TIMEOUT = 60.0
_EXTRACT_MAX_RETRIES = 1


@dataclass(frozen=True)
class ExtractedElement:
    """One extracted screen element.

    ``element_index`` is the full same-function family (every member index);
    ``key_element_index`` is the representative anchor(s) (a subset). Both live
    in the encoded XML index space.
    """

    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)
    element_index: list[int] = field(default_factory=list)
    key_element_index: list[int] = field(default_factory=list)


def _coerce_index_list(raw) -> list[int]:
    """Coerce a raw index value into a deduped, non-negative int list (order kept).

    ``None`` / scalar are tolerated; negatives and non-ints are dropped. Twin of
    MobileGPT-V2 ``subtask_extractor_agent._coerce_index_list``.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    seen: set[int] = set()
    out: list[int] = []
    for item in raw:
        try:
            idx = int(item)
        except (ValueError, TypeError):
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _extract_json(raw: str) -> str:
    """Pull a JSON object out of a model response, tolerating code fences."""
    fenced = _JSON_FENCE.search(raw)
    if fenced:
        return fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


class ElementExtractor:
    """Extracts screen elements (family + anchor indices) via the shared LLM client."""

    def __init__(self, llm_client: LLMClient):
        self._client = llm_client

    def extract(
        self,
        encoded_xml: str,
        known_elements: list[ExtractedElement] | None = None,
        screenshot_path: str | None = None,
    ) -> list[ExtractedElement]:
        """Extract elements from *encoded_xml*.

        Args:
            encoded_xml: Encoded HTML-like screen XML (full, or a masked partial
                view for the expand step).
            known_elements: Exclusion list; when non-empty the model returns only
                elements NOT overlapping these (always-on extract / expand).
            screenshot_path: Unused here (kept for signature parity / future
                vision support); the shared client is text-only.

        Returns:
            List of :class:`ExtractedElement`. Empty is valid (nothing new).
        """
        if not encoded_xml:
            return []
        known = list(known_elements or [])
        known_names = [e.name for e in known]
        messages = element_extractor_prompt.get_messages(encoded_xml, known_names)

        try:
            response = self._client.chat(
                messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                agent="element_extractor",
                max_tokens=_EXTRACT_MAX_TOKENS,
                timeout=_EXTRACT_TIMEOUT,
                max_retries=_EXTRACT_MAX_RETRIES,
            )
        except Exception as e:  # noqa: BLE001 — never fail the matcher
            logger.warning(f"element_extractor: query failed, returning []: {e}")
            return []

        items = self._parse_items(response)

        # Local import breaks the package-init import cycle (screen_matching's
        # __init__ pulls page_knowledge → element_extractor).
        from monkey_collector.pipeline.screen_matching.ui_attributes import (
            extract_interactable_indexes,
        )

        excluded = set(known_names)
        on_screen = set(extract_interactable_indexes(encoded_xml))
        elements: list[ExtractedElement] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or name in excluded:
                continue
            family = _coerce_index_list(item.get("element_index"))
            key = _coerce_index_list(item.get("key_element_index"))
            # Anchors must be members of the family; drop strays.
            key = [k for k in key if k in family] if family else key
            # Fallback: no anchor but a family exists → lowest on-screen member
            # (mirrors V2 TriggerUIAgent's min-candidate fallback).
            if not key and family:
                present = [i for i in family if i in on_screen]
                if present:
                    key = [min(present)]
            elements.append(
                ExtractedElement(
                    name=name,
                    description=str(item.get("description", "")),
                    parameters=item.get("parameters", {}) if isinstance(item.get("parameters"), dict) else {},
                    element_index=family,
                    key_element_index=key,
                )
            )
        return elements

    def _parse_items(self, response: str) -> list:
        if not response:
            return []
        try:
            data = json.loads(_extract_json(response))
        except (ValueError, TypeError) as e:
            logger.warning(f"element_extractor: malformed JSON ({e})")
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("elements", "subtasks", "result", "results", "items", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [data]
        return []


def create_element_extractor(llm_client: LLMClient | None) -> ElementExtractor | None:
    """Build an :class:`ElementExtractor`, or ``None`` when no client is available."""
    if llm_client is None:
        return None
    return ElementExtractor(llm_client)
