"""LLM-based screen element semantic grouping ("화면 나누기").

Ports the ``_gen_state_semantic_info`` idea from the reference ``LLM-Explorer``
project: given the current screen's element list, ask the LLM to (1) describe
the page, (2) summarize its control elements, and (3) group element indices
that share the same function (e.g. repeated list items, similar toggles,
redundant labels) — i.e. divide a single screen into semantic regions.

The grouping is computed over the **encoded** screen representation
(``encode_to_html_xml``), the same artifact saved as ``{step}_encoded.xml``,
so the ``index`` values in the returned groups line up with that file. Results
are persisted per screen as a session annotation (``{step}_groups.json``) for
world-model training data — they do not alter the exploration loop.

Cost is incurred per LLM call, so identical screen structures reuse the prior
result via an in-memory cache keyed on the structure-only XML (mirrors
LLM-Explorer's ``text_representation_frame`` reuse).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.xml.structured_parser import encode_to_html_xml, hierarchy_parse

if TYPE_CHECKING:
    from monkey_collector.llm.client import LLMClient

SYSTEM_PROMPT = (
    "You are a mobile GUI analyst. You are given the elements of a single "
    "Android app screen as HTML-like tags, each carrying an integer `index` "
    "attribute.\n"
    "Your job is to divide the screen into semantic groups: identify sets of "
    "element indices that share the same function or lead to the same kind of "
    "outcome (for example: items of the same list, similar toggles/checkboxes, "
    "repeated date or category labels, a row of equivalent navigation tabs).\n"
    "Elements with clearly different layouts or different redirect targets are "
    "unlikely to share a function. Singletons may be left ungrouped."
)

USER_PROMPT_TEMPLATE = (
    "Current screen elements:\n{screen_xml}\n\n"
    "Respond with ONLY a JSON object in exactly this shape (no markdown, no prose):\n"
    "{{\n"
    '  "page_description": "<short description of the page function, < 20 words>",\n'
    '  "elements_description": "<short comma-separated summary of main controls>",\n'
    '  "groups": [\n'
    '    {{"indices": [<int>, ...], "function": "<what this group does>"}}\n'
    "  ]\n"
    "}}"
)

# A group must contain at least this many elements to be meaningful.
MIN_GROUP_SIZE = 2

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _empty_grouping(model: str) -> dict:
    return {
        "model": model,
        "page_description": "",
        "elements_description": "",
        "groups": [],
    }


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


class ScreenGrouper:
    """Groups same-function elements of a screen via the shared LLM client."""

    def __init__(self, llm_client: LLMClient):
        self._client = llm_client
        # structure-only XML -> previously computed grouping (per session)
        self._cache: dict[str, dict] = {}

    def group(self, raw_xml: str) -> dict:
        """Return a semantic grouping for *raw_xml*.

        Never raises: parsing/model failures degrade to an empty grouping so
        the collection loop is unaffected.
        """
        model = getattr(self._client, "model", "")
        try:
            screen_xml = encode_to_html_xml(raw_xml) if raw_xml else ""
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Screen grouping: encode failed ({e})")
            return _empty_grouping(model)

        if not screen_xml:
            return _empty_grouping(model)

        cache_key = self._structure_key(raw_xml)
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        grouping = self._query(screen_xml, model)

        if cache_key is not None:
            self._cache[cache_key] = grouping
        return grouping

    def _query(self, screen_xml: str, model: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(screen_xml=screen_xml)},
        ]
        try:
            response = self._client.chat(
                messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                agent="screen_grouper",
            )
        except Exception as e:
            logger.warning(f"Screen grouping LLM call failed ({e})")
            return _empty_grouping(model)

        return self._parse(response, model)

    def _parse(self, response: str, model: str) -> dict:
        if not response:
            return _empty_grouping(model)
        try:
            data = json.loads(_extract_json(response))
        except (ValueError, TypeError) as e:
            logger.warning(f"Screen grouping: malformed JSON ({e})")
            return _empty_grouping(model)

        if not isinstance(data, dict):
            return _empty_grouping(model)

        groups = []
        for g in data.get("groups", []) or []:
            if not isinstance(g, dict):
                continue
            indices = [i for i in (g.get("indices") or []) if isinstance(i, int)]
            if len(indices) < MIN_GROUP_SIZE:
                continue
            groups.append({"indices": indices, "function": str(g.get("function", ""))})

        return {
            "model": model,
            "page_description": str(data.get("page_description", "")),
            "elements_description": str(data.get("elements_description", "")),
            "groups": groups,
        }

    @staticmethod
    def _structure_key(raw_xml: str) -> str | None:
        try:
            key = hierarchy_parse(raw_xml)
        except Exception:  # pragma: no cover - defensive
            return None
        return key or None


def create_screen_grouper(
    llm_client: LLMClient | None,
    enabled: bool = True,
) -> ScreenGrouper | None:
    """Build a :class:`ScreenGrouper`, or ``None`` when disabled / no client."""
    if not enabled or llm_client is None:
        return None
    return ScreenGrouper(llm_client)
