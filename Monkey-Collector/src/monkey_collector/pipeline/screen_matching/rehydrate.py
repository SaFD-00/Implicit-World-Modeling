"""Resume rehydration: rebuild a ScreenMatcher's page/observation knowledge
from disk after a session resume.

Before this module existed, resuming a session unconditionally reset the
matcher (``collector.py``), silently re-discovering every previously-seen page
as "new" again even though the durable ``data/{package}/pages/`` tree already
held its anchors, elements, and observations. This module closes that gap: it
walks that tree (via :class:`~monkey_collector.storage.DataWriter`) and
repopulates the matcher's registry, structural exact-match cache
(``_fp_to_key``), and per-page observation counters — the luminance
fingerprints are re-derived from each observation's saved screenshot rather
than cached separately, keeping the "pure PIL, no extra cache artifact" stance
the luminance prefilter was designed with.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from loguru import logger

from monkey_collector.domain.page_graph import compute_xml_fingerprint
from monkey_collector.pipeline.screen_matching.luminance import (
    extract_luminance_features,
)
from monkey_collector.pipeline.screen_matching.page_knowledge import PageKnowledge
from monkey_collector.pipeline.screen_matching.screen_matcher import ScreenMatcher

if TYPE_CHECKING:
    from monkey_collector.storage import DataWriter


def rehydrate_screen_matcher(matcher: ScreenMatcher, writer: DataWriter) -> None:
    """Rebuild *matcher*'s knowledge from *writer*'s on-disk pages/observations.

    Call once, right after ``matcher.reset()``, only when resuming an existing
    session (a fresh session's ``pages/`` tree is empty, so this is a natural
    no-op then too). Best-effort per page/observation: a malformed
    ``page.json`` or an unreadable screenshot logs a warning and is skipped
    rather than aborting the whole resume — matches the project's
    "extraction/matching failure never breaks collection" posture elsewhere.
    """
    pages: dict[str, PageKnowledge] = {}
    fp_to_key: dict[tuple[str, str], tuple[str, int]] = {}
    max_page_idx = -1

    for page_key in writer.list_pages():
        try:
            page = writer.load_page_knowledge(page_key)
        except Exception as e:
            logger.warning(f"rehydrate: {page_key}/page.json malformed, skipping ({e})")
            continue
        if page is None:
            continue

        obs_nums = writer.list_observations(page_key)
        page.next_observation_num = (max(obs_nums) + 1) if obs_nums else 0

        for obs_num in obs_nums:
            try:
                raw_xml = writer.load_observation_raw_xml(page_key, obs_num)
                if raw_xml is not None:
                    meta = writer.load_observation_elements_meta(page_key, obs_num) or {}
                    activity = meta.get("activity", "")
                    fp = compute_xml_fingerprint(raw_xml)
                    fp_to_key[(activity, fp)] = (page_key, obs_num)

                if matcher._luma_enabled:
                    shot = writer.load_observation_screenshot(page_key, obs_num)
                    if shot is not None:
                        feat = extract_luminance_features(shot, matcher._luma_width)
                        if feat is not None:
                            page.luminance_features.append((obs_num, feat))
            except Exception as e:
                logger.warning(
                    f"rehydrate: {page_key}/{obs_num} unreadable, skipping ({e})"
                )

        if len(page.luminance_features) > ScreenMatcher._MAX_LUMINANCE_OBS:
            page.luminance_features = page.luminance_features[-ScreenMatcher._MAX_LUMINANCE_OBS:]

        pages[page_key] = page
        with contextlib.suppress(ValueError):
            max_page_idx = max(max_page_idx, int(page_key))

    matcher.rehydrate(pages, fp_to_key, counter=max_page_idx + 1)
    logger.info(f"screen_matcher rehydrated: {len(pages)} pages")
