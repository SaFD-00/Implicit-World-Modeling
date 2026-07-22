"""S-9 diagnosis: offline replay of an archived session against a FRESH matcher.

Read-only diagnostic tool (not part of the collection loop). Feeds an archived
session's observations — in their original frame order — into a brand-new
``ScreenMatcher`` built with the exact ``config/run.yaml`` knobs, and checks
whether the same sequence of decisions (mint vs. merge) comes back out.

Ground truth for "this event was a mint in the live run" is ``observation_num
== 0``: under ``persist_filtered=True`` every sighting of a page — mint AND
every later revisit, prefilter hit or BM25 merge — gets a fresh, page-scoped,
monotonically increasing observation number starting at 0 (``_allocate_observation``
in ``screen_matcher.py``), so obs 0 occurs exactly once per page, at the event
that minted it. Comparing "obs==0" against the replay's own ``is_new_page``
event-for-event is a much stronger fidelity check than an aggregate page count:
it proves the replay mints at the SAME events the live run did, not merely the
same total.

For each such live mint whose event activity contains "MapActivity" (excluding
the session's very first page — the registry is empty then, so there is no
BM25 retrieval to diagnose), the pre-``match()`` matcher state is queried with
the matcher's OWN gate methods (``_bm25.top_k``, ``_element_ok``, ``_pixel_ok``)
to classify the cause: retrieval-miss / element-blocked / pixel-blocked. Using
the matcher's own methods (rather than a reimplementation) means the
classification can never drift from what ``match()`` itself decided.
"""

from __future__ import annotations

import argparse
import json
import os

from monkey_collector.pipeline.screen_matching.element_lines import (
    element_diff_count,
    serialize_element_lines,
)
from monkey_collector.pipeline.screen_matching.luminance import (
    extract_luminance_features,
    luminance_diff,
)
from monkey_collector.pipeline.screen_matching.screen_matcher import (
    ScreenMatcher,
    package_of,
)

MAP_ACTIVITY = "MapActivity"


def _load_events(runtime_dir: str) -> list[dict]:
    """Screen-match events from ``events.jsonl``, frame order, page_key-bearing only.

    Non-screen-match lines (``external_app``, ``open_app`` interrupts) carry no
    ``page_key`` and are skipped — they never reached the matcher. An event that
    carries the keys but with a ``null`` value is skipped for the same reason:
    the frame was never stamped, so there is no observation directory to load.
    Testing *presence* alone let those through and crashed ``_load_observation``
    on ``os.path.join(..., None)`` — the same null-join-key defect already fixed
    in ``export/converter.py``. ``observation_num == 0`` is a valid value, so the
    check must be ``is None``, not falsiness.
    """
    path = os.path.join(runtime_dir, "events.jsonl")
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if any(ev.get(k) is None for k in ("page_key", "observation_num")):
                continue
            events.append(ev)
    events.sort(key=lambda e: e["step"])
    return events


def _load_observation(data_dir: str, page_key: str, obs_num: int) -> tuple[str, str, bytes | None]:
    obs_dir = os.path.join(data_dir, "pages", page_key, str(obs_num))
    with open(os.path.join(obs_dir, "raw.xml"), encoding="utf-8") as f:
        raw_xml = f.read()
    with open(os.path.join(obs_dir, "encoded.xml"), encoding="utf-8") as f:
        encoded_xml = f.read()
    shot_path = os.path.join(obs_dir, "screenshot.png")
    screenshot = None
    if os.path.isfile(shot_path):
        with open(shot_path, "rb") as f:
            screenshot = f.read()
    return raw_xml, encoded_xml, screenshot


def _build_matcher(package_guard: bool = True) -> ScreenMatcher:
    """A fresh matcher with the ``config/run.yaml`` canonical knobs (verbatim).

    One knob is swept by the replay: ``--package-guard`` off reproduces the
    PRE-GUARD matcher exactly (the fidelity anchor); on is the shipped
    configuration.
    """
    return ScreenMatcher(
        luminance_prefilter=True,
        luminance_threshold=10,
        screenshot_diff_threshold=0.02,
        luminance_low_res_width=100,
        persist_filtered=True,
        bm25_top_k=5,
        element_criterion="diff",
        element_diff_max=5,
        element_jaccard_min=0.5,
        page_pixel_diff_threshold=0.3,
        package_guard=package_guard,
    )


def _classify_mint(
    matcher: ScreenMatcher,
    lines: list[str],
    cur_activity: str,
    feat,
    first_seen_activity: dict[str, str],
) -> dict:
    """Diagnose one mint using the matcher's OWN gate methods on its pre-match() state.

    Called BEFORE ``matcher.match()`` runs for this event, so ``matcher._bm25``
    / ``matcher._registry`` reflect exactly the candidate pool ``match()`` is
    about to see. The per-candidate verdict comes from ``_verify_candidate`` —
    the same method ``match()`` uses — so the gate is reflected here rather than
    re-implemented.
    """
    if len(matcher._registry) == 0 or not lines:
        return {"category": "retrieval-miss", "reason": "empty_registry_or_empty_query", "candidates": []}

    cur_set = set(lines)
    diag = []
    for cand_key, score in matcher._bm25.top_k(lines, matcher._bm25_top_k):
        page = matcher._registry.get(cand_key)
        if page is None:
            continue
        cand_activity = first_seen_activity.get(cand_key, "?")
        is_map = MAP_ACTIVITY in cand_activity
        cand_set = set(page.element_lines)
        diff = element_diff_count(cur_set, cand_set)
        package_ok, element_ok, pixel_ok = matcher._verify_candidate(
            cur_set, page, feat, cur_activity,
        )
        pixel_diff = None
        if feat is not None and page.luminance_features:
            pixel_diff = min(
                luminance_diff(feat, stored, matcher._luma_threshold)
                for _, stored in page.luminance_features
            )
        diag.append(
            {
                "candidate_page_key": cand_key,
                "candidate_activity": cand_activity,
                "is_map_activity": is_map,
                "bm25_score": score,
                "element_diff_count": diff,
                "package_ok": package_ok,
                "element_ok": element_ok,
                "pixel_diff_fraction": pixel_diff,
                "pixel_ok": pixel_ok if element_ok else None,
            }
        )

    map_cands = [c for c in diag if c["is_map_activity"]]
    if not map_cands:
        return {"category": "retrieval-miss", "candidates": diag}

    # The package guard runs before the element/pixel gates, so it gets its own
    # category: a mint whose every map candidate was cross-package was NOT
    # element- or pixel-blocked, it was a different app's screen.
    package_passing = [c for c in map_cands if c["package_ok"]]
    if not package_passing:
        return {"category": "package-blocked", "candidates": diag}

    element_passing = [c for c in package_passing if c["element_ok"]]
    if not element_passing:
        return {
            "category": "element-blocked",
            "min_element_diff_count": min(c["element_diff_count"] for c in package_passing),
            "candidates": diag,
        }

    # Self-check: at a GENUINE mint, no element-passing map candidate should
    # ALSO pass the pixel gate -- that combination means match() should have
    # merged, not minted. If this ever fires, the diagnostic snapshot diverged
    # from match()'s real state and this event's classification is untrustworthy.
    contradictions = [c for c in element_passing if c["pixel_ok"]]
    if contradictions:
        return {
            "category": "CONTRADICTION",
            "detail": "element+pixel-passing map candidate found at a mint event",
            "candidates": diag,
        }

    pixel_vals = [c["pixel_diff_fraction"] for c in element_passing if c["pixel_diff_fraction"] is not None]
    return {
        "category": "pixel-blocked",
        "min_pixel_diff_fraction": min(pixel_vals) if pixel_vals else None,
        "candidates": diag,
    }


def replay(data_dir: str, runtime_dir: str, package_guard: bool = True) -> dict:
    events = _load_events(runtime_dir)
    matcher = _build_matcher(package_guard)
    first_seen_activity: dict[str, str] = {}

    replay_mint_page_keys: list[str] = []
    mint_classifications: list[dict] = []
    per_event_mismatches: list[dict] = []
    # Live mints the replay merged away — the effect of whatever knob is swept.
    merged_at_live_mint: list[dict] = []
    decisions: list[dict] = []
    # Every activity each replay page absorbed (mint + all merges). A page that
    # absorbed two different PACKAGES is a cross-app merge: page_graph edges that
    # claim one screen belongs to two apps. This is the audit the merge guard
    # exists to zero out, computed here so no ad-hoc script has to.
    absorbed: dict[str, dict[str, int]] = {}
    first_page_seen = False

    for ev in events:
        page_key = ev["page_key"]
        obs_num = ev["observation_num"]
        activity = ev["activity_name"]
        raw_xml, encoded_xml, screenshot = _load_observation(data_dir, page_key, obs_num)

        lines = serialize_element_lines(encoded_xml)
        feat = (
            extract_luminance_features(screenshot, matcher._luma_width)
            if (matcher._luma_enabled and screenshot)
            else None
        )

        live_is_mint = obs_num == 0
        # Diagnose EVERY event's pre-match() state, not just the live mints: with
        # the merge guard on, the replay can mint where the live run merged (that
        # is the point), and those events need a classification too.
        diag = _classify_mint(matcher, lines, activity, feat, first_seen_activity)

        result = matcher.match(raw_xml, encoded_xml, activity, screenshot)

        if result.page_key:
            counts = absorbed.setdefault(result.page_key, {})
            counts[activity] = counts.get(activity, 0) + 1

        decisions.append(
            {
                "step": ev["step"],
                "live_page_key": page_key,
                "live_observation_num": obs_num,
                "replay_is_new_page": result.is_new_page,
                "replay_page_key": result.page_key,
                "replay_match_type": result.match_type,
            }
        )

        if result.is_new_page != live_is_mint:
            per_event_mismatches.append(
                {
                    "step": ev["step"],
                    "page_key": page_key,
                    "observation_num": obs_num,
                    "live_is_mint": live_is_mint,
                    "replay_is_new_page": result.is_new_page,
                    "replay_page_key": result.page_key,
                }
            )

        if live_is_mint and not result.is_new_page:
            merged_at_live_mint.append(
                {
                    "step": ev["step"],
                    "live_page_key": page_key,
                    "activity": activity,
                    "is_map_activity": MAP_ACTIVITY in activity,
                    "merged_into_replay_page_key": result.page_key,
                }
            )

        if result.is_new_page:
            first_seen_activity[result.page_key] = activity
            replay_mint_page_keys.append(result.page_key)
            is_map = MAP_ACTIVITY in activity
            if not first_page_seen:
                first_page_seen = True  # session's first page: excluded from classification
            elif is_map:
                mint_classifications.append(
                    {
                        "step": ev["step"],
                        "live_page_key": page_key,
                        "replay_page_key": result.page_key,
                        "activity": activity,
                        **diag,
                    }
                )

    total_replay_pages = len(replay_mint_page_keys)
    map_replay_pages = sum(1 for pk in replay_mint_page_keys if MAP_ACTIVITY in first_seen_activity.get(pk, ""))
    merged_non_map = [m for m in merged_at_live_mint if not m["is_map_activity"]]

    absorbed_activities = {
        pk: sorted(counts) for pk, counts in sorted(absorbed.items(), key=lambda kv: int(kv[0]))
    }
    cross_package_pages = [
        {
            "replay_page_key": pk,
            "packages": sorted({package_of(a) for a in acts}),
            "activities": acts,
        }
        for pk, acts in absorbed_activities.items()
        if len({package_of(a) for a in acts}) > 1
    ]

    return {
        "package_guard": package_guard,
        "fidelity": {
            "total_pages_replay": total_replay_pages,
            "map_pages_replay": map_replay_pages,
            "per_event_mismatch_count": len(per_event_mismatches),
            "per_event_mismatches": per_event_mismatches,
        },
        "merged_at_live_mint_count": len(merged_at_live_mint),
        "merged_non_map_pages": merged_non_map,
        "merged_non_map_page_count": len(merged_non_map),
        "absorbed_activities": absorbed_activities,
        "cross_package_pages": cross_package_pages,
        "cross_package_page_count": len(cross_package_pages),
        "mint_classifications": mint_classifications,
        "decisions": decisions,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="e.g. .../armA_poke_off/data_net.osmand")
    ap.add_argument("--runtime-dir", required=True, help="e.g. .../armA_poke_off/runtime_net.osmand")
    ap.add_argument(
        "--package-guard",
        choices=("on", "off"),
        default="on",
        help="same-package merge guard (default on; off = pre-guard matcher)",
    )
    ap.add_argument("--out", default=None, help="optional path to write the JSON result")
    args = ap.parse_args()

    result = replay(
        args.data_dir,
        args.runtime_dir,
        package_guard=(args.package_guard == "on"),
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
