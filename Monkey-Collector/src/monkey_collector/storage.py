"""Session-based raw data storage, split across two roots.

Directory structure::

    data/{package}/
    ├── pages/{page_key}/
    │   ├── page.json                 (frozen anchors, written once at page creation)
    │   └── {observation_num}/
    │       ├── screenshot.png, raw.xml, parsed.xml, hierarchy.xml, encoded.xml, pretty.xml
    │       └── elements.json         (only when a live match produced one)
    └── page_graph.json

    runtime/{package}/
    ├── metadata.json
    └── events.jsonl

``data/`` is the durable corpus root (pages/observations, page_graph);
``runtime/`` is ephemeral per-run bookkeeping (resume state, cost/coverage
CSVs, the action timeline — each event carries the ``page_key``/
``observation_num`` it maps onto, alongside a monotonic ``frame_index``). A
screen that reuses an existing observation (see
``pipeline.screen_matching.screen_matcher``) writes no new files at all — see
``save_observation``.

A pre-migration session (flat ``screenshots/``/``xml/`` under ``data/{package}/``,
no ``pages/``) is left as-is; ``regenerate_xml_variants`` still supports it,
but no other tooling here reads or writes that shape anymore.
"""

import json
import os
import threading
import xml.etree.ElementTree as ET
from datetime import datetime

from loguru import logger

from monkey_collector.domain.actions import OpenApp


class DataWriter:
    """Writes raw collection data to session directories (two-root split)."""

    def __init__(self, data_dir: str = "data", runtime_dir: str = "runtime"):
        self.data_dir = data_dir
        self.runtime_dir = runtime_dir
        self.data_session_dir: str | None = None
        self.runtime_session_dir: str | None = None
        self.step_count = 0
        # events.jsonl / metadata.json are written from two threads — the main
        # collection loop (actions, log_open_app) and the TCP receiver thread
        # (on_external_app -> log_external_app). Guard the append and the
        # read-modify-write so lines never interleave and counters never lose
        # an update.
        self._lock = threading.Lock()

    def find_existing_session(self, package: str) -> str | None:
        """Find existing session directory for a package.

        Returns the session_id (directory name) or None. Existence is
        determined by ``runtime/{package}/metadata.json`` — the durable
        ``data/{package}/`` half may exist without it (e.g. after a wipe of
        only the runtime side), which correctly reads as "no session".
        """
        meta_path = os.path.join(self.runtime_dir, package, "metadata.json")
        if os.path.isfile(meta_path):
            return package
        return None

    def resume_session(self, session_id: str) -> int:
        """Resume an existing session. Returns current step count.

        Step count is recomputed from ``events.jsonl``'s highest recorded
        ``frame_index`` (not by counting on-disk files — the observation
        layout only writes a file when a screen is genuinely new, so file
        count and frame count diverge by design). A gap left by a pending/
        empty-UI frame that consumed a frame_index but was never logged (see
        ``next_frame_index``) is harmless here: at worst a resumed session
        reuses a few frame_index numbers no observation file was ever written
        under, which is a labeling nicety, not a data-loss risk.
        """
        self.data_session_dir = os.path.join(self.data_dir, session_id)
        self.runtime_session_dir = os.path.join(self.runtime_dir, session_id)
        events_path = os.path.join(self.runtime_session_dir, "events.jsonl")

        max_frame_index = -1
        if os.path.isfile(events_path):
            with open(events_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    frame_index = event.get("frame_index")
                    if isinstance(frame_index, int):
                        max_frame_index = max(max_frame_index, frame_index)
        self.step_count = max_frame_index + 1

        # Update metadata with resumed_at timestamp
        meta_path = os.path.join(self.runtime_session_dir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta.setdefault("resumed_at", []).append(datetime.now().isoformat())
        meta["completed_at"] = None
        self._write_metadata(meta)

        logger.info(
            f"Session resumed: {self.data_session_dir} (step_count={self.step_count})"
        )
        return self.step_count

    def init_session(self, session_id: str, app_package: str):
        """Initialize a new session directory (both roots).

        ``pages/`` (data side) is created lazily by ``save_observation``/
        ``save_page_knowledge`` on first write — nothing to pre-create here.
        """
        self.data_session_dir = os.path.join(self.data_dir, session_id)
        self.runtime_session_dir = os.path.join(self.runtime_dir, session_id)
        self.step_count = 0

        os.makedirs(self.data_session_dir, exist_ok=True)
        os.makedirs(self.runtime_session_dir, exist_ok=True)

        meta = {
            "session_id": session_id,
            "package": app_package,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "total_steps": 0,
            "external_app_events": 0,
            "open_app_events": 0,
        }
        self._write_metadata(meta)
        logger.info(f"Session initialized: {self.data_session_dir}")

    def next_frame_index(self) -> int:
        """Allocate the next monotonic frame_index for ``events.jsonl``.

        Called exactly once per processed XML signal — pending/empty-UI
        frames included — so ``activity_coverage.csv``'s existing
        step_count-based keying contract holds exactly as before this
        refactor. Unlike the old ``save_xml``, this no longer implies a file
        write: a pending frame consumes an index but is never logged (no
        ``page_key`` to file it under), leaving a harmless gap in
        ``events.jsonl``'s frame_index sequence (see :meth:`resume_session`).
        """
        idx = self.step_count
        self.step_count += 1
        return idx

    def log_event(self, event: dict):
        """Append an event to the events JSONL file (thread-safe)."""
        path = os.path.join(self.runtime_session_dir, "events.jsonl")
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_external_app(self, payload: dict):
        """Log external app detection event."""
        event = {"type": "external_app", "step": self.step_count, **payload}
        self.log_event(event)
        self._increment_metadata("external_app_events")

    def log_open_app(
        self,
        package: str,
        app_name: str = "",
        step: int = -1,
        from_package: str | None = None,
    ):
        """Log an open_app action (target app relaunch on external recovery).

        Recorded for open_app learning. Marked ``transition: false`` so it is
        excluded from every transition/navigation consumer — the world-modeling
        converter (``ev.get("transition", True)``) and the offline page-graph
        rebuild (``_load_events`` skip) both drop it, and the live page graph
        never sees it because the loop clears ``state.last_action`` after
        recovery. ``from_package`` is the external app the device drifted to.
        """
        event = OpenApp(package=package, app_name=app_name).to_dict()
        event["step"] = step
        event["transition"] = False
        event["trigger"] = "external_recovery"
        if from_package is not None:
            event["from_package"] = from_package
        self.log_event(event)
        self._increment_metadata("open_app_events")

    def finalize_session(self):
        """Finalize session metadata."""
        meta_path = os.path.join(self.runtime_session_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta["total_steps"] = self.step_count
            # A session that collected zero steps failed (e.g. a stale finish
            # signal during the first-session handshake). Leave completed_at
            # unset so the next run re-collects it instead of skipping it.
            if self.step_count > 0:
                meta["completed_at"] = datetime.now().isoformat()
            self._write_metadata(meta)
        logger.info(f"Session finalized: {self.step_count} steps")

    def _write_metadata(self, meta: dict):
        meta_path = os.path.join(self.runtime_session_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def save_page_graph(self, graph_data: dict) -> str:
        """Save page graph JSON under the durable data root. Returns file path."""
        path = os.path.join(self.data_session_dir, "page_graph.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        return path

    # -- observation-oriented storage (data/{package}/pages/{page_key}/{obs}/) --

    def _observation_dir(self, page_key: str, observation_num: int) -> str:
        return os.path.join(
            self.data_session_dir, "pages", page_key, str(observation_num)
        )

    def save_observation(
        self,
        page_key: str,
        observation_num: int,
        screenshot_data: bytes | None,
        xml_content: str,
        match=None,
        activity: str = "",
    ) -> dict[str, str]:
        """Write one observation's full file set: ``screenshot.png``, ``raw.xml``
        + 4 derived variants, and — iff *match* is given — ``elements.json``.

        Callers must only call this when they've already decided this screen is
        a NEW observation (``ScreenMatch.is_new_observation``); a reused
        observation has nothing new to write. Returns the paths actually
        written, keyed by artifact name (a key is present only when that file
        was produced — e.g. no ``screenshot`` key when *screenshot_data* is
        falsy, matching today's "screenshot may be absent" degrade).
        """
        obs_dir = self._observation_dir(page_key, observation_num)
        os.makedirs(obs_dir, exist_ok=True)
        paths: dict[str, str] = {}

        if screenshot_data:
            screenshot_path = os.path.join(obs_dir, "screenshot.png")
            with open(screenshot_path, "wb") as f:
                f.write(screenshot_data)
            paths["screenshot"] = screenshot_path

        raw_path = os.path.join(obs_dir, "raw.xml")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        paths["raw"] = raw_path

        try:
            from monkey_collector.xml.structured_parser import (
                StructuredXmlParser,
                hierarchy_parse,
                indent_xml,
            )

            parser = StructuredXmlParser()
            parsed = parser.parse(xml_content)

            if parsed:
                parsed_path = os.path.join(obs_dir, "parsed.xml")
                with open(parsed_path, "w", encoding="utf-8") as f:
                    f.write(parsed)
                paths["parsed"] = parsed_path

                hierarchy = hierarchy_parse(xml_content)
                if hierarchy:
                    hierarchy_path = os.path.join(obs_dir, "hierarchy.xml")
                    with open(hierarchy_path, "w", encoding="utf-8") as f:
                        f.write(hierarchy)
                    paths["hierarchy"] = hierarchy_path

                encoded = parser._clear_bounds(parser.views)
                encoded_str = ET.tostring(ET.fromstring(encoded), encoding="unicode")
                encoded_path = os.path.join(obs_dir, "encoded.xml")
                with open(encoded_path, "w", encoding="utf-8") as f:
                    f.write(encoded_str)
                paths["encoded"] = encoded_path

                pretty = indent_xml(encoded_str)
                pretty_path = os.path.join(obs_dir, "pretty.xml")
                with open(pretty_path, "w", encoding="utf-8") as f:
                    f.write(pretty)
                paths["pretty"] = pretty_path
        except Exception as e:
            logger.warning(
                f"XML parsing failed for {page_key}/{observation_num}: {e}"
            )

        if match is not None:
            elements_path = os.path.join(obs_dir, "elements.json")
            data = {
                "page_key": match.page_key,
                "match_type": match.match_type,
                "is_new_page": match.is_new_page,
                "page_description": match.page_description,
                "activity": activity,
                "elements": [
                    {
                        "name": fam.name,
                        "description": fam.description,
                        "parameters": dict(fam.parameters),
                        "element_index": list(fam.element_index),
                        "key_element_index": list(fam.key_element_index),
                    }
                    for fam in match.families
                ],
            }
            with open(elements_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            paths["elements"] = elements_path

        return paths

    def load_observation_screenshot(
        self, page_key: str, observation_num: int
    ) -> bytes | None:
        path = os.path.join(self._observation_dir(page_key, observation_num), "screenshot.png")
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def load_observation_raw_xml(
        self, page_key: str, observation_num: int
    ) -> str | None:
        path = os.path.join(self._observation_dir(page_key, observation_num), "raw.xml")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()

    def load_observation_elements_meta(
        self, page_key: str, observation_num: int
    ) -> dict | None:
        path = os.path.join(self._observation_dir(page_key, observation_num), "elements.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            data: dict = json.load(f)
        return data

    def save_page_knowledge(self, page_key: str, page) -> str:
        """Write ``pages/{page_key}/page.json`` (a page's frozen anchors).

        *page* is a ``PageKnowledge``-like object exposing ``to_dict()``.
        Written once, at page creation — a merge never mutates a page's
        stored anchors, so there is nothing to re-save on a revisit.
        """
        page_dir = os.path.join(self.data_session_dir, "pages", page_key)
        os.makedirs(page_dir, exist_ok=True)
        path = os.path.join(page_dir, "page.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(page.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def load_page_knowledge(self, page_key: str):
        """Load a page's frozen anchors back into a ``PageKnowledge``, or
        ``None`` if ``page.json`` doesn't exist (e.g. a legacy-path page that
        never had one)."""
        from monkey_collector.pipeline.screen_matching.page_knowledge import (
            PageKnowledge,
        )

        path = os.path.join(self.data_session_dir, "pages", page_key, "page.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return PageKnowledge.from_dict(data)

    def list_pages(self) -> list[str]:
        """``page_key`` directories under ``pages/`` that have a ``page.json``
        — on-disk ground truth for rehydration (mirrors MobileGPT-V2's
        on-disk-is-truth observation-numbering philosophy)."""
        pages_dir = os.path.join(self.data_session_dir, "pages")
        if not os.path.isdir(pages_dir):
            return []
        return sorted(
            name for name in os.listdir(pages_dir)
            if os.path.isfile(os.path.join(pages_dir, name, "page.json"))
        )

    def list_observations(self, page_key: str) -> list[int]:
        """Sorted observation numbers under ``pages/{page_key}/`` — on-disk
        ground truth, used both to allocate the next observation number and
        for resume rehydration."""
        page_dir = os.path.join(self.data_session_dir, "pages", page_key)
        if not os.path.isdir(page_dir):
            return []
        nums = [
            int(name) for name in os.listdir(page_dir)
            if name.isdigit() and os.path.isdir(os.path.join(page_dir, name))
        ]
        return sorted(nums)

    def _increment_metadata(self, key: str):
        meta_path = os.path.join(self.runtime_session_dir, "metadata.json")
        with self._lock:
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                meta[key] = meta.get(key, 0) + 1
                self._write_metadata(meta)


def _regenerate_one(parser, indent_xml, raw_xml: str, out_dir: str, prefix: str) -> bool:
    """Regenerate the 4 derived variants for one raw XML into *out_dir*.

    ``prefix`` is ``"{stem}_"`` for the legacy flat layout (files share the
    XML dir with the raw file) or ``""`` for the new per-observation layout
    (files already live alone in their own observation dir). Returns False on
    parse failure (raw file is left untouched either way).
    """
    parsed = parser.parse(raw_xml)
    if not parsed:
        return False

    # {prefix}parsed.xml (semantic HTML tags + bounds + index)
    with open(os.path.join(out_dir, f"{prefix}parsed.xml"), "w", encoding="utf-8") as f:
        f.write(parsed)

    # {prefix}hierarchy.xml (structure only — no text/bounds/index)
    try:
        root = ET.fromstring(parser.views)
        for el in root.iter():
            el.attrib.pop("bounds", None)
            el.attrib.pop("index", None)
            el.text = None
        hierarchy = ET.tostring(root, encoding="unicode")
        with open(os.path.join(out_dir, f"{prefix}hierarchy.xml"), "w", encoding="utf-8") as f:
            f.write(hierarchy)
    except ET.ParseError:
        pass

    # {prefix}encoded.xml (bounds removed, index only)
    encoded = parser._clear_bounds(parser.views)
    encoded_str = ET.tostring(ET.fromstring(encoded), encoding="unicode")
    with open(os.path.join(out_dir, f"{prefix}encoded.xml"), "w", encoding="utf-8") as f:
        f.write(encoded_str)

    # {prefix}pretty.xml (pretty-printed encoded)
    pretty = indent_xml(encoded_str)
    with open(os.path.join(out_dir, f"{prefix}pretty.xml"), "w", encoding="utf-8") as f:
        f.write(pretty)

    return True


def regenerate_xml_variants(data_dir: str) -> int:
    """Re-parse all raw XML files and overwrite derived variants.

    Dispatches per package: the new layout (``data_dir/{package}/pages/{page_key}/
    {observation}/raw.xml``) regenerates ``parsed.xml``/``hierarchy.xml``/
    ``encoded.xml``/``pretty.xml`` in place inside each observation directory.
    A package with no ``pages/`` (a pre-migration session, never touched by
    this refactor) falls back to the legacy flat ``xml/{step}.xml`` ->
    ``{step}_parsed.xml`` etc. layout, unchanged — no migration needed for old
    sessions to keep working here.

    Returns the number of files successfully regenerated.
    """
    from monkey_collector.xml.structured_parser import (
        StructuredXmlParser,
        indent_xml,
    )

    count = 0
    for pkg_name in sorted(os.listdir(data_dir)):
        pages_dir = os.path.join(data_dir, pkg_name, "pages")
        xml_dir = os.path.join(data_dir, pkg_name, "xml")
        pkg_count = 0

        if os.path.isdir(pages_dir):
            for page_key in sorted(os.listdir(pages_dir)):
                page_dir = os.path.join(pages_dir, page_key)
                if not os.path.isdir(page_dir):
                    continue
                for obs_name in sorted(os.listdir(page_dir)):
                    obs_dir = os.path.join(page_dir, obs_name)
                    raw_path = os.path.join(obs_dir, "raw.xml")
                    if not os.path.isfile(raw_path):
                        continue
                    with open(raw_path, encoding="utf-8") as f:
                        raw_xml = f.read()
                    parser = StructuredXmlParser()
                    if _regenerate_one(parser, indent_xml, raw_xml, obs_dir, ""):
                        pkg_count += 1
                    else:
                        logger.warning(f"Parse failed, skipping: {raw_path}")
        elif os.path.isdir(xml_dir):
            for fname in sorted(os.listdir(xml_dir)):
                if not fname.endswith(".xml") or "_" in fname:
                    continue
                stem = fname.removesuffix(".xml")
                raw_path = os.path.join(xml_dir, fname)
                with open(raw_path, encoding="utf-8") as f:
                    raw_xml = f.read()
                parser = StructuredXmlParser()
                if _regenerate_one(parser, indent_xml, raw_xml, xml_dir, f"{stem}_"):
                    pkg_count += 1
                else:
                    logger.warning(f"Parse failed, skipping: {raw_path}")
        else:
            continue

        if pkg_count:
            logger.info(f"  {pkg_name}: {pkg_count} files regenerated")
        count += pkg_count

    return count
