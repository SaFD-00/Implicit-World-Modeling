"""Session-based raw data storage."""

import json
import os
import threading
import xml.etree.ElementTree as ET
from datetime import datetime

from loguru import logger

from monkey_collector.domain.actions import OpenApp


class DataWriter:
    """Writes raw collection data to session directories.

    Directory structure:
        data/raw/{package}/
        ├── metadata.json
        ├── screenshots/0000.png, 0001.png, ...
        ├── xml/0000.xml, 0001.xml, ...
        └── events.jsonl
    """

    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = base_dir
        self.session_dir: str | None = None
        self.step_count = 0
        # events.jsonl / metadata.json are written from two threads — the main
        # collection loop (actions, log_open_app) and the TCP receiver thread
        # (on_external_app -> log_external_app). Guard the append and the
        # read-modify-write so lines never interleave and counters never lose
        # an update.
        self._lock = threading.Lock()

    def find_existing_session(self, package: str) -> str | None:
        """Find existing session directory for a package.

        Returns the session_id (directory name) or None.
        """
        session_dir = os.path.join(self.base_dir, package)
        meta_path = os.path.join(session_dir, "metadata.json")
        if os.path.isfile(meta_path):
            return package
        return None

    def resume_session(self, session_id: str) -> int:
        """Resume an existing session. Returns current step count."""
        self.session_dir = os.path.join(self.base_dir, session_id)
        xml_dir = os.path.join(self.session_dir, "xml")

        # Count existing raw XML files (exclude _parsed, _hierarchy, etc.)
        existing = [
            f for f in os.listdir(xml_dir)
            if f.endswith(".xml") and "_" not in f
        ]
        self.step_count = len(existing)

        # Update metadata with resumed_at timestamp
        meta_path = os.path.join(self.session_dir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta.setdefault("resumed_at", []).append(datetime.now().isoformat())
        meta["completed_at"] = None
        self._write_metadata(meta)

        logger.info(f"Session resumed: {self.session_dir} (step_count={self.step_count})")
        return self.step_count

    def init_session(self, session_id: str, app_package: str):
        """Initialize a new session directory."""
        self.session_dir = os.path.join(self.base_dir, session_id)
        self.step_count = 0

        os.makedirs(os.path.join(self.session_dir, "screenshots"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "xml"), exist_ok=True)

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
        logger.info(f"Session initialized: {self.session_dir}")

    def save_screenshot(self, image_data: bytes) -> str:
        """Save screenshot data. Returns file path."""
        path = os.path.join(self.session_dir, "screenshots", f"{self.step_count:04d}.png")
        with open(path, "wb") as f:
            f.write(image_data)
        return path

    def save_xml(self, xml_content: str) -> str:
        """Save raw XML and 4 parsed variants. Increments step count.

        Files produced per step::

            {step}.xml              raw uiautomator dump
            {step}_parsed.xml       semantic HTML tags + bounds + index
            {step}_hierarchy.xml    structure only (no text/bounds/index)
            {step}_encoded.xml      bounds removed, index only (LLM input)
            {step}_pretty.xml       pretty-printed encoded
        """
        xml_dir = os.path.join(self.session_dir, "xml")
        step = self.step_count

        # 1. raw
        raw_path = os.path.join(xml_dir, f"{step:04d}.xml")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        # 2-5. parsed variants
        try:
            from monkey_collector.xml.structured_parser import (
                StructuredXmlParser,
                hierarchy_parse,
                indent_xml,
            )

            parser = StructuredXmlParser()
            parsed = parser.parse(xml_content)

            if parsed:
                # 2. parsed (bounds + index)
                parsed_path = os.path.join(xml_dir, f"{step:04d}_parsed.xml")
                with open(parsed_path, "w", encoding="utf-8") as f:
                    f.write(parsed)

                # 3. hierarchy (structure only)
                hierarchy = hierarchy_parse(xml_content)
                if hierarchy:
                    hierarchy_path = os.path.join(xml_dir, f"{step:04d}_hierarchy.xml")
                    with open(hierarchy_path, "w", encoding="utf-8") as f:
                        f.write(hierarchy)

                # 4. encoded (bounds removed)
                encoded = parser._clear_bounds(parser.views)
                encoded_str = ET.tostring(ET.fromstring(encoded), encoding="unicode")
                encoded_path = os.path.join(xml_dir, f"{step:04d}_encoded.xml")
                with open(encoded_path, "w", encoding="utf-8") as f:
                    f.write(encoded_str)

                # 5. pretty (encoded pretty-print)
                pretty = indent_xml(encoded_str)
                pretty_path = os.path.join(xml_dir, f"{step:04d}_pretty.xml")
                with open(pretty_path, "w", encoding="utf-8") as f:
                    f.write(pretty)
        except Exception as e:
            logger.warning(f"XML parsing failed for step {step}: {e}")

        self.step_count += 1
        return raw_path

    def save_elements(self, match) -> str | None:
        """Save the element-set match annotation for the just-saved step.

        Writes ``xml/{step}_elements.json`` for the step most recently produced
        by :meth:`save_xml` (i.e. ``step_count - 1``): the page identity
        (``page_key`` / ``match_type``) and the extracted element families
        (``name`` + ``description`` + ``parameters`` + ``element_index`` +
        ``key_element_index``; the indices are encoded-XML indices, the
        description/parameters carry the LLM's extracted semantics). On a new
        page these are the freshly-extracted families; on a merge / structural
        revisit they are the matched page's stored elements re-grounded on the
        current screen (anchors re-matched here, so the indices are current-step
        — ``element_index`` approximates to the anchor indices, and elements not
        rendered on this screen are omitted). A pending (loading / splash) screen
        the matcher declined produces NO ``_elements.json`` at all (the loop
        skips this call). This artifact depends on a live LLM call, so it is NOT
        reproduced by ``regenerate_xml_variants``. Returns the path, or ``None``
        if no step has been saved yet.
        """
        if self.session_dir is None or self.step_count == 0:
            return None
        step = self.step_count - 1
        path = os.path.join(self.session_dir, "xml", f"{step:04d}_elements.json")
        data = {
            "page_key": match.page_key,
            "match_type": match.match_type,
            "is_new_page": match.is_new_page,
            "page_description": match.page_description,
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def log_event(self, event: dict):
        """Append an event to the events JSONL file (thread-safe)."""
        path = os.path.join(self.session_dir, "events.jsonl")
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
        meta_path = os.path.join(self.session_dir, "metadata.json")
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
        meta_path = os.path.join(self.session_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def save_page_graph(self, graph_data: dict) -> str:
        """Save page graph JSON. Returns file path."""
        path = os.path.join(self.session_dir, "page_graph.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        return path

    def _increment_metadata(self, key: str):
        meta_path = os.path.join(self.session_dir, "metadata.json")
        with self._lock:
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                meta[key] = meta.get(key, 0) + 1
                self._write_metadata(meta)


def regenerate_xml_variants(raw_dir: str) -> int:
    """Re-parse all raw XML files and overwrite derived variants.

    Walks ``raw_dir/{package}/xml/`` directories, reads each raw XML
    (files without ``_`` in the name), and regenerates the 4 variant files:
    ``_parsed.xml``, ``_hierarchy.xml``, ``_encoded.xml``, ``_pretty.xml``.

    Returns the number of files successfully regenerated.
    """
    from monkey_collector.xml.structured_parser import (
        StructuredXmlParser,
        indent_xml,
    )

    count = 0
    for pkg_name in sorted(os.listdir(raw_dir)):
        xml_dir = os.path.join(raw_dir, pkg_name, "xml")
        if not os.path.isdir(xml_dir):
            continue
        pkg_count = 0
        for fname in sorted(os.listdir(xml_dir)):
            if not fname.endswith(".xml") or "_" in fname:
                continue
            stem = fname.removesuffix(".xml")
            raw_path = os.path.join(xml_dir, fname)
            with open(raw_path, encoding="utf-8") as f:
                raw_xml = f.read()

            parser = StructuredXmlParser()
            parsed = parser.parse(raw_xml)
            if not parsed:
                logger.warning(f"Parse failed, skipping: {raw_path}")
                continue

            # _parsed.xml (semantic HTML tags + bounds + index)
            with open(os.path.join(xml_dir, f"{stem}_parsed.xml"), "w", encoding="utf-8") as f:
                f.write(parsed)

            # _hierarchy.xml (structure only — no text/bounds/index)
            try:
                root = ET.fromstring(parser.views)
                for el in root.iter():
                    el.attrib.pop("bounds", None)
                    el.attrib.pop("index", None)
                    el.text = None
                hierarchy = ET.tostring(root, encoding="unicode")
                with open(os.path.join(xml_dir, f"{stem}_hierarchy.xml"), "w", encoding="utf-8") as f:
                    f.write(hierarchy)
            except ET.ParseError:
                pass

            # _encoded.xml (bounds removed, index only)
            encoded = parser._clear_bounds(parser.views)
            encoded_str = ET.tostring(ET.fromstring(encoded), encoding="unicode")
            with open(os.path.join(xml_dir, f"{stem}_encoded.xml"), "w", encoding="utf-8") as f:
                f.write(encoded_str)

            # _pretty.xml (pretty-printed encoded)
            pretty = indent_xml(encoded_str)
            with open(os.path.join(xml_dir, f"{stem}_pretty.xml"), "w", encoding="utf-8") as f:
                f.write(pretty)

            pkg_count += 1

        if pkg_count:
            logger.info(f"  {pkg_name}: {pkg_count} files regenerated")
        count += pkg_count

    return count
