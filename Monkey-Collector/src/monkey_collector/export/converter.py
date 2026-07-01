"""Convert raw collection sessions to gui-model_stage1.jsonl format.

Produces ShareGPT-format data for UI state transition prediction (World Modeling).
"""

import json
import shutil
from pathlib import Path

from loguru import logger

from monkey_collector.xml.ui_tree import UIElement, parse_uiautomator_xml

SYSTEM_PROMPT = (
    "You are a mobile UI transition predictor.\n"
    "Given the current screen represented as html-style XML and an action "
    "description, predict the next screen's html-style XML after the action "
    "is executed."
)


def _find_element_at(
    elements: list[UIElement], x: int, y: int
) -> UIElement | None:
    """Find the smallest element containing the point (x, y)."""
    candidates: list[UIElement] = []
    for elem in elements:
        left, top, right, bottom = elem.bounds
        if left <= x <= right and top <= y <= bottom:
            candidates.append(elem)
    if not candidates:
        return None
    return min(candidates, key=lambda e: e.area)


def _map_event_to_action(
    event: dict, elements: list[UIElement]
) -> dict | None:
    """Map a collector event to GUI-Model action format."""
    event_type = event.get("action_type", "")
    element_index = event.get("element_index", -1)

    if event_type == "tap":
        # Use recorded element_index if available, fallback to coordinate lookup
        if element_index >= 0:
            idx = element_index
        else:
            x, y = event.get("x", 0), event.get("y", 0)
            target = _find_element_at(elements, x, y)
            idx = target.index if target else -1
        return {
            "type": "Click",
            "params": {},
            "default": True,
            "index": idx,
        }
    elif event_type == "swipe":
        if "x1" in event and "x2" in event:
            dx = event.get("x2", 0) - event.get("x1", 0)
            dy = event.get("y2", 0) - event.get("y1", 0)
        else:
            dx, dy = 0, -1  # default up
        if abs(dy) >= abs(dx):
            direction = "Down" if dy > 0 else "Up"
        else:
            direction = "Right" if dx > 0 else "Left"
        return {
            "type": "Swipe",
            "params": {"direction": direction},
            "default": False,
            "index": element_index,
        }
    elif event_type == "input_text":
        return {
            "type": "Input",
            "params": {"text": event.get("text", "")},
            "default": False,
            "index": element_index,
        }
    elif event_type == "press_back":
        return {
            "type": "Back",
            "params": {},
            "default": False,
            "index": -1,
        }
    elif event_type == "long_press":
        if element_index < 0:
            x, y = event.get("x", 0), event.get("y", 0)
            target = _find_element_at(elements, x, y)
            element_index = target.index if target else -1
        return {
            "type": "LongClick",
            "params": {},
            "default": False,
            "index": element_index,
        }
    elif event_type == "press_home":
        return {
            "type": "Home",
            "params": {},
            "default": False,
            "index": -1,
        }
    return None


def generate_example(
    before_encoded_xml: str,
    after_encoded_xml: str,
    event: dict,
    screenshot_path: str,
    before_elements: list[UIElement] | None = None,
) -> dict | None:
    """Generate a world modeling training example.

    Args:
        before_encoded_xml: Encoded XML (_encoded.xml) for the before state.
        after_encoded_xml: Encoded XML (_encoded.xml) for the after state.
        event: Collector event dict.
        screenshot_path: Relative image path for the JSONL record.
        before_elements: UIElement list from raw XML for coordinate-based
            element lookup. Pass None if coordinate fallback is not needed.

    Returns:
        ShareGPT-format dict compatible with gui-model_stage1.jsonl,
        or None if no meaningful state change.
    """
    if not before_encoded_xml or not after_encoded_xml:
        return None

    if before_encoded_xml == after_encoded_xml:
        return None

    # Map event to action
    action = _map_event_to_action(event, before_elements or [])
    if action is None:
        return None

    action_json = json.dumps(action, indent=2)

    return {
        "messages": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {
                "from": "human",
                "value": (
                    f"<image>\n## Current State\n{before_encoded_xml}\n\n"
                    f"## Action\n{action_json}"
                ),
            },
            {"from": "gpt", "value": after_encoded_xml},
        ],
        "images": [screenshot_path],
    }


class Converter:
    """Convert raw session data to gui-model_stage1.jsonl."""

    def __init__(self, output_path: str, images_dir: str):
        self.output_path = Path(output_path)
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def convert_session(
        self, data_session_dir: str, runtime_session_dir: str, session_label: int
    ) -> int:
        """Convert a single session to JSONL examples.

        Each action event is joined to its before-screen via the event's
        ``page_key``/``observation_num`` — the durable
        ``data/{package}/pages/{page_key}/{observation_num:04d}/`` folder the
        collection loop wrote (or reused with no new write). The after-screen
        is the *next* action's before-screen, so transient loading frames
        captured between two actions are skipped and the model learns the
        settled next screen rather than a blank/IME intermediate. Two
        consecutive events resolving to the SAME observation (a reused
        observation — no visual change) are skipped too.

        Events are excluded when they lack ``frame_index``/``page_key``/
        ``observation_num`` (pre-migration sessions — no reliable join key, so
        they are dropped rather than guessed), are ``transition: false``
        markers (e.g. ``open_app`` on external recovery), or are
        ``no_change_retry`` retries (which saved no new frame). A session with
        no ``pages/`` directory but a legacy flat ``xml/`` dir degrades to
        :meth:`_convert_session_legacy`; one with neither is skipped, not a
        crash (decision: no migration script for pre-migration sessions).

        Returns:
            Number of examples generated.
        """
        data_session = Path(data_session_dir)
        pages_dir = data_session / "pages"

        if not pages_dir.is_dir():
            if (data_session / "xml").is_dir():
                logger.info(
                    f"{data_session_dir}: legacy flat layout, using legacy conversion"
                )
                return self._convert_session_legacy(
                    data_session_dir, runtime_session_dir, session_label,
                )
            logger.warning(f"{data_session_dir}: no pages/ or xml/ found, skipping")
            return 0

        events_path = Path(runtime_session_dir) / "events.jsonl"
        if not events_path.exists():
            logger.warning(f"{runtime_session_dir}: no events.jsonl, skipping")
            return 0

        events: list[dict] = []
        for line in events_path.read_text().splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if not ev.get("transition", True):
                continue  # open_app / external markers are not transitions
            if ev.get("no_change_retry"):
                continue  # retried action saved no new frame
            if not {"frame_index", "page_key", "observation_num"} <= ev.keys():
                continue  # pre-migration event: no reliable join key
            events.append(ev)
        events.sort(key=lambda e: e["frame_index"])

        count = 0
        for i, event in enumerate(events):
            # after = next action's before-screen (skips transient frames). The
            # last action has no following frame to pair with → skip it.
            if i + 1 >= len(events):
                continue
            nxt = events[i + 1]
            before_key, before_obs = event["page_key"], event["observation_num"]
            after_key, after_obs = nxt["page_key"], nxt["observation_num"]
            if (before_key, before_obs) == (after_key, after_obs):
                continue  # reused observation → no visual change

            before_dir = pages_dir / before_key / f"{before_obs:04d}"
            after_dir = pages_dir / after_key / f"{after_obs:04d}"
            before_encoded_path = before_dir / "encoded.xml"
            after_encoded_path = after_dir / "encoded.xml"
            if not before_encoded_path.exists() or not after_encoded_path.exists():
                logger.debug(
                    f"Encoded XML missing for {before_key}/{before_obs:04d} "
                    f"or {after_key}/{after_obs:04d}"
                )
                continue

            before_raw_path = before_dir / "raw.xml"
            before_elements = parse_uiautomator_xml(before_raw_path.read_text())

            # Image naming: episode_{label:06d}_step_{n:04d}.png (n = example seq)
            image_name = f"episode_{session_label:06d}_step_{count + 1:04d}.png"
            image_rel = f"GUI-Model/images/{image_name}"

            src_screenshot = before_dir / "screenshot.png"
            if not src_screenshot.exists():
                logger.debug(f"Screenshot not found for {before_key}/{before_obs:04d}")
                continue

            example = generate_example(
                before_encoded_path.read_text(),
                after_encoded_path.read_text(),
                event,
                image_rel,
                before_elements,
            )
            if example is None:
                continue

            dest_image = self.images_dir / image_name
            shutil.copy2(src_screenshot, dest_image)

            with open(self.output_path, "a") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

        logger.info(
            f"Converted session {data_session.name}: {count} examples generated"
        )
        return count

    def _convert_session_legacy(
        self, data_session_dir: str, runtime_session_dir: str, session_label: int
    ) -> int:
        """Convert a pre-migration, flat-layout session (no ``pages/``).

        Same join logic the tool has always used: events ordered by
        ``frame_index``, joined to ``xml/{idx:04d}*.xml`` / ``screenshots/
        {idx:04d}.png`` directly. ``events.jsonl`` now lives under
        ``runtime_session_dir`` (Stage 1's data/runtime split applies to every
        session going forward); a truly untouched pre-refactor backup keeps it
        alongside ``xml/``/``screenshots/``, so pass the same path for both
        roots when converting one of those.
        """
        data_session = Path(data_session_dir)
        xml_dir = data_session / "xml"
        screenshots_dir = data_session / "screenshots"
        events_path = Path(runtime_session_dir) / "events.jsonl"

        raw_xml_files = sorted(
            f for f in xml_dir.glob("*.xml") if "_" not in f.stem
        )
        if len(raw_xml_files) < 2:
            logger.warning(f"Session {data_session_dir}: not enough XML files")
            return 0
        max_saved = max(int(f.stem) for f in raw_xml_files)

        events: list[dict] = []
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                if not ev.get("transition", True):
                    continue
                if ev.get("no_change_retry"):
                    continue
                if "frame_index" not in ev:
                    continue
                events.append(ev)
        events.sort(key=lambda e: e["frame_index"])

        count = 0
        for i, event in enumerate(events):
            before_idx = event["frame_index"]
            after_idx = (
                events[i + 1]["frame_index"]
                if i + 1 < len(events)
                else max_saved
            )
            if after_idx <= before_idx:
                continue

            before_encoded_path = xml_dir / f"{before_idx:04d}_encoded.xml"
            after_encoded_path = xml_dir / f"{after_idx:04d}_encoded.xml"
            if (
                not before_encoded_path.exists()
                or not after_encoded_path.exists()
            ):
                logger.debug(
                    f"Encoded XML missing for frame {before_idx} or {after_idx}"
                )
                continue

            before_raw_path = xml_dir / f"{before_idx:04d}.xml"
            before_elements = parse_uiautomator_xml(before_raw_path.read_text())

            image_name = f"episode_{session_label:06d}_step_{count + 1:04d}.png"
            image_rel = f"GUI-Model/images/{image_name}"

            src_screenshot = screenshots_dir / f"{before_idx:04d}.png"
            if not src_screenshot.exists():
                logger.debug(f"Screenshot not found for frame {before_idx}")
                continue

            example = generate_example(
                before_encoded_path.read_text(),
                after_encoded_path.read_text(),
                event,
                image_rel,
                before_elements,
            )
            if example is None:
                continue

            dest_image = self.images_dir / image_name
            shutil.copy2(src_screenshot, dest_image)

            with open(self.output_path, "a") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

        logger.info(
            f"Converted session {data_session.name}: {count} examples generated"
        )
        return count

    def convert_all(self, data_dir: str, runtime_dir: str) -> int:
        """Convert all sessions under *data_dir* to JSONL.

        Returns:
            Total number of examples generated.
        """
        data_root = Path(data_dir)
        if not data_root.is_dir():
            logger.warning(f"Data directory not found: {data_dir}")
            return 0

        runtime_root = Path(runtime_dir)
        sessions = sorted(
            d for d in data_root.iterdir()
            if d.is_dir() and ((d / "pages").exists() or (d / "xml").exists())
        )

        if not sessions:
            logger.warning(f"No sessions found in {data_dir}")
            return 0

        total = 0
        for label, session_dir in enumerate(sessions, start=1):
            runtime_session_dir = runtime_root / session_dir.name
            n = self.convert_session(str(session_dir), str(runtime_session_dir), label)
            total += n

        logger.info(f"Total: {total} examples from {len(sessions)} sessions")
        return total
