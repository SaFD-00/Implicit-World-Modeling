"""Session directory scaffolding helpers for testing."""

import json
from pathlib import Path

from monkey_collector.xml.structured_parser import (
    encode_to_html_xml,
    parse_to_html_xml,
)
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

# Minimal 1x1 white PNG (67 bytes)
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def create_mock_session(
    base_dir: Path,
    session_id: str = "com.test.app",
    num_steps: int = 2,
) -> Path:
    """Create a mock session directory structure.

    Returns the session directory path.
    """
    session_dir = base_dir / session_id
    screenshots_dir = session_dir / "screenshots"
    xml_dir = session_dir / "xml"
    screenshots_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    xmls = [SIMPLE_XML, COMPLEX_XML]

    events = []
    for i in range(num_steps):
        # Screenshot
        (screenshots_dir / f"{i:04d}.png").write_bytes(TINY_PNG)
        # Raw XML
        raw_xml = xmls[i % len(xmls)]
        (xml_dir / f"{i:04d}.xml").write_text(raw_xml)
        # Pre-parsed XML (_parsed.xml)
        parsed = parse_to_html_xml(raw_xml)
        if parsed:
            (xml_dir / f"{i:04d}_parsed.xml").write_text(parsed)
        # Encoded XML (_encoded.xml) — bounds-stripped LLM input; Converter reads this
        encoded = encode_to_html_xml(raw_xml)
        if encoded:
            (xml_dir / f"{i:04d}_encoded.xml").write_text(encoded)
        # Event. `frame_index` is the join key the Converter / page-graph rebuild
        # use to map an action onto its before-frame file index. `step` is the
        # loop-counter label and is deliberately offset from frame_index here so
        # tests fail loudly if a consumer regresses to joining on `step`.
        events.append({
            "action_type": "tap",
            "x": 500,
            "y": 500,
            "element_index": i,
            "step": i + 100,
            "frame_index": i,
        })

    # events.jsonl
    events_path = session_dir / "events.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"
    )

    # metadata.json
    meta = {
        "session_id": session_id,
        "package": "com.test.app",
        "started_at": "2026-04-02T10:00:00",
        "completed_at": "2026-04-02T10:05:00",
        "total_steps": num_steps,
        "external_app_events": 0,
    }
    (session_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return session_dir


def _marker_xml(marker: str) -> str:
    """A minimal uiautomator dump whose single TextView carries ``marker`` —
    so each frame encodes to a unique ``_encoded.xml`` (before != after)."""
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        '<hierarchy rotation="0">'
        f'<node index="0" text="{marker}" class="android.widget.TextView" '
        'bounds="[0,0][100,100]" />'
        "</hierarchy>"
    )


def create_aligned_session(
    base_dir: Path,
    frames: list[tuple[int, str]],
    events: list[dict],
    session_id: str = "aligned",
) -> Path:
    """Build a session with explicit frame file indices and events.

    ``frames`` is a list of ``(frame_index, marker)`` — each writes
    ``{idx:04d}.png/.xml/_encoded.xml`` with a marker-unique encoding, letting a
    test decouple the saved file indices from the events that reference them
    (e.g. an empty-UI frame with no event, or events whose ``step`` differs from
    ``frame_index``). ``events`` is written verbatim to events.jsonl.

    Returns the session directory path.
    """
    session_dir = base_dir / session_id
    screenshots_dir = session_dir / "screenshots"
    xml_dir = session_dir / "xml"
    screenshots_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    for idx, marker in frames:
        (screenshots_dir / f"{idx:04d}.png").write_bytes(TINY_PNG)
        raw = _marker_xml(marker)
        (xml_dir / f"{idx:04d}.xml").write_text(raw)
        encoded = encode_to_html_xml(raw)
        (xml_dir / f"{idx:04d}_encoded.xml").write_text(
            encoded or f"<div>{marker}</div>"
        )

    events_path = session_dir / "events.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"
    )
    (session_dir / "metadata.json").write_text(
        json.dumps({"total_steps": len(frames)}, indent=2)
    )
    return session_dir
