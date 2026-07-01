"""Tests for monkey_collector.storage — DataWriter session data storage."""

import json
import os

import pytest

from monkey_collector.storage import DataWriter


@pytest.fixture
def writer(tmp_path):
    w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
    w.init_session("com.test.app", "com.test.app")
    return w


class TestInitSession:
    def test_creates_directories(self, writer, tmp_path):
        # pages/ (data side) is created lazily by save_observation/
        # save_page_knowledge on first write — init_session only needs the
        # bare session roots to exist.
        data_session_dir = tmp_path / "data" / "com.test.app"
        runtime_session_dir = tmp_path / "runtime" / "com.test.app"
        assert data_session_dir.is_dir()
        assert runtime_session_dir.is_dir()

    def test_writes_metadata(self, writer, tmp_path):
        meta_path = tmp_path / "runtime" / "com.test.app" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["session_id"] == "com.test.app"
        assert meta["package"] == "com.test.app"
        assert meta["started_at"] is not None
        assert meta["completed_at"] is None
        assert meta["total_steps"] == 0
        assert meta["external_app_events"] == 0
        assert meta["open_app_events"] == 0


class TestNextFrameIndex:
    def test_allocates_monotonically(self, writer):
        assert writer.next_frame_index() == 0
        assert writer.next_frame_index() == 1
        assert writer.next_frame_index() == 2
        assert writer.step_count == 3


class _FakeFamily:
    def __init__(self, name, element_index, key_element_index, description="", parameters=None):
        self.name = name
        self.element_index = element_index
        self.key_element_index = key_element_index
        self.description = description
        self.parameters = parameters or {}


class _FakeMatch:
    def __init__(self, page_key, match_type, is_new_page, families, page_description=""):
        self.page_key = page_key
        self.match_type = match_type
        self.is_new_page = is_new_page
        self.families = families
        self.page_description = page_description


class TestLogEvent:
    def test_appends_jsonl(self, writer, tmp_path):
        writer.log_event({"action_type": "tap", "x": 100, "y": 200})
        writer.log_event({"action_type": "swipe", "step": 1})

        events_path = tmp_path / "runtime" / "com.test.app" / "events.jsonl"
        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["action_type"] == "tap"
        assert json.loads(lines[1])["action_type"] == "swipe"


class TestLogExternalApp:
    def test_logs_and_increments(self, writer, tmp_path):
        writer.log_external_app({"detected_package": "com.other"})

        # Check event written
        events_path = tmp_path / "runtime" / "com.test.app" / "events.jsonl"
        lines = events_path.read_text().strip().split("\n")
        event = json.loads(lines[0])
        assert event["type"] == "external_app"
        assert event["detected_package"] == "com.other"

        # Check metadata counter
        meta = json.loads(
            (tmp_path / "runtime" / "com.test.app" / "metadata.json").read_text()
        )
        assert meta["external_app_events"] == 1


class TestLogOpenApp:
    def test_logs_action_and_increments(self, writer, tmp_path):
        writer.log_open_app(
            "com.target.app",
            app_name="Target App",
            step=42,
            from_package="com.android.chrome",
        )

        events_path = tmp_path / "runtime" / "com.test.app" / "events.jsonl"
        event = json.loads(events_path.read_text().strip().split("\n")[0])
        assert event["action_type"] == "open_app"
        assert event["package"] == "com.target.app"
        assert event["app_name"] == "Target App"
        assert event["step"] == 42
        assert event["from_package"] == "com.android.chrome"
        # Marked non-transition so navigation consumers skip it.
        assert event["transition"] is False
        assert event["trigger"] == "external_recovery"

        meta = json.loads(
            (tmp_path / "runtime" / "com.test.app" / "metadata.json").read_text()
        )
        assert meta["open_app_events"] == 1

    def test_from_package_omitted_when_none(self, writer, tmp_path):
        writer.log_open_app("com.target.app", step=1)
        events_path = tmp_path / "runtime" / "com.test.app" / "events.jsonl"
        event = json.loads(events_path.read_text().strip().split("\n")[0])
        assert "from_package" not in event
        assert event["app_name"] == ""


class TestFinalizeSession:
    def test_updates_metadata(self, writer, tmp_path):
        writer.next_frame_index()
        writer.next_frame_index()
        writer.finalize_session()

        meta = json.loads(
            (tmp_path / "runtime" / "com.test.app" / "metadata.json").read_text()
        )
        assert meta["completed_at"] is not None
        assert meta["total_steps"] == 2

    def test_zero_steps_not_marked_completed(self, writer, tmp_path):
        """A 0-step session failed (e.g. stale finish during handshake);
        it must not be marked completed, so the next run re-collects it."""
        writer.finalize_session()

        meta = json.loads(
            (tmp_path / "runtime" / "com.test.app" / "metadata.json").read_text()
        )
        assert meta["completed_at"] is None
        assert meta["total_steps"] == 0


class TestMultipleSteps:
    def test_sequential_operations(self, writer, tmp_path):
        for i in range(3):
            frame_index = writer.next_frame_index()
            writer.save_observation(
                f"page_{i}", 0, f"png_data_{i}".encode(), f"<xml>step_{i}</xml>",
            )
            writer.log_event({"step": i, "action_type": "tap", "frame_index": frame_index})

        assert writer.step_count == 3

        pages_dir = tmp_path / "data" / "com.test.app" / "pages"
        assert len(list(pages_dir.iterdir())) == 3
        for i in range(3):
            obs_dir = pages_dir / f"page_{i}" / "0"
            assert (obs_dir / "screenshot.png").exists()
            assert (obs_dir / "raw.xml").exists()


class TestFinalizeNoMetadata:
    def test_no_crash(self, tmp_path):
        """finalize_session when metadata.json doesn't exist -> no crash."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.runtime_session_dir = str(tmp_path / "runtime" / "nonexistent_session")
        os.makedirs(w.runtime_session_dir, exist_ok=True)
        # No metadata.json exists
        w.finalize_session()  # should not raise


class TestReinitSession:
    def test_reinit_resets_state(self, tmp_path):
        """Re-initializing session resets step_count."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("session_1", "com.test.app")
        w.next_frame_index()
        w.next_frame_index()
        assert w.step_count == 2

        w.init_session("session_2", "com.test.app")
        assert w.step_count == 0
        assert "session_2" in w.data_session_dir
        assert "session_2" in w.runtime_session_dir


class TestFindExistingSession:
    def test_returns_package(self, tmp_path):
        """Existing session for package → returns the package name."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.test.app", "com.test.app")

        result = w.find_existing_session("com.test.app")
        assert result == "com.test.app"

    def test_returns_none_when_no_session(self, tmp_path):
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        assert w.find_existing_session("com.test.app") is None

    def test_ignores_other_packages(self, tmp_path):
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.other.app", "com.other.app")

        assert w.find_existing_session("com.test.app") is None

    def test_ignores_dirs_without_metadata(self, tmp_path):
        """Directory without metadata.json is not a valid session."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        os.makedirs(tmp_path / "runtime" / "com.test.app")
        # No metadata.json created
        assert w.find_existing_session("com.test.app") is None


class TestResumeSession:
    def test_restores_step_count(self, tmp_path):
        """Step count restored from events.jsonl's highest frame_index — not
        by counting on-disk files, since a reused observation writes none."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.test.app", "com.test.app")
        for i in range(3):
            w.log_event({"step": i, "frame_index": w.next_frame_index()})
        w.finalize_session()

        w2 = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        step_count = w2.resume_session("com.test.app")
        assert step_count == 3
        assert w2.step_count == 3

    def test_restores_step_count_past_a_reused_observation_gap(self, tmp_path):
        """A frame_index with no on-disk observation (a pending/reused frame)
        must not desync resume — the max recorded frame_index is what counts."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.test.app", "com.test.app")
        w.log_event({"frame_index": 0})
        # frame_index 1 was a pending frame: consumed but never logged (no
        # observation written) — the log jumps straight to 2.
        w.log_event({"frame_index": 2})
        w.finalize_session()

        w2 = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        assert w2.resume_session("com.test.app") == 3

    def test_preserves_started_at(self, tmp_path):
        """Original started_at is preserved on resume."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.test.app", "com.test.app")
        meta_path = tmp_path / "runtime" / "com.test.app" / "metadata.json"
        original_meta = json.loads(meta_path.read_text())
        original_started = original_meta["started_at"]

        w2 = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w2.resume_session("com.test.app")

        meta = json.loads(meta_path.read_text())
        assert meta["started_at"] == original_started

    def test_adds_resumed_at(self, tmp_path):
        """Resume adds resumed_at timestamp array."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.test.app", "com.test.app")

        w2 = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w2.resume_session("com.test.app")

        meta_path = tmp_path / "runtime" / "com.test.app" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "resumed_at" in meta
        assert len(meta["resumed_at"]) == 1
        assert meta["completed_at"] is None

    def test_continues_numbering(self, tmp_path):
        """After resume, new frame_index values continue from the resumed count."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.test.app", "com.test.app")
        w.log_event({"frame_index": w.next_frame_index()})
        w.log_event({"frame_index": w.next_frame_index()})
        w.finalize_session()

        w2 = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w2.resume_session("com.test.app")
        assert w2.next_frame_index() == 2
        assert w2.step_count == 3


class TestIncrementMetadata:
    def test_increment_twice(self, tmp_path):
        """_increment_metadata twice -> value is 2."""
        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("test_session", "com.test.app")
        w._increment_metadata("external_app_events")
        w._increment_metadata("external_app_events")

        meta_path = tmp_path / "runtime" / "test_session" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["external_app_events"] == 2


class TestSaveObservation:
    def test_writes_screenshot_and_xml_variants(self, writer, tmp_path):
        from tests.fixtures.xml_samples import SIMPLE_XML

        paths = writer.save_observation("page_0", 0, b"\x89PNG_fake", SIMPLE_XML)

        obs_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0"
        assert (obs_dir / "screenshot.png").read_bytes() == b"\x89PNG_fake"
        assert (obs_dir / "raw.xml").read_text() == SIMPLE_XML
        assert (obs_dir / "parsed.xml").exists()
        assert (obs_dir / "hierarchy.xml").exists()
        assert (obs_dir / "encoded.xml").exists()
        assert (obs_dir / "pretty.xml").exists()
        assert "screenshot" in paths and "raw" in paths and "encoded" in paths

    def test_encoded_has_no_bounds(self, writer, tmp_path):
        import xml.etree.ElementTree as ET

        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_observation("page_0", 0, None, SIMPLE_XML)
        obs_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0"
        root = ET.fromstring((obs_dir / "encoded.xml").read_text())
        for el in root.iter():
            assert "bounds" not in el.attrib

    def test_invalid_xml_still_saves_raw(self, writer, tmp_path):
        """Invalid XML should still save the raw file without crashing."""
        writer.save_observation("page_0", 0, None, "<not valid!!!")
        obs_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0"
        assert (obs_dir / "raw.xml").read_text() == "<not valid!!!"
        assert not (obs_dir / "parsed.xml").exists()

    def test_no_screenshot_omits_file_and_key(self, writer, tmp_path):
        from tests.fixtures.xml_samples import SIMPLE_XML

        paths = writer.save_observation("page_0", 0, None, SIMPLE_XML)

        obs_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0"
        assert not (obs_dir / "screenshot.png").exists()
        assert "screenshot" not in paths

    def test_no_elements_json_without_match(self, writer, tmp_path):
        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_observation("page_0", 0, None, SIMPLE_XML)

        obs_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0"
        assert not (obs_dir / "elements.json").exists()

    def test_writes_elements_json_with_activity(self, writer, tmp_path):
        from tests.fixtures.xml_samples import SIMPLE_XML

        match = _FakeMatch(
            page_key="page_0",
            match_type="NEW",
            is_new_page=True,
            families=[
                _FakeFamily("open_search", [1, 2], [1], description="open search")
            ],
        )
        writer.save_observation(
            "page_0", 0, None, SIMPLE_XML, match=match, activity="act.Main",
        )

        obs_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0"
        data = json.loads((obs_dir / "elements.json").read_text())
        assert data["page_key"] == "page_0"
        assert data["activity"] == "act.Main"
        assert data["elements"][0]["name"] == "open_search"

    def test_second_observation_gets_own_directory(self, writer, tmp_path):
        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_observation("page_0", 0, None, SIMPLE_XML)
        writer.save_observation("page_0", 1, None, SIMPLE_XML)

        pages_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0"
        assert (pages_dir / "0").is_dir()
        assert (pages_dir / "1").is_dir()

    def test_empty_families_writes_empty_elements(self, writer, tmp_path):
        # elements.json serializes whatever families the match carries; an
        # empty families list (e.g. nothing re-grounded on this screen) yields
        # elements=[].
        from tests.fixtures.xml_samples import SIMPLE_XML

        match = _FakeMatch("page_0", "EQSET", False, families=[])
        writer.save_observation("page_0", 0, None, SIMPLE_XML, match=match)

        obs_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0"
        data = json.loads((obs_dir / "elements.json").read_text())
        assert data["match_type"] == "EQSET"
        assert data["elements"] == []


class TestPageKnowledgePersistence:
    def test_save_and_load_round_trips(self, writer, tmp_path):
        from monkey_collector.llm.element_extractor import ExtractedElement
        from monkey_collector.pipeline.screen_matching.page_knowledge import (
            PageKnowledge,
        )
        from monkey_collector.pipeline.screen_matching.ui_attributes import (
            UIAttributes,
        )

        page = PageKnowledge(
            page_key="page_0",
            elements=[
                ExtractedElement(
                    name="open_search", description="open search",
                    parameters={}, element_index=[1], key_element_index=[1],
                )
            ],
            key_elements={
                "open_search": [UIAttributes(self_attrs={"tag": "button"}, parent={}, children=[])]
            },
            extra_uis=[],
        )
        path = writer.save_page_knowledge("page_0", page)
        assert (tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "page.json").exists()
        assert str(path) == str(
            tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "page.json"
        )

        loaded = writer.load_page_knowledge("page_0")
        assert loaded is not None
        assert loaded.page_key == "page_0"
        assert loaded.elements[0].name == "open_search"
        assert loaded.key_elements["open_search"][0].self_attrs == {"tag": "button"}
        # luminance_features/next_observation_num are never persisted.
        assert loaded.luminance_features == []
        assert loaded.next_observation_num == 0

    def test_load_missing_page_returns_none(self, writer):
        assert writer.load_page_knowledge("page_absent") is None


class TestListPagesAndObservations:
    def test_list_pages_empty_before_any_write(self, writer):
        assert writer.list_pages() == []

    def test_list_pages_requires_page_json(self, writer, tmp_path):
        from tests.fixtures.xml_samples import SIMPLE_XML

        # An observation dir with no page.json doesn't count as a registered page.
        writer.save_observation("page_0", 0, None, SIMPLE_XML)
        assert writer.list_pages() == []

        from monkey_collector.pipeline.screen_matching.page_knowledge import (
            PageKnowledge,
        )
        writer.save_page_knowledge("page_0", PageKnowledge(page_key="page_0"))
        assert writer.list_pages() == ["page_0"]

    def test_list_observations_sorted_numerically(self, writer, tmp_path):
        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_observation("page_0", 2, None, SIMPLE_XML)
        writer.save_observation("page_0", 0, None, SIMPLE_XML)
        writer.save_observation("page_0", 1, None, SIMPLE_XML)

        assert writer.list_observations("page_0") == [0, 1, 2]

    def test_list_observations_empty_for_unknown_page(self, writer):
        assert writer.list_observations("page_absent") == []


class TestLoadObservationHelpers:
    def test_load_raw_xml_and_screenshot(self, writer):
        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_observation("page_0", 0, b"\x89PNG_fake", SIMPLE_XML)

        assert writer.load_observation_raw_xml("page_0", 0) == SIMPLE_XML
        assert writer.load_observation_screenshot("page_0", 0) == b"\x89PNG_fake"

    def test_load_missing_observation_returns_none(self, writer):
        assert writer.load_observation_raw_xml("page_0", 0) is None
        assert writer.load_observation_screenshot("page_0", 0) is None
        assert writer.load_observation_elements_meta("page_0", 0) is None

    def test_load_elements_meta(self, writer):
        from tests.fixtures.xml_samples import SIMPLE_XML

        match = _FakeMatch("page_0", "NEW", True, families=[])
        writer.save_observation(
            "page_0", 0, None, SIMPLE_XML, match=match, activity="act.Main",
        )
        meta = writer.load_observation_elements_meta("page_0", 0)
        assert meta is not None
        assert meta["activity"] == "act.Main"


class TestRegenerateXmlVariantsNewLayout:
    def test_regenerates_new_layout_observations(self, tmp_path):
        from monkey_collector.storage import regenerate_xml_variants
        from tests.fixtures.xml_samples import SIMPLE_XML

        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.test.app", "com.test.app")
        w.save_observation("page_0", 0, None, SIMPLE_XML)

        obs_dir = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0"
        (obs_dir / "parsed.xml").unlink()

        count = regenerate_xml_variants(str(tmp_path / "data"))
        assert count == 1
        assert (obs_dir / "parsed.xml").exists()

    def test_legacy_flat_layout_still_regenerates(self, tmp_path):
        """A pre-migration session (no pages/, only a flat xml/ dir with raw
        {step}.xml dumps) still works — no migration needed for old sessions."""
        from monkey_collector.storage import regenerate_xml_variants
        from tests.fixtures.xml_samples import SIMPLE_XML

        xml_dir = tmp_path / "data" / "com.legacy.app" / "xml"
        xml_dir.mkdir(parents=True)
        (xml_dir / "0000.xml").write_text(SIMPLE_XML, encoding="utf-8")

        count = regenerate_xml_variants(str(tmp_path / "data"))
        assert count == 1
        assert (xml_dir / "0000_parsed.xml").exists()
        assert (xml_dir / "0000_hierarchy.xml").exists()
        assert (xml_dir / "0000_encoded.xml").exists()
        assert (xml_dir / "0000_pretty.xml").exists()
