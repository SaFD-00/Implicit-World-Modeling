"""Tests for monkey_collector.storage — DataWriter session data storage."""

import json
import os

import pytest

from monkey_collector.storage import DataWriter


@pytest.fixture
def writer(tmp_path):
    w = DataWriter(base_dir=str(tmp_path))
    w.init_session("com.test.app", "com.test.app")
    return w


class TestInitSession:
    def test_creates_directories(self, writer, tmp_path):
        session_dir = tmp_path / "com.test.app"
        assert (session_dir / "screenshots").is_dir()
        assert (session_dir / "xml").is_dir()

    def test_writes_metadata(self, writer, tmp_path):
        meta_path = tmp_path / "com.test.app" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["session_id"] == "com.test.app"
        assert meta["package"] == "com.test.app"
        assert meta["started_at"] is not None
        assert meta["completed_at"] is None
        assert meta["total_steps"] == 0
        assert meta["external_app_events"] == 0
        assert meta["open_app_events"] == 0


class TestSaveScreenshot:
    def test_save(self, writer, tmp_path):
        path = writer.save_screenshot(b"\x89PNG_fake_data")
        assert "0000.png" in path
        saved = (tmp_path / "com.test.app" / "screenshots" / "0000.png").read_bytes()
        assert saved == b"\x89PNG_fake_data"


class TestSaveXml:
    def test_increments_step(self, writer, tmp_path):
        assert writer.step_count == 0

        path1 = writer.save_xml("<xml>first</xml>")
        assert "0000.xml" in path1
        assert writer.step_count == 1

        path2 = writer.save_xml("<xml>second</xml>")
        assert "0001.xml" in path2
        assert writer.step_count == 2

        content = (tmp_path / "com.test.app" / "xml" / "0001.xml").read_text()
        assert content == "<xml>second</xml>"

    def test_saves_five_variants(self, writer, tmp_path):
        """save_xml should produce 5 XML files for valid uiautomator input."""
        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_xml(SIMPLE_XML)

        xml_dir = tmp_path / "com.test.app" / "xml"
        assert (xml_dir / "0000.xml").exists()
        assert (xml_dir / "0000_parsed.xml").exists()
        assert (xml_dir / "0000_hierarchy.xml").exists()
        assert (xml_dir / "0000_encoded.xml").exists()
        assert (xml_dir / "0000_pretty.xml").exists()

    def test_encoded_has_no_bounds(self, writer, tmp_path):
        """Encoded XML should have no bounds attributes."""
        import xml.etree.ElementTree as ET

        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_xml(SIMPLE_XML)
        xml_dir = tmp_path / "com.test.app" / "xml"
        encoded = (xml_dir / "0000_encoded.xml").read_text()
        root = ET.fromstring(encoded)
        for el in root.iter():
            assert "bounds" not in el.attrib

    def test_invalid_xml_still_saves_raw(self, writer, tmp_path):
        """Invalid XML should still save raw file without crashing."""
        writer.save_xml("<not valid!!!")
        xml_dir = tmp_path / "com.test.app" / "xml"
        assert (xml_dir / "0000.xml").exists()
        assert not (xml_dir / "0000_parsed.xml").exists()


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


class TestSaveElements:
    def test_writes_elements_json_for_last_step(self, writer, tmp_path):
        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_xml(SIMPLE_XML)  # saves step 0, step_count -> 1
        match = _FakeMatch(
            page_key="page_0",
            match_type="NEW",
            is_new_page=True,
            families=[
                _FakeFamily(
                    "open_search", [1, 2], [1],
                    description="open the search bar",
                    parameters={"query": "what to search?"},
                )
            ],
            page_description="search screen",
        )
        path = writer.save_elements(match)

        assert path is not None
        assert "0000_elements.json" in path
        data = json.loads(
            (tmp_path / "com.test.app" / "xml" / "0000_elements.json").read_text()
        )
        assert data["page_key"] == "page_0"
        assert data["match_type"] == "NEW"
        assert data["is_new_page"] is True
        assert data["elements"][0]["name"] == "open_search"
        assert data["elements"][0]["description"] == "open the search bar"
        assert data["elements"][0]["parameters"] == {"query": "what to search?"}
        assert data["elements"][0]["element_index"] == [1, 2]
        assert data["elements"][0]["key_element_index"] == [1]

    def test_empty_families_writes_empty_elements(self, writer, tmp_path):
        # save_elements serializes whatever families the match carries; an empty
        # families list (e.g. nothing re-grounded on this screen) yields elements=[].
        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_xml(SIMPLE_XML)
        match = _FakeMatch("page_0", "EQSET", False, families=[])
        path = writer.save_elements(match)
        with open(path) as f:
            data = json.loads(f.read())
        assert data["match_type"] == "EQSET"
        assert data["elements"] == []

    def test_merge_with_families_writes_elements(self, writer, tmp_path):
        # On a merge/revisit the matcher now fills families from the matched page
        # (re-grounded on the current screen); save_elements writes them through.
        from tests.fixtures.xml_samples import SIMPLE_XML

        writer.save_xml(SIMPLE_XML)
        match = _FakeMatch(
            "page_0",
            "EQSET",
            False,
            families=[
                _FakeFamily(
                    "open_search", [5], [5],
                    description="open the search bar",
                    parameters={},
                )
            ],
        )
        path = writer.save_elements(match)
        with open(path) as f:
            data = json.loads(f.read())
        assert data["match_type"] == "EQSET"
        assert data["is_new_page"] is False
        assert data["elements"][0]["name"] == "open_search"
        assert data["elements"][0]["element_index"] == [5]
        assert data["elements"][0]["key_element_index"] == [5]

    def test_returns_none_before_any_step(self, writer):
        match = _FakeMatch("page_0", "NEW", True, families=[])
        assert writer.save_elements(match) is None


class TestLogEvent:
    def test_appends_jsonl(self, writer, tmp_path):
        writer.log_event({"action_type": "tap", "x": 100, "y": 200})
        writer.log_event({"action_type": "swipe", "step": 1})

        events_path = tmp_path / "com.test.app" / "events.jsonl"
        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["action_type"] == "tap"
        assert json.loads(lines[1])["action_type"] == "swipe"


class TestLogExternalApp:
    def test_logs_and_increments(self, writer, tmp_path):
        writer.log_external_app({"detected_package": "com.other"})

        # Check event written
        events_path = tmp_path / "com.test.app" / "events.jsonl"
        lines = events_path.read_text().strip().split("\n")
        event = json.loads(lines[0])
        assert event["type"] == "external_app"
        assert event["detected_package"] == "com.other"

        # Check metadata counter
        meta = json.loads(
            (tmp_path / "com.test.app" / "metadata.json").read_text()
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

        events_path = tmp_path / "com.test.app" / "events.jsonl"
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
            (tmp_path / "com.test.app" / "metadata.json").read_text()
        )
        assert meta["open_app_events"] == 1

    def test_from_package_omitted_when_none(self, writer, tmp_path):
        writer.log_open_app("com.target.app", step=1)
        events_path = tmp_path / "com.test.app" / "events.jsonl"
        event = json.loads(events_path.read_text().strip().split("\n")[0])
        assert "from_package" not in event
        assert event["app_name"] == ""


class TestFinalizeSession:
    def test_updates_metadata(self, writer, tmp_path):
        writer.save_xml("<xml>a</xml>")
        writer.save_xml("<xml>b</xml>")
        writer.finalize_session()

        meta = json.loads(
            (tmp_path / "com.test.app" / "metadata.json").read_text()
        )
        assert meta["completed_at"] is not None
        assert meta["total_steps"] == 2

    def test_zero_steps_not_marked_completed(self, writer, tmp_path):
        """A 0-step session failed (e.g. stale finish during handshake);
        it must not be marked completed, so the next run re-collects it."""
        writer.finalize_session()

        meta = json.loads(
            (tmp_path / "com.test.app" / "metadata.json").read_text()
        )
        assert meta["completed_at"] is None
        assert meta["total_steps"] == 0


class TestMultipleSteps:
    def test_sequential_operations(self, writer, tmp_path):
        for i in range(3):
            writer.save_screenshot(f"png_data_{i}".encode())
            writer.save_xml(f"<xml>step_{i}</xml>")
            writer.log_event({"step": i, "action_type": "tap"})

        assert writer.step_count == 3

        screenshots_dir = tmp_path / "com.test.app" / "screenshots"
        xml_dir = tmp_path / "com.test.app" / "xml"
        assert len(list(screenshots_dir.iterdir())) == 3
        # 3 raw files; parsed variants may or may not exist depending on XML validity
        raw_files = [f for f in xml_dir.iterdir() if "_" not in f.stem]
        assert len(raw_files) == 3


class TestFinalizeNoMetadata:
    def test_no_crash(self, tmp_path):
        """finalize_session when metadata.json doesn't exist -> no crash."""
        w = DataWriter(base_dir=str(tmp_path))
        w.session_dir = str(tmp_path / "nonexistent_session")
        os.makedirs(w.session_dir, exist_ok=True)
        # No metadata.json exists
        w.finalize_session()  # should not raise


class TestReinitSession:
    def test_reinit_resets_state(self, tmp_path):
        """Re-initializing session resets step_count."""
        w = DataWriter(base_dir=str(tmp_path))
        w.init_session("session_1", "com.test.app")
        w.save_xml("<xml>a</xml>")
        w.save_xml("<xml>b</xml>")
        assert w.step_count == 2

        w.init_session("session_2", "com.test.app")
        assert w.step_count == 0
        assert "session_2" in w.session_dir


class TestFindExistingSession:
    def test_returns_package(self, tmp_path):
        """Existing session for package → returns the package name."""
        w = DataWriter(base_dir=str(tmp_path))
        w.init_session("com.test.app", "com.test.app")

        result = w.find_existing_session("com.test.app")
        assert result == "com.test.app"

    def test_returns_none_when_no_session(self, tmp_path):
        w = DataWriter(base_dir=str(tmp_path))
        assert w.find_existing_session("com.test.app") is None

    def test_ignores_other_packages(self, tmp_path):
        w = DataWriter(base_dir=str(tmp_path))
        w.init_session("com.other.app", "com.other.app")

        assert w.find_existing_session("com.test.app") is None

    def test_ignores_dirs_without_metadata(self, tmp_path):
        """Directory without metadata.json is not a valid session."""
        w = DataWriter(base_dir=str(tmp_path))
        os.makedirs(tmp_path / "com.test.app")
        # No metadata.json created
        assert w.find_existing_session("com.test.app") is None


class TestResumeSession:
    def test_restores_step_count(self, tmp_path):
        """Step count restored from existing raw XML files."""
        w = DataWriter(base_dir=str(tmp_path))
        w.init_session("com.test.app", "com.test.app")
        w.save_xml("<xml>step0</xml>")
        w.save_xml("<xml>step1</xml>")
        w.save_xml("<xml>step2</xml>")
        w.finalize_session()

        w2 = DataWriter(base_dir=str(tmp_path))
        step_count = w2.resume_session("com.test.app")
        assert step_count == 3
        assert w2.step_count == 3

    def test_preserves_started_at(self, tmp_path):
        """Original started_at is preserved on resume."""
        w = DataWriter(base_dir=str(tmp_path))
        w.init_session("com.test.app", "com.test.app")
        meta_path = tmp_path / "com.test.app" / "metadata.json"
        original_meta = json.loads(meta_path.read_text())
        original_started = original_meta["started_at"]

        w2 = DataWriter(base_dir=str(tmp_path))
        w2.resume_session("com.test.app")

        meta = json.loads(meta_path.read_text())
        assert meta["started_at"] == original_started

    def test_adds_resumed_at(self, tmp_path):
        """Resume adds resumed_at timestamp array."""
        w = DataWriter(base_dir=str(tmp_path))
        w.init_session("com.test.app", "com.test.app")

        w2 = DataWriter(base_dir=str(tmp_path))
        w2.resume_session("com.test.app")

        meta_path = tmp_path / "com.test.app" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "resumed_at" in meta
        assert len(meta["resumed_at"]) == 1
        assert meta["completed_at"] is None

    def test_continues_numbering(self, tmp_path):
        """After resume, new files continue from existing step count."""
        w = DataWriter(base_dir=str(tmp_path))
        w.init_session("com.test.app", "com.test.app")
        w.save_xml("<xml>step0</xml>")
        w.save_xml("<xml>step1</xml>")
        w.finalize_session()

        w2 = DataWriter(base_dir=str(tmp_path))
        w2.resume_session("com.test.app")
        path = w2.save_xml("<xml>step2</xml>")
        assert "0002.xml" in path
        assert w2.step_count == 3


class TestIncrementMetadata:
    def test_increment_twice(self, tmp_path):
        """_increment_metadata twice -> value is 2."""
        w = DataWriter(base_dir=str(tmp_path))
        w.init_session("test_session", "com.test.app")
        w._increment_metadata("external_app_events")
        w._increment_metadata("external_app_events")

        meta_path = tmp_path / "test_session" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["external_app_events"] == 2
