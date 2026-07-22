"""Tests for pipeline.session_manager — init/resume dual-root wipe + rehydration."""

from unittest.mock import MagicMock

from monkey_collector.domain.page_graph import PageGraph
from monkey_collector.pipeline.collection_loop import CollectionState
from monkey_collector.pipeline.session_manager import (
    init_or_resume_session,
    rehydrate_session,
)
from monkey_collector.storage import DataWriter


def _collector(tmp_path, new_session=False):
    writer = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
    collector = MagicMock()
    collector.writer = writer
    collector._new_session = new_session
    collector._activity_tracker = None
    collector._cost_tracker = None
    collector._screen_matcher = None
    return collector


class TestNewSessionDualRootWipe:
    def test_wipes_both_data_and_runtime(self, tmp_path):
        """--new-session must remove BOTH roots — a surviving data/{pkg}/
        half would immediately rehydrate stale knowledge into what's supposed
        to be a fresh session."""
        collector = _collector(tmp_path, new_session=True)
        (tmp_path / "data" / "com.test.app" / "pages" / "page_0").mkdir(parents=True)
        (tmp_path / "runtime" / "apps" / "com.test.app").mkdir(parents=True)
        (tmp_path / "runtime" / "apps" / "com.test.app" / "metadata.json").write_text("{}")

        init_or_resume_session(collector, "com.test.app")

        # init_session() re-creates fresh empty screenshots/xml dirs, but the
        # stale pages/ subtree from the prior session must be gone.
        assert not (tmp_path / "data" / "com.test.app" / "pages").exists()

    def test_returns_is_resumed_false_for_fresh_session(self, tmp_path):
        collector = _collector(tmp_path, new_session=True)
        session_id, resume_step, is_resumed = init_or_resume_session(collector, "com.test.app")
        assert is_resumed is False
        assert resume_step == 0


class TestResumeReturnsTrue:
    def test_existing_session_reports_resumed(self, tmp_path):
        collector = _collector(tmp_path)
        collector.writer.init_session("com.test.app", "com.test.app")
        collector.writer.log_event({"frame_index": collector.writer.next_frame_index()})
        collector.writer.finalize_session()

        collector2 = _collector(tmp_path)
        _, resume_step, is_resumed = init_or_resume_session(collector2, "com.test.app")
        assert is_resumed is True
        assert resume_step == 1


class TestRehydrateSession:
    def test_loads_page_graph_into_state(self, tmp_path):
        collector = _collector(tmp_path)
        collector.writer.init_session("com.test.app", "com.test.app")

        graph = PageGraph()
        graph.get_or_create_page("act.Main", "<hierarchy/>", 0)
        collector.writer.save_page_graph(graph.to_dict())

        state = CollectionState()
        assert len(state.page_graph.nodes) == 0

        rehydrate_session(collector, state)

        assert len(state.page_graph.nodes) == 1

    def test_no_page_graph_file_leaves_state_untouched(self, tmp_path):
        collector = _collector(tmp_path)
        collector.writer.init_session("com.test.app", "com.test.app")

        state = CollectionState()
        rehydrate_session(collector, state)

        assert len(state.page_graph.nodes) == 0

    def test_malformed_page_graph_degrades_to_empty_not_crash(self, tmp_path):
        collector = _collector(tmp_path)
        collector.writer.init_session("com.test.app", "com.test.app")
        graph_path = tmp_path / "data" / "com.test.app" / "page_graph.json"
        graph_path.write_text("{not valid json")

        state = CollectionState()
        rehydrate_session(collector, state)  # must not raise

        assert len(state.page_graph.nodes) == 0
