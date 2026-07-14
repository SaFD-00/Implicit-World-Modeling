"""Tests for page_graph module: fingerprinting, PageGraph, and post-hoc build."""

import json

from monkey_collector.domain.page_graph import (
    PageGraph,
    _extract_structural_tuples,
    build_graph_from_new_layout,
    build_graph_from_session,
    compute_xml_fingerprint,
)
from tests.fixtures.session_fixtures import create_mock_session
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

# ── Fingerprinting ──


class TestComputeXmlFingerprint:
    def test_same_xml_same_fingerprint(self):
        fp1 = compute_xml_fingerprint(SIMPLE_XML)
        fp2 = compute_xml_fingerprint(SIMPLE_XML)
        assert fp1 == fp2

    def test_different_structure_different_fingerprint(self):
        fp1 = compute_xml_fingerprint(SIMPLE_XML)
        fp2 = compute_xml_fingerprint(COMPLEX_XML)
        assert fp1 != fp2

    def test_ignores_text_changes(self):
        xml_a = SIMPLE_XML
        xml_b = SIMPLE_XML.replace('text="Item title"', 'text="Different title"')
        assert compute_xml_fingerprint(xml_a) == compute_xml_fingerprint(xml_b)

    def test_ignores_bounds_changes(self):
        xml_a = SIMPLE_XML
        xml_b = SIMPLE_XML.replace(
            'bounds="[900,24][1056,144]"', 'bounds="[800,20][1000,140]"'
        )
        assert compute_xml_fingerprint(xml_a) == compute_xml_fingerprint(xml_b)

    def test_ignores_checked_state(self):
        xml_a = COMPLEX_XML
        xml_b = COMPLEX_XML.replace('checked="false"', 'checked="true"')
        assert compute_xml_fingerprint(xml_a) == compute_xml_fingerprint(xml_b)

    def test_empty_xml(self):
        fp = compute_xml_fingerprint('<?xml version="1.0"?><hierarchy/>')
        assert isinstance(fp, str)
        assert len(fp) == 32  # MD5 hex

    def test_malformed_xml(self):
        fp = compute_xml_fingerprint("not xml at all")
        assert isinstance(fp, str)
        assert len(fp) == 32


class TestExtractStructuralTuples:
    def test_returns_frozenset(self):
        result = _extract_structural_tuples(SIMPLE_XML)
        assert isinstance(result, frozenset)

    def test_simple_xml_has_tuples(self):
        result = _extract_structural_tuples(SIMPLE_XML)
        assert len(result) > 0

    def test_malformed_returns_empty(self):
        result = _extract_structural_tuples("<<<bad>>>")
        assert result == frozenset()

    def test_scrollable_children_limited(self):
        """Scrollable container with many children should only keep first 3."""
        children = "\n".join(
            f'<node index="{i}" text="item {i}" resource-id="item_{i}" '
            f'class="android.widget.TextView" content-desc="" '
            f'checkable="false" checked="false" clickable="false" '
            f'enabled="true" focusable="false" focused="false" scrollable="false" '
            f'long-clickable="false" password="false" selected="false" '
            f'bounds="[0,{i*100}][100,{i*100+80}]" package="com.test" '
            f'visible-to-user="true" important="false" />'
            for i in range(10)
        )
        xml = (
            '<?xml version="1.0"?>'
            '<hierarchy rotation="0">'
            '<node index="0" text="" resource-id="" '
            'class="android.widget.ScrollView" content-desc="" '
            'checkable="false" checked="false" clickable="false" '
            'enabled="true" focusable="false" focused="false" scrollable="true" '
            'long-clickable="false" password="false" selected="false" '
            'bounds="[0,0][100,1000]" package="com.test" '
            'visible-to-user="true" important="false">'
            f'{children}'
            '</node>'
            '</hierarchy>'
        )
        tuples = _extract_structural_tuples(xml)
        # After preprocessing: hierarchy collapses, Scroll tag kept.
        # TextViews become TextField after _reformat. Only first 3 kept.
        text_tuples = [t for t in tuples if t[0] == "TextField"]
        assert len(text_tuples) == 3


# ── PageGraph ──


class TestPageGraph:
    def test_create_new_page(self):
        g = PageGraph()
        pid = g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=0)
        assert pid == 0
        assert len(g.nodes) == 1
        assert g.nodes[0].activity == "com.test/.Main"

    def test_get_existing_page(self):
        g = PageGraph()
        pid1 = g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=0)
        pid2 = g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=1)
        assert pid1 == pid2
        assert len(g.nodes) == 1
        assert g.nodes[0].visit_count == 2

    def test_different_activity_different_page(self):
        g = PageGraph()
        pid1 = g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=0)
        pid2 = g.get_or_create_page("com.test/.Settings", SIMPLE_XML, step=1)
        assert pid1 != pid2
        assert len(g.nodes) == 2

    def test_same_activity_different_xml_different_page(self):
        g = PageGraph(threshold=1.0)  # strict: no fuzzy match
        pid1 = g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=0)
        pid2 = g.get_or_create_page("com.test/.Main", COMPLEX_XML, step=1)
        assert pid1 != pid2
        assert len(g.nodes) == 2

    def test_fuzzy_matching(self):
        """Minor XML changes within same activity should match with default threshold."""
        g = PageGraph(threshold=0.7)
        pid1 = g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=0)
        # Change only text content — structural tuples remain mostly the same
        xml_b = SIMPLE_XML.replace('text="Item title"', 'text="Other"')
        pid2 = g.get_or_create_page("com.test/.Main", xml_b, step=1)
        assert pid1 == pid2


class TestTransitionDedup:
    def test_add_new_transition(self):
        g = PageGraph()
        added = g.add_transition(0, 1, "tap", "button", step=0)
        assert added is True
        assert len(g.edges) == 1

    def test_duplicate_transition_increments_count(self):
        g = PageGraph()
        g.add_transition(0, 1, "tap", "button", step=0)
        added = g.add_transition(0, 1, "tap", "other_button", step=1)
        assert added is False
        assert len(g.edges) == 1
        assert g.edges[0].count == 2

    def test_same_pages_different_action_both_kept(self):
        g = PageGraph()
        g.add_transition(0, 1, "tap", "button", step=0)
        added = g.add_transition(0, 1, "swipe", "scroll", step=1)
        assert added is True
        assert len(g.edges) == 2

    def test_self_transition_skipped(self):
        g = PageGraph()
        added = g.add_transition(0, 0, "tap", "button", step=0)
        assert added is False
        assert len(g.edges) == 0


class TestGraphSerialization:
    def test_to_dict_roundtrip(self, tmp_path):
        g = PageGraph(threshold=0.9)
        g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=0)
        g.get_or_create_page("com.test/.Settings", COMPLEX_XML, step=1)
        g.add_transition(0, 1, "tap", "settings_btn", step=1)

        path = str(tmp_path / "graph.json")
        g.save(path)

        loaded = PageGraph.load(path)
        assert len(loaded.nodes) == 2
        assert len(loaded.edges) == 1
        assert loaded.nodes[0].activity == "com.test/.Main"
        assert loaded.edges[0].action_type == "tap"
        assert loaded.threshold == 0.9

    def test_to_dict_structure(self):
        g = PageGraph()
        g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=0)
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "metadata" in d
        assert d["metadata"]["total_pages"] == 1

    def test_observation_count_round_trips(self, tmp_path):
        g = PageGraph()
        page_id = g.get_or_create_page("com.test/.Main", SIMPLE_XML, step=0)
        g.next_observation_num(page_id)
        g.next_observation_num(page_id)
        assert g.nodes[page_id].observation_count == 2

        path = str(tmp_path / "graph.json")
        g.save(path)
        loaded = PageGraph.load(path)
        assert loaded.nodes[page_id].observation_count == 2

    def test_observation_count_defaults_to_zero_for_old_files(self, tmp_path):
        """A page_graph.json written before this field existed still loads."""
        data = {
            "nodes": [{
                "id": 0, "activity": "com.test/.Main", "xml_fingerprint": "abc",
                "first_seen_step": 0, "screenshot_step": 0,
            }],
            "edges": [],
            "metadata": {"total_pages": 1, "total_transitions": 0, "threshold": 0.85},
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(data))
        loaded = PageGraph.load(str(path))
        assert loaded.nodes[0].observation_count == 0


# ── Post-hoc build ──


class TestBuildGraphFromSession:
    def test_builds_from_mock_session(self, tmp_path):
        session_dir = create_mock_session(tmp_path, num_steps=3)
        # Add activity_coverage.csv
        csv_content = (
            "timestamp_sec,step,activity,unique_visited,total_activities,coverage\n"
            "0.5,0,com.test.app/.MainActivity,1,2,0.5\n"
            "1.0,1,com.test.app/.SettingsActivity,2,2,1.0\n"
            "1.5,2,com.test.app/.MainActivity,2,2,1.0\n"
        )
        (session_dir / "activity_coverage.csv").write_text(csv_content)

        graph = build_graph_from_session(str(session_dir))
        assert len(graph.nodes) >= 2  # at least 2 different pages
        assert len(graph.edges) >= 1  # at least 1 transition

    def test_handles_missing_activity_csv(self, tmp_path):
        session_dir = create_mock_session(tmp_path, num_steps=2)
        # No activity_coverage.csv — should still work with empty activity
        graph = build_graph_from_session(str(session_dir))
        assert len(graph.nodes) >= 1

    def test_handles_empty_session(self, tmp_path):
        session_dir = tmp_path / "empty_session"
        session_dir.mkdir()
        (session_dir / "xml").mkdir()
        graph = build_graph_from_session(str(session_dir))
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_no_change_retries_skipped(self, tmp_path):
        session_dir = create_mock_session(tmp_path, num_steps=2)
        # The retry carries no frame_index (it saved no new frame); the two real
        # actions reference frames 0 and 1.
        events = [
            {"action_type": "tap", "element_index": 0,
             "step": 0, "frame_index": 0},
            {"action_type": "tap", "element_index": 1,
             "step": 0, "no_change_retry": True},
            {"action_type": "tap", "element_index": 2,
             "step": 1, "frame_index": 1},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )
        graph = build_graph_from_session(str(session_dir))
        # The 0->1 edge is labeled by the frame-0 action (element_0); the
        # excluded retry (element_1) must never label an edge.
        assert any(e.element_info == "element_0" for e in graph.edges)
        assert all(e.element_info != "element_1" for e in graph.edges)

    def test_open_app_transition_false_not_an_edge_label(self, tmp_path):
        # An open_app event (transition:false) sitting where it would otherwise
        # label the page(0)->page(1) edge must be skipped: external recovery is
        # not navigation. The edge falls back to "unknown", never "open_app".
        session_dir = create_mock_session(tmp_path, num_steps=2)
        events = [
            {
                "action_type": "open_app",
                "element_index": -1,
                "package": "com.test.app",
                "app_name": "Test",
                "step": 0,
                "frame_index": 0,
                "transition": False,
                "trigger": "external_recovery",
            },
            {"action_type": "tap", "element_index": 2,
             "step": 1, "frame_index": 1},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )
        graph = build_graph_from_session(str(session_dir))
        assert all(e.action_type != "open_app" for e in graph.edges)

    def test_edge_label_uses_frame_index_not_step(self, tmp_path):
        # Events whose `step` (777, 888) is unrelated to their frame_index
        # (0, 1). A step-keyed join would mislabel the 0->1 edge; frame_index
        # join labels it with the frame-0 action (swipe).
        session_dir = create_mock_session(tmp_path, num_steps=2)
        events = [
            {"action_type": "swipe", "element_index": -1,
             "step": 777, "frame_index": 0},
            {"action_type": "tap", "element_index": 2,
             "step": 888, "frame_index": 1},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )
        graph = build_graph_from_session(str(session_dir))
        assert any(e.action_type == "swipe" for e in graph.edges)


class TestBuildGraphFromNewLayout:
    """Exact rebuild from pages/{page_key}/{obs}/ + events.jsonl — unlike
    build_graph_from_session, page_key is read directly, no fingerprint
    guessing."""

    def _writer(self, tmp_path):
        from monkey_collector.storage import DataWriter

        w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
        w.init_session("com.test.app", "com.test.app")
        return w

    def test_builds_nodes_and_edges_from_events(self, tmp_path):
        w = self._writer(tmp_path)
        w.save_observation("page_0", 0, None, SIMPLE_XML)
        w.save_observation("page_1", 0, None, COMPLEX_XML)
        w.log_event({
            "action_type": "tap", "element_index": 0, "activity_name": "act.Main",
            "frame_index": 0, "page_key": "page_0", "observation_num": 0,
        })
        w.log_event({
            "action_type": "tap", "element_index": 1, "activity_name": "act.Settings",
            "frame_index": 1, "page_key": "page_1", "observation_num": 0,
        })

        graph = build_graph_from_new_layout(w.data_session_dir, w.runtime_session_dir)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.edges[0].action_type == "tap"
        assert graph.edges[0].element_info == "element_0"

    def test_observation_count_counts_distinct_observations(self, tmp_path):
        w = self._writer(tmp_path)
        w.save_observation("page_0", 0, None, SIMPLE_XML)
        w.save_observation("page_0", 1, None, SIMPLE_XML)
        w.log_event({
            "action_type": "tap", "frame_index": 0,
            "page_key": "page_0", "observation_num": 0,
        })
        w.log_event({
            "action_type": "tap", "frame_index": 1,
            "page_key": "page_0", "observation_num": 1,
        })
        w.log_event({
            "action_type": "tap", "frame_index": 2,
            "page_key": "page_0", "observation_num": 0,  # revisit, not a 3rd observation
        })

        graph = build_graph_from_new_layout(w.data_session_dir, w.runtime_session_dir)
        assert len(graph.nodes) == 1
        assert graph.nodes[0].observation_count == 2

    def test_no_pages_dir_returns_empty_graph(self, tmp_path):
        w = self._writer(tmp_path)  # init_session only, no pages/ ever created
        graph = build_graph_from_new_layout(w.data_session_dir, w.runtime_session_dir)
        assert graph.nodes == []
        assert graph.edges == []

    def test_events_missing_page_key_are_skipped(self, tmp_path):
        w = self._writer(tmp_path)
        w.save_observation("page_0", 0, None, SIMPLE_XML)
        w.log_event({"action_type": "tap", "frame_index": 0})  # pre-migration
        graph = build_graph_from_new_layout(w.data_session_dir, w.runtime_session_dir)
        assert graph.nodes == []


class _FakeMatch:
    def __init__(self, page_key):
        self.page_key = page_key


class TestGetOrCreatePageByMatch:
    def test_same_page_key_reuses_node_and_bumps_visits(self):
        g = PageGraph()
        m = _FakeMatch("page_0")
        pid1 = g.get_or_create_page_by_match(m, "com.test/.Main", SIMPLE_XML, step=0)
        pid2 = g.get_or_create_page_by_match(m, "com.test/.Main", SIMPLE_XML, step=1)
        assert pid1 == pid2
        assert len(g.nodes) == 1
        assert g.nodes[pid1].visit_count == 2
        assert g.nodes[pid1].page_key == "page_0"

    def test_new_page_key_creates_node(self):
        g = PageGraph()
        a = _FakeMatch("page_0")
        b = _FakeMatch("page_1")
        pid_a = g.get_or_create_page_by_match(a, "com.test/.Main", SIMPLE_XML, step=0)
        pid_b = g.get_or_create_page_by_match(b, "com.test/.Main", COMPLEX_XML, step=1)
        assert pid_a != pid_b
        assert len(g.nodes) == 2

    def test_page_key_round_trips_through_serialization(self, tmp_path):
        g = PageGraph()
        m = _FakeMatch("page_0")
        g.get_or_create_page_by_match(m, "com.test/.Main", SIMPLE_XML, step=0)
        path = tmp_path / "page_graph.json"
        g.save(str(path))
        loaded = PageGraph.load(str(path))
        assert loaded.nodes[0].page_key == "page_0"
        # _key_to_id rebuilt → re-match reuses the node
        pid = loaded.get_or_create_page_by_match(m, "com.test/.Main", SIMPLE_XML, step=1)
        assert pid == 0 and len(loaded.nodes) == 1

    def test_old_page_graph_json_loads_without_page_key(self, tmp_path):
        # A legacy file with no page_key field must still load.
        legacy = {
            "nodes": [
                {
                    "id": 0,
                    "activity": "com.test/.Main",
                    "xml_fingerprint": "abc",
                    "first_seen_step": 0,
                    "screenshot_step": 0,
                    "visit_count": 1,
                }
            ],
            "edges": [],
            "metadata": {"threshold": 0.85},
        }
        path = tmp_path / "page_graph.json"
        path.write_text(json.dumps(legacy))
        loaded = PageGraph.load(str(path))
        assert loaded.nodes[0].page_key == ""
