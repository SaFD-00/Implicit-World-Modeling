"""Tests for monkey_collector.converter — session-to-JSONL conversion."""

import json

from monkey_collector.export.converter import (
    Converter,
    _find_element_at,
    _map_event_to_action,
    generate_example,
)
from monkey_collector.xml.structured_parser import (
    encode_to_html_xml,
    parse_to_html_xml,
)
from monkey_collector.xml.ui_tree import parse_uiautomator_xml
from tests.conftest import make_element
from tests.fixtures.session_fixtures import (
    TINY_PNG,
    create_aligned_session,
    create_mock_session,
)
from tests.fixtures.xml_samples import COMPLEX_XML, SIMPLE_XML

SIMPLE_PARSED = parse_to_html_xml(SIMPLE_XML) or ""
COMPLEX_PARSED = parse_to_html_xml(COMPLEX_XML) or ""
SIMPLE_ENCODED = encode_to_html_xml(SIMPLE_XML) or ""
COMPLEX_ENCODED = encode_to_html_xml(COMPLEX_XML) or ""


def _action_from_human(human_value: str) -> dict:
    """Extract the action JSON embedded after '## Action' in a human turn."""
    return json.loads(human_value.split("## Action\n", 1)[1])


def _before_from_human(human_value: str) -> str:
    """Extract the before-state encoded XML from a human turn."""
    return human_value.split("## Current State\n", 1)[1].split(
        "\n\n## Action", 1
    )[0]


class TestFindElementAt:
    def test_exact_center(self):
        elems = [make_element(index=0, bounds=(100, 100, 200, 200))]
        result = _find_element_at(elems, 150, 150)
        assert result is not None
        assert result.index == 0

    def test_smallest_wins(self):
        big = make_element(index=0, bounds=(0, 0, 1000, 1000))
        small = make_element(index=1, bounds=(100, 100, 200, 200))
        result = _find_element_at([big, small], 150, 150)
        assert result.index == 1

    def test_no_match(self):
        elems = [make_element(index=0, bounds=(100, 100, 200, 200))]
        assert _find_element_at(elems, 500, 500) is None

    def test_on_boundary(self):
        elems = [make_element(index=0, bounds=(100, 100, 200, 200))]
        result = _find_element_at(elems, 100, 100)
        assert result is not None


class TestMapEventToAction:
    def _elements(self):
        return [
            make_element(index=0, bounds=(0, 0, 500, 500)),
            make_element(index=1, bounds=(100, 100, 200, 200)),
        ]

    def test_tap_with_element_index(self):
        result = _map_event_to_action(
            {"action_type": "tap", "element_index": 3, "x": 0, "y": 0},
            self._elements(),
        )
        assert result["type"] == "Click"
        assert result["index"] == 3

    def test_tap_coordinate_fallback(self):
        result = _map_event_to_action(
            {"action_type": "tap", "element_index": -1, "x": 150, "y": 150},
            self._elements(),
        )
        assert result["type"] == "Click"
        assert result["index"] == 1  # smallest element at (150,150)

    def test_swipe_up(self):
        result = _map_event_to_action(
            {"action_type": "swipe", "x1": 500, "y1": 800, "x2": 500, "y2": 200},
            [],
        )
        assert result["type"] == "Swipe"
        assert result["params"]["direction"] == "Up"

    def test_swipe_down(self):
        result = _map_event_to_action(
            {"action_type": "swipe", "x1": 500, "y1": 200, "x2": 500, "y2": 800},
            [],
        )
        assert result["params"]["direction"] == "Down"

    def test_swipe_left(self):
        result = _map_event_to_action(
            {"action_type": "swipe", "x1": 800, "y1": 500, "x2": 200, "y2": 500},
            [],
        )
        assert result["params"]["direction"] == "Left"

    def test_swipe_right(self):
        result = _map_event_to_action(
            {"action_type": "swipe", "x1": 200, "y1": 500, "x2": 800, "y2": 500},
            [],
        )
        assert result["params"]["direction"] == "Right"

    def test_swipe_no_coords_default_up(self):
        result = _map_event_to_action({"action_type": "swipe"}, [])
        assert result["params"]["direction"] == "Up"

    def test_input_text(self):
        result = _map_event_to_action(
            {"action_type": "input_text", "text": "hello", "element_index": 1}, []
        )
        assert result["type"] == "Input"
        assert result["params"]["text"] == "hello"

    def test_press_back(self):
        result = _map_event_to_action({"action_type": "press_back"}, [])
        assert result["type"] == "Back"

    def test_long_press(self):
        result = _map_event_to_action(
            {"action_type": "long_press", "x": 150, "y": 150, "element_index": -1},
            self._elements(),
        )
        assert result["type"] == "LongClick"

    def test_press_home(self):
        result = _map_event_to_action({"action_type": "press_home"}, [])
        assert result["type"] == "Home"

    def test_unknown_type(self):
        assert _map_event_to_action({"action_type": "teleport"}, []) is None


class TestGenerateExample:
    def test_sharegpt_format(self):
        elements = parse_uiautomator_xml(SIMPLE_XML)
        result = generate_example(
            SIMPLE_PARSED, COMPLEX_PARSED,
            {"action_type": "tap", "element_index": 2, "x": 978, "y": 84},
            "images/0001.png",
            before_elements=elements,
        )
        assert result is not None
        msgs = result["messages"]
        assert len(msgs) == 3
        assert msgs[0]["from"] == "system"
        assert msgs[1]["from"] == "human"
        assert msgs[2]["from"] == "gpt"
        assert "images" in result

    def test_skip_no_state_change(self):
        result = generate_example(
            SIMPLE_PARSED, SIMPLE_PARSED,
            {"action_type": "tap", "element_index": 2},
            "img.png",
        )
        assert result is None

    def test_skip_empty_xml(self):
        result = generate_example(
            "", SIMPLE_PARSED,
            {"action_type": "tap"}, "img.png",
        )
        assert result is None


class TestConverterSession:
    def test_convert_session(self, tmp_path):
        # create_mock_session writes frames 0=SIMPLE, 1=COMPLEX with events
        # whose `step` (100, 101) deliberately differs from `frame_index` (0, 1)
        # — a session that joins on `step` would mismatch every frame.
        session_dir = create_mock_session(tmp_path)
        output_path = tmp_path / "output.jsonl"
        images_dir = tmp_path / "images"

        converter = Converter(str(output_path), str(images_dir))
        count = converter.convert_session(str(session_dir), str(session_dir), session_label=1)

        # frame 0 → frame 1 is the only pair; the frame-1 event is the last
        # action and has no after-frame, so exactly one example is produced.
        assert count == 1
        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        msgs = data["messages"]
        human = msgs[1]["value"]
        # before/after joined by frame_index, not step.
        assert _before_from_human(human) == SIMPLE_ENCODED
        assert msgs[2]["value"] == COMPLEX_ENCODED
        # action label comes from the frame-0 event (element_index 0).
        action = _action_from_human(human)
        assert action["type"] == "Click"
        assert action["index"] == 0

    def test_convert_session_insufficient_xml(self, tmp_path):
        session_dir = tmp_path / "short_session"
        xml_dir = session_dir / "xml"
        xml_dir.mkdir(parents=True)
        (xml_dir / "0000.xml").write_text(SIMPLE_XML)
        if SIMPLE_PARSED:
            (xml_dir / "0000_parsed.xml").write_text(SIMPLE_PARSED)
        (session_dir / "metadata.json").write_text("{}")

        output_path = tmp_path / "output.jsonl"
        converter = Converter(str(output_path), str(tmp_path / "images"))
        assert converter.convert_session(str(session_dir), str(session_dir), session_label=1) == 0

    def test_convert_all(self, tmp_path):
        raw_dir = tmp_path / "raw"
        create_mock_session(raw_dir, "session_a")
        create_mock_session(raw_dir, "session_b")

        output_path = tmp_path / "output.jsonl"
        converter = Converter(str(output_path), str(tmp_path / "images"))
        total = converter.convert_all(str(raw_dir), str(raw_dir))
        # 1 pair per 2-frame session, but both mock sessions replay the SAME
        # two frames and the same action — one triple, and dedup is global
        # across sessions. See test_dedup_spans_sessions.
        assert total == 1


class TestFrameIndexAlignment:
    """Regression: events join to frames by frame_index, not the loop step."""

    def _convert(self, tmp_path, session_dir):
        out = tmp_path / "out.jsonl"
        conv = Converter(str(out), str(tmp_path / "images"))
        count = conv.convert_session(str(session_dir), str(session_dir), session_label=1)
        lines = (
            out.read_text().strip().split("\n")
            if out.exists() and out.read_text().strip()
            else []
        )
        return count, [json.loads(line) for line in lines]

    def test_joins_on_frame_index_not_step(self, tmp_path):
        # Two frames; the single action's step (999) is unrelated to its
        # frame_index (0). A step-keyed join would never find frame 0.
        session = create_aligned_session(
            tmp_path,
            frames=[(0, "before"), (1, "after")],
            events=[{
                "action_type": "tap", "element_index": 7,
                "step": 999, "frame_index": 0,
            }],
        )
        count, examples = self._convert(tmp_path, session)
        assert count == 1
        human = examples[0]["messages"][1]["value"]
        assert "before" in _before_from_human(human)
        assert "after" in examples[0]["messages"][2]["value"]
        assert _action_from_human(human)["index"] == 7

    def test_after_is_next_action_frame_skipping_empty_ui(self, tmp_path):
        # Frame 1 is an empty-UI frame with no event. The action on frame 0
        # must pair with frame 2 (next action), skipping the transient frame 1.
        session = create_aligned_session(
            tmp_path,
            frames=[(0, "start"), (1, "loading"), (2, "settled")],
            events=[
                {"action_type": "tap", "element_index": 1,
                 "step": 0, "frame_index": 0},
                {"action_type": "tap", "element_index": 2,
                 "step": 1, "frame_index": 2},
            ],
        )
        count, examples = self._convert(tmp_path, session)
        assert count == 1  # frame-2 action is last → no after-frame
        assert "settled" in examples[0]["messages"][2]["value"]
        assert "loading" not in examples[0]["messages"][2]["value"]

    def test_no_change_retry_excluded(self, tmp_path):
        session = create_aligned_session(
            tmp_path,
            frames=[(0, "a"), (1, "b")],
            events=[
                {"action_type": "tap", "element_index": 1,
                 "step": 0, "frame_index": 0},
                {"action_type": "tap", "element_index": 2,
                 "step": 0, "no_change_retry": True},
            ],
        )
        count, _ = self._convert(tmp_path, session)
        assert count == 1  # the retry (no frame_index) contributes nothing

    def test_transition_false_excluded(self, tmp_path):
        session = create_aligned_session(
            tmp_path,
            frames=[(0, "a"), (1, "b"), (2, "c")],
            events=[
                {"action_type": "tap", "element_index": 1,
                 "step": 0, "frame_index": 0},
                {"action_type": "open_app", "transition": False,
                 "step": 0, "frame_index": 1},
                {"action_type": "tap", "element_index": 3,
                 "step": 1, "frame_index": 2},
            ],
        )
        count, examples = self._convert(tmp_path, session)
        # open_app (transition:false) is dropped; frame 0 pairs with frame 2.
        assert count == 1
        assert "c" in examples[0]["messages"][2]["value"]

    def test_missing_frame_index_skipped(self, tmp_path):
        session = create_aligned_session(
            tmp_path,
            frames=[(0, "a"), (1, "b")],
            events=[{"action_type": "tap", "element_index": 1, "step": 0}],
        )
        count, _ = self._convert(tmp_path, session)
        assert count == 0  # pre-alignment event has no join key

    def test_last_action_has_no_after_frame(self, tmp_path):
        session = create_aligned_session(
            tmp_path,
            frames=[(0, "a"), (1, "b")],
            events=[{"action_type": "tap", "element_index": 1,
                     "step": 0, "frame_index": 1}],
        )
        count, _ = self._convert(tmp_path, session)
        assert count == 0  # frame_index == max_saved → no after-frame


class TestConvertAllEmpty:
    def test_empty_dir(self, tmp_path):
        """Empty directory (no valid sessions) -> 0."""
        output = tmp_path / "output.jsonl"
        images = tmp_path / "images"
        converter = Converter(str(output), str(images))
        result = converter.convert_all(str(tmp_path), str(tmp_path))
        assert result == 0

    def test_missing_data_dir_no_crash(self, tmp_path):
        """A --data-dir that doesn't exist logs and returns 0 rather than
        raising FileNotFoundError."""
        output = tmp_path / "output.jsonl"
        images = tmp_path / "images"
        converter = Converter(str(output), str(images))
        result = converter.convert_all(str(tmp_path / "missing"), str(tmp_path / "missing_rt"))
        assert result == 0


def _writer(tmp_path):
    from monkey_collector.storage import DataWriter

    w = DataWriter(data_dir=str(tmp_path / "data"), runtime_dir=str(tmp_path / "runtime"))
    w.init_session("com.test.app", "com.test.app")
    return w


class TestConverterNewLayout:
    def test_convert_new_layout_session(self, tmp_path):
        w = _writer(tmp_path)
        w.save_observation("page_0", 0, TINY_PNG, SIMPLE_XML)
        w.save_observation("page_1", 0, TINY_PNG, COMPLEX_XML)
        w.log_event({
            "action_type": "tap", "element_index": 0,
            "frame_index": 0, "page_key": "page_0", "observation_num": 0,
        })
        w.log_event({
            "action_type": "tap", "element_index": 1,
            "frame_index": 1, "page_key": "page_1", "observation_num": 0,
        })

        output_path = tmp_path / "output.jsonl"
        converter = Converter(str(output_path), str(tmp_path / "images"))
        count = converter.convert_session(
            w.data_session_dir, w.runtime_session_dir, session_label=1,
        )

        # frame 0 (page_0) -> frame 1 (page_1) is the only pair; frame 1's
        # event is the last action and has no after-pair.
        assert count == 1
        data = json.loads(output_path.read_text().strip())
        human = data["messages"][1]["value"]
        assert _before_from_human(human) == SIMPLE_ENCODED
        assert data["messages"][2]["value"] == COMPLEX_ENCODED

    def test_same_observation_before_and_after_skipped(self, tmp_path):
        # A reused observation (e.g. luminance prefilter hit) between two
        # events means no visual change — must not produce an example.
        w = _writer(tmp_path)
        w.save_observation("page_0", 0, TINY_PNG, SIMPLE_XML)
        w.log_event({
            "action_type": "tap", "element_index": 0,
            "frame_index": 0, "page_key": "page_0", "observation_num": 0,
        })
        w.log_event({
            "action_type": "tap", "element_index": 1,
            "frame_index": 1, "page_key": "page_0", "observation_num": 0,
        })

        output_path = tmp_path / "output.jsonl"
        converter = Converter(str(output_path), str(tmp_path / "images"))
        count = converter.convert_session(
            w.data_session_dir, w.runtime_session_dir, session_label=1,
        )
        assert count == 0

    def test_no_pages_and_no_xml_skips_gracefully(self, tmp_path):
        w = _writer(tmp_path)  # init_session only, no observations written
        output_path = tmp_path / "output.jsonl"
        converter = Converter(str(output_path), str(tmp_path / "images"))
        count = converter.convert_session(
            w.data_session_dir, w.runtime_session_dir, session_label=1,
        )
        assert count == 0

    def test_duplicate_transition_emitted_once(self, tmp_path):
        # The same (before, action, after) triple observed twice is one
        # training example — always deduplicated, no flag involved. The second
        # occurrence writes no JSONL line and copies no screenshot.
        w = _writer(tmp_path)
        w.save_observation("page_0", 0, TINY_PNG, SIMPLE_XML)
        w.save_observation("page_1", 0, TINY_PNG, COMPLEX_XML)
        # page_0 obs 1 / page_1 obs 1 re-observe the SAME screens, so the
        # second A->B pair encodes identically to the first.
        w.save_observation("page_0", 1, TINY_PNG, SIMPLE_XML)
        w.save_observation("page_1", 1, TINY_PNG, COMPLEX_XML)
        for frame, (key, obs) in enumerate(
            [("page_0", 0), ("page_1", 0), ("page_0", 1), ("page_1", 1)]
        ):
            w.log_event({
                "action_type": "tap", "element_index": 0,
                "frame_index": frame, "page_key": key, "observation_num": obs,
            })

        output_path = tmp_path / "output.jsonl"
        images_dir = tmp_path / "images"
        converter = Converter(str(output_path), str(images_dir))
        count = converter.convert_session(
            w.data_session_dir, w.runtime_session_dir, session_label=1,
        )

        # Pairs: (p0/0 -> p1/0), (p1/0 -> p0/1), (p0/1 -> p1/1). The third is
        # an exact duplicate of the first, so only two survive.
        assert count == 2
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert len(list(images_dir.iterdir())) == 2

    def test_distinct_transitions_both_kept(self, tmp_path):
        # Dedup keys on the transition triple, not on the before-screen alone —
        # two different actions from the same screen must both survive.
        w = _writer(tmp_path)
        w.save_observation("page_0", 0, TINY_PNG, SIMPLE_XML)
        w.save_observation("page_1", 0, TINY_PNG, COMPLEX_XML)
        w.save_observation("page_0", 1, TINY_PNG, SIMPLE_XML)
        w.save_observation("page_1", 1, TINY_PNG, COMPLEX_XML)
        for frame, (key, obs, idx) in enumerate(
            [("page_0", 0, 0), ("page_1", 0, 0), ("page_0", 1, 7), ("page_1", 1, 0)]
        ):
            w.log_event({
                "action_type": "tap", "element_index": idx,
                "frame_index": frame, "page_key": key, "observation_num": obs,
            })

        output_path = tmp_path / "output.jsonl"
        images_dir = tmp_path / "images"
        converter = Converter(str(output_path), str(images_dir))
        count = converter.convert_session(
            w.data_session_dir, w.runtime_session_dir, session_label=1,
        )

        # Same before/after screens as the previous test, but the third pair's
        # action targets element 7 instead of 0 → a distinct triple.
        assert count == 3
        assert len(output_path.read_text().strip().split("\n")) == 3
        assert len(list(images_dir.iterdir())) == 3

    @staticmethod
    def _two_app_roots(tmp_path, first_element_index: dict[str, int]):
        """Build one page_0 -> page_1 transition per package under a shared root.

        ``first_element_index`` maps package -> the tapped element index of the
        *first* event, which is what varies the action JSON (and therefore the
        dedup key) between the two packages.
        """
        from monkey_collector.storage import DataWriter

        data_root = tmp_path / "data"
        runtime_root = tmp_path / "runtime"
        for pkg, idx in first_element_index.items():
            w = DataWriter(data_dir=str(data_root), runtime_dir=str(runtime_root))
            w.init_session(pkg, pkg)
            w.save_observation("page_0", 0, TINY_PNG, SIMPLE_XML)
            w.save_observation("page_1", 0, TINY_PNG, COMPLEX_XML)
            w.log_event({
                "action_type": "tap", "element_index": idx,
                "frame_index": 0, "page_key": "page_0", "observation_num": 0,
            })
            w.log_event({
                "action_type": "tap", "element_index": 1,
                "frame_index": 1, "page_key": "page_1", "observation_num": 0,
            })
        return data_root, runtime_root

    def test_convert_all_sums_distinct_sessions(self, tmp_path):
        # Guards the enumeration itself: two packages with *different* actions
        # produce two distinct triples, so both sessions must be walked and
        # summed. (Without this, a regression that only converted the first
        # session would still satisfy the dedup test below.)
        data_root, runtime_root = self._two_app_roots(
            tmp_path, {"com.a.app": 0, "com.b.app": 5},
        )
        converter = Converter(
            str(tmp_path / "output.jsonl"), str(tmp_path / "images"),
        )
        assert converter.convert_all(str(data_root), str(runtime_root)) == 2

    def test_dedup_spans_sessions(self, tmp_path):
        # convert_all reuses one Converter, so the seen-set is global: the same
        # transition collected in two apps yields one example.
        data_root, runtime_root = self._two_app_roots(
            tmp_path, {"com.a.app": 0, "com.b.app": 0},
        )
        converter = Converter(
            str(tmp_path / "output.jsonl"), str(tmp_path / "images"),
        )
        assert converter.convert_all(str(data_root), str(runtime_root)) == 1

    def test_events_missing_join_key_skipped(self, tmp_path):
        # A pre-migration event (no page_key/observation_num) can't be joined
        # under the new layout and is dropped, not guessed.
        w = _writer(tmp_path)
        w.save_observation("page_0", 0, TINY_PNG, SIMPLE_XML)
        w.save_observation("page_1", 0, TINY_PNG, COMPLEX_XML)
        w.log_event({"action_type": "tap", "element_index": 0, "frame_index": 0})
        w.log_event({
            "action_type": "tap", "element_index": 1,
            "frame_index": 1, "page_key": "page_1", "observation_num": 0,
        })

        output_path = tmp_path / "output.jsonl"
        converter = Converter(str(output_path), str(tmp_path / "images"))
        count = converter.convert_session(
            w.data_session_dir, w.runtime_session_dir, session_label=1,
        )
        assert count == 0
