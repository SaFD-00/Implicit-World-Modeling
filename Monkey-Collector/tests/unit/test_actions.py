"""Tests for monkey_collector.actions — action dataclasses and serialization."""

import pytest

from monkey_collector.domain.actions import (
    ACTION_REGISTRY,
    Action,
    InputText,
    LongPress,
    OpenApp,
    PressBack,
    PressHome,
    Swipe,
    Tap,
    action_from_dict,
)


class TestActionDefaults:
    def test_tap_defaults(self):
        t = Tap()
        assert t.action_type == "tap"
        assert t.x == 0
        assert t.y == 0
        assert t.element_index == -1

    def test_swipe_defaults(self):
        s = Swipe()
        assert s.action_type == "swipe"
        assert s.duration_ms == 300

    def test_input_text_defaults(self):
        it = InputText()
        assert it.action_type == "input_text"
        assert it.text == ""
        assert it.x == 0
        assert it.y == 0

    def test_press_back_defaults(self):
        pb = PressBack()
        assert pb.action_type == "press_back"

    def test_press_home_defaults(self):
        ph = PressHome()
        assert ph.action_type == "press_home"

    def test_long_press_defaults(self):
        lp = LongPress()
        assert lp.action_type == "long_press"
        assert lp.duration_ms == 1000


class TestRoundtrip:
    def test_tap_roundtrip(self):
        original = Tap(x=100, y=200, element_index=5)
        restored = Action.from_dict(original.to_dict())
        assert isinstance(restored, Tap)
        assert restored.x == 100
        assert restored.y == 200
        assert restored.element_index == 5

    def test_swipe_roundtrip(self):
        original = Swipe(x1=10, y1=20, x2=30, y2=40, duration_ms=500, element_index=3)
        restored = Action.from_dict(original.to_dict())
        assert isinstance(restored, Swipe)
        assert restored.x1 == 10 and restored.y2 == 40
        assert restored.duration_ms == 500
        assert restored.element_index == 3

    def test_input_text_roundtrip(self):
        original = InputText(text="hello", x=50, y=60, element_index=2)
        restored = Action.from_dict(original.to_dict())
        assert isinstance(restored, InputText)
        assert restored.text == "hello"
        assert restored.x == 50
        assert restored.y == 60

    def test_press_back_roundtrip(self):
        original = PressBack()
        restored = Action.from_dict(original.to_dict())
        assert isinstance(restored, PressBack)
        assert restored.action_type == "press_back"

    def test_press_home_roundtrip(self):
        original = PressHome()
        restored = Action.from_dict(original.to_dict())
        assert isinstance(restored, PressHome)

    def test_long_press_roundtrip(self):
        original = LongPress(x=100, y=200, duration_ms=2000, element_index=1)
        restored = Action.from_dict(original.to_dict())
        assert isinstance(restored, LongPress)
        assert restored.duration_ms == 2000


class TestFromDict:
    def test_unknown_type_returns_base_action(self):
        result = Action.from_dict({"action_type": "unknown_xyz"})
        assert isinstance(result, Action)
        assert result.action_type == "unknown_xyz"

    def test_extra_keys_ignored(self):
        result = Action.from_dict({"action_type": "tap", "x": 10, "y": 20, "extra": "ignored"})
        assert isinstance(result, Tap)
        assert result.x == 10
        assert not hasattr(result, "extra")

    def test_action_from_dict_empty_raises(self):
        with pytest.raises(ValueError):
            action_from_dict({})

    def test_action_from_dict_delegates(self):
        result = action_from_dict({"action_type": "tap", "x": 10, "y": 20})
        assert isinstance(result, Tap)
        assert result.x == 10


class TestRegistry:
    def test_all_types_registered(self):
        expected = {
            "tap", "swipe", "input_text", "press_back", "press_home",
            "long_press", "open_app",
        }
        assert set(ACTION_REGISTRY.keys()) == expected

    def test_registry_maps_to_correct_classes(self):
        assert ACTION_REGISTRY["tap"] is Tap
        assert ACTION_REGISTRY["swipe"] is Swipe
        assert ACTION_REGISTRY["input_text"] is InputText
        assert ACTION_REGISTRY["press_back"] is PressBack
        assert ACTION_REGISTRY["press_home"] is PressHome
        assert ACTION_REGISTRY["long_press"] is LongPress
        assert ACTION_REGISTRY["open_app"] is OpenApp


class TestOpenApp:
    def test_defaults(self):
        o = OpenApp()
        assert o.action_type == "open_app"
        assert o.package == ""
        assert o.app_name == ""
        assert o.element_index == -1

    def test_to_dict(self):
        o = OpenApp(package="com.target.app", app_name="Target App")
        assert o.to_dict() == {
            "action_type": "open_app",
            "element_index": -1,
            "package": "com.target.app",
            "app_name": "Target App",
        }

    def test_round_trip(self):
        d = OpenApp(package="com.x", app_name="X").to_dict()
        restored = action_from_dict(d)
        assert isinstance(restored, OpenApp)
        assert restored.package == "com.x"
        assert restored.app_name == "X"

    def test_from_dict_ignores_extra_log_fields(self):
        # The logged event carries step/transition/trigger/from_package on top
        # of the dataclass fields; from_dict must drop them, not choke.
        restored = action_from_dict({
            "action_type": "open_app",
            "package": "com.x",
            "app_name": "X",
            "step": 42,
            "transition": False,
            "trigger": "external_recovery",
            "from_package": "com.android.chrome",
        })
        assert isinstance(restored, OpenApp)
        assert restored.package == "com.x"
        assert not hasattr(restored, "trigger")
