"""Device-free tests for coverage-activity resolution (Bug 2).

The AccessibilityService reports a generic View class (e.g.
``.../android.view.ViewGroup``) for ``activity_name`` on many frames, which
never matches a declared activity and freezes coverage. The collection loop
resolves the REAL foreground activity via adb only when the a11y value is not a
declared activity — a declared a11y value is trusted as-is (no adb call, no
topResumedActivity race noise).
"""

from unittest.mock import MagicMock

from monkey_collector.domain.activity_coverage import ActivityCoverageTracker
from monkey_collector.pipeline.collection_loop import _resolve_coverage_activity

PKG = "com.google.android.contacts"
PEOPLE = f"{PKG}/.activities.PeopleActivity"
VIEWGROUP = f"{PKG}/android.view.ViewGroup"


def _collector(tracker, adb):
    collector = MagicMock()
    collector._activity_tracker = tracker
    collector.adb = adb
    return collector


def _tracker(tmp_path):
    tracker = ActivityCoverageTracker()
    tracker.initialize(
        str(tmp_path), [PEOPLE, f"{PKG}/.activities.ContactEditorActivity"],
        package=PKG, allow_dynamic_total=False,
    )
    return tracker


def test_generic_viewgroup_resolved_via_adb(tmp_path):
    tracker = _tracker(tmp_path)
    adb = MagicMock()
    adb.get_current_activity.return_value = PEOPLE
    collector = _collector(tracker, adb)

    resolved = _resolve_coverage_activity(collector, VIEWGROUP)

    # a11y ViewGroup is not declared → adb resolves the real activity.
    assert resolved == PEOPLE
    adb.get_current_activity.assert_called_once()
    # Recording the resolved activity increments coverage (ViewGroup would not).
    tracker.record(resolved, step=1)
    assert tracker.get_visited_count() == 1


def test_declared_a11y_activity_skips_adb(tmp_path):
    tracker = _tracker(tmp_path)
    adb = MagicMock()
    collector = _collector(tracker, adb)

    resolved = _resolve_coverage_activity(collector, PEOPLE)

    # a11y value is already a declared activity → trust it, no adb call.
    assert resolved == PEOPLE
    adb.get_current_activity.assert_not_called()


def test_empty_a11y_falls_back_to_adb(tmp_path):
    tracker = _tracker(tmp_path)
    adb = MagicMock()
    adb.get_current_activity.return_value = PEOPLE
    collector = _collector(tracker, adb)

    resolved = _resolve_coverage_activity(collector, "")

    assert resolved == PEOPLE
    adb.get_current_activity.assert_called_once()


def test_no_tracker_uses_adb(tmp_path):
    adb = MagicMock()
    adb.get_current_activity.return_value = PEOPLE
    collector = _collector(None, adb)

    resolved = _resolve_coverage_activity(collector, VIEWGROUP)

    assert resolved == PEOPLE
    adb.get_current_activity.assert_called_once()
