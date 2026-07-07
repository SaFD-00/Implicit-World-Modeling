"""Tests for monkey_collector.activity_coverage — Activity coverage tracking."""

import csv
import os

import pytest

from monkey_collector.domain.activity_coverage import (
    ActivityCoverageTracker,
    _normalize_activity_name,
)


class TestInitialize:
    def test_creates_csv(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        assert os.path.exists(tracker.csv_path)

    def test_writes_header(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        with open(tracker.csv_path) as f:
            header = next(csv.reader(f))
        assert header == ActivityCoverageTracker.CSV_COLUMNS

    def test_resets_state_on_reinitialize(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        tracker.record("A", step=1)
        assert tracker.get_visited_count() == 1

        # Re-initialize for new session
        session2 = tmp_path / "session2"
        session2.mkdir()
        tracker.initialize(str(session2), ["X", "Y", "Z"])
        assert tracker.get_visited_count() == 0
        assert len(tracker.total_activities) == 3


class TestRecord:
    def test_adds_activity(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        tracker.record("A", step=1)
        assert "A" in tracker.visited_activities

    def test_returns_entry_dict(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        entry = tracker.record("A", step=1)
        assert isinstance(entry, dict)
        for col in ActivityCoverageTracker.CSV_COLUMNS:
            assert col in entry

    def test_appends_to_csv(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B", "C"])
        tracker.record("A", step=1)
        tracker.record("B", step=2)
        with open(tracker.csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 data rows


class TestCoverage:
    def test_coverage_computation(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B", "C", "D"])
        tracker.record("A", step=1)
        tracker.record("B", step=2)
        assert tracker.get_coverage() == pytest.approx(0.5)

    def test_initial_coverage_zero(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["X", "Y"])
        assert tracker.get_coverage() == 0.0

    def test_full_coverage(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        tracker.record("A", step=1)
        tracker.record("B", step=2)
        assert tracker.get_coverage() == pytest.approx(1.0)

    def test_get_visited_count(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        assert tracker.get_visited_count() == 0
        tracker.record("A", step=1)
        assert tracker.get_visited_count() == 1


class TestEdgeCases:
    def test_duplicate_not_double_counted(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        tracker.record("A", step=1)
        tracker.record("A", step=2)
        assert tracker.get_visited_count() == 1
        assert tracker.get_coverage() == pytest.approx(0.5)

    def test_record_without_initialize(self):
        tracker = ActivityCoverageTracker()
        entry = tracker.record("A", step=1)
        assert "A" in tracker.visited_activities
        assert entry["activity"] == "A"
        assert not os.path.exists(tracker.csv_path)

    def test_empty_activity_name(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A"])
        tracker.record("", step=1)
        assert tracker.get_visited_count() == 0

    def test_empty_total_activities(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), [])
        entry = tracker.record("A", step=1)
        # A is dynamically added to total, so total_activities == 1
        assert entry["total_activities"] == 1
        assert entry["coverage"] == pytest.approx(1.0)

    def test_dynamic_total_expansion(self, tmp_path):
        """Visiting an undeclared activity expands total_activities."""
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A", "B"])
        tracker.record("A", step=1)
        tracker.record("C", step=2)  # C is not declared
        assert tracker.get_visited_count() == 2
        assert len(tracker.total_activities) == 3  # expanded
        assert tracker.get_coverage() <= 1.0

    def test_format_normalization(self, tmp_path):
        """Shorthand and full activity formats are treated as the same."""
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["com.test.app/.MainActivity"])
        # App sends full format
        tracker.record("com.test.app/com.test.app.MainActivity", step=1)
        # Should NOT expand total — same activity in different format
        assert len(tracker.total_activities) == 1
        assert tracker.get_coverage() == pytest.approx(1.0)

    def test_coverage_never_exceeds_one(self, tmp_path):
        """Coverage stays <= 1.0 even with many unknown activities."""
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["A"])
        tracker.record("A", step=1)
        tracker.record("B", step=2)
        tracker.record("C", step=3)
        tracker.record("D", step=4)
        assert tracker.get_coverage() <= 1.0
        assert len(tracker.total_activities) == 4


class TestStaticGroundTruth:
    def test_outside_ground_truth_not_counted(self, tmp_path):
        """allow_dynamic_total=False ignores activities not in the catalog set."""
        tracker = ActivityCoverageTracker()
        tracker.initialize(
            str(tmp_path), ["com.test/.A", "com.test/.B"],
            package="com.test", allow_dynamic_total=False,
        )
        tracker.record("com.test/.A", step=1)
        tracker.record("com.test/.NotInCatalog", step=2)
        tracker.record("androidx/.SomeDialog", step=3)
        assert len(tracker.total_activities) == 2  # fixed denominator
        assert tracker.get_visited_count() == 1    # only A counted
        assert tracker.get_coverage() == pytest.approx(0.5)

    def test_format_normalization_static(self, tmp_path):
        """Shorthand visit matches full-form catalog entry under static mode."""
        tracker = ActivityCoverageTracker()
        tracker.initialize(
            str(tmp_path), ["com.test.app/.MainActivity"],
            package="com.test.app", allow_dynamic_total=False,
        )
        # Catalog stores shorthand; observed activity arrives in full form
        tracker.record("com.test.app/com.test.app.MainActivity", step=1)
        assert tracker.get_visited_count() == 1
        assert tracker.get_coverage() == pytest.approx(1.0)

    def test_clamp_safety_net(self, tmp_path):
        """Coverage stays <= 1.0 even if visits ever exceed the denominator."""
        tracker = ActivityCoverageTracker()
        tracker.initialize(
            str(tmp_path), ["A"], package="", allow_dynamic_total=False,
        )
        tracker.record("A", step=1)
        assert tracker.get_coverage() == 1.0

    def test_resume_preserves_static_flag(self, tmp_path):
        """resume(allow_dynamic_total=False) keeps the same counting policy."""
        tracker = ActivityCoverageTracker()
        tracker.initialize(
            str(tmp_path), ["com.test/.A", "com.test/.B"],
            package="com.test", allow_dynamic_total=False,
        )
        tracker.record("com.test/.A", step=1)
        tracker.record("com.test/.NotInCatalog", step=2)

        tracker2 = ActivityCoverageTracker()
        tracker2.resume(
            str(tmp_path), ["com.test/.A", "com.test/.B"],
            package="com.test", allow_dynamic_total=False,
        )
        assert len(tracker2.total_activities) == 2
        assert tracker2.get_visited_count() == 1  # NotInCatalog ignored on rebuild
        tracker2.record("com.test/.AnotherNotInCatalog", step=3)
        assert len(tracker2.total_activities) == 2
        assert tracker2.get_visited_count() == 1


class TestIsDeclared:
    def test_declared_full_form(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["com.test/.A", "com.test/.B"])
        assert tracker.is_declared("com.test/com.test.A") is True

    def test_declared_shorthand_matches(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["com.test/.A"])
        # Shorthand catalog entry, full-form query — normalization merges them.
        assert tracker.is_declared("com.test/com.test.A") is True

    def test_generic_view_class_not_declared(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(
            str(tmp_path), ["com.google.android.contacts/.activities.PeopleActivity"]
        )
        assert tracker.is_declared(
            "com.google.android.contacts/android.view.ViewGroup"
        ) is False

    def test_empty_not_declared(self, tmp_path):
        tracker = ActivityCoverageTracker()
        tracker.initialize(str(tmp_path), ["com.test/.A"])
        assert tracker.is_declared("") is False


class TestNormalize:
    def test_shorthand_to_full(self):
        assert _normalize_activity_name("com.test.app/.MainActivity") == \
            "com.test.app/com.test.app.MainActivity"

    def test_full_format_unchanged(self):
        assert _normalize_activity_name("com.test.app/com.test.app.MainActivity") == \
            "com.test.app/com.test.app.MainActivity"

    def test_no_slash(self):
        assert _normalize_activity_name("SomeActivity") == "SomeActivity"

    def test_inner_class(self):
        assert _normalize_activity_name("com.test.app/.Outer$Inner") == \
            "com.test.app/com.test.app.Outer$Inner"
