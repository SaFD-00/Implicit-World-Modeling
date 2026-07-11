"""Activity coverage tracking for progressive measurement.

Logs which Android Activities are visited during exploration to a CSV file,
enabling generation of Progressive Activity Coverage charts.

The ground-truth set of declared activities (the denominator) is supplied
by the caller. The session manager prefers ``catalog/activities.json``
(static androguard extraction, fixed across sessions/devices) and falls
back to ``adb dumpsys package`` when the catalog has no entry for the
package — see ``pipeline.session_manager._resolve_declared_activities``.

When the ground truth is fixed (``allow_dynamic_total=False``), only
visits to activities present in that set increment ``unique_visited`` —
runtime activities outside the catalog (system dialogs, recovery screens,
etc.) are recorded in the ``activity`` column for traceability but ignored
for coverage. The legacy default ``allow_dynamic_total=True`` preserves
the original behaviour where target-package activities missing from
``dumpsys`` extend the total on first visit and every observed activity
counts toward ``unique_visited``.

CSV format:
    timestamp_sec,step,activity,unique_visited,total_activities,coverage
"""

import csv
import os
import time

from loguru import logger


def _normalize_activity_name(name: str) -> str:
    """Expand shorthand activity component name to full format.

    ``com.test.app/.MainActivity`` → ``com.test.app/com.test.app.MainActivity``

    This ensures names from ``dumpsys`` (shorthand) and from the Android
    AccessibilityService (full) map to the same key.
    """
    if "/" not in name:
        return name
    pkg, cls = name.split("/", 1)
    if cls.startswith("."):
        cls = pkg + cls
    return f"{pkg}/{cls}"


class ActivityCoverageTracker:
    """Tracks and persists activity coverage over time and steps."""

    CSV_COLUMNS = [
        "timestamp_sec", "step", "activity",
        "unique_visited", "total_activities", "coverage",
    ]

    def __init__(self):
        self.csv_path: str = ""
        self.visited_activities: set[str] = set()
        self.total_activities: list[str] = []
        self._total_set: set[str] = set()  # normalized names for O(1) lookup
        self._target_package: str = ""
        self._allow_dynamic_total: bool = True
        # normalized ``activity-alias → targetActivity`` map. An observed alias
        # visit resolves to its declared target so it counts toward coverage
        # (get_activities / dumpsys never surface aliases). Empty when the
        # ground truth came from dumpsys or the catalog had no aliases.
        self._alias_map: dict[str, str] = {}
        self.start_time: float = 0.0
        self._initialized = False

    def initialize(
        self,
        session_dir: str,
        total_activities: list[str],
        package: str = "",
        allow_dynamic_total: bool = True,
        aliases: dict[str, str] | None = None,
    ) -> None:
        """Set total activities and create CSV file with header.

        Resets internal state so the tracker can be reused across sessions.

        ``allow_dynamic_total=False`` fixes the denominator to the supplied
        list — used when the ground truth comes from ``catalog/activities.json``.

        ``aliases`` maps ``activity-alias`` component names to the declared
        ``targetActivity`` they resolve to; a visit to an alias then counts
        toward its target (aliases never extend the denominator).
        """
        self.csv_path = os.path.join(session_dir, "activity_coverage.csv")
        self.total_activities = list(total_activities)
        self._total_set = {_normalize_activity_name(a) for a in self.total_activities}
        self._target_package = package
        self._allow_dynamic_total = allow_dynamic_total
        self._alias_map = {
            _normalize_activity_name(k): _normalize_activity_name(v)
            for k, v in (aliases or {}).items()
        }
        self.visited_activities = set()
        self.start_time = time.time()

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
        self._initialized = True
        logger.info(
            f"Activity coverage tracker initialized: "
            f"{len(total_activities)} total activities, "
            f"csv={self.csv_path}"
        )

    def record(self, activity_name: str, step: int) -> dict:
        """Record a new observation and append to CSV.

        Args:
            activity_name: Current foreground Activity component name.
            step: Interaction step count.

        Returns:
            dict with coverage entry data.
        """
        if activity_name:
            if self._allow_dynamic_total:
                self.visited_activities.add(activity_name)
                # Expand total only for target package activities that
                # dumpsys missed (e.g. format mismatch). Skip other apps.
                normalized = _normalize_activity_name(activity_name)
                if normalized not in self._total_set:
                    pkg = normalized.split("/", 1)[0] if "/" in normalized else ""
                    if pkg == self._target_package:
                        self.total_activities.append(activity_name)
                        self._total_set.add(normalized)
            else:
                # Static ground truth: count only activities in the catalog,
                # keyed by normalized form so shorthand/full variants merge.
                normalized = _normalize_activity_name(activity_name)
                # An alias resolves to its declared target so the visit counts
                # (and alias/direct visits to the same screen merge).
                normalized = self._alias_map.get(normalized, normalized)
                if normalized in self._total_set:
                    self.visited_activities.add(normalized)

        total = max(len(self.total_activities), 1)
        coverage = min(1.0, len(self.visited_activities) / total)
        elapsed = time.time() - self.start_time

        entry = {
            "timestamp_sec": round(elapsed, 2),
            "step": step,
            "activity": activity_name,
            "unique_visited": len(self.visited_activities),
            "total_activities": len(self.total_activities),
            "coverage": round(coverage, 4),
        }

        if self._initialized:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writerow(entry)

        return entry

    def resume(
        self,
        session_dir: str,
        total_activities: list[str],
        package: str = "",
        allow_dynamic_total: bool = True,
        aliases: dict[str, str] | None = None,
    ) -> None:
        """Resume from existing activity_coverage.csv.

        Rebuilds visited_activities from CSV and appends new records. ``aliases``
        carries the same ``activity-alias → targetActivity`` map as
        ``initialize`` so alias rows in the CSV resolve onto their target.
        """
        self.csv_path = os.path.join(session_dir, "activity_coverage.csv")
        self.total_activities = list(total_activities)
        self._total_set = {_normalize_activity_name(a) for a in self.total_activities}
        self._target_package = package
        self._allow_dynamic_total = allow_dynamic_total
        self._alias_map = {
            _normalize_activity_name(k): _normalize_activity_name(v)
            for k, v in (aliases or {}).items()
        }
        self.visited_activities = set()
        self.start_time = time.time()

        if os.path.exists(self.csv_path):
            with open(self.csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    activity = row.get("activity", "")
                    if not activity:
                        continue
                    if self._allow_dynamic_total:
                        self.visited_activities.add(activity)
                        normalized = _normalize_activity_name(activity)
                        if normalized not in self._total_set:
                            pkg = normalized.split("/", 1)[0] if "/" in normalized else ""
                            if pkg == self._target_package:
                                self.total_activities.append(activity)
                                self._total_set.add(normalized)
                    else:
                        normalized = _normalize_activity_name(activity)
                        normalized = self._alias_map.get(normalized, normalized)
                        if normalized in self._total_set:
                            self.visited_activities.add(normalized)

        self._initialized = True
        logger.info(
            f"Activity coverage tracker resumed: "
            f"{len(self.visited_activities)} previously visited, "
            f"{len(total_activities)} total activities"
        )

    def is_declared(self, activity_name: str) -> bool:
        """Return whether *activity_name* is a known declared activity.

        Read-only membership check against the normalized denominator set. Used
        by the collection loop to decide whether an AccessibilityService
        ``activity_name`` can be trusted for coverage, or whether it is a
        generic View class (e.g. ``.../android.view.ViewGroup``) that must be
        resolved to the real foreground activity via adb.
        """
        if not activity_name:
            return False
        normalized = _normalize_activity_name(activity_name)
        # Trust an alias as declared: _resolve_coverage_activity then skips the
        # adb round-trip for a frame whose a11y name is a known alias.
        normalized = self._alias_map.get(normalized, normalized)
        return normalized in self._total_set

    def get_coverage(self) -> float:
        """Current coverage ratio (clamped to [0, 1])."""
        total = max(len(self.total_activities), 1)
        return min(1.0, len(self.visited_activities) / total)

    def get_visited_count(self) -> int:
        """Number of unique activities visited."""
        return len(self.visited_activities)
