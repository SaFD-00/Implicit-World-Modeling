"""Tests for run-command package resolution + completed-session skip."""

from __future__ import annotations

import json
from pathlib import Path

from monkey_collector.cli import (
    _load_completed_packages,
    _resolve_app_contexts,
    _resolve_run_packages,
)


def _apps_csv(path: Path, rows: list[tuple[str, bool]]) -> None:
    """Write a minimal apps.csv with (package_id, installed) rows."""
    lines = [
        "category,sub_category,app_name,package_id,source,priority,notes,installed"
    ]
    for pkg, installed in rows:
        lines.append(
            f"A,B,{pkg},{pkg},PlayStore,High,,{'true' if installed else 'false'}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_session(runtime_dir: Path, package: str, *, completed: bool) -> None:
    """Seed a session under the runtime root's ``apps/`` sub-root — where the
    completed-package scan looks."""
    d = runtime_dir / "apps" / package
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "session_id": package,
        "package": package,
        "completed_at": "2026-04-21T00:00:00" if completed else None,
    }
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


class TestLoadCompletedPackages:
    def test_empty_output_dir(self, tmp_path: Path) -> None:
        assert _load_completed_packages(str(tmp_path / "missing")) == set()

    def test_only_completed_counted(self, tmp_path: Path) -> None:
        _seed_session(tmp_path, "com.done", completed=True)
        _seed_session(tmp_path, "com.inprogress", completed=False)
        assert _load_completed_packages(str(tmp_path)) == {"com.done"}

    def test_logs_sibling_is_not_scanned(self, tmp_path: Path) -> None:
        # runtime/logs/ sits next to runtime/apps/, so the package scan never
        # sees it — no name-based exclusion needed.
        _seed_session(tmp_path, "com.done", completed=True)
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "run_20260101_000000.log").write_text("log\n")
        assert _load_completed_packages(str(tmp_path)) == {"com.done"}

    def test_missing_metadata_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "apps" / "com.bare").mkdir(parents=True)
        _seed_session(tmp_path, "com.done", completed=True)
        assert _load_completed_packages(str(tmp_path)) == {"com.done"}

    def test_malformed_metadata_ignored(self, tmp_path: Path) -> None:
        d = tmp_path / "apps" / "com.broken"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text("{not: valid}", encoding="utf-8")
        _seed_session(tmp_path, "com.done", completed=True)
        assert _load_completed_packages(str(tmp_path)) == {"com.done"}


class TestResolveRunPackages:
    def test_explicit_list_preserves_order_and_dedupes(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "catalog").mkdir()
        _apps_csv(tmp_path / "catalog" / "apps.csv", [("com.a", True), ("com.b", True)])
        result = _resolve_run_packages(
            ["com.a", "com.b", "com.a"], str(tmp_path / "data"), force=False,
        )
        assert result == ["com.a", "com.b"]

    def test_all_returns_installed_only(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "catalog").mkdir()
        _apps_csv(tmp_path / "catalog" / "apps.csv", [
            ("com.yes", True),
            ("com.no", False),
            ("com.yes2", True),
        ])
        result = _resolve_run_packages(
            ["all"], str(tmp_path / "data"), force=False,
        )
        assert result == ["com.yes", "com.yes2"]

    def test_completed_app_is_skipped(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "catalog").mkdir()
        _apps_csv(tmp_path / "catalog" / "apps.csv", [
            ("com.done", True),
            ("com.fresh", True),
        ])
        output = tmp_path / "data"
        _seed_session(output, "com.done", completed=True)

        result = _resolve_run_packages(
            ["all"], str(output), force=False,
        )
        assert result == ["com.fresh"]

    def test_force_includes_completed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "catalog").mkdir()
        _apps_csv(tmp_path / "catalog" / "apps.csv", [
            ("com.done", True),
            ("com.fresh", True),
        ])
        output = tmp_path / "data"
        _seed_session(output, "com.done", completed=True)

        result = _resolve_run_packages(
            ["all"], str(output), force=True,
        )
        assert result == ["com.done", "com.fresh"]

    def test_explicit_list_also_skips_completed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "catalog").mkdir()
        _apps_csv(tmp_path / "catalog" / "apps.csv", [
            ("com.done", True),
            ("com.fresh", True),
        ])
        output = tmp_path / "data"
        _seed_session(output, "com.done", completed=True)

        result = _resolve_run_packages(
            ["com.done", "com.fresh"], str(output), force=False,
        )
        assert result == ["com.fresh"]

    def test_in_progress_session_is_not_skipped(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A session with completed_at=null should be re-run (resume)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "catalog").mkdir()
        _apps_csv(tmp_path / "catalog" / "apps.csv", [("com.a", True)])
        output = tmp_path / "data"
        _seed_session(output, "com.a", completed=False)

        result = _resolve_run_packages(
            ["all"], str(output), force=False,
        )
        assert result == ["com.a"]


class TestResolveAppContexts:
    def test_maps_package_to_description(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "catalog").mkdir()
        _apps_csv(tmp_path / "catalog" / "apps.csv", [("com.a", True), ("com.b", True)])
        # _apps_csv writes app_name=pkg, category=A, sub_category=B, notes="".
        contexts = _resolve_app_contexts(["com.a", "com.b"])
        assert contexts == {"com.a": "com.a (A/B)", "com.b": "com.b (A/B)"}

    def test_missing_csv_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)  # no catalog/apps.csv here
        assert _resolve_app_contexts(["com.a"]) == {}

    def test_unknown_package_omitted(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "catalog").mkdir()
        _apps_csv(tmp_path / "catalog" / "apps.csv", [("com.a", True)])
        contexts = _resolve_app_contexts(["com.a", "com.unknown"])
        assert "com.unknown" not in contexts
        assert contexts == {"com.a": "com.a (A/B)"}
