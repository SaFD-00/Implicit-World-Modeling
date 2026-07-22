"""Tests for monkey_collector.migrate_layout — legacy → bare layout migration."""

import json

import pytest

import monkey_collector.migrate_layout as ml
from monkey_collector.migrate_layout import migrate, strip_page_key


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_legacy(tmp_path):
    """A pre-refactor session: page_{N}/ page dirs, {obs:04d}/ obs dirs, and
    ``page_{N}`` page_key strings embedded in every JSON."""
    data = tmp_path / "data" / "com.test.app"
    pages = data / "pages"

    # page_0 with one observation (0000) carrying elements.json + raw.xml
    _write(pages / "page_0" / "page.json", json.dumps({"page_key": "page_0"}))
    _write(pages / "page_0" / "0000" / "raw.xml", "<hierarchy/>")
    _write(
        pages / "page_0" / "0000" / "elements.json",
        json.dumps({"page_key": "page_0", "match_type": "NEW", "elements": []}),
    )
    # page_1 with two observations (0000, 0001)
    _write(pages / "page_1" / "page.json", json.dumps({"page_key": "page_1"}))
    _write(pages / "page_1" / "0000" / "raw.xml", "<hierarchy/>")
    _write(pages / "page_1" / "0001" / "raw.xml", "<hierarchy/>")

    # page_graph.json: node page_key strings, integer-id edges
    _write(
        data / "page_graph.json",
        json.dumps({
            "nodes": [{"id": 0, "page_key": "page_0"}, {"id": 1, "page_key": "page_1"}],
            "edges": [{"from_page": 0, "to_page": 1, "action_type": "tap"}],
        }),
    )

    # events.jsonl: two real events + one open_app line with no page_key
    runtime = tmp_path / "runtime" / "com.test.app"
    _write(
        runtime / "events.jsonl",
        "\n".join([
            json.dumps({"frame_index": 0, "page_key": "page_0", "observation_num": 0}),
            json.dumps({"frame_index": 1, "page_key": "page_1", "observation_num": 0}),
            json.dumps({"action_type": "open_app", "transition": False}),
        ]) + "\n",
    )

    # repo-level log to relocate
    _write(tmp_path / "logs" / "run_20260101_000000.log", "log line\n")
    return tmp_path


def _paths(tmp_path):
    return (
        str(tmp_path / "data"),
        str(tmp_path / "runtime"),
        str(tmp_path / "logs"),
    )


def test_strip_page_key_idempotent():
    assert strip_page_key("page_7") == "7"
    assert strip_page_key("page_0") == "0"
    assert strip_page_key("7") == "7"          # already bare
    assert strip_page_key("page_") == "page_"  # no digits → unchanged
    assert strip_page_key("home") == "home"


def test_dry_run_reports_without_touching_disk(tmp_path):
    _build_legacy(tmp_path)
    data, runtime, logs = _paths(tmp_path)

    result = migrate(data, runtime, logs, apply=False)

    assert result["applied"] is False
    assert result["operations"]  # non-empty plan
    # Nothing on disk changed.
    assert (tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0000").is_dir()
    assert not (tmp_path / "data" / "com.test.app" / "pages" / "0").exists()
    pj = json.loads((tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "page.json").read_text())
    assert pj["page_key"] == "page_0"
    assert (tmp_path / "logs" / "run_20260101_000000.log").is_file()
    assert (tmp_path / "runtime" / "com.test.app" / "events.jsonl").is_file()
    assert not (tmp_path / "runtime" / "apps").exists()


def test_dry_run_counts_events_still_at_the_runtime_root(tmp_path):
    # The apps/ move has not happened yet on a dry run, so the event scan has
    # to look at the legacy root too — otherwise the plan under-reports.
    _build_legacy(tmp_path)
    data, runtime, logs = _paths(tmp_path)

    result = migrate(data, runtime, logs, apply=False)

    assert result["counts"]["apps"] == 1
    assert result["counts"]["events"] == 2


def test_apply_renames_dirs_and_rewrites_keys(tmp_path):
    _build_legacy(tmp_path)
    data, runtime, logs = _paths(tmp_path)

    result = migrate(data, runtime, logs, apply=True)
    c = result["counts"]

    pages = tmp_path / "data" / "com.test.app" / "pages"
    # page dirs renamed, old names gone
    assert (pages / "0").is_dir() and (pages / "1").is_dir()
    assert not (pages / "page_0").exists() and not (pages / "page_1").exists()
    # observation dirs de-padded, raw.xml carried along
    assert (pages / "0" / "0" / "raw.xml").read_text() == "<hierarchy/>"
    assert (pages / "1" / "0").is_dir() and (pages / "1" / "1").is_dir()
    assert not (pages / "1" / "0000").exists()

    # embedded page_key rewritten everywhere
    assert json.loads((pages / "0" / "page.json").read_text())["page_key"] == "0"
    assert json.loads((pages / "0" / "0" / "elements.json").read_text())["page_key"] == "0"
    graph = json.loads((tmp_path / "data" / "com.test.app" / "page_graph.json").read_text())
    assert [n["page_key"] for n in graph["nodes"]] == ["0", "1"]
    # integer-id edges are untouched
    assert graph["edges"][0]["from_page"] == 0 and graph["edges"][0]["to_page"] == 1

    events = [
        json.loads(line)
        for line in (tmp_path / "runtime" / "apps" / "com.test.app" / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert events[0]["page_key"] == "0" and events[1]["page_key"] == "1"
    assert "page_key" not in events[2]  # open_app line preserved verbatim

    # package regrouped under runtime/apps/, log under the sibling runtime/logs/
    assert not (tmp_path / "runtime" / "com.test.app").exists()
    assert (tmp_path / "runtime" / "logs" / "run_20260101_000000.log").is_file()
    assert not (tmp_path / "logs" / "run_20260101_000000.log").exists()
    # logs/ stayed a sibling of apps/ instead of being swept into it
    assert not (tmp_path / "runtime" / "apps" / "logs").exists()

    assert c["pages"] == 2 and c["observations"] == 3
    assert c["page_json"] == 2 and c["elements_json"] == 1
    assert c["graphs"] == 1 and c["apps"] == 1 and c["events"] == 2 and c["logs"] == 1


def test_apply_is_idempotent(tmp_path):
    _build_legacy(tmp_path)
    data, runtime, logs = _paths(tmp_path)

    migrate(data, runtime, logs, apply=True)
    second = migrate(data, runtime, logs, apply=True)

    assert second["operations"] == []
    assert all(v == 0 for v in second["counts"].values())


def test_content_rewrite_is_atomic_on_failure(tmp_path, monkeypatch):
    # A failure during the atomic replace must leave the original file fully
    # intact (never truncated to 0 / half-written) and drop no stray temp file —
    # the migration's "safe to re-run" guarantee.
    _build_legacy(tmp_path)
    data, runtime, logs = _paths(tmp_path)

    def boom(*_a, **_k):
        raise OSError("simulated crash / disk-full during replace")

    monkeypatch.setattr(ml.os, "replace", boom)

    with pytest.raises(OSError):
        migrate(data, runtime, logs, apply=True)

    # First content rewrite site (elements.json) — original preserved verbatim.
    elements = tmp_path / "data" / "com.test.app" / "pages" / "page_0" / "0000" / "elements.json"
    assert json.loads(elements.read_text())["page_key"] == "page_0"
    # No rename happened before the failed write, and no temp file lingers.
    assert (tmp_path / "data" / "com.test.app" / "pages" / "page_0").is_dir()
    assert list(tmp_path.rglob("*.migrate.tmp")) == []


def test_missing_roots_are_noops(tmp_path):
    # Pointed at dirs that don't exist → empty plan, no crash.
    result = migrate(
        str(tmp_path / "nope_data"),
        str(tmp_path / "nope_runtime"),
        str(tmp_path / "nope_logs"),
        apply=True,
    )
    assert result["operations"] == []
    assert all(v == 0 for v in result["counts"].values())


def test_already_grouped_runtime_is_left_alone(tmp_path):
    # A tree already in the current layout: the package sits under apps/ and
    # the log under logs/. Nothing to move, and neither sub-root gets nested
    # inside the other.
    _write(
        tmp_path / "runtime" / "apps" / "com.test.app" / "events.jsonl",
        json.dumps({"frame_index": 0, "page_key": "0"}) + "\n",
    )
    _write(tmp_path / "runtime" / "logs" / "run_20260101_000000.log", "log line\n")
    data, runtime, logs = _paths(tmp_path)

    result = migrate(data, runtime, logs, apply=True)

    assert result["operations"] == []
    assert all(v == 0 for v in result["counts"].values())
    assert not (tmp_path / "runtime" / "apps" / "apps").exists()
    assert not (tmp_path / "runtime" / "apps" / "logs").exists()
    assert (tmp_path / "runtime" / "logs" / "run_20260101_000000.log").is_file()
