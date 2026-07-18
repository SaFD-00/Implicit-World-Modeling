"""Tests for catalog.download_apks — CLI parsing, F-Droid build selection,
partition logic, Play Store subprocess command, and the MISSING.json ledger
(plus its MISSING.md rendering).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MC_ROOT = Path(__file__).resolve().parents[2]
if str(_MC_ROOT) not in sys.path:
    sys.path.insert(0, str(_MC_ROOT))

from catalog.download_apks import (  # noqa: E402
    DownloadError,
    DownloadResult,
    LedgerError,
    _atomic_write_text,
    build_gplaydl_command,
    finalize_playstore_output,
    load_missing_ledger,
    main,
    parse_args,
    partition_jobs,
    pick_fdroid_build,
    render_missing_md,
    update_missing_ledger,
)

from monkey_collector.pipeline.app_catalog import AppJob  # noqa: E402

# ── CLI ────────────────────────────────────────────────────────────────────

def test_parse_args_defaults() -> None:
    ns = parse_args([])
    assert ns.source == "all"
    assert ns.abi == "x86_64"
    assert ns.playstore_arch == "arm64"
    assert ns.force is False
    assert ns.dry_run is False


def test_parse_args_overrides() -> None:
    ns = parse_args(
        ["--source", "fdroid", "--abi", "arm64-v8a", "--only", "a.b,c.d", "--force"]
    )
    assert ns.source == "fdroid"
    assert ns.abi == "arm64-v8a"
    assert ns.only == "a.b,c.d"
    assert ns.force is True


# ── partition_jobs ─────────────────────────────────────────────────────────

def _job(pkg: str, source: str) -> AppJob:
    return AppJob(
        category="X",
        sub_category="Y",
        app_name=pkg,
        package_id=pkg,
        source=source,
        priority="Medium",
    )


def test_partition_jobs_splits_by_source() -> None:
    jobs = [
        _job("a.fdroid", "F-Droid"),
        _job("b.playstore", "PlayStore"),
        _job("c.system", "System"),
    ]
    fdroid, playstore, system = partition_jobs(jobs, {"fdroid", "playstore"}, None)
    assert [j.package_id for j in fdroid] == ["a.fdroid"]
    assert [j.package_id for j in playstore] == ["b.playstore"]
    assert [j.package_id for j in system] == ["c.system"]


def test_partition_jobs_honors_only_allowlist() -> None:
    jobs = [
        _job("keep.me", "F-Droid"),
        _job("skip.me", "F-Droid"),
        _job("also.keep", "PlayStore"),
    ]
    fdroid, playstore, _ = partition_jobs(
        jobs, {"fdroid", "playstore"}, only={"keep.me", "also.keep"}
    )
    assert [j.package_id for j in fdroid] == ["keep.me"]
    assert [j.package_id for j in playstore] == ["also.keep"]


def test_partition_jobs_source_filter_skips_other_source() -> None:
    jobs = [_job("a", "F-Droid"), _job("b", "PlayStore")]
    fdroid, playstore, _ = partition_jobs(jobs, {"fdroid"}, None)
    assert len(fdroid) == 1 and len(playstore) == 0


# ── F-Droid build selection ────────────────────────────────────────────────
# `pick_fdroid_build` now accepts an index-v2.json package entry of shape:
#   {"metadata": {...}, "versions": {<key>: {"manifest": {...}, "file": {...}}}}


def _entry(*versions: dict) -> dict:
    return {"versions": {f"k{i}": v for i, v in enumerate(versions)}}


def _version(version_code: int, nativecode: list[str] | None, file_name: str, sha: str = "") -> dict:
    manifest: dict = {"versionCode": version_code}
    if nativecode is not None:
        manifest["nativecode"] = nativecode
    return {"manifest": manifest, "file": {"name": file_name, "sha256": sha or "a" * 64}}


def test_pick_fdroid_build_prefers_matching_abi() -> None:
    entry = _entry(
        _version(100, ["arm64-v8a"], "/old_100.apk"),
        _version(200, ["x86_64", "arm64-v8a"], "/new_200.apk"),
    )
    build = pick_fdroid_build(entry, "x86_64")
    assert build["file_name"] == "/new_200.apk"
    assert build["versionCode"] == 200


def test_pick_fdroid_build_universal_when_no_nativecode() -> None:
    entry = _entry(_version(42, None, "/any_42.apk"))
    build = pick_fdroid_build(entry, "x86_64")
    assert build["file_name"] == "/any_42.apk"


def test_pick_fdroid_build_treats_empty_nativecode_as_universal() -> None:
    entry = _entry(_version(42, [], "/any_42.apk"))
    build = pick_fdroid_build(entry, "x86_64")
    assert build["file_name"] == "/any_42.apk"


def test_pick_fdroid_build_raises_when_no_matching_abi() -> None:
    entry = _entry(
        _version(1, ["arm64-v8a"], "/a.apk"),
        _version(2, ["armeabi-v7a"], "/b.apk"),
    )
    with pytest.raises(DownloadError) as ei:
        pick_fdroid_build(entry, "x86_64")
    assert "x86_64" in str(ei.value)


def test_pick_fdroid_build_raises_when_empty_entry() -> None:
    with pytest.raises(DownloadError):
        pick_fdroid_build({"versions": {}}, "x86_64")


def test_pick_fdroid_build_returns_highest_version_code() -> None:
    entry = _entry(
        _version(2, None, "/v2.apk"),
        _version(5, None, "/v5.apk"),
        _version(3, None, "/v3.apk"),
    )
    assert pick_fdroid_build(entry, "x86_64")["file_name"] == "/v5.apk"


# ── gplaydl command ────────────────────────────────────────────────────────

def test_build_gplaydl_command_shape(tmp_path: Path) -> None:
    cmd = build_gplaydl_command("com.example", "arm64", tmp_path)
    assert cmd[0] == "gplaydl"
    assert cmd[1] == "download"
    assert cmd[2] == "com.example"
    assert "--output" in cmd and str(tmp_path) in cmd
    assert "--arch" in cmd and "arm64" in cmd
    assert "--no-extras" in cmd


# ── finalize_playstore_output ──────────────────────────────────────────────

def test_finalize_playstore_output_picks_base_over_splits(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "com.example"
    pkg_dir.mkdir()
    base = pkg_dir / "base.apk"
    split = pkg_dir / "split_config.xhdpi.apk"
    base.write_bytes(b"A" * 2048)
    split.write_bytes(b"B" * 128)

    dest = tmp_path / "out.apk"
    result_path, had_splits = finalize_playstore_output(tmp_path, dest)
    assert result_path == dest
    assert had_splits is True
    assert dest.read_bytes() == b"A" * 2048


def test_finalize_playstore_output_no_splits(tmp_path: Path) -> None:
    (tmp_path / "pkg.apk").write_bytes(b"X" * 512)
    dest = tmp_path / "out.apk"
    _, had_splits = finalize_playstore_output(tmp_path, dest)
    assert had_splits is False
    assert dest.exists()


def test_finalize_playstore_output_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        finalize_playstore_output(tmp_path, tmp_path / "out.apk")


# ── Ledger rendering (MISSING.json → MISSING.md) ───────────────────────────
# The ledger, not the current run's results, is what gets rendered. These two
# tests carry over the pre-ledger renderer's contract: group by source, omit
# empty sections, never list a package that was downloaded successfully.


def _ledger_from(
    results: list[DownloadResult],
    system: list,
    apks_dir: Path,
    *,
    run: dict | None = None,
) -> dict:
    ledger = update_missing_ledger(
        {"schema_version": 1, "entries": {}, "last_run": {}},
        results=results,
        system_jobs=system,
        scope_ids={r.package_id for r in results} | {j.package_id for j in system},
        apks_dir=apks_dir,
        now=datetime(2026, 4, 22, 14, 30, tzinfo=timezone.utc),
    )
    ledger["last_run"] = run or {
        "generated": "2026-04-22T14:30:00+00:00",
        "abi": "x86_64",
        "playstore_arch": "arm64",
        "scope": {"source": "all", "only": []},
        "total_targets": 3,
        "downloaded": 1,
        "skipped": 0,
        "failed": 2,
    }
    return ledger


def test_render_missing_md_groups_by_source(tmp_path: Path) -> None:
    results = [
        DownloadResult("a.ok", "F-Droid", "downloaded"),
        DownloadResult("b.bad", "F-Droid", "failed", reason="no x86_64 build"),
        DownloadResult("c.bad", "PlayStore", "failed", reason="gplaydl timeout"),
    ]
    system = [_job("com.android.settings", "System")]

    out = render_missing_md(_ledger_from(results, system, tmp_path))

    assert "# Missing APKs" in out
    assert "Generated: 2026-04-22T14:30:00+00:00" in out
    assert "## F-Droid (1)" in out
    assert "b.bad" in out and "no x86_64 build" in out
    assert "## PlayStore (1)" in out
    assert "c.bad" in out and "gplaydl timeout" in out
    assert "## System (1)" in out
    assert "com.android.settings" in out
    # Successful downloads are NOT listed
    assert "a.ok" not in out
    # Ledger entries carry first/last seen dates
    assert "(first seen: 2026-04-22, last seen: 2026-04-22)" in out


def test_render_missing_md_omits_empty_sections(tmp_path: Path) -> None:
    out = render_missing_md(
        _ledger_from(
            [DownloadResult(f"pkg.{i}", "F-Droid", "downloaded") for i in range(5)],
            [],
            tmp_path,
            run={
                "generated": "2026-04-22T14:30:00+00:00",
                "abi": "x86_64",
                "playstore_arch": "arm64",
                "scope": {"source": "all", "only": []},
                "total_targets": 5,
                "downloaded": 5,
                "skipped": 0,
                "failed": 0,
            },
        )
    )
    assert "## F-Droid" not in out
    assert "## PlayStore" not in out
    assert "## System" not in out
    assert "Failed: 0" in out


# ── End-to-end F-Droid mock ────────────────────────────────────────────────

def _mk_index(package_id: str, *, sha: str, file_name: str = "/pkg_1.apk") -> dict:
    return {
        "packages": {
            package_id: _entry(_version(1, ["x86_64"], file_name, sha=sha)),
        }
    }


def test_download_fdroid_writes_and_verifies_hash(tmp_path: Path) -> None:
    from catalog.download_apks import download_fdroid

    body = b"fake-apk-contents"
    sha = hashlib.sha256(body).hexdigest()

    apk_resp = MagicMock()
    apk_resp.status_code = 200
    apk_resp.iter_content = lambda chunk_size=65536: [body]
    apk_resp.raise_for_status = MagicMock()

    session = MagicMock()
    session.get.return_value = apk_resp

    dest = tmp_path / "com.example.apk"
    result = download_fdroid(
        session, "com.example", "x86_64", dest, index=_mk_index("com.example", sha=sha)
    )

    assert result.status == "downloaded"
    assert dest.exists() and dest.read_bytes() == body


def test_download_fdroid_records_hash_mismatch(tmp_path: Path) -> None:
    from catalog.download_apks import download_fdroid

    apk_resp = MagicMock()
    apk_resp.status_code = 200
    apk_resp.iter_content = lambda chunk_size=65536: [b"actual"]
    apk_resp.raise_for_status = MagicMock()

    session = MagicMock()
    session.get.return_value = apk_resp

    dest = tmp_path / "com.example.apk"
    result = download_fdroid(
        session,
        "com.example",
        "x86_64",
        dest,
        index=_mk_index("com.example", sha="deadbeef" * 8),
    )

    assert result.status == "failed"
    assert "mismatch" in result.reason
    assert not dest.exists()


def test_download_fdroid_raises_when_package_missing_from_index(tmp_path: Path) -> None:
    from catalog.download_apks import download_fdroid

    session = MagicMock()
    dest = tmp_path / "com.example.apk"
    result = download_fdroid(
        session, "missing.pkg", "x86_64", dest, index={"packages": {}}
    )

    assert result.status == "failed"
    assert "not in F-Droid index" in result.reason
    session.get.assert_not_called()


# ── PlayStore subprocess mock ──────────────────────────────────────────────

def test_download_playstore_invokes_gplaydl(tmp_path: Path) -> None:
    from catalog.download_apks import download_playstore

    def fake_run(cmd, capture_output, text, timeout, check):
        # Simulate gplaydl writing a base APK to the output dir.
        out_dir = Path(cmd[cmd.index("--output") + 1])
        (out_dir / "com.example").mkdir(parents=True, exist_ok=True)
        (out_dir / "com.example" / "base.apk").write_bytes(b"playstore-apk")
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        return proc

    dest = tmp_path / "com.example.apk"
    with patch("catalog.download_apks.subprocess.run", side_effect=fake_run):
        result = download_playstore("com.example", "arm64", dest)

    assert result.status == "downloaded"
    assert dest.read_bytes() == b"playstore-apk"


def test_download_playstore_failure_records_reason(tmp_path: Path) -> None:
    from catalog.download_apks import download_playstore

    proc = MagicMock()
    proc.returncode = 1
    proc.stdout = ""
    proc.stderr = "auth token expired"

    with patch("catalog.download_apks.subprocess.run", return_value=proc):
        result = download_playstore("com.example", "arm64", tmp_path / "x.apk")

    assert result.status == "failed"
    assert "gplaydl" in result.reason


# ── main() ledger semantics (end-to-end, fully mocked) ─────────────────────
# Regression guard for the clobber bug: MISSING.md used to be rewritten from
# scratch out of the current run's results, so any partial run (`--only` /
# `--source`) erased every record outside its own scope.

_CSV_HEADER = "category,sub_category,app_name,package_id,source,priority,notes,installed\n"
_SEEN = "2026-07-14T09:32:35+00:00"


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    body = "".join(f"Cat,Sub,{pkg},{pkg},{source},Medium,,false\n" for pkg, source in rows)
    path.write_text(_CSV_HEADER + body, encoding="utf-8")
    return path


def _ledger_entry(source: str, reason: str = "seeded reason", seen: str = _SEEN) -> dict:
    return {"source": source, "reason": reason, "first_seen": seen, "last_seen": seen}


def _seed_ledger(apks_dir: Path, entries: dict) -> Path:
    path = apks_dir / "MISSING.json"
    path.write_text(
        json.dumps({"schema_version": 1, "entries": entries, "last_run": {}}, indent=2),
        encoding="utf-8",
    )
    return path


def _run_main(argv: list[str], *, succeed: frozenset[str] = frozenset()) -> int:
    def fake_fdroid(session, package_id, abi, dest, *, index):
        if package_id in succeed:
            dest.write_bytes(b"apk-bytes")
            return DownloadResult(package_id, "F-Droid", "downloaded", path=dest)
        return DownloadResult(package_id, "F-Droid", "failed", reason="not in F-Droid index")

    def fake_playstore(package_id, arch, dest):
        if package_id in succeed:
            dest.write_bytes(b"apk-bytes")
            return DownloadResult(package_id, "PlayStore", "downloaded", path=dest)
        return DownloadResult(package_id, "PlayStore", "failed", reason="gplaydl rc=1: not found")

    with (
        patch("catalog.download_apks.fetch_fdroid_index", return_value={"packages": {}}),
        patch("catalog.download_apks.download_fdroid", side_effect=fake_fdroid),
        patch("catalog.download_apks.download_playstore", side_effect=fake_playstore),
    ):
        return main(argv)


@pytest.fixture
def apks_and_csv(tmp_path: Path) -> tuple[Path, Path]:
    apks_dir = tmp_path / "apks"
    apks_dir.mkdir()
    return apks_dir, tmp_path / "apps.csv"


def _argv(apks_dir: Path, csv_path: Path, *extra: str) -> list[str]:
    return ["--apks-dir", str(apks_dir), "--csv", str(csv_path), *extra]


def _entries(apks_dir: Path) -> dict:
    return json.loads((apks_dir / "MISSING.json").read_text(encoding="utf-8"))["entries"]


def test_main_only_scope_preserves_out_of_scope_ledger_entries(
    apks_and_csv: tuple[Path, Path],
) -> None:
    """THE regression: `--only b` must not erase the record for `a`."""
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("a.outside", "F-Droid"), ("b.target", "F-Droid")])
    _seed_ledger(apks_dir, {"a.outside": _ledger_entry("F-Droid", "no x86_64 build")})

    rc = _run_main(_argv(apks_dir, csv_path, "--only", "b.target"))

    assert rc == 1  # b.target failed
    entries = _entries(apks_dir)
    assert entries["a.outside"] == _ledger_entry("F-Droid", "no x86_64 build")
    assert entries["b.target"]["reason"] == "not in F-Droid index"

    md = (apks_dir / "MISSING.md").read_text(encoding="utf-8")
    assert "a.outside" in md and "no x86_64 build" in md
    assert "b.target" in md


def test_main_source_scope_preserves_other_source(apks_and_csv: tuple[Path, Path]) -> None:
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("f.pkg", "F-Droid"), ("p.pkg", "PlayStore")])
    _seed_ledger(
        apks_dir,
        {"f.pkg": _ledger_entry("F-Droid"), "p.pkg": _ledger_entry("PlayStore")},
    )

    _run_main(_argv(apks_dir, csv_path, "--source", "fdroid"))
    entries = _entries(apks_dir)
    assert entries["p.pkg"] == _ledger_entry("PlayStore")  # untouched
    assert entries["f.pkg"]["last_seen"] != _SEEN  # refreshed

    _run_main(_argv(apks_dir, csv_path, "--source", "playstore"))
    entries = _entries(apks_dir)
    assert entries["f.pkg"]["reason"] == "not in F-Droid index"  # still the fdroid run's
    assert entries["p.pkg"]["last_seen"] != _SEEN


def test_main_resolves_entry_when_download_succeeds(apks_and_csv: tuple[Path, Path]) -> None:
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("f.pkg", "F-Droid")])
    _seed_ledger(apks_dir, {"f.pkg": _ledger_entry("F-Droid")})

    rc = _run_main(_argv(apks_dir, csv_path), succeed=frozenset({"f.pkg"}))

    assert rc == 0
    assert _entries(apks_dir) == {}
    assert "f.pkg" not in (apks_dir / "MISSING.md").read_text(encoding="utf-8")


def test_main_resolves_entry_when_apk_already_present(apks_and_csv: tuple[Path, Path]) -> None:
    """A stale entry self-heals: the ledger means 'not held locally'."""
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("f.pkg", "F-Droid"), ("other.pkg", "F-Droid")])
    _seed_ledger(
        apks_dir,
        {"f.pkg": _ledger_entry("F-Droid"), "other.pkg": _ledger_entry("F-Droid")},
    )
    (apks_dir / "f.pkg.apk").write_bytes(b"already here")

    rc = _run_main(_argv(apks_dir, csv_path, "--only", "f.pkg"))

    assert rc == 0  # nothing attempted, nothing failed
    entries = _entries(apks_dir)
    assert "f.pkg" not in entries
    assert entries["other.pkg"] == _ledger_entry("F-Droid")  # out of scope, kept


def test_main_aborts_on_corrupt_ledger_without_writing(
    apks_and_csv: tuple[Path, Path],
) -> None:
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("f.pkg", "F-Droid")])
    corrupt = '{"entries": {"a.pkg": '
    (apks_dir / "MISSING.json").write_text(corrupt, encoding="utf-8")
    (apks_dir / "MISSING.md").write_text("# sentinel\n", encoding="utf-8")

    rc = _run_main(_argv(apks_dir, csv_path))

    assert rc == 2
    assert (apks_dir / "MISSING.json").read_text(encoding="utf-8") == corrupt
    assert (apks_dir / "MISSING.md").read_text(encoding="utf-8") == "# sentinel\n"
    assert not list(apks_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(None, id="null-entry"),
        pytest.param([], id="non-dict-entry"),
        pytest.param({"source": "F-Droid"}, id="missing-required-keys"),
        pytest.param(
            {"source": "F-Droid", "reason": "x", "first_seen": _SEEN, "last_seen": 17},
            id="non-string-field",
        ),
    ],
)
def test_main_aborts_on_structurally_broken_entry_before_downloading(
    apks_and_csv: tuple[Path, Path], entry: object
) -> None:
    """Syntactically valid JSON with a broken entry must die *before* downloads.

    The bug this guards: only the top-level shape was checked, so such a ledger
    passed load and crashed later in update_missing_ledger — after APKs had
    already been written to disk.
    """
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("f.pkg", "F-Droid"), ("p.pkg", "PlayStore")])
    corrupt = json.dumps({"schema_version": 1, "entries": {"a.pkg": entry}, "last_run": {}})
    (apks_dir / "MISSING.json").write_text(corrupt, encoding="utf-8")
    (apks_dir / "MISSING.md").write_text("# sentinel\n", encoding="utf-8")

    with (
        patch("catalog.download_apks.fetch_fdroid_index") as index_mock,
        patch("catalog.download_apks.download_fdroid") as fdroid_mock,
        patch("catalog.download_apks.download_playstore") as playstore_mock,
    ):
        rc = main(_argv(apks_dir, csv_path))

    assert rc == 2
    assert index_mock.call_count == 0
    assert fdroid_mock.call_count == 0
    assert playstore_mock.call_count == 0
    assert (apks_dir / "MISSING.json").read_text(encoding="utf-8") == corrupt
    assert (apks_dir / "MISSING.md").read_text(encoding="utf-8") == "# sentinel\n"
    assert not list(apks_dir.glob("*.tmp"))
    assert not list(apks_dir.glob("*.apk"))


def test_main_aborts_on_non_string_entry_key(apks_and_csv: tuple[Path, Path]) -> None:
    """JSON object keys are strings, so this only arrives via a hand-edited load."""
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("f.pkg", "F-Droid")])
    path = apks_dir / "MISSING.json"
    path.write_text("{}", encoding="utf-8")

    with patch(
        "catalog.download_apks.json.loads",
        return_value={"entries": {7: _ledger_entry("F-Droid")}},
    ):
        with pytest.raises(LedgerError, match="not a string"):
            load_missing_ledger(path)


def test_real_repo_ledger_passes_validation() -> None:
    """The recovered 24-entry ledger must not be rejected by the new checks."""
    ledger = load_missing_ledger(_MC_ROOT / "catalog" / "apks" / "MISSING.json")
    assert len(ledger["entries"]) == 24


def test_atomic_write_leaves_no_tmp_when_replace_fails(tmp_path: Path) -> None:
    target = tmp_path / "MISSING.json"
    target.write_text("original\n", encoding="utf-8")

    with patch.object(Path, "replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            _atomic_write_text(target, "new content")

    assert target.read_text(encoding="utf-8") == "original\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_main_preserves_first_seen_across_repeated_failures(
    apks_and_csv: tuple[Path, Path],
) -> None:
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("f.pkg", "F-Droid")])
    _seed_ledger(apks_dir, {"f.pkg": _ledger_entry("F-Droid", "old reason")})

    _run_main(_argv(apks_dir, csv_path))

    entry = _entries(apks_dir)["f.pkg"]
    assert entry["first_seen"] == _SEEN
    assert entry["last_seen"] != _SEEN
    assert entry["reason"] == "not in F-Droid index"


def test_main_creates_ledger_when_absent(apks_and_csv: tuple[Path, Path]) -> None:
    apks_dir, csv_path = apks_and_csv
    _write_csv(
        csv_path,
        [("f.pkg", "F-Droid"), ("p.pkg", "PlayStore"), ("com.android.settings", "System")],
    )
    assert not (apks_dir / "MISSING.json").exists()

    rc = _run_main(_argv(apks_dir, csv_path))

    assert rc == 1
    ledger = json.loads((apks_dir / "MISSING.json").read_text(encoding="utf-8"))
    assert ledger["schema_version"] == 1
    assert set(ledger["entries"]) == {"f.pkg", "p.pkg", "com.android.settings"}
    assert ledger["entries"]["com.android.settings"]["source"] == "System"
    assert ledger["last_run"]["scope"] == {"source": "all", "only": []}
    assert ledger["last_run"]["failed"] == 2
    md = (apks_dir / "MISSING.md").read_text(encoding="utf-8")
    assert "## F-Droid (1)" in md and "## PlayStore (1)" in md and "## System (1)" in md


def test_main_dry_run_writes_nothing(apks_and_csv: tuple[Path, Path]) -> None:
    apks_dir, csv_path = apks_and_csv
    _write_csv(csv_path, [("f.pkg", "F-Droid")])

    rc = _run_main(_argv(apks_dir, csv_path, "--dry-run"))

    assert rc == 0
    assert not (apks_dir / "MISSING.json").exists()
    assert not (apks_dir / "MISSING.md").exists()
