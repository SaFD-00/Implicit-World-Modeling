"""Tests for reset: delete collected session data by scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from monkey_collector.pipeline.reset import delete_targets, resolve_targets


def _seed(root: Path, package: str) -> Path:
    d = root / package
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text("{}")
    return d


class TestResolveTargets:
    def test_all_returns_both_roots(self, tmp_path):
        data_dir = tmp_path / "data" / "raw"
        runtime_dir = tmp_path / "runtime"
        _seed(data_dir, "com.a")
        _seed(runtime_dir, "com.a")
        targets = resolve_targets(data_dir, runtime_dir, all_=True)
        assert sorted(targets) == sorted([data_dir, runtime_dir])

    def test_all_excludes_missing_root(self, tmp_path):
        data_dir = tmp_path / "data" / "raw"
        runtime_dir = tmp_path / "runtime"  # never created
        _seed(data_dir, "com.a")
        targets = resolve_targets(data_dir, runtime_dir, all_=True)
        assert targets == [data_dir]

    def test_packages_returns_matching_dirs_in_both_roots(self, tmp_path):
        data_dir = tmp_path / "data" / "raw"
        runtime_dir = tmp_path / "runtime"
        _seed(data_dir, "com.a")
        _seed(data_dir, "com.c")
        _seed(runtime_dir / "apps", "com.a")
        _seed(runtime_dir / "apps", "com.c")
        targets = resolve_targets(data_dir, runtime_dir, packages=["com.a", "com.c"])
        assert sorted(targets) == sorted([
            data_dir / "com.a", data_dir / "com.c",
            runtime_dir / "apps" / "com.a", runtime_dir / "apps" / "com.c",
        ])

    def test_packages_filters_out_nonexistent(self, tmp_path):
        data_dir = tmp_path / "data" / "raw"
        runtime_dir = tmp_path / "runtime"
        _seed(data_dir, "com.a")
        _seed(runtime_dir / "apps", "com.a")
        targets = resolve_targets(data_dir, runtime_dir, packages=["com.a", "com.missing"])
        assert sorted(targets) == sorted([data_dir / "com.a", runtime_dir / "apps" / "com.a"])

    def test_packages_missing_one_root_returns_only_existing(self, tmp_path):
        # A package present only under data/raw (e.g. runtime/ already wiped)
        # still resolves — the other root's absence isn't an error.
        data_dir = tmp_path / "data" / "raw"
        runtime_dir = tmp_path / "runtime"
        _seed(data_dir, "com.a")
        targets = resolve_targets(data_dir, runtime_dir, packages=["com.a"])
        assert targets == [data_dir / "com.a"]

    def test_no_scope_raises(self, tmp_path):
        with pytest.raises(ValueError, match="scope"):
            resolve_targets(tmp_path / "data" / "raw", tmp_path / "runtime")


class TestProcessedCorpusPreserved:
    """``reset --all`` clears collection *session state* only.

    ``data/raw/`` and ``runtime/`` go; ``data/processed/`` — the derived
    training corpus produced by ``convert-all`` — is not session state and must
    survive. This holds structurally, not by a guard: ``data/processed`` is a
    *sibling* of the ``data/raw`` reset root, so ``rmtree`` never reaches it.
    """

    def test_all_scope_keeps_processed_corpus(self, tmp_path):
        data_root = tmp_path / "data"
        raw_dir = data_root / "raw"
        processed_dir = data_root / "processed"
        runtime_dir = tmp_path / "runtime"

        _seed(raw_dir, "com.a")
        _seed(runtime_dir, "com.a")
        (processed_dir / "images").mkdir(parents=True)
        (processed_dir / "images" / "episode_000001_step_0001.png").write_bytes(b"x")
        (processed_dir / "gui-model_stage1.jsonl").write_text('{"messages": []}\n')

        targets = resolve_targets(raw_dir, runtime_dir, all_=True)
        assert processed_dir not in targets
        delete_targets(targets, dry_run=False)

        assert not raw_dir.exists()
        assert not runtime_dir.exists()
        assert processed_dir.is_dir()
        assert (processed_dir / "gui-model_stage1.jsonl").read_text() == (
            '{"messages": []}\n'
        )
        assert (processed_dir / "images" / "episode_000001_step_0001.png").exists()
        # The data/ parent survives too — only its raw/ child was in scope.
        assert data_root.is_dir()


class TestDeleteTargets:
    def test_removes_dirs(self, tmp_path):
        d1 = _seed(tmp_path, "com.a")
        d2 = _seed(tmp_path, "com.b")
        count = delete_targets([d1, d2], dry_run=False)
        assert count == 2
        assert not d1.exists()
        assert not d2.exists()

    def test_dry_run_keeps_dirs(self, tmp_path):
        d = _seed(tmp_path, "com.a")
        count = delete_targets([d], dry_run=True)
        assert count == 0
        assert d.exists()

    def test_ignores_missing(self, tmp_path):
        count = delete_targets([tmp_path / "gone"], dry_run=False)
        assert count == 0
