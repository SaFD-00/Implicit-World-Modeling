"""Tests for monkey_collector.config — YAML loading, env overrides, CLI merge."""

import argparse
from pathlib import Path

from monkey_collector.config import (
    VALID_STRATEGIES,
    load_run_config,
    merge_with_cli_args,
)

NONEXISTENT = Path("/nonexistent/run.yaml")


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _full_args(**overrides) -> argparse.Namespace:
    """Namespace with every field merge_with_cli_args reads, defaulting to None."""
    base = dict(
        strategy=None,
        steps=None,
        seed=None,
        delay=None,
        port=None,
        data_dir=None,
        runtime_dir=None,
        input_mode=None,
        element_extraction=None,
        screen_grouping=None,
        cluster_merge_tolerance=None,
        max_expand_iters=None,
        luminance_prefilter=None,
        luminance_threshold=None,
        screenshot_diff_threshold=None,
        luminance_low_res_width=None,
        persist_filtered=None,
        config=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ── builtin defaults ──

def test_builtin_defaults_no_yaml():
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.exploration.strategy == "BFS"
    assert cfg.collection.max_steps == 1500
    assert cfg.collection.seed == 42
    assert cfg.collection.action_delay_ms == 1500
    assert cfg.collection.port == 12345
    assert cfg.collection.data_dir == "data"
    assert cfg.collection.runtime_dir == "runtime"
    assert cfg.llm.input_mode == "api"
    assert cfg.llm.element_extraction is False
    assert cfg.screen_matching.cluster_merge_tolerance == 0.2
    assert cfg.screen_matching.max_expand_iters == 3
    assert cfg.screen_matching.luminance_prefilter is True
    assert cfg.screen_matching.luminance_threshold == 10
    assert cfg.screen_matching.screenshot_diff_threshold == 0.02
    assert cfg.screen_matching.luminance_low_res_width == 100
    assert cfg.screen_matching.persist_filtered is True


def test_valid_strategies_set():
    assert frozenset({"DFS", "BFS", "GREEDY"}) == VALID_STRATEGIES


# ── YAML over builtin ──

def test_yaml_overrides_builtin(tmp_path):
    path = _write_yaml(tmp_path, "exploration:\n  strategy: DFS\ncollection:\n  max_steps: 7\n")
    cfg = load_run_config(path=path)
    assert cfg.exploration.strategy == "DFS"
    assert cfg.collection.max_steps == 7
    # untouched keys keep builtin values
    assert cfg.collection.seed == 42


def test_partial_yaml_keeps_builtin_for_missing_sections(tmp_path):
    path = _write_yaml(tmp_path, "collection:\n  seed: 99\n")
    cfg = load_run_config(path=path)
    assert cfg.collection.seed == 99
    assert cfg.exploration.strategy == "BFS"  # builtin


def test_empty_yaml_is_builtin(tmp_path):
    path = _write_yaml(tmp_path, "")
    cfg = load_run_config(path=path)
    assert cfg.exploration.strategy == "BFS"


# ── env over YAML ──

def test_env_overrides_yaml(tmp_path, monkeypatch):
    path = _write_yaml(tmp_path, "exploration:\n  strategy: GREEDY\n")
    monkeypatch.setenv("MC_EXPLORATION_STRATEGY", "BFS")
    cfg = load_run_config(path=path)
    assert cfg.exploration.strategy == "BFS"


def test_env_int_coercion(monkeypatch):
    monkeypatch.setenv("MC_COLLECTION_MAX_STEPS", "321")
    monkeypatch.setenv("MC_COLLECTION_PORT", "55555")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.collection.max_steps == 321
    assert cfg.collection.port == 55555


def test_env_float_coercion(monkeypatch):
    monkeypatch.setenv("MC_SCREEN_MATCHING_CLUSTER_MERGE_TOLERANCE", "0.45")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.screen_matching.cluster_merge_tolerance == 0.45


def test_env_luminance_coercion(monkeypatch):
    monkeypatch.setenv("MC_SCREEN_MATCHING_LUMINANCE_PREFILTER", "off")
    monkeypatch.setenv("MC_SCREEN_MATCHING_LUMINANCE_THRESHOLD", "25")
    monkeypatch.setenv("MC_SCREEN_MATCHING_SCREENSHOT_DIFF_THRESHOLD", "0.1")
    monkeypatch.setenv("MC_SCREEN_MATCHING_LUMINANCE_LOW_RES_WIDTH", "64")
    monkeypatch.setenv("MC_SCREEN_MATCHING_PERSIST_FILTERED", "off")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.screen_matching.luminance_prefilter is False
    assert cfg.screen_matching.luminance_threshold == 25
    assert cfg.screen_matching.screenshot_diff_threshold == 0.1
    assert cfg.screen_matching.luminance_low_res_width == 64
    assert cfg.screen_matching.persist_filtered is False


def test_env_bool_coercion(monkeypatch):
    monkeypatch.setenv("MC_LLM_ELEMENT_EXTRACTION", "false")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.llm.element_extraction is False


def test_env_bool_coercion_truthy(monkeypatch):
    monkeypatch.setenv("MC_LLM_ELEMENT_EXTRACTION", "on")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.llm.element_extraction is True


def test_mc_config_path_env_respected(tmp_path, monkeypatch):
    path = _write_yaml(tmp_path, "exploration:\n  strategy: DFS\n")
    monkeypatch.setenv("MC_CONFIG_PATH", str(path))
    cfg = load_run_config()  # no explicit path → reads MC_CONFIG_PATH
    assert cfg.exploration.strategy == "DFS"


# ── strategy validation / normalisation ──

def test_invalid_strategy_falls_back_to_greedy(tmp_path):
    path = _write_yaml(tmp_path, "exploration:\n  strategy: SIDEWAYS\n")
    cfg = load_run_config(path=path)
    assert cfg.exploration.strategy == "GREEDY"


def test_strategy_case_normalised(tmp_path):
    path = _write_yaml(tmp_path, "exploration:\n  strategy: dfs\n")
    cfg = load_run_config(path=path)
    assert cfg.exploration.strategy == "DFS"


def test_env_strategy_case_normalised(monkeypatch):
    monkeypatch.setenv("MC_EXPLORATION_STRATEGY", "greedy")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.exploration.strategy == "GREEDY"


# ── CLI merge ──

def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MC_COLLECTION_MAX_STEPS", "999")
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(cfg, _full_args(steps=200))
    assert cfg.collection.max_steps == 200


def test_cli_none_does_not_override(tmp_path):
    path = _write_yaml(tmp_path, "collection:\n  max_steps: 333\n")
    cfg = load_run_config(path=path)
    cfg = merge_with_cli_args(cfg, _full_args(steps=None))
    assert cfg.collection.max_steps == 333


def test_cli_strategy_normalised():
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(cfg, _full_args(strategy="dfs"))
    assert cfg.exploration.strategy == "DFS"


def test_cli_element_extraction_on_off():
    cfg = load_run_config(path=NONEXISTENT)
    off = merge_with_cli_args(cfg, _full_args(element_extraction="off"))
    assert off.llm.element_extraction is False
    on = merge_with_cli_args(cfg, _full_args(element_extraction="on"))
    assert on.llm.element_extraction is True


def test_cli_screen_grouping_off_disables_extraction():
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(cfg, _full_args(screen_grouping="off"))
    assert cfg.llm.element_extraction is False


def test_cli_invalid_strategy_falls_back_to_greedy():
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(cfg, _full_args(strategy="SIDEWAYS"))
    assert cfg.exploration.strategy == "GREEDY"


# ── regression: builtin defaults must not leak across calls ──

def test_no_state_leak_across_calls(monkeypatch):
    """An env override with no YAML must not mutate the module-global defaults."""
    monkeypatch.setenv("MC_COLLECTION_MAX_STEPS", "999")
    monkeypatch.setenv("MC_LLM_ELEMENT_EXTRACTION", "false")
    first = load_run_config(path=NONEXISTENT)
    assert first.collection.max_steps == 999
    assert first.llm.element_extraction is False

    # Remove the env vars; a fresh load must return the true builtin values,
    # not the previously-applied overrides.
    monkeypatch.delenv("MC_COLLECTION_MAX_STEPS")
    monkeypatch.delenv("MC_LLM_ELEMENT_EXTRACTION")
    second = load_run_config(path=NONEXISTENT)
    assert second.collection.max_steps == 1500
    assert second.llm.element_extraction is False


def test_cli_full_override():
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(
        cfg,
        _full_args(
            strategy="GREEDY",
            steps=10,
            seed=7,
            delay=250,
            port=4000,
            data_dir="/tmp/out",
            runtime_dir="/tmp/rt",
            input_mode="random",
            cluster_merge_tolerance=0.9,
            max_expand_iters=5,
            luminance_prefilter="off",
            luminance_threshold=30,
            screenshot_diff_threshold=0.05,
            luminance_low_res_width=80,
            persist_filtered="off",
        ),
    )
    assert cfg.exploration.strategy == "GREEDY"
    assert cfg.collection.max_steps == 10
    assert cfg.collection.seed == 7
    assert cfg.collection.action_delay_ms == 250
    assert cfg.collection.port == 4000
    assert cfg.collection.data_dir == "/tmp/out"
    assert cfg.collection.runtime_dir == "/tmp/rt"
    assert cfg.llm.input_mode == "random"
    assert cfg.screen_matching.cluster_merge_tolerance == 0.9
    assert cfg.screen_matching.max_expand_iters == 5
    assert cfg.screen_matching.luminance_prefilter is False
    assert cfg.screen_matching.luminance_threshold == 30
    assert cfg.screen_matching.screenshot_diff_threshold == 0.05
    assert cfg.screen_matching.luminance_low_res_width == 80
    assert cfg.screen_matching.persist_filtered is False


def test_cli_luminance_prefilter_on_off():
    cfg = load_run_config(path=NONEXISTENT)
    off = merge_with_cli_args(cfg, _full_args(luminance_prefilter="off"))
    assert off.screen_matching.luminance_prefilter is False
    on = merge_with_cli_args(cfg, _full_args(luminance_prefilter="on"))
    assert on.screen_matching.luminance_prefilter is True


def test_cli_persist_filtered_on_off():
    cfg = load_run_config(path=NONEXISTENT)
    off = merge_with_cli_args(cfg, _full_args(persist_filtered="off"))
    assert off.screen_matching.persist_filtered is False
    on = merge_with_cli_args(cfg, _full_args(persist_filtered="on"))
    assert on.screen_matching.persist_filtered is True
