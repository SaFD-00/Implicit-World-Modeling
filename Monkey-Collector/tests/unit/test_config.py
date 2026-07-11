"""Tests for monkey_collector.config — YAML loading, env overrides, CLI merge."""

import argparse
from pathlib import Path

from monkey_collector.config import (
    VALID_STRATEGIES,
    load_run_config,
    merge_with_cli_args,
    parse_duration,
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
        duration=None,
        budget_mode=None,
        signal_timeout=None,
        seed=None,
        delay=None,
        port=None,
        data_dir=None,
        runtime_dir=None,
        input_mode=None,
        element_extraction=None,
        screen_grouping=None,
        luminance_prefilter=None,
        luminance_threshold=None,
        screenshot_diff_threshold=None,
        luminance_low_res_width=None,
        persist_filtered=None,
        bm25_top_k=None,
        element_criterion=None,
        element_diff_max=None,
        element_jaccard_min=None,
        page_pixel_diff_threshold=None,
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
    assert cfg.collection.budget_mode == "time"
    assert cfg.collection.max_duration_sec == 7200
    assert cfg.collection.signal_timeout_sec == 12.0
    assert cfg.llm.input_mode == "api"
    assert cfg.llm.element_extraction is False
    assert cfg.screen_matching.luminance_prefilter is True
    assert cfg.screen_matching.luminance_threshold == 10
    assert cfg.screen_matching.screenshot_diff_threshold == 0.02
    assert cfg.screen_matching.luminance_low_res_width == 100
    assert cfg.screen_matching.persist_filtered is True
    assert cfg.screen_matching.bm25_top_k == 5
    assert cfg.screen_matching.element_criterion == "diff"
    assert cfg.screen_matching.element_diff_max == 5
    assert cfg.screen_matching.element_jaccard_min == 0.5
    assert cfg.screen_matching.page_pixel_diff_threshold == 0.3


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


def test_env_bm25_matching_coercion(monkeypatch):
    monkeypatch.setenv("MC_SCREEN_MATCHING_BM25_TOP_K", "3")
    monkeypatch.setenv("MC_SCREEN_MATCHING_ELEMENT_CRITERION", "jaccard")
    monkeypatch.setenv("MC_SCREEN_MATCHING_ELEMENT_DIFF_MAX", "8")
    monkeypatch.setenv("MC_SCREEN_MATCHING_ELEMENT_JACCARD_MIN", "0.7")
    monkeypatch.setenv("MC_SCREEN_MATCHING_PAGE_PIXEL_DIFF_THRESHOLD", "0.25")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.screen_matching.bm25_top_k == 3
    assert cfg.screen_matching.element_criterion == "jaccard"
    assert cfg.screen_matching.element_diff_max == 8
    assert cfg.screen_matching.element_jaccard_min == 0.7
    assert cfg.screen_matching.page_pixel_diff_threshold == 0.25


def test_invalid_element_criterion_falls_back_to_diff(monkeypatch):
    monkeypatch.setenv("MC_SCREEN_MATCHING_ELEMENT_CRITERION", "cosine")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.screen_matching.element_criterion == "diff"


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
            luminance_prefilter="off",
            luminance_threshold=30,
            screenshot_diff_threshold=0.05,
            luminance_low_res_width=80,
            persist_filtered="off",
            bm25_top_k=7,
            element_criterion="jaccard",
            element_diff_max=9,
            element_jaccard_min=0.6,
            page_pixel_diff_threshold=0.2,
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
    assert cfg.screen_matching.luminance_prefilter is False
    assert cfg.screen_matching.luminance_threshold == 30
    assert cfg.screen_matching.screenshot_diff_threshold == 0.05
    assert cfg.screen_matching.luminance_low_res_width == 80
    assert cfg.screen_matching.persist_filtered is False
    assert cfg.screen_matching.bm25_top_k == 7
    assert cfg.screen_matching.element_criterion == "jaccard"
    assert cfg.screen_matching.element_diff_max == 9
    assert cfg.screen_matching.element_jaccard_min == 0.6
    assert cfg.screen_matching.page_pixel_diff_threshold == 0.2


def test_cli_element_criterion_normalised():
    cfg = load_run_config(path=NONEXISTENT)
    valid = merge_with_cli_args(cfg, _full_args(element_criterion="jaccard"))
    assert valid.screen_matching.element_criterion == "jaccard"
    invalid = merge_with_cli_args(cfg, _full_args(element_criterion="cosine"))
    assert invalid.screen_matching.element_criterion == "diff"


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


def test_legacy_screen_matching_keys_are_ignored(tmp_path, monkeypatch):
    legacy_cluster = "_".join(("cluster", "merge", "tolerance"))
    legacy_expand = "_".join(("max", "expand", "iters"))
    path = _write_yaml(
        tmp_path,
        (
            "screen_matching:\n"
            f"  {legacy_cluster}: 0.9\n"
            f"  {legacy_expand}: 7\n"
            "  luminance_prefilter: false\n"
        ),
    )
    monkeypatch.setenv("MC_SCREEN_MATCHING_CLUSTER_MERGE_TOLERANCE", "0.45")

    cfg = load_run_config(path=path)

    assert cfg.screen_matching.luminance_prefilter is False
    assert not hasattr(cfg.screen_matching, legacy_cluster)
    assert not hasattr(cfg.screen_matching, legacy_expand)


# ── parse_duration ──

def test_parse_duration_hour_suffix():
    assert parse_duration("2h") == 7200


def test_parse_duration_minute_suffix():
    assert parse_duration("120m") == 7200


def test_parse_duration_second_suffix():
    assert parse_duration("7200s") == 7200


def test_parse_duration_bare_number_string():
    assert parse_duration("7200") == 7200


def test_parse_duration_int():
    assert parse_duration(7200) == 7200


def test_parse_duration_case_insensitive():
    assert parse_duration("2H") == 7200


def test_parse_duration_invalid_falls_back_to_7200():
    assert parse_duration("not-a-duration") == 7200


def test_parse_duration_non_positive_falls_back_to_7200():
    assert parse_duration("-5m") == 7200
    assert parse_duration(0) == 7200


# ── budget_mode / max_duration (YAML + env) ──

def test_yaml_max_duration_overrides_builtin(tmp_path):
    path = _write_yaml(tmp_path, "collection:\n  max_duration: 90m\n")
    cfg = load_run_config(path=path)
    assert cfg.collection.max_duration_sec == 5400
    assert cfg.collection.budget_mode == "time"  # builtin default retained


def test_yaml_budget_mode_steps(tmp_path):
    path = _write_yaml(tmp_path, "collection:\n  budget_mode: steps\n")
    cfg = load_run_config(path=path)
    assert cfg.collection.budget_mode == "steps"


def test_env_max_duration_coercion(monkeypatch):
    monkeypatch.setenv("MC_COLLECTION_MAX_DURATION", "45m")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.collection.max_duration_sec == 2700


def test_env_budget_mode_coercion(monkeypatch):
    monkeypatch.setenv("MC_COLLECTION_BUDGET_MODE", "steps")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.collection.budget_mode == "steps"


def test_invalid_budget_mode_falls_back_to_time(monkeypatch):
    monkeypatch.setenv("MC_COLLECTION_BUDGET_MODE", "bogus")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.collection.budget_mode == "time"


# ── signal_timeout_sec (YAML + env + CLI + non-positive fallback) ──

def test_yaml_signal_timeout_overrides_builtin(tmp_path):
    path = _write_yaml(tmp_path, "collection:\n  signal_timeout_sec: 8\n")
    cfg = load_run_config(path=path)
    assert cfg.collection.signal_timeout_sec == 8.0


def test_env_signal_timeout_coercion(monkeypatch):
    monkeypatch.setenv("MC_COLLECTION_SIGNAL_TIMEOUT_SEC", "20")
    cfg = load_run_config(path=NONEXISTENT)
    assert cfg.collection.signal_timeout_sec == 20.0


def test_non_positive_signal_timeout_falls_back_to_12(tmp_path):
    path = _write_yaml(tmp_path, "collection:\n  signal_timeout_sec: 0\n")
    cfg = load_run_config(path=path)
    assert cfg.collection.signal_timeout_sec == 12.0

    neg = _write_yaml(tmp_path, "collection:\n  signal_timeout_sec: -5\n")
    cfg2 = load_run_config(path=neg)
    assert cfg2.collection.signal_timeout_sec == 12.0


def test_cli_signal_timeout_override():
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(cfg, _full_args(signal_timeout=20.0))
    assert cfg.collection.signal_timeout_sec == 20.0


# ── CLI: budget-mode / duration resolution (D2) ──

def test_cli_duration_only_infers_time_mode():
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(cfg, _full_args(duration="90m"))
    assert cfg.collection.budget_mode == "time"
    assert cfg.collection.max_duration_sec == 5400


def test_cli_steps_only_infers_steps_mode():
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(cfg, _full_args(steps=50))
    assert cfg.collection.budget_mode == "steps"
    assert cfg.collection.max_steps == 50


def test_cli_explicit_budget_mode_wins_over_inference():
    """--budget-mode steps + --duration 90m: the explicit mode wins (steps),
    though --duration still updates the (unused) max_duration_sec value."""
    cfg = load_run_config(path=NONEXISTENT)
    cfg = merge_with_cli_args(
        cfg, _full_args(budget_mode="steps", duration="90m")
    )
    assert cfg.collection.budget_mode == "steps"
    assert cfg.collection.max_duration_sec == 5400


def test_cli_both_steps_and_duration_without_mode_keeps_config_and_warns(caplog):
    cfg = load_run_config(path=NONEXISTENT)  # builtin budget_mode == "time"
    with caplog.at_level("WARNING"):
        merged = merge_with_cli_args(cfg, _full_args(steps=50, duration="90m"))
    assert merged.collection.budget_mode == "time"
    assert merged.collection.max_steps == 50
    assert merged.collection.max_duration_sec == 5400
    assert any("budget_mode" in r.message for r in caplog.records)


def test_cli_neither_steps_nor_duration_keeps_config():
    cfg = load_run_config(path=NONEXISTENT)
    merged = merge_with_cli_args(cfg, _full_args())
    assert merged.collection.budget_mode == cfg.collection.budget_mode
    assert merged.collection.max_duration_sec == cfg.collection.max_duration_sec
