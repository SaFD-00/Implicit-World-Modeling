"""Monkey-Collector 중앙 설정 로딩: YAML → dataclass, 환경변수 오버라이드.

Resolution order (later wins):
  builtin defaults (this file) → config/run.yaml → MC_* env vars → CLI flags

CLI flags are applied in cli.py via merge_with_cli_args(), not here.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Builtin defaults — must match config/run.yaml canonical values exactly.
# ---------------------------------------------------------------------------

_BUILTIN_DEFAULTS: dict = {
    "exploration": {
        "strategy": "BFS",
    },
    "collection": {
        "max_steps": 1500,
        "seed": 42,
        "action_delay_ms": 1500,
        "port": 12345,
        "output_dir": "data/raw",
    },
    "llm": {
        "input_mode": "api",
        "element_extraction": True,
    },
    "screen_matching": {
        "cluster_merge_tolerance": 0.2,
        "max_expand_iters": 3,
    },
}

VALID_STRATEGIES: frozenset[str] = frozenset({"DFS", "BFS", "GREEDY"})

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "run.yaml"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExplorationConfig:
    strategy: str = "BFS"  # DFS | BFS | GREEDY


@dataclass
class CollectionConfig:
    max_steps: int = 1500
    seed: int = 42
    action_delay_ms: int = 1500
    port: int = 12345
    output_dir: str = "data/raw"


@dataclass
class LlmConfig:
    input_mode: str = "api"       # api | random
    element_extraction: bool = True


@dataclass
class ScreenMatchingConfig:
    cluster_merge_tolerance: float = 0.2
    max_expand_iters: int = 3


@dataclass
class RunConfig:
    exploration: ExplorationConfig
    collection: CollectionConfig
    llm: LlmConfig
    screen_matching: ScreenMatchingConfig


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return {}


def _coerce_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "on")


def _normalize_strategy(value: object, *, source: str) -> str:
    """Uppercase + validate a strategy value; fall back to GREEDY if invalid.

    Used by every layer (YAML/env via _from_raw, CLI via merge_with_cli_args)
    so an unknown strategy is coerced consistently no matter where it came from.
    """
    s = str(value).strip().upper()
    if s not in VALID_STRATEGIES:
        logger.warning(
            "Unknown exploration strategy %r (from %s) — falling back to GREEDY. "
            "Valid options: %s",
            value,
            source,
            ", ".join(sorted(VALID_STRATEGIES)),
        )
        return "GREEDY"
    return s


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto base (non-destructive)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(raw: dict) -> dict:
    """Layer MC_* environment variables on top of raw config dict."""
    env_map: list[tuple[str, str, str, str]] = [
        # (env_var, section, field, type_hint)
        ("MC_EXPLORATION_STRATEGY",                  "exploration",     "strategy",                  "str_upper"),
        ("MC_COLLECTION_MAX_STEPS",                  "collection",      "max_steps",                 "int"),
        ("MC_COLLECTION_SEED",                       "collection",      "seed",                      "int"),
        ("MC_COLLECTION_ACTION_DELAY_MS",            "collection",      "action_delay_ms",            "int"),
        ("MC_COLLECTION_PORT",                       "collection",      "port",                      "int"),
        ("MC_COLLECTION_OUTPUT_DIR",                 "collection",      "output_dir",                "str"),
        ("MC_LLM_INPUT_MODE",                        "llm",             "input_mode",                "str"),
        ("MC_LLM_ELEMENT_EXTRACTION",                "llm",             "element_extraction",        "bool"),
        ("MC_SCREEN_MATCHING_CLUSTER_MERGE_TOLERANCE", "screen_matching", "cluster_merge_tolerance", "float"),
        ("MC_SCREEN_MATCHING_MAX_EXPAND_ITERS",      "screen_matching", "max_expand_iters",          "int"),
    ]

    # raw is already an isolated deep copy (see load_run_config); mutate in place.
    for env_var, section, field, type_hint in env_map:
        val = os.environ.get(env_var)
        if val is None:
            continue
        raw.setdefault(section, {})
        if type_hint == "int":
            raw[section][field] = int(val)
        elif type_hint == "float":
            raw[section][field] = float(val)
        elif type_hint == "bool":
            raw[section][field] = _coerce_bool(val)
        elif type_hint == "str_upper":
            raw[section][field] = val.strip().upper()
        else:
            raw[section][field] = val
    return raw


def _from_raw(raw: dict) -> RunConfig:
    """Convert a merged raw dict into typed RunConfig."""
    expl = raw.get("exploration", {})
    coll = raw.get("collection", {})
    llm  = raw.get("llm", {})
    sm   = raw.get("screen_matching", {})

    strategy = _normalize_strategy(expl.get("strategy", "BFS"), source="config")

    return RunConfig(
        exploration=ExplorationConfig(strategy=strategy),
        collection=CollectionConfig(
            max_steps=int(coll.get("max_steps", 1500)),
            seed=int(coll.get("seed", 42)),
            action_delay_ms=int(coll.get("action_delay_ms", 1500)),
            port=int(coll.get("port", 12345)),
            output_dir=str(coll.get("output_dir", "data/raw")),
        ),
        llm=LlmConfig(
            input_mode=str(llm.get("input_mode", "api")),
            element_extraction=_coerce_bool(llm.get("element_extraction", True)),
        ),
        screen_matching=ScreenMatchingConfig(
            cluster_merge_tolerance=float(sm.get("cluster_merge_tolerance", 0.2)),
            max_expand_iters=int(sm.get("max_expand_iters", 3)),
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_run_config(path: Path | str | None = None) -> RunConfig:
    """Load config: builtin defaults → YAML → MC_* env vars.

    *path* overrides the YAML file location. Defaults to config/run.yaml
    relative to the package root. Pass ``Path("/nonexistent")`` in tests to
    skip the file entirely.
    """
    yaml_path: Path
    if path is not None:
        yaml_path = Path(path)
    elif (env_path := os.environ.get("MC_CONFIG_PATH")):
        yaml_path = Path(env_path)
    else:
        yaml_path = _DEFAULT_CONFIG_PATH

    # Deep copy so layering (YAML/env) never mutates the module-global defaults.
    raw = copy.deepcopy(_BUILTIN_DEFAULTS)
    if yaml_path.exists():
        yaml_raw = _load_yaml(yaml_path)
        raw = _deep_merge(raw, yaml_raw)

    raw = _apply_env_overrides(raw)
    return _from_raw(raw)


def merge_with_cli_args(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    """Apply CLI flag overrides on top of *config*.

    Only overrides fields where the CLI arg is not None (sentinel for
    "user did not specify"). Boolean flags (force, new_session) are
    CLI-only and not represented in RunConfig.
    """
    expl = config.exploration
    coll = config.collection
    llm  = config.llm
    sm   = config.screen_matching

    # exploration.strategy (validated like the YAML/env paths)
    strategy_arg = getattr(args, "strategy", None)
    if strategy_arg is not None:
        expl = replace(expl, strategy=_normalize_strategy(strategy_arg, source="--strategy"))

    # collection
    steps = getattr(args, "steps", None)
    seed = getattr(args, "seed", None)
    delay = getattr(args, "delay", None)
    port = getattr(args, "port", None)
    output = getattr(args, "output", None)
    if steps is not None:
        coll = replace(coll, max_steps=steps)
    if seed is not None:
        coll = replace(coll, seed=seed)
    if delay is not None:
        coll = replace(coll, action_delay_ms=delay)
    if port is not None:
        coll = replace(coll, port=port)
    if output is not None:
        coll = replace(coll, output_dir=output)

    # llm
    input_mode = getattr(args, "input_mode", None)
    elem_extr = getattr(args, "element_extraction", None)
    if input_mode is not None:
        llm = replace(llm, input_mode=input_mode)
    if elem_extr is not None:
        llm = replace(llm, element_extraction=(elem_extr == "on"))
    # deprecated --screen-grouping alias
    screen_grouping = getattr(args, "screen_grouping", None)
    if screen_grouping == "off":
        llm = replace(llm, element_extraction=False)

    # screen_matching
    cmt = getattr(args, "cluster_merge_tolerance", None)
    mei = getattr(args, "max_expand_iters", None)
    if cmt is not None:
        sm = replace(sm, cluster_merge_tolerance=cmt)
    if mei is not None:
        sm = replace(sm, max_expand_iters=mei)

    return RunConfig(exploration=expl, collection=coll, llm=llm, screen_matching=sm)
