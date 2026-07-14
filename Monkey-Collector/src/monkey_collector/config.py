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
        "data_dir": "data",
        "runtime_dir": "runtime",
        "budget_mode": "time",
        "max_duration": "2h",
        "signal_timeout_sec": 12.0,
        "poke_delay_sec": 1.5,
        "max_action_repeats": 8,
        "max_steps_without_new_page": 98,
    },
    "llm": {
        "input_mode": "api",
        "element_extraction": False,
    },
    "screen_matching": {
        "luminance_prefilter": True,
        "luminance_threshold": 10,
        "screenshot_diff_threshold": 0.02,
        "luminance_low_res_width": 100,
        "persist_filtered": True,
        # BM25 unique-page matching (Mobile3M mechanism).
        "bm25_top_k": 5,
        "element_criterion": "diff",
        "element_diff_max": 5,
        "element_jaccard_min": 0.5,
        "page_pixel_diff_threshold": 0.3,
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
    data_dir: str = "data"
    runtime_dir: str = "runtime"
    # Session end condition. "time": run until max_duration_sec elapses
    # (product default). "steps": run until max_steps actions (legacy).
    budget_mode: str = "time"
    max_duration_sec: int = 7200
    # Per-signal wait before a stuck episode escalates (nudge → force-relaunch).
    # Env: MC_COLLECTION_SIGNAL_TIMEOUT_SEC. See recovery.MAX_SIGNAL_TIMEOUTS.
    signal_timeout_sec: float = 12.0
    # Silence (seconds) inside one signal wait after which the server pokes the
    # client with CAPTURE (up to recovery.MAX_POKES_PER_WAIT times). The pokes
    # are carved out of signal_timeout_sec, so the total wait is unchanged. Env:
    # MC_COLLECTION_POKE_DELAY_SEC. 0 or negative — or >= signal_timeout_sec —
    # disables poking (single full-timeout wait).
    poke_delay_sec: float = 1.5
    # Repeat-action circuit breaker (D2): max times the same
    # (page_key, action_type, element_index) may execute on a page before the
    # next attempt breaks out via back/relaunch. Env:
    # MC_COLLECTION_MAX_ACTION_REPEATS. 0 or negative disables the guard.
    max_action_repeats: int = 8
    # Plateau early-stop (D3): real-action steps with no new page after which the
    # session clean-stops (app saturated). Env:
    # MC_COLLECTION_MAX_STEPS_WITHOUT_NEW_PAGE. 0 or negative disables the guard.
    # 98 = 2x the largest productive gap ever observed in the archive (49 steps,
    # iter6 armA_musicplayer); the 2x margin is the hedge for the 2h budget, whose
    # gaps are unobserved (longest archived session is 1800s / 542 steps).
    max_steps_without_new_page: int = 98


@dataclass
class LlmConfig:
    input_mode: str = "api"       # api | random
    # Default OFF: the LLM is used for input-text generation only. Turn on to
    # add LLM element extraction + element-set screen matching.
    element_extraction: bool = False


@dataclass
class ScreenMatchingConfig:
    # Luminance prefilter (MobileGPT-V2 port). Governs the tighter OBSERVATION
    # identity dedup + the PAGE-level pixel gate's fingerprints. Default ON.
    luminance_prefilter: bool = True
    luminance_threshold: int = 10           # per-pixel |ΔY| change cutoff (0–255)
    screenshot_diff_threshold: float = 0.02  # changed-pixel fraction → same OBSERVATION
    luminance_low_res_width: int = 100       # fingerprint downscale width (px)
    # Persist a prefilter/dedup revisit as its OWN fresh observation (per-visit
    # chain) instead of writing nothing. Default ON — filtered screens are saved.
    persist_filtered: bool = True
    # BM25 unique-page matching (Mobile3M mechanism).
    bm25_top_k: int = 5                       # BM25 candidates to verify per screen
    element_criterion: str = "diff"           # "diff" (|A△B|<max) | "jaccard" (>min)
    element_diff_max: int = 5                 # symmetric-diff cutoff → same page
    element_jaccard_min: float = 0.5          # Jaccard floor → same page ("jaccard")
    page_pixel_diff_threshold: float = 0.3    # PAGE-level pixel gate (changed frac)


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


VALID_ELEMENT_CRITERIA: frozenset[str] = frozenset({"diff", "jaccard"})


def _normalize_criterion(value: object, *, source: str) -> str:
    """Lower + validate an element criterion; fall back to "diff" if invalid."""
    s = str(value).strip().lower()
    if s not in VALID_ELEMENT_CRITERIA:
        logger.warning(
            "Unknown element_criterion %r (from %s) — falling back to 'diff'. "
            "Valid options: %s",
            value,
            source,
            ", ".join(sorted(VALID_ELEMENT_CRITERIA)),
        )
        return "diff"
    return s


VALID_BUDGET_MODES: frozenset[str] = frozenset({"time", "steps"})


def _normalize_budget_mode(value: object, *, source: str) -> str:
    """Lower + validate a budget mode; fall back to "time" if invalid."""
    s = str(value).strip().lower()
    if s not in VALID_BUDGET_MODES:
        logger.warning(
            "Unknown budget_mode %r (from %s) — falling back to 'time'. "
            "Valid options: %s",
            value,
            source,
            ", ".join(sorted(VALID_BUDGET_MODES)),
        )
        return "time"
    return s


def parse_duration(value: object) -> int:
    """Parse a wall-clock duration into whole seconds.

    Accepts an int/float (already seconds), or a string with an optional
    h/m/s suffix (case-insensitive): "2h" -> 7200, "120m" -> 7200,
    "7200s"/"7200" -> 7200. A non-positive or unparsable value falls back to
    7200 (2h) with a warning.
    """
    fallback = 7200
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = int(value)
    else:
        s = str(value).strip().lower()
        multiplier = 1
        if s.endswith("h"):
            multiplier, s = 3600, s[:-1]
        elif s.endswith("m"):
            multiplier, s = 60, s[:-1]
        elif s.endswith("s"):
            multiplier, s = 1, s[:-1]
        try:
            seconds = int(float(s) * multiplier)
        except ValueError:
            logger.warning(
                "Unparsable duration %r — falling back to %ds", value, fallback
            )
            return fallback
    if seconds <= 0:
        logger.warning(
            "Non-positive duration %r — falling back to %ds", value, fallback
        )
        return fallback
    return seconds


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
        ("MC_COLLECTION_DATA_DIR",                   "collection",      "data_dir",                  "str"),
        ("MC_COLLECTION_RUNTIME_DIR",                "collection",      "runtime_dir",               "str"),
        ("MC_COLLECTION_BUDGET_MODE",                "collection",      "budget_mode",               "str"),
        ("MC_COLLECTION_MAX_DURATION",               "collection",      "max_duration",              "str"),
        ("MC_COLLECTION_SIGNAL_TIMEOUT_SEC",         "collection",      "signal_timeout_sec",        "float"),
        ("MC_COLLECTION_POKE_DELAY_SEC",             "collection",      "poke_delay_sec",            "float"),
        ("MC_COLLECTION_MAX_ACTION_REPEATS",         "collection",      "max_action_repeats",        "int"),
        ("MC_COLLECTION_MAX_STEPS_WITHOUT_NEW_PAGE", "collection",      "max_steps_without_new_page", "int"),
        ("MC_LLM_INPUT_MODE",                        "llm",             "input_mode",                "str"),
        ("MC_LLM_ELEMENT_EXTRACTION",                "llm",             "element_extraction",        "bool"),
        ("MC_SCREEN_MATCHING_LUMINANCE_PREFILTER",     "screen_matching", "luminance_prefilter",       "bool"),
        ("MC_SCREEN_MATCHING_LUMINANCE_THRESHOLD",     "screen_matching", "luminance_threshold",       "int"),
        ("MC_SCREEN_MATCHING_SCREENSHOT_DIFF_THRESHOLD", "screen_matching", "screenshot_diff_threshold", "float"),
        ("MC_SCREEN_MATCHING_LUMINANCE_LOW_RES_WIDTH", "screen_matching", "luminance_low_res_width",   "int"),
        ("MC_SCREEN_MATCHING_PERSIST_FILTERED",        "screen_matching", "persist_filtered",          "bool"),
        ("MC_SCREEN_MATCHING_BM25_TOP_K",              "screen_matching", "bm25_top_k",                "int"),
        ("MC_SCREEN_MATCHING_ELEMENT_CRITERION",       "screen_matching", "element_criterion",         "str"),
        ("MC_SCREEN_MATCHING_ELEMENT_DIFF_MAX",        "screen_matching", "element_diff_max",          "int"),
        ("MC_SCREEN_MATCHING_ELEMENT_JACCARD_MIN",     "screen_matching", "element_jaccard_min",       "float"),
        ("MC_SCREEN_MATCHING_PAGE_PIXEL_DIFF_THRESHOLD", "screen_matching", "page_pixel_diff_threshold", "float"),
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

    signal_timeout_sec = float(coll.get("signal_timeout_sec", 12.0))
    if signal_timeout_sec <= 0:
        logger.warning(
            "Non-positive signal_timeout_sec %r — falling back to 12.0",
            signal_timeout_sec,
        )
        signal_timeout_sec = 12.0

    return RunConfig(
        exploration=ExplorationConfig(strategy=strategy),
        collection=CollectionConfig(
            max_steps=int(coll.get("max_steps", 1500)),
            seed=int(coll.get("seed", 42)),
            action_delay_ms=int(coll.get("action_delay_ms", 1500)),
            port=int(coll.get("port", 12345)),
            data_dir=str(coll.get("data_dir", "data")),
            runtime_dir=str(coll.get("runtime_dir", "runtime")),
            budget_mode=_normalize_budget_mode(coll.get("budget_mode", "time"), source="config"),
            max_duration_sec=parse_duration(coll.get("max_duration", "2h")),
            signal_timeout_sec=signal_timeout_sec,
            # No non-positive fallback here (unlike signal_timeout_sec above):
            # a 0-or-negative value is the documented way to DISABLE poking and
            # the D2/D3 guards (tests/experiments), so it is a valid input, not
            # an error.
            poke_delay_sec=float(coll.get("poke_delay_sec", 1.5)),
            max_action_repeats=int(coll.get("max_action_repeats", 8)),
            max_steps_without_new_page=int(coll.get("max_steps_without_new_page", 98)),
        ),
        llm=LlmConfig(
            input_mode=str(llm.get("input_mode", "api")),
            element_extraction=_coerce_bool(llm.get("element_extraction", False)),
        ),
        screen_matching=ScreenMatchingConfig(
            luminance_prefilter=_coerce_bool(sm.get("luminance_prefilter", True)),
            luminance_threshold=int(sm.get("luminance_threshold", 10)),
            screenshot_diff_threshold=float(sm.get("screenshot_diff_threshold", 0.02)),
            luminance_low_res_width=int(sm.get("luminance_low_res_width", 100)),
            persist_filtered=_coerce_bool(sm.get("persist_filtered", True)),
            bm25_top_k=int(sm.get("bm25_top_k", 5)),
            element_criterion=_normalize_criterion(
                sm.get("element_criterion", "diff"), source="config"
            ),
            element_diff_max=int(sm.get("element_diff_max", 5)),
            element_jaccard_min=float(sm.get("element_jaccard_min", 0.5)),
            page_pixel_diff_threshold=float(sm.get("page_pixel_diff_threshold", 0.3)),
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
    duration = getattr(args, "duration", None)
    budget_mode_arg = getattr(args, "budget_mode", None)
    seed = getattr(args, "seed", None)
    delay = getattr(args, "delay", None)
    port = getattr(args, "port", None)
    data_dir = getattr(args, "data_dir", None)
    runtime_dir = getattr(args, "runtime_dir", None)
    if steps is not None:
        coll = replace(coll, max_steps=steps)
    if duration is not None:
        coll = replace(coll, max_duration_sec=parse_duration(duration))
    # Mode resolution (value updates above are independent of mode): explicit
    # --budget-mode always wins; else infer from which single value flag was
    # given; both without an explicit mode keeps the config default + warns.
    if budget_mode_arg is not None:
        coll = replace(
            coll, budget_mode=_normalize_budget_mode(budget_mode_arg, source="--budget-mode")
        )
    elif steps is not None and duration is None:
        coll = replace(coll, budget_mode="steps")
    elif duration is not None and steps is None:
        coll = replace(coll, budget_mode="time")
    elif steps is not None and duration is not None:
        logger.warning(
            "both --steps and --duration given without --budget-mode; "
            "using config budget_mode=%s; pass --budget-mode to disambiguate",
            coll.budget_mode,
        )
    if seed is not None:
        coll = replace(coll, seed=seed)
    if delay is not None:
        coll = replace(coll, action_delay_ms=delay)
    if port is not None:
        coll = replace(coll, port=port)
    if data_dir is not None:
        coll = replace(coll, data_dir=data_dir)
    if runtime_dir is not None:
        coll = replace(coll, runtime_dir=runtime_dir)
    signal_timeout = getattr(args, "signal_timeout", None)
    if signal_timeout is not None:
        coll = replace(coll, signal_timeout_sec=float(signal_timeout))

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
    # luminance prefilter: --luminance-prefilter uses the {on,off} string sentinel
    # (like --element-extraction); the rest are typed scalars.
    lum = getattr(args, "luminance_prefilter", None)
    lt = getattr(args, "luminance_threshold", None)
    sdt = getattr(args, "screenshot_diff_threshold", None)
    lrw = getattr(args, "luminance_low_res_width", None)
    if lum is not None:
        sm = replace(sm, luminance_prefilter=(lum == "on"))
    if lt is not None:
        sm = replace(sm, luminance_threshold=lt)
    if sdt is not None:
        sm = replace(sm, screenshot_diff_threshold=sdt)
    if lrw is not None:
        sm = replace(sm, luminance_low_res_width=lrw)
    # persist_filtered: {on,off} string sentinel (like --luminance-prefilter).
    pf = getattr(args, "persist_filtered", None)
    if pf is not None:
        sm = replace(sm, persist_filtered=(pf == "on"))

    # BM25 unique-page matching knobs.
    btk = getattr(args, "bm25_top_k", None)
    ecrit = getattr(args, "element_criterion", None)
    edm = getattr(args, "element_diff_max", None)
    ejm = getattr(args, "element_jaccard_min", None)
    ppdt = getattr(args, "page_pixel_diff_threshold", None)
    if btk is not None:
        sm = replace(sm, bm25_top_k=btk)
    if ecrit is not None:
        sm = replace(sm, element_criterion=_normalize_criterion(ecrit, source="--element-criterion"))
    if edm is not None:
        sm = replace(sm, element_diff_max=edm)
    if ejm is not None:
        sm = replace(sm, element_jaccard_min=ejm)
    if ppdt is not None:
        sm = replace(sm, page_pixel_diff_threshold=ppdt)

    return RunConfig(exploration=expl, collection=coll, llm=llm, screen_matching=sm)
