"""CLI entrypoint for monkey-collector."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def cmd_run(args: argparse.Namespace) -> None:
    """Run server-driven data collection across one or more installed apps."""
    log_dir = Path(__file__).resolve().parents[2] / "runtime" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(str(log_path), level="DEBUG", enqueue=True)
    logger.info(f"[run] log file: {log_path}")

    from monkey_collector.adb import AdbClient
    from monkey_collector.config import load_run_config, merge_with_cli_args
    from monkey_collector.domain.activity_coverage import ActivityCoverageTracker
    from monkey_collector.domain.cost_tracker import CostTracker
    from monkey_collector.llm import create_element_extractor, create_llm_client
    from monkey_collector.pipeline.collector import Collector
    from monkey_collector.pipeline.exploration import LLMGuidedExplorer
    from monkey_collector.pipeline.screen_matching import create_screen_matcher
    from monkey_collector.pipeline.text_generator import create_text_generator
    from monkey_collector.storage import DataWriter
    from monkey_collector.tcp_server import CollectionServer

    # Resolve config: builtin defaults → run.yaml → MC_* env → CLI flags.
    # --screen-grouping deprecation is honoured by merge_with_cli_args; warn here.
    if getattr(args, "screen_grouping", None) == "off":
        logger.warning(
            "--screen-grouping is deprecated; use --element-extraction. "
            "Treating --screen-grouping off as --element-extraction off."
        )
    cfg = load_run_config(path=getattr(args, "config", None))
    cfg = merge_with_cli_args(cfg, args)
    logger.info(
        f"Config: strategy={cfg.exploration.strategy}, "
        f"budget_mode={cfg.collection.budget_mode}, "
        f"max_duration_sec={cfg.collection.max_duration_sec}, "
        f"max_steps={cfg.collection.max_steps}, seed={cfg.collection.seed}, "
        f"delay_ms={cfg.collection.action_delay_ms}, port={cfg.collection.port}, "
        f"input_mode={cfg.llm.input_mode}, "
        f"element_extraction={cfg.llm.element_extraction}, "
        f"luminance_prefilter={cfg.screen_matching.luminance_prefilter}, "
        f"persist_filtered={cfg.screen_matching.persist_filtered}"
    )

    packages = _resolve_run_packages(args.apps, cfg.collection.runtime_dir, args.force)
    if not packages:
        logger.info(
            "Nothing to collect. All requested apps are already marked "
            "complete (use --force to re-collect) or the apps.csv queue is "
            "empty."
        )
        return
    logger.info(f"Run queue ({len(packages)} app(s)): {packages}")
    app_contexts = _resolve_app_contexts(packages)
    app_names = _resolve_app_names(packages)

    adb = AdbClient()
    activity_tracker = ActivityCoverageTracker()
    cost_tracker = CostTracker()

    element_extraction_on = cfg.llm.element_extraction

    # Single shared OpenRouter client reused by input-text generation and
    # element extraction. Created only when an LLM feature is requested; returns
    # None (→ random text / structural-fingerprint matching) when
    # OPENROUTER_API_KEY is unset.
    llm_client = None
    if cfg.llm.input_mode == "api" or element_extraction_on:
        llm_client = create_llm_client(cost_tracker=cost_tracker)

    text_gen = create_text_generator(
        mode=cfg.llm.input_mode, seed=cfg.collection.seed, llm_client=llm_client,
    )
    # Page identity is decided by the BM25 matcher (LLM-free). With element
    # extraction on, one ElementExtractor additionally enriches a new page's
    # families (exploration same-function grouping). With it off (extractor
    # None), families are empty — see create_screen_matcher.
    extractor = create_element_extractor(llm_client) if element_extraction_on else None
    screen_matcher = create_screen_matcher(
        extractor,
        luminance_prefilter=cfg.screen_matching.luminance_prefilter,
        luminance_threshold=cfg.screen_matching.luminance_threshold,
        screenshot_diff_threshold=cfg.screen_matching.screenshot_diff_threshold,
        luminance_low_res_width=cfg.screen_matching.luminance_low_res_width,
        persist_filtered=cfg.screen_matching.persist_filtered,
        bm25_top_k=cfg.screen_matching.bm25_top_k,
        element_criterion=cfg.screen_matching.element_criterion,
        element_diff_max=cfg.screen_matching.element_diff_max,
        element_jaccard_min=cfg.screen_matching.element_jaccard_min,
        page_pixel_diff_threshold=cfg.screen_matching.page_pixel_diff_threshold,
    )
    explorer = LLMGuidedExplorer(
        adb,
        text_generator=text_gen,
        config={
            "seed": cfg.collection.seed,
            "action_delay_ms": cfg.collection.action_delay_ms,
        },
        strategy=cfg.exploration.strategy,
    )
    server = CollectionServer(host="0.0.0.0", port=cfg.collection.port)
    writer = DataWriter(
        data_dir=cfg.collection.data_dir, runtime_dir=cfg.collection.runtime_dir,
    )
    collector = Collector(
        adb=adb,
        explorer=explorer,
        server=server,
        writer=writer,
        max_steps=cfg.collection.max_steps,
        action_delay=cfg.collection.action_delay_ms / 1000.0,
        xml_timeout=cfg.collection.signal_timeout_sec,
        budget_mode=cfg.collection.budget_mode,
        max_duration_sec=cfg.collection.max_duration_sec,
        max_action_repeats=cfg.collection.max_action_repeats,
        max_steps_without_new_page=cfg.collection.max_steps_without_new_page,
        activity_coverage_tracker=activity_tracker,
        cost_tracker=cost_tracker,
        text_generator=text_gen,
        llm_client=llm_client,
        screen_matcher=screen_matcher,
        new_session=args.new_session,
        app_contexts=app_contexts,
        app_names=app_names,
    )

    session_ids = collector.run_queue(packages)
    logger.info(f"All sessions complete ({len(session_ids)}/{len(packages)})")
    for sid in session_ids:
        logger.info(f"  {cfg.collection.data_dir}/{sid}")


def _resolve_run_packages(
    apps_arg: list[str],
    runtime_dir: str,
    force: bool = False,
) -> list[str]:
    """Translate the ``--apps`` CLI argument into an ordered package list.

    * ``["all"]`` → every app marked ``installed=true`` in ``apps.csv``.
    * ``["com.X", "com.Y"]`` → exactly those package ids (preserves order,
      deduplicates, warns on unknown packages).

    Sessions whose ``{runtime_dir}/{pkg}/metadata.json`` has a non-empty
    ``completed_at`` field are skipped — those apps are treated as done.
    Pass ``force=True`` to include them anyway (useful for re-collection).
    """
    from monkey_collector.pipeline.app_catalog import AppCatalog

    if not apps_arg:
        return []

    if len(apps_arg) == 1 and apps_arg[0].strip().lower() == "all":
        try:
            catalog = AppCatalog.load("catalog/apps.csv")
        except FileNotFoundError:
            logger.error(
                "catalog/apps.csv not found. Run `sync-installed` first or "
                "add the catalog file."
            )
            sys.exit(2)
        jobs = catalog.installed_apps()
        if not jobs:
            logger.warning(
                "catalog/apps.csv has no rows with installed=true. "
                "Run `sync-installed` to refresh it."
            )
        candidates = [j.package_id for j in jobs]
    else:
        try:
            catalog = AppCatalog.load("catalog/apps.csv")
        except FileNotFoundError:
            catalog = None

        seen: set[str] = set()
        candidates = []
        for token in apps_arg:
            pkg = token.strip()
            if not pkg or pkg in seen:
                continue
            if catalog is not None and catalog.find_by_package(pkg) is None:
                logger.warning(
                    f"Package '{pkg}' not listed in apps.csv; session will "
                    f"be saved under the package id as-is."
                )
            seen.add(pkg)
            candidates.append(pkg)

    if force:
        return candidates

    completed = _load_completed_packages(runtime_dir)
    if not completed:
        return candidates

    filtered: list[str] = []
    skipped: list[str] = []
    for pkg in candidates:
        if pkg in completed:
            skipped.append(pkg)
        else:
            filtered.append(pkg)

    if skipped:
        logger.info(
            f"Skipping {len(skipped)} already-completed app(s) "
            f"(use --force to re-collect): {skipped}"
        )
    return filtered


def _resolve_app_contexts(packages: list[str]) -> dict[str, str]:
    """Map each package id to its human-readable app description from apps.csv.

    Used to ground LLM input-text generation in the app's domain. Best-effort:
    a missing catalog yields ``{}`` and packages absent from the catalog are
    simply omitted — the Collector falls back to the package id for those.
    """
    from monkey_collector.pipeline.app_catalog import AppCatalog

    try:
        catalog = AppCatalog.load("catalog/apps.csv")
    except FileNotFoundError:
        return {}

    contexts: dict[str, str] = {}
    for pkg in packages:
        job = catalog.find_by_package(pkg)
        if job is not None:
            contexts[pkg] = job.description
    return contexts


def _resolve_app_names(packages: list[str]) -> dict[str, str]:
    """Map each package id to its human-readable app name from apps.csv.

    Used to label open_app events on external recovery. Best-effort: a missing
    catalog yields ``{}`` and packages absent from the catalog are omitted — the
    open_app event then carries an empty ``app_name`` (downstream can join on
    ``package``).
    """
    from monkey_collector.pipeline.app_catalog import AppCatalog

    try:
        catalog = AppCatalog.load("catalog/apps.csv")
    except FileNotFoundError:
        return {}

    names: dict[str, str] = {}
    for pkg in packages:
        job = catalog.find_by_package(pkg)
        if job is not None and job.app_name:
            names[pkg] = job.app_name
    return names


def _load_completed_packages(runtime_dir: str) -> set[str]:
    """Return package ids whose session is already marked complete.

    Scans ``runtime_dir`` for ``{pkg}/metadata.json`` files and collects every
    package whose metadata has a non-empty ``completed_at`` value.
    """
    import json
    from pathlib import Path

    base = Path(runtime_dir)
    if not base.is_dir():
        return set()
    completed: set[str] = set()
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        meta = sub / "metadata.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("completed_at"):
            completed.add(sub.name)
    return completed


def cmd_reset(args: argparse.Namespace) -> None:
    """Delete collected session data by scope (all / apps)."""
    from monkey_collector.pipeline.reset import delete_targets, resolve_targets

    packages = _split_or_none(args.apps)
    if args.all and packages:
        logger.error("--all is mutually exclusive with --apps")
        sys.exit(2)
    if not args.all and not packages:
        logger.error("reset requires a scope: --all or --apps")
        sys.exit(2)

    targets = resolve_targets(
        data_dir=args.data_dir,
        runtime_dir=args.runtime_dir,
        all_=args.all,
        packages=packages,
    )

    if not targets:
        logger.info("No matching directories found; nothing to delete.")
        return

    logger.info(f"Reset scope resolved to {len(targets)} path(s):")
    for p in targets:
        logger.info(f"  {p}")

    if not args.yes and not args.dry_run:
        reply = input(f"Delete {len(targets)} path(s)? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            logger.info("Aborted.")
            return

    deleted = delete_targets(targets, dry_run=args.dry_run)
    if args.dry_run:
        logger.info(f"[dry-run] Would delete {len(targets)} path(s)")
    else:
        logger.info(f"Reset complete: deleted {deleted} path(s)")


def _split_or_none(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [p.strip() for p in raw.split(",") if p.strip()]
    return items or None


def cmd_sync_installed(args: argparse.Namespace) -> None:
    """Refresh the installed column of apps.csv from the connected device."""
    from monkey_collector.pipeline.installed_sync import sync

    sync(csv_path=args.apps_csv)


def cmd_convert(args: argparse.Namespace) -> None:
    """Convert a single session to JSONL."""
    import os

    from monkey_collector.export.converter import Converter

    data_session_dir = os.path.join(args.data_dir, args.package)
    runtime_session_dir = os.path.join(args.runtime_dir, args.package)
    converter = Converter(
        output_path=args.output,
        images_dir=args.images_dir,
    )
    count = converter.convert_session(data_session_dir, runtime_session_dir, args.label)
    logger.info(f"Generated {count} examples -> {args.output}")


def cmd_page_map(args: argparse.Namespace) -> None:
    """Build page map from a saved session.

    Dispatches on layout: ``pages/`` present → exact rebuild from
    events.jsonl's recorded page_key/observation_num; else a legacy flat
    ``xml/`` dir → the structural/Jaccard offline rebuild; else skipped (no
    migration script for pre-migration sessions).
    """
    import os

    from monkey_collector.domain.page_graph import (
        build_graph_from_new_layout,
        build_graph_from_session,
    )
    from monkey_collector.export.graph_visualizer import visualize_session

    data_session_dir = os.path.join(args.data_dir, args.package)
    runtime_session_dir = os.path.join(args.runtime_dir, args.package)

    if os.path.isdir(os.path.join(data_session_dir, "pages")):
        graph = build_graph_from_new_layout(
            data_session_dir, runtime_session_dir, threshold=args.threshold,
        )
    elif os.path.isdir(os.path.join(data_session_dir, "xml")):
        logger.info(f"{data_session_dir}: legacy flat layout, using structural rebuild")
        graph = build_graph_from_session(data_session_dir, threshold=args.threshold)
    else:
        logger.warning(f"{data_session_dir}: no pages/ or xml/ found, skipping")
        return

    graph.save(os.path.join(data_session_dir, "page_graph.json"))
    html = visualize_session(
        data_session_dir, output_path=args.output, open_browser=not args.no_open,
    )
    logger.info(
        f"Page map: {len(graph.nodes)} pages, "
        f"{len(graph.edges)} transitions"
    )
    if html:
        logger.info(f"Visualization: {html}")


def cmd_page_map_all(args: argparse.Namespace) -> None:
    """Build page maps for all sessions under a data directory."""
    import os

    from monkey_collector.domain.page_graph import (
        build_graph_from_new_layout,
        build_graph_from_session,
    )
    from monkey_collector.export.graph_visualizer import visualize_session

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        return

    total = 0
    for name in sorted(os.listdir(data_dir)):
        data_session_dir = os.path.join(data_dir, name)
        runtime_session_dir = os.path.join(args.runtime_dir, name)

        if os.path.isdir(os.path.join(data_session_dir, "pages")):
            graph = build_graph_from_new_layout(
                data_session_dir, runtime_session_dir, threshold=args.threshold,
            )
        elif os.path.isdir(os.path.join(data_session_dir, "xml")):
            graph = build_graph_from_session(data_session_dir, threshold=args.threshold)
        else:
            continue

        if graph.nodes:
            graph.save(os.path.join(data_session_dir, "page_graph.json"))
            visualize_session(data_session_dir, open_browser=False)
            total += 1
            logger.info(
                f"  {name}: {len(graph.nodes)} pages, "
                f"{len(graph.edges)} transitions"
            )

    logger.info(f"Built page maps for {total} sessions")


def cmd_regenerate(args: argparse.Namespace) -> None:
    """Regenerate all XML variants from raw XML files."""
    from monkey_collector.storage import regenerate_xml_variants

    logger.info(f"Regenerating XML variants under: {args.data_dir}")
    count = regenerate_xml_variants(args.data_dir)
    logger.info(f"Regenerated {count} files total")


def cmd_convert_all(args: argparse.Namespace) -> None:
    """Convert all sessions under a data directory to JSONL."""
    from monkey_collector.export.converter import Converter

    converter = Converter(
        output_path=args.output,
        images_dir=args.images_dir,
    )
    total = converter.convert_all(args.data_dir, args.runtime_dir)
    logger.info(f"Generated {total} total examples -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monkey-Collector: Android GUI data collector"
    )
    sub = parser.add_subparsers(dest="command")

    # run (server-driven single-device, one or more apps)
    p = sub.add_parser(
        "run",
        help="Collect GUI data across one or more apps on a single device",
    )
    p.add_argument(
        "--apps",
        nargs="+",
        required=True,
        metavar="PKG",
        help=(
            "Target apps. Use 'all' to sweep every app with installed=true "
            "in catalog/apps.csv, or pass one or more package ids explicitly "
            "(e.g. --apps com.google.android.deskclock com.google.android.calculator)."
        ),
    )
    # YAML-covered params default to None (sentinel): None means "not set on the
    # CLI", so the value resolves from config/run.yaml → MC_* env → builtin.
    # An explicit CLI flag always wins. See monkey_collector.config.
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to run.yaml (default: config/run.yaml, or MC_CONFIG_PATH)",
    )
    p.add_argument(
        "--strategy",
        choices=["DFS", "BFS", "GREEDY"],
        default=None,
        help="Exploration strategy (DFS | BFS | GREEDY; default from config/run.yaml)",
    )
    p.add_argument("--steps", type=int, default=None, help="Max steps per session")
    p.add_argument(
        "--duration",
        default=None,
        metavar="DURATION",
        help="Max wall-clock time per session, e.g. 2h/120m/7200s/7200 (default from config/run.yaml)",
    )
    p.add_argument(
        "--budget-mode",
        choices=["time", "steps"],
        default=None,
        help=(
            "Session end condition: 'time' (--duration) or 'steps' (--steps). "
            "Inferred from whichever of --steps/--duration is given when omitted; "
            "default from config/run.yaml."
        ),
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--delay", type=int, default=None, help="Action delay in ms")
    p.add_argument(
        "--signal-timeout",
        type=float,
        default=None,
        help=(
            "Seconds to wait for each screenshot/XML signal before a stuck "
            "screen escalates (nudge on timeouts 1-2, force-relaunch on the "
            "3rd); default from config/run.yaml (12s)."
        ),
    )
    p.add_argument("--port", type=int, default=None, help="TCP server port")
    p.add_argument("--data-dir", default=None, help="Durable data root (pages/observations, page_graph)")
    p.add_argument("--runtime-dir", default=None, help="Ephemeral runtime root (metadata, events, cost/coverage)")
    p.add_argument(
        "--input-mode",
        choices=["api", "random"],
        default=None,
        help="Input text generation mode: 'api' (LLM) or 'random' (hardcoded)",
    )
    p.add_argument(
        "--element-extraction",
        choices=["on", "off"],
        default=None,
        help=(
            "LLM element extraction + element-set screen matching. 'on' extracts "
            "each screen's elements (same-function family + representative anchor) "
            "in one call and uses them as page identity, saving "
            "xml/{step}_elements.json (requires OPENROUTER_API_KEY; auto-disabled "
            "to structural-fingerprint matching when no client is available). "
            "'off' uses structural matching only."
        ),
    )
    p.add_argument(
        "--screen-grouping",
        choices=["on", "off"],
        default=None,
        help="Deprecated alias for --element-extraction (off disables it).",
    )
    p.add_argument(
        "--luminance-prefilter",
        choices=["on", "off"],
        default=None,
        help=(
            "Stage-0 luminance prefilter: dedup a near-pixel-identical screen to a "
            "stored page with no LLM call (default on; runs standalone even with "
            "--element-extraction off, keeping page/observation dedup)"
        ),
    )
    p.add_argument(
        "--luminance-threshold",
        type=int,
        default=None,
        help="Per-pixel brightness |ΔY| change threshold, 0–255 (default 10)",
    )
    p.add_argument(
        "--screenshot-diff-threshold",
        type=float,
        default=None,
        help="Changed-pixel fraction below which two renders are the same OBSERVATION (default 0.02)",
    )
    p.add_argument(
        "--luminance-low-res-width",
        type=int,
        default=None,
        help="Downscale width (px) for the luminance fingerprint (default 100)",
    )
    p.add_argument(
        "--persist-filtered",
        choices=["on", "off"],
        default=None,
        help=(
            "Persist a prefilter/dedup revisit as its own fresh observation "
            "(per-visit chain pages/{page_key}/0,1,2,...) instead of skipping the "
            "write (default on; 'off' restores no-write-on-revisit dedup)"
        ),
    )
    p.add_argument(
        "--bm25-top-k",
        type=int,
        default=None,
        help="BM25 candidate pages verified per screen (default 5)",
    )
    p.add_argument(
        "--element-criterion",
        choices=["diff", "jaccard"],
        default=None,
        help=(
            "Element-set same-page criterion: 'diff' (|A△B| < --element-diff-max) "
            "or 'jaccard' (Jaccard > --element-jaccard-min) (default diff)"
        ),
    )
    p.add_argument(
        "--element-diff-max",
        type=int,
        default=None,
        help="Max differing element-lines to still count as the same page (default 5)",
    )
    p.add_argument(
        "--element-jaccard-min",
        type=float,
        default=None,
        help="Min element-line Jaccard to count as the same page ('jaccard' mode, default 0.5)",
    )
    p.add_argument(
        "--page-pixel-diff-threshold",
        type=float,
        default=None,
        help="Changed-pixel fraction below which the pixel gate confirms a page merge (default 0.3)",
    )
    p.add_argument(
        "--new-session",
        action="store_true",
        default=False,
        help=(
            "Delete any existing session for the app and start fresh "
            "(default: continue existing session for same app)"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Re-collect apps even if their sessions are already marked "
            "complete (completed_at set). Default: skip completed apps."
        ),
    )

    # reset (delete collected data)
    p = sub.add_parser(
        "reset",
        help="Delete collected session data by scope (all / apps)",
    )
    p.add_argument("--data-dir", default="data", help="Durable data root directory")
    p.add_argument("--runtime-dir", default="runtime", help="Ephemeral runtime root directory")
    p.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Wipe the entire output root (exclusive with --apps)",
    )
    p.add_argument(
        "--apps",
        default=None,
        help="Comma-separated package IDs to wipe",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print paths that would be deleted without deleting",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Skip interactive confirmation",
    )

    # sync-installed (refresh apps.csv installed column from device)
    p = sub.add_parser(
        "sync-installed",
        help="Update apps.csv 'installed' column from the connected device",
    )
    p.add_argument(
        "--apps-csv", default="catalog/apps.csv", help="Path to apps.csv"
    )

    # convert
    p = sub.add_parser("convert", help="Convert session to JSONL")
    p.add_argument("--data-dir", default="data", help="Durable data root directory")
    p.add_argument("--runtime-dir", default="runtime", help="Ephemeral runtime root directory")
    p.add_argument("--package", required=True, help="Package id (session directory name)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--images-dir", required=True, help="Images output directory")
    p.add_argument("--label", type=int, default=1, help="Session label for image naming")

    # page-map
    p = sub.add_parser("page-map", help="Build page map from session data")
    p.add_argument("--data-dir", default="data", help="Durable data root directory")
    p.add_argument("--runtime-dir", default="runtime", help="Ephemeral runtime root directory")
    p.add_argument("--package", required=True, help="Package id (session directory name)")
    p.add_argument(
        "--threshold", type=float, default=0.85,
        help="XML fingerprint similarity threshold (0.0-1.0, legacy flat-layout sessions only)",
    )
    p.add_argument("--output", default=None, help="Output HTML path")
    p.add_argument("--no-open", action="store_true", help="Do not open browser")

    # page-map-all
    p = sub.add_parser("page-map-all", help="Build page maps for all sessions")
    p.add_argument("--data-dir", default="data", help="Durable data root directory")
    p.add_argument("--runtime-dir", default="runtime", help="Ephemeral runtime root directory")
    p.add_argument(
        "--threshold", type=float, default=0.85,
        help="XML fingerprint similarity threshold (0.0-1.0, legacy flat-layout sessions only)",
    )
    p.add_argument("--no-open", action="store_true", help="Do not open browser")

    # regenerate
    p = sub.add_parser("regenerate", help="Regenerate XML variants from raw XML")
    p.add_argument("--data-dir", default="data", help="Durable data root directory")

    # convert-all
    p = sub.add_parser("convert-all", help="Convert all sessions to JSONL")
    p.add_argument("--data-dir", default="data", help="Durable data root directory")
    p.add_argument("--runtime-dir", default="runtime", help="Ephemeral runtime root directory")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--images-dir", required=True, help="Images output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "reset":
        cmd_reset(args)
    elif args.command == "sync-installed":
        cmd_sync_installed(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "convert-all":
        cmd_convert_all(args)
    elif args.command == "regenerate":
        cmd_regenerate(args)
    elif args.command == "page-map":
        cmd_page_map(args)
    elif args.command == "page-map-all":
        cmd_page_map_all(args)


if __name__ == "__main__":
    main()
