"""CLI entrypoint for monkey-collector."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def cmd_run(args: argparse.Namespace) -> None:
    """Run server-driven data collection across one or more installed apps."""
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(str(log_path), level="DEBUG", enqueue=True)
    logger.info(f"[run] log file: {log_path}")

    from monkey_collector.adb import AdbClient
    from monkey_collector.domain.activity_coverage import ActivityCoverageTracker
    from monkey_collector.domain.cost_tracker import CostTracker
    from monkey_collector.llm import create_element_extractor, create_llm_client
    from monkey_collector.pipeline.collector import Collector
    from monkey_collector.pipeline.exploration import LLMGuidedExplorer
    from monkey_collector.pipeline.screen_matching import create_screen_matcher
    from monkey_collector.pipeline.text_generator import create_text_generator
    from monkey_collector.storage import DataWriter
    from monkey_collector.tcp_server import CollectionServer

    packages = _resolve_run_packages(args.apps, args.output, args.force)
    if not packages:
        logger.info(
            "Nothing to collect. All requested apps are already marked "
            "complete (use --force to re-collect) or the apps.csv queue is "
            "empty."
        )
        return
    logger.info(f"Run queue ({len(packages)} app(s)): {packages}")
    app_contexts = _resolve_app_contexts(packages)

    adb = AdbClient()
    activity_tracker = ActivityCoverageTracker()
    cost_tracker = CostTracker()

    # --screen-grouping is a deprecated alias for --element-extraction.
    element_extraction_on = args.element_extraction == "on"
    if getattr(args, "screen_grouping", None) == "off":
        logger.warning(
            "--screen-grouping is deprecated; use --element-extraction. "
            "Treating --screen-grouping off as --element-extraction off."
        )
        element_extraction_on = False

    # Single shared OpenRouter client reused by input-text generation and
    # element extraction. Created only when an LLM feature is requested; returns
    # None (→ random text / structural-fingerprint matching) when
    # OPENROUTER_API_KEY is unset.
    llm_client = None
    if args.input_mode == "api" or element_extraction_on:
        llm_client = create_llm_client(cost_tracker=cost_tracker)

    text_gen = create_text_generator(
        mode=args.input_mode, seed=args.seed, llm_client=llm_client,
    )
    # One ElementExtractor feeds the ScreenMatcher, which the loop queries once
    # per new screen (plus expand passes) for element-set page identity.
    extractor = create_element_extractor(llm_client) if element_extraction_on else None
    screen_matcher = create_screen_matcher(
        extractor,
        enabled=element_extraction_on,
        cluster_merge_tolerance=args.cluster_merge_tolerance,
        max_expand_iters=args.max_expand_iters,
    )
    explorer = LLMGuidedExplorer(
        adb,
        text_generator=text_gen,
        config={
            "seed": args.seed,
            "action_delay_ms": args.delay,
        },
    )
    server = CollectionServer(host="0.0.0.0", port=args.port)
    writer = DataWriter(base_dir=args.output)
    collector = Collector(
        adb=adb,
        explorer=explorer,
        server=server,
        writer=writer,
        max_steps=args.steps,
        action_delay=args.delay / 1000.0,
        activity_coverage_tracker=activity_tracker,
        cost_tracker=cost_tracker,
        text_generator=text_gen,
        llm_client=llm_client,
        screen_matcher=screen_matcher,
        new_session=args.new_session,
        app_contexts=app_contexts,
    )

    session_ids = collector.run_queue(packages)
    logger.info(f"All sessions complete ({len(session_ids)}/{len(packages)})")
    for sid in session_ids:
        logger.info(f"  {args.output}/{sid}")


def _resolve_run_packages(
    apps_arg: list[str],
    output_dir: str,
    force: bool = False,
) -> list[str]:
    """Translate the ``--apps`` CLI argument into an ordered package list.

    * ``["all"]`` → every app marked ``installed=true`` in ``apps.csv``.
    * ``["com.X", "com.Y"]`` → exactly those package ids (preserves order,
      deduplicates, warns on unknown packages).

    Sessions whose ``{output_dir}/{pkg}/metadata.json`` has a non-empty
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

    completed = _load_completed_packages(output_dir)
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


def _load_completed_packages(output_dir: str) -> set[str]:
    """Return package ids whose session is already marked complete.

    Scans ``output_dir`` for ``{pkg}/metadata.json`` files and collects every
    package whose metadata has a non-empty ``completed_at`` value.
    """
    import json
    from pathlib import Path

    base = Path(output_dir)
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
        output_dir=args.output,
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
    from monkey_collector.export.converter import Converter

    converter = Converter(
        output_path=args.output,
        images_dir=args.images_dir,
    )
    count = converter.convert_session(args.session, args.label)
    logger.info(f"Generated {count} examples -> {args.output}")


def cmd_page_map(args: argparse.Namespace) -> None:
    """Build page map from a saved session."""
    import os

    from monkey_collector.domain.page_graph import build_graph_from_session
    from monkey_collector.export.graph_visualizer import visualize_session

    graph = build_graph_from_session(args.session, threshold=args.threshold)
    graph.save(os.path.join(args.session, "page_graph.json"))
    html = visualize_session(
        args.session, output_path=args.output, open_browser=not args.no_open,
    )
    logger.info(
        f"Page map: {len(graph.nodes)} pages, "
        f"{len(graph.edges)} transitions"
    )
    if html:
        logger.info(f"Visualization: {html}")


def cmd_page_map_all(args: argparse.Namespace) -> None:
    """Build page maps for all sessions in a directory."""
    import os

    from monkey_collector.domain.page_graph import build_graph_from_session
    from monkey_collector.export.graph_visualizer import visualize_session

    raw_dir = args.raw_dir
    if not os.path.isdir(raw_dir):
        logger.error(f"Directory not found: {raw_dir}")
        return

    total = 0
    for name in sorted(os.listdir(raw_dir)):
        session_dir = os.path.join(raw_dir, name)
        xml_dir = os.path.join(session_dir, "xml")
        if not os.path.isdir(xml_dir):
            continue
        graph = build_graph_from_session(session_dir, threshold=args.threshold)
        if graph.nodes:
            graph.save(os.path.join(session_dir, "page_graph.json"))
            visualize_session(session_dir, open_browser=False)
            total += 1
            logger.info(
                f"  {name}: {len(graph.nodes)} pages, "
                f"{len(graph.edges)} transitions"
            )

    logger.info(f"Built page maps for {total} sessions")


def cmd_regenerate(args: argparse.Namespace) -> None:
    """Regenerate all XML variants from raw XML files."""
    from monkey_collector.storage import regenerate_xml_variants

    logger.info(f"Regenerating XML variants under: {args.raw_dir}")
    count = regenerate_xml_variants(args.raw_dir)
    logger.info(f"Regenerated {count} files total")


def cmd_convert_all(args: argparse.Namespace) -> None:
    """Convert all sessions in a directory to JSONL."""
    from monkey_collector.export.converter import Converter

    converter = Converter(
        output_path=args.output,
        images_dir=args.images_dir,
    )
    total = converter.convert_all(args.raw_dir)
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
    p.add_argument("--steps", type=int, default=100, help="Max steps per session")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--delay", type=int, default=1500, help="Action delay in ms")
    p.add_argument("--port", type=int, default=12345, help="TCP server port")
    p.add_argument("--output", default="data/raw", help="Output directory")
    p.add_argument(
        "--input-mode",
        choices=["api", "random"],
        default="api",
        help="Input text generation mode: 'api' (LLM) or 'random' (hardcoded)",
    )
    p.add_argument(
        "--element-extraction",
        choices=["on", "off"],
        default="on",
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
        "--cluster-merge-tolerance",
        type=float,
        default=0.2,
        help="Two-sided tolerance band for OVERLAP element-set merges (default 0.2)",
    )
    p.add_argument(
        "--max-expand-iters",
        type=int,
        default=3,
        help="Max expand (re-extract on leftover UI) iterations per screen (default 3)",
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
    p.add_argument("--output", default="data/raw", help="Data root directory")
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
    p.add_argument("--session", required=True, help="Session directory path")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--images-dir", required=True, help="Images output directory")
    p.add_argument("--label", type=int, default=1, help="Session label for image naming")

    # page-map
    p = sub.add_parser("page-map", help="Build page map from session data")
    p.add_argument("--session", required=True, help="Session directory path")
    p.add_argument(
        "--threshold", type=float, default=0.85,
        help="XML fingerprint similarity threshold (0.0-1.0)",
    )
    p.add_argument("--output", default=None, help="Output HTML path")
    p.add_argument("--no-open", action="store_true", help="Do not open browser")

    # page-map-all
    p = sub.add_parser("page-map-all", help="Build page maps for all sessions")
    p.add_argument("--raw-dir", default="data/raw", help="Raw sessions directory")
    p.add_argument(
        "--threshold", type=float, default=0.85,
        help="XML fingerprint similarity threshold (0.0-1.0)",
    )
    p.add_argument("--no-open", action="store_true", help="Do not open browser")

    # regenerate
    p = sub.add_parser("regenerate", help="Regenerate XML variants from raw XML")
    p.add_argument("--raw-dir", default="data/raw", help="Raw sessions directory")

    # convert-all
    p = sub.add_parser("convert-all", help="Convert all sessions to JSONL")
    p.add_argument("--raw-dir", default="data/raw", help="Raw sessions directory")
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
