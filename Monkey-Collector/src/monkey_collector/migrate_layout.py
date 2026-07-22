"""One-off migration: legacy storage layout → the current one.

A session collected before the bare-name and runtime-split refactors looks
like::

    data/{pkg}/pages/page_0/0000/...          runtime/{pkg}/events.jsonl
    data/{pkg}/pages/page_0/page.json         logs/run_*.log        (repo root)

This module converts it to the current layout::

    data/{pkg}/pages/0/0/...                  runtime/apps/{pkg}/events.jsonl
    data/{pkg}/pages/0/page.json              runtime/logs/run_*.log

Three kinds of change:

* **Directory renames** — ``pages/page_{N}/`` → ``pages/{N}/`` and each
  zero-padded observation dir ``{obs:04d}/`` → ``{obs}/``.
* **Runtime regrouping** — package dirs sitting directly under ``runtime/``
  move into ``runtime/apps/``, making ``apps/`` and ``logs/`` siblings.
* **Embedded ``page_key`` strings** — the ``"page_{N}"`` identifier stored in
  ``page.json``, ``elements.json``, ``page_graph.json`` (``nodes[].page_key``),
  and ``events.jsonl`` is rewritten to ``"{N}"``. ``page_graph.json`` edges use
  integer node ids (``from_page``/``to_page``), so they need no change.

Design: **dry-run first** (``apply=False`` reports what would change without
touching disk) and **idempotent** (already-bare names/keys, packages already
under ``runtime/apps/``, and files already under ``runtime/logs/`` are left
untouched, so re-running is safe). These are
gitignored, freely-regenerated collection artifacts — a fresh run after the
refactor already writes the new layout, so this is only for pre-refactor
sessions you want to keep.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil

from loguru import logger

from monkey_collector.paths import APPS_SUBDIR, LOGS_SUBDIR, apps_root, logs_root

_PAGE_RE = re.compile(r"^page_(\d+)$")


def _atomic_write(path: str, text: str) -> None:
    """Overwrite *path* atomically: fully materialise a sibling temp file (flush
    + fsync), then ``os.replace`` it over the original. An interrupted or failed
    write (Ctrl-C / crash / disk-full) can therefore never truncate or
    half-overwrite the only copy — the original stays intact until the complete
    new content is in place. This is what makes the content rewrites re-runnable."""
    tmp = f"{path}.migrate.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def strip_page_key(page_key: str) -> str:
    """``"page_7"`` → ``"7"``; anything else (already bare, or not a page key)
    is returned unchanged, making every rewrite idempotent."""
    m = _PAGE_RE.match(page_key)
    return m.group(1) if m else page_key


def _canonical_obs_name(name: str) -> str | None:
    """Canonical bare name for an observation dir, or ``None`` if *name* is not
    an integer dir (leave non-observation entries alone)."""
    return str(int(name)) if name.isdigit() else None


def _rewrite_json_page_key(path: str, apply: bool) -> bool:
    """Rewrite a top-level ``page_key`` field in a JSON object file. Returns
    whether a change was (or would be) made."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"migrate: unreadable JSON {path}, skipping ({e})")
        return False
    if not isinstance(data, dict):
        return False
    old = data.get("page_key")
    if not isinstance(old, str):
        return False
    new = strip_page_key(old)
    if new == old:
        return False
    if apply:
        data["page_key"] = new
        _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False))
    return True


def _rewrite_graph_nodes(path: str, apply: bool) -> int:
    """Rewrite ``nodes[].page_key`` in a ``page_graph.json``. Returns the number
    of node keys changed (edges use integer ids and are left untouched)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"migrate: unreadable graph {path}, skipping ({e})")
        return 0
    changed = 0
    for node in data.get("nodes", []) if isinstance(data, dict) else []:
        if not isinstance(node, dict):
            continue
        old = node.get("page_key")
        if isinstance(old, str) and strip_page_key(old) != old:
            changed += 1
            if apply:
                node["page_key"] = strip_page_key(old)
    if changed and apply:
        _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False))
    return changed


def _rewrite_events(path: str, apply: bool) -> int:
    """Rewrite each JSONL event's ``page_key`` in place. Returns the number of
    lines changed. Malformed lines are copied through untouched."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        logger.warning(f"migrate: unreadable events {path}, skipping ({e})")
        return 0
    out: list[str] = []
    changed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            out.append(line)
            continue
        old = event.get("page_key")
        if isinstance(old, str) and strip_page_key(old) != old:
            event["page_key"] = strip_page_key(old)
            changed += 1
            out.append(json.dumps(event, ensure_ascii=False) + "\n")
        else:
            out.append(line)
    if changed and apply:
        _atomic_write(path, "".join(out))
    return changed


def _rename(src: str, dst: str, apply: bool, report: list[str]) -> bool:
    """Rename *src*→*dst*, refusing to clobber an existing *dst* (a partial
    prior migration). Returns whether the rename happened / would happen."""
    if src == dst:
        return False
    if os.path.exists(dst):
        logger.warning(f"migrate: target exists, skipping rename {src} -> {dst}")
        return False
    report.append(f"rename {src} -> {dst}")
    if apply:
        os.rename(src, dst)
    return True


def migrate_data_dir(data_dir: str, apply: bool, report: list[str]) -> dict[str, int]:
    """Rename page/observation dirs and rewrite embedded page_key JSON under
    every package in *data_dir*."""
    counts = {"pages": 0, "observations": 0, "page_json": 0, "elements_json": 0, "graphs": 0}
    if not os.path.isdir(data_dir):
        return counts

    for pkg in sorted(os.listdir(data_dir)):
        pkg_dir = os.path.join(data_dir, pkg)
        pages_dir = os.path.join(pkg_dir, "pages")
        if os.path.isdir(pages_dir):
            # Rename obs dirs and rewrite JSON while the page dir still has its
            # original name, then rename the page dir itself last.
            for page_name in sorted(os.listdir(pages_dir)):
                page_path = os.path.join(pages_dir, page_name)
                if not os.path.isdir(page_path):
                    continue
                m = _PAGE_RE.match(page_name)
                canonical_page = m.group(1) if m else (page_name if page_name.isdigit() else None)
                if canonical_page is None:
                    continue  # not a page dir we recognise

                for obs_name in sorted(os.listdir(page_path)):
                    obs_path = os.path.join(page_path, obs_name)
                    if not os.path.isdir(obs_path):
                        continue
                    canonical_obs = _canonical_obs_name(obs_name)
                    if canonical_obs is None:
                        continue
                    elements_path = os.path.join(obs_path, "elements.json")
                    if os.path.isfile(elements_path) and _rewrite_json_page_key(elements_path, apply):
                        counts["elements_json"] += 1
                    if _rename(obs_path, os.path.join(page_path, canonical_obs), apply, report):
                        counts["observations"] += 1

                page_json = os.path.join(page_path, "page.json")
                if os.path.isfile(page_json) and _rewrite_json_page_key(page_json, apply):
                    counts["page_json"] += 1

                if _rename(page_path, os.path.join(pages_dir, canonical_page), apply, report):
                    counts["pages"] += 1

        graph_path = os.path.join(pkg_dir, "page_graph.json")
        if os.path.isfile(graph_path):
            n = _rewrite_graph_nodes(graph_path, apply)
            if n:
                counts["graphs"] += 1

    return counts


def migrate_runtime_apps(runtime_dir: str, apply: bool, report: list[str]) -> int:
    """Move each package dir from the runtime root into ``runtime/apps/``.

    Legacy layout put package dirs and ``logs/`` as siblings directly under
    ``runtime/``; the current layout groups packages under ``apps/``. Only
    package directories move: the ``apps`` and ``logs`` sub-roots are skipped
    explicitly (moving ``logs`` would bury the run logs, and moving ``apps``
    would nest it inside itself), as are loose files. Already-migrated trees
    have nothing left at the root to move, so re-running is a no-op.

    Returns the number of package directories moved.
    """
    if not os.path.isdir(runtime_dir):
        return 0
    dest_root = apps_root(runtime_dir)
    moved = 0
    for name in sorted(os.listdir(runtime_dir)):
        if name in (APPS_SUBDIR, LOGS_SUBDIR):
            continue
        src = os.path.join(runtime_dir, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(dest_root, name)
        if os.path.exists(dst):
            logger.warning(f"migrate: app target exists, skipping {src} -> {dst}")
            continue
        report.append(f"move {src} -> {dst}")
        if apply:
            os.makedirs(dest_root, exist_ok=True)
            shutil.move(src, dst)
        moved += 1
    return moved


def migrate_runtime_dir(runtime_dir: str, apply: bool) -> int:
    """Rewrite ``page_key`` in every package's ``events.jsonl``. Returns the
    total number of event lines changed.

    Looks under ``{runtime_dir}/apps/`` *and* at package dirs still sitting at
    the runtime root: on a dry run :func:`migrate_runtime_apps` has not moved
    anything yet, so scanning only ``apps/`` would report zero events for a
    legacy tree. ``apps``/``logs`` are excluded from the root scan, and a
    package can only be found in one place, so nothing is counted twice.
    """
    total = 0
    if not os.path.isdir(runtime_dir):
        return 0
    apps_dir = apps_root(runtime_dir)
    roots = [apps_dir] if os.path.isdir(apps_dir) else []
    roots.append(runtime_dir)
    for root in roots:
        for pkg in sorted(os.listdir(root)):
            if root == runtime_dir and pkg in (APPS_SUBDIR, LOGS_SUBDIR):
                continue
            events_path = os.path.join(root, pkg, "events.jsonl")
            if os.path.isfile(events_path):
                total += _rewrite_events(events_path, apply)
    return total


def migrate_logs(logs_dir: str, runtime_dir: str, apply: bool, report: list[str]) -> int:
    """Move ``logs_dir/*.log`` under ``runtime_dir/logs/``. Returns the number
    of log files moved."""
    if not os.path.isdir(logs_dir):
        return 0
    dest_dir = logs_root(runtime_dir)
    moved = 0
    for name in sorted(os.listdir(logs_dir)):
        src = os.path.join(logs_dir, name)
        if not (os.path.isfile(src) and name.endswith(".log")):
            continue
        dst = os.path.join(dest_dir, name)
        if os.path.exists(dst):
            logger.warning(f"migrate: log target exists, skipping {src} -> {dst}")
            continue
        report.append(f"move {src} -> {dst}")
        if apply:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(src, dst)
        moved += 1
    return moved


def migrate(
    data_dir: str = "data/raw",
    runtime_dir: str = "runtime",
    logs_dir: str = "logs",
    apply: bool = False,
) -> dict:
    """Run the full layout migration. Returns a report dict of counts and the
    concrete rename/move operations (for logging or assertions in tests)."""
    report: list[str] = []
    counts = migrate_data_dir(data_dir, apply, report)
    # Move packages under apps/ before rewriting their events — the rewrite
    # scans apps/ first, so doing it in this order keeps the applied run from
    # walking the same package twice.
    counts["apps"] = migrate_runtime_apps(runtime_dir, apply, report)
    counts["events"] = migrate_runtime_dir(runtime_dir, apply)
    counts["logs"] = migrate_logs(logs_dir, runtime_dir, apply, report)
    return {"applied": apply, "counts": counts, "operations": report}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy storage layout (page_/zero-padded names, flat runtime root) to the current one."
    )
    parser.add_argument("--data-dir", default="data/raw", help="Durable raw-collection root (default: data/raw)")
    parser.add_argument("--runtime-dir", default="runtime", help="Runtime root holding apps/ and logs/ (default: runtime)")
    parser.add_argument("--logs-dir", default="logs", help="Repo-level logs dir to relocate (default: logs)")
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually perform the migration (default: dry-run, report only)",
    )
    args = parser.parse_args(argv)

    result = migrate(args.data_dir, args.runtime_dir, args.logs_dir, apply=args.apply)
    mode = "APPLIED" if result["applied"] else "DRY-RUN (use --apply to perform)"
    logger.info(f"[migrate-layout] {mode}")
    for op in result["operations"]:
        logger.info(f"  {op}")
    c = result["counts"]
    logger.info(
        f"[migrate-layout] pages={c['pages']} observations={c['observations']} "
        f"page.json={c['page_json']} elements.json={c['elements_json']} "
        f"graphs={c['graphs']} apps={c['apps']} events={c['events']} logs={c['logs']}"
    )


if __name__ == "__main__":
    main()
