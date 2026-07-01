"""Reset: delete collected session data by scope (all / apps)."""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger


def resolve_targets(
    data_dir: str | Path,
    runtime_dir: str | Path,
    all_: bool = False,
    packages: list[str] | None = None,
) -> list[Path]:
    """Return existing directories that match the reset scope, across BOTH
    roots — a full reset must clear ``data/{package}/`` and
    ``runtime/{package}/`` together, or a surviving ``data/`` half would
    immediately rehydrate stale page knowledge into what's supposed to be a
    wiped/fresh session.

    * ``all_=True``   → ``[data_dir, runtime_dir]`` (whichever exist).
    * ``packages``    → ``[data_dir / pkg, runtime_dir / pkg for each existing pkg dir]``.

    Raises ValueError if no scope is given.
    """
    data_dir = Path(data_dir)
    runtime_dir = Path(runtime_dir)

    if all_:
        return [p for p in (data_dir, runtime_dir) if p.exists()]

    if packages:
        return [
            p for pkg in packages
            for p in (data_dir / pkg, runtime_dir / pkg)
            if p.exists()
        ]

    raise ValueError("reset requires a scope: --all or --apps")


def delete_targets(targets: list[Path], dry_run: bool = False) -> int:
    """Delete directories via shutil.rmtree. Return number deleted."""
    deleted = 0
    for path in targets:
        if not path.exists():
            continue
        if dry_run:
            logger.info(f"[dry-run] would delete: {path}")
            continue
        shutil.rmtree(path)
        logger.info(f"Deleted: {path}")
        deleted += 1
    return deleted


__all__ = ["delete_targets", "resolve_targets"]
