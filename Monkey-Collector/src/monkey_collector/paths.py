"""Canonical sub-roots under the runtime root.

``runtime/`` holds two kinds of ephemeral artifact, and they are kept in
separate sub-directories so neither has to be filtered out of the other::

    runtime/
    ├── apps/{package}/     per-app session bookkeeping (metadata.json, events.jsonl,
    │                       cost.csv, activity_coverage.csv)
    └── logs/run_*.log      per-run loguru sinks

Everything that resolves a *per-app* path goes through :func:`apps_root` — a
single definition point, so a consumer can never write under ``apps/`` while
another reads the bare runtime root. Note the sub-roots are siblings: iterating
``apps_root(runtime_dir)`` yields package dirs only, with no ``logs`` entry to
skip.

``runtime_dir`` itself (config ``collection.runtime_dir``, CLI ``--runtime-dir``)
stays the *root* — it is the parent both sub-roots are derived from, so it must
not be pointed at ``runtime/apps`` directly.
"""

from __future__ import annotations

import os

APPS_SUBDIR = "apps"
LOGS_SUBDIR = "logs"


def apps_root(runtime_dir: str | os.PathLike[str]) -> str:
    """Root holding one directory per collected package: ``{runtime_dir}/apps``."""
    return os.path.join(str(runtime_dir), APPS_SUBDIR)


def app_dir(runtime_dir: str | os.PathLike[str], package: str) -> str:
    """Runtime directory for a single package: ``{runtime_dir}/apps/{package}``."""
    return os.path.join(apps_root(runtime_dir), package)


def logs_root(runtime_dir: str | os.PathLike[str]) -> str:
    """Root holding per-run log files: ``{runtime_dir}/logs``."""
    return os.path.join(str(runtime_dir), LOGS_SUBDIR)


__all__ = ["APPS_SUBDIR", "LOGS_SUBDIR", "app_dir", "apps_root", "logs_root"]
