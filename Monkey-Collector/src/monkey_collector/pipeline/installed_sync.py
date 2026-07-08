"""Sync the ``installed`` column of apps.csv from a live device.

The server drives collection based on which apps are reported as installed in
apps.csv (see ``AppCatalog.installed_apps()``).  This module provides the CLI
backend for ``python -m monkey_collector sync-installed``: it queries the device via
``adb shell pm list packages`` and rewrites the ``installed`` column in place
(atomically) without touching any other fields.
"""

from __future__ import annotations

import contextlib
import csv
import os
import tempfile
from pathlib import Path

from loguru import logger

from monkey_collector.adb import AdbClient

_INSTALLED_HEADER = "installed"
_PACKAGE_HEADER = "package_id"
_TRUE_VALUES = frozenset({"true", "1", "yes", "y", "t"})


def list_installed_packages(adb: AdbClient) -> set[str]:
    """Query ``pm list packages`` and return the set of installed package ids."""
    output = adb.shell("pm list packages")
    packages: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("package:"):
            continue
        pkg = line[len("package:"):].strip()
        if pkg:
            packages.add(pkg)
    return packages


def _is_true(value: str) -> bool:
    return value.strip().lower() in _TRUE_VALUES


def sync_installed_column(
    csv_path: str | Path,
    device_packages: set[str],
) -> tuple[int, int, int]:
    """Rewrite apps.csv so the ``installed`` column reflects *device_packages*.

    Returns ``(total_rows, installed_count, changed_count)``.

    Atomic: writes to a sibling tempfile then ``os.replace``s over the target,
    so a mid-write interrupt leaves the original file untouched.
    """
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    header = rows[0]
    normalized = [h.strip().lower() for h in header]
    if _PACKAGE_HEADER not in normalized:
        raise ValueError(
            f"apps.csv missing required column '{_PACKAGE_HEADER}' "
            f"(got {normalized})"
        )
    pkg_idx = normalized.index(_PACKAGE_HEADER)

    if _INSTALLED_HEADER in normalized:
        installed_idx = normalized.index(_INSTALLED_HEADER)
    else:
        header.append(_INSTALLED_HEADER)
        installed_idx = len(header) - 1

    changed = 0
    installed_count = 0
    total = 0
    for row in rows[1:]:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) <= pkg_idx:
            continue
        pkg = row[pkg_idx].strip()
        if not pkg:
            continue
        total += 1

        # Pad missing trailing cells so we can index installed_idx safely.
        while len(row) <= installed_idx:
            row.append("")

        old_is_installed = _is_true(row[installed_idx])
        new_is_installed = pkg in device_packages
        if old_is_installed != new_is_installed:
            changed += 1
        row[installed_idx] = "true" if new_is_installed else "false"
        if new_is_installed:
            installed_count += 1

    fd, tmp_path = tempfile.mkstemp(
        prefix=".apps.", suffix=".csv.tmp", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    return total, installed_count, changed


def sync(csv_path: str | Path) -> tuple[int, int, int]:
    """One-shot: query device, update apps.csv, log a summary."""
    adb = AdbClient()
    logger.info("Querying installed packages on connected device...")
    packages = list_installed_packages(adb)
    logger.info(f"Device reports {len(packages)} installed packages")

    total, installed, changed = sync_installed_column(csv_path, packages)
    logger.info(
        f"apps.csv synced: {installed}/{total} marked installed "
        f"({changed} row(s) changed)"
    )
    return total, installed, changed


__all__ = ["list_installed_packages", "sync_installed_column", "sync"]
