"""App catalog: parse apps.csv and filter by category/priority/installed."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

_REQUIRED_COLUMNS = (
    "category",
    "sub_category",
    "app_name",
    "package_id",
    "source",
    "priority",
    "notes",
)

# Optional column: present in production apps.csv, absent in older test fixtures.
_INSTALLED_COLUMN = "installed"
_TRUE_VALUES = frozenset({"true", "1", "yes", "y", "t"})


@dataclass(frozen=True)
class AppJob:
    category: str
    sub_category: str
    app_name: str
    package_id: str
    source: str
    priority: str
    notes: str = ""
    installed: bool = False

    @property
    def description(self) -> str:
        """Human-readable one-liner describing this app for LLM context.

        Example: ``Amazon Shopping (Shopping/General) — Top e-commerce; complex
        UI with search/filters/cart``. Falls back to the package id when the
        app name is missing.
        """
        label = self.app_name or self.package_id
        cat = "/".join(p for p in (self.category, self.sub_category) if p)
        text = f"{label} ({cat})" if cat else label
        if self.notes:
            text += f" — {self.notes}"
        return text


def _normalize(value: str) -> str:
    return value.strip().lower()


def _parse_installed(value: str) -> bool:
    return _normalize(value) in _TRUE_VALUES


class AppCatalog:
    def __init__(self, apps: list[AppJob]) -> None:
        self._apps: list[AppJob] = list(apps)

    @classmethod
    def load(cls, csv_path: str | Path) -> AppCatalog:
        """Parse apps.csv using the stdlib csv module (no pandas)."""
        path = Path(csv_path)
        apps: list[AppJob] = []
        # utf-8-sig strips a leading BOM if present.
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Empty CSV: {path}") from exc

            columns = [_normalize(h) for h in header]
            missing = [c for c in _REQUIRED_COLUMNS if c not in columns]
            if missing:
                raise ValueError(
                    f"apps.csv missing required columns: {missing} (got {columns})"
                )
            index = {name: columns.index(name) for name in _REQUIRED_COLUMNS}
            installed_idx = (
                columns.index(_INSTALLED_COLUMN)
                if _INSTALLED_COLUMN in columns
                else None
            )

            for line_no, row in enumerate(reader, start=2):
                if not row or all(not cell.strip() for cell in row):
                    continue
                try:
                    installed = False
                    if installed_idx is not None and len(row) > installed_idx:
                        installed = _parse_installed(row[installed_idx])
                    apps.append(
                        AppJob(
                            category=row[index["category"]].strip(),
                            sub_category=row[index["sub_category"]].strip(),
                            app_name=row[index["app_name"]].strip(),
                            package_id=row[index["package_id"]].strip(),
                            source=row[index["source"]].strip(),
                            priority=row[index["priority"]].strip(),
                            notes=row[index["notes"]].strip() if len(row) > index["notes"] else "",
                            installed=installed,
                        )
                    )
                except IndexError:
                    logger.warning(f"{path}:{line_no} skipped — malformed row: {row!r}")

        logger.info(f"Loaded {len(apps)} apps from {path}")
        return cls(apps)

    def filter(
        self,
        categories: list[str] | None = None,
        priorities: list[str] | None = None,
        installed: bool | None = None,
    ) -> list[AppJob]:
        """Return apps matching the given filters (case-insensitive, whitespace-trimmed).

        ``installed=True`` keeps only apps marked installed; ``False`` keeps only
        uninstalled; ``None`` ignores the installed column.
        """
        cat_set = self._prepare_filter("category", categories, self.categories())
        pri_set = self._prepare_filter("priority", priorities, self.priorities())

        result: list[AppJob] = []
        for app in self._apps:
            if cat_set is not None and _normalize(app.category) not in cat_set:
                continue
            if pri_set is not None and _normalize(app.priority) not in pri_set:
                continue
            if installed is not None and app.installed != installed:
                continue
            result.append(app)
        return result

    def installed_apps(self) -> list[AppJob]:
        """Convenience: apps where installed=True."""
        return self.filter(installed=True)

    def find_by_package(self, package_id: str) -> AppJob | None:
        """Lookup a single app by exact package_id (case-sensitive)."""
        for app in self._apps:
            if app.package_id == package_id:
                return app
        return None

    def categories(self) -> list[str]:
        return sorted({a.category for a in self._apps})

    def priorities(self) -> list[str]:
        return sorted({a.priority for a in self._apps})

    def _prepare_filter(
        self,
        field_name: str,
        requested: list[str] | None,
        known: list[str],
    ) -> set[str] | None:
        if requested is None:
            return None
        normalized = {_normalize(v) for v in requested}
        known_norm = {_normalize(v) for v in known}
        unknown = normalized - known_norm
        if unknown:
            logger.warning(
                f"AppCatalog.filter: unknown {field_name} values ignored: {sorted(unknown)}"
            )
        return normalized


__all__ = ["AppCatalog", "AppJob"]
