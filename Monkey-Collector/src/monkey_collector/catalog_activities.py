"""Static activity ground-truth source from ``catalog/activities.json``.

Produced by ``catalog/extract_activities.py`` (androguard parsing of each
APK's ``AndroidManifest.xml``) and loaded once per process. Used by the
activity coverage tracker so that the denominator (``total_activities``)
is fixed across sessions and devices, instead of varying with ``dumpsys``
output.

Falls back silently to ``None`` when the catalog file is missing, corrupt,
or the package is not registered — callers are expected to handle the
fallback path (typically ``adb.get_declared_activities``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from loguru import logger

# <repo>/src/monkey_collector/catalog_activities.py
#   parents[2] = <repo root>
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "catalog" / "activities.json"

# Class-name prefixes (the part after "/") that are never independently
# navigable by a user — they belong to third-party framework/SDK namespaces
# bundled into the APK, not the app's own code. Excluded from the coverage
# denominator (and from alias resolution) so "% of screens explored" reflects
# screens a monkey could plausibly reach, not vendored plumbing.
#
# Deliberately conservative: only third-party *framework* namespaces are
# listed here. Activities in the app's own namespace are kept even when
# their name suggests low navigability (e.g. permission/version-gate
# screens) because those can still appear in the foreground during a run.
NON_NAVIGABLE_CLASS_PREFIXES = (
    "androidx.car.app.",             # CarAppPermissionActivity (Android Auto)
    "com.android.billingclient.",    # ProxyBillingActivity(+V2)
    "com.google.android.gms.",       # GoogleApiActivity, SignInHubActivity
    "com.google.android.play.core.", # PlayCoreDialogWrapper/MissingSplits
)


def _is_navigable(component: str) -> bool:
    """``True`` unless ``component``'s class (after ``/``) is in the denylist."""
    _, _, class_name = component.partition("/")
    return not class_name.startswith(NON_NAVIGABLE_CLASS_PREFIXES)


class ActivityCatalog:
    """Process-lifetime cache of ``catalog/activities.json``."""

    _instance: ClassVar[ActivityCatalog | None] = None

    def __init__(self, path: Path | None = None) -> None:
        self._data: dict[str, list[str]] = {}
        self._aliases: dict[str, dict[str, str]] = {}
        self._loaded: bool = False
        self._try_load(path or _DEFAULT_PATH)

    @classmethod
    def instance(cls, path: Path | None = None) -> ActivityCatalog:
        if cls._instance is None:
            cls._instance = cls(path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear cached singleton — intended for tests only."""
        cls._instance = None

    def _try_load(self, path: Path) -> None:
        if not path.exists():
            logger.error(
                f"Activity catalog not found at {path}; "
                f"falling back to dumpsys for all packages"
            )
            return
        try:
            with path.open(encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("expected JSON object at top level")
            self._data = {
                pkg: list(entry.get("activities") or [])
                for pkg, entry in raw.items()
                if isinstance(entry, dict)
            }
            # activity-alias → targetActivity maps, added by a later catalog
            # revision. Older catalog files without the key parse to {} for
            # every package (backward-compatible: get_aliases returns an empty
            # map, and coverage resolution degrades to identity).
            self._aliases = {
                pkg: dict(aliases)
                for pkg, entry in raw.items()
                if isinstance(entry, dict)
                for aliases in [entry.get("aliases")]
                if isinstance(aliases, dict)
            }
            self._loaded = True
            logger.info(
                f"Activity catalog loaded: {len(self._data)} packages from {path}"
            )
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.error(
                f"Activity catalog at {path} unreadable ({e}); "
                f"falling back to dumpsys for all packages"
            )

    def is_loaded(self) -> bool:
        return self._loaded

    def get_declared(self, package: str) -> list[str] | None:
        """Return declared activities minus non-navigable framework classes.

        ``None`` on miss (catalog not loaded, or package unregistered).
        """
        if not self._loaded:
            return None
        acts = self._data.get(package)
        if acts is None:
            return None
        return [a for a in acts if _is_navigable(a)]

    def get_aliases(self, package: str) -> dict[str, str] | None:
        """Return a fresh ``alias → targetActivity`` map, or ``None`` on miss.

        ``None`` when the catalog is not loaded or the package is unknown
        (same miss semantics as ``get_declared``). A registered package with
        no aliases (or an older catalog file lacking the ``aliases`` key)
        yields an empty dict. Aliases whose target is non-navigable (see
        ``NON_NAVIGABLE_CLASS_PREFIXES``) are dropped, matching the
        ``get_declared`` filter — an alias never resolves onto a target that
        the denominator itself excludes.
        """
        if not self._loaded:
            return None
        if package not in self._data:
            return None
        return {
            alias: target
            for alias, target in (self._aliases.get(package) or {}).items()
            if _is_navigable(target)
        }
