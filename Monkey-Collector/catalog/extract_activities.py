"""Extract declared activities from APKs in catalog/apks/ using androguard.

Parses AndroidManifest.xml from each local APK and joins with apps.csv metadata
to produce a JSON mapping:

    {
      "<package_id>": {
        "app_name": str,
        "category": str,
        "sub_category": str,
        "source": str,
        "activities": ["<pkg>/<full.ClassName>", ...]
      }
    }

Usage:
    python -m catalog.extract_activities
    python -m catalog.extract_activities --output path/to/out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from androguard.core.apk import APK
from loguru import logger

CATALOG_DIR = Path(__file__).resolve().parent
APK_DIR = CATALOG_DIR / "apks"
APPS_CSV = CATALOG_DIR / "apps.csv"
DEFAULT_OUTPUT = CATALOG_DIR / "activities.json"


def silence_androguard() -> None:
    """androguard emits noisy DEBUG/WARNING logs while parsing AXML."""
    logging.getLogger("androguard").setLevel(logging.ERROR)
    try:
        from loguru import logger as loguru_logger

        loguru_logger.disable("androguard")
    except Exception:
        pass


def load_csv_meta(csv_path: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pkg = row["package_id"].strip()
            if not pkg:
                continue
            meta[pkg] = {
                "category": row["category"],
                "sub_category": row["sub_category"],
                "source": row["source"],
                "csv_app_name": row["app_name"],
            }
    return meta


_ANDROID_NS = "{http://schemas.android.com/apk/res/android}"


def _qualify(package: str, cls: str) -> str:
    """Expand a manifest class attribute to a fully-qualified class name.

    ``.Foo`` → ``<pkg>.Foo``; ``Foo`` (no dot) → ``<pkg>.Foo``; an already
    fully-qualified ``a.b.Foo`` is returned unchanged.
    """
    if cls.startswith("."):
        return package + cls
    if "." not in cls:
        return f"{package}.{cls}"
    return cls


def extract_from_apk(apk_path: Path) -> tuple[str, str, list[str], dict[str, str]]:
    a = APK(str(apk_path))
    package = a.get_package()
    app_name = a.get_app_name() or package
    activities = sorted(f"{package}/{act}" for act in a.get_activities())

    # androguard's get_activities() returns only <activity> elements, never
    # <activity-alias>. An alias is a launchable component name that resolves
    # to a real declared activity (its targetActivity) — Calendar's
    # `.AllInOneCalendarActivity` is an alias for the declared
    # `...allinone.AllInOneCalendarActivity`. Without the alias map, a visit to
    # the alias never matches the catalog denominator and coverage stays stuck.
    #
    # Read name/targetActivity from the SAME <activity-alias> element via the
    # lxml manifest tree: calling get_all_attribute_value('activity-alias',
    # 'name') and (...,'targetActivity') separately and zip-pairing them
    # reorders the pairs non-deterministically across runs (observed), so the
    # element-wise walk is the only correct source.
    declared = set(activities)
    aliases: dict[str, str] = {}
    manifest = a.get_android_manifest_xml()
    for el in manifest.findall(".//activity-alias"):
        name = el.get(_ANDROID_NS + "name")
        target = el.get(_ANDROID_NS + "targetActivity")
        if not name or not target:
            continue
        name = _qualify(package, name)
        target = _qualify(package, target)
        if f"{package}/{target}" not in declared:
            logger.warning(
                f"  alias {name} targets {target} not in declared activities; "
                f"skipping"
            )
            continue
        aliases[f"{package}/{name}"] = f"{package}/{target}"

    return package, app_name, activities, aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apk-dir", type=Path, default=APK_DIR)
    args = parser.parse_args()

    silence_androguard()

    if not args.apk_dir.exists():
        logger.error(f"APK directory not found: {args.apk_dir}")
        sys.exit(1)

    csv_meta = load_csv_meta(APPS_CSV)
    apk_files = sorted(args.apk_dir.glob("*.apk"))
    logger.info(f"Found {len(apk_files)} APKs in {args.apk_dir}")

    result: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []

    for apk_path in apk_files:
        try:
            package, app_name, activities, aliases = extract_from_apk(apk_path)
        except Exception as e:
            logger.warning(f"  FAIL {apk_path.name}: {e}")
            failures.append((apk_path.name, str(e)))
            continue

        meta = csv_meta.get(package, {})
        result[package] = {
            "app_name": app_name,
            "category": meta.get("category", ""),
            "sub_category": meta.get("sub_category", ""),
            "source": meta.get("source", ""),
            "activities": activities,
            "aliases": aliases,
        }
        logger.info(f"  {apk_path.name}: {package} ({app_name}) -> {len(activities)} activities")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=True)

    total = sum(len(v["activities"]) for v in result.values())
    logger.info(f"Saved {len(result)} apps / {total} activities to {args.output}")
    if failures:
        logger.warning(f"{len(failures)} APKs failed: {[f[0] for f in failures]}")


if __name__ == "__main__":
    main()
