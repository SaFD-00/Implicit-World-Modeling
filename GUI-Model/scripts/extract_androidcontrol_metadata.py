#!/usr/bin/env python3
"""
Extract AndroidControl per-episode metadata (goal, step_instructions, actions,
primary_app, ...) from GCS TFRecord files into a single JSONL, so it can be
joined with gui-model_stage{1,2}.jsonl by episode_id.

``primary_app`` is recovered from the ``accessibility_trees`` proto
(``AndroidAccessibilityForest``): for each step we pick the foreground
``TYPE_APPLICATION`` window, read its root node's ``package_name``, and keep
the most frequent non-system package across the episode. This covers
episodes that do not begin with an ``open_app`` action, which the earlier
"first open_app.app_name" heuristic left as ``null``. The resulting value is
a package identifier (e.g. ``com.ajnsnewmedia.kitchenstories``), not an app
label.

Screenshots are extracted separately by extract_androidcontrol_images.py.

Usage:
    python scripts/extract_androidcontrol_metadata.py \
        --output data/AndroidControl/episodes_meta.jsonl --verbose

    # Inspect a few episodes first to see which feature keys carry app info:
    python scripts/extract_androidcontrol_metadata.py \
        --output data/AndroidControl/episodes_meta.jsonl \
        --max-episodes 3 --verbose
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
from collections import Counter

from android_env.proto.a11y import android_accessibility_forest_pb2

from extract_androidcontrol_images import (
    GCS_BUCKET,
    GCS_PREFIX,
    gcs_download_to_file,
    gcs_list_objects,
    iter_tfrecord_gzip,
    parse_example,
)

# Per-step binary/huge features we never want to emit to JSONL. Screenshots are
# PNG bytes in the source TFRecord (re-encoded to JPEG by
# extract_androidcontrol_images.py); accessibility_trees is a serialized proto
# that we parse out-of-band to derive ``primary_app`` (see
# ``extract_primary_app_from_trees``).
SKIP_FEATURES = {"screenshots", "accessibility_trees"}

# Canonical JSON key order for each emitted episode. TFRecord feature iteration
# order is not stable across files, so without this pass the output had keys
# interleaved differently on every line. Keys not listed here are appended at
# the end in their original order.
CANONICAL_KEY_ORDER = (
    "episode_id",
    "goal",
    "primary_app",
    "screenshot_widths",
    "screenshot_heights",
    "step_instructions",
    "actions",
)

# Drop any single bytes entry larger than this (defense in depth for stray blobs).
MAX_BYTES_PER_ENTRY = 64 * 1024

# AccessibilityWindowInfo.TYPE_APPLICATION — matches both the Android SDK
# (https://developer.android.com/reference/android/view/accessibility/AccessibilityWindowInfo)
# and android_env's WindowType enum.
WINDOW_TYPE_APPLICATION = 1

# Launcher / system-chrome packages that can appear as the foreground
# application window but do not represent the episode's task app.
SYSTEM_PACKAGES = {
    "com.android.systemui",
    "com.android.launcher",
    "com.android.launcher3",
    "com.google.android.apps.nexuslauncher",
    "com.sec.android.app.launcher",
    "com.miui.home",
    "com.oppo.launcher",
    "com.huawei.android.launcher",
}


def feature_to_jsonable(feat: tuple[str, list]) -> object | None:
    kind, values = feat
    if kind == "int64_list":
        return values[0] if len(values) == 1 else values
    if kind == "bytes_list":
        decoded = []
        for v in values:
            if len(v) > MAX_BYTES_PER_ENTRY:
                decoded.append(f"<{len(v)} bytes omitted>")
            else:
                decoded.append(v.decode("utf-8", errors="replace"))
        return decoded[0] if len(decoded) == 1 else decoded
    return None


def _root_package(window) -> str | None:
    """Return the first non-empty ``package_name`` among the window's nodes.

    Every node inside a single ``TYPE_APPLICATION`` window carries the same
    owning ``package_name``, so taking the first non-empty value is equivalent
    to reading the root node's package.
    """
    for node in window.tree.nodes:
        pkg = (getattr(node, "package_name", "") or "").strip()
        if pkg:
            return pkg
    return None


def _foreground_package(forest, *, allow_system: bool) -> str | None:
    """Pick the foreground ``TYPE_APPLICATION`` window's root ``package_name``.

    Preference order:
      1. ``is_active == True`` application window.
      2. Application window with the greatest ``layer`` (top of the z-order).

    When ``allow_system`` is ``False``, packages in ``SYSTEM_PACKAGES`` are
    skipped so the caller can prefer real task apps over the launcher.
    """
    app_windows = [w for w in forest.windows if w.window_type == WINDOW_TYPE_APPLICATION]

    for w in app_windows:
        if getattr(w, "is_active", False):
            pkg = _root_package(w)
            if pkg and (allow_system or pkg not in SYSTEM_PACKAGES):
                return pkg

    for w in sorted(app_windows, key=lambda x: getattr(x, "layer", 0), reverse=True):
        pkg = _root_package(w)
        if pkg and (allow_system or pkg not in SYSTEM_PACKAGES):
            return pkg

    return None


def extract_primary_app_from_trees(a11y_bytes_list: list[bytes]) -> str | None:
    """Majority-vote the episode's primary app across per-step forests.

    Step 0 often captures the launcher before the first action runs, so a
    single-step read routinely picks ``com.google.android.apps.nexuslauncher``
    instead of the task app. We aggregate non-system foreground packages
    across every step and return the most common one. If every step only ever
    surfaces a system/launcher window, we fall back to the most common
    including system packages.
    """
    forests = []
    for raw in a11y_bytes_list:
        if not raw:
            continue
        try:
            forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
            forest.ParseFromString(raw)
        except Exception:
            continue
        forests.append(forest)

    non_system = Counter()
    for forest in forests:
        pkg = _foreground_package(forest, allow_system=False)
        if pkg:
            non_system[pkg] += 1
    if non_system:
        return non_system.most_common(1)[0][0]

    any_pkg = Counter()
    for forest in forests:
        pkg = _foreground_package(forest, allow_system=True)
        if pkg:
            any_pkg[pkg] += 1
    if any_pkg:
        return any_pkg.most_common(1)[0][0]

    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract AndroidControl per-episode metadata to JSONL."
    )
    ap.add_argument(
        "--output",
        default="data/AndroidControl/episodes_meta.jsonl",
        help="Output JSONL path (default: data/AndroidControl/episodes_meta.jsonl)",
    )
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Limit to N episodes (0 = unlimited)",
    )
    ap.add_argument("--verbose", action="store_true", help="Per-episode logging")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output: {args.output}")
    if args.max_episodes > 0:
        print(f"Max episodes: {args.max_episodes}")

    print("Listing TFRecord files from GCS...")
    try:
        obj_names = gcs_list_objects(GCS_BUCKET, GCS_PREFIX)
    except urllib.error.URLError as e:
        print(f"ERROR: Failed to list GCS objects: {e}", file=sys.stderr)
        sys.exit(1)
    if not obj_names:
        print("ERROR: No TFRecord files found in GCS bucket", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(obj_names)} TFRecord files\n")

    t_start = time.time()
    total_errors = 0
    seen_keys: set[str] = set()
    records: list[dict[str, object]] = []
    done = False

    for file_idx, obj_name in enumerate(obj_names):
        if done:
            break

        file_name = os.path.basename(obj_name)
        print(f"[{file_idx + 1}/{len(obj_names)}] Downloading {file_name} ...")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tfrecord.gz")
        os.close(tmp_fd)

        try:
            gcs_download_to_file(GCS_BUCKET, obj_name, tmp_path)
            size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            print(f"  Downloaded {size_mb:.1f} MB, parsing...")

            file_episodes = 0
            for record_data in iter_tfrecord_gzip(tmp_path):
                try:
                    features = parse_example(record_data)
                except Exception as e:
                    total_errors += 1
                    if args.verbose:
                        print(f"  [WARN] parse failed: {e}")
                    continue

                if "episode_id" not in features:
                    total_errors += 1
                    continue

                a11y_feat = features.get("accessibility_trees")
                if a11y_feat and a11y_feat[0] == "bytes_list":
                    a11y_bytes_list = a11y_feat[1]
                else:
                    a11y_bytes_list = []

                record: dict[str, object] = {}
                for name, feat in features.items():
                    if name in SKIP_FEATURES:
                        continue
                    val = feature_to_jsonable(feat)
                    if val is not None:
                        record[name] = val
                    seen_keys.add(name)

                record["primary_app"] = extract_primary_app_from_trees(a11y_bytes_list)

                records.append(record)
                file_episodes += 1

                if args.verbose:
                    print(
                        f"  episode {record.get('episode_id')}: "
                        f"primary_app={record['primary_app']!r} "
                        f"keys={sorted(record.keys())}"
                    )

                if args.max_episodes > 0 and len(records) >= args.max_episodes:
                    done = True
                    break

            elapsed = time.time() - t_start
            print(
                f"  -> {file_episodes} episodes "
                f"(cumulative: {len(records)} episodes, {elapsed:.0f}s)\n"
            )
        except urllib.error.URLError as e:
            print(f"  [ERROR] Download failed: {e}")
            total_errors += 1
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Sort by episode_id so line order is deterministic across runs, and emit
    # each record with a canonical key order so every line has the same shape.
    records.sort(key=lambda r: r["episode_id"])
    with open(args.output, "w", encoding="utf-8") as fout:
        for record in records:
            ordered = {k: record[k] for k in CANONICAL_KEY_ORDER if k in record}
            for k, v in record.items():
                if k not in ordered:
                    ordered[k] = v
            fout.write(json.dumps(ordered, ensure_ascii=False))
            fout.write("\n")

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"Done! {elapsed:.1f}s elapsed")
    print(f"Episodes written:  {len(records)}")
    print(f"Errors:            {total_errors}")
    print(f"Feature keys seen: {sorted(seen_keys)}")


if __name__ == "__main__":
    main()
