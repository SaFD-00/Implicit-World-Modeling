#!/usr/bin/env python3
"""Build EXP09 stage-1 domain-adaptation train/eval splits.

EXP09 studies whether a small LoRA can specialize the Stage-1 world model to a
semantic app domain. Each domain (Time Management, Media, ...) gets the same
treatment: resolve its app list, join to the current EXP08 Stage-1 state
source, carve out two OOD apps held out entirely from training, split the
remaining ID pool (seed 42) into Unseen/Train/(Seen subset), then apply a
length filter and a GPU-divisibility trim to Train.

Adding a new domain: write a `parse_<domain>_app_table()` that returns the
resolved package list from its source table, add one entry to `DOMAINS`, and
rerun this script. Everything downstream of "list of package names" (join,
OOD split, length filter, divisibility trim, token weights, file writing,
dataset_info.json registration) is shared across domains.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS_DIR))

PROPOSALS_DIR = REPO_ROOT / ".claude" / "proposals"
STATE_FILE = PROJECT_DIR / "data" / "AndroidControl" / "EXP08_stage1_state.jsonl"
EPISODES_META_FILE = PROJECT_DIR / "data" / "AndroidControl" / "episodes_meta.jsonl"
OUT_DIR = PROJECT_DIR / "data" / "AndroidControl_EXP09"
DATASET_INFO_FILE = PROJECT_DIR / "configs" / "lf_dataset" / "dataset_info.json"

SEED = 42
N_ID_UNSEEN = 100
N_ID_SEEN = 100

# EXP09 fixes a single global batch (per_device_train_batch_size x world_size x
# gradient_accumulation_steps) across every GPU count/type it supports, so the
# training trajectory (step count, LR schedule) is identical regardless of how
# many GPUs a given run uses (see scripts/run_exp09_stage1.sh). A quarter-epoch
# only lands on an integer optimizer step when N_train is a multiple of
# 4 x GLOBAL_BATCH, so every domain's Train split is trimmed to that multiple.
GLOBAL_BATCH = 32
TRIM_MULTIPLE = 4 * GLOBAL_BATCH  # 128

# Length filter (mirrors build_exp08_data.py's filter_pool_by_length, applied to
# the train pool only — eval/test pools stay lossless). Same model + pinned
# revision as build_exp08_data.py's DEFAULT_MODEL/DEFAULT_REVISION, so the
# mm-expanded length measurement is identical to EXP08's, not a fresh estimate.
CUTOFF_LEN = 24576
IMG_MAX_PIXELS = 1605632
IMG_MIN_PIXELS = 3136
LENGTH_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LENGTH_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"

# Diff-loss token weights — same values build_exp08_data.py passes, so EXP09's
# weighting convention is identical to the lineage it is compared against.
W_ADDED, W_MODIFIED, W_UNCHANGED = 1.0, 1.0, 0.25

SAMPLE_ID_RE = re.compile(r"episode_(\d+)_step_(\d+)")
PACKAGE_ROW_RE = re.compile(r"^[A-Za-z0-9_.]+(?:\.\.\.)?$")
SOURCE_IMG_RE = re.compile(r"^myset/images/episode_(\d+)_step_(\d+)\.jpg$")
_HANGUL_RE = re.compile(r"[가-힣]")


def remap_image(image_path: str) -> str:
    """``myset/images/episode_{N}_step_{M}.jpg`` -> ``AndroidControl/images/episode_{N:06d}_step_{M}.jpg``.

    Mirrors scripts/build_exp08_data.py:remap_image — the source episode number is
    unpadded while the shared image pool is 6-digit padded; step keeps its spelling.
    """
    if image_path == "myset/images/home.jpg":
        return "AndroidControl/images/home.jpg"
    match = SOURCE_IMG_RE.match(image_path)
    if not match:
        raise AssertionError(f"unexpected image path, cannot remap: {image_path!r}")
    return f"AndroidControl/images/episode_{int(match.group(1)):06d}_step_{match.group(2)}.jpg"


def ood_slug(package: str) -> str:
    """OOD display slug — second dot-segment of the package name (com.chegal.alarm -> chegal)."""
    parts = package.split(".")
    assert len(parts) >= 2, f"cannot derive an OOD slug from {package!r}"
    return parts[1]


# ── domain app-table parsers ────────────────────────────────────────────────
# Each returns a resolved, deduplicated list of package names. Source tables
# differ in shape per domain, so parsing is domain-specific; everything after
# "list of package names" is shared (see DOMAINS + build_domain below).


def _resolve_truncated(name: str, known_apps: set[str]) -> str:
    prefix = name[: -len("...")]
    matches = sorted(app for app in known_apps if app.startswith(prefix))
    assert len(matches) == 1, (
        f"truncated package {name!r} (prefix {prefix!r}) matched {len(matches)} "
        f"primary_app values: {matches}"
    )
    return matches[0]


def parse_time_mgmt_app_table(known_apps: set[str]) -> list[str]:
    """Time Management: a flat box-drawing table, one package per row."""
    path = PROPOSALS_DIR / "exp09-app-time-mgmt.md"
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("│"):
            continue
        name = line.strip().strip("│").split("│")[0].strip()
        if not name or name in ("TOTAL", "패키지"):
            continue
        # Guard against box-drawing leftovers silently entering as package names.
        assert PACKAGE_ROW_RE.match(name), f"unexpected package row: {name!r}"
        rows.append(name)

    resolved: list[str] = []
    truncated: dict[str, str] = {}
    for name in rows:
        if not name.endswith("..."):
            resolved.append(name)
            continue
        full = _resolve_truncated(name, known_apps)
        truncated[name] = full
        resolved.append(full)

    assert len(set(resolved)) == len(resolved), "duplicate package names in the app table"
    missing = [app for app in resolved if app not in known_apps]
    assert not missing, f"packages absent from episodes_meta: {missing}"

    print(f"[apps][time_mgmt] parsed {len(resolved)} packages from {path}")
    for name, full in truncated.items():
        print(f"[apps][time_mgmt] resolved truncated {name!r} -> {full!r}")
    return resolved


# Rows that belong to the Media table's summary block (subgroup totals), not to
# any individual package — matched exactly so a real package name never collides.
_MEDIA_SUMMARY_LABELS = {
    "서브그룹", "video_news", "reading_ebooks", "music_audio", "TOTAL",
    "패키지", "에피", "스텝", "평균스텝", "평균",
}


def parse_media_app_table(known_apps: set[str]) -> list[str]:
    """Media: grouped box-drawing tables (video_news/music_audio/reading_ebooks)
    with a leading summary block, occasional "pkg1 / pkg2" cells, and one
    unresolvable aggregate row ("기타 N개"). Per memory ac-exp09-setup-decisions
    (#5), no count in this table is trusted — only package names are sourced
    from it; totals are recomputed from the actual Stage-1 state file.
    """
    path = PROPOSALS_DIR / "exp09-app-media-data.md"
    resolved: list[str] = []
    skipped_aggregate: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("│"):
            continue
        name = line.strip().strip("│").split("│")[0].strip()
        if not name or name in _MEDIA_SUMMARY_LABELS:
            continue
        if _HANGUL_RE.search(name):
            # e.g. "기타 7개" — an aggregate row with no individually named packages.
            skipped_aggregate.append(name)
            continue
        parts = [p.strip() for p in name.split("/")] if "/" in name else [name]
        for part in parts:
            if not part:
                continue
            resolved.append(_resolve_truncated(part, known_apps) if part.endswith("...") else part)

    assert len(set(resolved)) == len(resolved), "duplicate package names in the app table"
    unresolved = [app for app in resolved if app not in known_apps]
    resolved = [app for app in resolved if app in known_apps]

    print(f"[apps][media] parsed {len(resolved)} packages from {path}")
    if skipped_aggregate:
        print(f"[apps][media] skipped unresolvable aggregate rows: {skipped_aggregate}")
    if unresolved:
        print(f"[apps][media] not found in episodes_meta primary_app, dropped: {unresolved}")
    return resolved


# ── domain registry ─────────────────────────────────────────────────────────
# n_domain_apps / n_source_domain / each ood count are lineage guards, not
# tunables — they are the values this script measured against the current
# data/AndroidControl/{episodes_meta,EXP08_stage1_state}.jsonl snapshot. A
# real drift in the source data should fail loudly here, not silently reshape
# the split.
DOMAINS = {
    "time_mgmt": {
        "label": "Time Management",
        "parse_apps": parse_time_mgmt_app_table,
        "n_domain_apps": 29,
        "n_source_domain": 4346,
        "ood": [("com.chegal.alarm", 88), ("com.digibites.calendar", 92)],
        # Frozen-spec choice (exp09-design.md §7.2): out of the 4,066 ID records
        # left after Unseen, Train is a round 4,000 and the remaining 66 are
        # deliberately left unused. This predates and is independent of the
        # length filter / divisibility trim applied below.
        "train_cap": 4000,
    },
    "media": {
        "label": "Media",
        "parse_apps": parse_media_app_table,
        "n_domain_apps": 64,
        "n_source_domain": 4397,
        "ood": [("org.readera", 87), ("com.bsbportal.music", 105)],
        # No round-number cap: Media's Train is the full ID-pool remainder,
        # shrunk only by the length filter and divisibility trim below.
        "train_cap": None,
    },
}


def load_episode_apps(path: Path) -> dict[int, str | None]:
    episode_app: dict[int, str | None] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            episode_app[rec["episode_id"]] = rec.get("primary_app")
    return episode_app


def load_episode_actions(path: Path) -> dict[int, list[str]]:
    episode_actions: dict[int, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            episode_actions[rec["episode_id"]] = rec.get("actions") or []
    return episode_actions


def collect_domain_records(domain_apps: set[str], episode_app: dict[int, str | None]) -> list[dict]:
    records: list[dict] = []
    with STATE_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if not raw:
                continue
            rec = json.loads(raw)
            # Identity is taken from the *source* (unpadded) path so that the split
            # stays stable regardless of the remap applied to the written images.
            match = SAMPLE_ID_RE.search(rec["images"][0])
            assert match, f"no (episode, step) in {rec['images'][0]!r}"
            episode_id = int(match.group(1))
            assert episode_id in episode_app, f"episode {episode_id} absent from episodes_meta"
            app = episode_app[episode_id]
            if app not in domain_apps:
                continue
            rec["images"] = [remap_image(ip) for ip in rec["images"]]
            # sample_id/content_hash travel inside the written record too (EXP08 carries
            # both on train and eval rows), so a prediction file can be audited back to
            # its source sample rather than relying on line order alone.
            gpt = next(m["value"] for m in rec["messages"] if m["from"] == "gpt")
            rec["sample_id"] = match.group(0)
            rec["content_hash"] = hashlib.sha1(gpt.encode("utf-8")).hexdigest()[:12]
            records.append(
                {
                    "sample_id": match.group(0),
                    "episode_id": episode_id,
                    "step": int(match.group(2)),
                    "app": app,
                    "line": json.dumps(rec, ensure_ascii=False),
                }
            )
    return records


def sort_key(rec: dict) -> tuple[int, int]:
    return (rec["episode_id"], rec["step"])


def action_type_of(rec: dict, episode_actions: dict[int, list[str]]) -> str:
    actions = episode_actions.get(rec["episode_id"], [])
    if rec["step"] >= len(actions):
        return "unknown"
    try:
        return json.loads(actions[rec["step"]]).get("action_type", "unknown")
    except (json.JSONDecodeError, AttributeError):
        return "unknown"


def filter_train_by_length(train: list[dict], id_seen: list[dict], length_of) -> tuple[list[dict], list[dict], dict]:
    """Drop mm-expanded-length > CUTOFF_LEN records from Train only.

    Mirrors build_exp08_data.py's filter_pool_by_length: the filter is applied to the
    train pool only, never to the eval/test splits. This is a post-split exclusion, not
    a resplit: Train membership only shrinks (no backfill). If an excluded sample_id
    happens to also be in id_seen (id_seen was drawn from the pre-filter Train), it is
    dropped from id_seen too, without backfill, rather than silently keeping a
    training-target that the trainer never actually sees.

    An unfiltered over-length sample crashes the training dataloader (Qwen-VL's
    get_rope_index indexes by the pre-cutoff image_grid_thw against cutoff_len-sized
    input_ids), so this is a correctness requirement, not an optional cleanup.
    """
    media_dir = PROJECT_DIR / "data"

    kept: list[dict] = []
    dropped: list[dict] = []
    for rec in train:
        length = length_of(json.loads(rec["line"]), media_dir)
        if length is None or length > CUTOFF_LEN:
            dropped.append({**rec, "measured_length": length})
        else:
            kept.append(rec)

    dropped_ids = {r["sample_id"] for r in dropped}
    id_seen_dropped = [r["sample_id"] for r in id_seen if r["sample_id"] in dropped_ids]
    id_seen_final = [r for r in id_seen if r["sample_id"] not in dropped_ids]

    print(
        f"[len-filter] train: {len(train)} -> keep {len(kept)}, drop {len(dropped)} "
        f"(>{CUTOFF_LEN}); id_seen: {len(id_seen)} -> {len(id_seen_final)}"
    )
    for rec in dropped:
        print(f"[len-filter] dropped {rec['sample_id']} (length={rec['measured_length']})")

    filter_meta = {
        "applied_to": "train only (id_unseen/ood_*/unused are unfiltered/lossless)",
        "cutoff_len": CUTOFF_LEN,
        "image_max_pixels": IMG_MAX_PIXELS,
        "image_min_pixels": IMG_MIN_PIXELS,
        "length_fn": "scripts/filter_long_samples.py:build_length_fn (same fn as scripts/build_exp08_data.py)",
        "model": LENGTH_MODEL,
        "revision": LENGTH_MODEL_REVISION,
        "n_train_pre_filter": len(train),
        "n_train_dropped": len(dropped),
        "train_dropped": [
            {"sample_id": r["sample_id"], "measured_length": r["measured_length"]} for r in dropped
        ],
        "id_seen_dropped_for_overlap": id_seen_dropped,
    }
    return kept, id_seen_final, filter_meta


def trim_train_to_multiple(train: list[dict], id_seen: list[dict], multiple: int) -> tuple[list[dict], list[dict], dict]:
    """Drop the tail of Train until len(Train) is a multiple of `multiple`.

    Same principle as filter_train_by_length: train-only, post-split, no backfill,
    id_seen loses any overlap. `train` is still in its seed-42 shuffled order at this
    point (filter_train_by_length preserves order), so "tail" is deterministic without
    an extra sort. This exists purely so GLOBAL_BATCH (see module docstring) divides
    every domain's final Train count evenly across GPU 1/2/4 without changing the
    per-GPU-count training trajectory.
    """
    target = (len(train) // multiple) * multiple
    n_drop = len(train) - target
    if n_drop == 0:
        return train, id_seen, {"multiple": multiple, "n_dropped": 0, "dropped_ids": []}

    kept, dropped = train[:-n_drop], train[-n_drop:]
    dropped_ids = {r["sample_id"] for r in dropped}
    id_seen_final = [r for r in id_seen if r["sample_id"] not in dropped_ids]

    print(f"[divisibility-trim] train: {len(train)} -> {len(kept)} (multiple of {multiple})")
    return kept, id_seen_final, {
        "multiple": multiple,
        "n_dropped": n_drop,
        "dropped_ids": sorted(dropped_ids),
        "id_seen_dropped_for_overlap": [r["sample_id"] for r in id_seen if r["sample_id"] not in {x["sample_id"] for x in id_seen_final}],
    }


def attach_token_weights(train: list[dict]) -> tuple[list[dict], dict]:
    """Return Train records carrying diff-based `token_weights`, plus a meta block.

    Returns copies rather than mutating: id_seen was sampled out of the Train list and
    holds the same dict objects, and it is an eval split that must stay unweighted.

    Train only — EXP08's eval/test JSONLs carry no token_weights either. Without this
    field `use_diff_token_weighted_loss: true` silently degrades to plain CE
    (converter.py reads example["token_weights"]; the loss fn treats None as all-ones).

    `--raw` and `--applied` are the same file: those differ in EXP08 only because its
    masked/dropped formats hide the current state, and the diff must be computed on the
    unmasked original. EXP09 is 100% full/unmasked, so the record *is* its own raw.
    """
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with src.open("w", encoding="utf-8") as handle:
            for rec in train:
                handle.write(rec["line"] + "\n")
        subprocess.run(
            [
                sys.executable, str(SCRIPTS_DIR / "diff_loss" / "build_diff_targets.py"),
                "--raw", str(src), "--applied", str(src), "--output", str(out),
                "--model", LENGTH_MODEL, "--revision", LENGTH_MODEL_REVISION,
                "--w-added", str(W_ADDED), "--w-modified", str(W_MODIFIED),
                "--w-unchanged", str(W_UNCHANGED), "--on-error", "fail",
            ],
            check=True,
        )
        weighted = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            weighted[payload["sample_id"]] = payload

    assert len(weighted) == len(train), f"weight builder returned {len(weighted)} of {len(train)}"
    uniform = 0
    out_train: list[dict] = []
    for rec in train:
        payload = weighted[rec["sample_id"]]
        weights = payload["token_weights"]
        assert weights, f"{rec['sample_id']}: empty token_weights"
        if set(weights) == {1.0}:
            uniform += 1
        out_train.append({**rec, "line": json.dumps(payload, ensure_ascii=False)})
    # A uniform-1.0 row is not necessarily a fallback: if _diff_counts genuinely has no
    # UNCHANGED element, all-1.0 is correct. drop_mismatched_uniform_weights (below)
    # is what tells the two apart and drops the real mismatches.
    if uniform:
        print(f"[weights] {uniform}/{len(train)} rows are uniformly 1.0 (checked for mismatch next)")

    ratios = sorted(
        sum(1 for w in weighted[r["sample_id"]]["token_weights"] if w > W_UNCHANGED)
        / len(weighted[r["sample_id"]]["token_weights"])
        for r in train
    )
    print(f"[weights] {len(train)} rows, {uniform} uniform, up-weighted p50={ratios[len(ratios) // 2]:.3f}")
    return out_train, {
        "applied_to": "train only (matches EXP08: its eval/test JSONLs carry no token_weights)",
        "weight_map": {"ADDED": W_ADDED, "MODIFIED": W_MODIFIED, "UNCHANGED": W_UNCHANGED},
        "builder": "scripts/diff_loss/build_diff_targets.py (--on-error fail)",
        "model": LENGTH_MODEL,
        "revision": LENGTH_MODEL_REVISION,
        "raw_equals_applied": "EXP09 is 100% full/unmasked, so each record is its own raw",
        "n_uniform": uniform,
        "upweighted_ratio_p50": round(ratios[len(ratios) // 2], 4),
    }


def drop_mismatched_uniform_weights(train: list[dict], id_seen: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Drop Train rows where token_weights are uniformly 1.0 despite _diff_counts
    reporting a real UNCHANGED element — a text-attribution mismatch observed in
    scripts/diff_loss/token_weight_builder_v2.py: a long or typography-heavy
    UNCHANGED span (seen in Media's ebook/article content; not in Time-Mgmt's short
    UI labels) fails to locate its token span in the assistant text, and the builder
    silently falls back to weighting the whole sample like a full ADDED/MODIFIED
    rewrite. This is caught here rather than fixed in the shared weight builder,
    which other experiments also depend on.

    A row where every element genuinely is ADDED/MODIFIED (_diff_counts UNCHANGED==0)
    is NOT dropped — uniform 1.0 there is the correct weighting, not a mismatch.

    Same principle as filter_train_by_length / trim_train_to_multiple: train-only,
    post-split, no backfill, id_seen loses any overlap. Must run before
    trim_train_to_multiple so the final Train count still lands on a clean multiple.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for rec in train:
        payload = json.loads(rec["line"])
        weights = payload["token_weights"]
        diff_counts = payload.get("_diff_counts") or {}
        if set(weights) == {1.0} and diff_counts.get("UNCHANGED", 0) > 0:
            dropped.append(rec)
        else:
            kept.append(rec)

    dropped_ids = {r["sample_id"] for r in dropped}
    id_seen_final = [r for r in id_seen if r["sample_id"] not in dropped_ids]

    print(f"[weight-mismatch] train: {len(train)} -> keep {len(kept)}, drop {len(dropped)} (uniform-1.0 vs UNCHANGED>0)")
    return kept, id_seen_final, {
        "reason": "token_weights uniformly 1.0 despite _diff_counts.UNCHANGED > 0 (weight-builder text-attribution mismatch)",
        "n_dropped": len(dropped),
        "dropped_ids": sorted(dropped_ids),
    }


TRAIN_ONLY_FIELDS = ("token_weights", "_diff_counts")


def write_jsonl(path: Path, records: list[dict], *, train: bool) -> None:
    """Write one split. Eval splits drop the train-only diff-loss fields.

    ID-Seen aliases the same record objects as Train (it is a subset of it), so
    attach_token_weights reaches its rows too; EXP08's eval JSONLs carry no
    token_weights, and all eval EXP09 files must have the same shape.
    """
    bad = sum(
        1
        for rec in records
        if any(not ip.startswith("AndroidControl/") for ip in json.loads(rec["line"])["images"])
    )
    assert not bad, f"{path.name}: {bad} rows with un-remapped images"
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            payload = json.loads(rec["line"])
            if train:
                assert payload.get("token_weights"), f"{rec['sample_id']}: train row without token_weights"
            else:
                for field in TRAIN_ONLY_FIELDS:
                    payload.pop(field, None)
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
    print(f"[write] {path} ({len(records)} records)")


def _lf_entry(rel_path: str) -> dict:
    return {
        "file_name": rel_path,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }


def register_dataset_info(info_path: Path, domain_results: list[dict]) -> list[str]:
    """(Re)register every EXP09 dataset key, replacing the old single-domain keys.

    Mirrors scripts/build_exp07_data.py:register_dataset_info — each experiment's data
    builder owns its own dataset_info.json entries (read-modify-write, other
    experiments' entries untouched).
    """
    rel_dir = "../../data/AndroidControl_EXP09"
    entries: dict[str, dict] = {}
    for res in domain_results:
        dk = res["domain_key"]
        entries[f"IWM-AC_EXP09_stage1_train_{dk}"] = _lf_entry(f"{rel_dir}/stage1_train_{dk}.jsonl")
        entries[f"IWM-AC_EXP09_stage1_eval_id_seen_{dk}"] = _lf_entry(f"{rel_dir}/stage1_eval_id_seen_{dk}.jsonl")
        entries[f"IWM-AC_EXP09_stage1_eval_id_unseen_{dk}"] = _lf_entry(f"{rel_dir}/stage1_eval_id_unseen_{dk}.jsonl")
        for slug in res["ood_slugs"]:
            entries[f"IWM-AC_EXP09_stage1_eval_ood_{slug}_{dk}"] = _lf_entry(
                f"{rel_dir}/stage1_eval_ood_{slug}_{dk}.jsonl"
            )

    info = json.loads(info_path.read_text(encoding="utf-8"))
    # Pre-domain-split keys (single implicit Time-Mgmt domain) — fully superseded.
    for stale in (
        "IWM-AC_EXP09_stage1_train",
        "IWM-AC_EXP09_stage1_eval_id_seen",
        "IWM-AC_EXP09_stage1_eval_id_unseen",
        "IWM-AC_EXP09_stage1_eval_ood_chegal",
        "IWM-AC_EXP09_stage1_eval_ood_digibites",
    ):
        info.pop(stale, None)
    info.update(entries)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sorted(entries)


def build_domain(
    domain_key: str,
    cfg: dict,
    episode_app: dict[int, str | None],
    episode_actions: dict[int, list[str]],
    known_apps: set[str],
    length_of,
) -> dict:
    print(f"\n=== domain: {domain_key} ({cfg['label']}) ===")
    domain_apps_ordered = cfg["parse_apps"](known_apps)
    assert len(domain_apps_ordered) == cfg["n_domain_apps"], (
        f"{domain_key}: parsed {len(domain_apps_ordered)} packages, expected {cfg['n_domain_apps']}"
    )
    domain_apps = set(domain_apps_ordered)

    domain_records = collect_domain_records(domain_apps, episode_app)
    assert len(domain_records) == cfg["n_source_domain"], (
        f"{domain_key}: source count is {len(domain_records)}, expected {cfg['n_source_domain']}"
    )

    ood_splits: dict[str, list[dict]] = {}
    ood_ids: set[str] = set()
    for pkg, expected_n in cfg["ood"]:
        slug = ood_slug(pkg)
        recs = sorted((r for r in domain_records if r["app"] == pkg), key=sort_key)
        assert len(recs) == expected_n, f"{domain_key}: {pkg} has {len(recs)}, expected {expected_n}"
        ood_splits[slug] = recs
        ood_ids |= {r["sample_id"] for r in recs}

    id_pool = sorted((r for r in domain_records if r["sample_id"] not in ood_ids), key=sort_key)
    n_id_pool_expected = cfg["n_source_domain"] - sum(n for _, n in cfg["ood"])
    assert len(id_pool) == n_id_pool_expected, f"{domain_key}: ID pool is {len(id_pool)}, expected {n_id_pool_expected}"

    shuffled = list(id_pool)
    random.Random(SEED).shuffle(shuffled)
    id_unseen = shuffled[:N_ID_UNSEEN]
    remainder = shuffled[N_ID_UNSEEN:]

    train_cap = cfg["train_cap"]
    if train_cap is not None:
        train, unused = remainder[:train_cap], remainder[train_cap:]
    else:
        train, unused = remainder, []

    # ID-Seen: a fresh Random(SEED) sampled over the (episode_id, step)-sorted Train
    # list, so the selection does not depend on the shuffled write order.
    train_sorted = sorted(train, key=sort_key)
    id_seen = random.Random(SEED).sample(train_sorted, N_ID_SEEN)

    def ids(records: list[dict]) -> set[str]:
        return {r["sample_id"] for r in records}

    train_ids, unseen_ids, seen_ids, unused_ids = ids(train), ids(id_unseen), ids(id_seen), ids(unused)
    assert seen_ids <= train_ids, f"{domain_key}: ID-Seen is not a subset of Train"
    assert not (unseen_ids & train_ids), f"{domain_key}: ID-Unseen overlaps Train"
    assert not (ood_ids & train_ids), f"{domain_key}: OOD overlaps Train"
    assert not (unseen_ids & unused_ids) and not (train_ids & unused_ids), f"{domain_key}: Unused overlaps Train/Unseen"
    assert unseen_ids | train_ids | unused_ids == ids(id_pool), f"{domain_key}: Unseen/Train/Unused do not partition the ID pool"
    assert len(train_ids) == len(train) and len(unseen_ids) == N_ID_UNSEEN, f"{domain_key}: duplicate sample ids within a split"

    # Post-split reductions, in this order: length filter -> token weights (needed to
    # detect weight-mismatch rows) -> weight-mismatch drop -> GPU-divisibility trim
    # (last, so the final count is guaranteed to land on a clean multiple). All are
    # train-only, post-split, no-backfill (see each function's docstring).
    train, id_seen, length_filter_meta = filter_train_by_length(train, id_seen, length_of)
    train, token_weights_meta = attach_token_weights(train)
    train, id_seen, weight_mismatch_meta = drop_mismatched_uniform_weights(train, id_seen)
    train, id_seen, trim_meta = trim_train_to_multiple(train, id_seen, TRIM_MULTIPLE)
    assert len(train) % TRIM_MULTIPLE == 0, f"{domain_key}: final train {len(train)} not a multiple of {TRIM_MULTIPLE}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / f"stage1_train_{domain_key}.jsonl", train, train=True)
    write_jsonl(OUT_DIR / f"stage1_eval_id_seen_{domain_key}.jsonl", id_seen, train=False)
    write_jsonl(OUT_DIR / f"stage1_eval_id_unseen_{domain_key}.jsonl", id_unseen, train=False)
    for slug, recs in ood_splits.items():
        write_jsonl(OUT_DIR / f"stage1_eval_ood_{slug}_{domain_key}.jsonl", recs, train=False)

    splits_for_stats = {"train": train, "id_seen": id_seen, "id_unseen": id_unseen, "unused_id": unused}
    splits_for_stats.update({f"ood_{slug}": recs for slug, recs in ood_splits.items()})

    def counted(records: list[dict], key) -> dict:
        return dict(sorted(collections.Counter(key(r) for r in records).items()))

    meta = {
        "domain": cfg["label"],
        "domain_key": domain_key,
        "seed": SEED,
        "source": "data/AndroidControl/EXP08_stage1_state.jsonl + data/AndroidControl/episodes_meta.jsonl",
        "n_domain_apps": cfg["n_domain_apps"],
        "domain_apps": domain_apps_ordered,
        "ood_apps": [pkg for pkg, _n in cfg["ood"]],
        "train_cap": train_cap,
        "global_batch": GLOBAL_BATCH,
        "counts": {
            "source_domain": len(domain_records),
            "id_pool": len(id_pool),
            "id_unseen": len(id_unseen),
            "train_precap_or_pool": len(remainder),
            "unused_precap": len(unused),
            "train_final": len(train),
            "id_seen": len(id_seen),
            **{f"ood_{slug}": len(recs) for slug, recs in ood_splits.items()},
        },
        "_note": (
            "id_unseen is sample-level unseen, not episode-level unseen: an ID-Unseen "
            "sample's episode may still have other steps present in train. train/id_seen "
            "counts are post-length-filter, post-weight-mismatch-drop, and "
            "post-divisibility-trim (see length_filter/weight_mismatch/divisibility_trim "
            "below); id_unseen/ood_*/unused_precap are the pre-reduction split, "
            "unaffected by any of the three."
        ),
        "length_filter": length_filter_meta,
        "weight_mismatch": weight_mismatch_meta,
        "divisibility_trim": trim_meta,
        "token_weights": token_weights_meta,
        "image_remap": {
            "applied": True,
            "rule": "myset/images/episode_{N}_step_{M}.jpg -> AndroidControl/images/episode_{N:06d}_step_{M}.jpg",
            "reference": "scripts/build_exp08_data.py:remap_image",
            "media_dir": "data/",
        },
        "sample_ids": {name: [r["sample_id"] for r in recs] for name, recs in splits_for_stats.items()},
        "realized_stats_for_observability_only": {
            "app_counts_by_split": {
                name: counted(recs, lambda r: r["app"]) for name, recs in splits_for_stats.items()
            },
            "action_counts_by_split": {
                name: counted(recs, lambda r: action_type_of(r, episode_actions))
                for name, recs in splits_for_stats.items()
            },
            "episode_counts_by_split": {
                name: len({r["episode_id"] for r in recs}) for name, recs in splits_for_stats.items()
            },
        },
    }
    meta_path = OUT_DIR / f"exp09_split_{domain_key}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {meta_path}")
    print(f"[done][{domain_key}] {json.dumps(meta['counts'])}")

    return {"domain_key": domain_key, "ood_slugs": list(ood_splits), "counts": meta["counts"]}


def main() -> None:
    requested = sys.argv[1:] or list(DOMAINS)
    unknown = [d for d in requested if d not in DOMAINS]
    assert not unknown, f"unknown domain(s) {unknown}, known: {list(DOMAINS)}"

    episode_app = load_episode_apps(EPISODES_META_FILE)
    known_apps = {app for app in episode_app.values() if app is not None}
    episode_actions = load_episode_actions(EPISODES_META_FILE)

    from filter_long_samples import build_length_fn  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415

    processor = AutoProcessor.from_pretrained(
        LENGTH_MODEL, revision=LENGTH_MODEL_REVISION, trust_remote_code=True
    )
    length_of = build_length_fn(processor, image_max_pixels=IMG_MAX_PIXELS, image_min_pixels=IMG_MIN_PIXELS)

    results = [
        build_domain(domain_key, DOMAINS[domain_key], episode_app, episode_actions, known_apps, length_of)
        for domain_key in requested
    ]

    keys = register_dataset_info(DATASET_INFO_FILE, results)
    print(f"\n[register] {DATASET_INFO_FILE}: {len(keys)} keys registered/updated")
    for k in keys:
        print(f"[register]   {k}")


if __name__ == "__main__":
    main()
