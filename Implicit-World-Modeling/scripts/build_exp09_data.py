#!/usr/bin/env python3
"""Build the EXP09 Time-Management-domain stage-1 train/eval splits."""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parent))

APP_LIST_FILE = REPO_ROOT / ".claude" / "prompts" / "exp09-app.txt"
STATE_FILE = PROJECT_DIR / "data" / "AndroidControl" / "EXP08_stage1_state.jsonl"
EPISODES_META_FILE = PROJECT_DIR / "data" / "AndroidControl" / "episodes_meta.jsonl"
OUT_DIR = PROJECT_DIR / "data" / "AndroidControl_EXP09"

SEED = 42
N_DOMAIN_APPS = 29
N_SOURCE_DOMAIN = 4346
N_OOD_CHEGAL = 88
N_OOD_DIGIBITES = 92
N_ID_POOL = 4166
N_ID_UNSEEN = 100
N_TRAIN = 4000
N_ID_SEEN = 100
N_UNUSED = 66

# Length filter (train pool only — mirrors build_exp08_data.py's filter_pool_by_length,
# which is applied to the train pool while eval/test pools stay lossless). Same model +
# pinned revision as build_exp08_data.py's DEFAULT_MODEL/DEFAULT_REVISION, so the
# mm-expanded length measurement is identical to EXP08's, not a fresh estimate.
CUTOFF_LEN = 24576
IMG_MAX_PIXELS = 1605632
IMG_MIN_PIXELS = 3136
LENGTH_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LENGTH_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"

OOD_CHEGAL = "com.chegal.alarm"
OOD_DIGIBITES = "com.digibites.calendar"

SAMPLE_ID_RE = re.compile(r"episode_(\d+)_step_(\d+)")
PACKAGE_ROW_RE = re.compile(r"^[A-Za-z0-9_.]+(?:\.\.\.)?$")
SOURCE_IMG_RE = re.compile(r"^myset/images/episode_(\d+)_step_(\d+)\.jpg$")


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


def parse_app_table(path: Path, known_apps: set[str]) -> list[str]:
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
        # The table is column-width-truncated: resolve the prefix against the
        # distinct primary_app values, requiring exactly one match.
        prefix = name[: -len("...")]
        matches = sorted(app for app in known_apps if app.startswith(prefix))
        if len(matches) != 1:
            raise AssertionError(
                f"truncated package {name!r} (prefix {prefix!r}) matched {len(matches)} "
                f"primary_app values: {matches}"
            )
        truncated[name] = matches[0]
        resolved.append(matches[0])

    assert len(resolved) == N_DOMAIN_APPS, f"parsed {len(resolved)} packages, expected {N_DOMAIN_APPS}"
    assert len(set(resolved)) == N_DOMAIN_APPS, "duplicate package names in the app table"
    missing = [app for app in resolved if app not in known_apps]
    assert not missing, f"packages absent from episodes_meta: {missing}"

    print(f"[apps] parsed {len(resolved)} Time Management packages from {path}")
    for name, full in truncated.items():
        print(f"[apps] resolved truncated {name!r} -> {full!r}")
    return resolved


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


def filter_train_by_length(train: list[dict], id_seen: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Drop mm-expanded-length > CUTOFF_LEN records from Train only.

    Mirrors build_exp08_data.py's filter_pool_by_length: the filter is applied to the
    train pool only, never to the eval/test splits (id_unseen/ood_*/unused stay
    lossless — untouched by this function). This is a post-split exclusion, not a
    resplit: Train membership only shrinks (no backfill), so id_unseen/ood/unused keep
    their originally-selected sample_ids exactly. If an excluded sample_id happens to
    also be in id_seen (id_seen was drawn from the pre-filter Train), it is dropped
    from id_seen too, without backfill, rather than silently keeping a training-target
    that the trainer never actually sees.

    An unfiltered over-length sample crashes the training dataloader (Qwen-VL's
    get_rope_index indexes by the pre-cutoff image_grid_thw against cutoff_len-sized
    input_ids), so this is a correctness requirement, not an optional cleanup.
    """
    from filter_long_samples import build_length_fn  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415

    processor = AutoProcessor.from_pretrained(
        LENGTH_MODEL, revision=LENGTH_MODEL_REVISION, trust_remote_code=True
    )
    length_of = build_length_fn(processor, image_max_pixels=IMG_MAX_PIXELS, image_min_pixels=IMG_MIN_PIXELS)
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


def write_jsonl(path: Path, records: list[dict]) -> None:
    bad = sum(
        1
        for rec in records
        if any(not ip.startswith("AndroidControl/") for ip in json.loads(rec["line"])["images"])
    )
    assert not bad, f"{path.name}: {bad} rows with un-remapped images"
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(rec["line"])
            handle.write("\n")
    print(f"[write] {path} ({len(records)} records)")


def main() -> None:
    episode_app = load_episode_apps(EPISODES_META_FILE)
    known_apps = {app for app in episode_app.values() if app is not None}

    domain_apps_ordered = parse_app_table(APP_LIST_FILE, known_apps)
    domain_apps = set(domain_apps_ordered)

    domain_records = collect_domain_records(domain_apps, episode_app)
    assert len(domain_records) == N_SOURCE_DOMAIN, (
        f"Time Mgmt source count is {len(domain_records)}, expected {N_SOURCE_DOMAIN}"
    )

    ood_chegal = sorted((r for r in domain_records if r["app"] == OOD_CHEGAL), key=sort_key)
    ood_digibites = sorted((r for r in domain_records if r["app"] == OOD_DIGIBITES), key=sort_key)
    assert len(ood_chegal) == N_OOD_CHEGAL, f"{OOD_CHEGAL} has {len(ood_chegal)}, expected {N_OOD_CHEGAL}"
    assert len(ood_digibites) == N_OOD_DIGIBITES, (
        f"{OOD_DIGIBITES} has {len(ood_digibites)}, expected {N_OOD_DIGIBITES}"
    )

    id_pool = sorted(
        (r for r in domain_records if r["app"] not in (OOD_CHEGAL, OOD_DIGIBITES)),
        key=sort_key,
    )
    assert len(id_pool) == N_ID_POOL, f"ID pool is {len(id_pool)}, expected {N_ID_POOL}"

    shuffled = list(id_pool)
    random.Random(SEED).shuffle(shuffled)

    id_unseen = shuffled[:N_ID_UNSEEN]
    train = shuffled[N_ID_UNSEEN : N_ID_UNSEEN + N_TRAIN]
    unused = shuffled[N_ID_UNSEEN + N_TRAIN :]

    # ID-Seen: a fresh Random(SEED) sampled over the (episode_id, step)-sorted Train
    # list, so the selection does not depend on the shuffled write order.
    train_sorted = sorted(train, key=sort_key)
    id_seen = random.Random(SEED).sample(train_sorted, N_ID_SEEN)

    assert len(id_unseen) == N_ID_UNSEEN, f"ID-Unseen is {len(id_unseen)}"
    assert len(train) == N_TRAIN, f"Train is {len(train)}"
    assert len(id_seen) == N_ID_SEEN, f"ID-Seen is {len(id_seen)}"
    assert len(unused) == N_UNUSED, f"Unused is {len(unused)}"

    def ids(records: list[dict]) -> set[str]:
        return {r["sample_id"] for r in records}

    train_ids, unseen_ids, seen_ids, unused_ids = ids(train), ids(id_unseen), ids(id_seen), ids(unused)
    chegal_ids, digibites_ids = ids(ood_chegal), ids(ood_digibites)

    assert seen_ids <= train_ids, "ID-Seen is not a subset of Train"
    assert not (unseen_ids & train_ids), "ID-Unseen overlaps Train"
    assert not ((chegal_ids | digibites_ids) & train_ids), "OOD overlaps Train"
    assert not (unseen_ids & unused_ids) and not (train_ids & unused_ids), "Unused overlaps Train/Unseen"
    assert unseen_ids | train_ids | unused_ids == ids(id_pool), "Unseen/Train/Unused do not partition the ID pool"
    assert len(train_ids) == N_TRAIN and len(unseen_ids) == N_ID_UNSEEN, "duplicate sample ids within a split"

    # Post-split length filter: shrinks train (and, only on overlap, id_seen) in place.
    # id_unseen/ood_chegal/ood_digibites/unused are untouched — see filter_train_by_length.
    train, id_seen, length_filter_meta = filter_train_by_length(train, id_seen)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "stage1_train_time_mgmt.jsonl", train)
    write_jsonl(OUT_DIR / "stage1_eval_id_seen_full.jsonl", id_seen)
    write_jsonl(OUT_DIR / "stage1_eval_id_unseen_full.jsonl", id_unseen)
    write_jsonl(OUT_DIR / "stage1_eval_ood_chegal_full.jsonl", ood_chegal)
    write_jsonl(OUT_DIR / "stage1_eval_ood_digibites_full.jsonl", ood_digibites)

    episode_actions = load_episode_actions(EPISODES_META_FILE)
    splits = {
        "train": train,
        "id_seen": id_seen,
        "id_unseen": id_unseen,
        "ood_chegal": ood_chegal,
        "ood_digibites": ood_digibites,
        "unused_id": unused,
    }

    def counted(records: list[dict], key) -> dict:
        return dict(sorted(collections.Counter(key(r) for r in records).items()))

    meta = {
        "seed": SEED,
        "source": "data/AndroidControl/EXP08_stage1_state.jsonl + data/AndroidControl/episodes_meta.jsonl",
        "domain": "Time Management",
        "n_domain_apps": N_DOMAIN_APPS,
        "domain_apps": domain_apps_ordered,
        "ood_apps": [OOD_CHEGAL, OOD_DIGIBITES],
        "counts": {
            "source_domain": len(domain_records),
            "id_pool": len(id_pool),
            "train": len(train),
            "id_seen": len(id_seen),
            "id_unseen": len(id_unseen),
            "ood_chegal": len(ood_chegal),
            "ood_digibites": len(ood_digibites),
            "unused_id": len(unused),
        },
        "_note": (
            "id_unseen is sample-level unseen, not episode-level unseen: an ID-Unseen "
            "sample's episode may still have other steps present in train. The split is "
            "sample-level by design. train/id_seen counts are post-length-filter (see "
            "length_filter below); id_unseen/ood_chegal/ood_digibites/unused counts are "
            "the original frozen-spec split (100/88/92/66), unaffected by the filter."
        ),
        "length_filter": length_filter_meta,
        "image_remap": {
            "applied": True,
            "rule": "myset/images/episode_{N}_step_{M}.jpg -> AndroidControl/images/episode_{N:06d}_step_{M}.jpg",
            "special_case": "myset/images/home.jpg -> AndroidControl/images/home.jpg (0 occurrences in EXP09)",
            "reference": "scripts/build_exp08_data.py:remap_image",
            "media_dir": "data/",
            "_note": (
                "Only the images field is remapped; messages are byte-identical to the "
                "source. sample_ids below keep the SOURCE (unpadded) spelling "
                "'episode_{N}_step_{M}', not the padded on-disk filename, so the split "
                "identity is stable across the remap."
            ),
        },
        "sample_ids": {name: [r["sample_id"] for r in recs] for name, recs in splits.items()},
        "realized_stats_for_observability_only": {
            "app_counts_by_split": {
                name: counted(recs, lambda r: r["app"]) for name, recs in splits.items()
            },
            "action_counts_by_split": {
                name: counted(recs, lambda r: action_type_of(r, episode_actions))
                for name, recs in splits.items()
            },
            "episode_counts_by_split": {
                name: len({r["episode_id"] for r in recs}) for name, recs in splits.items()
            },
        },
    }

    meta_path = OUT_DIR / "exp09_split.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {meta_path}")
    print(f"[done] {json.dumps(meta['counts'])}")


if __name__ == "__main__":
    main()
