#!/usr/bin/env python3
"""Train / Test split builder for GUI-Model datasets.

학습 대상 DS 는 {AC (AndroidControl), AC_2 (AndroidControl_2), MC (MonkeyCollection)} 를 지원한다.
AC_2 는 AC 와 동일한 episodes_meta 기반 app-level split 규칙을 따른다.
MobiBench(MB) 는 평가 전용 벤치마크이므로 split 하지 않는다 —
``data/MobiBench/gui-model_stage{1,2}.jsonl`` 두 단일 파일이 eval 입력.

Stage 1 (World Modeling)
    AC: ``episodes_meta.jsonl.primary_app`` 기반 **app-level ID/OOD** split.
        Stage 2 와 동일 partition 을 공유 (한 번 계산 → 양쪽 재사용) 해
        Stage 2 OOD 앱이 world-modeling 학습에서도 노출되지 않게 한다.
    MC: meta 없음 → 자동 random split (단일 ``_test.jsonl``).
Stage 2 (Action Prediction)
    AC, AC_2, AC_3: app-level ID/OOD split (Stage 1 과 partition 공유).
        AC_3 는 Stage 1 partition (action_pred 기준) 을 그대로 재사용해
        Stage 1↔Stage 2 OOD app 집합을 일치시킨다.
    MC: 데이터 없음 → 자동 skip (``--skip-stage2`` 기본 적용).

``primary_app`` 값은 앱 라벨이 아닌 package 식별자 (예:
``com.ajnsnewmedia.kitchenstories``) 이며,
``scripts/extract_androidcontrol_metadata.py`` 가
AndroidAccessibilityForest proto 에서 전경 application window 의
``package_name`` 을 다수결로 집계해 생성한다.

AC_3 Stage 1 은 항상 ``gui-model_stage1_{state,action}_pred_filtered.jsonl``
을 입력으로 사용한다 (mm-expanded length > cutoff_len 샘플을 사전 제거한
파일). 필터는 ``scripts/filter_long_samples.py`` 가 만든다 — 누락 시
명시적으로 에러를 발생시킨다. Stage 2 는 ``gui-model_stage2.jsonl`` 원본
을 그대로 사용한다 (마지막 message 가 ``<thought>...</thought>
<action>{...}</action>`` 래핑이라 ``_parse_action_payload`` 로 추출).

Usage
-----
  # AC: Stage 1 + Stage 2 모두 ID/OOD (defaults: train=50000, test_id=3000, test_ood=3000)
  python scripts/split_data.py --dataset AndroidControl

  # MC: Stage 1 random split 만 수행 (Stage 2 자동 skip, meta 없음)
  python scripts/split_data.py --dataset MonkeyCollection

  # AC 인데 Stage 1 만 random 으로 강제하고 싶을 때
  python scripts/split_data.py --dataset AndroidControl --stage1-mode random
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


DATASET_DIRS = {
    "AndroidControl":   "AndroidControl",
    "AC":               "AndroidControl",
    "AndroidControl_2": "AndroidControl_2",
    "AC_2":             "AndroidControl_2",
    "AndroidControl_3": "AndroidControl_3",
    "AC_3":             "AndroidControl_3",
    "MonkeyCollection": "MonkeyCollection",
    "MC":               "MonkeyCollection",
}

# Stage 2 분할을 지원하지 않는 데이터셋 (Stage 1 전용).
_STAGE1_ONLY = {"MonkeyCollection", "MC"}

# AC_3: stage1 이 state_pred / action_pred 두 파일로 갈라져 있고, EXP 별 비율 mix
# train 파일을 만들어야 한다 (train 단일 + ID/OOD test 가 아니라 별도 흐름).
_AC3_RATIO_MIX = {"AndroidControl_3", "AC_3"}

EPISODE_RE = re.compile(r"episode_(\d+)")
_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def _parse_action_payload(value: str) -> dict:
    """messages[-1].value 에서 action JSON dict 추출.

    AC stage2 / AC_3 stage1 action_pred 는 pure JSON,
    AC_3 stage2 는 ``<thought>...</thought>\\n<action>{...}</action>`` 래핑."""
    m = _ACTION_TAG_RE.search(value)
    payload = m.group(1) if m else value
    return json.loads(payload)


# ── IO helpers ────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(entries: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Stage 1 random split (MC fallback) ────────────────────────────────────
def split_stage1_random(entries: list, ratio: float, seed: int) -> tuple[list, list]:
    """Random split for Stage 1 (used when meta is absent — e.g., MC)."""
    rng = random.Random(seed)
    shuffled = entries.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


# ── Sampling helpers ──────────────────────────────────────────────────────
def stratified_subsample(
    entries: list, target_size: int, seed: int, type_key: str = "type"
) -> list:
    """Subsample preserving action-type ratio via largest-remainder method.

    마지막 message 가 action JSON 임을 가정한다.
    Stage 2 (AC) 는 ``type`` 키, AC_3 action_pred 는 ``action_type`` 키 사용 →
    ``type_key`` 인자로 분기."""
    rng = random.Random(seed)
    type_groups = defaultdict(list)
    for entry in entries:
        action = _parse_action_payload(entry["messages"][-1]["value"])
        type_groups[action.get(type_key, "unknown")].append(entry)

    total = len(entries)
    if target_size >= total:
        return entries.copy()

    quotas: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for atype, group in type_groups.items():
        exact = len(group) / total * target_size
        quotas[atype] = int(exact)
        remainders[atype] = exact - int(exact)

    leftover = target_size - sum(quotas.values())
    for atype, _ in sorted(remainders.items(), key=lambda kv: -kv[1])[:leftover]:
        quotas[atype] += 1

    sampled: list = []
    for atype, group in type_groups.items():
        rng.shuffle(group)
        sampled.extend(group[: quotas[atype]])
    return sampled


def random_subsample(entries: list, target_size: int, seed: int) -> list:
    """Stage 1 용 단순 random subsample (action-type stratification 없음)."""
    if target_size >= len(entries):
        return entries.copy()
    rng = random.Random(seed)
    shuffled = entries.copy()
    rng.shuffle(shuffled)
    return shuffled[:target_size]


# ── Episode / app mapping ─────────────────────────────────────────────────
def episode_id_from_entry(entry: dict) -> str | None:
    images = entry.get("images") or []
    if not images:
        return None
    m = EPISODE_RE.search(str(images[0]))
    return m.group(1) if m else None


def _norm_ep(raw: object) -> str:
    """Strip leading zeros so '006881' and 6881 both key as '6881'."""
    s = str(raw).strip()
    try:
        return str(int(s))
    except ValueError:
        return s


def _build_ep_to_app(meta_entries: list[dict]) -> dict[str, str | None]:
    return {_norm_ep(m.get("episode_id")): m.get("primary_app") for m in meta_entries}


# ── App-level partition ───────────────────────────────────────────────────
def partition_apps(
    app_to_rows: dict[str, list[dict]],
    ood_budget: int,
    id_budget: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """OOD-first app partition so the test_ood pool is always feasible.

    Apps are shuffled then appended to the OOD bucket until ``ood_budget`` rows
    are reached; remaining apps form the in-domain bucket. Warnings are emitted
    if either bucket is under-budget so the caller can adjust sizes.
    """
    apps = list(app_to_rows.keys())
    rng.shuffle(apps)

    ood_apps: list[str] = []
    ood_rows = 0
    idx = 0
    while ood_rows < ood_budget and idx < len(apps):
        a = apps[idx]
        ood_apps.append(a)
        ood_rows += len(app_to_rows[a])
        idx += 1
    id_apps = apps[idx:]

    if ood_rows < ood_budget:
        print(
            f"[warn] OOD pool has {ood_rows} rows (< ood_budget={ood_budget}). "
            "Consider lowering --stage2-test-ood-size or labeling more episodes.",
            file=sys.stderr,
        )
    id_rows = sum(len(app_to_rows[a]) for a in id_apps)
    if id_rows < id_budget:
        print(
            f"[warn] IN-DOMAIN pool has {id_rows} rows (< id_budget={id_budget}). "
            "train/test_id will be smaller than requested.",
            file=sys.stderr,
        )
    return id_apps, ood_apps


def compute_app_partition(
    stage2_entries: list[dict],
    meta_entries: list[dict],
    ood_row_budget: int,
    id_row_budget: int,
    seed: int,
) -> tuple[list[str], list[str], dict[str, list[dict]], list[dict], dict[str, str | None]]:
    """Stage 2 행 수 기준으로 app partition 을 단일 계산.

    Returns
    -------
    id_apps, ood_apps : list[str]
        Partition 결과 (Stage 1 / Stage 2 모두에서 재사용).
    app_to_rows_stage2 : dict[str, list[dict]]
        Stage 2 행을 app 별로 그룹핑한 결과 (Stage 2 split 에서 직접 사용).
    null_rows_stage2 : list[dict]
        episode_id 추출 실패 또는 app 라벨 결측 행 (Stage 2 train 에 옵션 합류).
    ep_to_app : dict[str, str | None]
        Stage 1 라우팅에서 재사용할 episode → primary_app 매핑.
    """
    ep_to_app = _build_ep_to_app(meta_entries)

    null_rows: list[dict] = []
    app_to_rows: dict[str, list[dict]] = defaultdict(list)
    for entry in stage2_entries:
        ep = episode_id_from_entry(entry)
        if ep is None:
            null_rows.append(entry)
            continue
        app = ep_to_app.get(_norm_ep(ep))
        if app is None or not str(app).strip():
            null_rows.append(entry)
            continue
        app_to_rows[str(app).strip()].append(entry)

    rng = random.Random(seed)
    id_apps, ood_apps = partition_apps(
        app_to_rows,
        ood_budget=ood_row_budget,
        id_budget=id_row_budget,
        rng=rng,
    )
    return id_apps, ood_apps, dict(app_to_rows), null_rows, ep_to_app


def route_entries_by_app(
    entries: list[dict],
    ep_to_app: dict[str, str | None],
    id_apps: set[str],
    ood_apps: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """주어진 partition 으로 entries 를 (id_pool, ood_pool, null_pool) 로 라우팅."""
    id_pool: list[dict] = []
    ood_pool: list[dict] = []
    null_pool: list[dict] = []
    for entry in entries:
        ep = episode_id_from_entry(entry)
        if ep is None:
            null_pool.append(entry)
            continue
        app = ep_to_app.get(_norm_ep(ep))
        if app is None or not str(app).strip():
            null_pool.append(entry)
            continue
        a = str(app).strip()
        if a in id_apps:
            id_pool.append(entry)
        elif a in ood_apps:
            ood_pool.append(entry)
        else:
            # partition 밖 (이론상 도달 불가) — 안전하게 null 로
            null_pool.append(entry)
    return id_pool, ood_pool, null_pool


# ── Stage 1 ID/OOD split (AC) ─────────────────────────────────────────────
def build_stage1_id_ood_split(
    stage1_entries: list[dict],
    ep_to_app: dict[str, str | None],
    id_apps: list[str],
    ood_apps: list[str],
    train_size: int,
    test_id_size: int,
    test_ood_size: int,
    seed: int,
    exclude_null_app: bool = False,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Stage 2 와 동일 partition 으로 Stage 1 을 ID/OOD 분할.

    Stage 1 entry 는 마지막 message 가 UI XML (action JSON 이 아님) 이므로
    action-type stratification 대신 단순 random subsample 을 사용한다.
    """
    id_set = set(id_apps)
    ood_set = set(ood_apps)
    id_pool, ood_pool, null_pool = route_entries_by_app(
        stage1_entries, ep_to_app, id_set, ood_set
    )

    test_id = random_subsample(id_pool, test_id_size, seed + 1)
    marks = {id(e) for e in test_id}
    id_remaining = [e for e in id_pool if id(e) not in marks]

    train_pool = list(id_remaining)
    if not exclude_null_app:
        train_pool.extend(null_pool)
    train = random_subsample(train_pool, train_size, seed)

    test_ood = random_subsample(ood_pool, test_ood_size, seed + 2)

    info = {
        "total_rows": len(stage1_entries),
        "labeled_rows": len(id_pool) + len(ood_pool),
        "null_rows": len(null_pool),
        "id_pool_rows": len(id_pool),
        "ood_pool_rows": len(ood_pool),
    }
    return train, test_id, test_ood, info


# ── Stage 2 ID/OOD split (AC) ─────────────────────────────────────────────
def build_stage2_id_ood_split(
    app_to_rows: dict[str, list[dict]],
    null_rows: list[dict],
    id_apps: list[str],
    ood_apps: list[str],
    train_size: int,
    test_id_size: int,
    test_ood_size: int,
    seed: int,
    exclude_null_app: bool = False,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Stage 2 ID/OOD split — precomputed partition 을 받아 샘플링만 수행."""
    id_pool: list[dict] = []
    for a in id_apps:
        id_pool.extend(app_to_rows.get(a, []))
    ood_pool: list[dict] = []
    for a in ood_apps:
        ood_pool.extend(app_to_rows.get(a, []))

    test_id = stratified_subsample(id_pool, test_id_size, seed + 1)
    marks = {id(e) for e in test_id}
    id_remaining = [e for e in id_pool if id(e) not in marks]

    train_pool = list(id_remaining)
    if not exclude_null_app:
        train_pool.extend(null_rows)
    train = stratified_subsample(train_pool, train_size, seed)

    test_ood = stratified_subsample(ood_pool, test_ood_size, seed + 2)

    info = {
        "total_rows": sum(len(v) for v in app_to_rows.values()) + len(null_rows),
        "labeled_rows": sum(len(v) for v in app_to_rows.values()),
        "null_rows": len(null_rows),
        "unique_labeled_apps": len(app_to_rows),
        "id_apps": len(id_apps),
        "ood_apps": len(ood_apps),
        "id_pool_rows": len(id_pool),
        "ood_pool_rows": len(ood_pool),
    }
    return train, test_id, test_ood, info


# ── Reporting ─────────────────────────────────────────────────────────────
def print_stage2_distribution(entries: list, label: str, type_key: str = "type") -> None:
    action_types: list[str] = []
    for entry in entries:
        try:
            action = _parse_action_payload(entry["messages"][-1]["value"])
            action_types.append(action.get(type_key, "unknown"))
        except (json.JSONDecodeError, KeyError):
            action_types.append("parse_error")

    counts = Counter(action_types)
    total = len(action_types)
    print(f"  {label} ({total}):")
    if total == 0:
        return
    for atype, count in counts.most_common():
        print(f"    {atype}: {count} ({count / total:.1%})")


# ── AC_3 (state_pred + action_pred ratio mix) ─────────────────────────────
def _parse_ratios(spec: str) -> list[tuple[int, int]]:
    """``"7:3,3:7,5:5"`` → ``[(7, 3), (3, 7), (5, 5)]``."""
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(":")
        ra, rb = int(a), int(b)
        if ra <= 0 or rb <= 0:
            raise ValueError(f"비율 쌍은 양수여야 함: {chunk}")
        out.append((ra, rb))
    if not out:
        raise ValueError(f"비어 있는 ratio spec: {spec!r}")
    return out


def run_ac3_split(args, dataset_dir: Path) -> int:
    # AC_3 는 항상 _filtered 소스만 사용 (mm-expanded length > cutoff_len 샘플
    # 제거 후의 jsonl). 학습 시 Qwen3-VL get_rope_index 의 broadcast shape
    # mismatch 를 피하기 위함. 필터 산출은 scripts/filter_long_samples.py.
    state_pred_path  = dataset_dir / "gui-model_stage1_state_pred_filtered.jsonl"
    action_pred_path = dataset_dir / "gui-model_stage1_action_pred_filtered.jsonl"
    meta_path        = dataset_dir / "episodes_meta.jsonl"

    for p in (state_pred_path, action_pred_path):
        if not p.exists():
            print(
                f"[ERROR] AC_3 source 가 없습니다: {p}\n"
                f"        먼저 `python scripts/filter_long_samples.py --dataset AC_3` 로 "
                f"_filtered.jsonl 을 생성하세요.",
                file=sys.stderr,
            )
            return 1
    if not meta_path.exists():
        print(f"[ERROR] AC_3 split 에 필요한 파일이 없습니다: {meta_path}", file=sys.stderr)
        return 1

    try:
        ratios = _parse_ratios(args.ac3_ratios)
    except ValueError as exc:
        print(f"[ERROR] --ac3-ratios 파싱 실패: {exc}", file=sys.stderr)
        return 1

    total = args.ac3_train_total
    test_id_size = args.stage1_test_id_size
    test_ood_size = args.stage1_test_ood_size
    seed = args.seed

    # 단일 task 가 한 EXP 에서 가져갈 최대 row 수 (ID partition budget 산정용)
    max_per_side = max(
        max(total * rs // (rs + ra) for rs, ra in ratios),
        max(total - total * rs // (rs + ra) for rs, ra in ratios),
    )

    print(f"Dataset: AC_3 ({dataset_dir})")
    print(f"Seed: {seed}")
    print(f"EXP ratios (state:action): "
          f"{', '.join(f'{a}:{b}' for a, b in ratios)}")
    print(f"Train total per EXP: {total}")
    print()

    sp_entries = load_jsonl(state_pred_path)
    ap_entries = load_jsonl(action_pred_path)
    meta_entries = load_jsonl(meta_path)

    # 두 파일은 같은 episode set 을 공유 → action_pred 기준 1회 partition.
    id_apps, ood_apps, _, _, ep_to_app = compute_app_partition(
        ap_entries, meta_entries,
        ood_row_budget=test_ood_size,
        id_row_budget=max_per_side + test_id_size,
        seed=seed,
    )
    id_set, ood_set = set(id_apps), set(ood_apps)

    sp_id, sp_ood, sp_null = route_entries_by_app(sp_entries, ep_to_app, id_set, ood_set)
    ap_id, ap_ood, ap_null = route_entries_by_app(ap_entries, ep_to_app, id_set, ood_set)

    print("=== App Partition (state_pred / action_pred 공유) ===")
    print(f"  Unique labeled apps: {len(id_apps) + len(ood_apps)}")
    print(f"  IN-DOMAIN apps:  {len(id_apps)}")
    print(f"  OUT-OF-DOMAIN apps: {len(ood_apps)}")
    print(f"  state_pred  ID/OOD pool: {len(sp_id)} / {len(sp_ood)} "
          f"(null {len(sp_null)})")
    print(f"  action_pred ID/OOD pool: {len(ap_id)} / {len(ap_ood)} "
          f"(null {len(ap_null)})")
    print()

    # state_pred 는 random, action_pred 는 action_type stratified.
    def sp_subsample(entries, n, s):
        return random_subsample(entries, n, s)

    def ap_subsample(entries, n, s):
        return stratified_subsample(entries, n, s, type_key="action_type")

    # Test 표본 (각 task 별 3000)
    sp_test_id  = sp_subsample(sp_id,  test_id_size,  seed + 11)
    sp_test_ood = sp_subsample(sp_ood, test_ood_size, seed + 12)
    ap_test_id  = ap_subsample(ap_id,  test_id_size,  seed + 21)
    ap_test_ood = ap_subsample(ap_ood, test_ood_size, seed + 22)

    # train pool: ID pool 에서 test_id 제거 (episode 가 아닌 row id 단위)
    def _disjoint(pool, test):
        marks = {id(e) for e in test}
        return [e for e in pool if id(e) not in marks]
    sp_train_pool = _disjoint(sp_id, sp_test_id)
    ap_train_pool = _disjoint(ap_id, ap_test_id)

    # Test 4 파일 작성
    test_id_sp_path  = dataset_dir / "gui-model_stage1_test_id_state_pred.jsonl"
    test_id_ap_path  = dataset_dir / "gui-model_stage1_test_id_action_pred.jsonl"
    test_ood_sp_path = dataset_dir / "gui-model_stage1_test_ood_state_pred.jsonl"
    test_ood_ap_path = dataset_dir / "gui-model_stage1_test_ood_action_pred.jsonl"
    write_jsonl(sp_test_id,  test_id_sp_path)
    write_jsonl(ap_test_id,  test_id_ap_path)
    write_jsonl(sp_test_ood, test_ood_sp_path)
    write_jsonl(ap_test_ood, test_ood_ap_path)

    print("=== Stage 1 Test Sets ===")
    print(f"  → {test_id_sp_path.name}  ({len(sp_test_id)})")
    print(f"  → {test_id_ap_path.name}  ({len(ap_test_id)})")
    print(f"  → {test_ood_sp_path.name} ({len(sp_test_ood)})")
    print(f"  → {test_ood_ap_path.name} ({len(ap_test_ood)})")
    print_stage2_distribution(ap_test_id,  "action_pred test_id",  type_key="action_type")
    print_stage2_distribution(ap_test_ood, "action_pred test_ood", type_key="action_type")
    print()

    # EXP ratio mixing
    print("=== EXP Ratio Mix (state_pred : action_pred) ===")
    for i, (rs, ra) in enumerate(ratios):
        n_state = total * rs // (rs + ra)
        n_action = total - n_state
        if n_state > len(sp_train_pool):
            print(f"[warn] state_pred {n_state} > pool {len(sp_train_pool)}; "
                  f"truncating to pool size.", file=sys.stderr)
        if n_action > len(ap_train_pool):
            print(f"[warn] action_pred {n_action} > pool {len(ap_train_pool)}; "
                  f"truncating to pool size.", file=sys.stderr)
        state_chunk  = sp_subsample(sp_train_pool,  n_state,  seed + 100 + i)
        action_chunk = ap_subsample(ap_train_pool, n_action, seed + 200 + i)
        mixed = state_chunk + action_chunk
        random.Random(seed + 300 + i).shuffle(mixed)

        out_path = dataset_dir / f"gui-model_stage1_train_{rs}_{ra}.jsonl"
        write_jsonl(mixed, out_path)
        print(f"  EXP {rs}:{ra} → {out_path.name} "
              f"({len(mixed)} = state {len(state_chunk)} + action {len(action_chunk)})")
        print_stage2_distribution(
            action_chunk, f"  action_pred chunk ({rs}:{ra})", type_key="action_type"
        )
    print()

    # ── Stage 2 (Action Prediction, ID/OOD) ─────────────────────────────
    # Stage 1 partition (id_apps/ood_apps) 을 그대로 적용해 Stage 1↔Stage 2
    # OOD app 집합을 일치시킨다.
    stage2_path = dataset_dir / "gui-model_stage2.jsonl"
    if args.skip_stage2:
        print("[skip] Stage 2 split (per --skip-stage2)")
    elif not stage2_path.exists():
        print(f"[skip] Stage 2 file not found: {stage2_path}")
    else:
        s2_entries = load_jsonl(stage2_path)
        s2_id, s2_ood, s2_null = route_entries_by_app(
            s2_entries, ep_to_app, id_set, ood_set
        )

        s2_test_id = stratified_subsample(
            s2_id, args.stage2_test_id_size, seed + 31, type_key="action_type"
        )
        marks = {id(e) for e in s2_test_id}
        s2_id_remaining = [e for e in s2_id if id(e) not in marks]
        s2_train_pool = list(s2_id_remaining)
        if not args.stage2_exclude_null_app:
            s2_train_pool.extend(s2_null)
        s2_train = stratified_subsample(
            s2_train_pool, args.stage2_train_size, seed, type_key="action_type"
        )
        s2_test_ood = stratified_subsample(
            s2_ood, args.stage2_test_ood_size, seed + 32, type_key="action_type"
        )

        s2_train_path   = dataset_dir / "gui-model_stage2_train.jsonl"
        s2_test_id_path = dataset_dir / "gui-model_stage2_test_id.jsonl"
        s2_test_ood_path = dataset_dir / "gui-model_stage2_test_ood.jsonl"
        write_jsonl(s2_train,    s2_train_path)
        write_jsonl(s2_test_id,  s2_test_id_path)
        write_jsonl(s2_test_ood, s2_test_ood_path)

        print("=== Stage 2 (Action Prediction, ID/OOD) ===")
        print(f"  Total rows: {len(s2_entries)} "
              f"(labeled {len(s2_id) + len(s2_ood)}, null {len(s2_null)})")
        print(f"  IN-DOMAIN pool: {len(s2_id)} rows")
        print(f"  OUT-OF-DOMAIN pool: {len(s2_ood)} rows")
        print(f"  → {s2_train_path.name} ({len(s2_train)})")
        print(f"  → {s2_test_id_path.name} ({len(s2_test_id)})")
        print(f"  → {s2_test_ood_path.name} ({len(s2_test_ood)})")
        print_stage2_distribution(s2_train,    "train",    type_key="action_type")
        print_stage2_distribution(s2_test_id,  "test_id",  type_key="action_type")
        print_stage2_distribution(s2_test_ood, "test_ood", type_key="action_type")
        print()

    print("Done.")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train/Test split builder (Stage 1/2 ID/OOD; MC Stage 1 random)",
    )
    parser.add_argument(
        "--dataset", required=True, choices=sorted(DATASET_DIRS),
        help="Dataset short or full name",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Data root (default: <repo>/data)",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")

    # Stage 1
    parser.add_argument(
        "--stage1-mode", choices=("auto", "random", "id-ood"), default="auto",
        help="auto: meta 가 있으면 id-ood, 없으면 random (default).",
    )
    parser.add_argument(
        "--stage1-ratio", type=float, default=0.95,
        help="Stage 1 random mode 전용 train 비율 (default 0.95).",
    )
    parser.add_argument("--stage1-train-size", type=int, default=50000)
    parser.add_argument("--stage1-test-id-size", type=int, default=3000)
    parser.add_argument("--stage1-test-ood-size", type=int, default=3000)
    parser.add_argument(
        "--stage1-exclude-null-app", action="store_true",
        help="Stage 1 ID/OOD 모드에서 null primary_app 행을 train pool 에 합치지 않음.",
    )

    # Stage 2
    parser.add_argument("--stage2-train-size", type=int, default=15000)
    parser.add_argument("--stage2-test-id-size", type=int, default=3000)
    parser.add_argument("--stage2-test-ood-size", type=int, default=3000)
    parser.add_argument(
        "--stage2-exclude-null-app", action="store_true",
        help="Drop episodes with null primary_app instead of pooling them into train.",
    )

    # AC_3 (state_pred + action_pred ratio mixing)
    parser.add_argument(
        "--ac3-train-total", type=int, default=50000,
        help="AC_3 EXP 별 train 파일의 총 row 수 (default 50000).",
    )
    parser.add_argument(
        "--ac3-ratios", type=str, default="7:3,3:7,5:5",
        help="AC_3 EXP 비율 list (state:action), 콤마 구분. "
             "Default '7:3,3:7,5:5' → 출력 파일명 train_7_3 / train_3_7 / train_5_5.",
    )

    args = parser.parse_args()

    if args.dataset in _STAGE1_ONLY and not args.skip_stage2:
        print(f"[info] {args.dataset} 은 Stage 1 전용입니다. Stage 2 는 자동 skip.")
        args.skip_stage2 = True

    ds_dir_name = DATASET_DIRS[args.dataset]
    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        data_root = Path(__file__).resolve().parent.parent / "data"
    dataset_dir = data_root / ds_dir_name
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    # AC_3 는 stage1 이 (state_pred, action_pred) 두 파일이라 별도 흐름.
    if args.dataset in _AC3_RATIO_MIX:
        return run_ac3_split(args, dataset_dir)

    stage1_path = dataset_dir / "gui-model_stage1.jsonl"
    stage2_path = dataset_dir / "gui-model_stage2.jsonl"
    meta_path = dataset_dir / "episodes_meta.jsonl"
    meta_available = meta_path.exists()

    # Resolve Stage 1 mode (auto → id-ood/random by meta 존재)
    stage1_mode = args.stage1_mode
    if stage1_mode == "auto":
        stage1_mode = "id-ood" if meta_available else "random"
    if stage1_mode == "id-ood" and not meta_available:
        print(f"[ERROR] Stage 1 id-ood requires {meta_path} — run "
              f"extract_{ds_dir_name.lower()}_metadata.py first.", file=sys.stderr)
        return 1

    print(f"Dataset: {args.dataset} ({dataset_dir})")
    print(f"Seed: {args.seed}")
    print(f"Stage 1 mode: {stage1_mode}")
    print()

    # ── Shared partition (id-ood 모드에서 한 번만 계산) ──────────────────
    partition_needed = (
        (not args.skip_stage1 and stage1_mode == "id-ood") or
        (not args.skip_stage2)
    )
    id_apps: list[str] = []
    ood_apps: list[str] = []
    app_to_rows_s2: dict[str, list[dict]] = {}
    null_rows_s2: list[dict] = []
    ep_to_app: dict[str, str | None] = {}
    stage2_entries: list[dict] = []

    if partition_needed:
        if not stage2_path.exists():
            print(f"[ERROR] App partition requires {stage2_path}.", file=sys.stderr)
            return 1
        if not meta_available:
            print(f"[ERROR] App partition requires {meta_path}.", file=sys.stderr)
            return 1
        stage2_entries = load_jsonl(stage2_path)
        meta_entries = load_jsonl(meta_path)
        id_apps, ood_apps, app_to_rows_s2, null_rows_s2, ep_to_app = (
            compute_app_partition(
                stage2_entries,
                meta_entries,
                ood_row_budget=args.stage2_test_ood_size,
                id_row_budget=args.stage2_test_id_size * 2,
                seed=args.seed,
            )
        )
        print("=== App Partition (Stage 1 / Stage 2 공유) ===")
        print(f"  Total Stage 2 rows: {len(stage2_entries)} "
              f"(labeled {sum(len(v) for v in app_to_rows_s2.values())}, "
              f"null {len(null_rows_s2)})")
        print(f"  Unique labeled apps: {len(app_to_rows_s2)}")
        print(f"  IN-DOMAIN apps:  {len(id_apps)}")
        print(f"  OUT-OF-DOMAIN apps: {len(ood_apps)}")
        print()

    # ── Stage 1 ───────────────────────────────────────────────────────
    if args.skip_stage1:
        print("[skip] Stage 1 split (per --skip-stage1)")
    elif not stage1_path.exists():
        print(f"[skip] Stage 1 file not found: {stage1_path}")
    elif stage1_mode == "id-ood":
        stage1_entries = load_jsonl(stage1_path)
        train, test_id, test_ood, info = build_stage1_id_ood_split(
            stage1_entries,
            ep_to_app,
            id_apps,
            ood_apps,
            train_size=args.stage1_train_size,
            test_id_size=args.stage1_test_id_size,
            test_ood_size=args.stage1_test_ood_size,
            seed=args.seed,
            exclude_null_app=args.stage1_exclude_null_app,
        )

        train_path = dataset_dir / "gui-model_stage1_train.jsonl"
        test_id_path = dataset_dir / "gui-model_stage1_test_id.jsonl"
        test_ood_path = dataset_dir / "gui-model_stage1_test_ood.jsonl"
        write_jsonl(train, train_path)
        write_jsonl(test_id, test_id_path)
        write_jsonl(test_ood, test_ood_path)

        print("=== Stage 1 (World Modeling, ID/OOD) ===")
        print(f"  Total rows: {info['total_rows']} "
              f"(labeled {info['labeled_rows']}, null {info['null_rows']})")
        print(f"  IN-DOMAIN pool: {info['id_pool_rows']} rows")
        print(f"  OUT-OF-DOMAIN pool: {info['ood_pool_rows']} rows")
        print(f"  → {train_path.name} ({len(train)})")
        print(f"  → {test_id_path.name} ({len(test_id)})")
        print(f"  → {test_ood_path.name} ({len(test_ood)})")
        print()
    else:  # random
        stage1_entries = load_jsonl(stage1_path)
        train, test = split_stage1_random(stage1_entries, args.stage1_ratio, args.seed)
        train_path = dataset_dir / "gui-model_stage1_train.jsonl"
        test_path = dataset_dir / "gui-model_stage1_test.jsonl"
        write_jsonl(train, train_path)
        write_jsonl(test, test_path)
        print("=== Stage 1 (World Modeling, random) ===")
        print(f"  Total: {len(stage1_entries)}")
        print(f"  Train: {len(train)} ({len(train) / max(len(stage1_entries), 1):.1%})")
        print(f"  Test:  {len(test)} ({len(test) / max(len(stage1_entries), 1):.1%})")
        print(f"  → {train_path.name}")
        print(f"  → {test_path.name}")
        print()

    # ── Stage 2 (ID/OOD) ──────────────────────────────────────────────
    if args.skip_stage2:
        print("[skip] Stage 2 split (per --skip-stage2)")
    else:
        train, test_id, test_ood, info = build_stage2_id_ood_split(
            app_to_rows_s2,
            null_rows_s2,
            id_apps,
            ood_apps,
            train_size=args.stage2_train_size,
            test_id_size=args.stage2_test_id_size,
            test_ood_size=args.stage2_test_ood_size,
            seed=args.seed,
            exclude_null_app=args.stage2_exclude_null_app,
        )

        train_path = dataset_dir / "gui-model_stage2_train.jsonl"
        test_id_path = dataset_dir / "gui-model_stage2_test_id.jsonl"
        test_ood_path = dataset_dir / "gui-model_stage2_test_ood.jsonl"
        write_jsonl(train, train_path)
        write_jsonl(test_id, test_id_path)
        write_jsonl(test_ood, test_ood_path)

        print("=== Stage 2 (Action Prediction, ID/OOD) ===")
        print(f"  Total rows: {info['total_rows']} "
              f"(labeled {info['labeled_rows']}, null {info['null_rows']})")
        print(f"  IN-DOMAIN pool: {info['id_pool_rows']} rows")
        print(f"  OUT-OF-DOMAIN pool: {info['ood_pool_rows']} rows")
        print(f"  → {train_path.name} ({len(train)})")
        print(f"  → {test_id_path.name} ({len(test_id)})")
        print(f"  → {test_ood_path.name} ({len(test_ood)})")
        print_stage2_distribution(train, "train")
        print_stage2_distribution(test_id, "test_id")
        print_stage2_distribution(test_ood, "test_ood")
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
