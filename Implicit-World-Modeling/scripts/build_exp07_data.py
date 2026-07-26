#!/usr/bin/env python3
"""AC_EXP07 stage1/stage2 train+test 데이터 정본 빌더 — 신규 0725 소스 (myset), 누출 0.

이 스크립트가 **AC_EXP07 train/test jsonl 의 유일한 커밋된 생성 경로**다.
소스는 조병웅 제공 0725 필터링본(myset)이며, 2026-07-26 부터 공유 원본 디렉토리
``data/AndroidControl/`` 에 명명규칙(ARCHITECTURE §3 규칙 2 — 공유 디렉토리의 변형본은
``EXP{NN}_`` 접두)대로 평탄하게 놓인다. 세 파일에서 파생한다:

  EXP07_stage1_state.jsonl   65,408  state-pred (NEXT_STATE_PREDICTION, img1, gpt=XML)
  EXP07_stage2.jsonl         88,387  downstream (NEXT_ACTION_PREDICTION, img1, gpt=<thought><action>)
                                     — stage2 15K 가 주 소비자지만 stage1 down 10K·stage1 action test 도
                                       같은 풀에서 나온다 (§3 규칙 3: 접두는 최초 소비 EXP 하나).
  EXP07_open_aug.jsonl          100  open 증강 (전부 open, home.jpg) — s1/s2 에 50 씩 균등 분배라 stage 중립명.

⚠ ``--source-dir`` 는 이제 **공유 디렉토리**(``data/AndroidControl/``)를 가리킨다. 빌더는 위 세
이름만 열므로 같은 디렉토리의 다른 EXP 자산에는 손대지 않는다.

산출 (``data/AndroidControl_EXP07/``):
  train  stage1_train.jsonl  50,000 = state 40,000(가중) + downstream 10,000(이미지 제거)
         stage2_train.jsonl  15,000 (이미지 유지, 무가중)
  test   stage1_test_{id,ood}_state.jsonl   (state_pred·이미지유지)   ← EXP05 state test 키
         stage1_test_{id,ood}_action.jsonl  (with_history·이미지제거) ← EXP05 stage2 test 키
         stage2_test_{id,ood}.jsonl         (with_history·이미지유지) ← EXP05 stage2 test 키

핵심 규격 (2026-07-25 회의 + 사용자 확정)
------------------------------------------
1. 이미지 경로 remap: ``myset/images/episode_{N}_step_{M}.jpg`` →
   ``AndroidControl/images/episode_{N:06d}_step_{M}.jpg``, ``home.jpg`` → ``AndroidControl/images/home.jpg``.
2. stage1 train 50K = state 40K(8:2 의 8, 이미지 유지, diff v2 1:0.2) + downstream 10K(2, 이미지 제거).
3. stage2 train 15K = downstream (이미지 유지, 무가중).
4. **비중복**: stage1-down(10K) ∩ stage2(15K) = ∅ (with_history 풀 disjoint 분할, action 비율 largest-remainder).
5. **우선포함**: open 증강 100 → s1 50 / s2 50 먼저; answer-terminate 는 가용분·쿼터에 맞춘
   동적 절반 (``per = min(len(ans) // 2, q1, q2)``) 을 각 stage 에 먼저 배치 — 실현치는 sidecar 의
   ``priority`` 참조 (seed 7 / 0725 소스 기준 가용 1,318 → s1 659 / s2 659).
6. **누출 0 (사용자 확정)**: EXP07 test = EXP05 test (ep,step) 키를 myset 포맷으로 재현.
   그 test 키 union 을 **train 두 풀에서 전량 제외** → EXP07 train ∩ EXP07 test = 0 (교차목적 포함).
   test 는 EXP07 자체 파일(EXP05 심링크 대체). id/ood 는 EXP05 배정 승계.

7. **길이 필터**: mm-expanded 길이 > ``cutoff_len``(24576) 인 이미지-보유 샘플은 학습
   dataloader 에서 죽으므로(잘리기 전 image_grid_thw vs 잘린 input_ids 불일치 →
   Qwen2.5-VL "Image features and image tokens do not match") **샘플링 전에 두 풀에서
   제외**한다. ``scripts/filter_long_samples.py`` 의 길이 계산(build_length_fn)을 재사용해
   LlamaFactory collator 길이와 일치시킨다. 제외 실현치는 sidecar 의 ``length_filter`` 참조.

**W 상수 불변식**: UNCHANGED=0.2. **metric v2 고정**. 샘플은 ``--seed`` 로 재현.

Usage
-----
  .venv/bin/python scripts/build_exp07_data.py
  .venv/bin/python scripts/build_exp07_data.py --source-dir data/AndroidControl --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJ = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / "diff_loss"))
sys.path.insert(0, str(SCRIPTS))  # filter_long_samples (길이 필터) import 용

# ── 불변식 ──────────────────────────────────────────────────────────────────
W_ADDED = 1.0
W_MODIFIED = 1.0
W_UNCHANGED = 0.2
METRIC_VERSION = "v2"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

N_STATE = 40000
N_S1_DOWN = 10000
N_S2 = 15000
AUG_PER_STAGE = 50
DEFAULT_SEED = 7
MAX_DEGENERATE_FRAC = 0.30

# ── 길이 필터 (mm-expanded > cutoff 인 이미지-보유 샘플을 build 시점에 제외) ─────
# EXP07 stage{1,2} 학습 YAML 과 동일한 값. 이 길이를 넘는 이미지-보유 샘플은 학습
# dataloader 에서 잘리기 *전* 의 image_grid_thw 로 위치를 만들어, 잘린 input_ids 의
# image_pad 수와 vision feature 수가 어긋나 죽는다 (Qwen2.5-VL:
# "Image features and image tokens do not match"). scripts/filter_long_samples.py 와
# 같은 근본책 — 풀에서 초과 샘플을 빼 40000/10000/15000 은 필터된 풀에서 채운다.
CUTOFF_LEN = 24576
IMG_MAX_PIXELS = 1605632
IMG_MIN_PIXELS = 3136

# ── 파일명 ────────────────────────────────────────────────────────────────
SRC_STATE = "EXP07_stage1_state.jsonl"
SRC_DOWN = "EXP07_stage2.jsonl"
SRC_AUG = "EXP07_open_aug.jsonl"
SRC_DEFAULT_SUBDIR = "AndroidControl"

OUT_SUBDIR = "AndroidControl_EXP07"
EXP05_SUBDIR = "AndroidControl_EXP05"
TRAIN1_NAME = "stage1_train.jsonl"
TRAIN2_NAME = "stage2_train.jsonl"

# EXP07 test 산출 파일 (우리 형식)
TEST_STATE = {"id": "stage1_test_id_state.jsonl", "ood": "stage1_test_ood_state.jsonl"}
TEST_ACTION = {
    "id": "stage1_test_id_action.jsonl",
    "ood": "stage1_test_ood_action.jsonl",
}
TEST_S2 = {"id": "stage2_test_id.jsonl", "ood": "stage2_test_ood.jsonl"}
# EXP05 test 원천 (키 추출용)
EXP05_STATE = {"id": "stage1_test_id_state.jsonl", "ood": "stage1_test_ood_state.jsonl"}
EXP05_S2 = {"id": "stage2_test_id.jsonl", "ood": "stage2_test_ood.jsonl"}
# 과거 잠정본이 남긴 EXP05 심링크(4종) — 정리 대상
LEGACY_SYMLINKS = list(TEST_STATE.values()) + list(TEST_S2.values())

# ── image-strip 치환표 (fail-closed, 전 레코드 정확히 1회 매칭 검증됨) ─────────
_SYS_SUBS = {
    "You are a mobile GUI agent. You analyze screenshots and UI hierarchy XML to predict coordinates for actions.": "You are a mobile GUI agent. You analyze UI hierarchy XML to predict coordinates for actions.",
    "# Given: Task Instruction, Action History, Current UI State (XML + Screenshot)": "# Given: Task Instruction, Action History, Current UI State (XML)",
    "- Current UI State is provided as html-style XML and a screenshot.": "- Current UI State is provided as html-style XML.",
    "\n- Use the screenshot for visual context and layout verification.": "",
}
_HUM_SUBS = {"\n\n[Screenshot]\n<image>": ""}

_ACTION_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)
_GIVEN_ACT_RE = re.compile(r'"action"\s*:\s*"([a-z_]+)"')
_KEY_RE = re.compile(r"episode_(\d+)_step_(\d+)")
_IMG_RE = re.compile(r"^myset/images/episode_(\d+)_step_(\d+)\.jpg$")
_EP_RE = re.compile(r"episode_(\d+)")


def _abort(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


# ── 이미지 remap / 키 ────────────────────────────────────────────────────────


def remap_image(ip: str) -> str:
    if ip == "myset/images/home.jpg":
        return "AndroidControl/images/home.jpg"
    m = _IMG_RE.match(ip)
    if not m:
        _abort(f"이미지 remap 실패(예상 밖 패턴): {ip!r}")
    return f"AndroidControl/images/episode_{int(m.group(1)):06d}_step_{m.group(2)}.jpg"


def remap_record(rec: dict) -> dict:
    rec["images"] = [remap_image(ip) for ip in rec.get("images", [])]
    return rec


def _step_key(rec: dict) -> tuple[int, int] | None:
    """레코드 현재화면 (episode,step) 키 (images[0] 기준). home.jpg 등 비매칭이면 None."""
    imgs = rec.get("images", [])
    if not imgs:
        return None
    m = _KEY_RE.search(imgs[0])
    return (int(m.group(1)), int(m.group(2))) if m else None


def _keys_from_file(path: Path) -> set[tuple[int, int]]:
    """EXP05 test 파일에서 (episode,step) 키 집합 추출 (images 전체 기준)."""
    ks: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for ip in json.loads(line).get("images", []):
                m = _KEY_RE.search(ip)
                if m:
                    ks.add((int(m.group(1)), int(m.group(2))))
    return ks


# ── 파싱 ────────────────────────────────────────────────────────────────────


def _msg(rec: dict, role: str) -> str:
    return next((m["value"] for m in rec["messages"] if m["from"] == role), "")


def _episode_of(rec: dict, src: str, i: int) -> str:
    imgs = rec.get("images", [])
    if not imgs:
        _abort(f"{src}:{i} images 비어 episode 파싱 불가")
    m = _EP_RE.search(imgs[0])
    if not m:
        _abort(f"{src}:{i} images[0]={imgs[0]!r} episode 파싱 실패")
    return m.group(1)


def _action_of_gpt(rec: dict, src: str, i: int) -> tuple[str, bool]:
    gpt = _msg(rec, "gpt")
    ms = _ACTION_RE.findall(gpt)
    if len(ms) != 1:
        _abort(f"{src}:{i} gpt <action> {len(ms)}개 (1 기대)")
    try:
        a = json.loads(ms[0])
    except json.JSONDecodeError as e:
        _abort(f"{src}:{i} action JSON 실패: {e}")
    at = a.get("action")
    if not at:
        _abort(f"{src}:{i} action type 없음")
    return at, (at == "terminate" and a.get("answer") not in (None, "", "null"))


def _given_action_of_state(rec: dict, src: str, i: int) -> str:
    ms = _GIVEN_ACT_RE.findall(_msg(rec, "human"))
    if not ms:
        _abort(f"{src}:{i} state given-action 파싱 실패")
    return ms[0]


# ── image-strip (downstream → XML-only, fail-closed) ─────────────────────────


def strip_action_images(rec: dict, src: str, i: int) -> dict:
    orig_n_img = len(rec.get("images", []))
    img_before = sum(m["value"].count("<image>") for m in rec["messages"])
    if img_before != orig_n_img:
        _abort(f"{src}:{i} <image> {img_before} != images {orig_n_img}")
    new_msgs = []
    for m in rec["messages"]:
        v = m["value"]
        subs = (
            _SYS_SUBS
            if m["from"] == "system"
            else _HUM_SUBS
            if m["from"] == "human"
            else {}
        )
        for key, repl in subs.items():
            if v.count(key) != 1:
                _abort(
                    f"{src}:{i} {m['from']} 치환키 {key[:40]!r} {v.count(key)}회 (1 기대)"
                )
            v = v.replace(key, repl)
        new_msgs.append({**m, "value": v})
    if sum(m["value"].count("<image>") for m in new_msgs) != 0:
        _abort(f"{src}:{i} 변환 후 <image> 잔존")
    if any("[Screenshot]" in m["value"] for m in new_msgs):
        _abort(f"{src}:{i} 변환 후 [Screenshot] 잔존")
    return {"messages": new_msgs, "images": []}


# ── 길이 필터 ────────────────────────────────────────────────────────────────


def filter_pool_by_length(pool, length_of, media_dir, cutoff, *, label):
    """이미지-보유 형태의 mm-expanded 길이 > cutoff 인 레코드를 pool 에서 제거.

    pool = [(rec, ...) ...] 튜플 리스트. rec 은 원본(myset 경로)이므로 remap 후 길이를
    잰다 (state=이미지유지, downstream=stage2 이미지-보유 형태 = 더 긴 쪽 → stage1-down
    stripped 은 이보다 짧아 함께 보장된다). 이미지 열기 실패 레코드는 보수적으로 제외한다.
    풀에서 제거만 하므로 누출 0·층화 카운트 불변식은 그대로 유지된다.
    """
    kept: list = []
    dropped: list[int] = []
    for t in pool:
        L = length_of(remap_record(dict(t[0])), media_dir)
        if L is None or L > cutoff:
            dropped.append(L if L is not None else -1)
        else:
            kept.append(t)
    print(
        f"[len-filter] {label}: {len(pool)} → keep {len(kept)}, "
        f"drop {len(dropped)} (mm-expanded > {cutoff})"
    )
    return kept, dropped


# ── 층화 배분 ────────────────────────────────────────────────────────────────


def _largest_remainder(marginal: dict[str, int], target: int) -> dict[str, int]:
    types = sorted(marginal)
    total = sum(marginal.values())
    raw = {t: target * marginal[t] / total for t in types}
    alloc = {t: int(raw[t]) for t in types}
    rem = target - sum(alloc.values())
    for t in sorted(types, key=lambda t: (-(raw[t] - alloc[t]), t))[:rem]:
        alloc[t] += 1
    return alloc


def _episode_roundrobin(items: list, rng: random.Random, ep_of) -> list:
    by_ep: dict[str, list] = defaultdict(list)
    for it in items:
        by_ep[ep_of(it)].append(it)
    eps = sorted(by_ep)
    for ep in eps:
        rng.shuffle(by_ep[ep])
    ep_order = eps[:]
    rng.shuffle(ep_order)
    out: list = []
    pos = {ep: 0 for ep in eps}
    remaining = sum(len(v) for v in by_ep.values())
    while remaining > 0:
        for ep in ep_order:
            if pos[ep] < len(by_ep[ep]):
                out.append(by_ep[ep][pos[ep]])
                pos[ep] += 1
                remaining -= 1
    return out


def sample_state(pool: list[tuple], target: int, seed: int) -> tuple[list[dict], dict]:
    """pool=[(rec, given_action, episode)] → target 층화 샘플 (action 층 × episode 라운드로빈)."""
    rng = random.Random(seed)
    by_act: dict[str, list[tuple]] = defaultdict(list)
    for rec, act, ep in pool:
        by_act[act].append((rec, ep))
    marginal = {a: len(v) for a, v in by_act.items()}
    alloc = _largest_remainder(marginal, target)
    selected: list[dict] = []
    realized: Counter = Counter()
    for act in sorted(by_act):
        k = alloc[act]
        recs = by_act[act]
        if k > len(recs):
            _abort(f"state 층 {act!r}: 배분 {k} > 재고 {len(recs)}")
        ordered = _episode_roundrobin(recs, rng, ep_of=lambda x: x[1])
        selected.extend(rec for rec, _ in ordered[:k])
        realized[act] = k
    return selected, {
        "target": target,
        "source_marginal": dict(sorted(marginal.items())),
        "realized": dict(sorted(realized.items())),
    }


def sample_downstream_joint(
    pool: list[tuple], aug: list[dict], seed: int
) -> tuple[list[dict], list[dict], dict]:
    """with_history 풀 → stage1-down(10K, 이미지제거) + stage2(15K, 이미지유지) 비중복 분할."""
    rng = random.Random(seed)

    def ep_of(t):
        return t[2]

    by_act: dict[str, list[tuple]] = defaultdict(list)
    for item in pool:
        by_act[item[1]].append(item)
    marginal = {a: len(v) for a, v in by_act.items()}
    q1 = _largest_remainder(marginal, N_S1_DOWN)
    q2 = _largest_remainder(marginal, N_S2)

    aug_shuf = aug[:]
    rng.shuffle(aug_shuf)
    if len(aug_shuf) < 2 * AUG_PER_STAGE:
        _abort(f"open 증강 {len(aug_shuf)} < {2 * AUG_PER_STAGE}")
    aug1, aug2 = aug_shuf[:AUG_PER_STAGE], aug_shuf[AUG_PER_STAGE : 2 * AUG_PER_STAGE]

    s1: list[dict] = []
    s2: list[dict] = []
    s1_keys: set[int] = set()
    s2_keys: set[int] = set()
    r1: Counter = Counter()
    r2: Counter = Counter()
    down_meta_ansterm: dict = {}

    def emit(t1: list[tuple], t2: list[tuple], act: str) -> None:
        for it in t1:
            s1.append(strip_action_images(remap_record(dict(it[0])), SRC_DOWN, it[4]))
            s1_keys.add(it[4])
            r1[act] += 1
        for it in t2:
            s2.append(remap_record(dict(it[0])))
            s2_keys.add(it[4])
            r2[act] += 1

    def slice_disjoint(items, n1, n2, act):
        if n1 < 0 or n2 < 0:
            _abort(f"층 {act!r}: 음수 쿼터 {n1}/{n2}")
        if n1 + n2 > len(items):
            _abort(f"층 {act!r}: 필요 {n1}+{n2} > 재고 {len(items)}")
        return items[:n1], items[n1 : n1 + n2]

    for act in sorted(by_act):
        bucket = by_act[act]
        if act == "open":
            for rec in aug1:
                s1.append(strip_action_images(remap_record(dict(rec)), "aug", 0))
                r1["open"] += 1
            for rec in aug2:
                s2.append(remap_record(dict(rec)))
                r2["open"] += 1
            ordered = _episode_roundrobin(bucket, rng, ep_of)
            p1, p2 = slice_disjoint(
                ordered, q1[act] - AUG_PER_STAGE, q2[act] - AUG_PER_STAGE, act
            )
            emit(p1, p2, act)
        elif act == "terminate":
            ans = [it for it in bucket if it[3]]
            noans = [it for it in bucket if not it[3]]
            # answer-terminate 를 각 stage 에 절반씩 우선 배치(가용분·쿼터에 맞춰 동적).
            per = min(len(ans) // 2, q1[act], q2[act])
            rng.shuffle(ans)
            ans1, ans2 = ans[:per], ans[per : 2 * per]
            ordered = _episode_roundrobin(noans, rng, ep_of)
            r_1, r_2 = slice_disjoint(ordered, q1[act] - per, q2[act] - per, act)
            emit(ans1 + r_1, ans2 + r_2, act)
            down_meta_ansterm["per_stage"] = per
            down_meta_ansterm["available"] = len(ans)
        else:
            ordered = _episode_roundrobin(bucket, rng, ep_of)
            p1, p2 = slice_disjoint(ordered, q1[act], q2[act], act)
            emit(p1, p2, act)

    overlap = s1_keys & s2_keys
    if overlap:
        _abort(f"비중복 위반: stage1-down ∩ stage2 = {len(overlap)}")
    return (
        s1,
        s2,
        {
            "s1_alloc": dict(sorted(q1.items())),
            "s2_alloc": dict(sorted(q2.items())),
            "s1_realized": dict(sorted(r1.items())),
            "s2_realized": dict(sorted(r2.items())),
            "priority": {
                "aug_open_per_stage": AUG_PER_STAGE,
                "answer_terminate_per_stage": down_meta_ansterm.get("per_stage"),
                "answer_terminate_available": down_meta_ansterm.get("available"),
            },
        },
    )


# ── I/O ────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 방어: path 가 심링크면 반드시 먼저 제거 — open("w") 가 심링크를 따라가 타깃(EXP05)
    # 을 덮어쓰는 사고를 막는다. 실파일은 정상 truncate.
    if path.is_symlink():
        path.unlink()
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── build ──────────────────────────────────────────────────────────────────


def build(
    source_dir: Path,
    out_dir: Path,
    exp05_dir: Path,
    model: str,
    revision: str | None,
    seed: int,
) -> dict:
    from preprocess_dataset_v2 import preprocess  # noqa: PLC0415

    for p in (source_dir / SRC_STATE, source_dir / SRC_DOWN, source_dir / SRC_AUG):
        if not p.is_file():
            _abort(f"소스 없음: {p}")

    # ── EXP05 test 키 로드 (state / stage2, id/ood) → test_union ─────────────
    state_keys = {
        sp: _keys_from_file(exp05_dir / EXP05_STATE[sp]) for sp in ("id", "ood")
    }
    s2_keys = {sp: _keys_from_file(exp05_dir / EXP05_S2[sp]) for sp in ("id", "ood")}
    test_union = set().union(*state_keys.values(), *s2_keys.values())
    print(
        f"[test-keys] state id={len(state_keys['id'])} ood={len(state_keys['ood'])} | "
        f"stage2 id={len(s2_keys['id'])} ood={len(s2_keys['ood'])} | union={len(test_union)}"
    )

    # ── state 풀 (train-eligible = key ∉ union) + state test ────────────────
    state_pool: list[tuple] = []
    state_test = {"id": [], "ood": []}
    for i, rec in enumerate(_read_jsonl(source_dir / SRC_STATE)):
        if len(rec.get("images", [])) != 1:
            _abort(f"state:{i} images != 1")
        k = _step_key(rec)
        if k in state_keys["id"]:
            state_test["id"].append(remap_record(dict(rec)))
        elif k in state_keys["ood"]:
            state_test["ood"].append(remap_record(dict(rec)))
        elif k not in test_union:
            state_pool.append(
                (
                    rec,
                    _given_action_of_state(rec, "state", i),
                    _episode_of(rec, "state", i),
                )
            )
        # else: union(=s2 test 화면)에만 걸린 state 레코드 → train/test 어디에도 안 씀(누출 방지)

    # ── downstream 풀 + stage2/action test ──────────────────────────────────
    down_pool: list[tuple] = []
    s2_test = {"id": [], "ood": []}
    action_test = {"id": [], "ood": []}
    for i, rec in enumerate(_read_jsonl(source_dir / SRC_DOWN)):
        if len(rec.get("images", [])) != 1:
            _abort(f"down:{i} images != 1")
        k = _step_key(rec)
        sp = "id" if k in s2_keys["id"] else "ood" if k in s2_keys["ood"] else None
        if sp is not None:
            s2_test[sp].append(remap_record(dict(rec)))  # 이미지 유지
            action_test[sp].append(
                strip_action_images(remap_record(dict(rec)), SRC_DOWN, i)
            )  # 이미지 제거
        elif k not in test_union:
            at, ans = _action_of_gpt(rec, "down", i)
            down_pool.append((rec, at, _episode_of(rec, "down", i), ans, i))
        # else: state test 화면에만 걸린 down 레코드 → 제외
    aug = _read_jsonl(source_dir / SRC_AUG)

    # 누출-free 불변식 (하드 제약, advisor tier-2): train 풀 어디에도 test_union 키가
    # 있으면 안 된다. image-stripped 10k downstream 은 산출물에서 (ep,step) 키를 못 뽑으므로
    # (images=[]) 산출물 검증만으로는 그 10k 의 누출을 못 잡는다 — 풀 단계에서 못박는다.
    for rec, *_ in state_pool:
        if _step_key(rec) in test_union:
            _abort("누출: state_pool 에 test_union 키 잔존")
    for rec, *_ in down_pool:
        if _step_key(rec) in test_union:
            _abort("누출: down_pool 에 test_union 키 잔존 (10k stripped 포함)")

    print(
        f"[pool] state(train-eligible)={len(state_pool)}  down(train-eligible)={len(down_pool)}  aug={len(aug)}"
    )
    print(
        f"[test] state id/ood={len(state_test['id'])}/{len(state_test['ood'])}  "
        f"stage2 id/ood={len(s2_test['id'])}/{len(s2_test['ood'])}  "
        f"action id/ood={len(action_test['id'])}/{len(action_test['ood'])}"
    )

    # ── mm-expanded 길이 필터 (cutoff 초과 이미지-보유 샘플 제거) ─────────────
    # 학습 dataloader 가 죽는 근본 원인(길이>cutoff → image_grid_thw vs 잘린 input_ids
    # 불일치)을 build 시점에 원천 차단한다. 풀에서 제거만 하므로 누출 0 은 유지되고,
    # 40000/10000/15000 은 필터된 풀에서 채운다 (풀에 충분한 여유가 있어야 함 — 없으면
    # sample_* 가 fail-closed 로 죽는다). test 는 무손실이라 필터하지 않는다.
    from filter_long_samples import build_length_fn  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415

    _proc = AutoProcessor.from_pretrained(
        model, revision=revision, trust_remote_code=True
    )
    _length_of = build_length_fn(
        _proc, image_max_pixels=IMG_MAX_PIXELS, image_min_pixels=IMG_MIN_PIXELS
    )
    _media_dir = (
        out_dir.parent
    )  # remap 경로 "AndroidControl/images/..." 의 기준 (=data/)
    state_pool, state_drop = filter_pool_by_length(
        state_pool, _length_of, _media_dir, CUTOFF_LEN, label="state"
    )
    down_pool, down_drop = filter_pool_by_length(
        down_pool, _length_of, _media_dir, CUTOFF_LEN, label="downstream"
    )

    # ── 샘플링 ──────────────────────────────────────────────────────────────
    state_sel, state_meta = sample_state(state_pool, N_STATE, seed)
    s1_down, s2_sel, down_meta = sample_downstream_joint(down_pool, aug, seed)
    for nm, got, exp in [
        ("state", len(state_sel), N_STATE),
        ("s1_down", len(s1_down), N_S1_DOWN),
        ("s2", len(s2_sel), N_S2),
    ]:
        if got != exp:
            _abort(f"{nm} 샘플 {got} != {exp}")
    print(f"[sample] state={len(state_sel)}  s1_down={len(s1_down)}  s2={len(s2_sel)}")

    # ── stage1 train (state remap 유지 + down strip) → 가중 ─────────────────
    combined = [remap_record(dict(r)) for r in state_sel] + s1_down
    interim = out_dir / (TRAIN1_NAME + ".sampled")
    _write_jsonl(interim, combined)
    train1 = out_dir / TRAIN1_NAME
    print(f"\n[weight] diff v2 (UNCHANGED={W_UNCHANGED}) → {TRAIN1_NAME}")
    pp_meta = preprocess(
        input_jsonl=str(interim),
        output_jsonl=str(train1),
        model_name=model,
        w_added=W_ADDED,
        w_modified=W_MODIFIED,
        w_unchanged=W_UNCHANGED,
        metric_version=METRIC_VERSION,
        revision=revision,
        on_error="fail",
    )
    interim.unlink()
    dt = pp_meta["diff_totals"]
    if not (dt["ADDED"] > 0 and dt["MODIFIED"] > 0 and dt["UNCHANGED"] > 0):
        _abort(f"diff-loss degenerate: {dt}")

    # ── stage2 train ────────────────────────────────────────────────────────
    _write_jsonl(out_dir / TRAIN2_NAME, s2_sel)
    print(f"[stage2] {TRAIN2_NAME} → {len(s2_sel)}행")

    # ── test 6종 (EXP07 자체 파일) ──────────────────────────────────────────
    for sp in ("id", "ood"):
        _write_jsonl(out_dir / TEST_STATE[sp], state_test[sp])
        _write_jsonl(out_dir / TEST_S2[sp], s2_test[sp])
        _write_jsonl(out_dir / TEST_ACTION[sp], action_test[sp])
    print(
        f"[test] 6종 기록: state {len(state_test['id'])}/{len(state_test['ood'])}, "
        f"stage2 {len(s2_test['id'])}/{len(s2_test['ood'])}, action {len(action_test['id'])}/{len(action_test['ood'])}"
    )

    # ── sidecar ─────────────────────────────────────────────────────────────
    meta_path = train1.with_name(train1.name + ".meta.json")
    merged = json.loads(meta_path.read_text(encoding="utf-8"))
    merged["exp07_sampling"] = {
        "source_dir": str(source_dir),
        "seed": seed,
        "sizes": {
            "stage1_state": N_STATE,
            "stage1_downstream": N_S1_DOWN,
            "stage2": N_S2,
        },
        "state": state_meta,
        "downstream_joint": down_meta,
        "test_keys": {
            "state_id": len(state_keys["id"]),
            "state_ood": len(state_keys["ood"]),
            "s2_id": len(s2_keys["id"]),
            "s2_ood": len(s2_keys["ood"]),
            "union": len(test_union),
        },
        "test_counts": {
            "state_id": len(state_test["id"]),
            "state_ood": len(state_test["ood"]),
            "s2_id": len(s2_test["id"]),
            "s2_ood": len(s2_test["ood"]),
            "action_id": len(action_test["id"]),
            "action_ood": len(action_test["ood"]),
        },
        "leakage_free": "EXP05 test 키 union 을 train 두 풀에서 제외 (EXP07 train ∩ test = 0)",
        "length_filter": {
            "cutoff_len": CUTOFF_LEN,
            "image_max_pixels": IMG_MAX_PIXELS,
            "image_min_pixels": IMG_MIN_PIXELS,
            "state_dropped": len(state_drop),
            "downstream_dropped": len(down_drop),
            "note": "mm-expanded > cutoff_len 인 이미지-보유 샘플을 풀에서 제외 (샘플링 전)",
        },
    }
    meta_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return {
        "state": state_meta,
        "downstream": down_meta,
        "preprocess": pp_meta,
        "test_union": test_union,
        "state_keys": state_keys,
        "s2_keys": s2_keys,
    }


# ── dataset_info (train 2 + test 6 = 8키, 전부 EXP07 자체) ────────────────────


def _entry(rel_path: str) -> dict:
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


def register_dataset_info(info_path: Path) -> list[str]:
    e = f"../../data/{OUT_SUBDIR}"
    entries = {
        "IWM-AC_EXP07_stage1_train": _entry(f"{e}/{TRAIN1_NAME}"),
        "IWM-AC_EXP07_stage2_train": _entry(f"{e}/{TRAIN2_NAME}"),
        "IWM-AC_EXP07_stage1_test_id_state": _entry(f"{e}/{TEST_STATE['id']}"),
        "IWM-AC_EXP07_stage1_test_ood_state": _entry(f"{e}/{TEST_STATE['ood']}"),
        "IWM-AC_EXP07_stage1_test_id_action": _entry(f"{e}/{TEST_ACTION['id']}"),
        "IWM-AC_EXP07_stage1_test_ood_action": _entry(f"{e}/{TEST_ACTION['ood']}"),
        "IWM-AC_EXP07_stage2_test_id": _entry(f"{e}/{TEST_S2['id']}"),
        "IWM-AC_EXP07_stage2_test_ood": _entry(f"{e}/{TEST_S2['ood']}"),
    }
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for k, v in entries.items():
        info[k] = v
    info_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return sorted(entries)


def cleanup_legacy_symlinks(out_dir: Path) -> None:
    """과거 잠정본이 남긴 EXP05 심링크(4종) 제거 — 이제 EXP07 자체 test 파일이 정본."""
    for name in LEGACY_SYMLINKS:
        p = out_dir / name
        if p.is_symlink():
            p.unlink()
            print(f"[cleanup] 심링크 제거 {name}")


# ── fail-closed 검증 ────────────────────────────────────────────────────────


def verify(out_dir: Path, res: dict) -> int:
    problems: list[str] = []
    train_keys: set = set()

    # (1) stage1 train
    t1 = out_dir / TRAIN1_NAME
    n = ns = nd = deg = 0
    with t1.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            w = set(r.get("token_weights", []))
            ni = len(r["images"])
            if ni == 1:
                ns += 1
                if not w <= {W_UNCHANGED, W_ADDED, W_MODIFIED}:
                    problems.append(f"stage1 {i}: state w {sorted(w)}")
                if w in ({W_UNCHANGED}, {W_ADDED}):
                    deg += 1
                if not r["images"][0].startswith("AndroidControl/images/"):
                    problems.append(f"stage1 {i}: 미remap")
                k = _step_key(r)
                if k:
                    train_keys.add(k)
            elif ni == 0:
                nd += 1
                if w and w != {1.0}:
                    problems.append(f"stage1 {i}: down w {sorted(w)}")
                if sum(m["value"].count("<image>") for m in r["messages"]):
                    problems.append(f"stage1 {i}: down <image> 잔존")
            else:
                problems.append(f"stage1 {i}: images {ni}")
    if n != N_STATE + N_S1_DOWN:
        problems.append(f"stage1 총 {n}")
    if ns != N_STATE:
        problems.append(f"stage1 state {ns}")
    if nd != N_S1_DOWN:
        problems.append(f"stage1 down {nd}")
    if deg / max(ns, 1) >= MAX_DEGENERATE_FRAC:
        problems.append(f"stage1 degenerate {deg / max(ns, 1):.2%}")

    # (2) stage2 train
    t2 = out_dir / TRAIN2_NAME
    m = 0
    with t2.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            m += 1
            if set(r.keys()) != {"messages", "images"}:
                problems.append(f"stage2 {i}: 키 {sorted(r.keys())}")
            if len(r["images"]) != 1 or not r["images"][0].startswith(
                "AndroidControl/images/"
            ):
                problems.append(f"stage2 {i}: 이미지")
            k = _step_key(r)
            if k:
                train_keys.add(k)
    if m != N_S2:
        problems.append(f"stage2 총 {m}")

    # (3) 누출 0: train_keys ∩ test_union == 0
    leak = train_keys & res["test_union"]
    if leak:
        problems.append(f"누출: train ∩ test_union = {len(leak)}")

    # (4) test 파일 계약 + 키 정합
    def check_test(names, key_src, keep_img, label):
        for sp in ("id", "ood"):
            p = out_dir / names[sp]
            cnt = 0
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cnt += 1
                    r = json.loads(line)
                    if keep_img:
                        if len(r["images"]) != 1 or not r["images"][0].startswith(
                            "AndroidControl/images/"
                        ):
                            problems.append(f"{label}/{sp}: 이미지")
                            break
                    else:
                        if r["images"] != [] or sum(
                            mm["value"].count("<image>") for mm in r["messages"]
                        ):
                            problems.append(f"{label}/{sp}: strip 위반")
                            break
            if cnt == 0:
                problems.append(f"{label}/{sp}: 0행")

    check_test(TEST_STATE, res["state_keys"], True, "state_test")
    check_test(TEST_S2, res["s2_keys"], True, "stage2_test")
    check_test(TEST_ACTION, res["s2_keys"], False, "action_test")

    print("\n── fail-closed 검증 ─────────────────────────────────────")
    print(f"  (1) stage1 {n} = state {ns}(deg {deg / max(ns, 1):.2%}) + down {nd}")
    print(f"  (2) stage2 {m}")
    print(f"  (3) 누출 train ∩ test_union = {len(leak)}  (0 이어야 함)")
    print("  (4) test 6종 계약")
    if problems:
        print(f"  [FAIL] {len(problems)}건")
        for p in problems[:15]:
            print(f"    - {p}")
        return 1
    print("  [OK] 전 계약 만족 (누출 0 포함)")
    return 0


# ── main ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="AC_EXP07 train+test 빌드 (신규 0725 소스, 누출 0)"
    )
    p.add_argument(
        "--source-dir", type=Path, default=PROJ / "data" / SRC_DEFAULT_SUBDIR
    )
    p.add_argument("--data-root", type=Path, default=PROJ / "data")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args(argv)

    out_dir: Path = args.data_root / OUT_SUBDIR
    exp05_dir: Path = args.data_root / EXP05_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[src] {args.source_dir}\n[out] {out_dir}\n[seed] {args.seed}\n")
    # ★ 심링크 먼저 제거 — build 가 test 파일을 쓸 때 EXP05 향 심링크를 따라가
    #   타깃을 덮어쓰는 사고를 원천 차단한다 (_write_jsonl 방어와 이중 안전).
    cleanup_legacy_symlinks(out_dir)
    res = build(
        args.source_dir, out_dir, exp05_dir, args.model, args.revision, args.seed
    )

    keys = register_dataset_info(PROJ / "configs" / "lf_dataset" / "dataset_info.json")
    print(f"\n[register] dataset_info {len(keys)}키(train2+test6): {keys}")

    rc = verify(out_dir, res)
    print(f"\nDone. → {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
