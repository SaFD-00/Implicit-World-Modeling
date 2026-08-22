#!/usr/bin/env python3
"""AC_EXP08 stage1/stage2 train+test 데이터 정본 빌더 — 3-포맷 관측성 분할 + diff loss.

이 스크립트가 **AC_EXP08 train/test jsonl 의 유일한 커밋된 생성 경로**다.
소스는 조병웅 제공 0822 필터링본이며, 공유 원본 디렉토리 ``data/AndroidControl/`` 에
명명규칙(ARCHITECTURE §3 규칙 2 — 공유 디렉토리의 변형본은 ``EXP{NN}_`` 접두)대로
평탄하게 놓인다. 두 파일에서 파생한다:

  EXP08_stage1_state.jsonl   60,871  state-pred (NEXT_STATE_PREDICTION, img1, gpt=XML)
  EXP08_stage2.jsonl         86,431  downstream (NEXT_ACTION_PREDICTION, img1,
                                     gpt=<thought><action>, **이미 img-first**)

산출 (``data/AndroidControl_EXP08/``):
  train  stage1_train.jsonl  50,000 = state 40,000(3-포맷·가중) + downstream 10,000(균일 1.0)
         stage2_train.jsonl  15,000 (무가중)
  test   stage1_test_state_full.jsonl     500  ┐ **같은 500 원본**을 세 포맷으로 각각 변환.
         stage1_test_state_masked.jsonl   500  │ 포맷 간 난이도 교란을 없애려는 설계라
         stage1_test_state_dropped.jsonl  500  ┘ 세 파일의 sample_id 집합이 동일하다.
         stage1_test_action.jsonl         500  (downstream)
         stage2_test.jsonl                500  (downstream, stage1_test_action 과 disjoint)

  감사용 중간 산출물 (``data/AndroidControl_EXP08/_build/``):
         state_train_split_raw.jsonl / _applied.jsonl / _weighted.jsonl 등

EXP07 과 무엇이 같고 무엇이 다른가
-----------------------------------
같다: 이미지 remap 규약, 좌표계(절대 픽셀 840x1876 / budget 1,605,632), ``cutoff_len``
24576 길이 필터, 95% 복사-편향 필터, 8:2 (state 40K : downstream 10K) 구성, stage2 15K.

다르다:
1. **3-포맷 관측성 분할** (full 25% / masked 55% / dropped 20%) 을 state 학습분에 적용한다.
   변환 정본은 ``scripts/build_wm_formats.py`` 이고 불변식 검증은
   ``scripts/validate_wm_formats.py`` (C1~C11) 다. anti-copy 설계 배경은
   ``docs/WM_FORMATS.md``.
2. **diff 는 raw 로, 학습은 applied 로.** 마스킹된 current XML 로 헝가리안을 돌리면
   가려진 요소가 next 에서 ADDED 로 오분류된다. 두 파일을 ``sample_id`` 로 조인해
   ``token_weights`` 를 붙이는 것이 ``scripts/diff_loss/build_diff_targets.py`` 다.
3. **파서 스키마가 다르다** — Cerebra html-like (``data-bbox="x1 y1 x2 y2"`` / ``aria-label``).
   그래서 헝가리안·가중치 모듈은 ``*_v2c.py`` (v2 의 Cerebra 확장 복제본) 를 쓴다.
   ``*_v2.py`` 는 EXP02/05/06/07 재현을 위해 불가침이다 (AGENTS 하드 제약 9).
4. **UNCHANGED 가중치 0.25** (EXP07 v1=0.2, v2=0.05). Cerebra XML 은 토큰의 상당수가
   ``data-bbox`` 좌표라 baseline 을 너무 낮추면 렌더 골격이 무너진다는 근거 (docs/WM_FORMATS.md §4.3).
5. **ID/OOD 구분 없음.** 원본에 앱 파티션 메타가 없어 앱 단위 분할을 재현할 수 없다.
   대신 **에피소드 단위 홀드아웃**으로 train ∩ test = 0 을 보장한다 (state·downstream
   두 소스가 에피소드를 13,967/14,095 공유하므로 step 단위 분리로는 누출이 남는다).
6. **stage1 downstream 의 이미지를 제거하지 않는다.** EXP07 은 image budget 때문에
   XML-only 로 만들었지만, EXP08 소스는 이미 img-first 프롬프트 완전체이고 3-포맷
   설계 자체가 "스크린샷에서 복원하라"를 전제하므로 이미지를 유지한다.

Usage
-----
  python scripts/build_exp08_data.py
  python scripts/build_exp08_data.py --source-dir data/AndroidControl --seed 8
  python scripts/build_exp08_data.py --limit 2000        # 스모크 (작은 목표치로 축소)
  python scripts/build_exp08_data.py --skip-length-filter  # 이미지 없을 때 임시 우회
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
SCRIPTS = PROJ / "scripts"
sys.path.insert(0, str(SCRIPTS))  # filter_long_samples
sys.path.insert(0, str(SCRIPTS / "diff_loss"))  # hungarian_diff_v2c

# ── 상수 ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_SEED = 8

N_STATE = 40000
N_S1_DOWN = 10000
N_S2 = 15000

N_TEST_STATE = 500
N_TEST_ACTION = 500
N_TEST_S2 = 500

# 3-포맷 비율 (build_wm_formats 기본값과 동일 — 여기서 명시해 sidecar 에 기록한다)
RATIO_FULL, RATIO_MASKED, RATIO_DROPPED = 0.25, 0.55, 0.20

# diff loss 가중치 (2026-08-22 사용자 확정: docs/WM_FORMATS.md §4.3 권장값)
W_ADDED, W_MODIFIED, W_UNCHANGED = 1.0, 1.0, 0.25

# 95% 복사-편향 필터 — EXP07 v2 와 동일 임계값.
COPY_FILTER_THR = 0.95
COPY_FILTER_PROCS = 12

# 길이 필터 — EXP07 과 동일. 이 길이를 넘는 이미지-보유 샘플은 학습 dataloader 에서
# 잘리기 *전* 의 image_grid_thw 로 위치를 만들어 vision feature 수와 어긋나 죽는다.
CUTOFF_LEN = 24576
IMG_MAX_PIXELS = 1605632
IMG_MIN_PIXELS = 3136

# 에피소드 홀드아웃 비율 — test 3종(각 500)을 채우고도 남을 만큼만 뗀다.
TEST_EPISODE_FRAC = 0.06

SRC_STATE = "EXP08_stage1_state.jsonl"
SRC_DOWN = "EXP08_stage2.jsonl"
SRC_DEFAULT_SUBDIR = "AndroidControl"
OUT_SUBDIR = "AndroidControl_EXP08"

_ACTION_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)
_GIVEN_ACT_RE = re.compile(r'"action"\s*:\s*"([a-z_]+)"')
_IMG_RE = re.compile(r"^myset/images/episode_(\d+)_step_(\d+)\.jpg$")
_EP_RE = re.compile(r"episode_(\d+)")


def _abort(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


# ── 이미지 remap / 키 ────────────────────────────────────────────────────────


def remap_image(ip: str) -> str:
    """``myset/images/episode_{N}_step_{M}.jpg`` → ``AndroidControl/images/episode_{N:06d}_step_{M}.jpg``.

    EXP08 소스의 episode 번호는 zero-pad 가 없고 공유 이미지 풀은 6자리다. 2026-08-22
    전수 확인: state 60,871 + downstream 86,431 전부 이 규칙으로 실재 파일에 매핑된다.
    """
    if ip == "myset/images/home.jpg":
        return "AndroidControl/images/home.jpg"
    m = _IMG_RE.match(ip)
    if not m:
        _abort(f"이미지 remap 실패(예상 밖 패턴): {ip!r}")
    return f"AndroidControl/images/episode_{int(m.group(1)):06d}_step_{m.group(2)}.jpg"


def remap_record(rec: dict) -> dict:
    rec["images"] = [remap_image(ip) for ip in rec.get("images", [])]
    return rec


def _msg(rec: dict, role: str) -> str:
    return next((m["value"] for m in rec["messages"] if m["from"] == role), "")


def _episode_of(rec: dict, src: str, i: int) -> str:
    imgs = rec.get("images", [])
    if not imgs:
        _abort(f"{src}:{i} images 비어 episode 파싱 불가")
    m = _EP_RE.search(imgs[0])
    if not m:
        _abort(f"{src}:{i} images[0]={imgs[0]!r} episode 파싱 실패")
    return str(int(m.group(1)))  # zero-pad 유무와 무관하게 정규화


def _action_of_gpt(rec: dict, src: str, i: int) -> str:
    ms = _ACTION_RE.findall(_msg(rec, "gpt"))
    if len(ms) != 1:
        _abort(f"{src}:{i} gpt <action> {len(ms)}개 (1 기대)")
    try:
        a = json.loads(ms[0])
    except json.JSONDecodeError as e:
        _abort(f"{src}:{i} action JSON 실패: {e}")
    at = a.get("action")
    if not at:
        _abort(f"{src}:{i} action type 없음")
    return at


def _given_action_of_state(rec: dict, src: str, i: int) -> str:
    ms = _GIVEN_ACT_RE.findall(_msg(rec, "human"))
    if not ms:
        _abort(f"{src}:{i} state given-action 파싱 실패")
    return ms[0]


# ── 필터 ─────────────────────────────────────────────────────────────────────


def filter_pool_by_length(pool, length_of, media_dir, cutoff, *, label):
    """mm-expanded 길이 > cutoff 인 레코드를 pool 에서 제거 (풀에서 제거만 → 누출 불변)."""
    kept, dropped = [], 0
    for t in pool:
        L = length_of(remap_record(dict(t[0])), media_dir)
        if L is None or L > cutoff:
            dropped += 1
        else:
            kept.append(t)
    print(f"[len-filter] {label}: {len(pool)} → keep {len(kept)}, drop {dropped} (>{cutoff})")
    return kept, dropped


def _copy_ratio_of_rec(rec: dict) -> float:
    """current(human) → future(gpt) diff 의 UNCHANGED 비율. **v2c** (Cerebra 스키마) 로 잰다.

    v2 로 재면 ``data-bbox`` 를 못 읽어 위치 신호가 죽고 비율이 통째로 어긋난다.
    """
    from hungarian_diff_v2c import classify_diff, summarize_diff  # noqa: PLC0415

    human = _msg(rec, "human")
    cur = human.split("Current UI State:", 1)[1] if "Current UI State:" in human else human
    cur = cur.split("[Screenshot]", 1)[0].strip()
    d = summarize_diff(classify_diff(cur, _msg(rec, "gpt").strip()))
    tot = d["ADDED"] + d["MODIFIED"] + d["UNCHANGED"]
    return d["UNCHANGED"] / tot if tot else 1.0


def filter_pool_by_copy(pool, thr=COPY_FILTER_THR, *, label, procs=COPY_FILTER_PROCS):
    """UNCHANGED 비율 >= thr 인 (복사가 정답인) 레코드를 제거. (kept, n_dropped) 반환."""
    recs = [t[0] for t in pool]
    if recs:
        with multiprocessing.Pool(procs) as mp:
            ratios = mp.map(_copy_ratio_of_rec, recs, chunksize=64)
    else:
        ratios = []
    kept = [t for t, r in zip(pool, ratios) if r < thr]
    n = len(pool) - len(kept)
    print(f"[copy-filter] {label}: {len(pool)} → keep {len(kept)}, drop {n} (UNCHANGED >= {thr})")
    return kept, n


# ── 층화 샘플링 ──────────────────────────────────────────────────────────────


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
    """에피소드를 라운드로빈으로 훑어 한 에피소드가 표본을 독점하지 않게 한다."""
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


def sample_stratified(pool: list[tuple], target: int, rng: random.Random, *, label: str):
    """pool=[(rec, action, episode)] → action 층화 + 에피소드 라운드로빈 표본."""
    by_act: dict[str, list[tuple]] = defaultdict(list)
    for rec, act, ep in pool:
        by_act[act].append((rec, ep))
    marginal = {a: len(v) for a, v in by_act.items()}
    if sum(marginal.values()) < target:
        _abort(f"{label}: 재고 {sum(marginal.values())} < 목표 {target}")
    alloc = _largest_remainder(marginal, target)
    selected: list[dict] = []
    realized: Counter = Counter()
    for act in sorted(by_act):
        k = min(alloc[act], len(by_act[act]))
        ordered = _episode_roundrobin(by_act[act], rng, ep_of=lambda x: x[1])
        selected.extend(rec for rec, _ in ordered[:k])
        realized[act] = k
    # largest-remainder 가 재고를 넘긴 층이 있으면 남은 몫을 여유 있는 층에서 채운다.
    deficit = target - len(selected)
    if deficit > 0:
        spare = [
            (rec, ep)
            for act in sorted(by_act)
            for rec, ep in by_act[act][realized[act] :]
        ]
        rng.shuffle(spare)
        selected.extend(rec for rec, _ in spare[:deficit])
    if len(selected) != target:
        _abort(f"{label}: 표본 {len(selected)} != 목표 {target}")
    return selected, {
        "target": target,
        "source_marginal": dict(sorted(marginal.items())),
        "realized": dict(sorted(realized.items())),
    }


def sample_disjoint(pool: list[tuple], n1: int, n2: int, rng: random.Random, *, label: str):
    """하나의 downstream 풀에서 **겹치지 않는** 두 표본을 뽑는다 (stage1-down / stage2)."""
    if len(pool) < n1 + n2:
        _abort(f"{label}: 재고 {len(pool)} < {n1}+{n2}")
    first, meta1 = sample_stratified(pool, n1, rng, label=f"{label}-1")
    taken = {id(r) for r in first}
    rest = [t for t in pool if id(t[0]) not in taken]
    second, meta2 = sample_stratified(rest, n2, rng, label=f"{label}-2")
    return first, second, meta1, meta2


# ── I/O ──────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"[write] {path}  ({len(records)} 행)")


def _run(cmd: list[str]) -> None:
    print("[run] " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=PROJ)
    if r.returncode != 0:
        _abort(f"서브커맨드 실패 (rc={r.returncode}): {' '.join(str(c) for c in cmd)}")


# ── 3-포맷 변환 + diff 가중치 ────────────────────────────────────────────────


def make_three_formats(
    records: list[dict],
    workdir: Path,
    prefix: str,
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[Path, Path]:
    """records → build_wm_formats → (raw, applied). 불변식 C1~C11 검증까지 통과시킨다."""
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / f"{prefix}.jsonl"
    _write_jsonl(src, records)
    rf, rm, rd = ratios
    _run([
        sys.executable, SCRIPTS / "build_wm_formats.py",
        "--input", src, "--outdir", workdir, "--prefix", prefix, "--seed", seed,
        "--ratio-full", rf, "--ratio-masked", rm, "--ratio-dropped", rd,
    ])
    raw = workdir / f"{prefix}_split_raw.jsonl"
    applied = workdir / f"{prefix}_split_applied.jsonl"
    _run([
        sys.executable, SCRIPTS / "validate_wm_formats.py",
        "--raw", raw, "--applied", applied,
        "--expect-full", rf, "--expect-masked", rm, "--expect-dropped", rd,
    ])
    return raw, applied


def attach_diff_weights(raw: Path, applied: Path, out: Path, *, model: str,
                        revision: str | None = None) -> Path:
    """raw 로 헝가리안 diff 를 계산해 applied 에 token_weights 를 부착한다.

    ``--revision`` 을 **반드시 고정하라**. 기본값 None 은 Hub HEAD 를 다시 해석하므로
    토크나이저가 바뀌면 token_weights 가 조용히 달라진다 (EXP07 이 같은 함정을 겪었다).
    실제로 해석된 SHA 는 ``<output>.meta.json`` 의 ``revision_resolved`` 에 남는다.
    """
    cmd = [
        sys.executable, SCRIPTS / "diff_loss" / "build_diff_targets.py",
        "--raw", raw, "--applied", applied, "--output", out, "--model", model,
        "--w-added", W_ADDED, "--w-modified", W_MODIFIED, "--w-unchanged", W_UNCHANGED,
        "--on-error", "fail",
    ]
    if revision:
        cmd += ["--revision", revision]
    _run(cmd)
    return out


def attach_raw_current_state(applied: Path, raw: Path) -> list[dict]:
    """applied 레코드에 **마스킹 전 원본** current XML 을 `raw_current_state` 로 실어 반환한다.

    왜 필요한가: copy-bias 진단(`_state_diff_eval.py`)은 current state 를 예측 파일의
    프롬프트에서 뽑는데, masked/dropped 포맷은 그 프롬프트의 current 가 이미 가려져 있다.
    그대로 채점하면 **masked/dropped 가 구조적으로 낮은 copy_rate 를 받아 가짜 개선**이 되고,
    `dropped` 는 current 가 `(none)` 이라 `copy_excess` 가 0 으로 붕괴한다. diff 타깃을 raw 로
    계산해야 하는 것과 같은 이유다 — 채점기는 이 필드가 있으면 프롬프트보다 우선한다.
    """
    raw_by_sid = {r["sample_id"]: r for r in _read_jsonl(raw)}
    out = []
    for rec in _read_jsonl(applied):
        src = _msg(raw_by_sid[rec["sample_id"]], "human")
        if "Current UI State:" not in src or "[Screenshot]" not in src:
            _abort(f"raw human 레이아웃 이상 (sample_id={rec['sample_id']})")
        cur = src.split("Current UI State:", 1)[1].split("[Screenshot]", 1)[0].strip()
        out.append({**rec, "raw_current_state": cur})
    return out


def attach_uniform_weights(records: list[dict], model: str) -> list[dict]:
    """downstream(action) 샘플에 균일 1.0 가중치를 붙인다.

    diff 경로로 흘려보내면 "diff 없음 → 전부 baseline" 으로 잘못 감쇠된다
    (AGENTS 하드 제약 10 의 action-uniform 분기와 같은 이유).
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model)
    out = []
    for r in records:
        asst = _msg(r, "gpt")
        n = len(tok(asst, add_special_tokens=False)["input_ids"])
        out.append({**r, "token_weights": [1.0] * n})
    return out


# ── 빌드 ─────────────────────────────────────────────────────────────────────


def build(args: argparse.Namespace) -> dict:
    src_dir = args.source_dir
    out_dir = args.data_root / OUT_SUBDIR
    work = out_dir / "_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    n_state, n_down, n_s2 = args.n_state, args.n_s1_down, args.n_s2
    n_t_state, n_t_act, n_t_s2 = args.n_test_state, args.n_test_action, args.n_test_s2

    # ── 1. 로드 + 층 정보 ────────────────────────────────────────────────────
    state_raw = _read_jsonl(src_dir / SRC_STATE)
    down_raw = _read_jsonl(src_dir / SRC_DOWN)
    if args.limit:
        state_raw, down_raw = state_raw[: args.limit], down_raw[: args.limit]
    print(f"[load] state={len(state_raw)}  downstream={len(down_raw)}")

    state_all = [
        (r, _given_action_of_state(r, SRC_STATE, i), _episode_of(r, SRC_STATE, i))
        for i, r in enumerate(state_raw)
    ]
    down_all = [
        (r, _action_of_gpt(r, SRC_DOWN, i), _episode_of(r, SRC_DOWN, i))
        for i, r in enumerate(down_raw)
    ]

    # ── 2. 에피소드 단위 홀드아웃 (누출 0 의 근거) ───────────────────────────
    # 두 소스가 에피소드를 대부분 공유하므로 step 단위로 나누면 같은 에피소드의
    # 앞뒤 스텝이 train/test 로 갈라져 누출이 남는다. 에피소드 자체를 가른다.
    all_eps = sorted({ep for _, _, ep in state_all} | {ep for _, _, ep in down_all}, key=int)
    shuffled = all_eps[:]
    rng.shuffle(shuffled)
    n_test_ep = max(1, int(len(shuffled) * TEST_EPISODE_FRAC))
    test_eps = set(shuffled[:n_test_ep])
    print(f"[split] 전체 에피소드 {len(all_eps)} → test {len(test_eps)} / train {len(all_eps)-len(test_eps)}")

    state_tr = [t for t in state_all if t[2] not in test_eps]
    state_te = [t for t in state_all if t[2] in test_eps]
    down_tr = [t for t in down_all if t[2] not in test_eps]
    down_te = [t for t in down_all if t[2] in test_eps]
    print(f"[split] state train/test = {len(state_tr)}/{len(state_te)}   "
          f"downstream train/test = {len(down_tr)}/{len(down_te)}")

    # ── 3. 길이 필터 (train 풀에만 — test 는 무손실 유지) ────────────────────
    len_drop = {"state": 0, "downstream": 0, "skipped": bool(args.skip_length_filter)}
    if not args.skip_length_filter:
        from filter_long_samples import build_length_fn  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        proc = AutoProcessor.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
        length_of = build_length_fn(proc, image_max_pixels=IMG_MAX_PIXELS, image_min_pixels=IMG_MIN_PIXELS)
        media_dir = out_dir.parent  # remap 경로 "AndroidControl/images/..." 의 기준(=data/)
        state_tr, len_drop["state"] = filter_pool_by_length(
            state_tr, length_of, media_dir, CUTOFF_LEN, label="state")
        down_tr, len_drop["downstream"] = filter_pool_by_length(
            down_tr, length_of, media_dir, CUTOFF_LEN, label="downstream")

    # ── 4. 95% 복사-편향 필터 (state 풀 — train/test 양쪽) ───────────────────
    # test 에도 거는 이유: 이 실험의 지표가 복사율이라 "복사가 정답"인 샘플이 섞이면
    # copy_excess 류 지표가 구조적으로 부풀어 anti-copy 효과가 가려진다.
    state_tr, copy_drop_tr = filter_pool_by_copy(state_tr, args.copy_thr, label="state-train")
    state_te, copy_drop_te = filter_pool_by_copy(state_te, args.copy_thr, label="state-test")

    # ── 5. 표본 추출 ─────────────────────────────────────────────────────────
    state_sel, state_meta = sample_stratified(state_tr, n_state, rng, label="state")
    s1_down, s2_train, down_meta1, down_meta2 = sample_disjoint(
        down_tr, n_down, n_s2, rng, label="downstream")

    test_state, test_state_meta = sample_stratified(state_te, n_t_state, rng, label="test-state")
    test_act, test_s2, _, _ = sample_disjoint(down_te, n_t_act, n_t_s2, rng, label="test-downstream")

    # ── 6. 이미지 remap (여기서 한 번에) ─────────────────────────────────────
    for group in (state_sel, s1_down, s2_train, test_state, test_act, test_s2):
        for r in group:
            remap_record(r)

    # ── 7. state 학습분: 3-포맷 변환 → 불변식 검증 → diff 가중치 ────────────
    raw_p, app_p = make_three_formats(
        state_sel, work, "state_train", seed=args.seed,
        ratios=(RATIO_FULL, RATIO_MASKED, RATIO_DROPPED))
    weighted_p = attach_diff_weights(
        raw_p, app_p, work / "state_train_split_weighted.jsonl",
        model=args.model, revision=args.revision)
    state_weighted = _read_jsonl(weighted_p)

    # ── 8. downstream 학습분: 균일 1.0 ───────────────────────────────────────
    s1_down_w = attach_uniform_weights(s1_down, args.model)

    # ── 9. stage1_train = state(가중) + downstream(균일), 섞어서 출력 ────────
    stage1 = state_weighted + s1_down_w
    rng.shuffle(stage1)
    _write_jsonl(out_dir / "stage1_train.jsonl", stage1)
    _write_jsonl(out_dir / "stage2_train.jsonl", s2_train)

    # ── 10. state test: 같은 500 원본을 세 포맷으로 각각 ─────────────────────
    fmt_ratio = {
        "full": (1.0, 0.0, 0.0),
        "masked": (0.0, 1.0, 0.0),
        "dropped": (0.0, 0.0, 1.0),
    }
    for fmt, ratios in fmt_ratio.items():
        raw_t, applied_t = make_three_formats(
            test_state, work, f"test_state_{fmt}", seed=args.seed, ratios=ratios)
        _write_jsonl(out_dir / f"stage1_test_state_{fmt}.jsonl",
                     attach_raw_current_state(applied_t, raw_t))

    _write_jsonl(out_dir / "stage1_test_action.jsonl", test_act)
    _write_jsonl(out_dir / "stage2_test.jsonl", test_s2)

    # ── 11. sidecar ──────────────────────────────────────────────────────────
    wmeta_p = Path(str(weighted_p) + ".meta.json")
    wmeta = json.loads(wmeta_p.read_text()) if wmeta_p.exists() else {}
    res = {
        "source_dir": str(src_dir),
        "seed": args.seed,
        "model": args.model,
        "revision_arg": args.revision,
        "sizes": {
            "stage1_state": n_state, "stage1_downstream": n_down, "stage2": n_s2,
            "test_state": n_t_state, "test_action": n_t_act, "test_stage2": n_t_s2,
        },
        "format_ratio": {"full": RATIO_FULL, "masked": RATIO_MASKED, "dropped": RATIO_DROPPED},
        "weight_map": {"ADDED": W_ADDED, "MODIFIED": W_MODIFIED, "UNCHANGED": W_UNCHANGED},
        "episode_holdout": {
            "total_episodes": len(all_eps),
            "test_episodes": len(test_eps),
            "frac": TEST_EPISODE_FRAC,
            "note": "train ∩ test = 0 (에피소드 단위). 원본에 앱 파티션 메타가 없어 ID/OOD 는 만들지 않는다.",
        },
        "length_filter": {"cutoff_len": CUTOFF_LEN, "image_max_pixels": IMG_MAX_PIXELS,
                          "image_min_pixels": IMG_MIN_PIXELS, **len_drop},
        "copy_filter": {"version": "v2c", "threshold": args.copy_thr,
                        "train_dropped": copy_drop_tr, "test_dropped": copy_drop_te},
        "state": state_meta,
        "downstream_s1": down_meta1,
        "downstream_s2": down_meta2,
        "test_state": test_state_meta,
        "diff_targets_meta": wmeta,
    }
    (out_dir / "stage1_train.jsonl.meta.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    return res


# ── 검증 ─────────────────────────────────────────────────────────────────────


def verify(out_dir: Path, res: dict) -> int:
    """산출물 불변식 검사. 0 = OK."""
    fails: list[str] = []
    files = {
        "stage1_train.jsonl": res["sizes"]["stage1_state"] + res["sizes"]["stage1_downstream"],
        "stage2_train.jsonl": res["sizes"]["stage2"],
        "stage1_test_state_full.jsonl": res["sizes"]["test_state"],
        "stage1_test_state_masked.jsonl": res["sizes"]["test_state"],
        "stage1_test_state_dropped.jsonl": res["sizes"]["test_state"],
        "stage1_test_action.jsonl": res["sizes"]["test_action"],
        "stage2_test.jsonl": res["sizes"]["test_stage2"],
    }
    loaded: dict[str, list[dict]] = {}
    for name, want in files.items():
        p = out_dir / name
        if not p.exists():
            fails.append(f"{name} 없음")
            continue
        recs = _read_jsonl(p)
        loaded[name] = recs
        if len(recs) != want:
            fails.append(f"{name}: {len(recs)} != {want}")

    tr = loaded.get("stage1_train.jsonl", [])
    if tr:
        no_w = sum(1 for r in tr if "token_weights" not in r)
        if no_w:
            fails.append(f"stage1_train: token_weights 없는 행 {no_w}")
        fmt = Counter(r.get("fmt", "<action>") for r in tr)
        print(f"[verify] stage1_train fmt 분포: {dict(fmt)}")
        bad_img = sum(1 for r in tr if any(not i.startswith("AndroidControl/") for i in r.get("images", [])))
        if bad_img:
            fails.append(f"stage1_train: remap 안 된 images {bad_img}행")

    # 세 포맷의 raw_current_state 는 **같은 원본**이어야 한다 (copy 지표의 전제)
    raw_maps = []
    for f in ("full", "masked", "dropped"):
        recs = loaded.get(f"stage1_test_state_{f}.jsonl", [])
        miss = sum(1 for r in recs if not r.get("raw_current_state"))
        if recs and miss:
            fails.append(f"stage1_test_state_{f}: raw_current_state 없는 행 {miss}")
        raw_maps.append({r["sample_id"]: r.get("raw_current_state") for r in recs})
    if all(raw_maps) and not (raw_maps[0] == raw_maps[1] == raw_maps[2]):
        fails.append("test_state 세 포맷의 raw_current_state 가 서로 다르다")

    # test 세 포맷이 같은 sample_id 집합인지 (포맷 간 난이도 교란 제거의 전제)
    sids = [
        {r["sample_id"] for r in loaded.get(f"stage1_test_state_{f}.jsonl", [])}
        for f in ("full", "masked", "dropped")
    ]
    if all(sids) and not (sids[0] == sids[1] == sids[2]):
        fails.append("test_state 세 포맷의 sample_id 집합이 다르다")

    # 학습/평가 에피소드 분리
    def eps_of(recs):
        return {_EP_RE.search(r["images"][0]).group(1) for r in recs if r.get("images")}

    tr_eps = eps_of(tr) | eps_of(loaded.get("stage2_train.jsonl", []))
    te_eps = set()
    for n in ("stage1_test_state_full.jsonl", "stage1_test_action.jsonl", "stage2_test.jsonl"):
        te_eps |= eps_of(loaded.get(n, []))
    overlap = tr_eps & te_eps
    print(f"[verify] train 에피소드 {len(tr_eps)} / test 에피소드 {len(te_eps)} / 교집합 {len(overlap)}")
    if overlap:
        fails.append(f"train ∩ test 에피소드 {len(overlap)}개 (0 이어야 함)")

    for f in fails:
        print(f"[FAIL] {f}")
    print("[verify] " + ("OK" if not fails else f"{len(fails)}건 실패"))
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", type=Path, default=PROJ / "data" / SRC_DEFAULT_SUBDIR)
    p.add_argument("--data-root", type=Path, default=PROJ / "data")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None,
                   help="tokenizer/processor commit SHA 고정. 재현성을 위해 반드시 주라 — "
                        "None 이면 Hub HEAD 를 다시 해석해 token_weights 가 조용히 달라진다. "
                        "실제 해석값은 sidecar 의 diff_targets_meta.revision_resolved 에 남는다")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-state", type=int, default=N_STATE)
    p.add_argument("--n-s1-down", type=int, default=N_S1_DOWN)
    p.add_argument("--n-s2", type=int, default=N_S2)
    p.add_argument("--n-test-state", type=int, default=N_TEST_STATE)
    p.add_argument("--n-test-action", type=int, default=N_TEST_ACTION)
    p.add_argument("--n-test-s2", type=int, default=N_TEST_S2)
    p.add_argument("--copy-thr", type=float, default=COPY_FILTER_THR)
    p.add_argument("--limit", type=int, default=0, help="스모크용: 소스 앞 N 행만 사용")
    p.add_argument("--skip-length-filter", action="store_true",
                   help="이미지가 없는 환경에서 임시 우회 (프로덕션 빌드에서는 쓰지 마라)")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)

    out_dir = args.data_root / OUT_SUBDIR
    if args.verify_only:
        meta = json.loads((out_dir / "stage1_train.jsonl.meta.json").read_text())
        return verify(out_dir, meta)

    res = build(args)
    return verify(out_dir, res)


if __name__ == "__main__":
    raise SystemExit(main())
