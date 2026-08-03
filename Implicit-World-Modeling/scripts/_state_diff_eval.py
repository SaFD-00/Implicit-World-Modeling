#!/usr/bin/env python3
"""Copy-bias 진단 채점기 — next-state 예측이 "예측"인지 "입력 복사"인지 가른다.

왜 별도 지표가 필요한가
-----------------------
`_hungarian_eval` 의 `hungarian_f1` 은 예측 전체를 GT 전체와 맞춘다. 그런데
next-state 는 current state 와 **대부분 겹친다** — 한 번의 클릭으로 화면 전체가
바뀌지는 않으니, GT 요소의 다수는 current 에 이미 있던 것(UNCHANGED)이다.
그래서 current 를 그대로 베껴 내기만 해도 `hungarian_f1` 이 높게 나온다.
모델이 world model 을 배운 것인지 "입력 복사"라는 지름길을 배운 것인지,
기존 지표만으로는 구분할 수 없다.

이 모듈은 **current state 를 세 번째 인자로 받아** 그 구분을 만든다.

지표 설계 — 두 가지 함정을 피한다
---------------------------------
1. `copy_rate` 단독은 해석 불가다. GT 자체가 current 와 대부분 겹치므로 **완벽한
   예측도 copy_rate 가 높다.** 높은 값이 "베꼈다"의 증거가 되지 않는다. 그래서 GT
   기준선을 같은 방식으로 함께 재고(`copy_rate_gt`), 그 초과분
   `copy_excess = copy_rate_pred - copy_rate_gt` 를 판별량으로 쓴다. 0 근처면
   "GT 가 겹치는 만큼만 겹쳤다"이고, 큰 양수면 **바뀌었어야 할 자리까지 베꼈다**이다.

2. diff 부분집합에 대한 precision 은 분모가 정규화되지 않는다. 예측 전체는 수십~수백
   요소인데 GT 의 변경분은 한 자릿수라, precision 상한이 `|diff_gt|/|pred|` 로 묶이고
   그 상한이 행마다 다르다. 평균을 내면 "F1 0.05" 같은 숫자가 나와 전부 실패로
   오독된다. 그래서 **헤드라인은 recall** 이다: `diff_recall` = 실제로 바뀐 GT 요소 중
   몇 개를 예측이 맞혔나. 0~1 범위가 그대로 의미를 갖는다.
   (precision/F1 도 낸다 — 다만 분모를 pred-side diff 로 대칭화한 별도 키다. 아래 참고.)

정본 지표와의 관계 — 층 분해
----------------------------
recall 계열은 `pred ↔ gt` **단 한 번의 Hungarian 매칭**(정본
`compute_hungarian_acc` 와 같은 매칭)을 GT diff 유형별로 쪼갠 것이다. 따라서

    (n_hit_unchanged + n_hit_modified + n_hit_added) / n_gt  ==  hungarian_rec

가 항등식으로 성립한다 (`tests/test_state_diff_eval.py` 가 검증). diff 부분집합만
따로 매칭하면 이 성질이 깨진다 — UNCHANGED 에 붙었어야 할 예측 요소가 MODIFIED 로
재배정되면서 recall 이 부풀기 때문이다. 그래서 부분집합 재매칭을 하지 않는다.

UNCHANGED 판정 기준
-------------------
`diff_loss/hungarian_diff_v2.classify_diff` 는 `match_cost <= 0.05` 를 UNCHANGED 로
본다. 그 임계는 v2 의 pos cost 스케일에서 잡힌 값이라 **index 모드에서는 뜻이 달라진다**
— tag·text 가 같아도 index 만 밀린 요소가 `W_INDEX * |Δidx|/max_idx > 0.05` 로
MODIFIED 가 되어, EXP01 과 EXP05 의 숫자가 조용히 비교 불가가 된다.
여기서는 **mode 독립 기준**을 쓴다: 매칭된 쌍 중 `text_sim == 1.0` 이면 UNCHANGED.
(tag 불일치는 `W_TAG=3.0` 이 두 모드의 threshold 를 모두 넘으므로 애초에 매칭되지
않는다 — 매칭된 쌍은 항상 tag 가 같다.) 위치만 바뀐 재배치는 "새로 지어낸 내용"이
아니므로 UNCHANGED 로 본다. `diff_loss/` 의 학습용 정의와는 의도적으로 다르며,
그쪽은 학습 데이터 생성 경로라 건드리지 않는다.

무엇을 채점 대상에서 빼야 하나
------------------------------
`max_new_tokens` 1024 에서 잘린 leaf 는 채점하지 않는다. 잘린 예측은 요소 수가 줄어
`copy_rate_pred` 를 **과소평가**하는데, 하필 측정하려는 방향으로 편향되어 실제보다
좋게 보인다 — hungarian 계열이 그냥 무효인 것과 달리 한쪽으로 틀린다.

판정은 **예측 토큰 수 실측**이다 (`truncated_reason()`): 정확히 1024 토큰인 행이
전체의 5% 이상이면 절단. 자유 생성은 특정 길이에 몰리지 않으므로 상한에 쌓인 모드가
지문이다. 앞서 쓰던 두 기준이 2026-08-03 전수 실측으로 차례로 반증됐다.

1. **mtime** (`MAX_NEW_TOKENS_FIX_UTC` 이전이면 절단) — 절단은 날짜가 아니라 추론
   실행 경로를 따른다. 같은 2026-05~06 산출물 안에 정상과 절단이 섞여 있다.
2. **문자 길이** (predict 최대가 짧으면 절단) — **양방향으로 틀린다.** EXP02
   `base/on-AC_EXP01-state` 는 predict 최장이 29,286자인데 그 행이 정확히 1024
   토큰이다(꼬리의 공백 수천 개가 문자 수만 부풀렸다) → 절단을 놓친다. 반대로 EXP02
   `lora_world-model/epoch-1` 은 최장 5,567자로 짧지만 절단의 근거는 길이가 아니라
   37.9% 가 정확히 1024 토큰이라는 사실이다.

문자 기준이 더 위험하다 — mtime 의 과잉 거부와 달리 **잘린 것을 통과시킨다.**

가드는 **`score` 진입부와 `_hungarian_eval._write_state_diff` 양쪽이 같은 함수를
부른다.** 백필 스크립트에만 뒀다면 `rebuild_woa_metrics.sh` 경로가 그대로 통과했을
자리다. 판정의 정본도 이 모듈이며 `_compare_site` 와 `rebuild_state_diff_metrics.sh`
가 여기서 가져다 쓴다.

Subcommand
----------
score : prediction jsonl + GT test jsonl 로 state_diff_metrics.json 을 만든다.
        `_hungarian_eval score` 가 같은 계산을 인라인으로 수행하므로, 정규 eval 에서는
        이 CLI 를 따로 부를 필요가 없다 — 백필/재산출용 진입점이다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _hungarian_eval as _he  # noqa: E402
from _prompt_sections import parse_prompt  # noqa: E402

# "그냥 복사" 판정 — pred 와 current 의 hungarian_f1 이 이 값 이상이면 사실상 복사본.
COPY_NEAR_F1 = 0.98

# state 예측이 vllm 기본값 1024 토큰에서 잘리던 버그의 수정 시각 (커밋 6a4b59e).
# **판정에는 쓰지 않는다** — 절단은 날짜가 아니라 추론 실행 경로를 따르며, 이 시각
# 이전에도 절단 없는 leaf 가 많다 (모듈 docstring 의 반증 사례). 길이를 잴 수 없을
# 때의 폴백과 이력 참조용으로만 남긴다. `_compare_site` 가 여기서 import 한다.
MAX_NEW_TOKENS_FIX_UTC = datetime(2026, 7, 28, 23, 38, tzinfo=UTC)

# 구 `max_new_tokens` 기본값. 절단의 지문은 "예측 토큰 수가 정확히 이 값인 행이
# 대량으로 존재한다"이다 — 자유 생성은 특정 값에 몰리지 않는다.
LEGACY_MAX_NEW_TOKENS = 1024
# 그 모드에 몇 할이 몰리면 절단으로 볼 것인가. 2026-08-03 전수 실측의 마진:
#   절단 state leaf 14개  → 0.2913 ~ 0.7116
#   정상 state leaf 19개  → 0.0000 ~ 0.0015
#   비-state leaf 97개    → 0.0000 ~ 0.0030 (미학습 base 의 발산 생성이 상한)
# 두 무리가 두 자릿수 배율로 떨어져 있어 0.05 는 어느 쪽으로도 여유가 크다.
TRUNC_MODE_SHARE = 0.05

# 예측을 만든 모델의 tokenizer 로 세야 한다. `_common.sh:297 MODEL_ID` 와 동기 유지.
_MODEL_ID = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}

# (path, mtime_ns, size) → (모드 행 수, 전체 행 수).
# 같은 leaf 를 여러 번 판정하는 호출자(`_compare_site` 19 사이트)를 위한 것이다.
_MODE_SHARE_CACHE: dict[tuple[str, int, int], tuple[int, int]] = {}
_TOKENIZERS: dict[str, object] = {}


def _model_key_for(path: str) -> str | None:
    """prediction 경로에서 모델 키를 뽑는다 (`outputs/<EXP>/eval/<model>[_변형]/...`).

    `qwen3-vl-8b_ratio37` 처럼 변형 접미사가 붙으므로 prefix 로 맞춘다. 못 찾으면
    None — 호출자가 mtime 폴백으로 넘어간다.
    """
    parts = Path(path).parts
    if "eval" not in parts:
        return None
    idx = parts.index("eval") + 1
    if idx >= len(parts):
        return None
    d = parts[idx]
    for key in sorted(_MODEL_ID, key=len, reverse=True):
        if d == key or d.startswith(key + "_"):
            return key
    return None


def _token_lengths(texts: list[str], model_key: str) -> list[int]:
    """`texts` 각각의 토큰 수. transformers 는 **여기서만** 늦게 로드한다.

    `_hungarian_eval._lazy_deps()` 와 같은 이유다 — 모듈 top-level 에서 import 하면
    채점기 전체가 transformers 에 묶인다. 테스트는 이 함수를 갈아끼운다.
    """
    from transformers import AutoTokenizer

    if model_key not in _TOKENIZERS:
        _TOKENIZERS[model_key] = AutoTokenizer.from_pretrained(
            _MODEL_ID[model_key], local_files_only=True
        )
    tok = _TOKENIZERS[model_key]
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def _mode_share(path: str) -> tuple[int, int] | None:
    """(토큰 수가 정확히 `LEGACY_MAX_NEW_TOKENS` 인 행 수, 전체 행 수). 못 재면 None.

    jsonl 이 50 MB 급이라 **배치 단위로 스트리밍**한다 — 전량을 메모리에 올리지 않는다.
    """
    model_key = _model_key_for(path)
    if model_key is None:
        return None
    try:
        st = Path(path).stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    cached = _MODE_SHARE_CACHE.get(key)
    if cached is not None:
        return cached

    n_mode = n_rows = 0
    batch: list[str] = []

    def _flush() -> None:
        nonlocal n_mode, n_rows
        if not batch:
            return
        lens = _token_lengths(batch, model_key)
        n_rows += len(lens)
        n_mode += sum(1 for n in lens if n == LEGACY_MAX_NEW_TOKENS)
        batch.clear()

    try:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                batch.append(obj.get("predict") or obj.get("output") or "")
                if len(batch) >= 128:
                    _flush()
            _flush()
    except (OSError, UnicodeDecodeError, ImportError, ValueError):
        return None
    if not n_rows:
        return None

    _MODE_SHARE_CACHE[key] = (n_mode, n_rows)
    return n_mode, n_rows


def truncated_reason(*pred_paths: str) -> str | None:
    """절단(1024)된 prediction 이면 사유 문자열, 아니면 None.

    잘린 예측은 element 수가 줄어 `copy_rate_pred` 를 **과소평가**한다 — 하필 이
    지표가 재려는 방향("얼마나 베꼈나")으로 편향되어 실제보다 좋게 보인다.
    hungarian 계열이 그냥 무효인 것과 달리 **한쪽으로 틀리므로** 아예 산출하지 않는다.

    판정은 **토큰 실측**이다: 예측 토큰 수가 정확히 `LEGACY_MAX_NEW_TOKENS` 인 행이
    전체의 `TRUNC_MODE_SHARE` 이상이면 절단. 자유 생성은 특정 길이에 몰리지 않으므로
    "상한에 대량으로 쌓인 모드"가 절단의 지문이다. 틀린 기준 둘이 이미 반증됐다:

    - **mtime 은 틀렸다.** 절단은 날짜가 아니라 추론 실행 경로를 따른다 — 2026-05~06
      산출물 중에도 정상(EXP01 `qwen2.5-vl-7b_ratio73/base`)과 절단이 섞여 있다.
    - **문자 길이는 양방향으로 틀렸다.** false negative: EXP02 `base/on-AC_EXP01-state`
      는 predict 최장이 29,286자인데 그 행의 토큰이 **정확히 1024** 다 (꼬리의 공백
      수천 개가 문자 수만 부풀렸다). false positive: EXP02 `lora_world-model/epoch-1`
      은 최장 5,567자로 짧아 "잘렸다"처럼 보이지만, 실제로 잘린 근거는 길이가 아니라
      **37.9% 가 정확히 1024 토큰**이라는 사실이다.

    `max_tok <= 1024` 를 조건으로 넣으면 안 된다. 디토크나이즈→재토크나이즈는 길이를
    보존하지 않아 절단 leaf 에도 1024 를 몇 토큰 넘긴 행이 **1개** 섞인다 (EXP01
    `ratio73/lora/epoch-3` 1060, EXP02 `epoch-1` 1025, `epoch-2` 1030). 그 세 leaf 는
    1024-토큰 행의 96% 가 루트 태그를 못 닫고 끝나(나머지 행은 0.8%) 절단이 확실하다.

    tokenizer 를 못 구하면(모델 미해석·로컬 캐시 없음) `MAX_NEW_TOKENS_FIX_UTC` 로
    폴백한다 — 문자 기준으로는 절대 물러서지 않는다.

    가드가 **채점 경로 안에** 있어야 하는 이유: 백필 스크립트에만 두면
    `rebuild_woa_metrics.sh` → `_hungarian_eval score` 경로가 그대로 통과해,
    백필이 거부하는 편향된 산출물이 woa sibling 에 생긴다.
    """
    for p in pred_paths:
        if not p:
            continue
        scanned = _mode_share(p)
        if scanned is None:
            # 토큰을 못 쟀다 (파일 없음·빈 파일·모델 미해석·tokenizer 부재).
            try:
                mtime = datetime.fromtimestamp(Path(p).stat().st_mtime, tz=UTC)
            except OSError:
                continue
            if mtime < MAX_NEW_TOKENS_FIX_UTC:
                return (
                    f"{Path(p).name} 의 예측 토큰 수를 못 쟀고 mtime 이 절단(1024) 수정 "
                    f"{MAX_NEW_TOKENS_FIX_UTC:%Y-%m-%d %H:%M} UTC 이전 "
                    f"({mtime:%Y-%m-%d %H:%M} UTC) — copy_rate 가 과소평가된다"
                )
            continue
        n_mode, n_rows = scanned
        share = n_mode / n_rows
        if share >= TRUNC_MODE_SHARE:
            return (
                f"{Path(p).name} 이 절단(1024)됐다 — 예측 {n_rows}행 중 {n_mode}행"
                f"({share:.1%})이 정확히 {LEGACY_MAX_NEW_TOKENS} 토큰 "
                f"(임계 {TRUNC_MODE_SHARE:.0%}): copy_rate 가 과소평가된다"
            )
    return None


# 예측이 닫히지 않은 채 끝났는지 보는 sanity 신호. 절단 leaf 를 사후에 싸게 식별한다.
_ROOT_TAG_RE = re.compile(r"\s*<([A-Za-z][\w.:-]*)")


class StateDiffError(RuntimeError):
    """채점 중단 — 값이 아니라 배선이 깨졌을 때만 낸다."""


# ── diff 분류 ────────────────────────────────────────────────────────────
def classify_diff(
    current_str: str, gt_str: str, match_mode: str = "index"
) -> list[dict]:
    """GT(next state)의 각 요소를 current 대비 UNCHANGED/MODIFIED/ADDED 로 분류한다.

    `diff_loss/hungarian_diff_v2.classify_diff` 의 포팅이다. 그 모듈을 직접 import
    하지 않는 이유는 셋이다: (a) `hungarian_metric_v2` 를 bare import 해서 sys.path
    해킹이 필요하고, (b) munkres 고정이라 scipy 로 통일한 채점 체제와 어긋나며,
    (c) **pos 모드 전용**이라 EXP01~04/MB 의 index HTML 을 넣으면 bounds 가 전부
    빈 문자열이 되어 위치 cost 0 으로 조용히 퇴화한다. 게다가 `diff_loss/` 는 학습
    데이터 생성 경로라 채점 요구로 건드릴 수 없다.
    """
    _he._lazy_deps()
    cur_els = _he.extract_elements(current_str, match_mode)
    gt_els = _he.extract_elements(gt_str, match_mode)

    if not gt_els:
        return []
    if not cur_els:
        return [
            {"gt_idx": i, "diff_type": "ADDED", "text_sim": 0.0}
            for i in range(len(gt_els))
        ]

    pairs, _ = _he._hungarian_match(cur_els, gt_els, match_mode)
    gt_to_cur = {j: i for i, j, _ in pairs}

    out: list[dict] = []
    for j, gt_el in enumerate(gt_els):
        if j not in gt_to_cur:
            out.append({"gt_idx": j, "diff_type": "ADDED", "text_sim": 0.0})
            continue
        sim = _he._text_sim(cur_els[gt_to_cur[j]]["text"], gt_el["text"])
        out.append(
            {
                "gt_idx": j,
                "diff_type": "UNCHANGED" if sim >= 1.0 else "MODIFIED",
                "text_sim": round(sim, 4),
            }
        )
    return out


def summarize_diff(diff_result: list[dict]) -> dict[str, int]:
    counts = {"ADDED": 0, "MODIFIED": 0, "UNCHANGED": 0}
    for d in diff_result:
        counts[d["diff_type"]] += 1
    return counts


# ── 행 단위 채점 ─────────────────────────────────────────────────────────
def _unclosed_root(pred_str: str) -> float:
    """예측이 root tag 를 닫지 않고 끝났으면 1.0. 절단 leaf 의 사후 식별용 sanity."""
    m = _ROOT_TAG_RE.match(pred_str)
    if not m:
        return 1.0
    tail = pred_str.rstrip()
    return 0.0 if tail.endswith(f"</{m.group(1)}>") or tail.endswith("/>") else 1.0


def compute_state_diff(
    pred_str: str, gt_str: str, current_str: str, match_mode: str = "index"
) -> dict:
    """한 행의 state-diff 진단값. 정의되지 않는 지표는 None (평균에서 제외된다).

    반환 키
      diff_recall / added_recall / modified_recall / unchanged_recall
          GT 를 diff 유형으로 층화한 recall. `diff` = MODIFIED + ADDED.
          UNCHANGED 는 높고 diff 는 낮으면 = 복사만 잘한다.
      diff_prec / diff_f1
          precision 분모를 **pred-side diff**(current 와 매칭되지 않은 예측 요소)로
          잡아 대칭화한 값. recall 과 다른 정의이므로 키를 분리해 둔다.
      copy_rate_pred / copy_rate_gt / copy_excess
          예측·GT 가 각각 current 와 겹치는 비율과 그 차이. `copy_excess` 가 판별량.
      copy_exact / copy_near
          예측이 current 와 문자열 완전일치 / hungarian_f1 >= COPY_NEAR_F1 인가.
      n_* : 해석용 원자료 개수. unclosed_root : 절단 sanity.
    """
    _he._lazy_deps()
    zero_counts = {
        "n_pred": 0,
        "n_gt": 0,
        "n_cur": 0,
        "n_gt_added": 0,
        "n_gt_modified": 0,
        "n_gt_unchanged": 0,
        "n_pred_diff": 0,
    }
    undefined = {
        k: None
        for k in (
            "diff_recall",
            "added_recall",
            "modified_recall",
            "unchanged_recall",
            "diff_prec",
            "diff_f1",
            "copy_rate_pred",
            "copy_rate_gt",
            "copy_excess",
        )
    }

    try:
        pred_els = _he.extract_elements(pred_str, match_mode)
        gt_els = _he.extract_elements(gt_str, match_mode)
        cur_els = _he.extract_elements(current_str, match_mode)
    except Exception:
        return {
            **undefined,
            **zero_counts,
            "copy_exact": 0.0,
            "copy_near": 0.0,
            "unclosed_root": _unclosed_root(pred_str),
        }

    counts = {
        "n_pred": len(pred_els),
        "n_gt": len(gt_els),
        "n_cur": len(cur_els),
    }
    row = {
        **undefined,
        **zero_counts,
        **counts,
        "copy_exact": 1.0 if pred_str.strip() == current_str.strip() else 0.0,
        "copy_near": 0.0,
        "unclosed_root": _unclosed_root(pred_str),
    }
    if not gt_els:
        return row

    diff = classify_diff(current_str, gt_str, match_mode)
    by_type: dict[str, set[int]] = {
        "ADDED": set(),
        "MODIFIED": set(),
        "UNCHANGED": set(),
    }
    for d in diff:
        by_type[d["diff_type"]].add(d["gt_idx"])
    diff_idx = by_type["ADDED"] | by_type["MODIFIED"]
    row["n_gt_added"] = len(by_type["ADDED"])
    row["n_gt_modified"] = len(by_type["MODIFIED"])
    row["n_gt_unchanged"] = len(by_type["UNCHANGED"])

    # GT 기준선 — GT 자체가 current 와 겹치는 비율. copy_rate_pred 의 해석 기준.
    row["copy_rate_gt"] = round((len(gt_els) - len(by_type["ADDED"])) / len(gt_els), 4)

    # 정본과 같은 pred↔gt 매칭 한 번. 층 분해의 근거라 부분집합 재매칭을 쓰지 않는다.
    pairs_pg, _ = _he._hungarian_match(pred_els, gt_els, match_mode)
    hit_gt = {j for _, j, _ in pairs_pg}
    pred_to_gt = {i: j for i, j, _ in pairs_pg}

    def _recall(subset: set[int]) -> float | None:
        return round(len(hit_gt & subset) / len(subset), 4) if subset else None

    row["diff_recall"] = _recall(diff_idx)
    row["added_recall"] = _recall(by_type["ADDED"])
    row["modified_recall"] = _recall(by_type["MODIFIED"])
    row["unchanged_recall"] = _recall(by_type["UNCHANGED"])

    # pred ↔ current: 복사량 + pred-side diff 산출
    if pred_els and cur_els:
        pairs_pc, _ = _he._hungarian_match(pred_els, cur_els, match_mode)
        n_copy = len(pairs_pc)
        row["copy_rate_pred"] = round(n_copy / len(pred_els), 4)
        row["copy_excess"] = round(row["copy_rate_pred"] - row["copy_rate_gt"], 4)
        prec_c = n_copy / len(pred_els)
        rec_c = n_copy / len(cur_els)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
        row["copy_near"] = 1.0 if f1_c >= COPY_NEAR_F1 else 0.0
        pred_diff_idx = set(range(len(pred_els))) - {i for i, _, _ in pairs_pc}
    elif pred_els and not cur_els:
        # current 에 요소가 없으면 베낄 것도 없다 — 예측 전체가 pred-side diff.
        row["copy_rate_pred"] = 0.0
        row["copy_excess"] = round(0.0 - row["copy_rate_gt"], 4)
        pred_diff_idx = set(range(len(pred_els)))
    else:
        pred_diff_idx = set()

    row["n_pred_diff"] = len(pred_diff_idx)
    if pred_diff_idx and diff_idx:
        hit = sum(1 for i in pred_diff_idx if pred_to_gt.get(i) in diff_idx)
        prec = hit / len(pred_diff_idx)
        rec = row["diff_recall"] or 0.0
        row["diff_prec"] = round(prec, 4)
        row["diff_f1"] = round(
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0, 4
        )
    return row


# ── 배선 self-test ───────────────────────────────────────────────────────
# `_he` 의 bs4/솔버는 지연 로드다. 초기화 없이 부르면 `extract_elements` 가 예외를 내고
# `compute_hungarian_acc` 의 except 가 그걸 삼켜 **전 행 0점**을 조용히 돌려준다
# (2026-08-01 실측: 표본 f1 0.0 vs aggregate 0.71). 행 단위로는 구분할 수 없으므로
# 데이터와 무관한 고정 XML 로 배선만 검사한다.
_PROBE = {
    "index": {
        "cur": '<node index="0"><button index="1" aria-label="OK"/>'
        '<p index="2">hello</p></node>',
        "gt": '<node index="0"><button index="1" aria-label="OK"/>'
        '<p index="2">world</p><p index="3">brand new</p></node>',
    },
    "pos": {
        "cur": '<node bounds="[0,0][10,10]" point="[5,5]">'
        '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
        '<p bounds="[6,6][9,9]" point="[7,7]">hello</p></node>',
        "gt": '<node bounds="[0,0][10,10]" point="[5,5]">'
        '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
        '<p bounds="[6,6][9,9]" point="[7,7]">world</p>'
        '<p bounds="[6,20][9,24]" point="[7,22]">brand new</p></node>',
    },
}


def assert_scorer_wired(match_mode: str) -> None:
    """표본 채점 전에 채점기가 실제로 동작하는지 한 번 확인한다."""
    _he._lazy_deps()
    probe = _PROBE[match_mode]
    els = _he.extract_elements(probe["cur"], match_mode)
    if not els:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            "probe XML 에서 element 0개 — bs4/scipy 의존성을 확인하세요."
        )
    # current 를 그대로 예측 = 복사. diff 를 하나도 못 맞히고 copy_excess 가 양수여야 한다.
    copied = compute_state_diff(probe["cur"], probe["gt"], probe["cur"], match_mode)
    perfect = compute_state_diff(probe["gt"], probe["gt"], probe["cur"], match_mode)
    if not (copied["copy_rate_pred"] == 1.0 and copied["copy_excess"] > 0):
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"복사 probe 가 copy_rate={copied['copy_rate_pred']} "
            f"copy_excess={copied['copy_excess']} — 1.0 / 양수 여야 합니다."
        )
    if perfect["diff_recall"] != 1.0:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"정답 probe 의 diff_recall={perfect['diff_recall']} — 1.0 이어야 합니다."
        )


# ── 집계 ─────────────────────────────────────────────────────────────────
# None 은 "그 행에서 정의되지 않음"이다 (예: GT 에 ADDED 요소가 없는 행의 added_recall).
# 0 으로 세면 평균이 아래로 끌려가 실제보다 나쁘게 보인다. 평균의 분모가 되는 행 수를
# 함께 기록해 몇 행 위에서 잰 값인지 드러낸다.
_MEAN_KEYS = [
    "diff_recall",
    "added_recall",
    "modified_recall",
    "unchanged_recall",
    "diff_prec",
    "diff_f1",
    "copy_rate_pred",
    "copy_rate_gt",
    "copy_excess",
]
_RATE_KEYS = ["copy_exact", "copy_near", "unclosed_root"]
_COUNT_KEYS = [
    "n_pred",
    "n_gt",
    "n_cur",
    "n_gt_added",
    "n_gt_modified",
    "n_gt_unchanged",
    "n_pred_diff",
]


def aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    out: dict = {"total": total}
    for k in _MEAN_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[f"avg_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
        out[f"n_{k}"] = len(vals)  # 이 평균이 몇 행 위에서 계산됐는지
    for k in _RATE_KEYS:
        out[f"{k}_rate"] = (
            round(sum(r.get(k, 0.0) for r in rows) / total, 4) if total else 0.0
        )
    for k in _COUNT_KEYS:
        out[f"avg_{k}"] = (
            round(sum(r.get(k, 0) for r in rows) / total, 4) if total else 0.0
        )
    return out


def evaluate_pairs(gt_entries, pred_entries, match_mode: str = "index") -> dict:
    """GT test entries + prediction entries → 집계 dict.

    GT next-state 는 **정본과 같은 출처**(`messages[-1]["value"]`)에서 읽는다.
    prediction 의 `label` 을 쓰면 chat template 정규화 차이로 소수 행이 어긋나
    (2026-08-01 실측 EXP01/EXP05 각 4행) 층 분해가 `hungarian_rec` 과 안 맞는다.
    current state 는 prediction 의 `prompt` 에서 읽는다 — 행 정렬이 보장되고
    필터(woa) leaf 도 그대로 커버된다.

    프롬프트에서 current state 를 못 읽으면 `classify_diff` 는 전부 ADDED 를,
    copy_rate 는 0 을 돌려준다 — **그럴듯한 완전 오답 표**가 조용히 나온다.
    그래서 실패를 세서 터뜨린다 (`_compare_site` 설계원칙 #2 와 같은 이유).
    """
    assert_scorer_wired(match_mode)
    rows = []
    failures = 0
    for gt_entry, pred_entry in zip(gt_entries, pred_entries, strict=False):
        gt_text = gt_entry["messages"][-1]["value"]
        pred_text = pred_entry.get("predict", pred_entry.get("output", ""))
        current = parse_prompt(pred_entry.get("prompt", "")).get("current_state", "")
        if not current:
            failures += 1
            continue
        rows.append(compute_state_diff(pred_text, gt_text, current, match_mode))
    if failures:
        raise StateDiffError(
            f"프롬프트에서 current state 를 못 읽은 행 {failures}건 "
            f"(전체 {failures + len(rows)}). 계열 마커가 "
            "'## Current State' / 'Current UI State:' 중 어느 쪽도 아닙니다 — "
            "scripts/_prompt_sections.py 에 계열을 등록하세요."
        )
    return aggregate(rows)


# ── CLI ──────────────────────────────────────────────────────────────────
def _print_row(label: str, m: dict) -> None:
    print(
        f"[state-diff:{label}] total={m['total']}  "
        f"diff_rec={m['avg_diff_recall']:.4f}  "
        f"added_rec={m['avg_added_recall']:.4f}  "
        f"unch_rec={m['avg_unchanged_recall']:.4f}  "
        f"copy={m['avg_copy_rate_pred']:.4f}(gt {m['avg_copy_rate_gt']:.4f})  "
        f"excess={m['avg_copy_excess']:+.4f}"
    )


def _cmd_score(args) -> int:
    split = bool(args.test_id or args.pred_id or args.test_ood or args.pred_ood)
    mm = args.match_mode
    if not args.include_truncated:
        reason = truncated_reason(args.pred, args.pred_id, args.pred_ood)
        if reason:
            print(f"[state-diff] 건너뜀 — {reason}", file=sys.stderr)
            return 0
    if split:
        missing = [
            n
            for n, v in [
                ("--test-id", args.test_id),
                ("--pred-id", args.pred_id),
                ("--test-ood", args.test_ood),
                ("--pred-ood", args.pred_ood),
            ]
            if not v
        ]
        if missing:
            print(f"[state-diff] ERROR: split mode needs {missing}", file=sys.stderr)
            return 2
        gt_id = _he._load_jsonl(args.test_id)
        pr_id = _he._load_jsonl(args.pred_id)
        gt_ood = _he._load_jsonl(args.test_ood)
        pr_ood = _he._load_jsonl(args.pred_ood)
        if args.exclude_action:
            gt_id, pr_id = _he._filter_pairs(gt_id, pr_id, args.exclude_action)
            gt_ood, pr_ood = _he._filter_pairs(gt_ood, pr_ood, args.exclude_action)
        metrics = build_metrics(gt_id, pr_id, gt_ood, pr_ood, mm)
        for k in ("overall", "in_domain", "out_of_domain"):
            _print_row(k, metrics[k])
    else:
        if not (args.test and args.pred):
            print(
                "[state-diff] ERROR: --test and --pred required in single-pair mode",
                file=sys.stderr,
            )
            return 2
        gts = _he._load_jsonl(args.test)
        preds = _he._load_jsonl(args.pred)
        if args.exclude_action:
            gts, preds = _he._filter_pairs(gts, preds, args.exclude_action)
        metrics = evaluate_pairs(gts, preds, mm)
        _print_row("all", metrics)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[state-diff] saved: {out}")
    return 0


def build_metrics(gt_id, pr_id, gt_ood, pr_ood, match_mode: str) -> dict:
    """ID/OOD 3-섹션 산출. `hungarian_metrics.json` 과 **동일한 섹션 구조**여야 한다 —
    `eval_viewer.load_metrics` 의 section 조회는 부재를 silent skip 하므로, 구조가
    어긋나면 표에 빈칸이 뜰 뿐 아무도 오류를 못 본다."""
    return {
        "overall": evaluate_pairs(gt_id + gt_ood, pr_id + pr_ood, match_mode),
        "in_domain": evaluate_pairs(gt_id, pr_id, match_mode),
        "out_of_domain": evaluate_pairs(gt_ood, pr_ood, match_mode),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Copy-bias 진단 채점기 (state 예측 전용)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="state_diff_metrics.json 산출")
    s.add_argument("--test", default=None)
    s.add_argument("--pred", default=None)
    s.add_argument("--test-id", default=None, dest="test_id")
    s.add_argument("--pred-id", default=None, dest="pred_id")
    s.add_argument("--test-ood", default=None, dest="test_ood")
    s.add_argument("--pred-ood", default=None, dest="pred_ood")
    s.add_argument("--output", required=True)
    s.add_argument(
        "--match-mode",
        default="index",
        choices=["index", "pos"],
        dest="match_mode",
        help="정본 채점(_hungarian_eval)과 **반드시 같은 값**이어야 비교 가능하다. "
        "EXP05/06/07 은 pos, 나머지는 index.",
    )
    s.add_argument("--exclude-action", default=None, dest="exclude_action")
    s.add_argument(
        "--include-truncated",
        action="store_true",
        dest="include_truncated",
        help="절단(1024) 경계 이전 prediction 에도 강제로 산출한다. 기본은 건너뛴다 — "
        "잘린 예측은 copy_rate 를 한쪽으로 과소평가한다 (truncated_reason 참고).",
    )
    s.set_defaults(func=_cmd_score)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
