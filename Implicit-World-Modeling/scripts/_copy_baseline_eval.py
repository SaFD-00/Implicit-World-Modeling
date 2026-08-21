#!/usr/bin/env python3
"""복사기(copy baseline) 기준선 채점기 — "그냥 베끼면 몇 점인가"를 지표마다 낸다.

왜 필요한가
-----------
`_state_diff_eval` 은 `copy_excess` 라는 **단일 판별량**으로 복사 편향을 잡고,
`change_f1_floor` 로 퇴화 바닥을 눈금으로 준다. 둘 다 맞는 설계지만 **지표별로는
읽히지 않는다** — "내 `hungarian_f1` 0.728 은 복사기보다 나은가?" 에 답하려면
복사기가 그 지표에서 몇 점인지를 알아야 하는데, `copy_excess` 하나로는 그 질문의
답이 나오지 않는다. 그래서 지표마다 같은 질문을 던진다:

    similarity gain = 내 점수 − 복사기 점수

복사기 = "프롬프트에 주어진 current state XML 을 그대로 예측으로 낸 가상 모델".
학습 없이 정의되고, 같은 test 파일·같은 채점기·같은 매칭 모드 위에서 산출된다.

채점은 **재구현하지 않는다**
----------------------------
`pred_text := parse_prompt(prompt)["current_state"]` 로 치환하고 정본 채점기
(`_hungarian_eval.compute_hungarian_acc` · `_state_diff_eval.compute_state_diff`)를
그대로 부른다. 지표를 여기서 다시 구현하면 두 벌이 언젠가 조용히 갈린다.

⚠️ 치환할 때 **재직렬화·정규화·strip 을 하지 않는다.** `parse_prompt` 가 돌려준
문자열 객체를 pred 슬롯과 current 슬롯 **양쪽에 그대로** 넣어야 `copy_exact_rate`
가 1.0 이 된다 — 그게 "정말 복사기를 채점했나"의 닫힌 형태 증명이다 (아래 불변식).

집계만 한 군데 다시 쓴다. `_hungarian_eval.evaluate_pairs` 는 행 단위 값을 밖으로
내주지 않는데 gain 은 행 단위 교집합이 필요해서, 같은 공식(`sum/total` → round 4)을
`_aggregate_hungarian` 이 미러한다. 그 동치성은 `tests/test_copy_baseline_eval.py`
가 `_he.evaluate_pairs` 와 직접 비교해 못박는다 — 여기가 유일한 드리프트 지점이다.

gain 은 **행 단위 교집합**에서만 낸다 — 두 JSON 의 뺄셈이 아니다
-----------------------------------------------------------------
`aggregate()` 의 `avg_*` 는 그 지표가 정의된 행에서만의 평균이고 `n_<key>` 가 그
분모다. 그런데 모델과 복사기의 **정의행 집합이 다르다**:

  - `addmod_prec` / `addmod_f1` — 복사기는 pred-side diff 가 항상 공집합이라
    (예측 요소가 전부 current 와 매칭된다) **전 행 미정의**다.
  - `copy_excess` — 모델은 파싱 실패행에서 None 인데 복사기는 파싱 실패가 없다.

두 평균을 그냥 빼면 **서로 다른 population 위의 수**를 빼게 된다. 그래서 같은 pass
안에서 행별로 model row 와 copy row 를 둘 다 만들고, gain 키마다 **양쪽 모두
non-None 인 행**에서만 `model − copy` 를 평균한다. `n_gain_<key>` 가 그 교집합
크기이고 항상 함께 싣는다. 교집합이 0행이면 `avg_gain_<key>` 는 **None** 이다 —
0.0 은 "차이가 없다"로 읽히므로 쓰면 안 된다.

gain 을 붙이지 않는 키 (`_NO_GAIN_KEYS`)
----------------------------------------
`copy_rate_pred` / `copy_rate_gt` / `copy_excess` / `change_f1_floor` / `copy_exact` /
`copy_near` / `unclosed_root` / `parse_fail` / `parse_fail_long` / 모든 `n_*`.
이유는 하나다 — **방향성이 없거나 뺄셈이 오독을 만든다.** 이 키들은 "높을수록 좋음"
이 아니다: `copy_excess` 의 gain 은 복사기가 정의상 최댓값이라 항상 큰 음수가 되고
(`−0.77` 같은 수), 부호만 보면 그게 "좋아 보인다". `change_f1_floor` 는 예측과
무관한 눈금이라 gain 이 항상 0 이다. `copy_exact`/`copy_near` 는 복사기가 정의상
1.0 이라 gain 이 "복사하지 않은 비율"의 음수일 뿐 성능이 아니다. `n_*` 는 개수라
뺄셈의 뜻이 없다. 이 키들은 `copy_baseline` 쪽에 값만 싣는다.

산출물
------
leaf 옆 `copy_baseline_metrics.json`. 섹션마다
`{"copy_baseline": {...}, "model": {...}, "gain": {...}}` 이고, `copy_baseline` /
`model` 은 hungarian 집계와 state-diff 집계를 **한 dict 로 병합**한 것이다
(충돌은 `total` 하나뿐이고 같은 행 리스트의 길이라 값이 같다 — `_merge_aggregates`).

절단(1024) leaf 의 처리가 이 채점기만 다르다. `_state_diff_eval._cmd_score` 는 leaf
전체를 건너뛰지만, 여기서는 **`copy_baseline` 을 정상 산출하고 `model`/`gain` 만
null 로 둔다** — 잘린 것은 예측이지 프롬프트가 아니므로 복사기 점수는 정확하다.
사유는 최상위 `truncated` 에 문자열로 남는다.

single-pair 모드(MB/MC)도 3-섹션 스키마를 쓰되 `overall` 만 채운다. 소비자가 아직
없어 스키마를 하나로 두는 편이 낫다 (`_state_diff_eval` 은 flat 을 낸다 — 다르다).

Subcommand
----------
score : prediction jsonl + GT test jsonl 로 copy_baseline_metrics.json 을 만든다.
        정본 eval 경로에는 얹지 않는다 — 백필/재산출 전용 진입점이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _hungarian_eval as _he  # noqa: E402
import _state_diff_eval as _sd  # noqa: E402
from _prompt_sections import parse_prompt  # noqa: E402

# 산출물 스키마 버전. `metrics_schema`(_state_diff_eval 쪽 버전)와 **별개**로 박는다 —
# gain 키 목록이나 병합 규칙이 바뀌면 여기가 움직이고, 행 단위 지표 정의가 바뀌면
# 저쪽이 움직인다. 두 축을 한 문자열에 섞으면 어느 쪽이 바뀐 건지 파일만 보고 모른다.
COPY_BASELINE_SCHEMA = "2026-08-11"

# `avg_copy_excess ≈ 1 − avg_copy_rate_gt` 의 허용 오차. 정확히 같지 않은 이유는 둘이다:
# (a) 행마다 round(.., 4) 를 거치고, (b) 두 평균의 **분모가 다르다** — `copy_rate_gt` 는
# gt_els 만 있으면 정의되는데 `copy_excess` 는 pred_els(=cur_els)도 필요하다.
# 실측(EXP05 full ep3 overall): n_copy_rate_gt 5343 vs n_copy_excess 5338.
INVARIANT_TOL = 5e-3


class CopyBaselineError(RuntimeError):
    """채점 중단 — 값이 아니라 배선/불변식이 깨졌을 때만 낸다."""


# ── gain 키 ──────────────────────────────────────────────────────────────
# hungarian 계열: 전 행에서 정의된다 (None 이 없다). 따라서 `n_gain_*` 는 total 과 같다.
_HUNGARIAN_GAIN_KEYS = (
    "bleu",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "exact_match",
    "hungarian_ea",
    "hungarian_f1",
    "hungarian_prec",
    "hungarian_rec",
    "hungarian_text",
)
# state-diff 계열: 조건부 정의다. `n_gain_*` 가 model·copy 정의행의 **교집합** 크기이고,
# `addmod_prec`/`addmod_f1` 은 복사기가 전 행 미정의라 항상 0 이 된다 (설계상 정상).
_STATE_GAIN_KEYS = (
    "addmod_recall",
    "added_recall",
    "modified_recall",
    "unchanged_recall",
    "addmod_prec",
    "addmod_f1",
    "change_prec_strict",
    "change_recall_strict",
    "change_f1_strict",
    "change_prec_loose",
    "change_recall_loose",
    "change_f1_loose",
    # ⚠️ `no_change_acc` gain 은 **천장에 붙어 있다.** 복사기는 변화 없는 행에서 정의상
    # 1.0 이라 gain 이 항상 ≤ 0 이고, 실질적으로 "모델이 정지 화면을 못 재현한 비율"의
    # 음수다. 다른 gain 과 같은 축("클수록 좋다, 0 이 기준")으로 읽으면 오독한다.
    "no_change_acc",
)
# gain 을 만들지 않는 행 키. 모듈 docstring 의 이유를 코드에서도 붙잡아 둔다 —
# 목록만 있으면 "왜 빠졌지" 하고 누가 다시 넣는다.
_NO_GAIN_KEYS = (
    "copy_rate_pred",  # 복사기는 정의상 1.0 — gain 은 "복사 안 한 비율"의 음수일 뿐
    "copy_rate_gt",  # 예측과 무관한 GT 기준선. 두 행이 같은 값이라 gain 이 항상 0
    "copy_excess",  # 복사기가 정의상 최댓값 → gain 이 항상 큰 음수 = 좋아 보인다
    "change_f1_floor",  # 예측과 무관한 눈금 → gain 항상 0
    "copy_exact",  # 복사기 정의상 1.0
    "copy_near",  # 복사기 정의상 1.0
    "unclosed_root",  # sanity 신호이지 성능이 아니다
    "parse_fail",  # 복사기는 파싱 실패가 없다 (0.0) → gain 이 곧 모델 실패율
    "parse_fail_long",
)


def gain_keys(match_mode: str) -> tuple[str, ...]:
    """gain 을 붙일 행 키 목록. 위치 키는 매칭 모드마다 이름이 다르다."""
    pos_key = "hungarian_pos" if match_mode == "pos" else "hungarian_idx"
    return (*_HUNGARIAN_GAIN_KEYS, pos_key, *_STATE_GAIN_KEYS)


# ── 행 단위 채점 ─────────────────────────────────────────────────────────
def _hungarian_row(
    pred_str: str,
    gt_str: str,
    match_mode: str = "index",
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
) -> dict:
    """`_hungarian_eval.evaluate_pairs` 의 행 루프와 **같은 계산**을 flat dict 으로.

    저쪽은 `{"hungarian": {...}}` 로 한 겹 감싸는데 여기서는 gain 계산이 행 키를
    평평하게 훑어야 해서 펼친다. 값 자체는 손대지 않는다 (bleu/rouge 는 raw,
    hungarian 계열은 `compute_hungarian_acc` 안에서 이미 round(.,4) 된 값).
    """
    return {
        "bleu": _he.calc_bleu(gt_str, pred_str),
        "rouge_1": _he.calc_rouge_n(gt_str, pred_str, 1),
        "rouge_2": _he.calc_rouge_n(gt_str, pred_str, 2),
        "rouge_l": _he.calc_rouge_l(gt_str, pred_str),
        "exact_match": 1.0 if gt_str.strip() == pred_str.strip() else 0.0,
        **_he.compute_hungarian_acc(
            pred_str,
            gt_str,
            match_mode,
            strict_pos=strict_pos,
            include_aria=include_aria,
        ),
    }


def score_row(
    pred_str: str,
    gt_str: str,
    current_str: str,
    match_mode: str = "index",
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
) -> dict:
    """한 행의 hungarian 지표 + state-diff 지표를 한 dict 으로.

    두 채점기의 **행 키는 서로 겹치지 않는다** (집계 키는 `total` 하나가 겹친다).
    겹치면 한쪽이 조용히 덮여 gain 이 엉뚱한 축에서 계산되므로 매 행 확인한다 —
    집합 교집합은 Hungarian 매칭 비용에 비하면 무시할 수 있다.
    """
    hung = _hungarian_row(
        pred_str, gt_str, match_mode, strict_pos=strict_pos, include_aria=include_aria
    )
    diff = _sd.compute_state_diff(
        pred_str,
        gt_str,
        current_str,
        match_mode,
        strict_pos=strict_pos,
        include_aria=include_aria,
    )
    clash = set(hung) & set(diff)
    if clash:
        raise CopyBaselineError(
            f"hungarian 행 키와 state-diff 행 키가 충돌한다: {sorted(clash)} "
            "— 한쪽이 조용히 덮인다."
        )
    return {**hung, **diff}


def score_rows(
    gt_entries,
    pred_entries,
    match_mode: str = "index",
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
    with_model: bool = True,
) -> tuple[list[dict] | None, list[dict]]:
    """(model_rows, copy_rows). `with_model=False` 면 model_rows 는 None.

    GT next-state 는 정본과 **같은 출처**(`messages[-1]["value"]`)에서, current state
    는 prediction 의 `prompt` 에서 읽는다 — `_state_diff_eval.evaluate_pairs` 와 같은
    규칙이라야 세 산출물(hungarian / state_diff / copy_baseline)이 같은 행 위에 선다.

    current 를 못 읽으면 복사기가 **빈 문자열을 예측으로 내는 다른 모델**이 되어
    그럴듯한 완전 오답표가 조용히 나온다. 그래서 세서 터뜨린다.
    """
    opts = {"strict_pos": strict_pos, "include_aria": include_aria}
    # ⚠️ `_lazy_deps()` 를 안 부르면 `compute_hungarian_acc` 의 except 가 예외를 삼켜
    # **전 행 0점**을 조용히 돌려준다. 재사용 코드가 전부 밟는 함정이라 여기서 막는다.
    _he._lazy_deps()
    _sd.assert_scorer_wired(match_mode, **opts)

    model_rows: list[dict] | None = [] if with_model else None
    copy_rows: list[dict] = []
    failures = 0
    for gt_entry, pred_entry in zip(gt_entries, pred_entries, strict=False):
        gt_text = gt_entry["messages"][-1]["value"]
        current = parse_prompt(pred_entry.get("prompt", "")).get("current_state", "")
        if not current:
            failures += 1
            continue
        # ⚠️ `current` 객체를 pred 슬롯과 current 슬롯에 **그대로** 넣는다. 어느 한쪽만
        # 정규화하면 copy_exact 가 1.0 이 아니게 되고 불변식이 무너진다.
        copy_rows.append(score_row(current, gt_text, current, match_mode, **opts))
        if model_rows is not None:
            pred_text = pred_entry.get("predict", pred_entry.get("output", ""))
            model_rows.append(
                score_row(pred_text, gt_text, current, match_mode, **opts)
            )
    if failures:
        raise CopyBaselineError(
            f"프롬프트에서 current state 를 못 읽은 행 {failures}건 "
            f"(전체 {failures + len(copy_rows)}). 계열 마커가 "
            "'## Current State' / 'Current UI State:' 중 어느 쪽도 아닙니다 — "
            "scripts/_prompt_sections.py 에 계열을 등록하세요."
        )
    return model_rows, copy_rows


# ── 집계 ─────────────────────────────────────────────────────────────────
def _aggregate_hungarian(rows: list[dict], match_mode: str = "index") -> dict:
    """`_hungarian_eval.evaluate_pairs` 의 집계부 미러.

    저쪽을 그대로 부르지 못하는 이유는 하나다 — 그 함수는 entries 를 받아 행을 **내부에서
    만들고 버린다.** gain 은 행 단위 교집합이 필요해서 행을 손에 쥐고 있어야 한다.
    공식(`sum/total` → round 4)과 키 이름은 저쪽과 한 글자도 다르면 안 되고, 그 동치성은
    `tests/test_copy_baseline_eval.py::TestHungarianAggregateParity` 가 못박는다.
    """
    pos_key = "hungarian_pos" if match_mode == "pos" else "hungarian_idx"
    total = len(rows)

    def avg(key: str) -> float:
        return sum(r[key] for r in rows) / total if total else 0.0

    return {
        "total": total,
        "avg_bleu": round(avg("bleu"), 4),
        "avg_rouge_1": round(avg("rouge_1"), 4),
        "avg_rouge_2": round(avg("rouge_2"), 4),
        "avg_rouge_l": round(avg("rouge_l"), 4),
        "exact_match_rate": round(avg("exact_match"), 4),
        "avg_hungarian_ea": round(avg("hungarian_ea"), 4),
        "avg_hungarian_f1": round(avg("hungarian_f1"), 4),
        "avg_hungarian_prec": round(avg("hungarian_prec"), 4),
        "avg_hungarian_rec": round(avg("hungarian_rec"), 4),
        "avg_hungarian_text": round(avg("hungarian_text"), 4),
        f"avg_{pos_key}": round(avg(pos_key), 4),
    }


def _merge_aggregates(hung: dict, state: dict) -> dict:
    """두 집계를 한 dict 으로. 충돌은 `total` 하나뿐이고 값이 같아야 한다.

    같은 행 리스트의 길이라 정의상 같지만, 두 집계가 서로 다른 행 집합 위에서 계산되는
    변경이 언젠가 들어오면 여기서 걸린다 (그때 조용히 덮이면 `total` 이 어느 쪽 것인지
    아무도 모른다).
    """
    clash = set(hung) & set(state)
    if clash != {"total"}:
        raise CopyBaselineError(
            f"집계 키 충돌이 예상 밖입니다: {sorted(clash - {'total'})} "
            "— 접두어를 붙이지 말고 키 이름 설계를 다시 보세요."
        )
    if hung["total"] != state["total"]:
        raise CopyBaselineError(
            f"두 집계의 total 이 다릅니다 (hungarian={hung['total']}, "
            f"state-diff={state['total']}) — 서로 다른 행 집합 위에서 쟀습니다."
        )
    return {**hung, **state}


def aggregate(rows: list[dict], match_mode: str = "index") -> dict:
    """행 리스트 → hungarian + state-diff 병합 집계.

    ⚠️ overall 은 반드시 `rows_id + rows_ood` 를 **그 순서로 이어붙여** 넣는다. 섹션
    평균의 가중합으로 만들면 부동소수 합산 순서가 달라져 round 4 에서 갈릴 수 있고,
    그러면 기존 `hungarian_metrics.json` 과 키별 일치가 깨진다.
    """
    return _merge_aggregates(
        _aggregate_hungarian(rows, match_mode), _sd.aggregate(rows)
    )


def compute_gain(
    model_rows: list[dict], copy_rows: list[dict], match_mode: str = "index"
) -> dict:
    """행 단위 교집합 위의 `model − copy` 평균. 키마다 분모(`n_gain_*`)를 함께 낸다.

    교집합이 0행이면 **None** 이다 — 0.0 은 "차이가 없다"로 읽힌다. `addmod_prec` /
    `addmod_f1` 이 그 경우이고 (복사기는 pred-side diff 가 항상 공집합), 그것이
    버그가 아니라 설계라는 점은 테스트가 고정한다.
    """
    out: dict = {}
    for k in gain_keys(match_mode):
        deltas = [
            m[k] - c[k]
            for m, c in zip(model_rows, copy_rows, strict=True)
            if m.get(k) is not None and c.get(k) is not None
        ]
        out[f"avg_gain_{k}"] = round(sum(deltas) / len(deltas), 4) if deltas else None
        out[f"n_gain_{k}"] = len(deltas)
    return out


def assert_copy_baseline_invariants(metrics: dict, label: str = "") -> None:
    """복사기 점수는 `(current, gt)` 만의 함수라 **닫힌 형태로 검증된다.**

    여기가 이 채점기의 self-test 다. 어긋나면 값이 아니라 배선이 깨진 것이므로 죽인다
    — 복사기 점수는 데이터가 무엇이든 아래 등식을 만족해야 한다.
    `_state_diff_eval.aggregate` 는 정의행이 0 일 때 None 이 아니라 **0.0** 을 내므로
    분모(`n_*`)를 먼저 보고 검사한다.
    """
    tag = f"[{label}] " if label else ""

    def die(msg: str) -> None:
        raise CopyBaselineError(f"{tag}복사기 불변식 위반 — {msg}")

    if metrics["total"] and metrics["copy_exact_rate"] != 1.0:
        die(
            f"copy_exact_rate={metrics['copy_exact_rate']} — 1.0 이어야 합니다. "
            "예측 슬롯에 current 문자열을 그대로 넣지 않았습니다(정규화/strip 금지)."
        )
    if metrics["n_copy_rate_pred"] and metrics["avg_copy_rate_pred"] != 1.0:
        die(
            f"avg_copy_rate_pred={metrics['avg_copy_rate_pred']} — 1.0 이어야 합니다 "
            "(예측 요소가 전부 current 와 매칭되어야 합니다)."
        )
    if metrics["n_copy_excess"]:
        expected = 1.0 - metrics["avg_copy_rate_gt"]
        if abs(metrics["avg_copy_excess"] - expected) > INVARIANT_TOL:
            die(
                f"avg_copy_excess={metrics['avg_copy_excess']} 가 "
                f"1 − avg_copy_rate_gt={expected:.4f} 와 {INVARIANT_TOL} 넘게 다릅니다."
            )
    if metrics["avg_change_f1_strict"] != 0.0:
        die(
            f"avg_change_f1_strict={metrics['avg_change_f1_strict']} — 0.0 이어야 합니다 "
            "(복사기는 어떤 변화도 주장하지 않습니다)."
        )
    if metrics["n_no_change_acc"] and metrics["avg_no_change_acc"] != 1.0:
        die(
            f"avg_no_change_acc={metrics['avg_no_change_acc']} — 1.0 이어야 합니다 "
            "(변화 없는 행에서는 복사가 정답입니다)."
        )
    # recall 층 항등식 — 근사가 아니라 **닫힌 형태**다. 복사기는 `pred_els == cur_els`
    # 이므로 hit 을 만드는 매칭(`_hungarian_match(pred, gt)`)과 GT 를 diff 유형으로
    # 나누는 매칭(`_hungarian_match(cur, gt)`)이 **같은 호출**이 된다. 따라서 행마다
    #     hit_gt == MODIFIED ∪ UNCHANGED
    # 이고 modified/unchanged recall 이 1.0, added recall 이 0.0 으로 **정확히** 떨어진다.
    # 여기서 나오는 따름정리가 이 채점기의 핵심 발견이다:
    #     addmod_recall(복사기) = |MODIFIED| / (|MODIFIED| + |ADDED|)
    # 즉 복사기의 `addmod_recall` 은 성능이 아니라 **test set 구성 통계**이며, 그래서
    # `addmod_recall` 은 "복사에 면역인 축"이 아니다 (pos 모드 실측 0.63). 복사에 진짜
    # 면역인 것은 `added_recall`(0.0)과 `change_f1_strict`(0.0)뿐이다.
    # 어긋나면 값 해석 문제가 아니라 solver 비결정성이나 element 추출 불일치다.
    for key, want in (
        ("modified_recall", 1.0),
        ("unchanged_recall", 1.0),
        ("added_recall", 0.0),
    ):
        # `n_*` 가 0 이면 그 층이 없는 것이라 `avg_*`(=0.0)에 뜻이 없다 — 먼저 본다.
        if metrics[f"n_{key}"] and metrics[f"avg_{key}"] != want:
            die(
                f"avg_{key}={metrics[f'avg_{key}']} — {want} 여야 합니다 "
                f"(n={metrics[f'n_{key}']}). 복사기에서는 pred↔gt 매칭과 cur→gt 분류가 "
                "같은 매칭이라 이 값이 항등적으로 결정됩니다."
            )
    if metrics["total"] and metrics["avg_hungarian_f1"] <= 0.0:
        die(
            "avg_hungarian_f1=0.0 — 이것은 지표 해석 문제가 아니라 **배선이 죽은 것**"
            "입니다. `_hungarian_eval._lazy_deps()` 미호출 시 compute_hungarian_acc 의 "
            "except 가 예외를 삼켜 전 행 0점을 조용히 돌려줍니다."
        )
    if metrics["parse_fail_rate"]:
        print(
            f"[copy-baseline] 경고: {tag}복사기 parse_fail_rate="
            f"{metrics['parse_fail_rate']} — current state 에서 element 를 하나도 못 뽑은 "
            "행이 있습니다 (프롬프트가 잘렸거나 계열이 다릅니다).",
            file=sys.stderr,
        )


# ── 섹션 조립 ────────────────────────────────────────────────────────────
def build_section(
    model_rows: list[dict] | None,
    copy_rows: list[dict],
    match_mode: str,
    label: str = "",
) -> dict:
    """한 섹션의 `{copy_baseline, model, gain}`.

    `model_rows is None` = 절단 leaf. `model`/`gain` 은 null 이지만 `copy_baseline` 은
    낸다 — 잘린 것은 예측이고 프롬프트는 멀쩡하므로 복사기 점수는 정확하다.
    """
    cb = aggregate(copy_rows, match_mode)
    assert_copy_baseline_invariants(cb, label)
    if model_rows is None:
        return {"copy_baseline": cb, "model": None, "gain": None}
    return {
        "copy_baseline": cb,
        "model": aggregate(model_rows, match_mode),
        "gain": compute_gain(model_rows, copy_rows, match_mode),
    }


def evaluate_pairs(
    gt_entries,
    pred_entries,
    match_mode: str = "index",
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
    with_model: bool = True,
    label: str = "",
) -> dict:
    """entries 한 쌍 → 한 섹션 dict."""
    model_rows, copy_rows = score_rows(
        gt_entries,
        pred_entries,
        match_mode,
        strict_pos=strict_pos,
        include_aria=include_aria,
        with_model=with_model,
    )
    return build_section(model_rows, copy_rows, match_mode, label)


def build_metrics(
    gt_id,
    pr_id,
    gt_ood,
    pr_ood,
    match_mode: str,
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
    with_model: bool = True,
) -> dict:
    """ID/OOD 3-섹션 산출. `hungarian_metrics.json` 과 **같은 섹션 구조**여야 한다.

    행을 split 마다 **한 번만** 채점하고 overall 은 그 둘을 이어붙여 집계한다. 정본
    (`_hungarian_eval` / `_state_diff_eval`)은 overall 을 위해 전 행을 다시 채점하는데,
    집계가 행 리스트의 순수 함수라 이어붙이기와 결과가 같다 — 여기는 복사기까지 함께
    돌아 비용이 두 배라 재채점을 피한다.
    """
    opts = {
        "strict_pos": strict_pos,
        "include_aria": include_aria,
        "with_model": with_model,
    }
    m_id, c_id = score_rows(gt_id, pr_id, match_mode, **opts)
    m_ood, c_ood = score_rows(gt_ood, pr_ood, match_mode, **opts)
    m_all = None if m_id is None or m_ood is None else m_id + m_ood
    return {
        "overall": build_section(m_all, c_id + c_ood, match_mode, "overall"),
        "in_domain": build_section(m_id, c_id, match_mode, "in_domain"),
        "out_of_domain": build_section(m_ood, c_ood, match_mode, "out_of_domain"),
    }


def stamp(metrics: dict, match_mode: str, truncated: str | None) -> dict:
    """최상위 메타 5종. `metrics_schema` 는 행 단위 지표 정의의 버전이라
    `_state_diff_eval` 에서 가져온다 (`copy_baseline_schema` 와 축이 다르다).

    `element_set` 은 인자가 아니라 **채점기가 실제로 읽은 전역**(`_he.ELEMENT_SET`)에서
    읽는다 (`_state_diff_eval.stamp_schema` 와 같은 방식). 호출자가 믿는 값을 적으면
    전파가 끊긴 바로 그 경우에 산출물이 거짓을 말한다 — 전역은 모듈 사본마다 따로
    있어서 실제로 끊길 수 있다. 형제 산출물(`hungarian_metrics.json` /
    `state_diff_metrics.json`)이 같은 규칙으로 스탬프하므로, 세 파일의 `element_set`
    을 나란히 놓으면 전파 실패가 그대로 드러난다.
    """
    metrics["copy_baseline_schema"] = COPY_BASELINE_SCHEMA
    metrics["metrics_schema"] = _sd.METRICS_SCHEMA
    metrics["match_mode"] = match_mode
    metrics["element_set"] = _he.ELEMENT_SET
    metrics["truncated"] = truncated
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────
def _print_row(label: str, section: dict, match_mode: str) -> None:
    cb = section["copy_baseline"]
    pos_key = "avg_hungarian_pos" if match_mode == "pos" else "avg_hungarian_idx"
    head = (
        f"[copy-baseline:{label}] total={cb['total']}  "
        f"copy: f1={cb['avg_hungarian_f1']:.4f} rec={cb['avg_hungarian_rec']:.4f} "
        f"prec={cb['avg_hungarian_prec']:.4f} pos={cb[pos_key]:.4f} "
        f"addmod_rec={cb['avg_addmod_recall']:.4f} unch_rec={cb['avg_unchanged_recall']:.4f}"
    )
    if section["model"] is None:
        print(f"{head}  | model/gain: 절단이라 미산출")
        return
    g = section["gain"]

    def fmt(key: str) -> str:
        v = g[f"avg_gain_{key}"]
        return "n/a" if v is None else f"{v:+.4f}"

    print(
        f"{head}\n"
        f"[copy-baseline:{label}] gain: f1={fmt('hungarian_f1')}(n={g['n_gain_hungarian_f1']}) "
        f"rec={fmt('hungarian_rec')} addmod_rec={fmt('addmod_recall')}"
        f"(n={g['n_gain_addmod_recall']}) "
        f"added_rec={fmt('added_recall')} change_f1={fmt('change_f1_strict')} "
        f"bleu={fmt('bleu')}"
    )


def _cmd_score(args) -> int:
    split = bool(args.test_id or args.pred_id or args.test_ood or args.pred_ood)
    mm = args.match_mode
    # 매칭 기준 스위치는 여기 한 진입점에서만 읽어 아래로 넘긴다 (전역 금지 —
    # `_hungarian_eval._cmd_score` / `_state_diff_eval._cmd_score` 와 같은 규칙이다).
    opts = {"strict_pos": args.strict_pos_match, "include_aria": args.include_aria}
    # element 집합만 전역이다 (`_he.set_element_set` 위 주석의 (a)(b)(c) 참고).
    # 이 채점기는 정본 eval 배선에 얹혀 있지 않은 **백필 전용 진입점**이라 스스로
    # 설정해야 한다 — 안 하면 기본값에만 의존하게 되어 legacy 재산출 경로가 없다.
    # `getattr` 기본값은 `_hungarian_eval._cmd_score` 와 같은 규칙이다 — CLI 를 거치지
    # 않고 Namespace 를 직접 만들어 부르는 호출부(테스트·백필 스크립트)가 있다.
    _he.set_element_set(getattr(args, "element_set", _he._default_element_set()))

    # 절단 판정은 **한 번만** 하고 세 섹션에 같은 값을 내린다. 여기서는 leaf 를 건너뛰지
    # 않는다 (`_state_diff_eval._cmd_score` 와 다른 점) — 복사기 점수는 절단과 무관하다.
    truncated = _sd.truncated_reason(args.pred, args.pred_id, args.pred_ood)
    if truncated:
        print(
            f"[copy-baseline] model/gain 미산출 (copy_baseline 은 산출) — {truncated}",
            file=sys.stderr,
        )
    opts["with_model"] = truncated is None

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
            print(f"[copy-baseline] ERROR: split mode needs {missing}", file=sys.stderr)
            return 2
        gt_id = _he._load_jsonl(args.test_id)
        pr_id = _he._load_jsonl(args.pred_id)
        gt_ood = _he._load_jsonl(args.test_ood)
        pr_ood = _he._load_jsonl(args.pred_ood)
        if args.exclude_action:
            gt_id, pr_id = _he._filter_pairs(gt_id, pr_id, args.exclude_action)
            gt_ood, pr_ood = _he._filter_pairs(gt_ood, pr_ood, args.exclude_action)
        metrics = build_metrics(gt_id, pr_id, gt_ood, pr_ood, mm, **opts)
    else:
        if not (args.test and args.pred):
            print(
                "[copy-baseline] ERROR: --test and --pred required in single-pair mode",
                file=sys.stderr,
            )
            return 2
        gts = _he._load_jsonl(args.test)
        preds = _he._load_jsonl(args.pred)
        if args.exclude_action:
            gts, preds = _he._filter_pairs(gts, preds, args.exclude_action)
        metrics = {"overall": evaluate_pairs(gts, preds, mm, label="overall", **opts)}
    for sec in ("overall", "in_domain", "out_of_domain"):
        if sec in metrics:
            _print_row(sec, metrics[sec], mm)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(stamp(metrics, mm, truncated), f, ensure_ascii=False, indent=2)
    print(f"[copy-baseline] saved: {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="복사기(copy baseline) 기준선 채점기 — state 예측 전용"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="copy_baseline_metrics.json 산출")
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
    s.add_argument(
        "--strict-pos-match",
        action="store_true",
        dest="strict_pos_match",
        help="pos 모드 매칭 임계를 1.7 → 1.5 로 조인다. **기본은 꺼짐** — 정본 채점과 "
        "**반드시 같은 값**이어야 model 섹션이 기존 산출물과 일치한다.",
    )
    s.add_argument(
        "--include-aria",
        action="store_true",
        dest="include_aria",
        help="pos 모드에서 aria-label 만 가진 요소도 채점 대상에 넣는다. **기본은 꺼짐** — "
        "정본 채점과 **반드시 같은 값**이어야 한다 (element 집합 자체가 달라진다).",
    )
    s.add_argument(
        "--element-set",
        # 세 채점기가 같은 규칙이어야 `ELEMENT_SET=legacy bash scripts/rebuild_*.sh`
        # 한 줄로 셋 다 옛 기준으로 재산출된다. 여기만 "full" 로 굳으면 gain 의
        # 피감수(model)와 감수(복사기)가 서로 다른 element 집합에서 나온다.
        default=_he._default_element_set(),
        choices=["full", "legacy"],
        dest="element_set",
        help="채점 대상 element 집합. 정본 채점(_hungarian_eval)과 **반드시 같은 값**"
        "이어야 gain 의 피감수(model)와 감수(복사기)가 같은 세계의 수가 된다. legacy 는 "
        "2026-08-21 이전의 화이트리스트 집합(실제 요소의 약 24%% 를 버린다) 재현용이다.",
    )
    s.add_argument("--exclude-action", default=None, dest="exclude_action")
    # `--include-truncated` 는 두지 않는다. 절단이어도 복사기 점수는 항상 산출하고
    # (프롬프트는 안 잘린다), model/gain 만 null 로 빠지므로 스위치가 필요 없다.
    s.set_defaults(func=_cmd_score)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
