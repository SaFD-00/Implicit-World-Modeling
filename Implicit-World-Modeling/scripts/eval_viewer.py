"""Build pair-aligned HTML comparison of Stage 1/2 eval outputs.

두 가지 산출 모드가 있다.

**pairs 모드 (기본)** — 각 (stage × logical-dataset) 별로 (EXP × MODEL × variant)
prediction 전량을 하나의 HTML 로 묶고, in-page checkbox 로 column/row 를 토글한다.

**site 모드 (`--site`)** — EXP 별 정성 비교 사이트를 만든다. stage × task 조합마다
자체 완결형 index.html + README.md 한 벌이고, 표본을 시드 고정으로 뽑아 정본 채점기
점수(Hungarian / step-accuracy / thought ROUGE-L)를 카드에 함께 싣는다. 이미지는 제외.

Output
------
pairs · single spec : outputs/{DS_DATADIR(exp)}/eval/{model}/stage{N}_eval/pairs_*.html
pairs · multi spec  : outputs/_compare/stage{N}_eval/pairs_*.html
                      (같은 위치에 `pairs_summary.md` 도 생성.)
site                : outputs/_compare/{on_ac_expNN}_stage{N}_{state|action}_compare/
                      {index.html, README.md}  + outputs/_compare/index.html (목록)

Examples
--------
    # 단일 EXP — 그 EXP 의 eval/ 디렉토리에 산출
    python scripts/eval_viewer.py --include AC_EXP02:qwen3-vl-8b
    python scripts/eval_viewer.py --include AC_EXP01:qwen3-vl-8b_ratio73 --stages 2

    # 다중 EXP cross-compare — outputs/_compare/ 에 산출
    python scripts/eval_viewer.py --include AC_EXP01:qwen3-vl-8b_ratio73 AC_EXP02:qwen3-vl-8b

    # EXP 별 정성 비교 사이트 (stage1 state/action + stage2)
    python scripts/eval_viewer.py --site --include AC_EXP02:qwen3-vl-8b
    python scripts/eval_viewer.py --site --samples 30 --seed 42 \\
        --include AC_EXP05:qwen2.5-vl-3b AC_EXP07_v1:qwen2.5-vl-3b_v1

    # 데이터셋/variant 필터
    python scripts/eval_viewer.py --include AC_EXP02:qwen3-vl-8b \\
        --datasets on-AC-state-id on-AC-action-id \\
        --variants "lora_world-model/epoch-1"

EVAL_DATASETS 는 (stage, EXP, logical_key) → on-disk dir / predictions jsonl /
test jsonl / metric files 를 단일 매핑으로 갖는다. 디렉토리 명명은
`scripts/stage{1,2}_eval.sh` (`on-{EVAL_DS}[-state|-action][-without-open_app]`)
와 `scripts/_common.sh::DS_DATADIR` 에 정합.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _compare_site  # noqa: E402

# DS key → outputs/ 직속 디렉토리 (scripts/_common.sh::DS_DATADIR 와 정합).
# AC_EXP07_v1/_v2 는 model/train 아티팩트만 버전이 갈리고 data 디렉토리는 공유한다.
DS_DATADIR: dict[str, str] = {
    "AC_EXP01": "AndroidControl_EXP01",
    "AC_EXP02": "AndroidControl_EXP02",
    "AC_EXP03": "AndroidControl_EXP03",
    "AC_EXP04": "AndroidControl_EXP04",
    "AC_EXP05": "AndroidControl_EXP05",
    "AC_EXP06": "AndroidControl_EXP06",
    "AC_EXP07_v1": "AndroidControl_EXP07",
    "AC_EXP07_v2": "AndroidControl_EXP07",
    "AC_EXP08": "AndroidControl_EXP08",
    "MC": "MonkeyCollection",
}

STATE_METRIC_KEYS = [
    "total",
    "exact_match_rate",
    "avg_bleu",
    "avg_rouge_l",
    "predict_bleu-4",
    "predict_rouge-l",
    # ── state-diff (copy-bias) 진단 — state_diff_metrics.json ──────────────
    # hungarian 계열은 예측 전체를 GT 전체와 맞춘다. next-state 는 current 와 대부분
    # 겹치므로 **current 를 베끼기만 해도 f1 이 높다.** 아래 키가 그 구분을 만든다.
    # avg_copy_excess 가 판별량 — 0 근처면 "GT 가 겹치는 만큼만 겹쳤다", 큰 양수면
    # "바뀌었어야 할 자리까지 베꼈다". 2026-07-28 23:38 UTC 이전 leaf 는 예측이
    # 1024 로 잘려 element 수가 줄어 copy_rate 가 과소평가되므로 산출하지 않는다
    # (그래서 구 leaf 는 이 컬럼이 비는 게 정상이다).
    # 2026-08-04 개명(scripts/_state_diff_eval.py 모듈 docstring 참고): diff_*→addmod_*,
    # change_f1/change_f1_null→change_f1_strict/change_f1_floor. 기존 34개 leaf 는
    # 옛 키만 갖고 있어 `_metric()` 이 새 이름 우선, 옛 이름 폴백으로 읽는다.
    "avg_addmod_recall",
    "avg_added_recall",
    "avg_modified_recall",
    "avg_unchanged_recall",
    # ── 복사기(copy baseline) 대비 gain — copy_baseline_metrics.json ────────
    # 같은 행에 current 를 그대로 베낀 "복사기" 예측을 넣어 채점하고 그 차를 뺀 값이다
    # (`scripts/_copy_baseline_eval.py`). **모델 − 복사기** 이므로 0 이 기준이고 음수면
    # 복사기보다 못한 것이다 — 하드 제약 13b 가 요구하는 "예측인가 입력 복사인가"를
    # 한 열로 답한다. 절대값(avg_addmod_recall)은 복사기도 높게 받을 수 있어 단독으로는
    # 그 구분이 안 된다.
    # `n_gain_*` 는 **model·copy 양쪽이 정의된 행의 교집합** 크기다. 조건부 정의 지표라
    # 세팅마다 분모가 달라질 수 있어 gain 과 반드시 함께 읽는다 (분모가 갈리면 두 gain
    # 은 서로 다른 population 위의 평균이다). 정본이 `avg_`/`n_` 를 쌍으로 내는 이유다.
    "avg_gain_addmod_recall",
    "n_gain_addmod_recall",
    # change 축 — "변화 자체를 맞혔나". 위 recall 층은 GT 요소를 분모로 잡아 **없어져야
    # 할 요소**를 못 세고, hit 판정이 매칭뿐이라 자리만 맞고 내용이 틀려도 맞힌 게 된다.
    # change_f1_strict 은 pred/gt 양쪽에서 같은 절차로 변화 항목을 뽑아 내용 일치까지 본다.
    # avg_n_change_gt 를 함께 읽어야 "몇 개 중 몇 개"인지 드러난다.
    # **avg_change_f1_floor 없이 avg_change_f1_strict 만 읽지 말 것** — 이 축의 바닥은
    # 0 이 아니다. 빈 예측은 current 를 전부 지운 것으로 분류돼 gt_deleted 와 공짜로
    # 겹치고, 그 바닥값이 데이터에 따라 0.2~0.4 다. 실제로 EXP07v1 학습 모델(0.114)은
    # 바닥(0.258)에 진다 — floor 열을 빼면 그게 "base > trained" 로 읽힌다.
    # change_prec 은 2026-08-03 부터 계산되고 있었는데 노출이 안 돼 있었다. 과대변화
    # (있지도 않은 변화를 지어내는 실패)는 copy_excess 가 못 잡고 이 열만 잡는다.
    "avg_change_prec_strict",
    "avg_change_recall_strict",
    "avg_change_f1_strict",
    # loose 는 τ 게이트 없이 자리만 본다. strict 와의 **갭**이 "자리는 찾았는데 내용이
    # 틀린" 양이다 — 둘을 나란히 두지 않으면 그 구분이 사라진다.
    "avg_change_f1_loose",
    "avg_change_f1_floor",
    # 복사기 대비 gain. floor(퇴화 바닥)와 **다른 눈금**이다 — floor 는 "아무것도
    # 재현하지 않는 예측"의 상한이고 이건 "current 를 통째로 베낀 예측"과의 차다.
    # 복사기는 정의상 change 를 하나도 주장하지 않아 이 축이 낮으므로, gain 이 0 근처면
    # "변화를 맞히는 능력이 복사기와 다르지 않다"는 뜻이다.
    # `change_f1_floor` 는 예측과 무관한 눈금이라 정본이 gain 을 만들지 않는다
    # (`_copy_baseline_eval._NO_GAIN_KEYS`) — 여기 없는 것이 정상이다.
    "avg_gain_change_f1_strict",
    "avg_gain_change_f1_loose",
    "n_gain_change_f1_strict",
    "avg_n_change_gt",
    "avg_n_change_pred",
    "avg_addmod_f1",
    "avg_copy_rate_pred",
    "avg_copy_rate_gt",
    "avg_copy_excess",
    "copy_near_rate",
    # 예측에서 element 를 하나도 못 뽑은 행의 비율. **avg_copy_excess 와 반드시 함께
    # 읽는다** — 그 행들은 copy_excess 평균에서 빠지므로, 이 값이 모델마다 다르면
    # copy_excess 를 서로 다른 population 위에서 비교하게 된다.
    "parse_fail_rate",
    "parse_fail_long_rate",
    # GT 가 current 와 같은 행("화면이 안 바뀌는 step")에서의 정확도. 그 구간에서는
    # 복사가 정답이라 다른 지표가 전부 None 이다 — 이 열이 없으면 아무도 안 잰다.
    "avg_no_change_acc",
    "avg_hungarian_ea",
    "avg_hungarian_f1",
    # hungarian gain 은 전 행에서 정의된다 (None 이 없어 `n_gain_*` 가 total 과 같다)
    # → 분모 열을 따로 싣지 않는다. 실측상 대부분 음수다: 복사기가 hungarian_f1 을
    # 매우 높게 받기 때문이고, 그게 정확히 하드 제약 13b 가 말하는 바다.
    "avg_gain_hungarian_f1",
    "avg_hungarian_prec",
    "avg_hungarian_rec",
    "avg_hungarian_text",
    "avg_hungarian_idx",
    "avg_hungarian_pos",
    "unclosed_root_rate",
]

# 하위호환 폴백 — 2026-08-04 개명 전 34개 leaf 의 state_diff_metrics.json 은 옛
# 키만 갖고 있고 재빌드하지 않는다(`scripts/_state_diff_eval.py` aggregate() 의
# alias 는 **새로 채점하는 것에만** 붙는다). 그래서 이 방향(새 이름 우선, 옛 이름
# 폴백)으로 읽지 않으면 기존 leaf 의 이 컬럼들이 전부 빈 칸으로 렌더링된다.
_LEGACY_METRIC_FALLBACK = {
    "avg_addmod_recall": "avg_diff_recall",
    "avg_addmod_prec": "avg_diff_prec",
    "avg_addmod_f1": "avg_diff_f1",
    "avg_change_f1_floor": "avg_change_f1_null",
    "avg_change_f1_strict": "avg_change_f1",
    "avg_change_prec_strict": "avg_change_prec",
    "avg_change_recall_strict": "avg_change_recall",
}


def _metric(d: dict, k: str):
    """`d[k]` 를 새 이름 우선으로 찾고, 없으면 개명 전 옛 이름으로 폴백한다."""
    if k in d:
        return d[k]
    legacy = _LEGACY_METRIC_FALLBACK.get(k)
    return d.get(legacy) if legacy else None


# state leaf 의 metric_files 에 공통으로 붙는 state-diff 산출. section 은 호출부에서
# hungarian_metrics.json 과 **같은 값**을 준다 — load_metrics 의 section 조회는 부재를
# silent skip 하므로, 어긋나면 표에 빈칸만 뜨고 아무도 오류를 못 본다.
def _state_diff_file(section: str | None) -> tuple[str, str | None]:
    return ("state_diff_metrics.json", section)


# 복사기 기준선 산출도 state leaf 옆에 이미 있다 (`scripts/_copy_baseline_eval.py`).
# 이 파일은 섹션이 **2단**이다 — `{section: {copy_baseline, model, gain}}` — 이라
# 1단 section 조회로는 numeric scalar 가 하나도 안 잡혀 조용히 빈 dict 가 된다.
# `load_metrics` 의 점(dot) 경로가 그 두 번째 단이다.
#
# `gain` 만 싣는다: `model` 은 hungarian/state_diff 파일과 같은 수를 다른 이름으로
# 한 번 더 싣는 것이고, `copy_baseline` 절대값은 세팅마다 같은 값(예측과 무관한
# 기준선)이라 세팅 비교 표에서 열 하나를 통째로 낭비한다. 판별량은 차(gain)다.
def _copy_baseline_file(section: str) -> tuple[str, str]:
    return ("copy_baseline_metrics.json", f"{section}.gain")


ACTION_METRIC_KEYS = [
    "total",
    "parse_rate",
    "type_accuracy",
    "step_accuracy",
    "macro_step_accuracy",
    "cond_index_acc",
    "cond_bbox_acc",
    "cond_dir_acc",
    "cond_app_acc",
    "cond_text_acc",
    "no_bbox_n",
    "predict_bleu-4",
    "predict_rouge-l",
]


def _ac_stage1_entries(exp: str) -> dict:
    """AC_EXP01/AC_EXP02 stage1 dual-task entries (ID/OOD × state/action × ±without-open_app)."""
    ds = DS_DATADIR[exp]
    data = REPO / "data" / ds
    actual = f"on-{exp}"  # on-AC_EXP01 / on-AC_EXP02
    return {
        "on-AC-state-id": {
            "dir": f"{actual}-state",
            "pred": "generated_predictions_id.jsonl",
            "test": data / "stage1_test_id_state.jsonl",
            "metric_files": [
                ("predict_results_id.json", None),
                ("hungarian_metrics.json", "in_domain"),
                _state_diff_file("in_domain"),
                _copy_baseline_file("in_domain"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-AC-state-ood": {
            "dir": f"{actual}-state",
            "pred": "generated_predictions_ood.jsonl",
            "test": data / "stage1_test_ood_state.jsonl",
            "metric_files": [
                ("predict_results_ood.json", None),
                ("hungarian_metrics.json", "out_of_domain"),
                _state_diff_file("out_of_domain"),
                _copy_baseline_file("out_of_domain"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-AC-state-id-without-open_app": {
            "dir": f"{actual}-state-without-open_app",
            "pred": "generated_predictions_id.jsonl",
            "test": data / "stage1_test_id_state_without_open_app.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", "in_domain"),
                _state_diff_file("in_domain"),
                _copy_baseline_file("in_domain"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-AC-state-ood-without-open_app": {
            "dir": f"{actual}-state-without-open_app",
            "pred": "generated_predictions_ood.jsonl",
            "test": data / "stage1_test_ood_state_without_open_app.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", "out_of_domain"),
                _state_diff_file("out_of_domain"),
                _copy_baseline_file("out_of_domain"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-AC-action-id": {
            "dir": f"{actual}-action",
            "pred": "generated_predictions_id.jsonl",
            "test": data / "stage1_test_id_action.jsonl",
            "metric_files": [
                ("predict_results_id.json", None),
                ("action_metrics.json", "in_domain"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
        "on-AC-action-ood": {
            "dir": f"{actual}-action",
            "pred": "generated_predictions_ood.jsonl",
            "test": data / "stage1_test_ood_action.jsonl",
            "metric_files": [
                ("predict_results_ood.json", None),
                ("action_metrics.json", "out_of_domain"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
    }


def _mb_stage1_entries() -> dict:
    data = REPO / "data" / "MobiBench"
    return {
        "on-MB": {
            "dir": "on-MB",
            "pred": "generated_predictions.jsonl",
            "test": data / "stage1.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", None),  # single-pair: top-level flat
                ("hungarian_metrics.json", "overall"),  # 호환: 혹시 nested 면 overall
                _state_diff_file(None),
                _state_diff_file("overall"),
                # single-pair(MB/MC)도 copy_baseline 은 3-섹션 스키마를 쓰되 `overall`
                # 만 채운다 — state_diff 처럼 top-level flat 변형이 없다.
                _copy_baseline_file("overall"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-MB-without-open_app": {
            "dir": "on-MB-without-open_app",
            "pred": "generated_predictions.jsonl",
            "test": data / "stage1_without_open_app.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", None),
                ("hungarian_metrics.json", "overall"),
                _state_diff_file(None),
                _state_diff_file("overall"),
                # single-pair(MB/MC)도 copy_baseline 은 3-섹션 스키마를 쓰되 `overall`
                # 만 채운다 — state_diff 처럼 top-level flat 변형이 없다.
                _copy_baseline_file("overall"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
    }


def _mc_stage1_entries() -> dict:
    data = REPO / "data" / "MonkeyCollection"
    return {
        "on-MC": {
            "dir": "on-MC",
            "pred": "generated_predictions.jsonl",
            "test": data / "stage1_test.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", None),
                ("hungarian_metrics.json", "overall"),
                _state_diff_file(None),
                _state_diff_file("overall"),
                # single-pair(MB/MC)도 copy_baseline 은 3-섹션 스키마를 쓰되 `overall`
                # 만 채운다 — state_diff 처럼 top-level flat 변형이 없다.
                _copy_baseline_file("overall"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        # MC 의 without-open_app GT 는 data/MonkeyCollection/ 에 없으므로 등록하지 않는다.
    }


def _exp08_stage1_entries() -> dict:
    """AC_EXP08 stage1 entries — ID/OOD 가 없는 단일 test 계열.

    leaf 4 종 (state_full / state_masked / state_dropped / action) 이 각각 단일 파일
    1-회 추론이라 섹션이 overall 하나뿐이다 (MB/MC 와 같은 single-pair 모드). state leaf
    3 종은 `-without-open_app` sibling 도 함께 등록한다 (stage1_eval.sh 가 산출한다).
    디렉토리 이름은 stem 의 `_` 를 `-` 로 바꾼 on-AC_EXP08-state-full 형태.
    """
    data = REPO / "data" / DS_DATADIR["AC_EXP08"]
    entries: dict[str, dict] = {}
    for stem in ("state_full", "state_masked", "state_dropped"):
        leaf = stem.replace("_", "-")
        for suffix in ("", "-without-open_app"):
            test_stem = f"stage1_test_{stem}" + (
                "_without_open_app" if suffix else ""
            )
            entries[f"on-AC-{leaf}{suffix}"] = {
                "dir": f"on-AC_EXP08-{leaf}{suffix}",
                "pred": "generated_predictions.jsonl",
                "test": data / f"{test_stem}.jsonl",
                "metric_files": [
                    ("predict_results.json", None),
                    ("hungarian_metrics.json", None),  # single-pair: top-level flat
                    ("hungarian_metrics.json", "overall"),  # 호환: nested 면 overall
                    _state_diff_file(None),
                    _state_diff_file("overall"),
                    # single-pair 도 copy_baseline 은 3-섹션 스키마를 쓰되 overall 만 채운다.
                    _copy_baseline_file("overall"),
                ],
                "metric_keys": STATE_METRIC_KEYS,
            }
    entries["on-AC-action"] = {
        "dir": "on-AC_EXP08-action",
        "pred": "generated_predictions.jsonl",
        "test": data / "stage1_test_action.jsonl",
        "metric_files": [
            ("predict_results.json", None),
            ("action_metrics.json", None),
            ("action_metrics.json", "overall"),
        ],
        "metric_keys": ACTION_METRIC_KEYS,
    }
    return entries


def _exp08_stage2_entries() -> dict:
    """AC_EXP08 stage2 entries — 단일 stage2_test.jsonl (overall 1-섹션)."""
    data = REPO / "data" / DS_DATADIR["AC_EXP08"]
    return {
        "on-AC": {
            "dir": "on-AC_EXP08",
            "pred": "generated_predictions.jsonl",
            "test": data / "stage2_test.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("action_metrics.json", None),
                ("action_metrics.json", "overall"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
    }


def _ac_stage2_entries(exp: str) -> dict:
    ds = DS_DATADIR[exp]
    data = REPO / "data" / ds
    actual = f"on-{exp}"
    return {
        "on-AC-id": {
            "dir": actual,
            "pred": "generated_predictions_id.jsonl",
            "test": data / "stage2_test_id.jsonl",
            "metric_files": [
                ("predict_results_id.json", None),
                ("action_metrics.json", "in_domain"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
        "on-AC-ood": {
            "dir": actual,
            "pred": "generated_predictions_ood.jsonl",
            "test": data / "stage2_test_ood.jsonl",
            "metric_files": [
                ("predict_results_ood.json", None),
                ("action_metrics.json", "out_of_domain"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
    }


def _mb_stage2_entries() -> dict:
    data = REPO / "data" / "MobiBench"
    return {
        "on-MB": {
            "dir": "on-MB",
            "pred": "generated_predictions.jsonl",
            "test": data / "stage2.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("action_metrics.json", None),
                ("action_metrics.json", "overall"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
    }


STAGE_CONFIG: dict[int, dict] = {
    1: {"eval_subdir": "stage1_eval"},
    2: {"eval_subdir": "stage2_eval"},
}

# (stage, EXP) → {logical_key: entry}
EVAL_DATASETS: dict[int, dict[str, dict[str, dict]]] = {
    1: {
        "AC_EXP01": {
            **_ac_stage1_entries("AC_EXP01"),
            **_mb_stage1_entries(),
            **_mc_stage1_entries(),
        },
        "AC_EXP02": {
            **_ac_stage1_entries("AC_EXP02"),
            **_mb_stage1_entries(),
            **_mc_stage1_entries(),
        },
        "AC_EXP03": {
            **_ac_stage1_entries("AC_EXP03"),
            **_mb_stage1_entries(),
            **_mc_stage1_entries(),
        },
        "AC_EXP04": {
            **_ac_stage1_entries("AC_EXP04"),
            **_mb_stage1_entries(),
            **_mc_stage1_entries(),
        },
        "AC_EXP05": {
            **_ac_stage1_entries("AC_EXP05"),
            **_mb_stage1_entries(),
            **_mc_stage1_entries(),
        },
        # AC_EXP06 은 Stage 2 전용 대조군이라 stage1 entry 가 없다.
        "AC_EXP07_v1": _ac_stage1_entries("AC_EXP07_v1"),
        "AC_EXP07_v2": _ac_stage1_entries("AC_EXP07_v2"),
        # AC_EXP08 은 ID/OOD 가 없어 _ac_stage1_entries (id/ood 전제) 를 쓸 수 없다.
        "AC_EXP08": _exp08_stage1_entries(),
        "MC": {**_mc_stage1_entries(), **_mb_stage1_entries()},
    },
    2: {
        "AC_EXP01": {**_ac_stage2_entries("AC_EXP01"), **_mb_stage2_entries()},
        "AC_EXP02": {**_ac_stage2_entries("AC_EXP02"), **_mb_stage2_entries()},
        "AC_EXP03": {**_ac_stage2_entries("AC_EXP03"), **_mb_stage2_entries()},
        "AC_EXP05": _ac_stage2_entries("AC_EXP05"),
        "AC_EXP06": _ac_stage2_entries("AC_EXP06"),
        "AC_EXP07_v1": _ac_stage2_entries("AC_EXP07_v1"),
        "AC_EXP07_v2": _ac_stage2_entries("AC_EXP07_v2"),
        "AC_EXP08": _exp08_stage2_entries(),
    },
}


PROMPT_RE = re.compile(
    r"^system\n(?P<sys>.*?)\nuser\n\n## Current State\n(?P<xml>.*?)\n\n## Action\n(?P<act>.*?)\nassistant\n?$",
    re.DOTALL,
)


def split_prompt(prompt: str) -> tuple[str, str, str]:
    m = PROMPT_RE.match(prompt)
    if not m:
        return prompt, "", ""
    return m.group("sys"), m.group("xml"), m.group("act")


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def action_oneliner(act_json: str) -> str:
    try:
        a = json.loads(act_json)
    except Exception:
        return act_json[:80].replace("\n", " ")
    t = a.get("type", "?")
    params = a.get("params", {})
    extras = []
    if "index" in a:
        extras.append(f"index={a['index']}")
    if "default" in a:
        extras.append(f"default={a['default']}")
    p = json.dumps(params, ensure_ascii=False) if params else ""
    return f"{t}  {p}  {' '.join(extras)}".strip()


def fmt_num(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return str(v) if v is not None else ""


def load_metrics(target_dir: Path, metric_files: list[tuple[str, str | None]]) -> dict:
    """metric_files = [(filename, section_or_None)] 을 차례로 읽어 flat dict 로 합친다.

    section 이 None 이면 JSON top-level 의 numeric scalar 만 merge.
    section 이 str 이면 JSON[section] 이 dict 일 때만 그 안의 numeric scalar 만 merge.
    section 에 점이 있으면 그만큼 내려간다 (`"in_domain.gain"`) — copy_baseline
    산출물이 `{section: {copy_baseline, model, gain}}` 2단이라 필요하다.
    파일/섹션 부재는 silent skip.
    """
    merged: dict = {}
    for fn, section in metric_files:
        p = target_dir / fn
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if section is not None:
            for part in section.split("."):
                if not isinstance(data, dict):
                    break
                data = data.get(part)
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if not isinstance(v, (dict, list)):
                merged.setdefault(k, v)
    return merged


def discover_variants(
    eval_root: Path, actual_dir: str, pred_filename: str
) -> list[str]:
    """eval_root 아래에서 `{variant_path}/{actual_dir}/{pred_filename}` 가 존재하는 variant_path 들을 찾는다.

    variant_path 는 1-level (예: `base`) 또는 2-level (예: `lora_world-model/epoch-3`).
    """
    found: list[str] = []
    if not eval_root.is_dir():
        return found

    def has_target(v: Path) -> bool:
        return (v / actual_dir / pred_filename).is_file()

    for child in sorted(eval_root.iterdir()):
        if not child.is_dir():
            continue
        if has_target(child):
            found.append(child.name)
            continue
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and has_target(sub):
                found.append(f"{child.name}/{sub.name}")
    return found


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; margin: 1rem; color: #222; }
h1 { font-size: 18px; margin: 0 0 8px; }
.meta { color: #666; font-size: 12px; margin-bottom: 12px; }
#variant-controls { padding: 8px 0 12px; font-size: 12px; border-bottom: 1px solid #eee; margin-bottom: 12px; }
#variant-controls strong { margin-right: 8px; }
#variant-controls label { margin-right: 12px; cursor: pointer; white-space: nowrap; display: inline-block; }
#variant-controls .actions { margin-left: 8px; }
#variant-controls button { font-size: 11px; padding: 2px 6px; cursor: pointer; }
table.metric { border-collapse: collapse; margin-bottom: 16px; font-size: 12px; }
table.metric th, table.metric td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
table.metric th:first-child, table.metric td:first-child { text-align: left; font-weight: 600; }
details.row { border: 1px solid #e0e0e0; border-radius: 6px; margin: 4px 0; padding: 4px 10px; }
details.row[open] { background: #fafbfc; }
details.row > summary { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; cursor: pointer; line-height: 1.5; list-style: none; }
details.row > summary::-webkit-details-marker { display: none; }
details.row > summary::before { content: "\\25B8 "; color: #888; }
details.row[open] > summary::before { content: "\\25BE "; }
.idx { color: #888; }
.img { color: #0a64ad; }
.act { color: #444; }
.grid { display: grid; grid-template-columns: 1.3fr 1fr repeat(var(--n-active, 1), minmax(0, 1fr)); gap: 8px; margin: 8px 0 4px; }
.grid section { border: 1px solid #eee; border-radius: 4px; padding: 6px 8px; min-width: 0; background: #fff; }
.grid h3 { margin: 0 0 4px; font-size: 12px; color: #555; font-weight: 600; }
.grid pre { white-space: pre; overflow-x: auto; font-size: 11px; line-height: 1.4; margin: 4px 0; background: #f7f7f9; padding: 6px; border-radius: 3px; font-family: ui-monospace, Menlo, Consolas, monospace; }
pre.sys { background: #fff7e6; }
pre.action { background: #e6f4ff; }
pre.imginfo { background: #f0f0f0; color: #555; }
.col-input pre.xml { max-height: 480px; overflow: auto; }
.col-label pre.xml, .col-pred pre.xml { max-height: 480px; overflow: auto; }
"""

JS = """
(function () {
  const controls = document.querySelectorAll('#variant-controls input[type=checkbox]');
  function refresh() {
    const active = new Set();
    controls.forEach(cb => { if (cb.checked) active.add(cb.dataset.variant); });
    document.querySelectorAll('[data-variant]').forEach(el => {
      el.style.display = active.has(el.dataset.variant) ? '' : 'none';
    });
    document.documentElement.style.setProperty('--n-active', Math.max(1, active.size));
  }
  controls.forEach(cb => cb.addEventListener('change', refresh));
  document.getElementById('btn-all').addEventListener('click', () => {
    controls.forEach(cb => { cb.checked = true; });
    refresh();
  });
  document.getElementById('btn-none').addEventListener('click', () => {
    controls.forEach(cb => { cb.checked = false; });
    refresh();
  });
  refresh();
})();
"""


def variant_label(exp: str, model: str, variant_path: str, multi: bool) -> str:
    """multi-spec 일 때는 `[EXP] model/variant_path`, 단일이면 `variant_path` 그대로."""
    if multi:
        return f"[{exp}] {model}/{variant_path}"
    return variant_path


def build_dataset(
    stage: int,
    logical_key: str,
    spec_variants: list[tuple[str, str, str, dict, Path]],
    multi: bool,
) -> tuple[str, dict, int]:
    """spec_variants = [(exp, model, variant_path, entry, eval_root)]"""
    metric_keys = spec_variants[0][3]["metric_keys"]

    # predictions 적재. 첫 spec 의 entry 의 test 파일이 있으면 anchor 로 사용.
    pred_lists: dict[str, list[dict]] = {}
    metrics_by_label: dict[str, dict] = {}
    anchor_test: Path | None = None

    for exp, model, vpath, entry, eval_root in spec_variants:
        label = variant_label(exp, model, vpath, multi)
        target_dir = eval_root / vpath / entry["dir"]
        recs = read_jsonl(target_dir / entry["pred"])
        pred_lists[label] = recs
        metrics_by_label[label] = load_metrics(target_dir, entry["metric_files"])
        if anchor_test is None:
            tp = entry.get("test")
            if tp is not None and Path(tp).is_file():
                anchor_test = Path(tp)

    # 행 수 일관성 검증 — 모든 prediction 의 row 수가 같아야 같은 인덱스로 정렬됨.
    lengths = {label: len(recs) for label, recs in pred_lists.items()}
    n_set = set(lengths.values())
    if len(n_set) > 1:
        raise SystemExit(
            f"stage{stage}/{logical_key}: prediction row count mismatch — {lengths}. "
            "EXP01/EXP02 stage 데이터는 byte-identical copy 여야 cross-compare 가 가능합니다."
        )
    n = n_set.pop()

    # images[]: anchor_test 가 있으면 거기서, 없으면 빈 문자열.
    if anchor_test is not None:
        test_recs = read_jsonl(anchor_test)
        if len(test_recs) != n:
            raise SystemExit(
                f"stage{stage}/{logical_key}: anchor test ({anchor_test.relative_to(REPO)}) "
                f"len {len(test_recs)} != predictions len {n}"
            )
        images = [
            r.get("images", [""])[0] if r.get("images") else "" for r in test_recs
        ]
    else:
        images = ["" for _ in range(n)]

    labels = list(pred_lists.keys())

    cb_html = "".join(
        f'<label><input type="checkbox" data-variant="{esc(lab)}"'
        f"{' checked' if i < 4 else ''}> {esc(lab)}</label>"
        for i, lab in enumerate(labels)
    )
    controls = (
        '<div id="variant-controls"><strong>모델 선택:</strong>'
        f"{cb_html}"
        '<span class="actions">'
        '<button id="btn-all" type="button">all</button> '
        '<button id="btn-none" type="button">none</button>'
        "</span></div>"
    )

    metric_header = "".join(f"<th>{esc(k)}</th>" for k in metric_keys)
    metric_body = ""
    for lab in labels:
        d = metrics_by_label[lab]
        cells = "".join(f"<td>{fmt_num(_metric(d, k))}</td>" for k in metric_keys)
        metric_body += f'<tr data-variant="{esc(lab)}"><th>{esc(lab)}</th>{cells}</tr>'
    metric_table = (
        f'<table class="metric"><thead><tr><th>variant</th>{metric_header}</tr></thead>'
        f"<tbody>{metric_body}</tbody></table>"
    )

    anchor_rel = (
        str(anchor_test.relative_to(REPO))
        if anchor_test is not None
        else "(no GT jsonl — prediction file 의 prompt/label 만 사용)"
    )
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Eval pairs · stage{stage} · {logical_key}</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>Eval pairs · stage{stage} · {logical_key} · n={n}</h1>",
        f'<div class="meta">test: {esc(anchor_rel)}<br>',
        f"variants: {len(labels)} (multi-spec)"
        if multi
        else f"variants: {len(labels)} (single-spec)",
        "</div>",
        controls,
        metric_table,
    ]

    base_label = labels[0]
    for i in range(n):
        anchor = pred_lists[base_label][i]
        sys_msg, cur_xml, act_json = split_prompt(anchor.get("prompt", ""))
        label_xml = anchor.get("label", "")
        summary_line = (
            f'<span class="idx">#{i:04d}</span> · '
            f'<span class="img">{esc(images[i])}</span> · '
            f'<span class="act">{esc(action_oneliner(act_json))}</span>'
        )
        parts.append(f'<details class="row"><summary>{summary_line}</summary>')
        parts.append('<div class="grid">')
        parts.append(
            '<section class="col-input"><h3>Input</h3>'
            f'<pre class="imginfo">image: {esc(images[i])}</pre>'
            f'<pre class="sys">{esc(sys_msg)}</pre>'
            f'<pre class="xml">{esc(cur_xml)}</pre>'
            f'<pre class="action">{esc(act_json)}</pre>'
            "</section>"
        )
        parts.append(
            f'<section class="col-label"><h3>Label</h3><pre class="xml">{esc(label_xml)}</pre></section>'
        )
        for lab in labels:
            pred = pred_lists[lab][i].get("predict", "")
            parts.append(
                f'<section class="col-pred" data-variant="{esc(lab)}">'
                f"<h3>{esc(lab)}</h3>"
                f'<pre class="xml">{esc(pred)}</pre>'
                "</section>"
            )
        parts.append("</div></details>")

    parts.append(f"<script>{JS}</script>")
    parts.append("</body></html>")
    return "".join(parts), metrics_by_label, n


def build_summary_md(
    stage: int,
    out_root: Path,
    per_ds: dict[str, tuple[list[str], dict[str, dict], int]],
    spec_specs: list[tuple[str, str]],
) -> str:
    out = [f"# Eval pairs summary · stage{stage}", ""]
    out.append(f"- out_root: `{out_root.relative_to(REPO)}`")
    out.append("- script: `scripts/eval_viewer.py`")
    out.append(f"- include: {', '.join(f'`{e}:{m}`' for e, m in spec_specs)}")
    out.append("")
    for ds, (metric_keys, metrics_by_label, n) in per_ds.items():
        out.append(f"## {ds} (n={n})")
        out.append("")
        out.append("| variant | " + " | ".join(metric_keys) + " |")
        out.append("|" + "---|" * (len(metric_keys) + 1))
        for lab, d in metrics_by_label.items():
            row = (
                "| "
                + lab
                + " | "
                + " | ".join(fmt_num(_metric(d, k)) for k in metric_keys)
                + " |"
            )
            out.append(row)
        out.append("")
    return "\n".join(out)


def task_of(entry: dict) -> str:
    """logical entry → 사이트 kind ('state' | 'action').

    화면 레이아웃은 여기서 정하지 않는다 — EXP07 stage1 `-action` 처럼 같은 kind 인데
    프롬프트 구조가 다른 경우가 있어 `_compare_site.detect_layout` 이 실물로 판정한다.
    """
    return "state" if entry["metric_keys"] == STATE_METRIC_KEYS else "action"


def run_site_mode(args: argparse.Namespace, specs: list[tuple[str, str]]) -> None:
    """EXP 별 정성 비교 사이트를 outputs/_compare/ 아래에 만든다.

    site 는 EXP 단위다 — 같은 EXP 의 여러 MODEL 은 하나의 사이트 안에서 setting 으로
    나란히 놓인다 (같은 test 를 쓰므로 행이 정렬된다). EXP 가 다르면 test 자체가
    달라 행이 정렬되지 않으므로 사이트를 나눈다.
    """
    compare_root = REPO / "outputs" / "_compare"
    built = 0

    by_exp: dict[str, list[str]] = {}
    for exp, model in specs:
        by_exp.setdefault(exp, []).append(model)

    for stage in args.stages:
        eval_subdir = STAGE_CONFIG[stage]["eval_subdir"]
        for exp, models in by_exp.items():
            entries = EVAL_DATASETS[stage].get(exp, {})
            if not entries:
                print(f"skip site stage{stage}/{exp}: 등록된 logical key 없음")
                continue

            groups: dict[str, list[str]] = {}
            for key, entry in entries.items():
                if args.datasets is not None and key not in args.datasets:
                    continue
                groups.setdefault(task_of(entry), []).append(key)

            key_order = {k: i for i, k in enumerate(entries)}
            for task, keys in groups.items():
                settings_by_id: dict[str, dict] = {}
                splits: list[dict] = []
                for key in sorted(keys, key=lambda k: key_order[k]):
                    entry = entries[key]
                    dirs: dict[str, Path] = {}
                    metrics: dict[str, dict] = {}
                    ids: list[str] = []
                    for model in models:
                        eval_root = (
                            REPO
                            / "outputs"
                            / DS_DATADIR[exp]
                            / "eval"
                            / model
                            / eval_subdir
                        )
                        discovered = discover_variants(
                            eval_root, entry["dir"], entry["pred"]
                        )
                        if args.variants is not None:
                            discovered = [v for v in discovered if v in args.variants]
                        for v in discovered:
                            sid = f"{model}|{v}"
                            dirs[sid] = eval_root / v / entry["dir"]
                            metrics[sid] = load_metrics(
                                dirs[sid], entry["metric_files"]
                            )
                            ids.append(sid)
                            settings_by_id.setdefault(
                                sid, {"id": sid, "model": model, "vpath": v}
                            )
                    if not ids:
                        continue
                    splits.append(
                        {
                            "key": key,
                            "pred_filename": entry["pred"],
                            "test_path": entry.get("test"),
                            "setting_ids": ids,
                            "dirs": dirs,
                            "metrics": metrics,
                        }
                    )

                if not splits:
                    print(f"skip site stage{stage}/{exp}/{task}: variant 없음")
                    continue

                multi_model = len({s["model"] for s in settings_by_id.values()}) > 1
                settings = sorted(
                    settings_by_id.values(),
                    key=lambda s: _compare_site.setting_sort_key(
                        s["model"], s["vpath"]
                    ),
                )
                for s in settings:
                    s["label"] = _compare_site.humanize_setting(
                        s["model"], s["vpath"], multi_model
                    )
                rank = {s["id"]: i for i, s in enumerate(settings)}
                for sp in splits:
                    sp["setting_ids"].sort(key=lambda sid: rank[sid])

                out_dir = compare_root / _compare_site.site_dirname(exp, stage, task)
                stats = _compare_site.build_site(
                    stage=stage,
                    exp=exp,
                    kind=task,
                    settings=settings,
                    splits=splits,
                    metric_keys=entries[splits[0]["key"]]["metric_keys"],
                    out_dir=out_dir,
                    samples=args.samples,
                    seed=args.seed,
                )
                built += 1
                print(
                    f"wrote {stats['path'].relative_to(REPO)}  "
                    f"settings={stats['n_settings']}  splits={stats['n_splits']}  "
                    f"samples={stats['n_samples']}  "
                    f"size={stats['size'] / 1024 / 1024:.1f}MB"
                )

    if built:
        idx = _compare_site.write_root_index(compare_root)
        if idx is not None:
            print(f"wrote {idx.relative_to(REPO)}")
    else:
        print("사이트를 하나도 만들지 못했습니다 — --include / --stages 를 확인하세요.")


def parse_spec(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise SystemExit(
            f"--include 항목 '{s}' 은 EXP:MODEL 형식이어야 함 (예: AC_EXP02:qwen3-vl-8b)."
        )
    exp, model = s.split(":", 1)
    exp = exp.strip()
    model = model.strip()
    if exp not in DS_DATADIR:
        raise SystemExit(
            f"--include 항목 '{s}' 의 EXP '{exp}' 미등록 — 허용: {sorted(DS_DATADIR)}"
        )
    if not model:
        raise SystemExit(f"--include 항목 '{s}' 의 MODEL 이 비어있음.")
    return exp, model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--include",
        nargs="+",
        required=True,
        metavar="EXP:MODEL",
        help="비교할 (EXP, MODEL) 쌍. 1개면 단일-EXP 모드, 2개 이상이면 cross-EXP 모드. "
        "EXP ∈ {AC_EXP01, AC_EXP02, AC_EXP03, AC_EXP04, AC_EXP05, AC_EXP08, MC}, MODEL = outputs/<DS_DATADIR(EXP)>/eval/ 아래 디렉토리 명. "
        "예: --include AC_EXP01:qwen3-vl-8b_ratio73 AC_EXP02:qwen3-vl-8b",
    )
    p.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=[1, 2],
        default=[1, 2],
        help="처리할 stage (기본 1 2 모두).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="LOGICAL_KEY",
        help="처리할 logical key (예: on-AC-state-id, on-MB). 기본 = 각 EXP 가 가진 logical key 합집합.",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="처리할 variant_path 화이트리스트 (예: base 'lora_world-model/epoch-3'). 기본 = auto-discover.",
    )
    p.add_argument(
        "--site",
        action="store_true",
        help="EXP 별 정성 비교 사이트(index.html + README.md)를 outputs/_compare/ 에 만든다. "
        "stage × task(state/action) 별로 한 벌씩, 표본은 시드 고정 추출.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=20,
        help="--site 모드에서 분할(ID/OOD/woa)마다 뽑을 표본 수 (기본 20).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="--site 모드의 표본 추출 시드 (기본 42 — 같은 시드면 같은 표본).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs: list[tuple[str, str]] = [parse_spec(s) for s in args.include]
    multi = len(specs) > 1

    if args.site:
        run_site_mode(args, specs)
        return

    for stage in args.stages:
        eval_subdir = STAGE_CONFIG[stage]["eval_subdir"]

        # 처리할 logical key 결정
        if args.datasets is not None:
            logical_keys = list(args.datasets)
        else:
            seen: set[str] = set()
            logical_keys = []
            for exp, _ in specs:
                for k in EVAL_DATASETS[stage].get(exp, {}):
                    if k not in seen:
                        seen.add(k)
                        logical_keys.append(k)

        if not logical_keys:
            print(f"skip stage{stage}: spec 들이 가진 logical key 없음")
            continue

        # 출력 경로 분기
        if multi:
            out_root = REPO / "outputs" / "_compare" / eval_subdir
        else:
            exp, model = specs[0]
            out_root = REPO / "outputs" / DS_DATADIR[exp] / "eval" / model / eval_subdir
        out_root.mkdir(parents=True, exist_ok=True)

        per_ds: dict[str, tuple[list[str], dict[str, dict], int]] = {}
        for logical_key in logical_keys:
            spec_variants: list[tuple[str, str, str, dict, Path]] = []
            for exp, model in specs:
                entry = EVAL_DATASETS[stage].get(exp, {}).get(logical_key)
                if entry is None:
                    continue
                eval_root = (
                    REPO / "outputs" / DS_DATADIR[exp] / "eval" / model / eval_subdir
                )
                discovered = discover_variants(eval_root, entry["dir"], entry["pred"])
                if args.variants is not None:
                    discovered = [v for v in discovered if v in args.variants]
                for v in discovered:
                    spec_variants.append((exp, model, v, entry, eval_root))

            if not spec_variants:
                print(
                    f"skip stage{stage}/{logical_key}: no variants found across specs"
                )
                continue

            try:
                doc, metrics_by_label, n = build_dataset(
                    stage, logical_key, spec_variants, multi
                )
            except SystemExit:
                raise
            except Exception as e:
                print(f"error stage{stage}/{logical_key}: {e}")
                continue

            target = out_root / f"pairs_{logical_key}.html"
            target.write_text(doc)
            size_mb = target.stat().st_size / 1024 / 1024
            print(
                f"wrote {target.relative_to(REPO)}  rows={n}  variants={len(spec_variants)}  size={size_mb:.1f}MB"
            )
            per_ds[logical_key] = (
                spec_variants[0][3]["metric_keys"],
                metrics_by_label,
                n,
            )

        if per_ds:
            summary_path = out_root / "pairs_summary.md"
            summary_path.write_text(build_summary_md(stage, out_root, per_ds, specs))
            print(f"wrote {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
