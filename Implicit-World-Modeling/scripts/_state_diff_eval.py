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
   오독된다. 그래서 **헤드라인은 recall** 이다: `addmod_recall` = 실제로 바뀐 GT 요소 중
   몇 개를 예측이 맞혔나. 0~1 범위가 그대로 의미를 갖는다.
   (precision/F1 도 낸다 — 다만 분모를 pred-side diff 로 대칭화한 별도 키다. 아래 참고.)

2026-08-04 개명 — 옛 이름과의 관계
-----------------------------------
`diff_recall`/`diff_prec`/`diff_f1` → `addmod_recall`/`addmod_prec`/`addmod_f1`,
`change_f1`/`change_f1_null` → `change_f1_strict`/`change_f1_floor` 로 개명했다.
`diff_*` 는 ADDED∪MODIFIED 뿐이고 DELETED 가 빠져 있는데, 같은 파일의 `change_*` 는
DELETED 를 포함한다 — 두 접두어가 서로 다른 변화 집합인데 이름에서 구분이 안 됐다.
`_null` 은 "무엇의 null 인지" 모호하고 이 모듈엔 진짜 None 이 되는 지표가 따로 있어
(`copy_excess` 빈 예측, `addmod_f1` 조건부) 오독된다 — `_floor`(퇴화 바닥)가 실제 뜻이다.
`change_f1_strict` 는 훗날 "자리만 맞으면 hit" 인 loose 축이 추가될 때의 대칭을 위한
이름이다. 기존 34개 leaf 의 `state_diff_metrics.json` 을 재빌드하지 않기 위해
`aggregate()` 가 옛 키를 alias 로 함께 낸다 — 새 코드는 이 문서의 새 이름을 쓴다.

정본 지표와의 관계 — 층 분해
----------------------------
recall 계열은 `pred ↔ gt` **단 한 번의 Hungarian 매칭**(정본
`compute_hungarian_acc` 와 같은 매칭)을 GT diff 유형별로 쪼갠 것이다. 따라서

    (n_hit_unchanged + n_hit_modified + n_hit_added) / n_gt  ==  hungarian_rec

가 항등식으로 성립한다 (`tests/test_state_diff_eval.py` 가 검증). diff 부분집합만
따로 매칭하면 이 성질이 깨진다 — UNCHANGED 에 붙었어야 할 예측 요소가 MODIFIED 로
재배정되면서 recall 이 부풀기 때문이다. 그래서 부분집합 재매칭을 하지 않는다.

change 축 — recall 층 분해가 못 보는 두 가지
--------------------------------------------
층 분해는 GT 요소를 분모로 잡으므로 (a) **사라져야 할 요소를 지웠는가**를 볼 수 없고
(DELETED 는 GT 에 대응물이 없다), hit 판정이 Hungarian 매칭뿐이라 (b) 자리만 맞고
내용이 틀린 예측도 맞힌 것으로 센다 (매칭 임계 1.5/1.7 은 `_text_sim` 이 0 이어도
붙을 만큼 느슨하다). `change_f1_strict` 가 그 두 구멍을 메운다 — current 대비 변화 목록을
pred/gt 양쪽에서 **같은 절차**로 뽑아 집합 비교하고, 내용 일치를 τ(0.9)로 한 번 더
건다. 정의와 경계는 `compute_change_items` 참고.

매칭 기준 스위치 (`strict_pos` / `include_aria`)
------------------------------------------------
둘 다 **기본 꺼짐**이다 — 켜면 임계나 element 집합이 달라져 기존 산출물과 나란히
놓을 수 없다. 그리고 둘 다 **전역이 아니라 인자로** 흐른다: 이 모듈은
`_hungarian_eval` 의 매칭 함수를 그대로 쓰므로, 한쪽만 전역을 읽으면 정본 채점과
설정이 어긋나 위 항등식이 조용히 깨진다.

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

# change 축의 "내용까지 맞혔나" 임계. 매칭만으로는 부족하다 — 매칭 임계(1.5/1.7)는
# `_text_sim` 이 0 에 가까워도 붙을 만큼 느슨해서, "자리는 맞고 내용은 틀린" 예측이
# 변화를 맞힌 것으로 세어진다. 여기서 한 번 더 걸러야 change_f1_strict 이 내용 지표가 된다.
CHANGE_TEXT_SIM_TAU = 0.9

# `parse_fail_long` 의 경계. 이 길이를 넘겼는데 element 가 0 개면 "안 냈다"가 아니라
# **"파싱 가능한 태그가 하나도 없는 장문을 냈다"** 이다. 채점기는 둘을 같은 값으로
# 취급하므로(둘 다 `pred_els == []`) 이 플래그가 유일한 구분 수단이다.
PARSE_FAIL_LONG_CHARS = 100

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


# probe_forget 은 `eval/<model>/` 규약 밖 경로다 (`outputs/<EXP>/probe_forget/<variant>`)
# — 경로 어디에도 모델명이 없어 아래 prefix 매칭이 실패한다. 그러면 토큰 판정을 못 해
# mtime 폴백으로 내려가므로, EXP 단위로 모델을 명시한다 (EXP07 은 qwen2.5-vl-3b 단독).
# `rebuild_state_diff_metrics.sh` 가 같은 경로에서 EVAL_DS 를 명시 매핑하는 것과 같은 이유다.
_PROBE_FORGET_MODEL = {"AndroidControl_EXP07": "qwen2.5-vl-3b"}


def _model_key_for(path: str) -> str | None:
    """prediction 경로에서 모델 키를 뽑는다 (`outputs/<EXP>/eval/<model>[_변형]/...`).

    `qwen3-vl-8b_ratio37` 처럼 변형 접미사가 붙으므로 prefix 로 맞춘다. 못 찾으면
    None — 호출자가 mtime 폴백으로 넘어간다.
    """
    parts = Path(path).parts
    if "probe_forget" in parts:
        idx = parts.index("probe_forget")
        if idx >= 1:
            return _PROBE_FORGET_MODEL.get(parts[idx - 1])
        return None
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


def _cached_snapshot(model_id: str) -> str | None:
    """HF 캐시에서 tokenizer 파일이 실제로 있는 snapshot 디렉터리 경로. 없으면 None.

    repo id 로 부르는 경로가 실패하는 캐시가 존재하기 때문에 필요하다 (2026-08-11
    실측): `Qwen/Qwen2.5-VL-3B-Instruct` 캐시에는 tokenizer 4종이 다 있는데
    **`config.json` 이 없다.** `AutoTokenizer.from_pretrained(repo_id)` 는 tokenizer
    클래스를 정하려 config 를 먼저 찾고, 없으면 hub 로 나가려다 오프라인에서
    `OSError` 로 죽는다. snapshot 디렉터리를 **직접** 주면 `tokenizer_config.json` 의
    `tokenizer_class` 로 해결돼 정상 로드된다.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    repo_dir = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir()):
        if (snap / "tokenizer.json").exists() or (snap / "tokenizer_config.json").exists():
            return str(snap)
    return None


def _token_lengths(texts: list[str], model_key: str) -> list[int]:
    """`texts` 각각의 토큰 수. transformers 는 **여기서만** 늦게 로드한다.

    `_hungarian_eval._lazy_deps()` 와 같은 이유다 — 모듈 top-level 에서 import 하면
    채점기 전체가 transformers 에 묶인다. 테스트는 이 함수를 갈아끼운다.

    로드 실패는 **조용히 넘어가면 안 되는 실패**다. `_mode_share` 가 None 을 돌려주면
    `truncated_reason` 이 mtime 폴백으로 내려가고, 그러면 절단 판정이 이미 두 번
    반증된 기준으로 되돌아간다 — 판정을 **안 한 것**이 "정상"으로 보고된다
    (2026-08-11 실측: 3B 계열 EXP05·EXP07 22 leaf 전량이 이 상태였다). 그래서
    repo id 가 실패하면 캐시 snapshot 경로로 한 번 더 시도한다.
    """
    from transformers import AutoTokenizer

    if model_key not in _TOKENIZERS:
        model_id = _MODEL_ID[model_key]
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception:
            snap = _cached_snapshot(model_id)
            if snap is None:
                raise
            tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        _TOKENIZERS[model_key] = tok
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
# 항목 스키마: ADDED/MODIFIED/UNCHANGED 는 next state 요소를 가리키므로 `gt_idx`,
# DELETED 는 next state 에 대응물이 없어 current 요소를 가리키므로 `cur_idx` 다.
# 두 축을 한 리스트에 담되 **GT 축을 먼저, DELETED 를 뒤에** 붙인다 — 기존 소비자
# (recall 층 분해)가 순서와 스키마를 그대로 보게 하기 위한 것이다.
def _classify_from_els(
    cur_els: list[dict],
    next_els: list[dict],
    match_mode: str = "index",
    *,
    strict_pos: bool = False,
    empty_next_is_deletion: bool = True,
) -> list[dict]:
    """이미 추출된 element 리스트로 diff 를 분류한다 (`classify_diff` 의 본체).

    호출자가 같은 XML 을 이미 파싱해 뒀을 때 `extract_elements` 를 다시 돌리지 않기
    위한 진입점이다 — `compute_state_diff` 는 pred/gt/current 셋을 이미 들고 있고,
    change 축 때문에 분류를 두 번(→gt, →pred) 하므로 재파싱 비용이 그대로 두 배가 된다.

    `empty_next_is_deletion` (2026-08-04 신설) — `next_els` 가 비었을 때의 해석을
    **호출자별로** 가른다. 전역 동작이 아니라 파라미터인 것이 요점이다:

      - GT 쪽(`cur → gt`)은 `True`. GT 가 실제로 빈 state 라면 그건 관측된 사실이고,
        정말로 전량 삭제다.
      - **예측 쪽(`cur → pred`)은 `False`.** 예측이 비었다는 건 관측이 아니라
        **생성 실패**다. 그걸 "current 를 전부 지웠다는 적극적 주장"으로 읽으면
        아무것도 못 낸 모델이 `pred_deleted ∩ gt_deleted` 를 공짜 hit 으로 받아
        `change_f1_strict` 을 0.24~0.38 씩 가져간다 (2026-08-04 실측). ScratchWorld
        선례가 명시적이다 — *"Outputs that fail schema parsing are scored as
        **incorrect**."* 그래서 주장 자체를 비운다.
    """
    if not next_els:
        if not empty_next_is_deletion:
            # 주장 없음. `n_change_pred == 0` 이 되어 prec/recall/f1 이 전부 0.0 이다
            # (None 이 아니다 — 평균에서 빠지면 실패가 감춰진다).
            return []
        # next state 에 요소가 없다 = current 를 전부 지웠다. (예전에는 빈 리스트를
        # 돌려줬는데, DELETED 축이 생긴 지금은 그게 "변화 없음"과 구분되지 않는다.)
        return [
            {"cur_idx": i, "diff_type": "DELETED", "text_sim": 0.0}
            for i in range(len(cur_els))
        ]
    if not cur_els:
        return [
            {"gt_idx": i, "diff_type": "ADDED", "text_sim": 0.0}
            for i in range(len(next_els))
        ]

    pairs, _ = _he._hungarian_match(cur_els, next_els, match_mode, strict_pos)
    gt_to_cur = {j: i for i, j, _ in pairs}

    out: list[dict] = []
    for j, gt_el in enumerate(next_els):
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
    matched_cur = {i for i, _, _ in pairs}
    out.extend(
        {"cur_idx": i, "diff_type": "DELETED", "text_sim": 0.0}
        for i in range(len(cur_els))
        if i not in matched_cur
    )
    return out


def classify_diff(
    current_str: str,
    gt_str: str,
    match_mode: str = "index",
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
) -> list[dict]:
    """GT(next state)의 각 요소를 current 대비 UNCHANGED/MODIFIED/ADDED 로 분류하고,
    짝을 못 찾은 current 요소를 DELETED 로 덧붙인다.

    `diff_loss/hungarian_diff_v2.classify_diff` 의 포팅이다. 그 모듈을 직접 import
    하지 않는 이유는 셋이다: (a) `hungarian_metric_v2` 를 bare import 해서 sys.path
    해킹이 필요하고, (b) munkres 고정이라 scipy 로 통일한 채점 체제와 어긋나며,
    (c) **pos 모드 전용**이라 EXP01~04/MB 의 index HTML 을 넣으면 bounds 가 전부
    빈 문자열이 되어 위치 cost 0 으로 조용히 퇴화한다. 게다가 `diff_loss/` 는 학습
    데이터 생성 경로라 채점 요구로 건드릴 수 없다.

    DELETED 는 여기가 유일한 산출 경로다 (레퍼런스에는 없다). recall 층 분해는 GT
    요소를 분모로 잡아 이 축을 볼 수 없는데, "사라져야 할 요소를 지웠나"는 복사 편향의
    직접 증거라 `change_f1_strict` 가 그것을 센다.
    """
    _he._lazy_deps()
    return _classify_from_els(
        _he.extract_elements(current_str, match_mode, include_aria),
        _he.extract_elements(gt_str, match_mode, include_aria),
        match_mode,
        strict_pos=strict_pos,
    )


def summarize_diff(diff_result: list[dict]) -> dict[str, int]:
    counts = {"ADDED": 0, "MODIFIED": 0, "UNCHANGED": 0, "DELETED": 0}
    for d in diff_result:
        counts[d["diff_type"]] += 1
    return counts


# ── change 축 (변화 자체를 예측했는가) ───────────────────────────────────
# `change_f1_floor` 는 지표가 아니라 **눈금**이다. 복사기(=current 그대로)는 정의상 0.0
# 이지만 반대쪽 퇴화 — "current 를 하나도 재현하지 않는 예측" — 은 0 이 아니다: 그런
# 예측은 current 전체를 지운 것으로 분류되고, 화면 전환은 실제로 current 의 상당 부분을
# 지우므로 그 교집합이 공짜 hit 이 된다. 2026-08-04 실측(200행): 그 바닥이 EXP01 0.383 ·
# EXP05 0.235 · EXP07v1 0.258 이다. 같은 leaf 의 **학습된** 모델이 EXP07v1 0.114 라
# 바닥에 진다. 그래서 change_f1_strict 은 바닥값을 옆에 두지 않으면 읽을 수 없다
# (0 기준으로 읽으면 "base > trained" 가 결과처럼 보인다 — 그건 결과가 아니라 눈금
# 없이 읽은 것이다).
#
# ⚠️ 2026-08-04 의미 변경 — 이 바닥에 **빈 예측으로는 더 이상 도달할 수 없다.**
# `_classify_from_els(..., empty_next_is_deletion=False)` 가 생성 실패를 "주장 없음"으로
# 읽어 `change_f1_strict = 0.0` 을 주기 때문이다. 지금 `change_f1_floor` 가 재는 것은
# **"비어 있지 않으면서 current 와 아무것도 공유하지 않는 최대삭제 예측"의 점수 상한**
# 이다. 닫힌식과 수치는 그대로이고 라벨만 바뀌었다 — 이전에 발표한 값은 유효하다.
#
# ⚠️ 그리고 이 바닥은 여전히 0 이 아니다. 요소가 단 1개라도 있으면 나머지 current 요소
# 전부가 그대로 DELETED 주장이 되므로, 위 수정은 **"생성 실패"만 0 으로 만들 뿐 DELETED
# 공짜 hit 문제 자체를 없애지 않는다.** "빈 예측을 막았으니 바닥이 0 이 됐다"고 읽지 말 것.
# (2026-08-04 개명: change_f1→change_f1_strict, change_f1_null→change_f1_floor —
#  `_null` 은 이 모듈의 진짜 None 값과 헷갈렸고 `_floor` 가 실제 뜻인 퇴화 바닥이다.
#  change_prec/change_recall 도 loose 축과의 대칭을 위해 `_strict` 를 붙였다.)
_CHANGE_METRIC_KEYS = (
    "change_prec_strict",
    "change_recall_strict",
    "change_f1_strict",
    # loose 축 (2026-08-04 신설) — strict 와 유일한 차이는 ADDED/MODIFIED hit 에서
    # 내용 임계 τ 를 **빼고** 매칭만 본다는 것이다. ScratchWorld 의 `F₁^pres`
    # (presence-only) ↔ `F₁^VA` (value-aware) 대응. 둘을 나란히 두면 "자리를 못 찾은
    # 건가, 자리는 찾고 내용이 틀린 건가"가 갈린다 — strict 하나로는 구분 불가다.
    "change_prec_loose",
    "change_recall_loose",
    "change_f1_loose",
    # 바닥은 **하나뿐이다.** 퇴화 예측의 hit 은 전부 DELETED 에서 오는데 DELETED 판정에는
    # τ 가 걸리지 않으므로 strict/loose 의 바닥이 같은 값이다. 두 개를 만들면 같은 수를
    # 두 이름으로 싣게 된다.
    "change_f1_floor",
)


def compute_change_items(
    diff_gt: list[dict],
    diff_pred: list[dict],
    pred_els: list[dict],
    gt_els: list[dict],
    pairs_pg: list[tuple],
    n_cur: int,
) -> dict:
    """current 대비 바뀐 항목을 pred/gt 양쪽에서 뽑아 맞춰 본다.

    recall 계열이 못 보는 것을 본다. `addmod_recall` 의 분모는 GT 요소라 **없어져야 할
    요소**를 세지 못하고, hit 판정이 Hungarian 매칭뿐이라 자리만 맞고 내용이 틀린
    예측도 맞힌 것으로 센다. 여기서는 변화 자체를 항목으로 만들어 집합 비교한다.

        C_gt   = classify(current → gt)   의 ADDED ∪ MODIFIED ∪ DELETED
        C_pred = classify(current → pred) 의 ADDED ∪ MODIFIED ∪ DELETED

    두 집합을 **같은 절차**로 뽑는 것이 핵심이다 (대칭). hit 은 ADDED/MODIFIED 면
    정본 pred↔gt 매칭에서 짝이 맞고 그 짝의 `text_sim` 이 `CHANGE_TEXT_SIM_TAU`
    이상일 때, DELETED 면 **같은 current 요소를 양쪽 다 지웠을 때**다.

    정의 경계
      - 양쪽 다 비면(변화 없는 행) 세 지표 모두 None — 잴 것이 없다.
      - 한쪽만 비면 0.0 이다. **None 으로 빼면 안 된다**: C_pred 가 비는 행이 곧
        복사기이고, C_gt 가 비는데 C_pred 가 차 있으면 환각이다. 둘 다 이 지표가
        잡으라고 만든 것인데 정의불능으로 빼면 평균에서 사라진다.
      - **예측이 파싱 불능이면(`pred_els == []`) C_pred 가 빈다** → prec/rec/f1 = 0.0.
        2026-08-04 이전에는 이 경우 current 전체를 DELETED 로 주장한 것으로 읽어
        `change_f1` 이 바닥값(0.24~0.38)을 받았다. `_classify_from_els` 참고.

    strict / loose
      두 축은 **ADDED/MODIFIED hit 판정에서만** 갈린다. strict 는 매칭된 짝의
      `text_sim ≥ CHANGE_TEXT_SIM_TAU`(0.9)까지 요구하고, loose 는 매칭만 본다.
      DELETED hit(같은 current 요소를 양쪽 다 지웠나)은 τ 와 무관해 양쪽 공통이다.
      따라서 항상 `change_f1_loose ≥ change_f1_strict` 이고, 그 **갭이 "자리는 찾았는데
      내용이 틀린" 양**이다. 정의 구간(None 여부)은 두 축이 완전히 같아야 한다 —
      어긋나면 두 평균의 분모가 갈려 나란히 못 읽는다.

    `change_f1_floor` 는 같은 행에서 **current 를 하나도 재현하지 않는 예측**이 받는
    점수의 상한이다 (그런 예측은 current 전체가 DELETED 로 분류되므로 hits =
    |gt_deleted|, n_pred = n_cur — 매칭이 필요 없어 Hungarian 을 한 번 더 돌리지 않는다).
    빈 문자열은 위 규칙 변경으로 더 이상 이 바닥에 도달하지 않는다. **change_f1_strict 와
    정확히 같은 행에서만 정의한다** — 정의 구간이 어긋나면 두 평균의 분모가 달라져
    나란히 못 읽는다.
    """
    gt_changed = {
        d["gt_idx"] for d in diff_gt if d["diff_type"] in ("ADDED", "MODIFIED")
    }
    gt_deleted = {d["cur_idx"] for d in diff_gt if d["diff_type"] == "DELETED"}
    pred_changed = {
        d["gt_idx"] for d in diff_pred if d["diff_type"] in ("ADDED", "MODIFIED")
    }
    pred_deleted = {d["cur_idx"] for d in diff_pred if d["diff_type"] == "DELETED"}

    n_gt = len(gt_changed) + len(gt_deleted)
    n_pred = len(pred_changed) + len(pred_deleted)
    counts = {"n_change_gt": n_gt, "n_change_pred": n_pred}
    if not n_gt and not n_pred:
        return {**{k: None for k in _CHANGE_METRIC_KEYS}, **counts}

    pred_to_gt = {i: j for i, j, _ in pairs_pg}
    # 매칭된 ADDED/MODIFIED 짝을 한 번만 돌면서 두 축을 동시에 센다 — 같은 루프를
    # 두 번 도는 것보다 싸고, 무엇보다 **두 축이 같은 짝 집합을 보게** 강제된다.
    hits_loose = 0
    hits_strict = 0
    for i in pred_changed:
        j = pred_to_gt.get(i)
        if j not in gt_changed:
            continue
        hits_loose += 1
        if _he._text_sim(pred_els[i]["text"], gt_els[j]["text"]) >= CHANGE_TEXT_SIM_TAU:
            hits_strict += 1
    # DELETED hit 은 τ 와 무관하다 (지워진 요소엔 비교할 내용이 없다) — 양축 공통.
    n_del_hit = len(pred_deleted & gt_deleted)
    hits_loose += n_del_hit
    hits_strict += n_del_hit

    def _prf(hits: int) -> tuple[float, float, float]:
        prec = hits / n_pred if n_pred else 0.0
        rec = hits / n_gt if n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    prec_s, rec_s, f1_s = _prf(hits_strict)
    prec_l, rec_l, f1_l = _prf(hits_loose)

    # 퇴화 눈금 — current 를 하나도 재현하지 않는 예측(= current 전체가 DELETED 로
    # 분류되는 예측)의 같은 행 점수 상한. 빈 예측은 이제 여기 도달하지 않는다.
    prec0 = len(gt_deleted) / n_cur if n_cur else 0.0
    rec0 = len(gt_deleted) / n_gt if n_gt else 0.0
    f1_floor = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
    return {
        "change_prec_strict": round(prec_s, 4),
        "change_recall_strict": round(rec_s, 4),
        "change_f1_strict": round(f1_s, 4),
        "change_prec_loose": round(prec_l, 4),
        "change_recall_loose": round(rec_l, 4),
        "change_f1_loose": round(f1_l, 4),
        "change_f1_floor": round(f1_floor, 4),
        **counts,
    }


# ── 행 단위 채점 ─────────────────────────────────────────────────────────
def _unclosed_root(pred_str: str) -> float:
    """예측이 root tag 를 닫지 않고 끝났으면 1.0. 절단 leaf 의 사후 식별용 sanity."""
    m = _ROOT_TAG_RE.match(pred_str)
    if not m:
        return 1.0
    tail = pred_str.rstrip()
    return 0.0 if tail.endswith(f"</{m.group(1)}>") or tail.endswith("/>") else 1.0


def compute_state_diff(
    pred_str: str,
    gt_str: str,
    current_str: str,
    match_mode: str = "index",
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
) -> dict:
    """한 행의 state-diff 진단값. 정의되지 않는 지표는 None (평균에서 제외된다).

    반환 키
      addmod_recall / added_recall / modified_recall / unchanged_recall
          GT 를 diff 유형으로 층화한 recall. `addmod` = MODIFIED + ADDED.
          UNCHANGED 는 높고 addmod 는 낮으면 = 복사만 잘한다.
      addmod_prec / addmod_f1
          precision 분모를 **pred-side diff**(current 와 매칭되지 않은 예측 요소)로
          잡아 대칭화한 값. recall 과 다른 정의이므로 키를 분리해 둔다.
      change_{prec,recall,f1}_strict / change_{prec,recall,f1}_loose / change_f1_floor
          "current 대비 무엇이 바뀌었나"를 pred/gt 양쪽에서 같은 절차로 뽑아 맞춘 값.
          DELETED 축을 포함한다. strict 는 내용 임계(τ)까지, loose 는 자리(매칭)만.
          `_floor` 는 같은 행에서 **current 를 하나도 재현하지 않는 예측**이 받는 점수
          = 이 축의 퇴화 바닥. strict/loose 공통이다 (DELETED 엔 τ 가 없다).
          change_f1_* 는 이 값과 나란히 읽어야 한다 (0 이 바닥이 아니다).
      no_change_acc
          **GT 가 current 와 같은 행에서만** 정의 (`n_change_gt == 0`). 그 행에서는
          복사가 정답이므로 "예측도 아무 변화를 주장하지 않았나"를 1/0 으로 센다.
          변화가 있는 행에서는 None — 다른 지표와 분모가 다르다.
      copy_rate_pred / copy_rate_gt / copy_excess
          예측·GT 가 각각 current 와 겹치는 비율과 그 차이. `copy_excess` 가 판별량.
          ⚠️ **`copy_excess` 를 `parse_fail_rate` 없이 두 모델 사이에서 비교하지 마라.**
          파싱 불능 행은 이 지표가 None 이라 평균에서 빠지는데, 그 비율이 모델마다
          다르면 **서로 다른 population 위의 평균**을 나란히 놓게 된다 (실측: EXP07
          probe_forget 에서 onlyS2 계열 11~15% 제외 vs mergeO 계열 1.6~2.6% 제외).
      copy_exact / copy_near
          예측이 current 와 문자열 완전일치 / hungarian_f1 >= COPY_NEAR_F1 인가.
      parse_fail / parse_fail_long
          예측에서 element 를 하나도 못 뽑은 행 / 그중 예측 문자열이 100자를 넘는 행.
          둘을 나눠야 "아무것도 안 냈다"와 "**태그가 하나도 없는 장문 쓰레기를 냈다**"가
          구분된다 (실측 사례: `predict` 58,303자인데 추출 요소 0개).
      n_* : 해석용 원자료 개수. unclosed_root : 절단 sanity.

      2026-08-04 개명: diff_recall/diff_prec/diff_f1 → addmod_recall/addmod_prec/
      addmod_f1, change_f1/change_f1_null → change_f1_strict/change_f1_floor
      (모듈 docstring "2026-08-04 개명" 참고). `aggregate()` 가 옛 키를 하위호환
      alias 로 함께 낸다 — 이 함수(row-dict)에는 옛 키가 없다.
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
        "n_change_gt": 0,
        "n_change_pred": 0,
    }
    undefined = {
        k: None
        for k in (
            "addmod_recall",
            "added_recall",
            "modified_recall",
            "unchanged_recall",
            "addmod_prec",
            "addmod_f1",
            *_CHANGE_METRIC_KEYS,
            "no_change_acc",
            "copy_rate_pred",
            "copy_rate_gt",
            "copy_excess",
        )
    }

    try:
        pred_els = _he.extract_elements(pred_str, match_mode, include_aria)
        gt_els = _he.extract_elements(gt_str, match_mode, include_aria)
        cur_els = _he.extract_elements(current_str, match_mode, include_aria)
    except Exception:
        # 추출 자체가 터진 행도 **파싱 실패로 세어야 한다.** 여기서 안 채우면
        # `aggregate` 의 `r.get(k, 0.0)` 이 조용히 0 으로 세어 실패율이 과소보고된다.
        return {
            **undefined,
            **zero_counts,
            "copy_exact": 0.0,
            "copy_near": 0.0,
            "parse_fail": 1.0,
            "parse_fail_long": 1.0 if len(pred_str.strip()) > PARSE_FAIL_LONG_CHARS else 0.0,
            "unclosed_root": _unclosed_root(pred_str),
        }

    counts = {
        "n_pred": len(pred_els),
        "n_gt": len(gt_els),
        "n_cur": len(cur_els),
    }
    parse_fail = 1.0 if not pred_els else 0.0
    row = {
        **undefined,
        **zero_counts,
        **counts,
        "copy_exact": 1.0 if pred_str.strip() == current_str.strip() else 0.0,
        "copy_near": 0.0,
        "parse_fail": parse_fail,
        "parse_fail_long": (
            1.0
            if parse_fail and len(pred_str.strip()) > PARSE_FAIL_LONG_CHARS
            else 0.0
        ),
        "unclosed_root": _unclosed_root(pred_str),
    }
    if not gt_els:
        return row

    diff_gt = _classify_from_els(cur_els, gt_els, match_mode, strict_pos=strict_pos)
    by_type: dict[str, set[int]] = {
        "ADDED": set(),
        "MODIFIED": set(),
        "UNCHANGED": set(),
    }
    for d in diff_gt:
        # DELETED 는 GT 요소가 아니라 current 요소를 가리킨다 — GT 를 분모로 잡는
        # recall 층 분해에는 들어갈 자리가 없다. change 축에서만 쓴다.
        if d["diff_type"] != "DELETED":
            by_type[d["diff_type"]].add(d["gt_idx"])
    diff_idx = by_type["ADDED"] | by_type["MODIFIED"]
    row["n_gt_added"] = len(by_type["ADDED"])
    row["n_gt_modified"] = len(by_type["MODIFIED"])
    row["n_gt_unchanged"] = len(by_type["UNCHANGED"])

    # GT 기준선 — GT 자체가 current 와 겹치는 비율. copy_rate_pred 의 해석 기준.
    row["copy_rate_gt"] = round((len(gt_els) - len(by_type["ADDED"])) / len(gt_els), 4)

    # 정본과 같은 pred↔gt 매칭 한 번. 층 분해의 근거라 부분집합 재매칭을 쓰지 않는다.
    pairs_pg, _ = _he._hungarian_match(pred_els, gt_els, match_mode, strict_pos)
    hit_gt = {j for _, j, _ in pairs_pg}
    pred_to_gt = {i: j for i, j, _ in pairs_pg}

    def _recall(subset: set[int]) -> float | None:
        return round(len(hit_gt & subset) / len(subset), 4) if subset else None

    row["addmod_recall"] = _recall(diff_idx)
    row["added_recall"] = _recall(by_type["ADDED"])
    row["modified_recall"] = _recall(by_type["MODIFIED"])
    row["unchanged_recall"] = _recall(by_type["UNCHANGED"])

    # change 축 — GT 쪽과 **인자 순서까지 같은** 분류를 pred 쪽에도 돌려 대칭을 지킨다.
    # (`_hungarian_match(cur, X)` 로 통일. 아래 pairs_pc 는 인자 순서가 반대라 재사용
    #  하면 동점 배정이 갈릴 수 있고, 그러면 두 집합이 같은 절차의 산물이 아니게 된다.)
    # `empty_next_is_deletion=False` — 예측이 비면 "최대 삭제 주장"이 아니라
    # "주장 없음"이다 (2026-08-04, `_classify_from_els` 참고). GT 쪽 호출은 기본값
    # True 를 그대로 쓴다: GT 의 빈 state 는 생성 실패가 아니라 관측된 사실이다.
    diff_pred = _classify_from_els(
        cur_els,
        pred_els,
        match_mode,
        strict_pos=strict_pos,
        empty_next_is_deletion=False,
    )
    row.update(
        compute_change_items(
            diff_gt, diff_pred, pred_els, gt_els, pairs_pg, len(cur_els)
        )
    )
    # 변화 없는 행("화면이 안 바뀌는 step")에서는 복사가 정답이다 — 다른 지표는 이
    # 행에서 전부 None 이라 이 축이 없으면 그 구간의 성능을 아무도 안 잰다.
    #
    # ⚠️ `pred_els` 를 조건에 **반드시** 넣어야 한다. 빈 예측은 위 규칙 변경으로
    # `n_change_pred == 0` 이므로, 이걸 빼면 **아무것도 못 낸 모델이 1.0 을 받는다**
    # (실측: EXP01 base 는 파싱 실패율 93.9% 인데 no_change_acc 가 204/204 = 1.0 이
    #  나왔다). 이 축이 묻는 것은 "변화를 지어내지 않았나"가 아니라 **"화면을 그대로
    # 재현했나"** 다 — 생성 실패는 재현이 아니라 오답이다 (copy_excess 를 0 으로 채우지
    # 않기로 한 것과 같은 이유: 실패가 미덕으로 집계되면 안 된다).
    if row["n_change_gt"] == 0:
        row["no_change_acc"] = (
            1.0 if (pred_els and row["n_change_pred"] == 0) else 0.0
        )

    # pred ↔ current: 복사량 + pred-side diff 산출
    #
    # ⚠️ `pred_els` 가 비는 행은 `copy_rate_pred`/`copy_excess` 가 None 으로 남아
    # 평균에서 빠진다. **이걸 0 으로 채우자는 제안은 2026-08-04 에 기각했다.**
    # 빈 예측의 `copy_rate_pred` 는 0 이므로 `copy_excess = 0 − copy_rate_gt ≈ −0.77`
    # 이 되는데, 그러면 **아무것도 못 낸 행이 "복사를 안 했다"는 미덕으로 집계된다** —
    # 이 지표가 잡으라고 만든 것의 정반대다. 실측(EXP07 probe_forget onlyS2-ep1,
    # ID n=2032): 채우면 avg_copy_excess 가 +0.2392 → +0.0850 으로 **떨어져** 퇴화
    # 모델이 되레 좋아 보인다.
    # 남는 진짜 문제는 값이 아니라 **분모**다 — 그건 `n_copy_excess` 와
    # `parse_fail_rate` 를 함께 실어서 드러낸다 (compute_state_diff docstring 참고).
    if pred_els and cur_els:
        pairs_pc, _ = _he._hungarian_match(pred_els, cur_els, match_mode, strict_pos)
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

    # `n_pred_diff` 는 이름을 유지한다 — addmod_prec 의 분모이지만 `_COUNT_KEYS`
    # 소속이라 이번 개명 범위 밖이다 (이름이 어긋나지만 의도적 유지).
    row["n_pred_diff"] = len(pred_diff_idx)
    if pred_diff_idx and diff_idx:
        hit = sum(1 for i in pred_diff_idx if pred_to_gt.get(i) in diff_idx)
        prec = hit / len(pred_diff_idx)
        rec = row["addmod_recall"] or 0.0
        row["addmod_prec"] = round(prec, 4)
        row["addmod_f1"] = round(
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0, 4
        )
    return row


# ── 배선 self-test ───────────────────────────────────────────────────────
# `_he` 의 bs4/솔버는 지연 로드다. 초기화 없이 부르면 `extract_elements` 가 예외를 내고
# `compute_hungarian_acc` 의 except 가 그걸 삼켜 **전 행 0점**을 조용히 돌려준다
# (2026-08-01 실측: 표본 f1 0.0 vs aggregate 0.71). 행 단위로는 구분할 수 없으므로
# 데이터와 무관한 고정 XML 로 배선만 검사한다.
# cur 에는 **gt 에서 사라지는 요소**가 반드시 하나 있어야 한다 (`img`/`gone`).
# 없으면 `gt_deleted` 가 공집합이 되고, 그러면 change 축의 지배항인 DELETED 가
# self-test 를 한 번도 통과하지 않는다 — 빈 예측 퇴화를 probe 가 구조적으로 못 본다.
# 태그를 `img` 로 둔 것은 의도다: 같은 `p` 로 두면 gt 의 "brand new" 와 짝이 맞아
# ADDED 가 사라지고(copy_excess 0), 위치를 멀리 밀어도 tag 가 같으면 cost 상한이
# W_TEXT+W_POS=1.9 라 임계 1.7 을 못 넘긴다. tag 불일치는 W_TAG=3.0 이라 확실하다.
# `maxdel` = "요소가 딱 하나뿐이고 current 를 하나도 재현하지 않는" 예측 —
# **비어 있지 않은 예측 중 가장 퇴화한 것**이다. 태그는 `select` 를 쓴다: cur/gt 의
# 어떤 태그(`button`/`p`/`img`)와도 달라 tag cost W_TAG=3.0 이 임계(1.5/1.7)를 확실히
# 넘고(위치·텍스트를 아무리 밀어도 **같은 태그면** cost 상한이 임계 아래라 붙는다),
# 동시에 `INTERACTIVE_TAGS` 소속이라 텍스트 없이도 추출된다 — 화이트리스트 밖 태그
# (`zzz` 등)를 쓰면 element 가 0 개로 추출돼 빈 예측과 구분이 안 된다.
# 이 probe 가 **바닥이 여전히 0 이 아님**을 배선 단계에서 증명한다: 빈 예측을 0 으로
# 만들어도 요소가 1개만 있으면 나머지 current 전부가 그대로 DELETED 주장이 된다.
_PROBE = {
    "index": {
        "cur": '<node index="0"><button index="1" aria-label="OK"/>'
        '<p index="2">hello</p><img index="4">gone</img></node>',
        "gt": '<node index="0"><button index="1" aria-label="OK"/>'
        '<p index="2">world</p><p index="3">brand new</p></node>',
        "maxdel": '<node index="900"><select index="901"/></node>',
    },
    "pos": {
        "cur": '<node bounds="[0,0][10,10]" point="[5,5]">'
        '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
        '<p bounds="[6,6][9,9]" point="[7,7]">hello</p>'
        '<img bounds="[6,30][9,34]" point="[7,32]">gone</img></node>',
        "gt": '<node bounds="[0,0][10,10]" point="[5,5]">'
        '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
        '<p bounds="[6,6][9,9]" point="[7,7]">world</p>'
        '<p bounds="[6,20][9,24]" point="[7,22]">brand new</p></node>',
        "maxdel": '<node bounds="[900,900][910,910]" point="[905,905]">'
        '<select bounds="[920,920][930,930]" point="[925,925]"/></node>',
    },
}


def assert_scorer_wired(
    match_mode: str, *, strict_pos: bool = False, include_aria: bool = False
) -> None:
    """표본 채점 전에 채점기가 실제로 동작하는지 한 번 확인한다."""
    _he._lazy_deps()
    probe = _PROBE[match_mode]
    els = _he.extract_elements(probe["cur"], match_mode, include_aria)
    if not els:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            "probe XML 에서 element 0개 — bs4/scipy 의존성을 확인하세요."
        )
    opts = {"strict_pos": strict_pos, "include_aria": include_aria}
    # current 를 그대로 예측 = 복사. diff 를 하나도 못 맞히고 copy_excess 가 양수여야 한다.
    copied = compute_state_diff(
        probe["cur"], probe["gt"], probe["cur"], match_mode, **opts
    )
    perfect = compute_state_diff(
        probe["gt"], probe["gt"], probe["cur"], match_mode, **opts
    )
    if not (copied["copy_rate_pred"] == 1.0 and copied["copy_excess"] > 0):
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"복사 probe 가 copy_rate={copied['copy_rate_pred']} "
            f"copy_excess={copied['copy_excess']} — 1.0 / 양수 여야 합니다."
        )
    if perfect["addmod_recall"] != 1.0:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"정답 probe 의 addmod_recall={perfect['addmod_recall']} — 1.0 이어야 합니다."
        )
    # change 축은 배선이 끊겨도 **그럴듯한 0** 을 낸다 (변화를 하나도 못 잡았을 때와
    # 계산이 안 돌았을 때의 값이 같다). 두 끝을 다 찍어야 구분된다.
    if copied["change_f1_strict"] != 0.0:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"복사 probe 의 change_f1_strict={copied['change_f1_strict']} — 0.0 이어야 합니다."
        )
    if perfect["change_f1_strict"] != 1.0:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"정답 probe 의 change_f1_strict={perfect['change_f1_strict']} — 1.0 이어야 합니다."
        )
    # 규칙의 양 끝을 둘 다 박는다 (2026-08-04). 예전에는 **빈 예측** 하나가 두 역할을
    # 겸했는데 — 퇴화 눈금이 0 이 아님을 보이는 것과 DELETED 축이 살아 있음을 보이는 것 —
    # 빈 예측이 이제 0 을 받도록 규칙이 바뀌어 그 겸직이 불가능하다. 역할을 둘로 쪼갠다.
    #
    # (1) 생성 실패는 0 이어야 한다. `empty_next_is_deletion=False` 가 pred 쪽에
    #     실제로 걸렸는지 확인한다 — 이게 안 걸리면 아무것도 못 낸 모델이 바닥값을
    #     공짜로 가져간다.
    empty = compute_state_diff("", probe["gt"], probe["cur"], match_mode, **opts)
    if empty["change_f1_strict"] != 0.0:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"빈 예측 probe 의 change_f1_strict={empty['change_f1_strict']} — "
            "0.0 이어야 합니다 (빈 예측을 '최대 삭제 주장'으로 읽고 있습니다)."
        )
    if empty["parse_fail"] != 1.0:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"빈 예측 probe 의 parse_fail={empty['parse_fail']} — 1.0 이어야 합니다."
        )
    # (2) DELETED 축은 살아 있어야 한다. current 와 아무것도 공유하지 않는 **비어 있지
    #     않은** 예측이 그 역할을 이어받는다. 0 이 나오면 probe 에 DELETED 가 없다는
    #     뜻이고(=`_PROBE` 훼손), 그러면 change_f1_strict 을 0 기준으로 읽게 된다.
    #     `change_f1_floor` 는 이 전략의 **상한**이다 — 합성 요소 1개가 ADDED 로
    #     `n_change_pred` 에 더해져 precision 분모가 커지므로 실제 값은 그보다 조금 낮다.
    maxdel = compute_state_diff(
        probe["maxdel"], probe["gt"], probe["cur"], match_mode, **opts
    )
    floor = maxdel["change_f1_floor"]
    if not 0.0 < maxdel["change_f1_strict"] <= floor:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"최대삭제 probe 의 change_f1_strict={maxdel['change_f1_strict']} 이 "
            f"(0, change_f1_floor={floor}] 밖입니다 — DELETED 축이 죽었거나 "
            "바닥 닫힌식이 어긋났습니다."
        )
    if maxdel["change_f1_strict"] < 0.5 * floor:
        raise StateDiffError(
            f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
            f"최대삭제 probe 의 change_f1_strict={maxdel['change_f1_strict']} 이 "
            f"바닥({floor})의 절반에도 못 미칩니다 — 합성 요소가 current 와 매칭돼 "
            "DELETED 주장이 줄었을 수 있습니다 (`_PROBE['maxdel']` 확인)."
        )
    # loose 는 strict 의 상계다. 두 축이 같은 짝 집합을 보는지 확인한다 — 뒤집히면
    # τ 게이트가 엉뚱한 쪽에 걸린 것이다.
    for name, r in (("복사", copied), ("정답", perfect), ("최대삭제", maxdel)):
        if r["change_f1_loose"] < r["change_f1_strict"]:
            raise StateDiffError(
                f"state-diff 채점기 배선 실패 (match_mode={match_mode}): "
                f"{name} probe 의 change_f1_loose={r['change_f1_loose']} < "
                f"change_f1_strict={r['change_f1_strict']} — loose 는 strict 의 "
                "상계여야 합니다 (τ 게이트가 반대로 걸렸습니다)."
            )


# ── 집계 ─────────────────────────────────────────────────────────────────
# None 은 "그 행에서 정의되지 않음"이다 (예: GT 에 ADDED 요소가 없는 행의 added_recall).
# 0 으로 세면 평균이 아래로 끌려가 실제보다 나쁘게 보인다. 평균의 분모가 되는 행 수를
# 함께 기록해 몇 행 위에서 잰 값인지 드러낸다.
_MEAN_KEYS = [
    "addmod_recall",
    "added_recall",
    "modified_recall",
    "unchanged_recall",
    "addmod_prec",
    "addmod_f1",
    "copy_rate_pred",
    "copy_rate_gt",
    "copy_excess",
    *_CHANGE_METRIC_KEYS,
    # 변화 없는 행에서만 정의된다 — `n_no_change_acc` 가 곧 "그런 step 이 몇 행인가"
    # (= 데이터셋 sparsity 통계) 라서 평균과 분모를 함께 봐야 한다.
    "no_change_acc",
]
_RATE_KEYS = [
    "copy_exact",
    "copy_near",
    "unclosed_root",
    # `parse_fail_rate` / `parse_fail_long_rate` 로 나간다 (`f"{k}_rate"`).
    # copy_excess 의 분모 손실을 드러내는 짝이므로 항상 함께 읽는다.
    "parse_fail",
    "parse_fail_long",
]
_COUNT_KEYS = [
    "n_pred",
    "n_gt",
    "n_cur",
    "n_gt_added",
    "n_gt_modified",
    "n_gt_unchanged",
    "n_pred_diff",
    "n_change_gt",
    "n_change_pred",
]


# 산출물 스키마 버전. 개명 alias 때문에 **옛 키 이름이 그대로 남아 있는데
# `change_f1` 의 정의는 바뀌었다**(빈 예측 규칙, `_classify_from_els` 참고). 파일만
# 보고 옛/새를 구분할 수단이 없으면 정의가 다른 두 수를 같은 키로 나란히 놓게 된다.
METRICS_SCHEMA = "2026-08-04"

# 하위호환 alias — 2026-08-04 개명(모듈 docstring 참고) 전 산출물은 옛 키만 갖고 있다.
# `aggregate()` 가 새 키와 나란히 옛 키도 내어 소비자가 한 번에 갈아타지 않아도 되게 한다.
#
# ⚠️ **`change_f1` alias 는 "이름만 다른 같은 수"가 아니다.** 같은 실행 안에서는
# `avg_change_f1 == avg_change_f1_strict` 이지만, 2026-08-04 이전에 채점된 파일의
# `avg_change_f1` 은 **빈 예측이 바닥값을 받던 옛 정의**의 값이다. 두 파일을 나란히
# 놓기 전에 `metrics_schema` 를 먼저 볼 것.
_LEGACY_KEY_ALIAS = {
    "addmod_recall": "diff_recall",
    "addmod_prec": "diff_prec",
    "addmod_f1": "diff_f1",
    "change_f1_floor": "change_f1_null",
    "change_f1_strict": "change_f1",
    "change_prec_strict": "change_prec",
    "change_recall_strict": "change_recall",
}


def stamp_schema(metrics: dict) -> dict:
    """산출물 최상위에 스키마 버전을 박는다. **파일로 쓰는 모든 경로**가 불러야 한다
    (`_state_diff_eval._cmd_score` · `_hungarian_eval._write_state_diff` 둘 다)."""
    metrics["metrics_schema"] = METRICS_SCHEMA
    return metrics


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
    # 옛 이름 alias — avg_/n_ 두 접미사 모두 채운다. n_ 은 조건부 분모라 해석에
    # 필수인데(예: n_diff_prec), avg_ 만 alias 하면 이 값이 옛 이름으로는 사라진다.
    for new, old in _LEGACY_KEY_ALIAS.items():
        if f"avg_{new}" in out:
            out[f"avg_{old}"] = out[f"avg_{new}"]
        if f"n_{new}" in out:
            out[f"n_{old}"] = out[f"n_{new}"]
    return out


def evaluate_pairs(
    gt_entries,
    pred_entries,
    match_mode: str = "index",
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
) -> dict:
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
    opts = {"strict_pos": strict_pos, "include_aria": include_aria}
    assert_scorer_wired(match_mode, **opts)
    rows = []
    failures = 0
    for gt_entry, pred_entry in zip(gt_entries, pred_entries, strict=False):
        gt_text = gt_entry["messages"][-1]["value"]
        pred_text = pred_entry.get("predict", pred_entry.get("output", ""))
        current = parse_prompt(pred_entry.get("prompt", "")).get("current_state", "")
        if not current:
            failures += 1
            continue
        rows.append(compute_state_diff(pred_text, gt_text, current, match_mode, **opts))
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
    # 이 함수는 방금 aggregate() 가 만든 m 을 받으므로 새 키가 항상 있다 — 옛 이름
    # alias 는 디스크의 34개 leaf(재빌드 안 함)를 읽는 소비자를 위한 것이라 여기선
    # 새 이름을 쓴다.
    print(
        f"[state-diff:{label}] total={m['total']}  "
        f"addmod_rec={m['avg_addmod_recall']:.4f}  "
        f"added_rec={m['avg_added_recall']:.4f}  "
        f"unch_rec={m['avg_unchanged_recall']:.4f}  "
        f"copy={m['avg_copy_rate_pred']:.4f}(gt {m['avg_copy_rate_gt']:.4f})  "
        f"excess={m['avg_copy_excess']:+.4f}  "
        f"change_f1={m['avg_change_f1_strict']:.4f}"
        f"/{m['avg_change_f1_loose']:.4f}(floor {m['avg_change_f1_floor']:.4f})  "
        f"pfail={m['parse_fail_rate']:.4f}"
    )


def _cmd_score(args) -> int:
    split = bool(args.test_id or args.pred_id or args.test_ood or args.pred_ood)
    mm = args.match_mode
    # 매칭 기준 스위치는 여기 한 진입점에서만 읽어 아래로 넘긴다 (전역 금지 —
    # `_hungarian_eval._cmd_score` 와 같은 규칙이다).
    opts = {"strict_pos": args.strict_pos_match, "include_aria": args.include_aria}
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
        metrics = build_metrics(gt_id, pr_id, gt_ood, pr_ood, mm, **opts)
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
        metrics = evaluate_pairs(gts, preds, mm, **opts)
        _print_row("all", metrics)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(stamp_schema(metrics), f, ensure_ascii=False, indent=2)
    print(f"[state-diff] saved: {out}")
    return 0


def build_metrics(
    gt_id,
    pr_id,
    gt_ood,
    pr_ood,
    match_mode: str,
    *,
    strict_pos: bool = False,
    include_aria: bool = False,
) -> dict:
    """ID/OOD 3-섹션 산출. `hungarian_metrics.json` 과 **동일한 섹션 구조**여야 한다 —
    `eval_viewer.load_metrics` 의 section 조회는 부재를 silent skip 하므로, 구조가
    어긋나면 표에 빈칸이 뜰 뿐 아무도 오류를 못 본다."""
    opts = {"strict_pos": strict_pos, "include_aria": include_aria}
    return {
        "overall": evaluate_pairs(gt_id + gt_ood, pr_id + pr_ood, match_mode, **opts),
        "in_domain": evaluate_pairs(gt_id, pr_id, match_mode, **opts),
        "out_of_domain": evaluate_pairs(gt_ood, pr_ood, match_mode, **opts),
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
    s.add_argument(
        "--strict-pos-match",
        action="store_true",
        dest="strict_pos_match",
        help="pos 모드 매칭 임계를 1.7 → 1.5 로 조인다. **기본은 꺼짐** — 정본 채점"
        "(_hungarian_eval)과 **반드시 같은 값**이어야 층 분해 항등식이 유지된다.",
    )
    s.add_argument(
        "--include-aria",
        action="store_true",
        dest="include_aria",
        help="pos 모드에서 aria-label 만 가진 요소도 채점 대상에 넣는다. **기본은 꺼짐** — "
        "정본 채점과 **반드시 같은 값**이어야 한다 (element 집합 자체가 달라진다).",
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
