#!/usr/bin/env bash
# state-diff (copy-bias) 메트릭 백필 — GPU 를 쓰지 않는 순수 재채점 배치.
#
# 무엇을 채우나
# -------------
# `_hungarian_eval.py score` 는 2026-08-01 부터 정본 채점과 **같은 실행에서**
# state_diff_metrics.json 을 함께 낸다. 그 이전에 채점된 leaf 에는 그 파일이 없다.
# 이 스크립트가 저장된 generated_predictions_*.jsonl 로 그 구멍만 메운다
# (추론 재실행 없음, 결과는 인라인 산출과 동치).
#
# 왜 rebuild_eval_metrics.sh 에 못 얹나
# ------------------------------------
# 그쪽은 `[ -f "$leaf/$metrics" ] && continue` 로 hungarian_metrics.json 존재를 보고
# 건너뛴다. 백필 대상은 정확히 "이미 hungarian 채점이 끝난" leaf 라, 그 스크립트에
# 얹으면 전부 스킵된다. 조건이 반대라 별도 진입점이 맞다.
#
# 왜 절단 leaf 를 제외하나 — 이 스크립트의 핵심 판단
# --------------------------------------------------
# 2026-07-28 23:38 UTC 이전 예측은 vllm 기본값 1024 토큰에서 하드 컷됐다
# (`--max_new_tokens` 미주입 버그, 커밋 6a4b59e). 잘린 예측은 element 수가 줄어
# **copy_rate_pred 를 과소평가**한다 — 하필 이 지표가 재려는 방향("얼마나 베꼈나")
# 으로 편향되므로, 절단 leaf 의 copy_excess 는 실제보다 낮게, 즉 실제보다 좋게
# 보인다. hungarian 계열처럼 "무효"인 정도가 아니라 **한쪽으로 틀린다.**
# 그래서 경계 이전 leaf 는 산출하지 않는다. 표에 컬럼이 비는 게 정직하다.
#
# **진짜 가드는 채점기 안에 있다** (`_state_diff_eval.truncated_reason`, `score` 진입부).
# 여기 필터는 dry-run 목록을 정직하게 보여주기 위한 선별일 뿐이다 — 백필 스크립트에만
# 두면 `rebuild_woa_metrics.sh → _hungarian_eval score` 경로가 그대로 통과해 편향된
# 산출물이 woa sibling 에 생긴다. 경계 상수도 `_state_diff_eval` 한 곳이 정본이다.
#
# usage: rebuild_state_diff_metrics.sh [-j N] [-n] [-f] [--include-truncated] [FILTER]
#   -j N                동시 실행 수 (기본 4). 채점기는 사실상 단일 스레드다.
#   -n                  dry-run — 대상만 출력.
#   -f                  이미 state_diff_metrics.json 이 있어도 다시 산출.
#   --include-truncated 절단 경계 이전 leaf 도 포함 (위 이유로 기본은 제외).
#   FILTER              leaf 경로에 이 문자열이 든 것만 대상 (예: EXP05, epoch-3).
set -euo pipefail

JOBS=4
DRY=0
FORCE=0
INCLUDE_TRUNCATED=0
FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    -n) DRY=1; shift ;;
    -f) FORCE=1; shift ;;
    --include-truncated) INCLUDE_TRUNCATED=1; shift ;;
    -h|--help) sed -n '1,32p' "$0"; exit 0 ;;
    *)  FILTER="$1"; shift ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

exec 9>"${TMPDIR:-/tmp}/.iwm_state_diff_rebuild.lock"
if ! flock -n 9; then
  echo "[=] 다른 rebuild_state_diff_metrics 인스턴스가 실행 중 — 종료"
  exit 0
fi

# DS_DATADIR (EVAL_DS → data/ 하위 디렉토리) 는 _common.sh 가 정본이다.
# shellcheck source=./_common.sh
source "$BASE_DIR/scripts/_common.sh"

LOG_ROOT="${LOG_DIR:-$BASE_DIR/logs}/state_diff_rebuild"
mkdir -p "$LOG_ROOT"

# ── 1) 대상 수집 ────────────────────────────────────────────────────────────
# 조건: ID/OOD prediction 둘 다 있고, 정본 state 채점(hungarian_metrics.json)이 끝난
# leaf. 아직 채점 중이거나 재측정 대기로 메트릭을 지워둔 leaf 는 건드리지 않는다.
declare -a LEAVES=()
while IFS= read -r leaf; do
  [ -n "$FILTER" ] && [[ "$leaf" != *"$FILTER"* ]] && continue
  [ -f "$leaf/generated_predictions_id.jsonl" ]  || continue
  [ -f "$leaf/generated_predictions_ood.jsonl" ] || continue
  [ -f "$leaf/hungarian_metrics.json" ]          || continue
  if [ "$FORCE" -eq 0 ] && [ -f "$leaf/state_diff_metrics.json" ]; then
    continue
  fi
  LEAVES+=("$leaf")
done < <(find outputs -type d \( -name 'on-*-state' -o -name 'on-*-state-without-open_app' \) \
           -not -path '*_backup*' | sort)

if [ "${#LEAVES[@]}" -eq 0 ]; then
  echo "[=] 대상 없음 (FILTER='${FILTER}', FORCE=$FORCE)"
  exit 0
fi

# ── 2) 절단 경계 필터 (dry-run 표시용 — 강제는 채점기가 한다) ──────────────
# 경계 시각은 _state_diff_eval 이 정본. mtime 은 leaf 의 prediction 파일 것을 본다.
declare -a KEPT=()
if [ "$INCLUDE_TRUNCATED" -eq 1 ]; then
  KEPT=("${LEAVES[@]}")
  echo "[!] --include-truncated: 절단 leaf 포함 — copy_rate 가 과소평가된다" >&2
else
  # leaf 목록은 **argv 로** 넘긴다. `python - <<'PY'` 는 stdin 을 heredoc 이 가져가므로
  # 파이프로 넣은 목록은 조용히 버려지고 결과가 항상 빈 목록이 된다.
  while IFS= read -r leaf; do
    [ -n "$leaf" ] && KEPT+=("$leaf")
  done < <(python - "${LEAVES[@]}" <<'PY'
import os, sys
from datetime import UTC, datetime
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from _state_diff_eval import MAX_NEW_TOKENS_FIX_UTC as FIX

skipped = []
for leaf in sys.argv[1:]:
    pred = os.path.join(leaf, "generated_predictions_id.jsonl")
    if datetime.fromtimestamp(os.path.getmtime(pred), tz=UTC) >= FIX:
        print(leaf)
    else:
        skipped.append(leaf)
if skipped:
    print(
        f"[=] 절단(1024) 경계 {FIX:%Y-%m-%d %H:%M} UTC 이전이라 제외: "
        f"{len(skipped)} leaf",
        file=sys.stderr,
    )
    for s in skipped:
        print(f"      {s}", file=sys.stderr)
PY
  )
fi

if [ "${#KEPT[@]}" -eq 0 ]; then
  echo "[=] 절단 필터 후 대상 없음"
  exit 0
fi

# ── 3) EVAL_DS 해석 + 명령 조립 ────────────────────────────────────────────
# 채점 모드는 stage1_eval.sh / rebuild_woa_metrics.sh 와 **같은 규칙**이어야 한다.
# 다른 모드로 매기면 element 집합 자체가 달라져 hungarian 표와 나란히 못 놓는다.
CMDS="$(mktemp)"; trap 'rm -f "$CMDS"' EXIT
for leaf in "${KEPT[@]}"; do
  base="$(basename "$leaf")"                       # on-AC_EXP05-state[-without-open_app]
  woa=0
  case "$base" in *-without-open_app) woa=1 ;; esac
  eval_ds="${base#on-}"
  eval_ds="${eval_ds%-without-open_app}"
  eval_ds="${eval_ds%-state}"
  datadir="${DS_DATADIR[$eval_ds]:-}"
  if [ -z "$datadir" ]; then
    echo "[!] DS_DATADIR 에 '$eval_ds' 가 없다 — 건너뜀: $leaf" >&2
    continue
  fi
  mode_flag=""
  case "$eval_ds" in
    AC_EXP05|AC_EXP06|AC_EXP07_v1|AC_EXP07_v2) mode_flag="--match-mode pos" ;;
  esac

  # woa leaf 는 이미 필터된 prediction 을 들고 있으므로 필터된 GT 와 짝지어야 한다.
  # 여기서 --exclude-action 을 다시 주면 이미 빠진 행을 또 거르려다 정렬이 어긋난다.
  if [ "$woa" -eq 1 ]; then
    test_id="$BASE_DIR/data/${datadir}/stage1_test_id_state_without_open_app.jsonl"
    test_ood="$BASE_DIR/data/${datadir}/stage1_test_ood_state_without_open_app.jsonl"
  else
    test_id="$BASE_DIR/data/${datadir}/stage1_test_id_state.jsonl"
    test_ood="$BASE_DIR/data/${datadir}/stage1_test_ood_state.jsonl"
  fi
  if [ ! -f "$test_id" ] || [ ! -f "$test_ood" ]; then
    echo "[!] test jsonl 없음 — 건너뜀: $leaf ($datadir)" >&2
    continue
  fi
  # 행 수가 안 맞으면 `zip(..., strict=False)` 이 조용히 짧은 쪽에 맞춰 잘리고,
  # 파싱 실패 카운터는 0 이라 그대로 통과한다. **ID·OOD 를 둘 다** 본다 — woa leaf 는
  # 필터된 GT 와 짝짓는데 그 파일은 rebuild_woa_metrics.sh 가 따로 만들어서
  # 두 파일의 생성 이력이 갈릴 수 있는 자리가 정확히 여기다.
  skip=0
  for split in id ood; do
    case "$split" in id) t="$test_id" ;; ood) t="$test_ood" ;; esac
    if [ "$(wc -l < "$t")" -ne "$(wc -l < "$leaf/generated_predictions_${split}.jsonl")" ]; then
      echo "[!] ${split^^} 행 수 불일치 — 건너뜀: $leaf" >&2
      skip=1
    fi
  done
  [ "$skip" -eq 1 ] && continue

  tag="$(echo "${leaf#outputs/}" | tr '/' '_')"
  printf '%s\t%s\n' "$tag" \
    "python '$BASE_DIR/scripts/_state_diff_eval.py' score \
       --test-id  '$test_id'  --pred-id  '$leaf/generated_predictions_id.jsonl' \
       --test-ood '$test_ood' --pred-ood '$leaf/generated_predictions_ood.jsonl' \
       $mode_flag --output '$leaf/state_diff_metrics.json'" >> "$CMDS"
done

n=$(wc -l < "$CMDS")
echo "[=] 대상 $n leaf · 동시 $JOBS · 로그 $LOG_ROOT"
if [ "$DRY" -eq 1 ]; then
  cut -f1 "$CMDS" | sed 's/^/  /'
  exit 0
fi

# ── 4) 팬아웃 ──────────────────────────────────────────────────────────────
run_one() {
  local tag cmd log
  tag="${1%%$'\t'*}"; cmd="${1#*$'\t'}"
  log="$LOG_ROOT/${tag}.log"
  if bash -c "$cmd" > "$log" 2>&1; then
    echo "[+] OK   $tag"
  else
    echo "[!] FAIL $tag  (log: $log)" >&2
  fi
}
export -f run_one
export LOG_ROOT

xargs -d '\n' -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} < "$CMDS"
echo "[=] 완료"
