#!/usr/bin/env bash
# copy-baseline (복사기 기준선) 메트릭 백필 — GPU 를 쓰지 않는 순수 재채점 배치.
#
# 무엇을 채우나
# -------------
# leaf 옆 `copy_baseline_metrics.json`. "프롬프트의 current state 를 그대로 예측으로
# 낸 가상 모델"의 점수를 지표마다 내고, 같은 행 위에서 `gain = 모델 − 복사기` 를 낸다.
# 저장된 generated_predictions_*.jsonl 만 쓰므로 추론 재실행이 없다.
#
# 왜 절단 필터가 없나 — 이 스크립트가 rebuild_state_diff_metrics.sh 와 갈리는 지점
# ------------------------------------------------------------------------------
# 저쪽은 절단(1024) leaf 를 배치 단계에서 통째로 뺀다. 여기서는 **빼지 않는다.**
# 잘린 것은 예측이지 프롬프트가 아니므로 복사기 점수는 절단과 무관하게 정확하다.
# 판정은 채점기 안(`_copy_baseline_eval`)에서 `_state_diff_eval.truncated_reason` 이
# 하고, 결과는 산출물의 `truncated` 필드 + `model`/`gain` = null 로 남는다.
# 판정을 여기서 다시 구현하면 두 목록이 언젠가 조용히 갈린다 — 안 한다.
#
# usage: rebuild_copy_baseline.sh [-j N] [-n] [-f] [FILTER]
#   -j N     동시 실행 수 (기본 4). 채점기는 사실상 단일 스레드다.
#   -n       dry-run — 대상만 출력.
#   -f       이미 copy_baseline_metrics.json 이 있어도 다시 산출.
#   FILTER   leaf 경로에 이 문자열이 든 것만 대상 (예: EXP05, epoch-3).
set -euo pipefail

JOBS=4
DRY=0
FORCE=0
FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    -n) DRY=1; shift ;;
    -f) FORCE=1; shift ;;
    -h|--help) sed -n '1,22p' "$0"; exit 0 ;;
    *)  FILTER="$1"; shift ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

exec 9>"${TMPDIR:-/tmp}/.iwm_copy_baseline_rebuild.lock"
if ! flock -n 9; then
  echo "[=] 다른 rebuild_copy_baseline 인스턴스가 실행 중 — 종료"
  exit 0
fi

# DS_DATADIR (EVAL_DS → data/ 하위 디렉토리) 는 _common.sh 가 정본이다.
# shellcheck source=./_common.sh
source "$BASE_DIR/scripts/_common.sh"

LOG_ROOT="${LOG_DIR:-$BASE_DIR/logs}/copy_baseline_rebuild"
mkdir -p "$LOG_ROOT"

# ── 1) 대상 수집 ────────────────────────────────────────────────────────────
# 조건: ID/OOD prediction 둘 다 있고, 정본 state 채점(hungarian_metrics.json)이 끝난
# leaf. 아직 채점 중이거나 재측정 대기로 메트릭을 지워둔 leaf 는 건드리지 않는다.
# 절단 leaf 도 여기 남는다 — 그쪽은 model/gain 만 null 로 빠진 산출물이 된다.
declare -a LEAVES=()
while IFS= read -r leaf; do
  [ -n "$FILTER" ] && [[ "$leaf" != *"$FILTER"* ]] && continue
  [ -f "$leaf/generated_predictions_id.jsonl" ]  || continue
  [ -f "$leaf/generated_predictions_ood.jsonl" ] || continue
  [ -f "$leaf/hungarian_metrics.json" ]          || continue
  if [ "$FORCE" -eq 0 ] && [ -f "$leaf/copy_baseline_metrics.json" ]; then
    continue
  fi
  LEAVES+=("$leaf")
done < <( { find outputs -type d \( -name 'on-*-state' -o -name 'on-*-state-without-open_app' \) \
              -not -path '*_backup*'
            # probe_forget = stage2 체크포인트를 **stage1 state test** 로 재평가하는 망각
            # 프로브. leaf 이름이 `on-*-state` 규약을 안 따라서 위 패턴에 안 걸린다 —
            # 그래서 이 계열은 백필 배치에 **한 번도 들어온 적이 없었다**(2026-08-04
            # 발견). 채점 대상은 다른 state leaf 와 완전히 같은 test 파일이다.
            find outputs -type d -path '*/probe_forget/*' -not -path '*_backup*'
          } | sort)

if [ "${#LEAVES[@]}" -eq 0 ]; then
  echo "[=] 대상 없음 (FILTER='${FILTER}', FORCE=$FORCE)"
  exit 0
fi

# ── 2) EVAL_DS 해석 + 명령 조립 ────────────────────────────────────────────
# 채점 모드는 stage1_eval.sh / rebuild_state_diff_metrics.sh 와 **같은 규칙**이어야 한다.
# 다른 모드로 매기면 element 집합 자체가 달라져 model 섹션이 기존 hungarian 표와
# 어긋나고, 그러면 gain 의 피감수와 감수가 다른 세계의 수가 된다.
CMDS="$(mktemp)"; trap 'rm -f "$CMDS"' EXIT
for leaf in "${LEAVES[@]}"; do
  base="$(basename "$leaf")"                       # on-AC_EXP05-state[-without-open_app]
  woa=0
  if [[ "$leaf" == */probe_forget/* ]]; then
    # 이름 규약(on-<DS>-state) 밖이라 EVAL_DS 를 경로에서 읽는다. woa 변형은 없다.
    # 못 읽으면 **건너뛴다** — 엉뚱한 test 와 짝지으면 행 수가 우연히 맞는 한
    # 조용히 완전 오답표가 나온다.
    case "$leaf" in
      outputs/AndroidControl_EXP07/probe_forget/*-v1-*) eval_ds="AC_EXP07_v1" ;;
      outputs/AndroidControl_EXP07/probe_forget/*-v2-*) eval_ds="AC_EXP07_v2" ;;
      *) echo "[!] probe_forget leaf 의 EVAL_DS 를 못 읽었다 — 건너뜀: $leaf" >&2
         continue ;;
    esac
  else
    case "$base" in *-without-open_app) woa=1 ;; esac
    eval_ds="${base#on-}"
    eval_ds="${eval_ds%-without-open_app}"
    eval_ds="${eval_ds%-state}"
  fi
  datadir="${DS_DATADIR[$eval_ds]:-}"
  if [ -z "$datadir" ]; then
    echo "[!] DS_DATADIR 에 '$eval_ds' 가 없다 — 건너뜀: $leaf" >&2
    continue
  fi
  mode_flag="$(ds_score_mode_flag "$eval_ds" state)"

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
    "python '$BASE_DIR/scripts/_copy_baseline_eval.py' score \
       --test-id  '$test_id'  --pred-id  '$leaf/generated_predictions_id.jsonl' \
       --test-ood '$test_ood' --pred-ood '$leaf/generated_predictions_ood.jsonl' \
       $mode_flag --output '$leaf/copy_baseline_metrics.json'" >> "$CMDS"
done

n=$(wc -l < "$CMDS")
echo "[=] 대상 $n leaf · 동시 $JOBS · 로그 $LOG_ROOT"
if [ "$DRY" -eq 1 ]; then
  cut -f1 "$CMDS" | sed 's/^/  /'
  exit 0
fi

# ── 3) 팬아웃 ──────────────────────────────────────────────────────────────
run_rebuild_batch "$CMDS" "$JOBS" "$LOG_ROOT"
echo "[=] 완료"
