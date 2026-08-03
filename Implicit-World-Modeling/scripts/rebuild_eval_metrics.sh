#!/usr/bin/env bash
# 정규 eval 메트릭 일괄 산출 — GPU 를 쓰지 않는 순수 채점 배치.
#
# 왜 필요한가
# ----------
# stage1_eval.sh 는 생성 직후 같은 job 안에서 채점까지 했다. 채점은 **단일 스레드 CPU**
# 작업인데 job 을 붙잡고 있어 그동안 그 job 의 GPU 가 통째로 논다. 2026-07-30 실측으로
# EXP01 7B `lora ep1`(3000+3000행)은 생성 2h41m + 채점 2h08m 이라 wall 의 44% 가 GPU
# 유휴였다. 남은 state leaf 15개 기준 누적 30시간 규모다.
# 그래서 EVAL_SKIP_SCORE=1 로 job 에서 채점을 떼고, 여기서 64코어를 써서 병렬로 돌린다.
# 입력이 이미 저장된 generated_predictions_*.jsonl 이라 결과는 인라인 채점과 동치다.
#
# 대상 선정 — .gen_done 마커
# --------------------------
# "predictions 줄 수가 test 행 수와 같다"만으로 고르면 안 된다. 재측정 대기 중인 구
# predictions 는 1024 로 잘렸거나 시드 없이 뽑혔어도 **행 수는 온전**해서(2026-07-30
# 실측 37 leaf) 무효 산출물을 채점해 정본 자리에 쓰게 된다. stage1_eval.sh 가 현재
# 예산·시드로 생성을 마쳤을 때만 남기는 `.gen_done` 마커를 유일한 기준으로 삼는다.
#
# usage: rebuild_eval_metrics.sh [-j N] [-n] [FILTER]
#   -j N    동시 실행 수 (기본 4)
#   -n      dry-run
#   FILTER  leaf 경로 부분 문자열로 대상 좁히기
set -euo pipefail

JOBS=4; DRY=0; FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    -n) DRY=1; shift ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *)  FILTER="$1"; shift ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"
# shellcheck source=./_common.sh
source "$BASE_DIR/scripts/_common.sh"

exec 9>"${TMPDIR:-/tmp}/.iwm_eval_rebuild.lock"
if ! flock -n 9; then
  echo "[=] 다른 rebuild_eval_metrics 인스턴스가 실행 중 — 종료"; exit 0
fi

LOG_ROOT="${LOG_DIR:-$BASE_DIR/logs}/eval_rebuild"; mkdir -p "$LOG_ROOT"
CMDS="$(mktemp)"; trap 'rm -f "$CMDS"' EXIT

while IFS= read -r leaf; do
  [ -n "$FILTER" ] && [[ "$leaf" != *"$FILTER"* ]] && continue
  [ -f "$leaf/.gen_done" ] || continue            # ← 유일한 신선도 근거
  base="$(basename "$leaf")"
  case "$base" in
    *-state)  task=state;  scorer=_hungarian_eval.py; metrics=hungarian_metrics.json ;;
    *-action) task=action; scorer=_action_eval.py;    metrics=action_metrics.json ;;
    *) continue ;;
  esac
  [ -f "$leaf/$metrics" ] && continue             # 이미 채점됨
  eval_ds="${base#on-}"; eval_ds="${eval_ds%-$task}"
  datadir="${DS_DATADIR[$eval_ds]:-}"
  if [ -z "$datadir" ]; then echo "[!] DS_DATADIR 에 '$eval_ds' 없음 — 건너뜀: $leaf" >&2; continue; fi

  # 채점 모드 판정은 _common.sh 가 정본 (stage1_eval.sh 와 동일 규칙).
  mode_flag="$(ds_score_mode_flag "$eval_ds" "$task")"
  test_id="$BASE_DIR/data/${datadir}/stage1_test_id_${task}.jsonl"
  test_ood="$BASE_DIR/data/${datadir}/stage1_test_ood_${task}.jsonl"
  if [ ! -f "$test_id" ] || [ ! -f "$test_ood" ]; then
    echo "[!] test jsonl 없음 — 건너뜀: $leaf" >&2; continue
  fi
  # predictions 완성도 재확인 (마커가 있어도 파일이 잘렸으면 채점하면 안 된다)
  if [ "$(wc -l < "$leaf/generated_predictions_id.jsonl" 2>/dev/null || echo 0)"  -ne "$(wc -l < "$test_id")" ] || \
     [ "$(wc -l < "$leaf/generated_predictions_ood.jsonl" 2>/dev/null || echo 0)" -ne "$(wc -l < "$test_ood")" ]; then
    echo "[!] predictions 불완전 — 건너뜀: $leaf" >&2; continue
  fi
  tag="$(echo "${leaf#outputs/}" | tr '/' '_')"
  printf '%s\t%s\n' "$tag" \
    "python '$BASE_DIR/scripts/$scorer' score \
       --test-id  '$test_id'  --pred-id  '$leaf/generated_predictions_id.jsonl' \
       --test-ood '$test_ood' --pred-ood '$leaf/generated_predictions_ood.jsonl' \
       $mode_flag --output '$leaf/$metrics'" >> "$CMDS"
done < <(find outputs -type d \( -name 'on-*-state' -o -name 'on-*-action' \) -not -path '*_backup*' | sort)

n=$(wc -l < "$CMDS")
if [ "$n" -eq 0 ]; then echo "[=] 대상 없음 (FILTER='${FILTER}')"; exit 0; fi
echo "[=] 대상 $n leaf · 동시 $JOBS · 로그 $LOG_ROOT"
if [ "$DRY" -eq 1 ]; then cut -f1 "$CMDS" | sed 's/^/  /'; exit 0; fi

run_rebuild_batch "$CMDS" "$JOBS" "$LOG_ROOT"
echo "[=] 완료"
