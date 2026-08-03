#!/usr/bin/env bash
# without_open_app (woa) 메트릭 일괄 (재)산출 — GPU 를 쓰지 않는 순수 재채점 배치.
#
# 왜 별도 스크립트인가
# --------------------
# stage1_eval.sh 는 정규 state 채점 직후 같은 job 안에서 woa 채점을 이어 돌렸다.
# 이 단계는 GPU 를 한 톨도 쓰지 않으면서 job 을 30~60 분 붙잡고, 그동안 그 job 에
# 배정된 GPU 는 논다 (state leaf 하나당 통째로 낭비). EVAL_SKIP_WOA=1 로 job 에서
# 떼어내고, 남은 채점은 이 스크립트가 CPU 만 써서 몰아 돌린다. 입력이 이미 저장된
# generated_predictions_*.jsonl 이라 추론 재실행이 없고, 결과는 인라인 산출과 동치다.
#
# 2026-07-30 함께 고친 것
# -----------------------
# _hungarian_eval.py::_gt_action_type 이 GT 의 action 을 못 읽어 --exclude-action 이
# 전 실험에서 한 행도 거르지 못했다 (woa 산출물이 정규 산출물과 byte-identical, 9/9).
# 파서를 고친 지금 woa 는 비로소 실제 ablation 이 된다 — EXP01/02/03 5.7~6.6%,
# EXP05 6.2~6.9%, MobiBench 14.1% 가 빠진다. EXP07 은 test 에 open 계열이 0 행이라
# 고친 뒤에도 정규 산출물과 같은 값이 나오는 게 정상이다.
# 따라서 파서 수정 이전에 만들어진 woa 메트릭은 전부 무효이며 -f 로 다시 만들어야 한다.
#
# usage: rebuild_woa_metrics.sh [-j N] [-n] [-f] [FILTER]
#   -j N    동시 실행 수 (기본 4). 채점기는 사실상 단일 스레드라 leaf 단위로 벌린다.
#   -n      dry-run — 대상만 출력하고 실행하지 않는다.
#   -f      이미 woa 메트릭이 있어도 다시 산출 (기본은 건너뜀).
#   FILTER  leaf 경로에 이 문자열이 든 것만 대상 (예: EXP05, epoch-3).
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
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *)  FILTER="$1"; shift ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# 주기 실행(워처)과 수동 실행이 겹치면 같은 leaf 에 채점기가 둘 붙는다. 산출은 같지만
# CPU 를 두 배로 먹고 --filtered-pred-dir 을 동시에 쓴다. 한 번에 하나만 돌게 한다.
exec 9>"${TMPDIR:-/tmp}/.iwm_woa_rebuild.lock"
if ! flock -n 9; then
  echo "[=] 다른 rebuild_woa_metrics 인스턴스가 실행 중 — 종료"
  exit 0
fi

# DS_DATADIR (EVAL_DS → data/ 하위 디렉토리) 는 _common.sh 가 정본이다. 여기서 따로
# 표를 만들면 데이터셋이 늘 때마다 조용히 어긋난다.
# shellcheck source=./_common.sh
source "$BASE_DIR/scripts/_common.sh"

LOG_ROOT="${LOG_DIR:-$BASE_DIR/logs}/woa_rebuild"
mkdir -p "$LOG_ROOT"

# ── 1) 대상 수집 ────────────────────────────────────────────────────────────
# 조건: 정규 state 채점이 끝난(hungarian_metrics.json 이 있는) leaf 만. 아직 채점
# 중이거나 재측정 대기로 메트릭을 지워둔 leaf 를 건드리면 반쪽짜리가 남는다.
declare -a LEAVES=()
while IFS= read -r leaf; do
  [ -n "$FILTER" ] && [[ "$leaf" != *"$FILTER"* ]] && continue
  [ -f "$leaf/generated_predictions_id.jsonl" ]  || continue
  [ -f "$leaf/generated_predictions_ood.jsonl" ] || continue
  [ -f "$leaf/hungarian_metrics.json" ]          || continue
  if [ "$FORCE" -eq 0 ] && [ -f "${leaf}-without-open_app/hungarian_metrics.json" ]; then
    continue
  fi
  LEAVES+=("$leaf")
done < <(find outputs -type d -name 'on-*-state' -not -path '*_backup*' | sort)

if [ "${#LEAVES[@]}" -eq 0 ]; then
  echo "[=] 대상 없음 (FILTER='${FILTER}', FORCE=$FORCE)"
  exit 0
fi

# ── 2) EVAL_DS 해석 + 명령 조립 ────────────────────────────────────────────
# on-{EVAL_DS}-state → EVAL_DS. 채점 모드는 stage1_eval.sh 와 같은 규칙을 쓴다
# (AC_EXP05 / AC_EXP07_* 는 xy 통일 액션 스페이스 + index 없는 HTML 이라 --match-mode pos).
CMDS="$(mktemp)"; trap 'rm -f "$CMDS"' EXIT
for leaf in "${LEAVES[@]}"; do
  base="$(basename "$leaf")"          # on-AC_EXP05-state
  eval_ds="${base#on-}"; eval_ds="${eval_ds%-state}"
  datadir="${DS_DATADIR[$eval_ds]:-}"
  if [ -z "$datadir" ]; then
    echo "[!] DS_DATADIR 에 '$eval_ds' 가 없다 — 건너뜀: $leaf" >&2
    continue
  fi
  mode_flag="$(ds_score_mode_flag "$eval_ds" state)"
  test_id="$BASE_DIR/data/${datadir}/stage1_test_id_state.jsonl"
  test_ood="$BASE_DIR/data/${datadir}/stage1_test_ood_state.jsonl"
  if [ ! -f "$test_id" ] || [ ! -f "$test_ood" ]; then
    echo "[!] test jsonl 없음 — 건너뜀: $leaf ($datadir)" >&2
    continue
  fi

  # 필터가 이 데이터셋에서 한 행도 거르지 않으면 woa 는 정규 산출물과 **같은 입력에
  # 같은 결정적 계산**이라 결과가 반드시 같다. EXP07 이 그렇다(test 에 open 계열 0 행).
  # 그런데 채점 비용은 그대로 든다 — 2026-07-30 실측으로 EXP07 leaf 하나에 58분이
  # 걸렸고, 그 CPU 는 같은 장비에서 도는 eval job 의 전처리(num_proc=8)와 경합해
  # GPU 생성 속도를 1.36 → 7.48 s/it 로 떨어뜨렸다. 그래서 계산하지 않고 건너뛴다.
  # (사본을 만들어 두지도 않는다 — 없는 편이 "같은 값"이라는 오해보다 정직하다.)
  woa_id="$BASE_DIR/data/${datadir}/stage1_test_id_state_without_open_app.jsonl"
  woa_ood="$BASE_DIR/data/${datadir}/stage1_test_ood_state_without_open_app.jsonl"
  if [ -f "$woa_id" ] && [ -f "$woa_ood" ] \
     && [ "$(wc -l < "$woa_id")"  -eq "$(wc -l < "$test_id")" ] \
     && [ "$(wc -l < "$woa_ood")" -eq "$(wc -l < "$test_ood")" ]; then
    echo "[=] $eval_ds 는 필터가 0 행을 거른다(정규 산출물과 동일) — 건너뜀: ${leaf#outputs/}" >&2
    continue
  fi
  woa="${leaf}-without-open_app"
  tag="$(echo "${leaf#outputs/}" | tr '/' '_')"
  printf '%s\t%s\n' "$tag" \
    "mkdir -p '$woa' && python '$BASE_DIR/scripts/_hungarian_eval.py' score \
       --test-id  '$test_id'  --pred-id  '$leaf/generated_predictions_id.jsonl' \
       --test-ood '$test_ood' --pred-ood '$leaf/generated_predictions_ood.jsonl' \
       $mode_flag --exclude-action open_app \
       --filtered-test-dir '$BASE_DIR/data/${datadir}' \
       --filtered-pred-dir '$woa' \
       --output '$woa/hungarian_metrics.json'" >> "$CMDS"
done

n=$(wc -l < "$CMDS")
echo "[=] 대상 $n leaf · 동시 $JOBS · 로그 $LOG_ROOT"
if [ "$DRY" -eq 1 ]; then
  cut -f1 "$CMDS" | sed 's/^/  /'
  exit 0
fi

# ── 3) 공유 필터 test jsonl 선점 생성 ──────────────────────────────────────
# --filtered-test-dir 은 같은 datadir 을 쓰는 leaf 들이 공유한다. 병렬로 들어가면
# 여러 프로세스가 같은 경로를 동시에 만든다. _write_jsonl_idempotent 를 atomic 하게
# 고쳐 두긴 했지만, 어차피 한 번만 만들면 되는 파일이라 팬아웃 전에 직렬로 끝낸다.
python - "$CMDS" <<'PY'
import json, re, sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import _hungarian_eval as H

dirs = set()
for line in open(sys.argv[1]):
    for m in re.finditer(r"--filtered-test-dir '([^']+)'", line):
        dirs.add(m.group(1))
for d in sorted(dirs):
    for split in ("id", "ood"):
        src = os.path.join(d, f"stage1_test_{split}_state.jsonl")
        dst = os.path.join(d, f"stage1_test_{split}_state_without_open_app.jsonl")
        if os.path.exists(dst):
            continue
        recs = [json.loads(l) for l in open(src, encoding="utf-8")]
        kept, _ = H._filter_pairs(recs, list(range(len(recs))), "open_app")
        H._write_jsonl_idempotent(kept, dst)
        print(f"[+] filtered test: {dst}  {len(recs)} → {len(kept)}", flush=True)
PY

# ── 4) 팬아웃 ──────────────────────────────────────────────────────────────
run_rebuild_batch "$CMDS" "$JOBS" "$LOG_ROOT"
echo "[=] 완료"
