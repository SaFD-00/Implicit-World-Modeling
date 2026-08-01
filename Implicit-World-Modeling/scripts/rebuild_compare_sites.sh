#!/usr/bin/env bash
# EXP 별 정성 비교 사이트 일괄 (재)생성 — outputs/_compare/ 아래.
#
# 무엇을 만드는가
# ---------------
# eval_viewer.py --site 는 (EXP × stage × task) 하나마다 자체 완결형 index.html +
# README.md 를 만든다. 이 스크립트는 outputs/ 에 실제로 존재하는 (EXP, MODEL) 조합을
# 스캔해 EXP 단위로 묶어 --include 를 조립한다. 사이트는 EXP 단위인데(같은 EXP 의 여러
# MODEL 은 test 가 같아 한 사이트 안에서 세팅으로 나란히 놓인다) --include 는
# EXP:MODEL 단위라, 손으로 적으면 조합을 빠뜨리기 쉽다.
#
# EXP07 은 _v1/_v2 를 반드시 분리한다 — stage2 학습셋이 27% 다른 별개 실험군이라
# 한 사이트에 섞으면 "stage1 유무"와 "데이터 교체"가 동시에 움직이는 비교가 된다.
#
# GPU 를 쓰지 않는다. 입력은 이미 저장된 generated_predictions_*.jsonl 이고,
# 표본 행만 정본 채점기로 다시 채점한다 (시드 고정 — 같은 시드면 같은 표본).
#
# usage: rebuild_compare_sites.sh [-n] [--samples N] [--seed N] [FILTER]
#   -n           dry-run — 조립된 --include 만 출력하고 실행하지 않는다.
#   --samples N  분할(ID/OOD/woa)마다 뽑을 표본 수 (기본 20).
#   --seed N     표본 추출 시드 (기본 42).
#   FILTER       EXP 키에 이 문자열이 든 것만 (예: EXP07, EXP05).
set -euo pipefail

cd "$(dirname "$0")/.."

DRY=0
SAMPLES=20
SEED=42
FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n) DRY=1; shift ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) sed -n '1,22p' "$0"; exit 0 ;;
    *)  FILTER="$1"; shift ;;
  esac
done

PYBIN="${PYBIN:-python}"
command -v "$PYBIN" >/dev/null || PYBIN="python3"

# (EXP 키) → "EXP:MODEL EXP:MODEL ..." 조립.
declare -A SPECS=()
for eval_dir in outputs/AndroidControl_EXP*/eval/*/; do
  [ -d "$eval_dir" ] || continue
  model="$(basename "$eval_dir")"
  datadir="$(basename "$(dirname "$(dirname "$eval_dir")")")"   # AndroidControl_EXP07
  exp="AC_${datadir#AndroidControl_}"                            # AC_EXP07
  # EXP07 은 model 디렉토리의 trailing _v1/_v2 가 곧 실험군 버전이다.
  case "$model" in
    *_v1) [ "$exp" = "AC_EXP07" ] && exp="AC_EXP07_v1" ;;
    *_v2) [ "$exp" = "AC_EXP07" ] && exp="AC_EXP07_v2" ;;
  esac
  [ -n "$FILTER" ] && case "$exp" in *"$FILTER"*) ;; *) continue ;; esac
  SPECS["$exp"]="${SPECS[$exp]:-} ${exp}:${model}"
done

if [ "${#SPECS[@]}" -eq 0 ]; then
  echo "[=] 대상 없음 (FILTER='${FILTER}')"; exit 0
fi

fail=0
for exp in $(printf '%s\n' "${!SPECS[@]}" | sort); do
  include="${SPECS[$exp]# }"
  if [ "$DRY" -eq 1 ]; then
    echo "  --include $include"
    continue
  fi
  echo "[=] $exp"
  # 한 EXP 가 실패해도 나머지는 계속 만든다 (사이트끼리 독립).
  if ! "$PYBIN" scripts/eval_viewer.py --site \
        --samples "$SAMPLES" --seed "$SEED" --include $include; then
    echo "[!] FAIL $exp" >&2
    fail=1
  fi
done

[ "$DRY" -eq 1 ] && exit 0
echo "[=] 완료 — outputs/_compare/index.html 을 브라우저로 여세요."
exit "$fail"
