#!/usr/bin/env bash
# AC_EXP08 stage1 eval v2 스윕 — 구 stage1 매트릭스를 새 11-leaf eval 로 재평가한다.
#
# 왜 이 스크립트가 필요한가
# -------------------------
# `stage1_eval.sh` 는 (variant, stage1-variant, epoch) 한 조합을 순차로 돈다. 구
# 매트릭스는 그런 조합이 32 개이고 각 조합이 11 leaf(state 6 + action 5) 를 낳는다.
# leaf 마다 `skip_if_done` marker 가 독립이라 **중단·재개가 안전하다** — 같은 커맨드를
# 다시 걸어도 끝난 leaf 는 다시 안 돈다. stage2 쪽(run_exp08_stage2_sweep.sh)과 같은 관례다.
#
# stage2 스윕과 다른 점
# ---------------------
#   1. leaf 가 5 개가 아니라 **11 개**다. 그 중 state 6 개는 max_new_tokens=12288 로
#      추론해 (전체 UI XML 예측) action leaf 보다 훨씬 느리다.
#   2. state leaf 는 추론 없이 `--exclude-action open_app` 재채점 sibling
#      (`-without-open_app`) 을 하나 더 낳는다. GPU 를 안 쓰면서 큐를 붙잡는 구간이라
#      급하면 `EVAL_SKIP_WOA=1` 로 끄고 나중에 `rebuild_woa_metrics.sh` 로 몰아 돌린다.
#   3. 그래서 **`base` 를 맨 앞에 둔다** — 가장 싸고 (모델이 이미 캐시돼 있다) 11 leaf
#      실측 시간을 제일 먼저 준다. 전체 ETA 는 그 유닛이 끝난 뒤에 다시 계산해야 한다.
#
# 재고 현실 (HF 조회 + 로컬 체크포인트 확인, 2026-09-08)
# -----------------------------------------------------
#   - `lora_world-model/epoch-1` 은 **재현 불가**다. HF 에 없고(401) 로컬에 stage1 lora
#     어댑터도 없다. 구 아카이브에서도 leaf 하나(state-full)만 있는 미완성 유닛이었다.
#     → 이 스윕에서 빠진다. 32/33 유닛만 돈다.
#   - `world-model-action-only` 11 유닛 중 **8 개가 HF 에 없다**(0.77·1.02·1.28·1.53·
#     1.79·2.29·2.55·2.81). 다만 로컬 `adapters/.../checkpoint-*` 에 전량 남아 있어
#     `stage1_merge.sh --no-hf-upload --stage1-variant action-only` 로 merged/ 를 채우면
#     `resolve_eval_model_path` 가 local hit 으로 잡는다. **스윕 전에 그 merge 가
#     끝나 있어야 한다** — 안 그러면 그 8 유닛이 HF 401 로 죽는다.
#
# 구 아카이브에 있으나 여기서 못 도는 것
# --------------------------------------
# `stage1_eval_2026-08-22/{full,lora}_stage2_*` 11 유닛 (= stage2 체크포인트를 stage1
# state 과제에 얹은 교차 조합) 은 **오늘 저장소의 어떤 스크립트도 만들 수 없다.**
# `stage1_eval.sh` 의 variant 목록은 base/full_world_model/lora_world_model 뿐이고
# (`_common.sh::STAGE1_ALL_VARIANTS`), `probe_forget_eval.sh` 는 EXP07 전용 좌표계다.
# 축을 새로 뚫는 것은 사용자 결정 사항이라 여기서는 **보고만 하고 건드리지 않는다**.
#
# 사용법
# ------
#   bash run_exp08_stage1_sweep.sh --repo /path/to/Implicit-World-Modeling --gpus 0 --dry-run
#   bash run_exp08_stage1_sweep.sh --repo /path/to/Implicit-World-Modeling --gpus 0
#   EVAL_SKIP_WOA=1 bash run_exp08_stage1_sweep.sh --repo ... --gpus 0
#
# 필수 환경: conda env(기본 implicit-world-modeling) · LlamaFactory 체크아웃 · HF 접근
set -euo pipefail

REPO=""
GPUS="0"
CONDA_ENV="${CONDA_ENV:-/opt/miniconda3/envs/implicit-world-modeling}"
DRY_RUN=0
LOGDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --logdir) LOGDIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '1,50p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$REPO" ]] || { echo "--repo 는 필수" >&2; exit 2; }
REPO="$(cd "$REPO" && pwd)"
[[ -f "$REPO/scripts/stage1_eval.sh" ]] || { echo "stage1_eval.sh 가 없다: $REPO" >&2; exit 2; }
LOGDIR="${LOGDIR:-$REPO/logs/sweep_stage1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOGDIR"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"

# ── 작업 목록 ────────────────────────────────────────────────────────────────
# 한 줄 = stage1_eval.sh 인자 (한 유닛 = 11 leaf). 순서는 아카이브 매트릭스 그대로이되
# base 를 맨 앞에 둔다 (위 "stage2 스윕과 다른 점" 3 번 참조).
TASKS=()
add() { TASKS+=("$*"); }

add "--variants base"

# ── 우선순위: 계보별 epoch 1 · 3 먼저 (2026-09-10 사용자 지시) ─────────────────
# "2" 에 정확히 해당하는 체크포인트가 계보마다 없어(가장 가까운 건 2.01/2.04) 근사치는
# 우선순위에 넣지 않고 아래 "나머지" 순서 그대로 둔다 — 사용자가 명시적으로 선택.
add "--variants full_world_model --epochs 1"
add "--variants full_world_model --epochs 3"
add "--variants full_world_model --stage1-variant action-only --epochs 1"
add "--variants full_world_model --stage1-variant action-only --epochs 3"
add "--variants full_world_model --stage1-variant inverse-mix --epochs 1"
add "--variants full_world_model --stage1-variant inverse-mix --epochs 3"

# ── 나머지 (계보별 원래 순서, 위에서 뺀 epoch 1·3 제외) ────────────────────────
for e in 0.25 0.5 0.75 1.25 1.5 1.76 2.01 2.26 2.51 2.76; do
  add "--variants full_world_model --epochs $e"
done
for e in 0.77 1.02 1.28 1.53 1.79 2.04 2.29 2.55 2.81; do
  add "--variants full_world_model --stage1-variant action-only --epochs $e"
done
for e in 0.25 0.5 0.75 1.5 2.01 2.51; do
  add "--variants full_world_model --stage1-variant inverse-mix --epochs $e"
done
# lora_world_model/epoch-1 은 재현 불가 (위 "재고 현실" 참조).

echo "[sweep1] 작업 ${#TASKS[@]} 개 (= 유닛), leaf ${#TASKS[@]}×11 = $(( ${#TASKS[@]} * 11 ))"
echo "[sweep1] GPU ${GPUS} (${NGPU} 장) · 로그 $LOGDIR"

if (( DRY_RUN )); then
  for i in "${!TASKS[@]}"; do
    printf "  gpu%-3s %s\n" "${GPU_ARR[$(( i % NGPU ))]}" "${TASKS[$i]}"
  done
  exit 0
fi

# ── merge 선행 확인 ──────────────────────────────────────────────────────────
# HF 에 없는 action-only 8 유닛은 로컬 merged/ 가 있어야만 돈다. 없으면 스윕이
# 8 유닛을 401 로 날리게 되므로, 조용히 실패하지 말고 **먼저 경고**한다.
AO_MERGED="$REPO/outputs/AndroidControl_EXP08/merged/qwen2.5-vl-3b_stage1_full_world-model-action-only"
for ep in 0.77 1.02 1.28 1.53 1.79 2.29 2.55 2.81; do
  if [[ ! -d "$AO_MERGED/epoch-$ep" ]]; then
    echo "[sweep1][!] action-only epoch-$ep 의 로컬 merged 가 없다 — HF 에도 없어 이 유닛은 실패한다." >&2
    echo "[sweep1][!] 먼저: bash scripts/stage1_merge.sh --model qwen2.5-vl-3b --dataset AC_EXP08 \\" >&2
    echo "[sweep1][!]         --stage1-mode full --stage1-variant action-only --no-hf-upload" >&2
    break
  fi
done

# ── GPU 별 워커 ──────────────────────────────────────────────────────────────
for g in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$g]}"
  (
    for i in "${!TASKS[@]}"; do
      (( i % NGPU == g )) || continue
      echo "[gpu$gpu] $(date +%H:%M:%S) start: ${TASKS[$i]}"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES="$gpu" \
      CONDA_PREFIX="$CONDA_ENV" \
      PYTHONPATH="$REPO/LlamaFactory/src${PYTHONPATH:+:$PYTHONPATH}" \
      LF_CUDA_GUARD_SKIP=1 \
        bash "$REPO/scripts/stage1_eval.sh" \
          --model qwen2.5-vl-3b --train-dataset AC_EXP08 --eval-datasets AC_EXP08 \
          ${TASKS[$i]} \
        || echo "[gpu$gpu] FAILED: ${TASKS[$i]}"
      echo "[gpu$gpu] $(date +%H:%M:%S) done:  ${TASKS[$i]}"
    done
    echo "[gpu$gpu] ALL DONE"
  ) > "$LOGDIR/gpu${gpu}.log" 2>&1 &
  echo "[sweep1] gpu$gpu 워커 pid=$!"
done

wait
echo "[sweep1] 전체 완료. 진행 확인: grep -c 'done:' $LOGDIR/gpu*.log"
