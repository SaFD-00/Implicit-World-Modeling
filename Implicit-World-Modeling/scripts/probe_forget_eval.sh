#!/usr/bin/env bash
# 망각 프로브 — stage2 체크포인트를 **stage1 state test** 로 재평가한다.
#
# 무엇을 재나
# -----------
# stage1 world-model 학습으로 얻은 "변화를 예측하는 능력"이 stage2(action) 학습을
# 거치며 얼마나 되돌아가는지. 평가면은 stage1 state test 그대로라 stage1 leaf 와
# 같은 표에 놓고 읽을 수 있다 (2026-08-04 실측: stage1 lora ep1 의 `added_recall`
# 0.237 이 stage2 1 epoch 만에 0.045 로, `copy_excess` 는 +0.063 → +0.184 로 복귀).
#
# 왜 stage1_eval.sh 로 못 하나
# ----------------------------
# 그쪽은 산출 경로가 `eval/{MODEL}/stage1_eval/{variant}/epoch-{E}/on-{DS}-state` 로
# 고정이고 variant 목록도 stage1 계열(base/full/lora world-model)뿐이다. 여기서
# 평가하는 것은 **stage2 산출물을 stage1 과제에 얹은** 교차 조합이라 그 좌표계에
# 자리가 없다. 그래서 별도 진입점 + 별도 디렉토리(`probe_forget/`)를 쓴다.
#
# ⚠️ 디렉토리 이름이 `on-*-state` 규약 밖이라 `rebuild_*.sh` 의 leaf 발견에 안 걸린다.
#    `rebuild_state_diff_metrics.sh` 에는 2026-08-04 에 경로를 따로 넣었다. 새 배치
#    스크립트를 만들 때 같은 함정을 밟지 않도록 이 주석을 남긴다.
#
# 셀 좌표 (v1 한정 — v2 는 merge X 계보가 섞여 축이 하나 더 는다)
#   mergeO : stage1 lora ep1 → stage2 lora ep{E}   → probe_forget/mergeO-v1-s2ep{E}
#   onlyS2 : stage1 없이 base → stage2 lora ep{E}  → probe_forget/onlyS2-v1-ep{E}
#
# usage: probe_forget_eval.sh [-n] [-f] [--cells LIST]
#   -n            dry-run — 대상 셀과 모델 경로만 출력.
#   -f            산출물이 있어도 다시 돌린다.
#   --cells LIST  콤마 구분. 기본 전체 6셀. 예: mergeO-2,onlyS2-1,onlyS2-2
set -euo pipefail

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"

DRY=0
FORCE=0
CELLS="mergeO-1,mergeO-2,mergeO-3,onlyS2-1,onlyS2-2,onlyS2-3"
while [ $# -gt 0 ]; do
  case "$1" in
    -n) DRY=1; shift ;;
    -f) FORCE=1; shift ;;
    --cells) CELLS="$2"; shift 2 ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "[!] 알 수 없는 인자: $1" >&2; exit 2 ;;
  esac
done

export DISABLE_VERSION_CHECK=1

MODEL_SHORT="qwen2.5-vl-3b"
TRAIN_DS="AC_EXP07_v1"
TEMPLATE="${MODEL_TEMPLATE[$MODEL_SHORT]}"
DATADIR="${DS_DATADIR[$TRAIN_DS]}"
EVAL_PREFIX="${DS_PREFIX[$TRAIN_DS]}"
OUT_ROOT="outputs/$(ds_outputs_code "$TRAIN_DS")/probe_forget"
# 정본 채점과 **같은 모드**여야 stage1 leaf 와 나란히 놓을 수 있다 (EXP07 = pos).
MODE_FLAG="$(ds_score_mode_flag "$TRAIN_DS" state)"
# state 예측 = 전체 UI XML. 라벨 max ~11k 토큰이라 1024 기본값이면 전량 무효가 된다.
INFER_MNT=12288

TEST_ID="$BASE_DIR/data/${DATADIR}/stage1_test_id_state.jsonl"
TEST_OOD="$BASE_DIR/data/${DATADIR}/stage1_test_ood_state.jsonl"
for f in "$TEST_ID" "$TEST_OOD"; do
  [ -f "$f" ] || { echo "[!] test jsonl 없음: $f" >&2; exit 1; }
done

IFS=',' read -r -a CELL_ARR <<< "$CELLS"
for cell in "${CELL_ARR[@]}"; do
  variant="${cell%-*}"
  ep="${cell##*-}"
  case "$variant" in
    mergeO)
      # merge O = stage1 lora ep1 을 병합한 모델 위에 stage2 LoRA 를 새로 학습.
      leaf_name="mergeO-v1-s2ep${ep}"
      model_path="$(resolve_eval_model_path stage2_world "$MODEL_SHORT" "$TRAIN_DS" lora 1 lora "$ep")"
      ;;
    onlyS2)
      # stage1 없이 base 에서 바로 stage2 — 망각의 대조군(잊을 것이 애초에 없다).
      leaf_name="onlyS2-v1-ep${ep}"
      model_path="$(resolve_eval_model_path stage2_base "$MODEL_SHORT" "$TRAIN_DS" lora "$ep")"
      ;;
    *) echo "[!] 알 수 없는 셀: $cell (mergeO-{1,2,3} / onlyS2-{1,2,3})" >&2; exit 2 ;;
  esac

  out_rel="${OUT_ROOT}/${leaf_name}"
  # LF_ROOT 기준 — build_infer_cmd 가 cwd=$LF_ROOT 에서 $out_rel(상대경로)로 쓰므로
  # 채점도 같은 기준으로 읽어야 한다 (stage1_eval.sh 의 out_dir 규약과 동일).
  # BASE_DIR 기준으로 두면 LlamaFactory/outputs 심링크가 없을 때 추론과 채점이
  # 서로 다른 물리 위치를 보게 되어 "방금 쓴 파일을 못 찾음"으로 조용히 죽는다
  # (2026-08-04 실측: mergeO-v1-s2ep2 가 이 버그로 90분 추론 후 채점 직전 FAILED,
  #  LlamaFactory/outputs 심링크가 깨져 있던 것과 겹쳐 원인이 두 겹이었다).
  out_dir="$LF_ROOT/$out_rel"
  subtag="probe_forget_${MODEL_SHORT}_${leaf_name}"

  if [ "$DRY" -eq 1 ]; then
    printf '  %-20s %s\n' "$leaf_name" "$model_path"
    continue
  fi
  if [ "$FORCE" -eq 0 ] && [ -f "$out_dir/hungarian_metrics.json" ]; then
    echo "[=] [$subtag] skip (이미 채점됨)"
    continue
  fi

  build_infer_cmd "$MODEL_SHORT" "$model_path" "${EVAL_PREFIX}_stage1_test_id_state" \
    "$TEST_ID" "$TEMPLATE" \
    "$out_rel/generated_predictions_id.jsonl" "$out_rel/predict_results_id.json" "$INFER_MNT"
  infer_id="$INFER_CMD"
  build_infer_cmd "$MODEL_SHORT" "$model_path" "${EVAL_PREFIX}_stage1_test_ood_state" \
    "$TEST_OOD" "$TEMPLATE" \
    "$out_rel/generated_predictions_ood.jsonl" "$out_rel/predict_results_ood.json" "$INFER_MNT"
  infer_ood="$INFER_CMD"

  # 채점은 정본 진입점 하나로 한다 — `_hungarian_eval score` 가 같은 실행에서
  # state_diff_metrics.json(copy-bias + change 축)까지 낸다.
  run_logged "$subtag" \
    bash -c "cd '$LF_ROOT' && mkdir -p '$out_rel' && \
      $infer_id && \
      $infer_ood && \
      python '$BASE_DIR/scripts/_hungarian_eval.py' score \
        --test-id  '$TEST_ID'  --pred-id  '$out_dir/generated_predictions_id.jsonl' \
        --test-ood '$TEST_OOD' --pred-ood '$out_dir/generated_predictions_ood.jsonl' \
        $MODE_FLAG --output '$out_dir/hungarian_metrics.json'"
done

[ "$DRY" -eq 1 ] || echo "[=] 완료"
