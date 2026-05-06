#!/usr/bin/env bash
# Stage 1 Merge — 전체 epoch checkpoint 를 각각 merge + HF Hub push.
#
# train → merge → eval 흐름 전환: BEST_CHECKPOINT 의존 제거. 모든
# outputs/{DS}/adapters/{MODEL}_stage1_{MODE}_world-model/checkpoint-*/ 를 순회하며
# epoch 별로 local merge + 개별 HF repo push 한다.
#
# AC_3: --dataset AC_3 입력 시 parse_args 가 DATASETS=(AC_3_r37 AC_3_r55 AC_3_r73)
# 로 펼쳐 ratio 별 outputs 디렉토리를 각각 merge → HF repo (ac-3-r37-, ac-3-r55-,
# ac-3-r73- slug) 로 push 한다. 부분 실행은 --ac3-ratios r55,r73.
#
# --stage1-mode full (default) | lora.
#
# 임시 merge YAML 생성 → llamafactory-cli export
# (lora 모드는 base model + adapter_name_or_path 블록 추가)
#
# HF repo id 규칙 (단일 정의: _common.sh::hf_repo_id_stage1):
#   SaFD-00/{short}-{slug}world-model-stage1-{MODE}-epoch{E}
#
# 로컬 산출물 (사용자 정책: 전부 보존):
#   outputs/{DS}/merged/{MODEL}_stage1_{MODE}_world-model/epoch-{E}/
#
# 요구: HF_TOKEN (.env 또는 환경변수)

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"
parse_args "$@"
export DISABLE_VERSION_CHECK=1

SCRIPT_TAG="stage1_merge_${STAGE1_MODE}"
MERGED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

for MODEL_SHORT in "${MODELS[@]}"; do
  BASE_MODEL="${MODEL_ID[$MODEL_SHORT]}"
  for DS in "${DATASETS[@]}"; do
    # LF cwd 기준 상대경로 (= BASE_DIR 기준 "outputs/...").
    TRAIN_DIR_REL="../outputs/${DS}/adapters/${MODEL_SHORT}_stage1_${STAGE1_MODE}_world-model"
    TRAIN_DIR="$LF_ROOT/$TRAIN_DIR_REL"

    shopt -s nullglob
    CKPTS=("$TRAIN_DIR"/checkpoint-*/)
    shopt -u nullglob
    if [ "${#CKPTS[@]}" -eq 0 ]; then
      echo "[WARN] [$MODEL_SHORT][$DS][$STAGE1_MODE] No checkpoints under $TRAIN_DIR — skipping. Run stage1_train.sh --stage1-mode ${STAGE1_MODE} first." >&2
      SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
      continue
    fi
    echo "[+] [$MODEL_SHORT][$DS][$STAGE1_MODE] Merging ${#CKPTS[@]} checkpoints" >&2

    for CKPT_DIR in "${CKPTS[@]}"; do
      CKPT_DIR="${CKPT_DIR%/}"
      CKPT_NAME=$(basename "$CKPT_DIR")
      EPOCH=$(ckpt_epoch_from_dir "$CKPT_DIR") || {
        echo "[!] [$MODEL_SHORT][$DS][$STAGE1_MODE][$CKPT_NAME] epoch 파싱 실패" >&2
        FAILED_COUNT=$((FAILED_COUNT + 1)); continue
      }

      HUB_ID=$(hf_repo_id_stage1 "$MODEL_SHORT" "$DS" "$STAGE1_MODE" "$EPOCH")
      MERGED_REL="../outputs/${DS}/merged/${MODEL_SHORT}_stage1_${STAGE1_MODE}_world-model/epoch-${EPOCH}"
      LOCAL_DIR="$(local_merged_epoch_dir stage1 "$MODEL_SHORT" "$DS" "$STAGE1_MODE" "$EPOCH")"
      CKPT_REL="./${TRAIN_DIR_REL}/${CKPT_NAME}"

      echo "[+] [$MODEL_SHORT][$DS][$STAGE1_MODE] ${CKPT_NAME} (epoch=${EPOCH}) → ${HUB_ID}" >&2

      TMP_YAML=$(mktemp -t "stage1_merge_${MODEL_SHORT}_${DS}_${STAGE1_MODE}_ep${EPOCH}_XXXXXX.yaml")
      if [ "$STAGE1_MODE" = "full" ]; then
        cat > "$TMP_YAML" <<EOF
### model
model_name_or_path: ${CKPT_REL}
trust_remote_code: true
template: ${MODEL_TEMPLATE[$MODEL_SHORT]}

### export
export_dir: ${MERGED_REL}
export_size: 5
export_device: cpu
export_legacy_format: false
export_hub_model_id: ${HUB_ID}
EOF
      else
        cat > "$TMP_YAML" <<EOF
### model
model_name_or_path: ${BASE_MODEL}
adapter_name_or_path: ${CKPT_REL}
trust_remote_code: true
finetuning_type: lora
template: ${MODEL_TEMPLATE[$MODEL_SHORT]}

### export
export_dir: ${MERGED_REL}
export_size: 5
export_device: cpu
export_legacy_format: false
export_hub_model_id: ${HUB_ID}
EOF
      fi

      if run_logged "${SCRIPT_TAG}_${MODEL_SHORT}_${DS}_epoch${EPOCH}" \
        bash -c "cd '$LF_ROOT' && llamafactory-cli export '$TMP_YAML'"; then
        :
      else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        rm -f "$TMP_YAML"
        continue
      fi
      rm -f "$TMP_YAML"

      if [ ! -d "$LOCAL_DIR" ]; then
        echo "[!] [$MODEL_SHORT][$DS][$STAGE1_MODE][epoch${EPOCH}] Expected output dir missing: $LOCAL_DIR" >&2
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
      fi
      MERGED_COUNT=$((MERGED_COUNT + 1))
    done
  done
done

echo "--- Stage 1 Merge (${STAGE1_MODE}): $MERGED_COUNT merged, $SKIPPED_COUNT skipped, $FAILED_COUNT failed ---" >&2
if [ "$FAILED_COUNT" -gt 0 ]; then
  echo "[!] Some epochs failed. Re-run after fixing." >&2
  exit 1
fi
if [ "$MERGED_COUNT" -eq 0 ] && [ "$SKIPPED_COUNT" -eq 0 ]; then
  echo "[!] No models were merged." >&2
  exit 1
fi
