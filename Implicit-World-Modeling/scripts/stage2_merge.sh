#!/usr/bin/env bash
# Stage 2 Merge — 전체 epoch adapter/checkpoint 를 각각 merge + HF Hub push.
#
# Variants:
#   base_${STAGE2_MODE}          - Base model + (full|lora) Stage 2 checkpoint
#   world-model_from_${STAGE1_MODE}-ep${STAGE1_EPOCH}_${STAGE2_MODE}
#                                - Stage 1 local merged (epoch = --stage1-epoch) +
#                                  (full|lora) Stage 2 checkpoint
#
# Flags (all required):
#   --stage1-mode {full|lora}    Stage 1 상류 모델 종류 (world-model variant 전용)
#   --stage1-epoch N             Stage 1 local merged/{MODEL}_stage1_{MODE}_world-model/epoch-N
#                                world-model variant 전용. base variant 에서는 무시.
#   --stage2-mode {full|lora}    Stage 2 학습 방식 (adapter 디렉토리 + HF suffix 결정)
#   --no-hf-upload               local merge 만 수행하고 HF Hub push 는 생략
#   --model / --dataset          (공통)
#
# HF repo id 규칙 (단일 정의: _common.sh):
#   base variant:
#     SaFD-00/{short}-{slug}base-stage2-{STAGE2_MODE}-epoch{E2}
#   world-model variant:
#     SaFD-00/{short}-{slug}world-model-stage1-{STAGE1_MODE}-epoch{E1}-stage2-{STAGE2_MODE}-epoch{E2}
#
# 로컬 산출물 (전부 보존):
#   outputs/{DS}/merged/{MODEL}_stage2_{STAGE2_MODE}_{base|world-model_from_{STAGE1_MODE}-ep{E1}}/epoch-{E2}/
#
# 임시 merge YAML → llamafactory-cli export
#   · full:  model_name_or_path=ckpt (adapter 블록 없음)
#   · lora:  model_name_or_path=base + adapter_name_or_path=ckpt
#
# 요구: HF Hub upload 시 HF_TOKEN (.env 또는 환경변수)

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"
parse_args "$@"
export DISABLE_VERSION_CHECK=1

SCRIPT_TAG="stage2_merge_${STAGE2_MODE}_from_${STAGE1_MODE}"
MERGED_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

for MODEL_SHORT in "${MODELS[@]}"; do
  BASE_MODEL="${MODEL_ID[$MODEL_SHORT]}"

  for DS in "${DATASETS[@]}"; do
    # AC_EXP01 ratio variant: outputs/AndroidControl_EXP01 단일 부모 + model dir 에 _ratio{37,55,73} suffix.
    OUT_DS="$(ds_outputs_code "$DS")"
    SFX="$(ds_model_suffix "$DS")"

    # Stage 1 local merged base (world-model variant 전용). --stage1-epoch 기반.
    # ds_stage1_source 로 stage1 계보 소스 DS 를 해석한다 (예: AC_EXP06 → AC_EXP05,
    # stage2 비증강 대조군은 EXP06 stage1 을 따로 학습하지 않고 EXP05 를 승계).
    # stage2 산출물(OUT_DS/SFX, 위에서 정의)은 원래 DS 를 그대로 유지 — stage1 winner 참조만 소스 기준.
    S1_SRC_DS="$(ds_stage1_source "$DS")"
    S1_OUT_DS="$(ds_outputs_code "$S1_SRC_DS")"
    S1_SFX="$(ds_model_suffix "$S1_SRC_DS")"
    S1_WINNER_AVAILABLE=0
    S1_WINNER_ABS=""
    S1_WINNER_REL=""
    if [[ -n "$STAGE1_EPOCH" ]]; then
      S1_WINNER_ABS="$(local_merged_epoch_dir stage1 "$MODEL_SHORT" "$S1_SRC_DS" "$STAGE1_MODE" "$STAGE1_EPOCH")"
      S1_WINNER_REL="../outputs/${S1_OUT_DS}/merged/${MODEL_SHORT}${S1_SFX}_stage1_${STAGE1_MODE}_world-model/epoch-${STAGE1_EPOCH}"
      if [ -d "$S1_WINNER_ABS" ]; then
        S1_WINNER_AVAILABLE=1
      else
        echo "[WARN] [$MODEL_SHORT][$DS] Missing Stage 1 merged dir (stage1 소스 DS: ${S1_SRC_DS}): $S1_WINNER_ABS" >&2
        echo "       world-model variant 건너뜁니다. (base variant 는 계속 진행)" >&2
      fi
    else
      echo "[WARN] [$MODEL_SHORT][$DS] --stage1-epoch 미지정 → world-model variant 건너뜁니다." >&2
    fi

    # variant key (adapter 디렉토리 suffix 로 사용)
    BASE_VARIANT_KEY="${STAGE2_MODE}_base"
    WM_VARIANT_KEY="${STAGE2_MODE}_world-model_from_${STAGE1_MODE}-ep${STAGE1_EPOCH}"

    declare -A VARIANT_BASE_LF_REL=(
      [base]="$BASE_MODEL"
      [world_model]="${S1_WINNER_REL}"
    )
    declare -A VARIANT_ADAPTER_SUFFIX=(
      [base]="${BASE_VARIANT_KEY}"
      [world_model]="${WM_VARIANT_KEY}"
    )

    for VARIANT in base world_model; do
      if [ "$VARIANT" = "world_model" ] && [ "$S1_WINNER_AVAILABLE" -eq 0 ]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
      fi

      ADAPTER_SUFFIX="${VARIANT_ADAPTER_SUFFIX[$VARIANT]}"
      TRAIN_DIR="$BASE_DIR/outputs/${OUT_DS}/adapters/${MODEL_SHORT}${SFX}_stage2_${ADAPTER_SUFFIX}"
      TRAIN_DIR_REL="../outputs/${OUT_DS}/adapters/${MODEL_SHORT}${SFX}_stage2_${ADAPTER_SUFFIX}"

      shopt -s nullglob
      CKPTS=("$TRAIN_DIR"/checkpoint-*/)
      shopt -u nullglob
      if [ "${#CKPTS[@]}" -eq 0 ]; then
        echo "[WARN] [$MODEL_SHORT][$DS][stage2_${ADAPTER_SUFFIX}] No checkpoints under $TRAIN_DIR — skipping. Run stage2_train.sh first." >&2
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
      fi
      echo "[+] [$MODEL_SHORT][$DS][stage2_${ADAPTER_SUFFIX}] Merging ${#CKPTS[@]} checkpoints" >&2

      for CKPT_DIR in "${CKPTS[@]}"; do
        CKPT_DIR="${CKPT_DIR%/}"
        CKPT_NAME=$(basename "$CKPT_DIR")
        EPOCH=$(ckpt_epoch_from_dir "$CKPT_DIR") || {
          echo "[!] [$MODEL_SHORT][$DS][stage2_${ADAPTER_SUFFIX}][$CKPT_NAME] epoch 파싱 실패" >&2
          FAILED_COUNT=$((FAILED_COUNT + 1)); continue
        }

        if [ "$HF_UPLOAD" -eq 1 ]; then
          if [ "$VARIANT" = "base" ]; then
            HUB_ID=$(hf_repo_id_stage2_base "$MODEL_SHORT" "$DS" "$STAGE2_MODE" "$EPOCH")
          else
            HUB_ID=$(hf_repo_id_stage2_world_model "$MODEL_SHORT" "$DS" \
              "$STAGE1_MODE" "$STAGE1_EPOCH" "$STAGE2_MODE" "$EPOCH")
          fi
          TARGET_DESC="$HUB_ID"
        else
          HUB_ID=""
          TARGET_DESC="local-only"
        fi
        MERGED_REL="../outputs/${OUT_DS}/merged/${MODEL_SHORT}${SFX}_stage2_${ADAPTER_SUFFIX}/epoch-${EPOCH}"
        LOCAL_DIR="$(local_merged_epoch_dir stage2 "$MODEL_SHORT" "$DS" "$ADAPTER_SUFFIX" "$EPOCH")"
        ADAPTER_REL="${TRAIN_DIR_REL}/${CKPT_NAME}"

        echo "[+] [$MODEL_SHORT][$DS][stage2_${ADAPTER_SUFFIX}] ${CKPT_NAME} (epoch=${EPOCH}) → ${TARGET_DESC}" >&2

        TMP_YAML=$(mktemp -t "stage2_merge_${MODEL_SHORT}_${DS}_${VARIANT}_ep${EPOCH}_XXXXXX.yaml")
        if [ "$STAGE2_MODE" = "full" ]; then
          # Full FT: checkpoint 자체가 이미 전체 모델 → adapter 없음.
          cat > "$TMP_YAML" <<EOF
### model
model_name_or_path: ${ADAPTER_REL}
trust_remote_code: true
template: ${MODEL_TEMPLATE[$MODEL_SHORT]}

### export
export_dir: ${MERGED_REL}
export_size: 5
export_device: cpu
export_legacy_format: false
EOF
        else
          cat > "$TMP_YAML" <<EOF
### model
model_name_or_path: ${VARIANT_BASE_LF_REL[$VARIANT]}
adapter_name_or_path: ${ADAPTER_REL}
trust_remote_code: true
finetuning_type: lora
template: ${MODEL_TEMPLATE[$MODEL_SHORT]}

### export
export_dir: ${MERGED_REL}
export_size: 5
export_device: cpu
export_legacy_format: false
EOF
        fi
        if [ "$HF_UPLOAD" -eq 1 ]; then
          cat >> "$TMP_YAML" <<EOF
export_hub_model_id: ${HUB_ID}
EOF
        fi
        if ! run_logged "${SCRIPT_TAG}_${MODEL_SHORT}_${DS}_${VARIANT}_epoch${EPOCH}" \
          bash -c "cd '$LF_ROOT' && llamafactory-cli export '$TMP_YAML'"; then
          FAILED_COUNT=$((FAILED_COUNT + 1))
          rm -f "$TMP_YAML"
          continue
        fi
        rm -f "$TMP_YAML"

        if [ ! -d "$LOCAL_DIR" ]; then
          echo "[!] [$MODEL_SHORT][$DS][stage2_${ADAPTER_SUFFIX}][epoch${EPOCH}] Expected output dir missing: $LOCAL_DIR" >&2
          FAILED_COUNT=$((FAILED_COUNT + 1))
          continue
        fi
        MERGED_COUNT=$((MERGED_COUNT + 1))
      done
    done

    unset VARIANT_BASE_LF_REL VARIANT_ADAPTER_SUFFIX
  done
done

echo "--- Stage 2 Merge (stage2=${STAGE2_MODE} from stage1=${STAGE1_MODE}): $MERGED_COUNT merged, $SKIPPED_COUNT skipped, $FAILED_COUNT failed ---" >&2
if [ "$FAILED_COUNT" -gt 0 ]; then
  echo "[!] Some epochs failed. Re-run after fixing." >&2
  exit 1
fi
if [ "$MERGED_COUNT" -eq 0 ] && [ "$SKIPPED_COUNT" -eq 0 ]; then
  echo "[!] No variants were merged." >&2
  exit 1
fi
