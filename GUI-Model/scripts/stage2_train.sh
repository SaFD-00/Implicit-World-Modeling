#!/usr/bin/env bash
# Stage 2 Fine-tuning — 2 variants × {full, lora}:
#
#   base                              - Base model + stage2 학습
#   world-model-${STAGE1_MODE}        - Stage 1 local merged (epoch = --stage1-epoch)
#                                       을 base 로 삼아 stage2 학습
#
# Flags:
#   --stage1-mode {full|lora}    Stage 1 상류 소스 (world-model variant 용)
#   --stage1-epoch N             Stage 1 local merged/{MODEL}_stage1_{MODE}/epoch-N
#                                을 world-model variant 의 base 로 사용.
#                                world-model variant 에서 필수.
#   --stage2-mode {full|lora}    Stage 2 학습 방식 (기본 lora).
#   --model / --dataset          (공통)
#
# YAML 위치:
#   examples/custom/GUI-Model-${DS}/stage2_{MODE}/{MODEL}_{VARIANT}.yaml
#
# world-model variant 는 노트북이 생성한 YAML 의 model_name_or_path 를 런타임에
# Stage 1 local merged 경로로 sed 치환한다 (임시 YAML). 또한 output_dir 의
# `__STAGE1_EPOCH__` 플레이스홀더를 `$STAGE1_EPOCH` 값으로 치환하여 stage1
# upstream epoch 별 분리 저장 (`..._world-model_from_${MODE1}-ep${STAGE1_EPOCH}`).
#
# NPROC_PER_NODE 은 .env 에서 관리 (기본값 2).

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"
parse_args "$@"
export DISABLE_VERSION_CHECK=1
: "${NPROC_PER_NODE:=2}"

SCRIPT_TAG="stage2_train_${STAGE2_MODE}_from_${STAGE1_MODE}"

resolve_stage1_base() {
  # 반환: LF cwd 기준 상대경로 "../outputs/{OUT_DS}/merged/{MODEL}{SFX}_stage1_{MODE}_world-model/epoch-N".
  # AC_3 ratio variant 는 OUT_DS=AC_3, SFX=_r{37,55,73} 로 ratio 별 stage1 merged 를 가리킨다.
  local model_short="$1" ds="$2" mode="$3" epoch="$4"
  local abs; abs="$(local_merged_epoch_dir stage1 "$model_short" "$ds" "$mode" "$epoch")"
  if [ ! -d "$abs" ]; then
    echo "[!] Missing Stage 1 merged dir: $abs" >&2
    echo "    먼저 stage1_train.sh + stage1_merge.sh 를 --stage1-mode ${mode} 로 돌리고, epoch-${epoch} 가 로컬에 있는지 확인하세요." >&2
    return 1
  fi
  local out_ds; out_ds="$(ds_outputs_code "$ds")"
  local sfx;    sfx="$(ds_model_suffix "$ds")"
  echo "../outputs/${out_ds}/merged/${model_short}${sfx}_stage1_${mode}_world-model/epoch-${epoch}"
}

for MODEL_SHORT in "${MODELS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    VARIANTS_LOCAL=("base" "world-model-${STAGE1_MODE}")
    # --variants 로 일부 variant 만 선택 (예: world-model-lora).
    if [[ "${#VARIANTS[@]}" -gt 0 ]]; then
      FILTERED=()
      for v in "${VARIANTS_LOCAL[@]}"; do
        for w in "${VARIANTS[@]}"; do
          if [[ "$v" == "$w" ]]; then FILTERED+=("$v"); break; fi
        done
      done
      VARIANTS_LOCAL=("${FILTERED[@]}")
    fi

    for VARIANT in "${VARIANTS_LOCAL[@]}"; do
      # world-model variant 는 --stage1-epoch 가 필수.
      if [[ "$VARIANT" == world-model-* ]]; then
        if [[ -z "$STAGE1_EPOCH" ]]; then
          echo "[!] [$MODEL_SHORT][$DS][$VARIANT] --stage1-epoch 가 지정되어야 합니다." >&2
          exit 2
        fi
      fi

      YAML_REL="examples/custom/GUI-Model-${DS}/stage2_${STAGE2_MODE}/${MODEL_SHORT}_${VARIANT}.yaml"
      YAML_ABS="$LF_ROOT/$YAML_REL"
      require_yaml "$YAML_REL" "run notebook Cell (Stage 2 ${STAGE2_MODE}) to generate this YAML"
      RUN_YAML_REL="$YAML_REL"

      if [[ "$VARIANT" == world-model-* ]]; then
        S1_BASE=$(resolve_stage1_base "$MODEL_SHORT" "$DS" "$STAGE1_MODE" "$STAGE1_EPOCH") || exit 1
        TMP_YAML=$(mktemp -t "stage2_train_${MODEL_SHORT}_${DS}_${VARIANT}_${STAGE2_MODE}_XXXXXX.yaml")
        sed -e "0,/^model_name_or_path:/{s|^model_name_or_path:.*|model_name_or_path: ${S1_BASE}|}" \
            -e "s|__STAGE1_EPOCH__|${STAGE1_EPOCH}|g" \
          "$YAML_ABS" > "$TMP_YAML"
        LINK_REL="examples/custom/GUI-Model-${DS}/stage2_${STAGE2_MODE}/.${MODEL_SHORT}_${VARIANT}.runtime.yaml"
        ln -sfn "$TMP_YAML" "$LF_ROOT/$LINK_REL"
        RUN_YAML_REL="$LINK_REL"
        echo "[+] [$MODEL_SHORT][$DS][$VARIANT][stage2=${STAGE2_MODE}] Stage 1 base = $S1_BASE" >&2
        trap 'rm -f "$TMP_YAML" "$LF_ROOT/$LINK_REL"' RETURN
      fi

      run_logged "${SCRIPT_TAG}_${MODEL_SHORT}_${DS}_${VARIANT}" \
        bash -c "cd '$LF_ROOT' && llamafactory-cli train '$RUN_YAML_REL'"

      if [[ "$VARIANT" == world-model-* ]]; then
        rm -f "$TMP_YAML" "$LF_ROOT/$LINK_REL"
        trap - RETURN
      fi
    done
  done
done
