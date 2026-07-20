#!/usr/bin/env bash
# Stage 2 Evaluation — local merged 우선 + HF Hub fallback sweep × 교차 데이터셋.
#
# 학습 DS (TRAIN_DATASET ∈ {AC_EXP01, AC_EXP02, AC_EXP03}) 에서 얻은 merged 모델을
# 여러 평가 DS 에서 sweep. MC 는 Stage 2 학습 데이터/YAML 부재로 미지원.
# (variant, epoch) 별 model path 는 _common.sh::resolve_eval_model_path
# (kind=stage2_base | stage2_world | stage1 for epoch-0) 가 결정하며, local
# merged dir 이 존재하면 그 경로, 없으면 HF Hub repo id 로 fallback.
# AC_EXP01 은 --exp01-ratio {ratio37|ratio55|ratio73} 으로 ratio 별 학습 모델을 지정한다
# (Stage 1 과 동일 패턴).
# AC_EXP02 는 AC_EXP01 ratio73 동일 데이터 + Stage1 state-pred diff loss 실험군.
# EVAL_DS 별 섹션 구성:
#   AC_EXP01 / AC_EXP02 : test_id + test_ood 2-회 inference → action_metrics.json
#        (overall / in_domain / out_of_domain 3-섹션)
#   MB : 단일 파일 implicit-world-modeling_stage2.jsonl 1-회 inference → action_metrics.json
#        (overall 1-섹션, single-pair 모드)
#
# Flags (공통은 _common.sh::parse_eval_args 참고):
#   --model / --train-dataset / --eval-datasets
#   --variants LIST      콤마 구분. 기본: base,full_base,lora_base,full_world_model,lora_world_model
#     base                    : Zero-shot baseline (base model)
#     {full|lora}_base        : SaFD-00/{short}-{slug}base-stage2-{mode2}-epoch{E2}
#     {full|lora}_world_model : SaFD-00/{short}-{slug}world-model-stage1-{STAGE1_MODE}-epoch{STAGE1_EPOCH}-stage2-{mode2}-epoch{E2}
#   --epochs LIST        콤마 구분 정수 (기본 1,2,3).
#     0 포함 시: {full|lora}_world_model 은 epoch-0 = stage1 merged repo
#       (SaFD-00/{short}-{slug}world-model-stage1-{STAGE1_MODE}-epoch{STAGE1_EPOCH},
#        stage2 미학습 베이스라인) 을 평가. {full|lora}_base 는 원본 base 모델과
#       중복(=base variant)이라 skip. base variant 는 epoch 무관(원본 모델 zero-shot).
#   --stage1-mode {full|lora}   world-model variant 의 상류 Stage 1 모드 (기본 full).
#   --stage1-epoch N            world-model variant 에서 HF repo 계보 번호 주입용. 필수.
#
# 산출물:
#   outputs/{TRAIN_DS}/eval/{MODEL}/stage2_eval/{variant}[/epoch-{E}]/on-{EVAL_DS}/
#     EVAL_DS=AC_EXP01 / AC_EXP02 : generated_predictions_{id,ood}.jsonl + action_metrics.json
#     EVAL_DS=MB                  : generated_predictions.jsonl         + action_metrics.json (overall only)

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"
parse_eval_args "$@"
resolve_stage2_variants
export DISABLE_VERSION_CHECK=1

SCRIPT_TAG="stage2_eval"
TRAIN_DS="$TRAIN_DATASET"

case "$TRAIN_DS" in
  # AC_EXP04 stage2 보류 — 데이터/등록 키 없음 (현재 *) 분기로 거부). 도입 시 case + 아래 에러문에 AC_EXP04 포함.
  # AC_EXP05 = xy 통일 액션 스페이스 실험군 — action 채점 시 --coord-mode xy (run_variant_epoch_eval_on 참조).
  AC_EXP01_ratio37|AC_EXP01_ratio55|AC_EXP01_ratio73|AC_EXP02|AC_EXP03|AC_EXP05|AC_EXP06) ;;
  MC)
    echo "[!] Stage 2 는 MonkeyCollection(MC) 학습 데이터를 갖지 않습니다 (got '$TRAIN_DS')." >&2
    echo "    --train-dataset 는 AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP05 만 사용하세요." >&2
    exit 2 ;;
  *)
    echo "[!] Stage 2 eval --train-dataset 는 AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP05 만 지원합니다 (got '$TRAIN_DS')." >&2
    exit 2 ;;
esac

# 한 (MODEL, TRAIN_DS, VARIANT, EPOCH, HUB_ID, EVAL_DS) 조합 평가 실행.
# - EVAL_DS=AC_EXP01 / AC_EXP02 : test_id + test_ood → 3-섹션 action_metrics.
# - EVAL_DS=MB                  : 단일 파일 → overall only action_metrics (single-pair 모드).
run_variant_epoch_eval_on() {
  local model_short="$1" train_ds="$2" variant="$3" epoch="$4" hub_id="$5" \
        out_rel_base="$6" template="$7" eval_ds="$8"
  local out_rel="${out_rel_base}/on-${eval_ds}"
  local out_dir="$LF_ROOT/$out_rel"
  local tag="${SCRIPT_TAG}_${model_short}_${train_ds}_${variant}"
  if [[ -n "$epoch" ]]; then
    tag="${tag}_epoch${epoch}"
  fi
  tag="${tag}_on-${eval_ds}"
  if skip_if_done "$tag" "$out_dir/action_metrics.json"; then
    return 0
  fi

  local datadir="${DS_DATADIR[$eval_ds]}"
  local eval_prefix="${DS_PREFIX[$eval_ds]}"

  # AC_EXP05 는 xy 통일 액션 스페이스라 action 채점 모드가 다르다 (stage1_eval 과 동일).
  # 나머지 EXP 는 플래그 없이 기존 index 채점 경로 그대로.
  local action_mode_flag=""
  if [[ "$eval_ds" == "AC_EXP05" || "$eval_ds" == "AC_EXP06" ]]; then
    action_mode_flag="--coord-mode xy"
  fi

  # Single-test 데이터셋 (overall only): MB.
  if [[ "$eval_ds" == "MB" ]]; then
    local test_jsonl="$BASE_DIR/data/${datadir}/implicit-world-modeling_stage2.jsonl"
    local ds_test="${eval_prefix}_stage2"
    if [ ! -f "$test_jsonl" ]; then
      echo "[!] [$model_short][train=$train_ds][eval=$eval_ds] Missing test file: $test_jsonl" >&2
      exit 1
    fi

    build_infer_cmd "$model_short" "$hub_id" "$ds_test" \
      "$test_jsonl" "$template" \
      "$out_rel/generated_predictions.jsonl" \
      "$out_rel/predict_results.json"

    run_logged "$tag" \
      bash -c "cd '$LF_ROOT' && mkdir -p '$out_rel' && \
        $INFER_CMD && \
        python '$BASE_DIR/scripts/_action_eval.py' score \
          --test   '$test_jsonl' \
          --pred   '$out_dir/generated_predictions.jsonl' \
          $action_mode_flag \
          --output '$out_dir/action_metrics.json'"
  else
    local test_id="$BASE_DIR/data/${datadir}/implicit-world-modeling_stage2_test_id.jsonl"
    local test_ood="$BASE_DIR/data/${datadir}/implicit-world-modeling_stage2_test_ood.jsonl"
    if [ ! -f "$test_id" ] || [ ! -f "$test_ood" ]; then
      echo "[!] [$model_short][train=$train_ds][eval=$eval_ds] Missing test_id/test_ood jsonl:" >&2
      echo "      $test_id" >&2
      echo "      $test_ood" >&2
      exit 1
    fi
    local ds_test_id="${eval_prefix}_stage2_test_id"
    local ds_test_ood="${eval_prefix}_stage2_test_ood"

    build_infer_cmd "$model_short" "$hub_id" "$ds_test_id" \
      "$test_id" "$template" \
      "$out_rel/generated_predictions_id.jsonl" \
      "$out_rel/predict_results_id.json"
    local infer_id="$INFER_CMD"
    build_infer_cmd "$model_short" "$hub_id" "$ds_test_ood" \
      "$test_ood" "$template" \
      "$out_rel/generated_predictions_ood.jsonl" \
      "$out_rel/predict_results_ood.json"
    local infer_ood="$INFER_CMD"

    run_logged "$tag" \
      bash -c "cd '$LF_ROOT' && mkdir -p '$out_rel' && \
        $infer_id && \
        $infer_ood && \
        python '$BASE_DIR/scripts/_action_eval.py' score \
          --test-id  '$test_id' \
          --pred-id  '$out_dir/generated_predictions_id.jsonl' \
          --test-ood '$test_ood' \
          --pred-ood '$out_dir/generated_predictions_ood.jsonl' \
          $action_mode_flag \
          --output   '$out_dir/action_metrics.json'"
  fi
}

for MODEL_SHORT in "${MODELS[@]}"; do
  BASE_MODEL="${MODEL_ID[$MODEL_SHORT]}"
  TEMPLATE="${MODEL_TEMPLATE[$MODEL_SHORT]}"

  # outputs/ 1-level 디렉토리는 ds_outputs_code 로 정규화 (AC_EXP01_ratio* → AndroidControl_EXP01),
  # AC_EXP01 ratio variant 만 model 디렉토리에 _ratio{37,55,73} suffix 를 붙여 충돌 방지.
  OUT_DS="$(ds_outputs_code "$TRAIN_DS")"
  EVAL_SFX="$(ds_model_suffix "$TRAIN_DS")"
  EVAL_DIR_REL="../outputs/${OUT_DS}/eval/${MODEL_SHORT}${EVAL_SFX}/stage2_eval"

  for VARIANT in "${VARIANTS[@]}"; do
    case "$VARIANT" in
      base)
        OUT_REL_BASE="${EVAL_DIR_REL}/base"
        for EVAL_DS in "${EVAL_DATASETS[@]}"; do
          run_variant_epoch_eval_on "$MODEL_SHORT" "$TRAIN_DS" base "" "$BASE_MODEL" \
            "$OUT_REL_BASE" "$TEMPLATE" "$EVAL_DS"
        done
        ;;

      full_base|lora_base)
        MODE2="${VARIANT%_base}"
        echo "[+] [$MODEL_SHORT][train=$TRAIN_DS][$VARIANT] Sweeping stage2 epochs: ${EPOCHS[*]}" >&2
        for EPOCH in "${EPOCHS[@]}"; do
          if [[ "$EPOCH" == "0" ]]; then
            echo "[=] [$MODEL_SHORT][train=$TRAIN_DS][$VARIANT] epoch-0 은 stage1 계보가 없어 원본 base 모델과 동일 — skip (base variant 사용)." >&2
            continue
          fi
          HUB_ID=$(resolve_eval_model_path stage2_base "$MODEL_SHORT" "$TRAIN_DS" "$MODE2" "$EPOCH")
          OUT_REL_BASE="${EVAL_DIR_REL}/${VARIANT}/epoch-${EPOCH}"
          for EVAL_DS in "${EVAL_DATASETS[@]}"; do
            run_variant_epoch_eval_on "$MODEL_SHORT" "$TRAIN_DS" "$VARIANT" "$EPOCH" "$HUB_ID" \
              "$OUT_REL_BASE" "$TEMPLATE" "$EVAL_DS"
          done
        done
        ;;

      full_world_model|lora_world_model)
        if [[ -z "$STAGE1_EPOCH" ]]; then
          echo "[!] [$MODEL_SHORT][train=$TRAIN_DS][$VARIANT] --stage1-epoch 필수." >&2
          exit 2
        fi
        MODE2="${VARIANT%_world_model}"
        VARIANT_PATH="${VARIANT/world_model/world-model}"
        echo "[+] [$MODEL_SHORT][train=$TRAIN_DS][$VARIANT] stage1=${STAGE1_MODE}ep${STAGE1_EPOCH} stage2 epochs: ${EPOCHS[*]}" >&2
        for EPOCH in "${EPOCHS[@]}"; do
          if [[ "$EPOCH" == "0" ]]; then
            # epoch-0 = stage2 미학습 = stage1 merged repo (stage2 mode 무관, 동일 모델).
            # ds_stage1_source 로 stage1 계보 소스 DS 를 해석 (예: AC_EXP06 → AC_EXP05,
            # stage2 비증강 대조군은 EXP06 stage1 을 따로 학습하지 않고 EXP05 를 승계 —
            # HF 폴백 id 도 자동으로 ac-exp05-...-stage1-... 이 된다).
            HUB_ID=$(resolve_eval_model_path stage1 "$MODEL_SHORT" "$(ds_stage1_source "$TRAIN_DS")" "$STAGE1_MODE" "$STAGE1_EPOCH")
          else
            HUB_ID=$(resolve_eval_model_path stage2_world "$MODEL_SHORT" "$TRAIN_DS" \
              "$STAGE1_MODE" "$STAGE1_EPOCH" "$MODE2" "$EPOCH")
          fi
          OUT_REL_BASE="${EVAL_DIR_REL}/${VARIANT_PATH}_from_${STAGE1_MODE}-ep${STAGE1_EPOCH}/epoch-${EPOCH}"
          for EVAL_DS in "${EVAL_DATASETS[@]}"; do
            run_variant_epoch_eval_on "$MODEL_SHORT" "$TRAIN_DS" "$VARIANT" "$EPOCH" "$HUB_ID" \
              "$OUT_REL_BASE" "$TEMPLATE" "$EVAL_DS"
          done
        done
        ;;
    esac
  done
done
