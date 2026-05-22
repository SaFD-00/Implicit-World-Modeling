#!/usr/bin/env bash
# Stage 1 Evaluation — HF Hub merged repo sweep × 교차 데이터셋.
#
# 학습 DS (TRAIN_DATASET, HF repo 식별) 와 평가 DS (EVAL_DATASETS, test JSONL)
# 를 분리한다. 학습한 모델 하나를 여러 벤치마크에서 sweep 할 수 있다.
#
# Flags (공통은 _common.sh::parse_eval_args 참고):
#   --model / --train-dataset / --eval-datasets
#   --variants LIST      콤마 구분. 기본: base,full_world_model,lora_world_model
#     base               : Zero-shot baseline (base model)
#     full_world_model   : SaFD-00/{short}-{slug}world-model-stage1-full-epoch{E}
#     lora_world_model   : SaFD-00/{short}-{slug}world-model-stage1-lora-epoch{E}
#   --epochs LIST        콤마 구분 정수 (기본 1,2,3). world-model variant 대상.
#
# EVAL_DS 별 분기 (Stage 2 와 동일 패턴):
#   AC   : test_id + test_ood 2-회 inference → hungarian_metrics.json
#          (overall / in_domain / out_of_domain 3-섹션)
#   MC   : 단일 파일 gui-model_stage1_test.jsonl 1-회 (random split 산출물)
#          → hungarian_metrics.json (overall 1-섹션, single-pair)
#   MB   : 단일 파일 gui-model_stage1.jsonl 1-회 (벤치마크 단일 파일)
#          → hungarian_metrics.json (overall 1-섹션, single-pair)
#   AC_3 : task 별 독립 평가 (state_pred + action_pred). 각 task 는 id/ood 2-section.
#          on-AC_3-state/  ← _hungarian_eval.py score (Stage1 채점, state transition)
#          on-AC_3-action/ ← _action_eval.py score    (Stage2 채점, action prediction)
#          ratio 차원은 학습 산출물(TRAIN_DATASET=AC_3_r{37,55,73}) 에 박혀있고
#          test 파일은 ratio 와 무관하게 4 개로 고정.
#
# without_open_app 자동 산출:
#   각 (variant, EVAL_DS) 마다 정규 eval 직후 추론 재실행 없이
#   _hungarian_eval.py score --exclude-action open_app 한 번을 더 돌려
#   sibling on-{EVAL_DS}-without-open_app/ 디렉토리에 필터된 jsonl + 메트릭을 산출.
#   skip marker 가 별도라서 정규/필터 각각 독립 idempotent.
#   주의: AC_3-action 분기는 _action_eval.py 가 --exclude-action 미지원이라 woa 미산출.
#
# 산출물:
#   outputs/{OUT_DS}/eval/{MODEL}{EVAL_SFX}/stage1_eval/{variant}[/epoch-{E}]/on-{EVAL_DS}/
#     OUT_DS   = ds_outputs_code(TRAIN_DS)  — AC_3_r* → AC_3, 그 외는 그대로.
#     EVAL_SFX = ds_eval_suffix(TRAIN_DS)   — AC=_ac, AC_2=_ac_2, AC_3_r*=_r{37,55,73}, MC="".
#     EVAL_DS=AC          : generated_predictions_{id,ood}.jsonl + hungarian_metrics.json
#     EVAL_DS=MC / MB     : generated_predictions.jsonl          + hungarian_metrics.json (overall only)
#     EVAL_DS=AC_3-state  : generated_predictions_{id,ood}.jsonl + hungarian_metrics.json
#     EVAL_DS=AC_3-action : generated_predictions_{id,ood}.jsonl + action_metrics.json
#   outputs/{OUT_DS}/eval/{MODEL}{EVAL_SFX}/stage1_eval/{variant}[/epoch-{E}]/on-{EVAL_DS}-without-open_app/
#     동일 파일 구조 + predict_results.json (정규 eval 의 schema 와 동일)

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"
parse_eval_args "$@"
resolve_stage1_variants
export DISABLE_VERSION_CHECK=1

SCRIPT_TAG="stage1_eval"
TRAIN_DS="$TRAIN_DATASET"

# AC_3 / AC_4 dual-task eval helper.
# state_pred / action_pred 각각 (id + ood) 2-section 으로 독립 채점.
#   on-{DS}-state/  ← _hungarian_eval.py score (Stage1 채점)
#   on-{DS}-action/ ← _action_eval.py score    (Stage2 채점)
# without_open_app 은 state branch 만 산출 (action branch 의 _action_eval.py 는
# --exclude-action 미지원).
# AC_4 는 AC_3 와 동일 test 파일을 쓴다 (DS_DATADIR[AC_4]=AndroidControl_3).
run_ac3_eval() {
  local model_short="$1" train_ds="$2" variant="$3" epoch="$4" hub_id="$5" \
        out_rel_base="$6" template="$7" eval_ds="${8:-AC_3}"
  local datadir="${DS_DATADIR[$eval_ds]}"
  local eval_prefix="${DS_PREFIX[$eval_ds]}"

  local task subtag scorer metrics_name
  for task in state action; do
    local out_rel="${out_rel_base}/on-${eval_ds}-${task}"
    local out_dir="$LF_ROOT/$out_rel"
    subtag="${SCRIPT_TAG}_${model_short}_${train_ds}_${variant}"
    if [[ -n "$epoch" ]]; then
      subtag="${subtag}_epoch${epoch}"
    fi
    subtag="${subtag}_on-${eval_ds}-${task}"

    if [[ "$task" == "state" ]]; then
      scorer="_hungarian_eval.py"
      metrics_name="hungarian_metrics.json"
    else
      scorer="_action_eval.py"
      metrics_name="action_metrics.json"
    fi

    if skip_if_done "$subtag" "$out_dir/$metrics_name"; then
      continue
    fi

    local test_id="$BASE_DIR/data/${datadir}/gui-model_stage1_test_id_${task}_pred.jsonl"
    local test_ood="$BASE_DIR/data/${datadir}/gui-model_stage1_test_ood_${task}_pred.jsonl"
    if [ ! -f "$test_id" ] || [ ! -f "$test_ood" ]; then
      echo "[!] [$model_short][train=$train_ds][eval=${eval_ds}-${task}] Missing test jsonl:" >&2
      echo "      $test_id" >&2
      echo "      $test_ood" >&2
      exit 1
    fi
    local ds_test_id="${eval_prefix}_stage1_test_id_${task}"
    local ds_test_ood="${eval_prefix}_stage1_test_ood_${task}"

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

    run_logged "$subtag" \
      bash -c "cd '$LF_ROOT' && mkdir -p '$out_rel' && \
        $infer_id && \
        $infer_ood && \
        python '$BASE_DIR/scripts/$scorer' score \
          --test-id  '$test_id' \
          --pred-id  '$out_dir/generated_predictions_id.jsonl' \
          --test-ood '$test_ood' \
          --pred-ood '$out_dir/generated_predictions_ood.jsonl' \
          --output   '$out_dir/$metrics_name'"

    # without_open_app sibling: state task 만 (hungarian_eval 만 --exclude-action 지원).
    if [[ "$task" == "state" ]]; then
      local out_rel_woa="${out_rel}-without-open_app"
      local out_dir_woa="$LF_ROOT/$out_rel_woa"
      local tag_woa="${subtag}_without_open_app"
      if ! skip_if_done "$tag_woa" "$out_dir_woa/$metrics_name"; then
        run_logged "$tag_woa" \
          bash -c "cd '$LF_ROOT' && mkdir -p '$out_rel_woa' && \
            python '$BASE_DIR/scripts/$scorer' score \
              --test-id  '$test_id' \
              --pred-id  '$out_dir/generated_predictions_id.jsonl' \
              --test-ood '$test_ood' \
              --pred-ood '$out_dir/generated_predictions_ood.jsonl' \
              --exclude-action open_app \
              --filtered-test-dir '$BASE_DIR/data/${datadir}' \
              --filtered-pred-dir '$out_dir_woa' \
              --output   '$out_dir_woa/$metrics_name'"
      fi
    fi
  done
}

# 한 (MODEL, TRAIN_DS, VARIANT, EPOCH, HUB_ID, EVAL_DS) 조합 평가 실행.
# - EVAL_DS=AC   : test_id + test_ood → 3-섹션 hungarian_metrics.
# - EVAL_DS=MC   : 단일 파일 gui-model_stage1_test.jsonl  → overall only.
# - EVAL_DS=MB   : 단일 파일 gui-model_stage1.jsonl       → overall only.
# - EVAL_DS=AC_3 : state_pred / action_pred 두 task 독립 채점.
#                  state → hungarian_metrics, action → action_metrics.
run_variant_epoch_eval_on() {
  local model_short="$1" train_ds="$2" variant="$3" epoch="$4" hub_id="$5" \
        out_rel_base="$6" template="$7" eval_ds="$8"

  # AC_3 / AC_4 는 task 별 독립 채점이라 별도 helper 위임.
  if [[ "$eval_ds" == "AC_3" || "$eval_ds" == "AC_4" ]]; then
    run_ac3_eval "$model_short" "$train_ds" "$variant" "$epoch" "$hub_id" \
                 "$out_rel_base" "$template" "$eval_ds"
    return $?
  fi

  local out_rel="${out_rel_base}/on-${eval_ds}"
  local out_dir="$LF_ROOT/$out_rel"
  local tag="${SCRIPT_TAG}_${model_short}_${train_ds}_${variant}"
  if [[ -n "$epoch" ]]; then
    tag="${tag}_epoch${epoch}"
  fi
  tag="${tag}_on-${eval_ds}"
  if skip_if_done "$tag" "$out_dir/hungarian_metrics.json"; then
    return 0
  fi

  local datadir="${DS_DATADIR[$eval_ds]}"
  local eval_prefix="${DS_PREFIX[$eval_ds]}"

  if [[ "$eval_ds" == "AC" ]]; then
    local test_id="$BASE_DIR/data/${datadir}/gui-model_stage1_test_id.jsonl"
    local test_ood="$BASE_DIR/data/${datadir}/gui-model_stage1_test_ood.jsonl"
    if [ ! -f "$test_id" ] || [ ! -f "$test_ood" ]; then
      echo "[!] [$model_short][train=$train_ds][eval=$eval_ds] Missing test_id/test_ood jsonl:" >&2
      echo "      $test_id" >&2
      echo "      $test_ood" >&2
      exit 1
    fi
    local ds_test_id="${eval_prefix}_stage1_test_id"
    local ds_test_ood="${eval_prefix}_stage1_test_ood"

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
        python '$BASE_DIR/scripts/_hungarian_eval.py' score \
          --test-id  '$test_id' \
          --pred-id  '$out_dir/generated_predictions_id.jsonl' \
          --test-ood '$test_ood' \
          --pred-ood '$out_dir/generated_predictions_ood.jsonl' \
          --output   '$out_dir/hungarian_metrics.json'"

    # without_open_app: 추론 재실행 없이 정규 eval 산출물에서 open_app 행만 drop.
    local out_rel_woa="${out_rel}-without-open_app"
    local out_dir_woa="$LF_ROOT/$out_rel_woa"
    local tag_woa="${tag}_without_open_app"
    if ! skip_if_done "$tag_woa" "$out_dir_woa/hungarian_metrics.json"; then
      run_logged "$tag_woa" \
        bash -c "cd '$LF_ROOT' && mkdir -p '$out_rel_woa' && \
          python '$BASE_DIR/scripts/_hungarian_eval.py' score \
            --test-id  '$test_id' \
            --pred-id  '$out_dir/generated_predictions_id.jsonl' \
            --test-ood '$test_ood' \
            --pred-ood '$out_dir/generated_predictions_ood.jsonl' \
            --exclude-action open_app \
            --filtered-test-dir '$BASE_DIR/data/${datadir}' \
            --filtered-pred-dir '$out_dir_woa' \
            --output   '$out_dir_woa/hungarian_metrics.json'"
    fi
  else
    local test_jsonl
    local ds_test
    if [[ "$eval_ds" == "MB" ]]; then
      test_jsonl="$BASE_DIR/data/${datadir}/gui-model_stage1.jsonl"
      ds_test="${eval_prefix}_stage1"
    else  # MC (random split)
      test_jsonl="$BASE_DIR/data/${datadir}/gui-model_stage1_test.jsonl"
      ds_test="${eval_prefix}_stage1_test"
    fi
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
        python '$BASE_DIR/scripts/_hungarian_eval.py' score \
          --test   '$test_jsonl' \
          --pred   '$out_dir/generated_predictions.jsonl' \
          --output '$out_dir/hungarian_metrics.json'"

    # without_open_app: 추론 재실행 없이 정규 eval 산출물에서 open_app 행만 drop.
    local out_rel_woa="${out_rel}-without-open_app"
    local out_dir_woa="$LF_ROOT/$out_rel_woa"
    local tag_woa="${tag}_without_open_app"
    if ! skip_if_done "$tag_woa" "$out_dir_woa/hungarian_metrics.json"; then
      run_logged "$tag_woa" \
        bash -c "cd '$LF_ROOT' && mkdir -p '$out_rel_woa' && \
          python '$BASE_DIR/scripts/_hungarian_eval.py' score \
            --test   '$test_jsonl' \
            --pred   '$out_dir/generated_predictions.jsonl' \
            --exclude-action open_app \
            --filtered-test-dir '$BASE_DIR/data/${datadir}' \
            --filtered-pred-dir '$out_dir_woa' \
            --output '$out_dir_woa/hungarian_metrics.json'"
    fi
  fi
}

for MODEL_SHORT in "${MODELS[@]}"; do
  BASE_MODEL="${MODEL_ID[$MODEL_SHORT]}"
  TEMPLATE="${MODEL_TEMPLATE[$MODEL_SHORT]}"

  # outputs/ 1-level 디렉토리는 ds_outputs_code 로 정규화 (AC_3_r* → AC_3),
  # 모델 디렉토리에는 ds_eval_suffix (AC=_ac, AC_2=_ac_2, AC_3_r*=_r{37,55,73}) 를 붙인다.
  OUT_DS="$(ds_outputs_code "$TRAIN_DS")"
  EVAL_SFX="$(ds_eval_suffix "$TRAIN_DS")"
  EVAL_DIR_REL="../outputs/${OUT_DS}/eval/${MODEL_SHORT}${EVAL_SFX}/stage1_eval"

  for VARIANT in "${VARIANTS[@]}"; do
    case "$VARIANT" in
      base)
        OUT_REL_BASE="${EVAL_DIR_REL}/base"
        for EVAL_DS in "${EVAL_DATASETS[@]}"; do
          run_variant_epoch_eval_on "$MODEL_SHORT" "$TRAIN_DS" base "" "$BASE_MODEL" \
            "$OUT_REL_BASE" "$TEMPLATE" "$EVAL_DS"
        done
        ;;

      full_world_model|lora_world_model)
        MODE="${VARIANT%_world_model}"    # full | lora
        VARIANT_PATH="${VARIANT/world_model/world-model}"
        echo "[+] [$MODEL_SHORT][train=$TRAIN_DS][$VARIANT] Sweeping epochs: ${EPOCHS[*]}" >&2
        for EPOCH in "${EPOCHS[@]}"; do
          HUB_ID=$(hf_repo_id_stage1 "$MODEL_SHORT" "$TRAIN_DS" "$MODE" "$EPOCH")
          OUT_REL_BASE="${EVAL_DIR_REL}/${VARIANT_PATH}/epoch-${EPOCH}"
          for EVAL_DS in "${EVAL_DATASETS[@]}"; do
            run_variant_epoch_eval_on "$MODEL_SHORT" "$TRAIN_DS" "$VARIANT" "$EPOCH" "$HUB_ID" \
              "$OUT_REL_BASE" "$TEMPLATE" "$EVAL_DS"
          done
        done
        ;;
    esac
  done
done
