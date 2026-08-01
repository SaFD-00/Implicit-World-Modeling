#!/usr/bin/env bash
# Stage 1 Evaluation — local merged 우선 + HF Hub fallback sweep × 교차 데이터셋.
#
# 학습 DS (TRAIN_DATASET, merged 모델 식별) 와 평가 DS (EVAL_DATASETS, test JSONL)
# 를 분리한다. 학습한 모델 하나를 여러 벤치마크에서 sweep 할 수 있다.
# (variant, epoch) 별 model path 는 _common.sh::resolve_eval_model_path 가
# local merged dir (outputs/.../merged/.../epoch-{E}/) 존재 여부로 결정한다 —
# 있으면 그 절대 경로, 없으면 HF Hub repo id 로 fallback.
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
#   MC       : 단일 파일 stage1_test.jsonl 1-회 (random split 산출물)
#              → hungarian_metrics.json (overall 1-섹션, single-pair)
#   MB       : 단일 파일 stage1.jsonl 1-회 (벤치마크 단일 파일)
#              → hungarian_metrics.json (overall 1-섹션, single-pair)
#   AC_EXP01 / AC_EXP02 : task 별 독립 평가 (state_pred + action_pred). 각 task 는 id/ood 2-section.
#              on-{DS}-state/  ← _hungarian_eval.py score (Stage1 채점, state transition)
#              on-{DS}-action/ ← _action_eval.py score    (Stage2 채점, action prediction)
#              ratio 차원은 학습 산출물(TRAIN_DATASET=AC_EXP01_ratio{37,55,73}) 에 박혀있고
#              test 파일은 ratio 와 무관하게 4 개로 고정.
#
# without_open_app 자동 산출:
#   각 (variant, EVAL_DS) 마다 정규 eval 직후 추론 재실행 없이
#   _hungarian_eval.py score --exclude-action open_app 한 번을 더 돌려
#   sibling on-{EVAL_DS}-without-open_app/ 디렉토리에 필터된 jsonl + 메트릭을 산출.
#   skip marker 가 별도라서 정규/필터 각각 독립 idempotent.
#   EVAL_SKIP_WOA=1 로 이 단계를 통째로 끌 수 있다 — GPU 를 안 쓰면서 job 을 붙잡는
#   구간이라, 배치로 몰아 돌리려면 scripts/rebuild_woa_metrics.sh 를 쓴다.
#   주의: AC_EXP01-action 분기는 _action_eval.py 가 --exclude-action 미지원이라 woa 미산출.
#
# 산출물:
#   outputs/{OUT_DS}/eval/{MODEL}{EVAL_SFX}/stage1_eval/{variant}[/epoch-{E}]/on-{EVAL_DS}/
#     OUT_DS   = ds_outputs_code(TRAIN_DS)  — AC_EXP01_ratio* → AndroidControl_EXP01, 그 외는 그대로.
#     EVAL_SFX = ds_model_suffix(TRAIN_DS)  — AC_EXP01_ratio*=_ratio{37,55,73}, 그 외="".
#     EVAL_DS=MC / MB           : generated_predictions.jsonl          + hungarian_metrics.json (overall only)
#     EVAL_DS=AC_EXP01-state / AC_EXP02-state   : generated_predictions_{id,ood}.jsonl + hungarian_metrics.json
#     EVAL_DS=AC_EXP01-action / AC_EXP02-action : generated_predictions_{id,ood}.jsonl + action_metrics.json
#   outputs/{OUT_DS}/eval/{MODEL}{EVAL_SFX}/stage1_eval/{variant}[/epoch-{E}]/on-{EVAL_DS}-without-open_app/
#     동일 파일 구조 + predict_results.json (정규 eval 의 schema 와 동일)

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"
parse_eval_args "$@"
resolve_stage1_variants
export DISABLE_VERSION_CHECK=1

SCRIPT_TAG="stage1_eval"
TRAIN_DS="$TRAIN_DATASET"

# AC_EXP01 / AC_EXP02 dual-task eval helper.
# state_pred / action_pred 각각 (id + ood) 2-section 으로 독립 채점.
#   on-{DS}-state/  ← _hungarian_eval.py score (Stage1 채점)
#   on-{DS}-action/ ← _action_eval.py score    (Stage2 채점)
# without_open_app 은 state branch 만 산출 (action branch 의 _action_eval.py 는
# --exclude-action 미지원).
# AC_EXP02 는 AndroidControl_EXP02 폴더의 자체 test 파일을 쓴다 (DS_DATADIR[AC_EXP02]=AndroidControl_EXP02).
run_exp01_eval() {
  local model_short="$1" train_ds="$2" variant="$3" epoch="$4" hub_id="$5" \
        out_rel_base="$6" template="$7" eval_ds="${8:-AC_EXP01}"
  local datadir="${DS_DATADIR[$eval_ds]}"
  local eval_prefix="${DS_PREFIX[$eval_ds]}"

  # AC_EXP05 는 xy 통일 액션 스페이스 + index 속성 없는 HTML 이라 채점 모드가 다르다.
  # 나머지 EXP 는 플래그를 붙이지 않아 기존 채점 경로 그대로.
  local state_mode_flag="" action_mode_flag=""
  if [[ "$eval_ds" == "AC_EXP05" || "$eval_ds" == "AC_EXP07_v1" || "$eval_ds" == "AC_EXP07_v2" ]]; then
    state_mode_flag="--match-mode pos"
    action_mode_flag="--coord-mode xy"
  fi

  # EVAL_TASKS (공백 구분, 기본 "state action") 로 task 를 좁힐 수 있다. state 와 action 은
  # skip marker / 산출 디렉토리가 서로 독립이라, 두 값을 각각 다른 GPU 프로세스에 배정해
  # (CUDA_VISIBLE_DEVICES=0 EVAL_TASKS=state / =1 EVAL_TASKS=action) 병렬 평가가 가능하다.
  # 미설정 시 기존 동작(state → action 순차) 그대로.
  local task subtag scorer metrics_name mode_flag
  for task in ${EVAL_TASKS:-state action}; do
    local out_rel="${out_rel_base}/on-${eval_ds}-${task}"
    local out_dir="$LF_ROOT/$out_rel"
    subtag="${SCRIPT_TAG}_${model_short}_${train_ds}_${variant}"
    if [[ -n "$epoch" ]]; then
      subtag="${subtag}_epoch${epoch}"
    fi
    subtag="${subtag}_on-${eval_ds}-${task}"

    local infer_mnt
    if [[ "$task" == "state" ]]; then
      scorer="_hungarian_eval.py"
      metrics_name="hungarian_metrics.json"
      mode_flag="$state_mode_flag"
      # state 예측 = 전체 UI XML (라벨 max ~11k 토큰) → 데이터 최대치를 덮는 예산.
      infer_mnt=12288
    else
      scorer="_action_eval.py"
      metrics_name="action_metrics.json"
      mode_flag="$action_mode_flag"
      # action 예측 = action JSON (라벨 max ~440 토큰) → 기본 예산으로 충분.
      infer_mnt=2048
    fi

    if skip_if_done "$subtag" "$out_dir/$metrics_name"; then
      continue
    fi

    local test_id="$BASE_DIR/data/${datadir}/stage1_test_id_${task}.jsonl"
    local test_ood="$BASE_DIR/data/${datadir}/stage1_test_ood_${task}.jsonl"
    if [ ! -f "$test_id" ] || [ ! -f "$test_ood" ]; then
      echo "[!] [$model_short][train=$train_ds][eval=${eval_ds}-${task}] Missing test jsonl:" >&2
      echo "      $test_id" >&2
      echo "      $test_ood" >&2
      exit 1
    fi
    local ds_test_id="${eval_prefix}_stage1_test_id_${task}"
    local ds_test_ood="${eval_prefix}_stage1_test_ood_${task}"

    # EVAL_SKIP_SCORE=1 이면 생성만 하고 채점은 배치(scripts/rebuild_eval_metrics.sh)로
    # 미룬다. 채점은 단일 스레드 CPU 작업인데 job 안에 있으면 그동안 GPU 가 통째로 논다 —
    # 2026-07-30 실측으로 EXP01 7B ep1 은 생성 2h41m + 채점 2h08m 이라 wall 의 44% 가
    # GPU 유휴였다. 이때는 metrics 가 안 생기므로 skip 기준을 predictions 완성 여부로
    # 바꾼다. vLLM 은 split 을 다 돌고 나서야 jsonl 을 쓰므로 "줄 수 == test 행 수"가
    # 완성의 충분조건이다 (반쯤 쓰인 파일은 줄 수가 모자라 다시 생성된다).
    #
    # ⚠️ 줄 수만으로 판단하면 안 된다. 재측정 대기 중인 구 predictions 는 1024 로 잘렸거나
    # 시드 없이 뽑혔어도 **행 수는 온전**하다 (2026-07-30 실측 37 leaf). 줄 수만 보면 그것들을
    # "완성"으로 오인해 재생성을 건너뛰고, 배치가 무효 predictions 를 채점해 정본 자리에
    # 쓴다. 그래서 이 코드 경로가 직접 남긴 마커(.gen_done)가 있을 때만 건너뛴다.
    if [[ "${EVAL_SKIP_SCORE:-0}" == "1" && -f "$out_dir/.gen_done" ]]; then
      local n_id n_ood
      n_id=$(wc -l < "$out_dir/generated_predictions_id.jsonl" 2>/dev/null || echo 0)
      n_ood=$(wc -l < "$out_dir/generated_predictions_ood.jsonl" 2>/dev/null || echo 0)
      if [ "$n_id" -eq "$(wc -l < "$test_id")" ] && [ "$n_ood" -eq "$(wc -l < "$test_ood")" ]; then
        echo "[=] [$subtag] skip (생성 완료 마커 확인 ${n_id}/${n_ood} · 채점은 배치 위임)" >&2
        continue
      fi
      echo "[!] [$subtag] .gen_done 은 있으나 predictions 가 불완전(${n_id}/${n_ood}) — 재생성한다" >&2
      rm -f "$out_dir/.gen_done"
    fi

    build_infer_cmd "$model_short" "$hub_id" "$ds_test_id" \
      "$test_id" "$template" \
      "$out_rel/generated_predictions_id.jsonl" \
      "$out_rel/predict_results_id.json" "$infer_mnt"
    local infer_id="$INFER_CMD"
    build_infer_cmd "$model_short" "$hub_id" "$ds_test_ood" \
      "$test_ood" "$template" \
      "$out_rel/generated_predictions_ood.jsonl" \
      "$out_rel/predict_results_ood.json" "$infer_mnt"
    local infer_ood="$INFER_CMD"

    local score_step=" && \
        python '$BASE_DIR/scripts/$scorer' score \
          --test-id  '$test_id' \
          --pred-id  '$out_dir/generated_predictions_id.jsonl' \
          --test-ood '$test_ood' \
          --pred-ood '$out_dir/generated_predictions_ood.jsonl' \
          $mode_flag \
          --output   '$out_dir/$metrics_name'"
    # 채점을 미룰 때는 대신 생성 완료 마커를 남긴다. 이 마커가 "이 predictions 는 현재
    # 예산·시드 설정으로 방금 뽑은 것"이라는 유일한 증거이고, 배치 채점의 대상 선정 기준이다.
    if [[ "${EVAL_SKIP_SCORE:-0}" == "1" ]]; then
      score_step=" && printf 'generated_at=%s\\nmax_new_tokens=%s\\nseed=%s\\nmodel=%s\\n' \\
        \"\$(date -u +%FT%TZ)\" '$infer_mnt' \"\${VLLM_SEED:-42}\" '$hub_id' > '$out_dir/.gen_done'"
    fi

    run_logged "$subtag" \
      bash -c "cd '$LF_ROOT' && mkdir -p '$out_rel' && \
        $infer_id && \
        $infer_ood${score_step}"

    # without_open_app sibling: state task 만 (hungarian_eval 만 --exclude-action 지원).
    # EVAL_SKIP_WOA=1 이면 건너뛴다. 이 채점은 GPU 를 한 톨도 안 쓰면서 job 을 30~60 분
    # 붙잡고, 그동안 이 GPU 는 논다. 추론 재실행이 필요 없는 순수 재채점이라
    # scripts/rebuild_woa_metrics.sh 로 나중에 CPU 만 써서 일괄 산출해도 결과가 같다.
    if [[ "$task" == "state" && "${EVAL_SKIP_WOA:-0}" != "1" && "${EVAL_SKIP_SCORE:-0}" != "1" ]]; then
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
              $mode_flag \
              --exclude-action open_app \
              --filtered-test-dir '$BASE_DIR/data/${datadir}' \
              --filtered-pred-dir '$out_dir_woa' \
              --output   '$out_dir_woa/$metrics_name'"
      fi
    fi
  done
}

# 한 (MODEL, TRAIN_DS, VARIANT, EPOCH, HUB_ID, EVAL_DS) 조합 평가 실행.
# - EVAL_DS=MC                : 단일 파일 stage1_test.jsonl → overall only.
# - EVAL_DS=MB                : 단일 파일 stage1.jsonl      → overall only.
# - EVAL_DS=AC_EXP01 / AC_EXP02 : state_pred / action_pred 두 task 독립 채점.
#                                  state → hungarian_metrics, action → action_metrics.
run_variant_epoch_eval_on() {
  local model_short="$1" train_ds="$2" variant="$3" epoch="$4" hub_id="$5" \
        out_rel_base="$6" template="$7" eval_ds="$8"

  # AC_EXP01 / AC_EXP02 / AC_EXP03 / AC_EXP04 / AC_EXP05 / AC_EXP07_v1 / AC_EXP07_v2 는 task 별 독립 채점이라 별도 helper 위임.
  if [[ "$eval_ds" == "AC_EXP01" || "$eval_ds" == "AC_EXP02" || "$eval_ds" == "AC_EXP03" || "$eval_ds" == "AC_EXP04" || "$eval_ds" == "AC_EXP05" || "$eval_ds" == "AC_EXP07_v1" || "$eval_ds" == "AC_EXP07_v2" ]]; then
    run_exp01_eval "$model_short" "$train_ds" "$variant" "$epoch" "$hub_id" \
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

  local test_jsonl
  local ds_test
  if [[ "$eval_ds" == "MB" ]]; then
    test_jsonl="$BASE_DIR/data/${datadir}/stage1.jsonl"
    ds_test="${eval_prefix}_stage1"
  else  # MC (random split)
    test_jsonl="$BASE_DIR/data/${datadir}/stage1_test.jsonl"
    ds_test="${eval_prefix}_stage1_test"
  fi
  if [ ! -f "$test_jsonl" ]; then
    echo "[!] [$model_short][train=$train_ds][eval=$eval_ds] Missing test file: $test_jsonl" >&2
    exit 1
  fi

  # MC/MB 는 state transition (hungarian) 단일 파일 → state 예측 예산 적용.
  build_infer_cmd "$model_short" "$hub_id" "$ds_test" \
    "$test_jsonl" "$template" \
    "$out_rel/generated_predictions.jsonl" \
    "$out_rel/predict_results.json" 12288

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
  if [[ "${EVAL_SKIP_WOA:-0}" != "1" ]] && \
     ! skip_if_done "$tag_woa" "$out_dir_woa/hungarian_metrics.json"; then
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
}

for MODEL_SHORT in "${MODELS[@]}"; do
  BASE_MODEL="${MODEL_ID[$MODEL_SHORT]}"
  TEMPLATE="${MODEL_TEMPLATE[$MODEL_SHORT]}"

  # outputs/ 1-level 디렉토리는 ds_outputs_code 로 정규화 (AC_EXP01_ratio* → AndroidControl_EXP01),
  # 모델 디렉토리에는 ds_model_suffix (AC_EXP01_ratio*=_ratio{37,55,73}) 를 붙인다.
  OUT_DS="$(ds_outputs_code "$TRAIN_DS")"
  EVAL_SFX="$(ds_model_suffix "$TRAIN_DS")"
  # VER(_v1): EXP07 버전 태그. v1/v2 eval 산출물이 같은 공유 부모(AndroidControl_EXP07)
  # 아래에서 겹치지 않도록 모델 eval 디렉토리 이름 끝에 버전을 붙인다 (그 외 DS 는 빈 문자열).
  EVAL_VER="$(ds_version_suffix "$TRAIN_DS")"
  EVAL_DIR_REL="../outputs/${OUT_DS}/eval/${MODEL_SHORT}${EVAL_SFX}${EVAL_VER}/stage1_eval"

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
          HUB_ID=$(resolve_eval_model_path stage1 "$MODEL_SHORT" "$TRAIN_DS" "$MODE" "$EPOCH")
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
