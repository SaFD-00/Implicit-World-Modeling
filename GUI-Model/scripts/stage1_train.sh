#!/usr/bin/env bash
# Stage 1 Fine-tuning (full / lora)
# - --stage1-mode full (default) 또는 lora 로 선택.
# - FORCE_TORCHRUN=1 NPROC_PER_NODE=${NPROC_PER_NODE} (DeepSpeed Z3)
#
# AC_3 (AndroidControl_3): --dataset AC_3 입력 시 _common.sh::parse_args 가
# DATASETS=(AC_3_r37 AC_3_r55 AC_3_r73) 로 펼쳐주므로, 본 스크립트는 ratio 별
# DS 키를 그대로 받아 examples/custom/GUI-Model-AC_3_r{37,55,73}/stage1_{full,lora}/
# {MODEL_SHORT}_world-model.yaml 을 require_yaml 한다 (notebook Cell 8 가 ratio 별
# 디렉토리에 YAML 을 생성). --ac3-ratios r55 처럼 부분 sweep 도 가능.
#
# NPROC_PER_NODE 은 .env 에서 관리 (기본값 2).

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"
parse_args "$@"
export DISABLE_VERSION_CHECK=1
: "${NPROC_PER_NODE:=2}"

SCRIPT_TAG="stage1_train_${STAGE1_MODE}"

for MODEL_SHORT in "${MODELS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    YAML="examples/custom/GUI-Model-${DS}/stage1_${STAGE1_MODE}/${MODEL_SHORT}_world-model.yaml"
    require_yaml "$YAML" "run notebook Cell 9 to generate this YAML"

    run_logged "${SCRIPT_TAG}_${MODEL_SHORT}_${DS}" \
      env FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="$NPROC_PER_NODE" \
      bash -c "cd '$LF_ROOT' && llamafactory-cli train '$YAML'"
  done
done
