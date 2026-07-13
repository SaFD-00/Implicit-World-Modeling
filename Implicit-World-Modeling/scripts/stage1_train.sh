#!/usr/bin/env bash
# Stage 1 Fine-tuning (full / lora)
# - --stage1-mode full (default) 또는 lora 로 선택.
# - FORCE_TORCHRUN=1 NPROC_PER_NODE=${NPROC_PER_NODE} (DeepSpeed Z3)
#
# AC_EXP01 (AndroidControl_EXP01): --dataset AC_EXP01 입력 시 _common.sh::parse_args 가
# DATASETS=(AC_EXP01_ratio37 AC_EXP01_ratio55 AC_EXP01_ratio73) 로 펼쳐주므로, 본 스크립트는
# ratio 별 DS 키를 그대로 받아
# configs/train/IWM-AC_EXP01_ratio{37,55,73}/stage1_{full,lora}/{MODEL_SHORT}_world-model.yaml
# 을 require_yaml 한다 (implicit_world_modeling.gen_configs 가 ratio 별 디렉토리에 YAML 을 생성).
# --exp01-ratios ratio55 처럼 부분 sweep 도 가능.
#
# NPROC_PER_NODE 은 .env 에서 관리 (기본값 2).

# shellcheck source=./_common.sh
source "$(dirname "$0")/_common.sh"
parse_args "$@"
export DISABLE_VERSION_CHECK=1
: "${NPROC_PER_NODE:=2}"

# DeepSpeed CPUAdam JIT 빌드에 필요한 CUDA 라이브러리 경로 (conda env 내 nvidia 패키지)
# LIBRARY_PATH: 빌드 타임 링커, LD_LIBRARY_PATH: 런타임 로더
_NVIDIA_PKGS="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
_CUDA_LIBS="${_NVIDIA_PKGS}/curand/lib:${_NVIDIA_PKGS}/cuda_runtime/lib"
export LIBRARY_PATH="${_CUDA_LIBS}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${_CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

SCRIPT_TAG="stage1_train_${STAGE1_MODE}"

for MODEL_SHORT in "${MODELS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    # YAML 정본은 repo 가 소유한다 (LF/examples/custom 이 아니라 configs/train).
    YAML="$BASE_DIR/configs/train/IWM-${DS}/stage1_${STAGE1_MODE}/${MODEL_SHORT}_world-model.yaml"
    require_model_eligible "$MODEL_SHORT" "${DS_DATADIR[$DS]}"
    require_yaml "$YAML" "python -m implicit_world_modeling.gen_configs --write 로 생성하세요"

    # GPU 트리오(pdbs/grad_accum/deepspeed) + repo-owned dataset_dir/media_dir 를 런타임 주입.
    OVERRIDES="$(resolve_overrides "$MODEL_SHORT" "${DS_DATADIR[$DS]}" "$STAGE1_MODE")"
    echo_resolved "$YAML" "$OVERRIDES"
    maybe_dry_run "$YAML" "$OVERRIDES" && continue

    run_logged "${SCRIPT_TAG}_${MODEL_SHORT}_${DS}" \
      env FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="$NPROC_PER_NODE" \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      bash -c "cd '$LF_ROOT' && llamafactory-cli train '$YAML' $OVERRIDES"
  done
done
