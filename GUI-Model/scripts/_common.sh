#!/usr/bin/env bash
# Shared helpers for GUI-Model stage{1,2}_{train,eval,merge}.sh
# Source from sibling scripts:  source "$(dirname "$0")/_common.sh"
# Requires: bash 4+ (associative array 사용). Linux 기본 bash 는 4+ 이므로 통상 OK.
#           macOS 기본 bash 는 3.2 → `brew install bash` 후 `/opt/homebrew/bin/bash` 권장.

set -euo pipefail

# 이 환경은 일부 deps (typing_extensions, regex, fsspec, peft, trl, deepspeed 등)
# 가 PYTHONUSERBASE 아래에만 설치되어 있으므로 user-site 는 비활성화하지 않는다.
# 다만 /root/.local/workspace/python-packages/bin 의 낡은 accelerate CLI 는
# shebang 이 base env python 을 가리킬 때가 있어 `No module named 'torch'` 를
# 유발한다. conda env (`gui-model`) 가 활성화되어 있다면 해당 env 의
# bin 을 PATH 맨 앞에 고정해 env 소속 CLI ($CONDA_PREFIX/bin/accelerate 등) 가
# 먼저 잡히도록 강제한다.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
else
  echo "[!] conda env 가 활성화되지 않았습니다. 먼저 실행하세요:" >&2
  echo "      conda activate gui-model" >&2
  exit 1
fi

if (( BASH_VERSINFO[0] < 4 )); then
  echo "[!] bash 4+ required (current: $BASH_VERSION)." >&2
  echo "    macOS 기본 /bin/bash 3.2 는 지원하지 않습니다. 'brew install bash' 후 재실행하세요." >&2
  exit 1
fi

# --- paths -------------------------------------------------------------------
# scripts/ 의 부모 디렉토리가 BASE_DIR (notebook Cell 3 의 BASE_DIR 대응)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LF_ROOT="$BASE_DIR/LlamaFactory"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

# --- data symlinks --------------------------------------
# vllm_infer.py 의 media_dir 는 dataset_dir 를 기본으로 사용.
# JSONL 이미지 경로가 "AndroidControl/images/..." 형태이므로
# LF_ROOT/data/ 아래에 심볼릭 링크가 필요함.
# 주의: eval script 에서 vllm_infer.py 호출 시 반드시 --dataset_dir '$LF_ROOT/data'
#       (절대 경로)를 전달해야 한다. 상대 경로("data") 사용 시 HF datasets 캐시가
#       다른 cwd 에서 생성된 stale 경로를 재사용하여 이미지 FileNotFoundError 발생.
for _ds_dir in "$BASE_DIR"/data/*/; do
  _ds_name=$(basename "$_ds_dir")
  _link="$LF_ROOT/data/$_ds_name"
  if [ ! -e "$_link" ]; then
    ln -sfn "$_ds_dir" "$_link"
  fi
done
unset _ds_dir _ds_name _link

# --- eval-only benchmark dataset_info entries (idempotent) --------------------
# MobiBench 는 평가 전용 단일 파일 (ID/OOD split 없음). notebook Cell 13/16 이
# 같은 엔트리를 기록하지만, notebook 을 돌리지 않은 fresh clone 에서도 eval
# 파이프라인이 성립하도록 여기서 보장한다. 이미 존재하면 no-op.
ensure_eval_only_dataset_info() {
  local di="$LF_ROOT/data/dataset_info.json"
  [ -f "$di" ] || return 0
  python3 - "$di" <<'PY'
import json, sys
from collections import OrderedDict
p = sys.argv[1]
with open(p) as f:
    d = json.load(f, object_pairs_hook=OrderedDict)
tags = OrderedDict([
    ("role_tag", "from"), ("content_tag", "value"),
    ("user_tag", "human"), ("assistant_tag", "gpt"),
    ("system_tag", "system"),
])
def entry(stage):
    return OrderedDict([
        ("file_name", f"../../data/MobiBench/gui-model_stage{stage}.jsonl"),
        ("formatting", "sharegpt"),
        ("columns", OrderedDict([("messages","messages"),("images","images")])),
        ("tags", tags),
    ])
changed = False
for stage, anchor in ((1, "GUI-Model-MB_stage1_train"), (2, "GUI-Model-MB_stage2_train")):
    key = f"GUI-Model-MB_stage{stage}"
    if key in d:
        continue
    new_d = OrderedDict()
    inserted = False
    for k, v in d.items():
        if k == anchor and not inserted:
            new_d[key] = entry(stage); inserted = True
        new_d[k] = v
    if not inserted:
        new_d[key] = entry(stage)
    d = new_d
    changed = True
if changed:
    with open(p, 'w') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        f.write('\n')
PY
}
ensure_eval_only_dataset_info

# --- .env (HF_TOKEN 등) -------------------------------------------------------
if [ -f "$BASE_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$BASE_DIR/.env"
  set +a
fi

# --- RTX5090 + DeepSpeed CPU offload: CUDA toolkit 정렬 가드 -----------------
# RTX5090 환경에서는 yaml 이 ds_z3_offload_config.json 으로 swap 되어 있어
# DeepSpeed 가 DeepSpeedCPUAdam → CPUAdamBuilder 를 JIT 컴파일한다. 이 빌드는
# nvcc 와 cu 헤더가 torch 가 빌드된 cu 버전과 정확히 일치해야 하며,
# 불일치 시 학습 시작 직후 CUDAMismatchException 으로 죽는다.
# CUDA_HOME 미설정 + 시스템 PATH 에서 다른 cu 버전 nvcc (예: 13.x) 가 잡히는
# 사고를 막기 위해 RTX5090 일 때만 /usr/local/cuda 를 강제 export 후 검증한다.
# (다른 GPU_TYPE 은 offload 를 안 쓰므로 가드 미적용.)
if [[ "${GPU_TYPE:-}" == "RTX5090" ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "[!] RTX5090: nvcc 가 $CUDA_HOME/bin 에 없습니다." >&2
    echo "    cu12.8 toolkit (nvcc + cuda.h 헤더 + lib64) 를 설치 후 /usr/local/cuda 로 link 하세요." >&2
    exit 1
  fi
  _nvcc_ver="$("$CUDA_HOME/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n1)"
  _torch_cuda="$(python3 -c 'import sys,torch; sys.stdout.write(torch.version.cuda or "")' 2>/dev/null || true)"
  if [[ -n "$_torch_cuda" && "$_nvcc_ver" != "$_torch_cuda" ]]; then
    echo "[!] CUDA mismatch: $CUDA_HOME nvcc=$_nvcc_ver != torch.version.cuda=$_torch_cuda" >&2
    echo "    DeepSpeed CPUAdamBuilder JIT 빌드 시 CUDAMismatchException 발생합니다." >&2
    echo "    /usr/local/cuda 가 cu$_torch_cuda toolkit 을 가리키도록 link 를 갱신하세요." >&2
    exit 1
  fi
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  unset _nvcc_ver _torch_cuda
fi

# --- dataset prefix / HF slug / data dir 매핑 (Cell 3 _DATASET_CONFIG 와 일치) -
# MB 는 평가 전용 벤치마크(학습 파이프라인 미사용). 학습 대상 DS 는 {AC, AC_2, AC_3, MC}.
# MB entry 는 평가 스크립트가 dataset_info 이름/slug 를 조합하는 데 사용.
#
# AC_3 (AndroidControl_3) 은 state_pred / action_pred 두 task 를 비율 혼합한
# 3 종 train (3:7, 5:5, 7:3) 으로 학습한다. ratio 가 학습 산출물의 정체성에
#영향을 주므로 내부적으로 ratio 별 가상 키 (AC_3_r37, AC_3_r55, AC_3_r73) 로
# 펼친다. 사용자 facing CLI 는 --dataset AC_3 (학습/merge) 또는
# --train-dataset AC_3 + --ac3-ratio r55 (eval) 만 받고 expansion 은 내부에서 처리.
declare -A DS_PREFIX=(
  [MB]="GUI-Model-MB"
  [AC]="GUI-Model-AC"
  [AC_2]="GUI-Model-AC_2"
  [AC_3]="GUI-Model-AC_3"
  [AC_3_r37]="GUI-Model-AC_3" [AC_3_r55]="GUI-Model-AC_3" [AC_3_r73]="GUI-Model-AC_3"
  [AC_4]="GUI-Model-AC_4"
  [MC]="GUI-Model-MC"
)
declare -A HF_SLUG=(
  [MB]="mb-"
  [AC]="ac-"
  [AC_2]="ac-2-"
  [AC_3]="ac-3-"
  [AC_3_r37]="ac-3-r37-" [AC_3_r55]="ac-3-r55-" [AC_3_r73]="ac-3-r73-"
  [AC_4]="ac-4-"
  [MC]="mc-"
)
declare -A DS_DATADIR=(
  [MB]="MobiBench"
  [AC]="AndroidControl"
  [AC_2]="AndroidControl_2"
  [AC_3]="AndroidControl_3"
  [AC_3_r37]="AndroidControl_3" [AC_3_r55]="AndroidControl_3" [AC_3_r73]="AndroidControl_3"
  # AC_4 = AC_3 r73 동일 데이터 + Stage1 state-pred diff loss 실험군.
  # train 은 diff-loss 전처리본, test/Stage2 는 AC_3 에서 복사 — 모두 AndroidControl_4/ 아래.
  [AC_4]="AndroidControl_4"
  [MC]="MonkeyCollection"
)

# AC_3 ratio variant 메타: ratio 키 ↔ split_data.py 산출 파일 stem.
# split_data.py 는 train_3_7.jsonl / train_5_5.jsonl / train_7_3.jsonl 을 생성한다.
declare -A AC3_RATIO_FILE=(
  [AC_3_r37]="train_3_7"
  [AC_3_r55]="train_5_5"
  [AC_3_r73]="train_7_3"
)
AC3_ALL_RATIOS=(r37 r55 r73)

# AC_3 ratio variant 인지 검사. usage: if is_ac3_ratio "$DS"; then ...
is_ac3_ratio() {
  case "$1" in
    AC_3_r37|AC_3_r55|AC_3_r73) return 0 ;;
    *) return 1 ;;
  esac
}

# DS 키 → outputs/ 1-level 디렉토리 코드 (notebook Cell 5 의 output_prefix 와 동치).
# AC_3 ratio variant 3 키는 단일 부모 'AC_3' 로 모인다.
ds_outputs_code() {
  case "$1" in
    AC_3_r37|AC_3_r55|AC_3_r73) echo "AC_3" ;;
    *) echo "$1" ;;
  esac
}

# DS 키 → adapters/ + merged/ 의 모델 디렉토리 이름에 붙일 suffix.
# AC_3 ratio variant 만 _r{37,55,73} 을 갖고, 다른 DS 는 빈 문자열.
ds_model_suffix() {
  case "$1" in
    AC_3_r37) echo "_r37" ;;
    AC_3_r55) echo "_r55" ;;
    AC_3_r73) echo "_r73" ;;
    *) echo "" ;;
  esac
}

# DS 키 → eval/ 의 모델 디렉토리 이름에 붙일 suffix.
# AC/AC_2 는 DS 코드 자체를 lower 로, AC_3 ratio 는 ratio 만, MC 는 빈 문자열.
ds_eval_suffix() {
  case "$1" in
    AC) echo "_ac" ;;
    AC_2) echo "_ac_2" ;;
    AC_3_r37) echo "_r37" ;;
    AC_3_r55) echo "_r55" ;;
    AC_3_r73) echo "_r73" ;;
    *) echo "" ;;
  esac
}

# --- 모델 레지스트리 (Cell 3 _MODEL_CONFIG 와 일치) ---------------------------
declare -A MODEL_ID=(
  [qwen2-vl-2b]="Qwen/Qwen2-VL-2B-Instruct"
  [qwen2-vl-7b]="Qwen/Qwen2-VL-7B-Instruct"
  [qwen2.5-vl-3b]="Qwen/Qwen2.5-VL-3B-Instruct"
  [qwen2.5-vl-7b]="Qwen/Qwen2.5-VL-7B-Instruct"
  [qwen3-vl-4b]="Qwen/Qwen3-VL-4B-Instruct"
  [qwen3-vl-8b]="Qwen/Qwen3-VL-8B-Instruct"
  [qwen3.5-4b-base]="Qwen/Qwen3.5-4B-Base"
  [qwen3.5-9b-base]="Qwen/Qwen3.5-9B-Base"
)
declare -A MODEL_TEMPLATE=(
  [qwen2-vl-2b]="qwen2_vl"
  [qwen2-vl-7b]="qwen2_vl"
  [qwen2.5-vl-3b]="qwen2_vl"
  [qwen2.5-vl-7b]="qwen2_vl"
  [qwen3-vl-4b]="qwen3_vl_nothink"
  [qwen3-vl-8b]="qwen3_vl_nothink"
  [qwen3.5-4b-base]="qwen3_5_nothink"
  [qwen3.5-9b-base]="qwen3_5_nothink"
)
# 정렬 순서: Qwen 이전세대 → 최신세대. 세대 내 작은 모델 먼저.
ALL_MODELS=(
  qwen2-vl-2b qwen2-vl-7b
  qwen2.5-vl-3b qwen2.5-vl-7b
  qwen3-vl-4b qwen3-vl-8b
  qwen3.5-4b-base qwen3.5-9b-base
)

# --- CLI 인자 파싱 (학습/merge 스크립트용): --model MODEL --dataset DS --------
# 사용법:
#   bash script.sh --model qwen3-vl-8b --dataset AC
#   bash script.sh --model qwen3-vl-8b --stage1-mode lora
#   bash script.sh --stage1-mode lora           # 전체 모델 LoRA 학습/평가/merge
#   bash script.sh                               # 기본값: 전체 모델 + 전체 학습 DS + full
#
# 학습 대상 DS 는 {AC, AC_2, MC} 만. MobiBench(MB) 는 평가 전용 벤치마크이므로
# --dataset MB 입력은 거절된다. 교차 평가는 stage{1,2}_eval.sh 가 제공하는
# parse_eval_args (--train-dataset / --eval-datasets) 를 사용한다.
parse_args() {
  local model_arg="all"
  local dataset_arg="all"
  local stage1_mode_arg="full"
  local stage2_mode_arg="lora"
  local hf_upload_arg=1
  local stage1_epoch_arg=""
  local epochs_arg="1,2,3"
  local variants_arg=""
  local ac3_ratios_arg=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        if [[ -z "${2:-}" ]]; then echo "Error: --model requires a value." >&2; exit 2; fi
        model_arg="$2"; shift 2 ;;
      --dataset)
        if [[ -z "${2:-}" ]]; then echo "Error: --dataset requires a value." >&2; exit 2; fi
        dataset_arg="$2"; shift 2 ;;
      --stage1-mode)
        if [[ -z "${2:-}" ]]; then echo "Error: --stage1-mode requires a value." >&2; exit 2; fi
        stage1_mode_arg="$2"; shift 2 ;;
      --stage2-mode)
        if [[ -z "${2:-}" ]]; then echo "Error: --stage2-mode requires a value." >&2; exit 2; fi
        stage2_mode_arg="$2"; shift 2 ;;
      --no-hf-upload)
        hf_upload_arg=0; shift ;;
      --stage1-epoch)
        if [[ -z "${2:-}" ]]; then echo "Error: --stage1-epoch requires a value." >&2; exit 2; fi
        stage1_epoch_arg="$2"; shift 2 ;;
      --epochs)
        if [[ -z "${2:-}" ]]; then echo "Error: --epochs requires a value." >&2; exit 2; fi
        epochs_arg="$2"; shift 2 ;;
      --variants)
        if [[ -z "${2:-}" ]]; then echo "Error: --variants requires a value." >&2; exit 2; fi
        variants_arg="$2"; shift 2 ;;
      --ac3-ratios)
        if [[ -z "${2:-}" ]]; then echo "Error: --ac3-ratios requires a value." >&2; exit 2; fi
        ac3_ratios_arg="$2"; shift 2 ;;
      -h|--help)
        cat <<EOF
Usage: $(basename "$0") [--model MODEL] [--dataset DS] [--stage1-mode MODE]
                         [--stage2-mode MODE] [--stage1-epoch N] [--epochs LIST]
                         [--variants LIST] [--ac3-ratios LIST] [--no-hf-upload]

Options:
  --model MODEL        모델 short_name 또는 "all" (기본: all)
  --dataset DS         AC | AC_2 | AC_3 | AC_4 | MC | all (기본: all) — 학습 대상 DS.
                       AC_3 는 ratio mix (3:7, 5:5, 7:3) 3 종을 모두 sweep 하므로
                       --ac3-ratios 로 부분 실행 가능. MB 는 평가 전용이라 사용 불가.
                       AC_4 는 AC_3 r73 동일 데이터 + Stage1 state-pred diff loss 실험군.
                       'all' 은 (AC AC_2 MC) 만 의미하며 AC_3/AC_4 는 명시적으로 선택해야 함.
  --stage1-mode MODE   full | lora (기본: full) — Stage 1 학습 방식.
  --stage2-mode MODE   full | lora (기본: lora) — Stage 2 학습 방식 (Stage 2 전용).
  --no-hf-upload       Hugging Face 업로드를 생략하고 local merge/export 만 수행.
                       merge 스크립트에서만 의미가 있다.
  --stage1-epoch N     Stage 2 world-model variant 가 상류 base 로 삼을 Stage 1 epoch.
                       stage2_{train,merge,eval}.sh 전용.
  --epochs LIST        콤마로 구분된 epoch 정수 리스트 (기본: 1,2,3)
                       stage{1,2}_eval.sh 에서 HF Hub merged repo sweep 대상.
  --variants LIST      콤마로 구분된 변형 목록. stage{1,2}_eval.sh 전용.
                       Stage1: base, full_world_model, lora_world_model
                       Stage2: base, full_base, lora_base, full_world_model, lora_world_model
  --ac3-ratios LIST    콤마로 구분된 AC_3 ratio 목록 (기본: r37,r55,r73).
                       --dataset AC_3 일 때만 의미가 있다.
  -h, --help           이 도움말 표시

Available models:
  ${ALL_MODELS[*]}
EOF
        exit 0
        ;;
      *)
        echo "Error: Unknown argument '$1'. Use --help for usage." >&2
        exit 2
        ;;
    esac
  done

  case "$stage1_mode_arg" in
    full|lora) STAGE1_MODE="$stage1_mode_arg" ;;
    *) echo "Error: --stage1-mode must be full | lora (got '$stage1_mode_arg')." >&2; exit 2 ;;
  esac
  case "$stage2_mode_arg" in
    full|lora) STAGE2_MODE="$stage2_mode_arg" ;;
    *) echo "Error: --stage2-mode must be full | lora (got '$stage2_mode_arg')." >&2; exit 2 ;;
  esac
  HF_UPLOAD="$hf_upload_arg"

  STAGE1_EPOCH=""
  if [[ -n "$stage1_epoch_arg" ]]; then
    if ! [[ "$stage1_epoch_arg" =~ ^[0-9]+$ ]]; then
      echo "Error: --stage1-epoch must be a positive integer (got '$stage1_epoch_arg')." >&2
      exit 2
    fi
    STAGE1_EPOCH="$stage1_epoch_arg"
  fi

  # model_arg → MODELS 배열
  if [[ "$model_arg" == "all" ]]; then
    MODELS=("${ALL_MODELS[@]}")
  elif [[ -n "${MODEL_ID[$model_arg]+x}" ]]; then
    MODELS=("$model_arg")
  else
    echo "Error: Unknown model '$model_arg'." >&2
    echo "Available: ${ALL_MODELS[*]} | all" >&2
    exit 2
  fi

  # AC_3 ratio 선택 파싱 (--ac3-ratios LIST). 기본: 3 ratio 전체.
  local ac3_ratios=()
  if [[ -n "$ac3_ratios_arg" ]]; then
    IFS=',' read -r -a ac3_ratios <<< "$ac3_ratios_arg"
    for _r in "${ac3_ratios[@]}"; do
      case "$_r" in
        r37|r55|r73) ;;
        *) echo "Error: --ac3-ratios item '$_r' invalid (use r37 | r55 | r73)." >&2; exit 2 ;;
      esac
    done
    unset _r
  else
    ac3_ratios=("${AC3_ALL_RATIOS[@]}")
  fi

  # AC_3 → 내부 ratio variant DS 키들로 expand. 다른 DS 는 그대로 전달.
  case "$dataset_arg" in
    AC)   DATASETS=(AC) ;;
    AC_2) DATASETS=(AC_2) ;;
    AC_4) DATASETS=(AC_4) ;;
    MC)   DATASETS=(MC) ;;
    AC_3)
      DATASETS=()
      for _r in "${ac3_ratios[@]}"; do DATASETS+=("AC_3_${_r}"); done
      unset _r
      ;;
    all)  DATASETS=(AC AC_2 MC) ;;
    MB)
      echo "Error: MobiBench (MB) 는 평가 전용 벤치마크입니다. 학습/merge 에는 사용할 수 없습니다." >&2
      echo "       교차 평가는 stage{1,2}_eval.sh --train-dataset {AC|AC_2|AC_3|MC} --eval-datasets AC,AC_2,AC_3,MC,MB 를 사용하세요." >&2
      exit 2
      ;;
    *) echo "Error: Unknown dataset '$dataset_arg'. Use AC | AC_2 | AC_3 | AC_4 | MC | all." >&2; exit 2 ;;
  esac

  IFS=',' read -r -a EPOCHS <<< "$epochs_arg"
  if [[ "${#EPOCHS[@]}" -eq 0 ]]; then
    echo "Error: --epochs 값이 비어있습니다." >&2; exit 2
  fi
  for _e in "${EPOCHS[@]}"; do
    if ! [[ "$_e" =~ ^[0-9]+$ ]]; then
      echo "Error: --epochs 는 콤마로 구분된 정수여야 합니다 (got: '$epochs_arg')." >&2
      exit 2
    fi
  done
  unset _e

  VARIANTS=()
  if [[ -n "$variants_arg" ]]; then
    IFS=',' read -r -a VARIANTS <<< "$variants_arg"
  fi
}

# --- CLI 인자 파싱 (eval 스크립트용): --train-dataset / --eval-datasets --------
# 학습 DS (HF Hub merged repo 식별용) 와 평가 DS (test JSONL 경로) 를 분리한다.
#
# 사용법:
#   bash stage1_eval.sh --model qwen3-vl-8b --train-dataset AC --eval-datasets AC,AC_2,MC,MB
#   bash stage2_eval.sh --model qwen3-vl-8b --train-dataset AC_2 --eval-datasets AC_2,MB \
#        --stage1-mode full --stage1-epoch 3 --stage2-mode lora
#
# 생성 변수:
#   MODELS          ALL_MODELS 또는 단일 모델 배열
#   TRAIN_DATASET   AC | AC_2 | MC  (필수, 단일)
#   EVAL_DATASETS   (AC|AC_2|MC|MB)+ 배열  (기본: 단일값 = TRAIN_DATASET)
#   STAGE1_MODE, STAGE2_MODE, STAGE1_EPOCH, EPOCHS, VARIANTS  (parse_args 와 동일)
parse_eval_args() {
  local model_arg="all"
  local train_arg=""
  local eval_arg=""
  local stage1_mode_arg="full"
  local stage2_mode_arg="lora"
  local stage1_epoch_arg=""
  local epochs_arg="1,2,3"
  local variants_arg=""
  local ac3_ratio_arg=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        if [[ -z "${2:-}" ]]; then echo "Error: --model requires a value." >&2; exit 2; fi
        model_arg="$2"; shift 2 ;;
      --train-dataset)
        if [[ -z "${2:-}" ]]; then echo "Error: --train-dataset requires a value." >&2; exit 2; fi
        train_arg="$2"; shift 2 ;;
      --eval-datasets)
        if [[ -z "${2:-}" ]]; then echo "Error: --eval-datasets requires a value." >&2; exit 2; fi
        eval_arg="$2"; shift 2 ;;
      --stage1-mode)
        if [[ -z "${2:-}" ]]; then echo "Error: --stage1-mode requires a value." >&2; exit 2; fi
        stage1_mode_arg="$2"; shift 2 ;;
      --stage2-mode)
        if [[ -z "${2:-}" ]]; then echo "Error: --stage2-mode requires a value." >&2; exit 2; fi
        stage2_mode_arg="$2"; shift 2 ;;
      --stage1-epoch)
        if [[ -z "${2:-}" ]]; then echo "Error: --stage1-epoch requires a value." >&2; exit 2; fi
        stage1_epoch_arg="$2"; shift 2 ;;
      --epochs)
        if [[ -z "${2:-}" ]]; then echo "Error: --epochs requires a value." >&2; exit 2; fi
        epochs_arg="$2"; shift 2 ;;
      --variants)
        if [[ -z "${2:-}" ]]; then echo "Error: --variants requires a value." >&2; exit 2; fi
        variants_arg="$2"; shift 2 ;;
      --ac3-ratio)
        if [[ -z "${2:-}" ]]; then echo "Error: --ac3-ratio requires a value." >&2; exit 2; fi
        ac3_ratio_arg="$2"; shift 2 ;;
      -h|--help)
        cat <<EOF
Usage: $(basename "$0") --train-dataset {AC|AC_2|AC_3|MC} [--eval-datasets LIST] [--model MODEL]
                         [--stage1-mode MODE] [--stage2-mode MODE] [--stage1-epoch N]
                         [--epochs LIST] [--variants LIST] [--ac3-ratio RATIO]

Options:
  --model MODEL           모델 short_name 또는 "all" (기본: all)
  --train-dataset DS      AC | AC_2 | AC_3 | AC_4 | MC (필수) — HF Hub merged repo 를 해석할 학습 DS.
                          AC_3 는 ratio 하나를 추가로 지정해야 함 (--ac3-ratio).
  --eval-datasets LIST    콤마로 구분된 평가 DS 리스트 (기본: --train-dataset 단일값)
                          허용값: AC, AC_2, AC_3, AC_4, MC, MB (MB 는 단일 파일 overall 채점).
                          AC_3 / AC_4 는 state_pred / action_pred 두 task 를 각각 채점한다.
  --stage1-mode MODE      full | lora (기본: full) — world-model variant 의 상류 Stage1 모드.
  --stage2-mode MODE      full | lora (기본: lora) — Stage 2 merge/eval 전용.
  --stage1-epoch N        Stage 2 world-model variant 의 HF repo 계보 번호.
  --epochs LIST           콤마 구분 정수 리스트 (기본: 1,2,3) — sweep 대상 epoch.
  --variants LIST         콤마 구분 평가 변형 목록.
                          Stage1: base, full_world_model, lora_world_model
                          Stage2: base, full_base, lora_base, full_world_model, lora_world_model
  --ac3-ratio RATIO       AC_3 학습 모델 식별용 단일 ratio (r37 | r55 | r73, 기본 r55).
                          --train-dataset AC_3 일 때만 의미가 있다.
  -h, --help              이 도움말 표시

Available models:
  ${ALL_MODELS[*]}
EOF
        exit 0
        ;;
      *)
        echo "Error: Unknown argument '$1'. Use --help for usage." >&2
        exit 2
        ;;
    esac
  done

  if [[ -z "$train_arg" ]]; then
    echo "Error: --train-dataset 는 필수입니다 (AC | AC_2 | AC_3 | AC_4 | MC)." >&2; exit 2
  fi
  case "$train_arg" in
    AC|AC_2|AC_4|MC) TRAIN_DATASET="$train_arg" ;;
    AC_3)
      # AC_3 는 ratio 별로 학습 가중치가 다르므로 평가 sweep 은 한 번에 한 ratio.
      # 미지정 시 r55 default. TRAIN_DATASET 은 ratio variant 키로 정규화.
      local _r="${ac3_ratio_arg:-r55}"
      case "$_r" in
        r37|r55|r73) ;;
        *) echo "Error: --ac3-ratio must be r37 | r55 | r73 (got '$_r')." >&2; exit 2 ;;
      esac
      TRAIN_DATASET="AC_3_${_r}"
      AC3_RATIO="$_r"
      unset _r ;;
    MB)
      echo "Error: --train-dataset MB 는 허용되지 않습니다 (MobiBench 는 평가 전용)." >&2
      exit 2 ;;
    *) echo "Error: --train-dataset must be AC | AC_2 | AC_3 | AC_4 | MC (got '$train_arg')." >&2; exit 2 ;;
  esac

  # --ac3-ratio 는 AC_3 train 일 때만 유효. 다른 train DS 와 함께 주면 에러.
  if [[ -n "$ac3_ratio_arg" && "$train_arg" != "AC_3" ]]; then
    echo "Error: --ac3-ratio 는 --train-dataset AC_3 와 함께만 사용할 수 있습니다." >&2
    exit 2
  fi

  if [[ -z "$eval_arg" ]]; then
    # AC_3 train 의 eval 기본값은 raw 'AC_3' (test 파일은 ratio 와 무관).
    if [[ "$train_arg" == "AC_3" ]]; then
      EVAL_DATASETS=(AC_3)
    else
      EVAL_DATASETS=("$TRAIN_DATASET")
    fi
  else
    IFS=',' read -r -a EVAL_DATASETS <<< "$eval_arg"
    if [[ "${#EVAL_DATASETS[@]}" -eq 0 ]]; then
      echo "Error: --eval-datasets 값이 비어있습니다." >&2; exit 2
    fi
    for _d in "${EVAL_DATASETS[@]}"; do
      case "$_d" in
        AC|AC_2|AC_3|AC_4|MC|MB) ;;
        *) echo "Error: --eval-datasets item '$_d' invalid (use AC | AC_2 | AC_3 | AC_4 | MC | MB)." >&2; exit 2 ;;
      esac
    done
    unset _d
  fi

  case "$stage1_mode_arg" in
    full|lora) STAGE1_MODE="$stage1_mode_arg" ;;
    *) echo "Error: --stage1-mode must be full | lora (got '$stage1_mode_arg')." >&2; exit 2 ;;
  esac
  case "$stage2_mode_arg" in
    full|lora) STAGE2_MODE="$stage2_mode_arg" ;;
    *) echo "Error: --stage2-mode must be full | lora (got '$stage2_mode_arg')." >&2; exit 2 ;;
  esac

  STAGE1_EPOCH=""
  if [[ -n "$stage1_epoch_arg" ]]; then
    if ! [[ "$stage1_epoch_arg" =~ ^[0-9]+$ ]]; then
      echo "Error: --stage1-epoch must be a positive integer (got '$stage1_epoch_arg')." >&2
      exit 2
    fi
    STAGE1_EPOCH="$stage1_epoch_arg"
  fi

  if [[ "$model_arg" == "all" ]]; then
    MODELS=("${ALL_MODELS[@]}")
  elif [[ -n "${MODEL_ID[$model_arg]+x}" ]]; then
    MODELS=("$model_arg")
  else
    echo "Error: Unknown model '$model_arg'." >&2
    echo "Available: ${ALL_MODELS[*]} | all" >&2
    exit 2
  fi

  IFS=',' read -r -a EPOCHS <<< "$epochs_arg"
  if [[ "${#EPOCHS[@]}" -eq 0 ]]; then
    echo "Error: --epochs 값이 비어있습니다." >&2; exit 2
  fi
  for _e in "${EPOCHS[@]}"; do
    if ! [[ "$_e" =~ ^[0-9]+$ ]]; then
      echo "Error: --epochs 는 콤마로 구분된 정수여야 합니다 (got: '$epochs_arg')." >&2
      exit 2
    fi
  done
  unset _e

  VARIANTS=()
  if [[ -n "$variants_arg" ]]; then
    IFS=',' read -r -a VARIANTS <<< "$variants_arg"
  fi
}

# --- tee 로거 ----------------------------------------------------------------
# usage: run_logged <tag> <cmd...>
# - LOG_DIR/<tag>_<timestamp>.log 로 저장
# - pipefail 로 커맨드 실패 시 스크립트 중단
run_logged() {
  local tag="$1"; shift
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local log="$LOG_DIR/${tag}_${ts}.log"
  echo "[+] [$tag] start  -> log: $log" >&2
  echo "[+] [$tag] cmd:    $*" >&2
  local rc=0
  "$@" 2>&1 | tee "$log" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[!] [$tag] FAILED (exit=$rc)  log: $log" >&2
    exit "$rc"
  fi
  echo "[+] [$tag] done   log: $log" >&2
}

# --- skip-if-exists 가드 -----------------------------------------------------
# usage: if skip_if_done <tag> <marker>; then continue; fi
# marker 파일이 이미 존재하면 stderr 에 skip 메시지를 찍고 0 (success) 을 반환.
# 호출부에서 `continue` / `:` 로 우회하는 패턴으로 사용한다.
skip_if_done() {
  local tag="$1" marker="$2"
  if [ -f "$marker" ]; then
    echo "[=] [$tag] skip (already done): $marker" >&2
    return 0
  fi
  return 1
}

# --- YAML / 디렉토리 가드 ----------------------------------------------------
# usage: require_yaml <절대 또는 LF_ROOT 상대 경로> <노트북 cell 안내>
require_yaml() {
  local yaml="$1"; local hint="${2:-}"
  local abs
  if [[ "$yaml" == /* ]]; then abs="$yaml"; else abs="$LF_ROOT/$yaml"; fi
  if [ ! -f "$abs" ]; then
    echo "[!] Missing YAML: $abs" >&2
    [ -n "$hint" ] && echo "    Hint: $hint" >&2
    exit 1
  fi
}

# --- checkpoint → epoch 매핑 -------------------------------------------------
# HF Trainer 가 저장한 trainer_state.json 의 "epoch" 필드를 int 로 반환.
# 학습 YAML 은 save_strategy=epoch 이므로 정수에 근접하지만 방어적으로 round.
ckpt_epoch_from_dir() {
  local ckpt_dir="$1"
  local state="$ckpt_dir/trainer_state.json"
  if [ ! -f "$state" ]; then
    echo "[!] trainer_state.json not found: $state" >&2
    return 1
  fi
  python - "$state" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
e = s.get("epoch")
if e is None:
    sys.stderr.write(f"[!] 'epoch' missing in {sys.argv[1]}\n")
    sys.exit(1)
print(int(round(float(e))))
PY
}

# --- HF Hub repo id 조립 (단일 실패 지점) ------------------------------------
# Stage 1: SaFD-00/{short}-{slug}world-model-stage1-{mode}-epoch{E}
#   ex: SaFD-00/qwen2.5-vl-7b-ac-world-model-stage1-full-epoch1
hf_repo_id_stage1() {
  local model_short="$1" ds="$2" mode="$3" epoch="$4"
  printf 'SaFD-00/%s-%sworld-model-stage1-%s-epoch%s' \
    "$model_short" "${HF_SLUG[$ds]}" "$mode" "$epoch"
}

# Stage 2 (base variant):
#   SaFD-00/{short}-{slug}base-stage2-{mode2}-epoch{E2}
#   ex: SaFD-00/qwen2.5-vl-7b-ac-base-stage2-full-epoch1
hf_repo_id_stage2_base() {
  local model_short="$1" ds="$2" mode2="$3" epoch2="$4"
  printf 'SaFD-00/%s-%sbase-stage2-%s-epoch%s' \
    "$model_short" "${HF_SLUG[$ds]}" "$mode2" "$epoch2"
}

# Stage 2 (world-model variant — Stage 1 계보 포함):
#   SaFD-00/{short}-{slug}world-model-stage1-{mode1}-epoch{E1}-stage2-{mode2}-epoch{E2}
#   ex: SaFD-00/qwen2.5-vl-7b-ac-world-model-stage1-full-epoch3-stage2-lora-epoch1
hf_repo_id_stage2_world_model() {
  local model_short="$1" ds="$2" mode1="$3" epoch1="$4" mode2="$5" epoch2="$6"
  printf 'SaFD-00/%s-%sworld-model-stage1-%s-epoch%s-stage2-%s-epoch%s' \
    "$model_short" "${HF_SLUG[$ds]}" "$mode1" "$epoch1" "$mode2" "$epoch2"
}

# --- Local merged 디렉토리 경로 ---------------------------------------------
# stage1: merged/{MODEL}{SFX}_stage1_{MODE}_world-model/epoch-{E}
#   variant_key = MODE (full|lora). Stage 1 은 항상 world-model 학습이므로 접미 고정.
#   SFX = ds_model_suffix(ds) — AC_3 ratio variant 만 _r{37,55,73}, 그 외는 "".
#   outputs/ 1-level 디렉토리는 ds_outputs_code(ds) 로 정규화 (AC_3_r* → AC_3).
# stage2: merged/{MODEL}{SFX}_stage2_{variant_key}/epoch-{E}
#   AC_3 ratio variant 가 같은 outputs/AC_3/ 부모를 공유하므로 model 디렉토리에
#   ratio suffix 를 붙여 r37/r55/r73 산출물을 구분한다.
local_merged_epoch_dir() {
  local stage="$1" model_short="$2" ds="$3" variant_key="$4" epoch="$5"
  local out_ds; out_ds="$(ds_outputs_code "$ds")"
  local sfx;    sfx="$(ds_model_suffix "$ds")"
  case "$stage" in
    stage1) printf '%s/outputs/%s/merged/%s%s_stage1_%s_world-model/epoch-%s' \
              "$BASE_DIR" "$out_ds" "$model_short" "$sfx" "$variant_key" "$epoch" ;;
    stage2) printf '%s/outputs/%s/merged/%s%s_stage2_%s/epoch-%s' \
              "$BASE_DIR" "$out_ds" "$model_short" "$sfx" "$variant_key" "$epoch" ;;
    *) echo "[!] local_merged_epoch_dir: unknown stage '$stage'" >&2; return 1 ;;
  esac
}

# --- Variant 유효성 체크 + 기본값 ---------------------------------------------
# Stage 1 변형: base, full_world_model, lora_world_model
STAGE1_ALL_VARIANTS=(base full_world_model lora_world_model)
# Stage 2 변형: base, full_base, lora_base, full_world_model, lora_world_model
STAGE2_ALL_VARIANTS=(base full_base lora_base full_world_model lora_world_model)

# Stage 1 variants 를 지정하지 않았으면 전체를 사용. 잘못된 항목은 error.
resolve_stage1_variants() {
  if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    VARIANTS=("${STAGE1_ALL_VARIANTS[@]}")
    return
  fi
  for v in "${VARIANTS[@]}"; do
    local ok=0
    for allowed in "${STAGE1_ALL_VARIANTS[@]}"; do
      if [[ "$v" == "$allowed" ]]; then ok=1; break; fi
    done
    if (( ok == 0 )); then
      echo "Error: unknown stage1 variant '$v'." >&2
      echo "Allowed: ${STAGE1_ALL_VARIANTS[*]}" >&2
      exit 2
    fi
  done
}

# --- Inference 커맨드 조립 (backend 분기) ------------------------------------
# stage{1,2}_eval.sh 의 base / world-model variant 블록이 generated_predictions.jsonl
# 을 만드는 핵심 커맨드. backend 에 따라 다른 runner 를 호출하지만 호출부의
# dispatch 로직을 중복시키지 않기 위해 여기에 모은다.
#
# usage:
#   build_infer_cmd <model_short> <model_path_or_hub_id> <lf_dataset_name> \
#                   <test_jsonl> <template> <save_rel> <matrix_rel>
#   → INFER_CMD 전역 변수에 커맨드 문자열을 할당.
#   호출부는 `bash -c "cd '$LF_ROOT' && mkdir -p '$OUT_REL' && $INFER_CMD && ..."`
#   형태로 체이닝한다. 경로는 cwd=$LF_ROOT 기준 상대 (save_rel/matrix_rel) 와
#   절대 (test_jsonl) 가 섞여 있으며 기존 스크립트 관행을 그대로 유지.
build_infer_cmd() {
  local model_short="$1" model_path="$2" ds_name="$3" \
        test_jsonl="$4" template="$5" save_rel="$6" matrix_rel="$7"
  local enable_thinking_flag=""
  if [[ "$template" == qwen3_vl* || "$template" == qwen3_5* ]]; then
    enable_thinking_flag="--enable_thinking False"
  fi
  # mm_processor_kwargs: LlamaFactory 학습 후 저장된 preprocessor_config.json 의
  # max_pixels/min_pixels 가 null 로 덮여 transformers smart_resize 가 터지는 것을
  # 런타임 오버라이드로 회피한다.
  # max_pixels 는 학습 YAML 의 image_max_pixels (노트북 Cell 5) 와 통일해 vllm
  # processor 가 추가 다운샘플하지 않도록 한다.
  #
  # 정책: token 예산은 학습 데이터셋 (TRAIN_DATASET) 으로 결정한다 — 학습된 모델은
  # 평가 데이터셋과 무관하게 학습 시 budget 을 그대로 사용해야 mismatch 가 없다.
  #   TRAIN_DATASET=AC_2  → max_tokens=5400  (AC2 학습 모델, 모든 평가 ds 에 동일)
  #   TRAIN_DATASET=AC|MC → max_tokens=2048  (family default)
  # family 별 factor (patch×merge) 에 따라 max_pixels = max_tokens × factor², 그리고
  # min_pixels = 4 × factor² 로 환산:
  #   Qwen2/2.5-VL  (factor 28): 2048→1,605,632 / 5400→4,233,600  | min=3,136
  #   Qwen3-VL/3.5  (factor 32): 2048→2,097,152 / 5400→5,529,600  | min=4,096
  local mm_max_tokens=2048
  if [[ "${TRAIN_DATASET:-}" == "AC_2" ]]; then
    mm_max_tokens=5400
  fi
  local _factor=28 mm_min=3136
  if [[ "$template" == qwen3_vl* || "$template" == qwen3_5* ]]; then
    _factor=32
    mm_min=4096
  fi
  local mm_max=$(( mm_max_tokens * _factor * _factor ))
  INFER_CMD="python scripts/vllm_infer.py \
      --model_name_or_path '$model_path' \
      --dataset '$ds_name' \
      --dataset_dir '$LF_ROOT/data' \
      --template $template \
      --cutoff_len 8192 \
      --image_max_pixels $mm_max \
      $enable_thinking_flag \
      --vllm_config '{\"gpu_memory_utilization\": 0.80, \"mm_processor_kwargs\": {\"min_pixels\": $mm_min, \"max_pixels\": $mm_max}}' \
      --save_name        '$save_rel' \
      --matrix_save_name '$matrix_rel'"
}

resolve_stage2_variants() {
  if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
    VARIANTS=("${STAGE2_ALL_VARIANTS[@]}")
    return
  fi
  for v in "${VARIANTS[@]}"; do
    local ok=0
    for allowed in "${STAGE2_ALL_VARIANTS[@]}"; do
      if [[ "$v" == "$allowed" ]]; then ok=1; break; fi
    done
    if (( ok == 0 )); then
      echo "Error: unknown stage2 variant '$v'." >&2
      echo "Allowed: ${STAGE2_ALL_VARIANTS[*]}" >&2
      exit 2
    fi
  done
}
