#!/usr/bin/env bash
# Shared helpers for Implicit-World-Modeling stage{1,2}_{train,eval,merge}.sh
# Source from sibling scripts:  source "$(dirname "$0")/_common.sh"
# Requires: bash 4+ (associative array 사용). Linux 기본 bash 는 4+ 이므로 통상 OK.
#           macOS 기본 bash 는 3.2 → `brew install bash` 후 `/opt/homebrew/bin/bash` 권장.

set -euo pipefail

# 이 환경은 일부 deps (typing_extensions, regex, fsspec, peft, trl, deepspeed 등)
# 가 PYTHONUSERBASE 아래에만 설치되어 있으므로 user-site 는 비활성화하지 않는다.
# 다만 /root/.local/workspace/python-packages/bin 의 낡은 accelerate CLI 는
# shebang 이 base env python 을 가리킬 때가 있어 `No module named 'torch'` 를
# 유발한다. conda env (`implicit-world-modeling`) 가 활성화되어 있다면 해당 env 의
# bin 을 PATH 맨 앞에 고정해 env 소속 CLI ($CONDA_PREFIX/bin/accelerate 등) 가
# 먼저 잡히도록 강제한다.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
else
  echo "[!] conda env 가 활성화되지 않았습니다. 먼저 실행하세요:" >&2
  echo "      conda activate implicit-world-modeling" >&2
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

# --- dataset_dir (repo-owned) -------------------------------------------------
# LF 는 dataset_dir/dataset_info.json 을 읽고, media_dir 기본값도 dataset_dir 다
# (LF hparams/data_args.py:156). 예전에는 LF_ROOT/data/ 를 dataset_dir 로 쓰면서
# 거기에 심링크를 만들고 dataset_info.json 을 in-place 로 뜯어고쳤다 — LF 는 git 밖
# 서드파티라 그 상태가 재클론 한 번에 증발했다.
# 이제 dataset_dir 은 repo 가 소유한다: configs/lf_dataset/ (dataset_info.json + 상대
# 심링크가 git 에 커밋돼 있다). 학습·평가 스크립트가 이 경로를 절대경로로 넘긴다.
#
# 주의: dataset_dir 은 반드시 절대경로로 전달할 것. 상대경로("data")를 쓰면 HF datasets
#       캐시가 다른 cwd 에서 만든 stale 경로를 재사용해 이미지 FileNotFoundError 가 난다
#       (vllm_infer.py 실측).
LF_DATASET_DIR="$BASE_DIR/configs/lf_dataset"

# data/ 에 새 데이터셋이 생기면 상대 심링크를 자기치유로 추가한다 (git 에 커밋할 것).
for _ds_dir in "$BASE_DIR"/data/*/; do
  [ -d "$_ds_dir" ] || continue
  _ds_name=$(basename "$_ds_dir")
  _link="$LF_DATASET_DIR/$_ds_name"
  if [ ! -e "$_link" ]; then
    ln -sfn "../../data/$_ds_name" "$_link"
  fi
done
unset _ds_dir _ds_name _link

# --- dataset_info 검증 (쓰지 않는다) ------------------------------------------
# 예전에는 이 함수가 LF_ROOT/data/dataset_info.json 을 in-place 로 수정해 MobiBench
# 평가 엔트리를 심었다. 이제 dataset_info.json 은 configs/lf_dataset/ 에 커밋된 정본이며
# 그 안에 MB 엔트리도 정적으로 들어있다 — 런타임에 파일을 뜯어고치지 않는다.
# 여기서는 정본이 온전한지 확인만 하고, 아니면 무엇을 해야 하는지 말하고 죽는다.
verify_dataset_info() {
  local di="$LF_DATASET_DIR/dataset_info.json"
  if [ ! -f "$di" ]; then
    echo "[!] dataset_info 정본이 없습니다: $di" >&2
    echo "    이 파일은 git 에 커밋돼 있어야 합니다 (LF 안이 아니라 repo 안이 정본)." >&2
    exit 1
  fi
  local missing
  missing=$(python3 - "$di" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
need = ["IWM-MB_stage1", "IWM-MB_stage2"]
print(" ".join(k for k in need if k not in d))
PY
)
  if [ -n "$missing" ]; then
    echo "[!] dataset_info 정본에 평가 전용 엔트리가 없습니다: $missing" >&2
    echo "    파일: $LF_DATASET_DIR/dataset_info.json" >&2
    echo "    (예전처럼 런타임에 심지 않습니다 — 정본에 넣고 커밋하세요.)" >&2
    exit 1
  fi
}
verify_dataset_info

# --- .env (HF_TOKEN 등) -------------------------------------------------------
# `set -a; source .env` 는 .env 값으로 프로세스 환경을 **덮어쓴다**. 그러면
# `GPU_TYPE=A100 NPROC_PER_NODE=4 bash scripts/stage1_train.sh ...` 처럼 호출 시점에
# GPU 조합을 지정할 수 없다 (.env 의 값이 이긴다). GPU 매트릭스를 쓰려면 프로세스 환경이
# 이겨야 하므로, .env 를 읽기 전에 값을 붙잡아 두었다가 되돌린다.
_PRE_ENV_GPU_TYPE="${GPU_TYPE:-}"
_PRE_ENV_NPROC="${NPROC_PER_NODE:-}"
if [ -f "$BASE_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$BASE_DIR/.env"
  set +a
fi
[[ -n "$_PRE_ENV_GPU_TYPE" ]] && export GPU_TYPE="$_PRE_ENV_GPU_TYPE"
[[ -n "$_PRE_ENV_NPROC" ]] && export NPROC_PER_NODE="$_PRE_ENV_NPROC"
unset _PRE_ENV_GPU_TYPE _PRE_ENV_NPROC
# Per-invocation NPROC override (레거시 — 이제 NPROC_PER_NODE 를 직접 넘겨도 이긴다)
if [[ -n "${NPROC_PER_NODE_OVERRIDE:-}" ]]; then
  NPROC_PER_NODE="$NPROC_PER_NODE_OVERRIDE"
fi

# --- DeepSpeed CPU offload: CUDA toolkit 정렬 가드 ---------------------------
# ZeRO-3 CPU offload 는 DeepSpeedCPUAdam → CPUAdamBuilder 를 JIT 컴파일한다. 이 빌드는
# nvcc 와 cu 헤더가 torch 가 빌드된 cu 버전과 정확히 일치해야 하며, 불일치 시 학습 시작
# 직후 CUDAMismatchException 으로 죽는다. CUDA_HOME 미설정 상태에서 시스템 PATH 의 다른
# cu 버전 nvcc (예: 13.x) 가 잡히는 사고를 막는다.
#
# 예전에는 이 가드를 GPU_TYPE == RTX5090 일 때만 적용했다 — "다른 GPU 는 offload 를 안 쓴다"
# 는 전제였는데, 그 전제가 이제 틀렸다. scripts/gpu_policy.py 는 GPU 종류와 무관하게 항상
# offload 를 반환하므로(as-trained 74/74 YAML 이 그러하고, A100/H100 에서 offload 를 빼면
# EXP05 7B full FT 가 확정 OOM) A100/H100 도 CPUAdam JIT 경로를 탄다. 따라서 가드도 항상 건다.
# LF_CUDA_GUARD_SKIP=1 로 우회 가능 (CPUAdam 이 미리 빌드돼 있는 이미지 등).
if [[ "${LF_CUDA_GUARD_SKIP:-0}" != "1" ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "[!] nvcc 가 $CUDA_HOME/bin 에 없습니다 (DeepSpeed CPU offload 는 CPUAdam JIT 빌드를 요구합니다)." >&2
    echo "    torch 와 같은 cu 버전의 toolkit (nvcc + cuda.h 헤더 + lib64) 을 설치 후 /usr/local/cuda 로 link 하세요." >&2
    echo "    CPUAdam 이 이미 빌드된 이미지라면 LF_CUDA_GUARD_SKIP=1 로 우회할 수 있습니다." >&2
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
# MB 는 평가 전용 벤치마크(학습 파이프라인 미사용). 학습 대상 DS 는 {AC_EXP01, AC_EXP02, AC_EXP03, AC_EXP04, AC_EXP05, MC}.
# MB entry 는 평가 스크립트가 dataset_info 이름/slug 를 조합하는 데 사용.
#
# AC_EXP01 (AndroidControl_EXP01) 은 state_pred / action_pred 두 task 를 비율
# 혼합한 3 종 train (3:7, 5:5, 7:3) 으로 학습한다. ratio 가 학습 산출물의
# 정체성에 영향을 주므로 내부적으로 ratio 별 가상 키 (AC_EXP01_ratio37,
# AC_EXP01_ratio55, AC_EXP01_ratio73) 로 펼친다. 사용자 facing CLI 는
# --dataset AC_EXP01 (학습/merge) 또는 --train-dataset AC_EXP01 + --exp01-ratio
# ratio55 (eval) 만 받고 expansion 은 내부에서 처리.
#
# AC_EXP02 (AndroidControl_EXP02) 는 AC_EXP01 ratio73 동일 데이터 + Stage1
# state-pred diff loss 실험군.
#
# AC_EXP03 (AndroidControl_EXP03) 는 AC_EXP01 ratio73 멤버십을 좌표(point) 표현으로
# 미러한 실험군 (index→x,y; scripts/mirror_experiment.py --experiment exp03). stage1 train 은 ratio73 단일,
# test/Stage2 는 EXP01 멤버십 미러 — 모두 AndroidControl_EXP03/ 아래.
#
# 원본 AndroidControl/ 디렉토리는 EXP01/EXP02/EXP03 의 source jsonl + 이미지 자산으로만
# 사용된다 (학습/평가 entry 아님). DS_DATADIR 에는 등재하지 않는다.
declare -A DS_PREFIX=(
  [MB]="IWM-MB"
  [AC_EXP01]="IWM-AC_EXP01"
  [AC_EXP01_ratio37]="IWM-AC_EXP01"
  [AC_EXP01_ratio55]="IWM-AC_EXP01"
  [AC_EXP01_ratio73]="IWM-AC_EXP01"
  [AC_EXP02]="IWM-AC_EXP02"
  [AC_EXP03]="IWM-AC_EXP03"
  [AC_EXP04]="IWM-AC_EXP04"
  [AC_EXP05]="IWM-AC_EXP05"
  [MC]="IWM-MC"
)
declare -A HF_SLUG=(
  [MB]="mb-"
  [AC_EXP01]="ac-exp01-"
  [AC_EXP01_ratio37]="ac-exp01-ratio37-"
  [AC_EXP01_ratio55]="ac-exp01-ratio55-"
  [AC_EXP01_ratio73]="ac-exp01-ratio73-"
  [AC_EXP02]="ac-exp02-"
  [AC_EXP03]="ac-exp03-"
  [AC_EXP04]="ac-exp04-"
  [AC_EXP05]="ac-exp05-"
  [MC]="mc-"
)
declare -A DS_DATADIR=(
  [MB]="MobiBench"
  [AC_EXP01]="AndroidControl_EXP01"
  [AC_EXP01_ratio37]="AndroidControl_EXP01"
  [AC_EXP01_ratio55]="AndroidControl_EXP01"
  [AC_EXP01_ratio73]="AndroidControl_EXP01"
  # AC_EXP02 = AC_EXP01 ratio73 동일 데이터 + Stage1 state-pred diff loss 실험군.
  # train 은 diff-loss 전처리본, test/Stage2 는 AC_EXP01 에서 복사 — 모두 AndroidControl_EXP02/ 아래.
  [AC_EXP02]="AndroidControl_EXP02"
  # AC_EXP03 = AC_EXP01 ratio73 멤버십을 좌표(point) 표현으로 미러 (scripts/mirror_experiment.py --experiment exp03).
  # stage1 train + test 6종 + Stage2 train/test 3종 모두 AndroidControl_EXP03/ 아래.
  [AC_EXP03]="AndroidControl_EXP03"
  # AC_EXP04 = AC_EXP03 와 동일 멤버십·좌표 표현 + stage1 프롬프트 업그레이드 (scripts/mirror_experiment.py --experiment exp04).
  # stage1 train + test 6종만 AndroidControl_EXP04/ 아래 (Stage 2 보류 — 데이터/등록 키 없음).
  [AC_EXP04]="AndroidControl_EXP04"
  # AC_EXP05 = AC_EXP01 ratio73 멤버십의 절대 픽셀(840×1876, budget 1605632) 미러 — scripts/mirror_experiment.py --experiment exp05.
  # Qwen2.5-VL 전용. stage1 7종만, Stage 2 보류.
  [AC_EXP05]="AndroidControl_EXP05"
  [MC]="MonkeyCollection"
)

# AC_EXP01 ratio variant 메타: ratio 키 ↔ split_data.py 산출 파일 stem.
# split_data.py 는 train_3_7.jsonl / train_5_5.jsonl / train_7_3.jsonl 을 생성한다.
declare -A EXP01_RATIO_FILE=(
  [AC_EXP01_ratio37]="train_3_7"
  [AC_EXP01_ratio55]="train_5_5"
  [AC_EXP01_ratio73]="train_7_3"
)
EXP01_ALL_RATIOS=(ratio37 ratio55 ratio73)

# AC_EXP01 ratio variant 인지 검사. usage: if is_ac_exp01_ratio "$DS"; then ...
is_ac_exp01_ratio() {
  case "$1" in
    AC_EXP01_ratio37|AC_EXP01_ratio55|AC_EXP01_ratio73) return 0 ;;
    *) return 1 ;;
  esac
}

# DS 키 → outputs/ 1-level 디렉토리 코드 (notebook Cell 5 의 output_prefix 와 동치).
# AC_EXP01 ratio variant 3 키는 단일 부모 'AndroidControl_EXP01' 로 모인다.
# AC_EXP02 는 단일 폴더 'AndroidControl_EXP02'.
ds_outputs_code() {
  case "$1" in
    AC_EXP01_ratio37|AC_EXP01_ratio55|AC_EXP01_ratio73|AC_EXP01) echo "AndroidControl_EXP01" ;;
    AC_EXP02) echo "AndroidControl_EXP02" ;;
    AC_EXP03) echo "AndroidControl_EXP03" ;;
    AC_EXP04) echo "AndroidControl_EXP04" ;;
    AC_EXP05) echo "AndroidControl_EXP05" ;;
    *) echo "$1" ;;
  esac
}

# DS 키 → adapters/ + merged/ 의 모델 디렉토리 이름에 붙일 suffix.
# AC_EXP01 ratio variant 만 _ratio{37,55,73} 을 갖고, 다른 DS 는 빈 문자열.
ds_model_suffix() {
  case "$1" in
    AC_EXP01_ratio37) echo "_ratio37" ;;
    AC_EXP01_ratio55) echo "_ratio55" ;;
    AC_EXP01_ratio73) echo "_ratio73" ;;
    *) echo "" ;;
  esac
}

# DS 키 → eval/ 의 모델 디렉토리 이름에 붙일 suffix.
# AC_EXP01 ratio 는 ratio 만, 그 외는 빈 문자열.
ds_eval_suffix() {
  case "$1" in
    AC_EXP01_ratio37) echo "_ratio37" ;;
    AC_EXP01_ratio55) echo "_ratio55" ;;
    AC_EXP01_ratio73) echo "_ratio73" ;;
    *) echo "" ;;
  esac
}

# --- 모델 레지스트리 (Cell 5 _MODEL_CONFIG 와 일치) ---------------------------
declare -A MODEL_ID=(
  [qwen3-vl-8b]="Qwen/Qwen3-VL-8B-Instruct"
  [qwen3-vl-4b]="Qwen/Qwen3-VL-4B-Instruct"
  [qwen2.5-vl-7b]="Qwen/Qwen2.5-VL-7B-Instruct"
  [qwen2.5-vl-3b]="Qwen/Qwen2.5-VL-3B-Instruct"
)
declare -A MODEL_TEMPLATE=(
  [qwen3-vl-8b]="qwen3_vl_nothink"
  [qwen3-vl-4b]="qwen3_vl_nothink"
  [qwen2.5-vl-7b]="qwen2_vl"
  [qwen2.5-vl-3b]="qwen2_vl"
)
# 등록 모델은 7-9B tier (qwen3-vl-8b, qwen2.5-vl-7b) + 3-4B tier (qwen3-vl-4b, qwen2.5-vl-3b).
ALL_MODELS=(
  qwen3-vl-8b qwen3-vl-4b qwen2.5-vl-7b qwen2.5-vl-3b
)

# --- CLI 인자 파싱 (학습/merge 스크립트용): --model MODEL --dataset DS --------
# 사용법:
#   bash script.sh --model qwen3-vl-8b --dataset AC_EXP01
#   bash script.sh --model qwen3-vl-8b --dataset AC_EXP02 --stage1-mode lora
#   bash script.sh --model qwen3-vl-8b --dataset MC
#
# 학습 대상 DS 는 {AC_EXP01, AC_EXP02, AC_EXP03, AC_EXP04, AC_EXP05, MC}. 각 DS 는 명시적으로 선택해야 하며,
# `--dataset all` 같은 일괄 sweep 모드는 지원하지 않는다.
# MobiBench(MB) 는 평가 전용 벤치마크이므로 --dataset MB 입력은 거절된다.
# 교차 평가는 stage{1,2}_eval.sh 가 제공하는 parse_eval_args
# (--train-dataset / --eval-datasets) 를 사용한다.
parse_args() {
  local model_arg="all"
  local dataset_arg=""
  local stage1_mode_arg="full"
  local stage2_mode_arg="lora"
  local hf_upload_arg=1
  local stage1_epoch_arg=""
  local epochs_arg="1,2,3"
  local variants_arg=""
  local exp01_ratios_arg=""
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
      --exp01-ratios)
        if [[ -z "${2:-}" ]]; then echo "Error: --exp01-ratios requires a value." >&2; exit 2; fi
        exp01_ratios_arg="$2"; shift 2 ;;
      -h|--help)
        cat <<EOF
Usage: $(basename "$0") [--model MODEL] [--dataset DS] [--stage1-mode MODE]
                         [--stage2-mode MODE] [--stage1-epoch N] [--epochs LIST]
                         [--variants LIST] [--exp01-ratios LIST] [--no-hf-upload]

Options:
  --model MODEL        모델 short_name 또는 "all" (기본: all)
  --dataset DS         AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP04 | AC_EXP05 | MC (필수) — 학습 대상 DS.
                       AC_EXP01 은 ratio mix (3:7, 5:5, 7:3) 3 종을 모두 sweep 하므로
                       --exp01-ratios 로 부분 실행 가능. MB 는 평가 전용이라 사용 불가.
                       AC_EXP02 는 AC_EXP01 ratio73 동일 데이터 + Stage1 state-pred
                       diff loss 실험군. AC_EXP03 은 AC_EXP01 ratio73 멤버십을 좌표(point)
                       표현으로 미러한 실험군 (index→x,y).
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
  --exp01-ratios LIST  콤마로 구분된 AC_EXP01 ratio 목록 (기본: ratio37,ratio55,ratio73).
                       --dataset AC_EXP01 일 때만 의미가 있다.
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

  # AC_EXP01 ratio 선택 파싱 (--exp01-ratios LIST). 기본: 3 ratio 전체.
  local exp01_ratios=()
  if [[ -n "$exp01_ratios_arg" ]]; then
    IFS=',' read -r -a exp01_ratios <<< "$exp01_ratios_arg"
    for _r in "${exp01_ratios[@]}"; do
      case "$_r" in
        ratio37|ratio55|ratio73) ;;
        *) echo "Error: --exp01-ratios item '$_r' invalid (use ratio37 | ratio55 | ratio73)." >&2; exit 2 ;;
      esac
    done
    unset _r
  else
    exp01_ratios=("${EXP01_ALL_RATIOS[@]}")
  fi

  # AC_EXP01 → 내부 ratio variant DS 키들로 expand. 다른 DS 는 그대로 전달.
  case "$dataset_arg" in
    AC_EXP02) DATASETS=(AC_EXP02) ;;
    AC_EXP03) DATASETS=(AC_EXP03) ;;
    AC_EXP04) DATASETS=(AC_EXP04) ;;
    AC_EXP05) DATASETS=(AC_EXP05) ;;
    MC)       DATASETS=(MC) ;;
    AC_EXP01)
      DATASETS=()
      for _r in "${exp01_ratios[@]}"; do DATASETS+=("AC_EXP01_${_r}"); done
      unset _r
      ;;
    "")
      echo "Error: --dataset 는 필수입니다 (AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP04 | AC_EXP05 | MC)." >&2; exit 2 ;;
    MB)
      echo "Error: MobiBench (MB) 는 평가 전용 벤치마크입니다. 학습/merge 에는 사용할 수 없습니다." >&2
      echo "       교차 평가는 stage{1,2}_eval.sh --train-dataset {AC_EXP01|AC_EXP02|MC} --eval-datasets AC_EXP01,AC_EXP02,MC,MB 를 사용하세요." >&2
      exit 2
      ;;
    *) echo "Error: Unknown dataset '$dataset_arg'. Use AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP04 | AC_EXP05 | MC." >&2; exit 2 ;;
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
#   bash stage1_eval.sh --model qwen3-vl-8b --train-dataset AC_EXP01 --exp01-ratio ratio55 --eval-datasets AC_EXP01,MC,MB
#   bash stage2_eval.sh --model qwen3-vl-8b --train-dataset AC_EXP02 --eval-datasets AC_EXP02,MB \
#        --stage1-mode full --stage1-epoch 3 --stage2-mode lora
#
# 생성 변수:
#   MODELS          ALL_MODELS 또는 단일 모델 배열
#   TRAIN_DATASET   AC_EXP01 | AC_EXP02 | MC  (필수, 단일)
#   EVAL_DATASETS   (AC_EXP01|AC_EXP02|MC|MB)+ 배열  (기본: 단일값 = TRAIN_DATASET)
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
  local exp01_ratio_arg=""
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
      --exp01-ratio)
        if [[ -z "${2:-}" ]]; then echo "Error: --exp01-ratio requires a value." >&2; exit 2; fi
        exp01_ratio_arg="$2"; shift 2 ;;
      -h|--help)
        cat <<EOF
Usage: $(basename "$0") --train-dataset {AC_EXP01|AC_EXP02|MC} [--eval-datasets LIST] [--model MODEL]
                         [--stage1-mode MODE] [--stage2-mode MODE] [--stage1-epoch N]
                         [--epochs LIST] [--variants LIST] [--exp01-ratio RATIO]

Options:
  --model MODEL           모델 short_name 또는 "all" (기본: all)
  --train-dataset DS      AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP04 | AC_EXP05 | MC (필수) — HF Hub merged repo 를
                          해석할 학습 DS. AC_EXP01 은 ratio 하나를 추가로 지정해야 함 (--exp01-ratio).
  --eval-datasets LIST    콤마로 구분된 평가 DS 리스트 (기본: --train-dataset 단일값)
                          허용값: AC_EXP01, AC_EXP02, AC_EXP03, AC_EXP04, AC_EXP05, MC, MB (MB 는 단일 파일 overall 채점).
                          AC_EXP01 / AC_EXP02 는 state_pred / action_pred 두 task 를 각각 채점한다.
  --stage1-mode MODE      full | lora (기본: full) — world-model variant 의 상류 Stage1 모드.
  --stage2-mode MODE      full | lora (기본: lora) — Stage 2 merge/eval 전용.
  --stage1-epoch N        Stage 2 world-model variant 의 HF repo 계보 번호.
  --epochs LIST           콤마 구분 정수 리스트 (기본: 1,2,3) — sweep 대상 epoch.
  --variants LIST         콤마 구분 평가 변형 목록.
                          Stage1: base, full_world_model, lora_world_model
                          Stage2: base, full_base, lora_base, full_world_model, lora_world_model
  --exp01-ratio RATIO     AC_EXP01 학습 모델 식별용 단일 ratio (ratio37 | ratio55 | ratio73, 기본 ratio55).
                          --train-dataset AC_EXP01 일 때만 의미가 있다.
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
    echo "Error: --train-dataset 는 필수입니다 (AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP04 | AC_EXP05 | MC)." >&2; exit 2
  fi
  case "$train_arg" in
    AC_EXP02|AC_EXP03|AC_EXP04|AC_EXP05|MC) TRAIN_DATASET="$train_arg" ;;
    AC_EXP01)
      # AC_EXP01 은 ratio 별로 학습 가중치가 다르므로 평가 sweep 은 한 번에 한 ratio.
      # 미지정 시 ratio55 default. TRAIN_DATASET 은 ratio variant 키로 정규화.
      local _r="${exp01_ratio_arg:-ratio55}"
      case "$_r" in
        ratio37|ratio55|ratio73) ;;
        *) echo "Error: --exp01-ratio must be ratio37 | ratio55 | ratio73 (got '$_r')." >&2; exit 2 ;;
      esac
      TRAIN_DATASET="AC_EXP01_${_r}"
      EXP01_RATIO="$_r"
      unset _r ;;
    MB)
      echo "Error: --train-dataset MB 는 허용되지 않습니다 (MobiBench 는 평가 전용)." >&2
      exit 2 ;;
    *) echo "Error: --train-dataset must be AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP04 | AC_EXP05 | MC (got '$train_arg')." >&2; exit 2 ;;
  esac

  # --exp01-ratio 는 AC_EXP01 train 일 때만 유효. 다른 train DS 와 함께 주면 에러.
  if [[ -n "$exp01_ratio_arg" && "$train_arg" != "AC_EXP01" ]]; then
    echo "Error: --exp01-ratio 는 --train-dataset AC_EXP01 와 함께만 사용할 수 있습니다." >&2
    exit 2
  fi

  if [[ -z "$eval_arg" ]]; then
    # AC_EXP01 train 의 eval 기본값은 raw 'AC_EXP01' (test 파일은 ratio 와 무관).
    if [[ "$train_arg" == "AC_EXP01" ]]; then
      EVAL_DATASETS=(AC_EXP01)
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
        AC_EXP01|AC_EXP02|AC_EXP03|AC_EXP04|AC_EXP05|MC|MB) ;;
        *) echo "Error: --eval-datasets item '$_d' invalid (use AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP04 | AC_EXP05 | MC | MB)." >&2; exit 2 ;;
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

# --- llamafactory-cli 런타임 override ----------------------------------------
# 커밋된 학습 YAML(configs/train/)은 GPU-불변 baseline (RTX5090 × 2 프로필:
# pdbs=1, ga=32) 이다. 실제 GPU 조합에 맞는 값은 실행 시점에 주입한다 —
# LF 가 `llamafactory-cli train cfg.yaml key=value` 를 OmegaConf merge 로 지원한다
# (LF hparams/parser.py:69-83). 그래야 GPU_TYPE × NPROC 조합마다 YAML 을 재생성하지
# 않아도 되고, GLOBAL_BATCH=64 가 모든 조합에서 유지된다.
#
# 주입하는 것:
#   per_device_train_batch_size / gradient_accumulation_steps / deepspeed  ← gpu_policy.py (SSoT)
#   dataset_dir / media_dir                                                ← repo-owned 절대경로
#
# ★ dataset_dir 은 반드시 넘겨야 한다. 커밋 YAML 의 `dataset: IWM-AC_*_stage1_train` 키는
#   configs/lf_dataset/dataset_info.json 에만 있고, LF 기본값(LF_ROOT/data)의 upstream
#   dataset_info.json 에는 없다. 안 넘기면 fresh clone 에서 학습이 시작조차 못 한다.
#
# ★ cwd 는 LF_ROOT 를 유지한다. 커밋 YAML 의 `output_dir: ../outputs/...` 가 cwd 상대경로라
#   cwd 를 BASE_DIR 로 옮기면 프로젝트 바깥을 가리킨다 (stage2 의 resolve_stage1_base 도 동일).
#   우리 목적은 "LF 에 쓰지 않는 것"이지 cwd 변경이 아니다 — 그래서 override 는 전부 절대경로로 넘긴다.

# 모델 short name → gpu_policy 의 size_class
model_size_class() {
  case "$1" in
    qwen3-vl-8b|qwen2.5-vl-7b) echo "7-9B" ;;
    qwen3-vl-4b|qwen2.5-vl-3b) echo "3-4B" ;;
    *) echo "[!] 알 수 없는 모델의 size_class: $1" >&2; exit 1 ;;
  esac
}

# resolve_overrides <model_short> <dataset_dir_name> <mode:full|lora>
# stdout: llamafactory-cli 에 append 할 override 인자들 (공백 구분 한 줄)
resolve_overrides() {
  local model_short="$1" ds_name="$2" mode="$3"
  local size_class gpu_over
  size_class="$(model_size_class "$model_short")"

  # gpu_policy 의 stdout 은 override 한 줄, warnings 는 stderr 로 나온다.
  if ! gpu_over="$(python3 "$BASE_DIR/scripts/gpu_policy.py" \
      --gpu-type "${GPU_TYPE:-RTX5090}" --nproc "${NPROC_PER_NODE:-2}" \
      --size-class "$size_class" --ds "$ds_name" --mode "$mode" --format cli)"; then
    echo "[!] gpu_policy 가 이 조합을 거부했습니다:" >&2
    echo "    GPU_TYPE=${GPU_TYPE:-RTX5090} NPROC_PER_NODE=${NPROC_PER_NODE:-2}" \
         "size_class=$size_class ds=$ds_name mode=$mode" >&2
    exit 1
  fi

  # global batch 불변식 재확인 (gpu_policy 가 이미 강제하지만, 주입 직전에 한 번 더).
  local pdbs ga total
  pdbs="$(sed -nE 's/.*per_device_train_batch_size=([0-9]+).*/\1/p' <<<"$gpu_over")"
  ga="$(sed -nE 's/.*gradient_accumulation_steps=([0-9]+).*/\1/p' <<<"$gpu_over")"
  total=$(( pdbs * ga * ${NPROC_PER_NODE:-2} ))
  if (( total != 64 )); then
    echo "[!] global batch 가 64 가 아닙니다: ${pdbs} × ${ga} × ${NPROC_PER_NODE:-2} = ${total}" >&2
    echo "    scripts/gpu_policy.py 와 .env 의 GPU_TYPE/NPROC_PER_NODE 를 확인하세요." >&2
    exit 1
  fi

  # gpu_policy 는 deepspeed 를 BASE_DIR 상대경로로 준다. cwd 는 LF_ROOT 이므로 절대경로로 바꾼다.
  local ds_path
  ds_path="$(sed -nE 's/.*deepspeed=([^ ]+).*/\1/p' <<<"$gpu_over")"
  [[ "$ds_path" == /* ]] || ds_path="$BASE_DIR/$ds_path"
  if [ ! -f "$ds_path" ]; then
    echo "[!] deepspeed config 가 없습니다: $ds_path" >&2
    exit 1
  fi
  gpu_over="$(sed -E "s|deepspeed=[^ ]+|deepspeed=$ds_path|" <<<"$gpu_over")"

  local smoke=""
  if [[ "${SMOKE:-0}" == "1" ]]; then
    smoke=" max_samples=8 max_steps=1 save_strategy=no report_to=none"
  fi

  printf '%s dataset_dir=%s media_dir=%s%s' \
    "$gpu_over" "$LF_DATASET_DIR" "$BASE_DIR/data" "$smoke"
}

# 해석된 실행 커맨드를 로그에 남긴다 — 커밋 YAML ≠ 실제 실행값인 override 설계에서
# 무엇이 실제로 쓰였는지 알 수 있는 유일한 장치다.
echo_resolved() {
  local yaml="$1" overrides="$2"
  echo "[cfg] YAML      : $yaml" >&2
  echo "[cfg] overrides : $overrides" >&2
  echo "[cfg] GPU       : ${GPU_TYPE:-RTX5090} × ${NPROC_PER_NODE:-2}" >&2
}

# DRY_RUN=1 이면 실행하지 않고 최종 커맨드만 출력하고 종료한다.
maybe_dry_run() {
  local yaml="$1" overrides="$2"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] cd '$LF_ROOT' && llamafactory-cli train '$yaml' $overrides"
    return 0
  fi
  return 1
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
#   ex: SaFD-00/qwen3-vl-8b-ac-exp01-ratio37-world-model-stage1-lora-epoch1
hf_repo_id_stage1() {
  local model_short="$1" ds="$2" mode="$3" epoch="$4"
  printf 'SaFD-00/%s-%sworld-model-stage1-%s-epoch%s' \
    "$model_short" "${HF_SLUG[$ds]}" "$mode" "$epoch"
}

# Stage 2 (base variant):
#   SaFD-00/{short}-{slug}base-stage2-{mode2}-epoch{E2}
#   ex: SaFD-00/qwen3-vl-8b-ac-exp01-ratio37-base-stage2-lora-epoch1
hf_repo_id_stage2_base() {
  local model_short="$1" ds="$2" mode2="$3" epoch2="$4"
  printf 'SaFD-00/%s-%sbase-stage2-%s-epoch%s' \
    "$model_short" "${HF_SLUG[$ds]}" "$mode2" "$epoch2"
}

# Stage 2 (world-model variant — Stage 1 계보 포함):
#   SaFD-00/{short}-{slug}world-model-stage1-{mode1}-epoch{E1}-stage2-{mode2}-epoch{E2}
#   ex: SaFD-00/qwen3-vl-8b-ac-exp01-ratio37-world-model-stage1-lora-epoch3-stage2-lora-epoch1
hf_repo_id_stage2_world_model() {
  local model_short="$1" ds="$2" mode1="$3" epoch1="$4" mode2="$5" epoch2="$6"
  printf 'SaFD-00/%s-%sworld-model-stage1-%s-epoch%s-stage2-%s-epoch%s' \
    "$model_short" "${HF_SLUG[$ds]}" "$mode1" "$epoch1" "$mode2" "$epoch2"
}

# --- Local merged 디렉토리 경로 ---------------------------------------------
# stage1: merged/{MODEL}{SFX}_stage1_{MODE}_world-model/epoch-{E}
#   variant_key = MODE (full|lora). Stage 1 은 항상 world-model 학습이므로 접미 고정.
#   SFX = ds_model_suffix(ds) — AC_EXP01 ratio variant 만 _ratio{37,55,73}, 그 외는 "".
#   outputs/ 1-level 디렉토리는 ds_outputs_code(ds) 로 정규화 (AC_EXP01_ratio* → AndroidControl_EXP01).
# stage2: merged/{MODEL}{SFX}_stage2_{variant_key}/epoch-{E}
#   AC_EXP01 ratio variant 가 같은 outputs/AndroidControl_EXP01/ 부모를 공유하므로
#   model 디렉토리에 ratio suffix 를 붙여 ratio37/ratio55/ratio73 산출물을 구분한다.
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

# --- Eval 모델 경로 해석 (local 우선 → HF Hub fallback) ----------------------
# stage{1,2}_eval.sh 가 vllm_infer 에 넘길 model_path 를 결정한다.
# local merged dir (local_merged_epoch_dir) 가 존재하면 그 절대경로를, 없으면
# 기존 HF Hub repo id 를 반환한다. HF push 없이 local merge 만 수행한 워크플로우
# (stage1_merge.sh --no-hf-upload) 에서도 eval 이 동작하게 한다.
# 사용:
#   resolve_eval_model_path stage1       <model_short> <ds> <mode>           <epoch>
#   resolve_eval_model_path stage2_base  <model_short> <ds> <mode2>          <epoch2>
#   resolve_eval_model_path stage2_world <model_short> <ds> <mode1> <epoch1> <mode2> <epoch2>
# Stage 2 world-model epoch-0 (stage2 미학습 = stage1 merged 동일) 은 호출부에서
# stage1 helper 로 위임 (기존 분기 그대로).
resolve_eval_model_path() {
  local kind="$1"; shift
  local local_dir="" hub_id=""
  case "$kind" in
    stage1)
      local _ms="$1" _ds="$2" _mode="$3" _ep="$4"
      local_dir="$(local_merged_epoch_dir stage1 "$_ms" "$_ds" "$_mode" "$_ep")"
      hub_id="$(hf_repo_id_stage1 "$_ms" "$_ds" "$_mode" "$_ep")"
      ;;
    stage2_base)
      local _ms="$1" _ds="$2" _m2="$3" _ep2="$4"
      local_dir="$(local_merged_epoch_dir stage2 "$_ms" "$_ds" "${_m2}_base" "$_ep2")"
      hub_id="$(hf_repo_id_stage2_base "$_ms" "$_ds" "$_m2" "$_ep2")"
      ;;
    stage2_world)
      local _ms="$1" _ds="$2" _m1="$3" _ep1="$4" _m2="$5" _ep2="$6"
      local_dir="$(local_merged_epoch_dir stage2 "$_ms" "$_ds" "${_m2}_world-model_from_${_m1}-ep${_ep1}" "$_ep2")"
      hub_id="$(hf_repo_id_stage2_world_model "$_ms" "$_ds" "$_m1" "$_ep1" "$_m2" "$_ep2")"
      ;;
    *)
      echo "[!] resolve_eval_model_path: unknown kind '$kind'" >&2; return 1 ;;
  esac
  if [ -d "$local_dir" ]; then
    echo "[=] resolve_eval_model_path[$kind]: local hit → $local_dir" >&2
    printf '%s' "$local_dir"
  else
    echo "[=] resolve_eval_model_path[$kind]: local miss → HF fallback $hub_id" >&2
    printf '%s' "$hub_id"
  fi
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
  # 현재 모든 학습 DS 는 family default 인 max_tokens=2048 을 사용한다.
  # family 별 factor (patch×merge) 에 따라 max_pixels = max_tokens × factor², 그리고
  # min_pixels = 4 × factor² 로 환산:
  #   Qwen2/2.5-VL  (factor 28): 2048 → 1,605,632  | min=3,136
  #   Qwen3-VL/3.5  (factor 32): 2048 → 2,097,152  | min=4,096
  local mm_max_tokens=2048
  local _factor=28 mm_min=3136
  if [[ "$template" == qwen3_vl* || "$template" == qwen3_5* ]]; then
    _factor=32
    mm_min=4096
  fi
  local mm_max=$(( mm_max_tokens * _factor * _factor ))
  # AC_EXP03/EXP04/EXP05 좌표(point) 표현은 평가 입력도 ~2.5x 길다 → cutoff_len 을 학습과 통일해
  # 상향(잘림 0), 그 외는 8192. vLLM max_model_len = cutoff + max_new_tokens 증가 →
  # KV cache 메모리↑/throughput↓ (필요 시 VLLM_GPU_MEM_UTIL 로 보정).
  local infer_cutoff=8192
  if [[ "$ds_name" == IWM-AC_EXP03* || "$ds_name" == IWM-AC_EXP04* || "$ds_name" == IWM-AC_EXP05* ]]; then
    infer_cutoff=24576
  fi
  INFER_CMD="python scripts/vllm_infer.py \
      --model_name_or_path '$model_path' \
      --dataset '$ds_name' \
      --dataset_dir '$LF_DATASET_DIR' \
      --template $template \
      --cutoff_len $infer_cutoff \
      --image_max_pixels $mm_max \
      $enable_thinking_flag \
      --vllm_config '{\"gpu_memory_utilization\": ${VLLM_GPU_MEM_UTIL:-0.80}, \"mm_processor_kwargs\": {\"min_pixels\": $mm_min, \"max_pixels\": $mm_max}}' \
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
