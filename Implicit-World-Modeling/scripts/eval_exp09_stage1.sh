#!/usr/bin/env bash
# AC_EXP09 stage1 eval 러너 — 도메인별 LoRA vs 레퍼런스 2종, 고정 eval 샘플.
#
# 15 조건(base, general-full@{1,2.01,3} 3개, general-inverse-mix@{1,2.01,3} 3개,
# {domain}-lora@{0.25..5} 8개) × 4 split = 60 leaf.
# leaf 하나 = "추론 → 채점" 이고 leaf 마다 marker 가 독립이라 중단·재개가 안전하다
# (run_exp08_stage1_sweep.sh / run_exp08_stage2_sweep.sh 와 같은 관례).
# max_new_tokens=12288 짜리 60 leaf 는 길어서 **반드시 중간에 끊긴다**.
#
# 도메인 (--domain 필수)
# -----------------------
# EXP09 는 도메인마다 독립된 학습 데이터·eval 세트·어댑터를 가진다 (Base/General-Full/
# General-Inverse-Mix 조차 도메인별 eval 세트로 따로 평가한다 — 같은 모델이지만 예측
# 결과가 다르다).
#   time_mgmt: split = id-seen/id-unseen/ood-chegal/ood-digibites
#   media:     split = id-seen/id-unseen/ood-readera/ood-bsbportal
#
# 정적 레퍼런스 두 계보 (EXP08 stage1, action-only 제외 — 유일하게 재현 가능한 "다른
# baseline"은 inverse-mix 뿐이다, lora_world_model 은 어댑터 자체가 없어 재현 불가)
# ------------------------------------------------------------------------------
# general-full/general-inverse-mix 는 각각 그 계보의 epoch 1/2.01/3 지점(2 는 어느
# 계보에도 정확히 없어 가장 가까운 2.01 을 쓴다 — EXP08 sweep 스크립트와 같은 선택)을
# 고정 레퍼런스로 삼는다. 로컬 병합본이 있으면 그걸 쓰고(이 머신 한정 지름길), 없으면
# HF 저장소로 폴백한다 — 병합본이 없는 환경(예: Megazone)에서도 그대로 돌아가야 한다.
# HF repo id 유도 규칙은 scripts/_common.sh::hf_repo_id_stage1 그대로다.
#
# 왜 stage1_eval.sh / _common.sh 를 쓰지 않는가
# ---------------------------------------------
# 그 경로는 평가 대상이 항상 **merge 된 디렉토리**라고 가정한다 (resolve_eval_model_path).
# EXP09 는 LoRA 어댑터를 merge 없이 그대로 평가한다 — vllm_infer.py 가
# `--adapter_name_or_path` 를 네이티브로 받아 vLLM 의 LoRARequest 로 넘기기 때문이다
# (vllm_infer.py:49 · 81-84 · 116 · 141-144). 또 `_common.sh::build_infer_cmd` 의
# cutoff_len 상향은 `IWM-AC_EXP08*` 데이터셋 이름 prefix allowlist 로 걸려 있어
# (_common.sh:1385-1387) EXP09 이름은 조용히 8192 로 떨어진다. 그래서 _common.sh 를
# source 하지 않고 필요한 것만 아래에서 재현한다 — run_exp09_stage1.sh 와 같은 판단.
#
# max_lora_rank (이게 없으면 LoRA 조건 8 개가 전부 죽는다)
# --------------------------------------------------------
# vllm_infer.py 의 engine_args (108-117) 에는 `max_lora_rank` 가 없고 vLLM 기본값은
# **16** 이다 (vllm/config/lora.py:33). EXP09 어댑터는 rank 64 라
# (configs/train/IWM-AC_EXP09/stage1_lora/qwen2.5-vl-3b_world-model_{domain}.yaml 의
# `lora_rank: 64`, 저장된 adapter_config.json 도 `r=64`) 기본값으로는 어댑터 로드가
# 거절된다. 유일한 주입 경로가 `--vllm_config` 의 engine_args.update (121-122) 라
# 거기에 넣는다. 허용값은 (8,16,32,64,128,256,320,512) — 64 는 그 안에 있다 (lora.py:106).
#
# vllm_config 를 통째로 갈아끼우면 안 되는 이유
# ----------------------------------------------
# 같은 update() 가 dict 를 **대체**하므로 `mm_processor_kwargs` 를 빠뜨리면
# _common.sh:1360-1362 이 막아 두었던 것 — merged 디렉토리의 preprocessor_config.json 이
# max_pixels/min_pixels 를 null 로 덮어 transformers smart_resize 가 터지는 문제 — 가
# 되살아난다. general-full 조건이 정확히 그 merged 디렉토리다. 그래서 네 키를 모두 싣고
# max_lora_rank 만 어댑터 조건에서 추가한다.
# (--image_max_pixels/--image_min_pixels 는 LlamaFactory 쪽 _regularize_images
#  경로라(vllm_infer.py:162-164) mm_processor_kwargs 와 별개로 둘 다 필요하다.)
#
# 채점 플래그
# -----------
# `--match-mode pos` 와 `--xml-schema cerebra` 는 **둘 다 필수**다. 기본값은
# index/android 이고, 틀린 기본값으로 채점해도 **에러가 나지 않은 채 지표만 무너진다**.
# EXP09 데이터는 EXP08 과 같은 state 소스에서 그대로 떠온 것이라(절대 픽셀 좌표 ·
# data-bbox 를 쓰고 index 속성이 없는 Cerebra XML) EXP08 의 판정을 그대로 상속한다.
# 정본은 _common.sh:268-297 (ds_is_pixel_xy / ds_score_mode_flag / ds_xml_schema_flag).
# score 한 번이 state_diff_metrics.json 을 같은 디렉토리에 **함께** 낸다
# (_hungarian_eval.py:1028 이 _state_diff_eval 을 직접 import). _state_diff_eval.py 를
# 따로 부르지 않는다.
#
# 사용법
#   bash scripts/eval_exp09_stage1.sh --domain time_mgmt --dry-run
#   bash scripts/eval_exp09_stage1.sh --domain media --conditions base,general-full@3  # 오늘 가능한 것만
#   bash scripts/eval_exp09_stage1.sh --domain time_mgmt --splits id-seen --dry-run
#   CUDA_DEVICE=1 bash scripts/eval_exp09_stage1.sh --domain media
#
# 환경변수
#   CUDA_DEVICE        기본 1   (GPU0 은 타인의 vLLM 서버)
#   CONDA_ENV          기본 /opt/miniconda3/envs/implicit-world-modeling
#   VLLM_GPU_MEM_UTIL  기본 0.80 — GPU 를 남과 나눠 쓸 때 낮춘다
#   VLLM_SEED          기본 42
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LF_ROOT="$BASE_DIR/LlamaFactory"
LF_DATASET_DIR="$BASE_DIR/configs/lf_dataset"
DATA_DIR="$BASE_DIR/data/AndroidControl_EXP09"

BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

# 정적 레퍼런스 epoch 지점 (두 계보 공통 — 위 헤더 주석 참조).
REF_EPOCHS=(1 2.01 3)
general_full_local()        { echo "$BASE_DIR/outputs/AndroidControl_EXP08/merged/qwen2.5-vl-3b_stage1_full_world-model/epoch-$1"; }
general_full_hf()           { echo "SaFD-00/qwen2.5-vl-3b-ac-exp08-world-model-stage1-full-epoch$1"; }
general_inverse_mix_local() { echo "$BASE_DIR/outputs/AndroidControl_EXP08/merged/qwen2.5-vl-3b_stage1_full_world-model-inverse-mix/epoch-$1"; }
general_inverse_mix_hf()    { echo "SaFD-00/qwen2.5-vl-3b-ac-exp08-world-model-inverse-mix-stage1-full-epoch$1"; }

CONDA_ENV="${CONDA_ENV:-/opt/miniconda3/envs/implicit-world-modeling}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.80}"
VLLM_SEED="${VLLM_SEED:-42}"

TEMPLATE="qwen2_vl"
# state 예측 = 전체 UI XML. 기본값 2048 로는 약 69% 가 잘려 Hungarian F1 이 무효가 된다
# (_common.sh:1350-1355 실측). cutoff_len 은 위 allowlist 이슈 때문에 명시해야 한다.
MAX_NEW_TOKENS=12288
CUTOFF_LEN=24576
# Qwen2.5-VL (factor 28) · max_tokens 2048 → 2048×28² = 1,605,632 / min = 4×28² = 3,136.
MM_MAX=1605632
MM_MIN=3136
LORA_RANK=64

# epoch 체크포인트 지점 — run_exp09_stage1.sh 의 저장 스케줄과 반드시 일치해야 한다.
LORA_EPOCHS=(0.25 0.5 0.75 1 2 3 4 5)

DOMAIN=""
DRY_RUN=0
CONDITIONS=()
SPLITS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)     DOMAIN="$2"; shift 2 ;;
    --conditions) IFS=',' read -r -a CONDITIONS <<< "$2"; shift 2 ;;
    --splits)     IFS=',' read -r -a SPLITS     <<< "$2"; shift 2 ;;
    --conda-env)  CONDA_ENV="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    sed -n '1,66p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ── 도메인별 OOD 슬러그 (build_exp09_data.py 의 DOMAINS 테이블과 같은 슬러그) ────
case "$DOMAIN" in
  time_mgmt) OOD_SLUG_1="chegal";   OOD_SLUG_2="digibites" ;;
  media)     OOD_SLUG_1="readera";  OOD_SLUG_2="bsbportal" ;;
  "") echo "[!] --domain 이 필요하다 (time_mgmt | media)" >&2; exit 2 ;;
  *) echo "[!] 알 수 없는 --domain: $DOMAIN (허용: time_mgmt, media)" >&2; exit 2 ;;
esac

EVAL_ROOT="$BASE_DIR/outputs/AndroidControl_EXP09/eval/qwen2.5-vl-3b/stage1_eval/$DOMAIN"
ADAPTER_DIR="$BASE_DIR/outputs/AndroidControl_EXP09/adapters/qwen2.5-vl-3b_stage1_lora_world-model_$DOMAIN"
CKPT_MAP="$ADAPTER_DIR/epoch_checkpoint_map.json"
LORA_COND_PREFIX="${DOMAIN}-lora"

ALL_CONDITIONS=(base)
for e in "${REF_EPOCHS[@]}"; do ALL_CONDITIONS+=("general-full@${e}"); done
for e in "${REF_EPOCHS[@]}"; do ALL_CONDITIONS+=("general-inverse-mix@${e}"); done
for e in "${LORA_EPOCHS[@]}"; do ALL_CONDITIONS+=("${LORA_COND_PREFIX}@${e}"); done
ALL_SPLITS=(id-seen id-unseen "ood-$OOD_SLUG_1" "ood-$OOD_SLUG_2")

[[ ${#CONDITIONS[@]} -eq 0 ]] && CONDITIONS=("${ALL_CONDITIONS[@]}")
[[ ${#SPLITS[@]}     -eq 0 ]] && SPLITS=("${ALL_SPLITS[@]}")

in_list() { local n="$1"; shift; local x; for x in "$@"; do [[ "$x" == "$n" ]] && return 0; done; return 1; }
for c in "${CONDITIONS[@]}"; do
  in_list "$c" "${ALL_CONDITIONS[@]}" || { echo "[!] 알 수 없는 condition: $c" >&2
    echo "    가능: ${ALL_CONDITIONS[*]}" >&2; exit 2; }
done
for s in "${SPLITS[@]}"; do
  in_list "$s" "${ALL_SPLITS[@]}" || { echo "[!] 알 수 없는 split: $s" >&2
    echo "    가능: ${ALL_SPLITS[*]}" >&2; exit 2; }
done

# split → (dataset 키, jsonl 경로). N 은 파일에서 직접 센다 — 상수로 박으면 데이터가
# 바뀌었을 때 "완료된 predictions" 판정이 조용히 틀린다.
split_ds_key()  { echo "IWM-AC_EXP09_stage1_eval_${1//-/_}_${DOMAIN}"; }
split_jsonl()   { echo "$DATA_DIR/stage1_eval_${1//-/_}_${DOMAIN}.jsonl"; }

# ── LoRA 체크포인트 해석 ──────────────────────────────────────────────────────
# run_exp09_stage1.sh --select-checkpoints 가 쓰는 map 을 읽는다. 값은 basename 이라
# ADAPTER_DIR 과 join 해야 한다. map 이 없으면 = 학습이 아직 안 끝난 것.
# checkpoint-N 을 추측하지 않는다. **LoRA 조건이 실제로 요청됐을 때만** 이 경로를
# 탄다 — base/general-full 만 돌리는 오늘 같은 상황에서 map 부재로 죽으면 안 된다.
resolve_lora_ckpt() {
  local epoch="$1"
  python3 - "$CKPT_MAP" "$ADAPTER_DIR" "$epoch" <<'PY'
import json, os, sys
map_path, adapter_dir, epoch = sys.argv[1:4]
with open(map_path) as f:
    m = json.load(f)["map"]
if epoch not in m:
    sys.exit(f"[!] epoch {epoch} 가 map 에 없다 (있는 키: {', '.join(sorted(m))})")
d = os.path.join(adapter_dir, m[epoch]["checkpoint"])
if not os.path.isdir(d):
    sys.exit(f"[!] map 이 가리키는 체크포인트 디렉토리가 없다: {d}")
print(d)
PY
}

NEED_LORA=0 HAS_NON_LORA=0
for c in "${CONDITIONS[@]}"; do
  if [[ "$c" == "${LORA_COND_PREFIX}@"* ]]; then NEED_LORA=1; else HAS_NON_LORA=1; fi
done
if (( NEED_LORA )) && [[ ! -f "$CKPT_MAP" ]]; then
  echo "[!] epoch_checkpoint_map.json 이 없다: $CKPT_MAP" >&2
  echo "    = stage1 학습이 아직 끝나지 않았다는 뜻이다. 먼저:" >&2
  echo "      bash scripts/run_exp09_stage1.sh --domain $DOMAIN" >&2
  echo "      bash scripts/run_exp09_stage1.sh --domain $DOMAIN --select-checkpoints" >&2
  echo "    지금 돌릴 수 있는 것만 먼저 돌리려면: --conditions base,general-full" >&2
  # 요청에 비-LoRA 조건이 섞여 있으면 그것들은 오늘 돌릴 수 있다 — 여기서 멈추지 않고
  # LoRA 조건만 resolve_condition 에서 leaf 단위로 실패시킨다. 전부 LoRA 면 할 일이
  # 없으므로 즉시 멈춘다. dry-run 은 계획 전체를 보여주는 게 목적이라 언제나 계속한다.
  if (( ! HAS_NON_LORA && ! DRY_RUN )); then exit 1; fi
  (( DRY_RUN )) && echo "[dry-run][!] LoRA 경로는 <UNRESOLVED> 로 출력한다." >&2
fi

# condition → (모델 경로, 어댑터 경로, leaf 디렉토리 prefix)
COND_MODEL="" COND_ADAPTER="" COND_SUBDIR=""
resolve_condition() {
  local cond="$1"
  COND_ADAPTER=""
  case "$cond" in
    base) COND_MODEL="$BASE_MODEL"; COND_SUBDIR="base" ;;
    general-full@*)
      local epoch="${cond#general-full@}" local_dir
      local_dir="$(general_full_local "$epoch")"
      if [[ -d "$local_dir" ]]; then COND_MODEL="$local_dir"; else COND_MODEL="$(general_full_hf "$epoch")"; fi
      COND_SUBDIR="general-full/epoch-$epoch"
      ;;
    general-inverse-mix@*)
      local epoch="${cond#general-inverse-mix@}" local_dir
      local_dir="$(general_inverse_mix_local "$epoch")"
      if [[ -d "$local_dir" ]]; then COND_MODEL="$local_dir"; else COND_MODEL="$(general_inverse_mix_hf "$epoch")"; fi
      COND_SUBDIR="general-inverse-mix/epoch-$epoch"
      ;;
    "${LORA_COND_PREFIX}@"*)
      local epoch="${cond#${LORA_COND_PREFIX}@}"
      COND_MODEL="$BASE_MODEL"
      COND_SUBDIR="${LORA_COND_PREFIX}/epoch-$epoch"
      if [[ -f "$CKPT_MAP" ]]; then
        COND_ADAPTER="$(resolve_lora_ckpt "$epoch")" || return 1
      elif (( DRY_RUN )); then
        # 계획 출력 전용 placeholder. 실제 런에서는 절대 여기로 새면 안 된다 —
        # 존재하지 않는 경로로 vLLM 을 띄우게 된다.
        COND_ADAPTER="<UNRESOLVED>"
      else
        echo "[!] map 이 없어 '$cond' 의 체크포인트를 해석할 수 없다: $CKPT_MAP" >&2
        return 1
      fi
      ;;
  esac
  return 0
}

# ── env (run_exp09_stage1.sh 가 이미 푼 문제를 그대로 재현) ───────────────────
if (( ! DRY_RUN )); then
  [[ -d "$CONDA_ENV" ]] || { echo "[!] conda env 가 없다: $CONDA_ENV" >&2; exit 1; }
  export PATH="$CONDA_ENV/bin:$PATH"
  # LlamaFactory editable install 의 .pth 가 없는 경로를 가리켜 PYTHONPATH 없이는
  # ModuleNotFoundError 로 죽는다.
  export PYTHONPATH="$LF_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  export DISABLE_VERSION_CHECK=1
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

LOG_DIR="$BASE_DIR/logs/eval_exp09_stage1_${DOMAIN}_$(date +%Y%m%d_%H%M%S)"
(( DRY_RUN )) || mkdir -p "$LOG_DIR"

TOTAL=$(( ${#CONDITIONS[@]} * ${#SPLITS[@]} ))
echo "[eval09] 도메인 $DOMAIN · 조건 ${#CONDITIONS[@]} × split ${#SPLITS[@]} = leaf $TOTAL 개"
echo "[eval09] 출력 $EVAL_ROOT"
(( DRY_RUN )) || echo "[eval09] 로그 $LOG_DIR · GPU CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"

DONE=0 SKIPPED=0 FAILED=0 IDX=0
FAILED_LEAVES=()

for cond in "${CONDITIONS[@]}"; do
  if ! resolve_condition "$cond"; then
    echo "[eval09][!] condition 해석 실패: $cond — 건너뛴다" >&2
    FAILED=$((FAILED + ${#SPLITS[@]})); FAILED_LEAVES+=("$cond/* (${#SPLITS[@]} leaf)")
    continue
  fi

  # 모델 경로 존재 확인 — 절대경로(로컬 병합본)만 검사한다. HF hub id 는
  # "org/repo" 형태라 "/" 로 시작하지 않으므로(base 의 "Qwen/..." 도, general-full/
  # general-inverse-mix 가 HF 로 폴백했을 때의 "SaFD-00/..." 도) 로컬 검사 대상이 아니다.
  if [[ "$COND_MODEL" == /* && ! -d "$COND_MODEL" ]]; then
    echo "[eval09][!] 모델 디렉토리가 없다: $COND_MODEL — condition '$cond' 건너뛴다" >&2
    FAILED=$((FAILED + ${#SPLITS[@]})); FAILED_LEAVES+=("$cond/* (${#SPLITS[@]} leaf)")
    continue
  fi

  for split in "${SPLITS[@]}"; do
    IDX=$((IDX+1))
    leaf_dir="$EVAL_ROOT/$COND_SUBDIR/$split"
    pred_file="$leaf_dir/generated_predictions.jsonl"
    metrics="$leaf_dir/hungarian_metrics.json"
    sd_metrics="$leaf_dir/state_diff_metrics.json"
    test_jsonl="$(split_jsonl "$split")"
    ds_key="$(split_ds_key "$split")"
    tag="${cond//[@\/]/_}_${split}"

    if [[ ! -f "$test_jsonl" ]]; then
      echo "[eval09][!] eval jsonl 이 없다: $test_jsonl" >&2
      FAILED=$((FAILED+1)); FAILED_LEAVES+=("$tag"); continue
    fi
    expect_n="$(wc -l < "$test_jsonl")"

    # marker = 최종 metrics JSON 자체 (_common.sh::skip_if_done 와 같은 관례).
    # 단 EXP08 state leaf 와 달리 여기는 산출물이 **두 개**다. _hungarian_eval.py 는
    # 정본(hungarian)을 **먼저 쓰고 그 다음에** state-diff 를 계산하므로(:989-1005 의
    # 명시적 설계), 그 사이에 끊기거나 state-diff 만 실패한 leaf 는 정본만 남는다.
    # 정본 하나만 marker 로 삼으면 그 leaf 가 sibling 없이 영구 skip 되고, 요약
    # 스크립트는 두 파일을 다 읽는다. 둘 다 있어야 완료로 본다 — 재채점 비용은
    # 아래 predictions-level skip 덕분에 추론 없이 끝난다.
    if [[ -f "$metrics" && -f "$sd_metrics" ]]; then
      echo "[=] [$IDX/$TOTAL][$tag] skip (already done): $metrics"
      SKIPPED=$((SKIPPED+1)); continue
    fi

    # vllm_infer.py:227-229 는 전부 모아 마지막에 한 번에 쓴다 → predictions 는
    # "완전하거나 아예 없거나" 다. 줄 수가 맞으면 채점만 실패한 leaf 이므로
    # 12288 토큰 재추론 없이 채점부터 다시 한다.
    need_infer=1
    if [[ -f "$pred_file" ]] && [[ "$(wc -l < "$pred_file")" == "$expect_n" ]]; then
      need_infer=0
      echo "[=] [$IDX/$TOTAL][$tag] predictions 완비 ($expect_n 행) — 추론 건너뛰고 채점만"
    fi

    # engine_args.update() 가 dict 를 대체하므로 네 키를 모두 싣는다 (상단 주석 참조).
    vcfg="{\"gpu_memory_utilization\": $VLLM_GPU_MEM_UTIL, \"tensor_parallel_size\": 1"
    vcfg="$vcfg, \"mm_processor_kwargs\": {\"min_pixels\": $MM_MIN, \"max_pixels\": $MM_MAX}"
    [[ -n "$COND_ADAPTER" ]] && vcfg="$vcfg, \"max_lora_rank\": $LORA_RANK"
    vcfg="$vcfg}"

    # --matrix_save_name 은 일부러 넘기지 않는다: BLEU/ROUGE 는 이 실험에서 쓰지 않고
    # (지표는 전부 Hungarian/state-diff), 그 경로가 load_dataset 으로 HF 캐시만 불린다.
    infer_cmd=(python scripts/vllm_infer.py
      --model_name_or_path "$COND_MODEL"
      --dataset "$ds_key"
      --dataset_dir "$LF_DATASET_DIR"
      --template "$TEMPLATE"
      --cutoff_len "$CUTOFF_LEN"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --image_max_pixels "$MM_MAX"
      --image_min_pixels "$MM_MIN"
      --seed "$VLLM_SEED"
      --vllm_config "$vcfg"
      --save_name "$pred_file")
    # temperature/top_p/top_k 는 건드리지 않는다 — vllm_infer.py 기본값
    # (0.95/0.7/50) + seed 고정이 EXP08 정본이고, greedy 로 바꾸면 EXP08 수치와
    # 나란히 놓을 수 없다 (_common.sh:1396-1402).
    [[ -n "$COND_ADAPTER" ]] && infer_cmd+=(--adapter_name_or_path "$COND_ADAPTER")

    score_cmd=(python "$BASE_DIR/scripts/_hungarian_eval.py" score
      --test "$test_jsonl"
      --pred "$pred_file"
      --match-mode pos
      --xml-schema cerebra
      --output "$metrics")

    if (( DRY_RUN )); then
      echo "--- [$IDX/$TOTAL] $tag -> $leaf_dir"
      (( need_infer )) && { printf '    (cd %q && ' "$LF_ROOT"; printf '%q ' "${infer_cmd[@]}"; printf ')\n'; }
      printf '    '; printf '%q ' "${score_cmd[@]}"; printf '\n'
      continue
    fi

    mkdir -p "$leaf_dir"
    log="$LOG_DIR/${tag}.log"
    echo "[+] [$IDX/$TOTAL] $(date +%H:%M:%S) start: $tag -> $log"
    rc=0
    # 서브셸 + 명시적 `|| exit $?`. `{ ...; } || rc=$?` 로 묶으면 그룹이 `||` 리스트의
    # 왼쪽 항이 되어 **그룹 안 전체에서 set -e 가 무시**된다 — 추론이 OOM 이나
    # max_lora_rank 거절로 죽어도 채점기가 그대로 이어 돌고, 이전 실행이 남긴 낡은
    # predictions 를 채점해 marker 까지 만들어 그 leaf 를 영구 skip 시킨다.
    (
      if (( need_infer )); then
        ( cd "$LF_ROOT" && "${infer_cmd[@]}" ) || exit $?
      fi
      "${score_cmd[@]}"
    ) >"$log" 2>&1 || rc=$?

    if (( rc != 0 )); then
      echo "[eval09][!] FAILED ($rc): $tag — log: $log" >&2
      FAILED=$((FAILED+1)); FAILED_LEAVES+=("$tag")
    else
      echo "[+] [$IDX/$TOTAL] $(date +%H:%M:%S) done:  $tag"
      DONE=$((DONE+1))
    fi
  done
done

if (( DRY_RUN )); then
  echo "[eval09] dry-run 종료 (실행한 것 없음)"
  exit 0
fi

echo "[eval09] 완료 $DONE · 건너뜀 $SKIPPED · 실패 $FAILED (전체 $TOTAL)"
if (( FAILED )); then
  echo "[eval09] 실패 leaf: ${FAILED_LEAVES[*]}" >&2
  echo "[eval09] 같은 커맨드를 다시 돌리면 끝난 leaf 는 건너뛴다." >&2
  exit 1
fi
