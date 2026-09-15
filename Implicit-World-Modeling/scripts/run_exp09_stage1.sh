#!/usr/bin/env bash
# AC_EXP09 stage1 LoRA 학습 러너 (Time Management 도메인 적응).
#
# 왜 stage1_train.sh 를 쓰지 않는가
# --------------------------------
# stage1_train.sh 는 `resolve_overrides` → `scripts/gpu_policy.py` 를 타는데, 그 정책은
# GLOBAL_BATCH_SIZE=64 를 하드코딩하고 gradient_accumulation_steps 를
# `64 / (pdbs × nproc)` 으로 **강제**한다 (_common.sh:1094-1103 이 주입 직전 재검증까지 한다).
# EXP09 는 grad_accum=37 (3996/37 = epoch 당 정확히 108 optimizer step) 이 필요해서
# 그 경로로는 애초에 통과할 수 없다. 그래서 _common.sh 를 source 하지 않고
# llamafactory-cli 를 직접 부른다. 대신 _common.sh 가 주입하던 것 중 실제로 필요한 것은
# 아래 env 블록에서 전부 재현한다.
#
# 사용법
#   bash scripts/run_exp09_stage1.sh --deepspeed        # 실제 학습 (3 epoch, ~13 시간)
#   bash scripts/run_exp09_stage1.sh --smoke-test       # 소수 step 스모크 (별도 output_dir)
#   bash scripts/run_exp09_stage1.sh --select-checkpoints  # 학습 후 epoch→checkpoint 매핑만
#   bash scripts/run_exp09_stage1.sh --dry-run          # 최종 커맨드만 출력
#
# --deepspeed (ZeRO-3 + CPU offload) — 2026-09-15 실측으로 필요성 확인
# ---------------------------------------------------------------------
# deepspeed 없이 돌리면 32GB 한 장에 **들어가지 않는다**. 스모크 실측: step 1 은 통과했으나
# step 2 에서 `diff_token_weighted_loss_func` 안의 cross_entropy 가 OOM (30.22 GiB 점유 중
# 594 MiB 추가 실패). OOM peak 을 지배하는 항은 lm_head logits (시퀀스 길이 × vocab 151,936)
# 이라 LoRA 여부·모델 크기와 무관하다 — scripts/gpu_policy.py:45 의 "(e) RTX5090(32GB): 크기·
# 모드와 무관하게 offload 없이는 들어가지 않는다" 와 같은 결론이다.
# 이 플래그를 주면 EXP08 과 동일한 ds_z3_offload_config.json + CPUAdam JIT 빌드에 필요한
# LIBRARY_PATH + torchrun(nproc=1) 을 함께 주입한다 (셋 다 있어야 돈다).
#
# 환경변수
#   CUDA_DEVICE          기본 1  (GPU0 은 타인의 vLLM 서버 — 건드리지 않는다)
#   CONDA_ENV            기본 /opt/miniconda3/envs/implicit-world-modeling
#   SMOKE_MAX_STEPS      기본 3
#   SMOKE_DATASET_DIR    스모크용 dataset_dir override (미지정 시 정본 dataset_dir)
#   SMOKE_DATASET        스모크용 dataset 키 override

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LF_ROOT="$BASE_DIR/LlamaFactory"
LOG_DIR="$BASE_DIR/logs"
YAML="$BASE_DIR/configs/train/IWM-AC_EXP09/stage1_lora/qwen2.5-vl-3b_time-mgmt-world-model.yaml"
DATASET_JSONL="$BASE_DIR/data/AndroidControl_EXP09/stage1_train_time_mgmt.jsonl"
LF_DATASET_DIR="$BASE_DIR/configs/lf_dataset"

CONDA_ENV="${CONDA_ENV:-/opt/miniconda3/envs/implicit-world-modeling}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-3}"

MODE="train"
DRY_RUN=0
USE_DEEPSPEED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-test)        MODE="smoke"; shift ;;
    --select-checkpoints) MODE="select"; shift ;;
    --deepspeed)         USE_DEEPSPEED=1; shift ;;
    --dry-run)           DRY_RUN=1; shift ;;
    -h|--help)           sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ── output_dir (YAML 의 cwd 상대경로를 셸에서도 알아야 한다) ──────────────────
OUT_REL="$(sed -nE 's/^output_dir:[[:space:]]*(.+)$/\1/p' "$YAML")"
[[ -n "$OUT_REL" ]] || { echo "[!] YAML 에 output_dir 이 없다: $YAML" >&2; exit 1; }
OUT_DIR="$(cd "$LF_ROOT" && python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_REL")"
[[ "$MODE" == "smoke" ]] && OUT_DIR="${OUT_DIR}_smoketest"

TARGET_EPOCHS="0.25 0.5 0.75 1.0 2.0 3.0"

# ── epoch → checkpoint 매핑 (GPU/conda 불요, stdlib only) ─────────────────────
# 실제 런이 끝난 뒤 이 스크립트를 --select-checkpoints 로 다시 돌려도 같은 결과가 나온다.
select_checkpoints() {
  local out_dir="$1" strict="$2"
  python3 - "$out_dir" "$strict" "$TARGET_EPOCHS" <<'PY'
import glob, json, os, re, sys

out_dir, strict, targets = sys.argv[1], sys.argv[2] == "strict", [float(t) for t in sys.argv[3].split()]
ckpts = []
for d in glob.glob(os.path.join(out_dir, "checkpoint-*")):
    m = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(d))
    state = os.path.join(d, "trainer_state.json")
    if not m or not os.path.isfile(state):
        continue
    with open(state) as f:
        s = json.load(f)
    ckpts.append({"dir": os.path.basename(d), "global_step": s["global_step"], "epoch": float(s["epoch"])})
ckpts.sort(key=lambda c: c["global_step"])

if not ckpts:
    print(f"[!] checkpoint-* 가 없다: {out_dir}", file=sys.stderr)
    sys.exit(1)

print(f"[ckpt] {out_dir}")
for c in ckpts:
    print(f"[ckpt]   {c['dir']:<18} global_step={c['global_step']:<5} epoch={c['epoch']!r}")

if not strict:
    sys.exit(0)

mapping, missing = {}, []
for t in targets:
    hit = next((c for c in ckpts if abs(c["epoch"] - t) <= 1e-6), None)
    if hit is None:
        nearest = min(ckpts, key=lambda c: abs(c["epoch"] - t))
        missing.append(f"  target epoch {t}: 정확히 일치하는 체크포인트 없음 "
                       f"(가장 가까운 것 {nearest['dir']} epoch={nearest['epoch']!r}, "
                       f"차이 {abs(nearest['epoch'] - t):.6g})")
        continue
    mapping[f"{t:g}"] = {"checkpoint": hit["dir"], "global_step": hit["global_step"], "epoch": hit["epoch"]}

if missing:
    print("[!] epoch→checkpoint 매핑 실패 — 정수 step 산술이 깨졌다는 뜻이므로 버그로 보고할 것.",
          file=sys.stderr)
    print("\n".join(missing), file=sys.stderr)
    sys.exit(1)

path = os.path.join(out_dir, "epoch_checkpoint_map.json")
with open(path, "w") as f:
    json.dump({"output_dir": out_dir, "tolerance": 1e-6, "map": mapping}, f, indent=2)
    f.write("\n")
print(f"[ckpt] wrote {path}")
for k, v in mapping.items():
    print(f"[ckpt]   epoch {k:<5} -> {v['checkpoint']}")
PY
}

if [[ "$MODE" == "select" ]]; then
  select_checkpoints "$OUT_DIR" strict
  exit 0
fi

# ── env (stage1_train.sh / _common.sh 가 주입하던 것 중 실제로 필요한 것) ──────
# 재현: PATH(conda bin 선두) · PYTHONPATH · DISABLE_VERSION_CHECK · alloc conf · cwd=LF_ROOT
#       · dataset_dir/media_dir 절대경로 주입.
# 의도적으로 제외: deepspeed 와 그에 딸린 CUDA_HOME 가드 / LIBRARY_PATH(nvidia curand,
#       cuda_runtime) — 전부 DeepSpeed CPUAdam JIT 빌드 전용인데 EXP09 는 deepspeed 를 쓰지 않는다.
[[ -d "$CONDA_ENV" ]] || { echo "[!] conda env 가 없다: $CONDA_ENV" >&2; exit 1; }
export PATH="$CONDA_ENV/bin:$PATH"
# LlamaFactory 는 editable install 의 .pth 가 존재하지 않는 체크아웃 경로를 가리키고 있어
# PYTHONPATH 없이는 ModuleNotFoundError 로 죽는다.
export PYTHONPATH="$LF_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

LAUNCH_ENV=()
if (( USE_DEEPSPEED )); then
  # CPUAdam JIT 빌드는 -lcudart -lcublas 를 링크하는데 그 .so 들은 $CONDA_PREFIX/lib 에만
  # 있다 (빌드가 기본으로 보는 lib64 에는 stubs 뿐이라 ld 가 실패한다).
  _NV="$CONDA_ENV/lib/python3.12/site-packages/nvidia"
  _CUDA_LIBS="$CONDA_ENV/lib:$_NV/curand/lib:$_NV/cuda_runtime/lib:$_NV/cublas/lib"
  export LIBRARY_PATH="${_CUDA_LIBS}${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="${_CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # deepspeed 는 프로세스 그룹을 요구한다. world_size=1 이라도 torchrun 이 세팅해 주는 게 가장 싸다.
  LAUNCH_ENV=(FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=1)
fi

DS_DIR="$LF_DATASET_DIR"
OVERRIDES=(
  "dataset_dir=$DS_DIR"
  "media_dir=$BASE_DIR/data"
  "output_dir=$OUT_DIR"
)
(( USE_DEEPSPEED )) && OVERRIDES+=("deepspeed=$LF_ROOT/examples/deepspeed/ds_z3_offload_config.json")
if [[ "$MODE" == "smoke" ]]; then
  [[ -n "${SMOKE_DATASET_DIR:-}" ]] && { DS_DIR="$SMOKE_DATASET_DIR"; OVERRIDES[0]="dataset_dir=$DS_DIR"; }
  [[ -n "${SMOKE_DATASET:-}" ]] && OVERRIDES+=("dataset=$SMOKE_DATASET")
  # warmup_ratio=0: 소수 step 에서는 warmup 이 전체를 덮어 lr=0 이 되고, 그러면 lora_B 가
  # 0 인 채 저장돼 "가중치가 갱신되는가" 를 검증하지 못한다.
  OVERRIDES+=("max_steps=$SMOKE_MAX_STEPS" "save_steps=1" "warmup_ratio=0" "report_to=none" "plot_loss=false")
fi

# ── run metadata ─────────────────────────────────────────────────────────────
write_run_meta() {
  mkdir -p "$OUT_DIR"
  python3 - "$OUT_DIR" "$YAML" "$DATASET_JSONL" "$BASE_DIR" "$MODE" "$USE_DEEPSPEED" "${OVERRIDES[@]}" <<'PY'
import hashlib, json, os, subprocess, sys

out_dir, yaml_path, jsonl, base_dir, mode, use_ds = sys.argv[1:7]
overrides = sys.argv[7:]
use_ds = use_ds == "1"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=base_dir,
                        capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=base_dir,
                       capture_output=True, text=True).stdout.strip()

meta = {
    "experiment": "AC_EXP09_stage1_lora_time_mgmt",
    "mode": mode,
    "git_commit": commit,
    "git_dirty": bool(dirty),
    "seed": 42,
    "world_size": 1,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "config_path": yaml_path,
    "config_yaml": open(yaml_path).read(),
    "cli_overrides": overrides,
    "dataset_jsonl": jsonl,
    "dataset_sha256": sha256(jsonl) if os.path.isfile(jsonl) else None,
    "dataset_n_records": sum(1 for _ in open(jsonl)) if os.path.isfile(jsonl) else None,
    "target_epoch_checkpoints": {
        "0.25": 27, "0.5": 54, "0.75": 81, "1.0": 108, "2.0": 216, "3.0": 324,
    },
    "deepspeed": "ds_z3_offload_config.json (ZeRO-3 + CPU offload)" if use_ds else None,
    "deviations_from_exp08_stage1_lora": [
        "deepspeed 키를 YAML 에서 제거 — 설계 지시(단일 32GB GPU 의 rank-64 LoRA 에는 불필요). "
        "단 2026-09-15 실측에서 이 구성은 16k 토큰 이상 샘플의 cross_entropy 에서 OOM 했다. "
        "--deepspeed 로 EXP08 과 동일한 ZeRO-3 + CPU offload 를 되돌릴 수 있고, "
        f"이 런은 {'그것을 사용했다' if use_ds else '사용하지 않았다'}.",
        "gradient_accumulation_steps 32 → 37 — pdbs=1 · world_size=1 · train=3996(길이필터 후) 에서 "
        "3996/37 = epoch 당 정확히 108 optimizer step (나머지 0) 이라 quarter-epoch 이 정수 step "
        "27 에 떨어진다. (최초 설계는 train=4000 가정하에 grad_accum=40 이었으나, "
        "build_exp09_data.py 의 length filter 가 train 을 3996 으로 줄이면서 40 은 더 이상 "
        "4000%(4×40)==0 조건을 만족하지 못해 37 로 재산정했다 — 999(=3996/4)의 약수 중 40 에 최근접.)",
        "save_steps 0.25 (float ratio) → 27 (정수 step) — float ratio 는 전체 step 수에 대한 균등 "
        "분할이라 목표 epoch 지점을 정확히 짚지 못한다.",
        "save_total_limit 5 → 20 — 3 epoch × 4 = 12 회 저장되므로 5 면 oldest-first 정리로 "
        "0.25 epoch 체크포인트가 학습 도중 삭제된다.",
        "learning_rate 1.0e-5 → 1.0e-4 (EXP09 설계 문서 고정값).",
        "num_train_epochs 1 → 3 (EXP09 설계 문서 고정값).",
        "seed 42 를 YAML 에 명시 — HF 기본값과 같은 값이라 동작은 불변, run_meta 와 대조 가능하게 한 것.",
        "gpu_policy.py / resolve_overrides 경로 전면 우회 — GLOBAL_BATCH=64 강제가 grad_accum=37 을 거절한다.",
        "torchrun 미사용 (world_size=1) — launcher 의 torchrun 재실행은 FORCE_TORCHRUN/multi-device 에서만 발동.",
    ],
}
path = os.path.join(out_dir, "run_meta.json")
with open(path, "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"[meta] wrote {path}")
PY
}

# 실제 런 전용 가드 ------------------------------------------------------------
# quarter-epoch 이 정수 global step 에 떨어지려면 train 행 수 N 이 4 × grad_accum 의
# 배수여야 한다 (epoch(step k) = k × ga / N). 어긋나면 6 개 목표 체크포인트가 존재하지
# 않게 되는데, 그 사실은 13 시간 뒤 --select-checkpoints 에서야 드러난다.
if [[ "$MODE" == "train" ]]; then
  GA="$(sed -nE 's/^gradient_accumulation_steps:[[:space:]]*([0-9]+).*/\1/p' "$YAML")"
  N="$(wc -l < "$DATASET_JSONL")"
  if (( N % (4 * GA) != 0 )); then
    echo "[!] train 행 수 N=$N 이 4 × grad_accum($GA) = $((4 * GA)) 의 배수가 아닙니다." >&2
    echo "    quarter-epoch 이 정수 step 에 떨어지지 않아 목표 체크포인트 6 개를 만들 수 없습니다." >&2
    echo "    N 을 되돌리거나 grad_accum 을 재산정한 뒤 다시 실행하세요." >&2
    exit 1
  fi
  if (( ! USE_DEEPSPEED )); then
    echo "[!] --deepspeed 없이 실제 런을 시작합니다. 2026-09-15 실측에서 이 구성은" >&2
    echo "    16k 토큰 이상 샘플(현재 데이터에 16 행)에서 cross_entropy OOM 으로 죽었습니다." >&2
    echo "    스크립트 상단의 --deepspeed 설명을 확인하세요." >&2
  fi
fi

CMD="cd '$LF_ROOT' && env ${LAUNCH_ENV[*]} llamafactory-cli train '$YAML' ${OVERRIDES[*]}"
echo "[cfg] mode       : $MODE"
echo "[cfg] deepspeed  : $( ((USE_DEEPSPEED)) && echo "ZeRO-3 + CPU offload" || echo "없음 (실측 OOM 구성)" )"
echo "[cfg] YAML       : $YAML"
echo "[cfg] output_dir : $OUT_DIR"
echo "[cfg] overrides  : ${OVERRIDES[*]}"
echo "[cfg] GPU        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (world_size=1)"

if (( DRY_RUN )); then
  echo "[dry-run] $CMD"
  exit 0
fi

write_run_meta

mkdir -p "$LOG_DIR"
TAG="run_exp09_stage1_${MODE}_$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/${TAG}.log"
echo "[+] start -> log: $LOG"
rc=0
bash -c "$CMD" 2>&1 | tee "$LOG" || rc=$?
if (( rc != 0 )); then
  echo "[!] 학습 실패 (exit=$rc) — log: $LOG" >&2
  exit "$rc"
fi
echo "[+] done  -> log: $LOG"

if [[ "$MODE" == "smoke" ]]; then
  select_checkpoints "$OUT_DIR" report
else
  select_checkpoints "$OUT_DIR" strict
fi
