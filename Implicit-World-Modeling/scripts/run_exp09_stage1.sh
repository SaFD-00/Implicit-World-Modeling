#!/usr/bin/env bash
# AC_EXP09 stage1 LoRA 학습 러너 — 도메인(time_mgmt/media) x GPU 타입
# (RTX5090/A100/H100) x GPU 대수(1/2/4) 일반화, 5 epoch / 8개 fractional 체크포인트.
#
# 왜 gpu_policy.py 의 resolve_overrides 경로를 통째로 쓰지 않는가
# --------------------------------------------------------------
# resolve_overrides -> gpu_policy.py 는 GLOBAL_BATCH_SIZE=64 를 하드코딩하고
# gradient_accumulation_steps 를 64/(pdbs*nproc) 로 강제한다. EXP09 는 도메인별
# train 표본 수에 맞춰 quarter-epoch 이 정수 step 에 떨어지는 global batch(=32)
# 를 GPU 대수/타입과 무관하게 고정해야 해서 그 강제 경로를 우회한다. 대신
# gpu_policy.py 의 CLI 를 서브프로세스로 불러 per_device_train_batch_size/
# deepspeed/offload 세 값만 그대로 재사용한다 — 이 값들은 GLOBAL_BATCH_SIZE
# 상수와 무관하게 gpu_type/size_class/mode/ds_name 조합만으로 결정되고,
# AndroidControl_EXP09_{time_mgmt,media} 는 _HALF_BATCH_DATASETS 어느 집합에도
# 없어 그대로 통과한다. grad_accum 은 GLOBAL_BATCH=32 기준으로 이 스크립트가
# 직접 계산한다. --nproc 유효성(RTX5090={1,2}, A100/H100={1,2,4,8})도
# gpu_policy.py 호출 자체가 검증하므로 여기서 다시 만들지 않는다.
#
# GLOBAL_BATCH=32 로 고정하는 이유
# --------------------------------
# GPU 1/2/4장, RTX5090/A100/H100 어느 조합이든 global batch 를 고정해야 step
# 수·LR 스케줄이 동일한 하나의 실험으로 남는다. build_exp09_data.py 가 각
# 도메인의 train 을 4x32=128 의 배수로 미리 trim 해 두므로 quarter-epoch(step)
# = N/128 은 항상 정수다.
#
# deepspeed offload 여부는 GPU 타입이 정한다 (gpu_policy.py 위임)
# --------------------------------------------------------------
# RTX5090(32GB)은 크기·모드 무관 offload 필수다 (2026-09-15 실측: OOM peak 을
# 지배하는 항은 lm_head logits — 시퀀스 길이 x vocab — 이라 LoRA 라도 예외가
# 없다). A100/H100(80GB)+lora 는 offload 없이도 들어가 pdbs=2 로 뛴다. 이 분기는
# gpu_policy.py::_is_no_offload_combo 가 이미 판정해 둔 것을 CLI 로 가져올
# 뿐이다. offload 가 꺼지는 조합에서는 CPUAdam JIT 빌드용 라이브러리 경로 확장이
# 불필요해 건너뛴다.
#
# 사용법
#   bash scripts/run_exp09_stage1.sh --domain time_mgmt                        # RTX5090x1 (기본), ~13시간대
#   bash scripts/run_exp09_stage1.sh --domain media --gpu-type A100 --nproc 4  # A100x4
#   bash scripts/run_exp09_stage1.sh --domain time_mgmt --nproc 2 --smoke-test
#   bash scripts/run_exp09_stage1.sh --domain time_mgmt --select-checkpoints
#   bash scripts/run_exp09_stage1.sh --domain media --gpu-type H100 --nproc 4 --dry-run
#
# 환경변수
#   CUDA_DEVICE     nproc=1 일 때 쓸 GPU 인덱스. 기본 1.
#   CUDA_DEVICES    nproc>1 일 때 쓸 콤마 리스트(예 "0,1"). 미지정 시 0..nproc-1.
#                   어느 GPU 가 비어있는지는 시점마다 다르다 — 특정 인덱스를
#                   "안전하다"고 하드코딩하지 않는다. 실행 전 nvidia-smi 로 확인할 것.
#   CONDA_ENV       기본 /opt/miniconda3/envs/implicit-world-modeling
#   SMOKE_MAX_STEPS 기본 3
#   SMOKE_DATASET_DIR / SMOKE_DATASET  스모크용 dataset_dir/dataset 키 override

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LF_ROOT="$BASE_DIR/LlamaFactory"
LOG_DIR="$BASE_DIR/logs"
LF_DATASET_DIR="$BASE_DIR/configs/lf_dataset"

CONDA_ENV="${CONDA_ENV:-/opt/miniconda3/envs/implicit-world-modeling}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-3}"
GLOBAL_BATCH=32
# 생성기가 하드코딩하는 save_total_limit=5 로는 5 epoch x 4(quarter) = 20 회 저장을
# 못 버틴다(oldest-first 정리로 앞쪽 fractional 체크포인트가 학습 도중 삭제됨).
SAVE_TOTAL_LIMIT=24

DOMAIN=""
GPU_TYPE="RTX5090"
NPROC=1
MODE="train"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)              DOMAIN="$2"; shift 2 ;;
    --gpu-type)            GPU_TYPE="$2"; shift 2 ;;
    --nproc)               NPROC="$2"; shift 2 ;;
    --smoke-test)          MODE="smoke"; shift ;;
    --select-checkpoints)  MODE="select"; shift ;;
    --dry-run)             DRY_RUN=1; shift ;;
    -h|--help)             sed -n '1,38p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$DOMAIN" in
  time_mgmt|media) ;;
  *) echo "[!] --domain 은 time_mgmt|media 중 하나여야 한다 (받은 값: '${DOMAIN}')" >&2; exit 2 ;;
esac

YAML="$BASE_DIR/configs/train/IWM-AC_EXP09/stage1_lora/qwen2.5-vl-3b_world-model_${DOMAIN}.yaml"
DATASET_JSONL="$BASE_DIR/data/AndroidControl_EXP09/stage1_train_${DOMAIN}.jsonl"
[[ -f "$YAML" ]] || { echo "[!] YAML 이 없다: $YAML (python -m implicit_world_modeling.gen_configs --write 먼저 실행했는지 확인)" >&2; exit 1; }
[[ -f "$DATASET_JSONL" ]] || { echo "[!] train jsonl 이 없다: $DATASET_JSONL (scripts/build_exp09_data.py 먼저 실행)" >&2; exit 1; }

# ── output_dir (YAML 의 cwd 상대경로를 셸에서도 알아야 한다) ──────────────────
OUT_REL="$(sed -nE 's/^output_dir:[[:space:]]*(.+)$/\1/p' "$YAML")"
[[ -n "$OUT_REL" ]] || { echo "[!] YAML 에 output_dir 이 없다: $YAML" >&2; exit 1; }
OUT_DIR="$(cd "$LF_ROOT" && python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_REL")"
[[ "$MODE" == "smoke" ]] && OUT_DIR="${OUT_DIR}_smoketest"

TARGET_EPOCHS="0.25 0.5 0.75 1.0 2.0 3.0 4.0 5.0"

# ── epoch -> checkpoint 매핑 (GPU/conda 불요, stdlib only) ────────────────────
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
    print("[!] epoch->checkpoint 매핑 실패 — 정수 step 산술이 깨졌다는 뜻이므로 버그로 보고할 것.",
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

# ── GPU 정책 (SSoT 는 scripts/gpu_policy.py) ──────────────────────────────────
POLICY_JSON="$(python3 "$BASE_DIR/scripts/gpu_policy.py" \
  --gpu-type "$GPU_TYPE" --nproc "$NPROC" --size-class 3-4B \
  --ds "AndroidControl_EXP09_${DOMAIN}" --mode lora --format json)"
PDBS="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["per_device_train_batch_size"])' "$POLICY_JSON")"
DEEPSPEED_PATH="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["deepspeed"])' "$POLICY_JSON")"
OFFLOAD="$(python3 -c 'import json,sys; print("1" if json.loads(sys.argv[1])["offload"] else "0")' "$POLICY_JSON")"

DENOM=$((PDBS * NPROC))
if (( GLOBAL_BATCH % DENOM != 0 )); then
  echo "[!] GLOBAL_BATCH($GLOBAL_BATCH) 이 per_device_train_batch_size($PDBS) x nproc($NPROC) = $DENOM 로 나누어떨어지지 않는다." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / DENOM))
# gpu_policy.py 의 deepspeed 경로는 BASE_DIR(repo root) 기준 상대경로다(모듈
# docstring: "cwd=BASE_DIR 전제"). 이 스크립트는 llamafactory-cli 를 cwd=$LF_ROOT
# 에서 부르므로 상대경로 그대로 넘기면 $LF_ROOT/LlamaFactory/... 를 찾다 실패한다
# (실측 확인) — 절대경로로 바꿔 cwd 와 무관하게 만든다.
DEEPSPEED_PATH="$BASE_DIR/$DEEPSPEED_PATH"

# ── 체크포인트 스텝 산술 ───────────────────────────────────────────────────────
# quarter-epoch 이 정수 global step 에 떨어지려면 train 행 수 N 이
# 4 x GLOBAL_BATCH 의 배수여야 한다 (build_exp09_data.py 가 이미 그렇게 trim
# 해 둔다). 어긋나면 8개 목표 체크포인트가 존재하지 않게 되는데, 그 사실은
# 학습이 끝난 뒤 --select-checkpoints 에서야 드러난다.
N="$(wc -l < "$DATASET_JSONL")"
if (( N % (4 * GLOBAL_BATCH) != 0 )); then
  echo "[!] train 행 수 N=$N 이 4 x GLOBAL_BATCH($GLOBAL_BATCH)=$((4 * GLOBAL_BATCH)) 의 배수가 아니다." >&2
  echo "    quarter-epoch 이 정수 step 에 떨어지지 않아 8개 목표 체크포인트를 만들 수 없다." >&2
  echo "    scripts/build_exp09_data.py 로 데이터를 다시 만들었는지 확인하라." >&2
  exit 1
fi
QUARTER=$((N / (4 * GLOBAL_BATCH)))

# ── GPU 디바이스 선택 ──────────────────────────────────────────────────────────
if [[ "$NPROC" == "1" ]]; then
  CUDA_DEVICE="${CUDA_DEVICE:-1}"
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
else
  if [[ -n "${CUDA_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  else
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC - 1)))"
    echo "[!] CUDA_DEVICES 미지정 — 기본값 '$CUDA_VISIBLE_DEVICES' 사용. nvidia-smi 로 해당 GPU 들이 비어있는지 먼저 확인할 것." >&2
  fi
fi

# ── env (stage1_train.sh / _common.sh 가 주입하던 것 중 실제로 필요한 것) ──────
[[ -d "$CONDA_ENV" ]] || { echo "[!] conda env 가 없다: $CONDA_ENV" >&2; exit 1; }
export PATH="$CONDA_ENV/bin:$PATH"
# LlamaFactory 는 editable install 의 .pth 가 존재하지 않는 체크아웃 경로를 가리키고 있어
# PYTHONPATH 없이는 ModuleNotFoundError 로 죽는다.
export PYTHONPATH="$LF_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CPUAdam JIT 빌드는 -lcudart -lcublas 를 링크하는데 그 .so 들은 $CONDA_PREFIX/lib 에만
# 있다(빌드가 기본으로 보는 lib64 에는 stubs 뿐이라 ld 가 실패한다). offload 가 꺼진
# 조합(A100/H100+lora)은 CPUAdam 자체를 안 쓰므로 이 확장이 불필요하다.
if [[ "$OFFLOAD" == "1" ]]; then
  _NV="$CONDA_ENV/lib/python3.12/site-packages/nvidia"
  _CUDA_LIBS="$CONDA_ENV/lib:$_NV/curand/lib:$_NV/cuda_runtime/lib:$_NV/cublas/lib"
  export LIBRARY_PATH="${_CUDA_LIBS}${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="${_CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# deepspeed 는 프로세스 그룹을 요구한다. nproc=1 이라도 torchrun 이 세팅해 주는 게 가장 싸다.
LAUNCH_ENV=(FORCE_TORCHRUN=1 NNODES=1 "NPROC_PER_NODE=$NPROC")

DS_DIR="$LF_DATASET_DIR"
OVERRIDES=(
  "dataset_dir=$DS_DIR"
  "media_dir=$BASE_DIR/data"
  "output_dir=$OUT_DIR"
  "per_device_train_batch_size=$PDBS"
  "gradient_accumulation_steps=$GRAD_ACCUM"
  "deepspeed=$DEEPSPEED_PATH"
  "save_steps=$QUARTER"
  "save_total_limit=$SAVE_TOTAL_LIMIT"
)
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
  python3 - "$OUT_DIR" "$YAML" "$DATASET_JSONL" "$BASE_DIR" "$MODE" "$DOMAIN" "$GPU_TYPE" \
    "$NPROC" "$QUARTER" "$GRAD_ACCUM" "$PDBS" "$DEEPSPEED_PATH" "$OFFLOAD" "${OVERRIDES[@]}" <<'PY'
import hashlib, json, os, subprocess, sys

(out_dir, yaml_path, jsonl, base_dir, mode, domain, gpu_type,
 nproc, quarter, grad_accum, pdbs, deepspeed_path, offload) = sys.argv[1:14]
overrides = sys.argv[14:]
quarter = int(quarter)

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

# epoch E 의 global step = E x N/GLOBAL_BATCH = E x 4 x quarter.
epoch_to_quarters = {"0.25": 1, "0.5": 2, "0.75": 3, "1.0": 4, "2.0": 8, "3.0": 12, "4.0": 16, "5.0": 20}
target_epoch_checkpoints = {k: v * quarter for k, v in epoch_to_quarters.items()}

meta = {
    "experiment": f"AC_EXP09_stage1_lora_{domain}",
    "mode": mode,
    "domain": domain,
    "git_commit": commit,
    "git_dirty": bool(dirty),
    "seed": 42,
    "gpu_type": gpu_type,
    "nproc": int(nproc),
    "per_device_train_batch_size": int(pdbs),
    "gradient_accumulation_steps": int(grad_accum),
    "global_batch_size": int(pdbs) * int(nproc) * int(grad_accum),
    "deepspeed": deepspeed_path,
    "offload": bool(int(offload)),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "config_path": yaml_path,
    "config_yaml": open(yaml_path).read(),
    "cli_overrides": overrides,
    "dataset_jsonl": jsonl,
    "dataset_sha256": sha256(jsonl) if os.path.isfile(jsonl) else None,
    "dataset_n_records": sum(1 for _ in open(jsonl)) if os.path.isfile(jsonl) else None,
    "quarter_epoch_steps": quarter,
    "target_epoch_checkpoints": target_epoch_checkpoints,
    "gpu_policy_source": "scripts/gpu_policy.py --format json (per_device_train_batch_size/deepspeed/offload only; "
                          "gradient_accumulation_steps recomputed against this experiment's GLOBAL_BATCH=32, "
                          "not gpu_policy.py's own GLOBAL_BATCH_SIZE=64)",
}
path = os.path.join(out_dir, "run_meta.json")
with open(path, "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"[meta] wrote {path}")
PY
}

CMD="cd '$LF_ROOT' && env ${LAUNCH_ENV[*]} llamafactory-cli train '$YAML' ${OVERRIDES[*]}"
echo "[cfg] mode        : $MODE"
echo "[cfg] domain      : $DOMAIN"
echo "[cfg] gpu         : $GPU_TYPE x $NPROC (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<dry-run: unset>})"
echo "[cfg] pdbs/ga     : $PDBS / $GRAD_ACCUM (global batch $((PDBS * NPROC * GRAD_ACCUM)))"
echo "[cfg] deepspeed   : $DEEPSPEED_PATH (offload=$OFFLOAD)"
echo "[cfg] YAML        : $YAML"
echo "[cfg] output_dir  : $OUT_DIR"
echo "[cfg] quarter-step: $QUARTER (targets: $(python3 -c "print(', '.join(str($QUARTER*m) for m in (1,2,3,4,8,12,16,20)))"))"
echo "[cfg] overrides   : ${OVERRIDES[*]}"

if (( DRY_RUN )); then
  echo "[dry-run] $CMD"
  exit 0
fi

write_run_meta

mkdir -p "$LOG_DIR"
TAG="run_exp09_stage1_${DOMAIN}_${MODE}_$(date +%Y%m%d_%H%M%S)"
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
