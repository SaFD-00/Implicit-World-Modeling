"""학습 YAML 생성기 SSoT (노트북 Cell 10/12 이식본).

``configs/train/**`` 아래의 LlamaFactory 학습 YAML 을 레지스트리
(``implicit_world_modeling.lf_registry``) + GPU 정책
(``scripts.gpu_policy.resolve_gpu_policy``) 로부터 결정론적으로 재생성한다.

    python -m implicit_world_modeling.gen_configs --check   # 커밋본과 대조 (CI 게이트)
    python -m implicit_world_modeling.gen_configs --write   # 재생성
    python -m implicit_world_modeling.gen_configs --write --out-base /tmp/x

★ 노트북 대비 의도적 제거
--------------------------------------------------------------------------
* ``_YAML_GEN_DS = {"AndroidControl_EXP05"}`` allowlist — 삭제. EXP03/04 를
  "hand-fix 라 복구 불가" 라는 이유로 보호하던 필터였으나, 그 YAML 은 디스크
  어디에도 존재하지 않았다 (전역 탐색 확인). 전 실험군을 생성한다.
* ``if GPU_TYPE == "RTX5090": _ds_path = ...offload...`` swap — 삭제.
  offload 는 GPU 종류만으로 갈리지 않는다 — ``resolve_gpu_policy`` 가
  ``(gpu_type, size_class)`` 쌍으로 판정한다 ((A100|H100) × 3-4B 만 no-offload,
  7-9B 와 RTX5090 은 offload).
* ``_PER_DEVICE_BS_BY_SIZE`` / ``_derive_grad_accum`` / half-batch 예외 — 삭제.
  batch/grad_accum/deepspeed 세 값은 전부 ``resolve_gpu_policy`` 가 준다.

★ 커밋 YAML 의 GPU 트리오는 RTX5090×2 baseline
--------------------------------------------------------------------------
커밋된 YAML 은 ``resolve_gpu_policy("RTX5090", 2, ...)`` 값
(pdbs=1, ga=32, deepspeed=offload) 으로 고정 emit 한다. 다른 GPU 조합에서는
YAML 을 다시 쓰지 않고 ``llamafactory-cli train cfg.yaml key=value`` 런타임
override 로 주입한다 (``scripts/gpu_policy.py`` docstring 참조).
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from implicit_world_modeling.lf_registry import (
    _MODEL_CONFIG,
    _STAGE1_ONLY,
    BASE_DIR,
    CONFIGS,
    eligible_models,
)

# scripts/ 는 패키지가 아니므로 (``__init__.py`` 없음) 경로를 직접 얹어 import 한다
# — tests/test_gpu_policy.py 와 동일한 관례.
_SCRIPTS_DIR = Path(BASE_DIR) / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gpu_policy import GpuPolicy, resolve_gpu_policy  # noqa: E402

# ============================================================
# === 상수 ===
# ============================================================

# 커밋 YAML 의 baseline GPU 조합 (as-trained). 다른 조합은 런타임 override.
BASELINE_GPU_TYPE = "RTX5090"
BASELINE_NPROC = 2

DEFAULT_OUT_BASE = Path(BASE_DIR) / "configs" / "train"

# YAML 의 deepspeed 필드는 cwd=LlamaFactory 기준 상대경로 (``media_dir: ../data`` ,
# ``output_dir: ../outputs/...`` 와 같은 기준) 인 반면, gpu_policy 는 cwd=repo root
# 기준 경로를 반환한다. 같은 파일을 가리키므로 접두만 벗겨 쓴다.
_LF_PATH_PREFIX = "LlamaFactory/"

# as-trained YAML 이 소실된 실험군 — 생성기 재구성본임을 파일 헤더에 명시한다.
RECONSTRUCTED_DATASETS = frozenset({"AndroidControl_EXP03", "AndroidControl_EXP04"})
RECONSTRUCTED_HEADER = (
    "# [reconstructed 2026-07-13] as-trained YAML 소실 — 생성기 재구성본, "
    "실제 학습 설정과 다를 수 있음\n"
)


def _deepspeed_field(policy: GpuPolicy) -> str:
    """gpu_policy 의 repo-root 상대경로를 YAML 의 LF-root 상대경로로 변환."""
    path = policy.deepspeed
    if path.startswith(_LF_PATH_PREFIX):
        return path[len(_LF_PATH_PREFIX) :]
    return path


def _header(ds_name: str) -> str:
    return RECONSTRUCTED_HEADER if ds_name in RECONSTRUCTED_DATASETS else ""


# ============================================================
# === Stage 1 YAML (노트북 Cell 10) ===
# ============================================================
def render_stage1(cfg: dict, mode: str, policy: GpuPolicy) -> str:
    """Stage 1 (World Modeling) 학습 YAML 본문. full / lora 두 벌."""
    s1 = cfg[f"stage1_{mode}"]
    mcfg = cfg["model_config"]

    ds_line = f"deepspeed: {_deepspeed_field(policy)}\n"

    # diff loss: stage1 config 에 플래그가 있으면 (AC_EXP02 / AC_EXP05) method 에 주입.
    diff_loss_line = (
        "use_diff_token_weighted_loss: true\n"
        if s1.get("use_diff_token_weighted_loss")
        else ""
    )

    output_dir = cfg[f"save_s1_{mode}"]

    if mode == "full":
        method_block = f"""\
### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: {str(mcfg["freeze_vision_tower"]).lower()}"""
    else:
        method_block = f"""\
### method
stage: sft
do_train: true
finetuning_type: lora
freeze_vision_tower: {str(mcfg["freeze_vision_tower"]).lower()}
lora_rank: {s1["lora_rank"]}
lora_alpha: {s1["lora_alpha"]}
lora_target: all
lora_dropout: {s1["lora_dropout"]}"""

    return f"""\
{_header(cfg["dataset_name"])}### model
model_name_or_path: {cfg["model_id"]}
trust_remote_code: true
image_max_pixels: {cfg["image_max_pixels"]}
image_min_pixels: {cfg["image_min_pixels"]}

{method_block}
{diff_loss_line}
### dataset
dataset: {cfg["ds_s1_train"]}
template: {cfg["template"]}
cutoff_len: {cfg["cutoff_len"]}
overwrite_cache: false
preprocessing_num_workers: 16
media_dir: ../data

### output
output_dir: {output_dir}
logging_steps: 1
save_strategy: {s1["save_strategy"]}
save_total_limit: 5
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: {policy.per_device_train_batch_size}
gradient_accumulation_steps: {policy.gradient_accumulation_steps}
learning_rate: {s1["lr"]}
num_train_epochs: {s1["epochs"]}
lr_scheduler_type: {s1["lr_scheduler_type"]}
warmup_ratio: {s1["warmup_ratio"]}
weight_decay: {s1["weight_decay"]}
max_grad_norm: {s1["max_grad_norm"]}
bf16: true
gradient_checkpointing: true
{ds_line}ddp_timeout: 18000000
# resume_from_checkpoint: true
"""


# ============================================================
# === Stage 2 YAML (노트북 Cell 12) ===
# ============================================================
def render_stage2(cfg: dict, mode: str, policy: GpuPolicy) -> dict[str, str]:
    """Stage 2 (Action Prediction) 학습 YAML — variant 3 종을 한 번에 렌더한다.

    variant: base / world-model-full / world-model-lora (Stage 1 계보).
    """
    s2 = cfg["stage2"]
    mcfg = cfg["model_config"]

    ds_line = f"deepspeed: {_deepspeed_field(policy)}\n"

    if mode == "full":
        method_block = (
            "### method\n"
            "stage: sft\n"
            "do_train: true\n"
            "finetuning_type: full\n"
            f"freeze_vision_tower: {str(mcfg['freeze_vision_tower']).lower()}"
        )
        # Full FT 는 LoRA 대비 lr 을 낮춰 안정화.
        lr_value = 1.5e-5
    else:
        method_block = (
            "### method\n"
            "stage: sft\n"
            "do_train: true\n"
            "finetuning_type: lora\n"
            f"freeze_vision_tower: {str(mcfg['freeze_vision_tower']).lower()}\n"
            f"lora_rank: {s2['lora_rank']}\n"
            f"lora_alpha: {s2['lora_alpha']}\n"
            "lora_target: all\n"
            f"lora_dropout: {s2['lora_dropout']}"
        )
        lr_value = s2["lr"]

    common_config = f"""\
{_header(cfg["dataset_name"])}### model
model_name_or_path: {{model_name_or_path}}
trust_remote_code: true
image_max_pixels: {cfg["image_max_pixels"]}
image_min_pixels: {cfg["image_min_pixels"]}

{method_block}

### dataset
dataset: {cfg["ds_s2_train"]}
template: {cfg["template"]}
cutoff_len: {cfg["cutoff_len"]}
overwrite_cache: false
preprocessing_num_workers: 16
media_dir: ../data

### output
output_dir: {{output_dir}}
logging_steps: 1
save_strategy: {s2["save_strategy"]}
save_total_limit: 5
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: {policy.per_device_train_batch_size}
gradient_accumulation_steps: {policy.gradient_accumulation_steps}
learning_rate: {lr_value}
num_train_epochs: {s2["epochs"]}
lr_scheduler_type: {s2["lr_scheduler_type"]}
warmup_ratio: {s2["warmup_ratio"]}
weight_decay: {s2["weight_decay"]}
max_grad_norm: {s2["max_grad_norm"]}
bf16: true
gradient_checkpointing: true
{ds_line}ddp_timeout: 18000000
# resume_from_checkpoint: true
"""

    variants = {
        "base": {
            "model_name_or_path": cfg["model_id"],
            "output_dir": cfg[f"save_s2_{mode}_base"],
        },
        "world-model-full": {
            "model_name_or_path": cfg["hf_s1_model_full"],
            "output_dir": cfg[f"save_s2_{mode}_world_from_full"],
        },
        "world-model-lora": {
            "model_name_or_path": cfg["hf_s1_model_lora"],
            "output_dir": cfg[f"save_s2_{mode}_world_from_lora"],
        },
    }

    return {
        variant: common_config.format(
            model_name_or_path=params["model_name_or_path"],
            output_dir=params["output_dir"],
        )
        for variant, params in variants.items()
    }


# ============================================================
# === 전체 생성 ===
# ============================================================
def generate_all(
    gpu_type: str = BASELINE_GPU_TYPE, nproc: int = BASELINE_NPROC
) -> dict[str, str]:
    """``{relpath: content}`` — 자격 있는 (모델 × DS × stage × mode) 조합 전부."""
    out: dict[str, str] = {}
    for ds_name, ds_cfgs in _iter_datasets():
        for model_key in eligible_models(ds_name):
            cfg = ds_cfgs[model_key]
            size_class = _MODEL_CONFIG[model_key]["size"]
            subfolder = cfg["lf_subfolder"]

            for mode in ("full", "lora"):
                policy = resolve_gpu_policy(
                    gpu_type=gpu_type,
                    nproc=nproc,
                    size_class=size_class,
                    ds_name=ds_name,
                    mode=mode,
                )
                rel = f"{subfolder}/stage1_{mode}/{model_key}_world-model.yaml"
                out[rel] = render_stage1(cfg, mode, policy)

                # Stage 2 를 지원하지 않는 DS (MC / EXP04 / EXP05) 는 skip.
                if ds_name in _STAGE1_ONLY:
                    continue
                for variant, content in render_stage2(cfg, mode, policy).items():
                    rel = f"{subfolder}/stage2_{mode}/{model_key}_{variant}.yaml"
                    out[rel] = content
    return out


def _iter_datasets():
    """``(ds_name, {model_key: cfg})`` — CONFIGS 를 DS 우선으로 뒤집어 순회."""
    ds_names = list(next(iter(CONFIGS.values())).keys())
    for ds_name in ds_names:
        yield ds_name, {mk: CONFIGS[mk][ds_name] for mk in CONFIGS}


def write_all(out_base: Path, generated: dict[str, str]) -> tuple[int, int]:
    """``(written, unchanged)``. 기존 파일과 내용이 같으면 mtime 을 건드리지 않는다."""
    written = unchanged = 0
    for rel, content in sorted(generated.items()):
        path = out_base / rel
        if path.exists() and path.read_text() == content:
            unchanged += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written += 1
    return written, unchanged


def check_all(out_base: Path, generated: dict[str, str]) -> list[str]:
    """디스크와 양방향 대조 — 내용 불일치 / 누락 / 고아 파일 전부 보고한다.

    고아(생성기가 만들지 않는데 디스크에 있는 파일) 도 실패로 취급한다: 그건
    자격 테이블이 틀렸다는 신호다.
    """
    problems: list[str] = []
    on_disk = {
        str(p.relative_to(out_base)) for p in out_base.rglob("*.yaml") if p.is_file()
    }

    for rel in sorted(generated):
        path = out_base / rel
        if not path.exists():
            problems.append(f"[missing] {rel}")
            continue
        actual = path.read_text()
        if actual != generated[rel]:
            diff = difflib.unified_diff(
                actual.splitlines(keepends=True),
                generated[rel].splitlines(keepends=True),
                fromfile=f"{rel} (on disk)",
                tofile=f"{rel} (generated)",
            )
            problems.append(f"[diff] {rel}\n" + "".join(diff))

    for rel in sorted(on_disk - set(generated)):
        problems.append(f"[orphan] {rel} — 생성기가 만들지 않는 파일")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="configs/train/** 학습 YAML 생성기 (레지스트리 + GPU 정책 SSoT)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="configs/train/** 재생성.")
    group.add_argument(
        "--check",
        action="store_true",
        help="디스크 산출물과 대조 — 불일치 시 diff 출력 후 exit 1.",
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=DEFAULT_OUT_BASE,
        help=f"출력 베이스 디렉토리 (기본: {DEFAULT_OUT_BASE}).",
    )
    args = parser.parse_args(argv)

    generated = generate_all()
    out_base: Path = args.out_base

    if args.write:
        written, unchanged = write_all(out_base, generated)
        print(
            f"[gen_configs] {len(generated)} YAML → {out_base} "
            f"(written={written}, unchanged={unchanged})"
        )
        return 0

    if not out_base.exists():
        print(f"[!] gen_configs: out-base 없음 — {out_base}", file=sys.stderr)
        return 1

    problems = check_all(out_base, generated)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print(
            f"[!] gen_configs --check FAILED: {len(problems)} 건 불일치 "
            f"(생성 {len(generated)} 개 기준)",
            file=sys.stderr,
        )
        return 1

    print(f"[gen_configs] --check OK — {len(generated)} YAML 이 {out_base} 와 일치")
    return 0


if __name__ == "__main__":
    sys.exit(main())
