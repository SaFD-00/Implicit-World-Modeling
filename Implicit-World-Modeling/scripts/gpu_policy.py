"""GPU 정책 SSoT (single source of truth).

``.env`` / 노트북 Cell 5 (``_PER_DEVICE_BS_BY_SIZE`` · ``_derive_grad_accum``) /
노트북 Cell 10 (``GPU_TYPE == "RTX5090"`` 일 때만 deepspeed 를 offload config 로
swap 하던 조건부 분기) / ``scripts/_common.sh`` (``CUDA_HOME`` 가드) 네 곳에
흩어져 있던 GPU 별 학습 하이퍼파라미터 결정 로직을 이 모듈 하나로 통합한다.

커밋된 학습 YAML (``configs/train/**``, 74 개)은 전부 RTX5090×2 baseline 값으로
고정돼 있다. 실행 시점에는 이 모듈이 계산한 ``per_device_train_batch_size`` /
``gradient_accumulation_steps`` / ``deepspeed`` 를
``llamafactory-cli train cfg.yaml key=value`` 런타임 override 인자로 주입한다
(LlamaFactory 의 ``src/llamafactory/hparams/parser.py:69-83`` 가 OmegaConf merge
로 이 오버라이드 패턴을 정식 지원한다). **이 모듈의 반환값이 틀리면 전 GPU
조합의 학습이 틀린다.**

★ 핵심 불변식 — deepspeed 는 항상 offload
--------------------------------------------------------------------------
``resolve_gpu_policy`` 는 ``allow_no_offload=True`` 를 명시적으로 넘기지 않는
한 ``GPU_TYPE`` · ``nproc`` · ``mode`` · ``ds_name`` 과 무관하게 항상
``ds_z3_offload_config.json`` (CPU offload 활성 DeepSpeed ZeRO-3 config) 를
반환한다. 다음 세 가지 근거로 이 불변식을 강제한다:

(a) as-trained 등가: ``configs/train/**`` 아래 커밋된 74 개 train YAML 이
    예외 0·누락 0 으로 전부 ``ds_z3_offload_config.json`` 을 쓴다. 노트북의
    ``_MODEL_CONFIG[*]["stage1_deepspeed"] = "examples/deepspeed/ds_z3_config.json"``
    (non-offload) 은 **한 번도 실행된 적 없는 죽은 기본값**이다 — 이 저장소의
    모든 생성·학습 실행은 RTX5090 경로였고, 노트북 Cell 10 은 그 경로에서만
    offload 로 swap 했다 (``if GPU_TYPE == "RTX5090": _ds_path = ...offload...``).
(b) A100/H100 에서 offload 를 빼면 EXP05 7B full FT 는 **확정 OOM** 이다
    (실측: 2GPU 기준 모델 상태만 GPU 당 ~77 GiB).
(c) OOM peak 을 지배하는 항은 lm_head logits (시퀀스 길이 × vocab 크기) 이며,
    이는 모델 파라미터 샤딩이나 GPU 대수 증설로는 줄어들지 않는다 — 따라서
    "GPU 가 크니까/많으니까 offload 없이도 될 것" 이라는 추론은 성립하지 않는다.

**조건부 offload (``if GPU_TYPE == "RTX5090": ...`` 같은 분기)는 금지된
패턴이다.** 노트북 Cell 10 이 바로 그 패턴이었고, RTX5090 이 아닌 조합에서는
한 번도 실행되지 않은 채 (a) 의 죽은 기본값으로 조용히 divergence 했다. 이
모듈은 그 실수를 반복하지 않기 위해 always-offload 를 기본값으로 하고,
opt-out 은 ``allow_no_offload=True`` 라는 명시적 플래그로만 허용한다 (미실측
경고 동반, `Out of Scope`: 이 opt-out 경로의 실측 검증은 이 브리프 밖이다).

CLI
--------------------------------------------------------------------------
다른 파이프라인 단위(``scripts/*_train.sh`` 등)가 이 모듈을
``llamafactory-cli`` override 인자 생성기로 호출할 수 있도록 CLI 진입점을
제공한다::

    python scripts/gpu_policy.py --gpu-type A100 --nproc 2 --size-class 7-9B \\
        --ds AndroidControl_EXP01 --mode full --format cli

``--format cli`` 는 stdout 에 정확히 한 줄만
(``per_device_train_batch_size=N gradient_accumulation_steps=M deepspeed=<path>``)
출력한다 — 그대로 ``llamafactory-cli`` override 인자로 append 할 수 있는
형식. 경고(``warnings``)는 stdout 을 오염시키면 안 되므로 반드시 stderr 로만
나간다. ``--format json`` 은 전 필드를 JSON 으로 stdout 에 출력한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

# ============================================================
# === 상수 ===
# ============================================================

GLOBAL_BATCH_SIZE = 64

_ALLOWED_GPU_TYPES: tuple[str, ...] = ("RTX5090", "A100", "H100")

# RTX5090 은 이 환경 baseline 상 {1,2} 장만 검증됐다. A100/H100 은 {1,2,4,8} 전부 허용.
_ALLOWED_NPROC: dict[str, tuple[int, ...]] = {
    "RTX5090": (1, 2),
    "A100": (1, 2, 4, 8),
    "H100": (1, 2, 4, 8),
}

_ALLOWED_SIZE_CLASS: tuple[str, ...] = ("7-9B", "3-4B")
_ALLOWED_MODE: tuple[str, ...] = ("full", "lora")

# per_device_train_batch_size 는 size_class/mode 와 무관하게 gpu_type 으로만 결정된다.
_BASE_PER_DEVICE_BS: dict[str, int] = {"RTX5090": 1, "A100": 2, "H100": 2}

# 긴 시퀀스(좌표/절대픽셀 표현) 실험군은 메모리 압박이 커 per_device 를 절반으로 낮춘다.
_HALF_BATCH_DATASETS: frozenset[str] = frozenset(
    {"AndroidControl_EXP03", "AndroidControl_EXP04", "AndroidControl_EXP05"}
)

# cwd=BASE_DIR (repo root) 전제의 상대경로. LlamaFactory 서브프로젝트 안의
# examples/deepspeed/ 에 두 config 모두 존재함을 확인했다 (ls LlamaFactory/examples/deepspeed/).
DEEPSPEED_OFFLOAD = "LlamaFactory/examples/deepspeed/ds_z3_offload_config.json"
DEEPSPEED_NO_OFFLOAD = "LlamaFactory/examples/deepspeed/ds_z3_config.json"
# 테스트 전용 opt-out — 프로덕션 호출자 0건 (쉘/노트북/gen_configs 는 allow_no_offload 를 넘기지 않는다).
# tests/test_gpu_policy.py:205 가 이 값을 고정한다.

_NO_OFFLOAD_WARNING = (
    "no-offload 미실측 — 8×H100 등에서 속도 이득 가능하나 검증 안 됨, "
    "EXP05 7B full FT 는 확정 OOM"
)
_HOST_RAM_WARNING = "host RAM: steady ~154GB/노드, 체크포인트 저장 시 ~175GB+"


@dataclass(frozen=True)
class GpuPolicy:
    """``resolve_gpu_policy`` 의 반환 타입. 모든 필드는 순수하게 입력으로부터 결정된다."""

    gpu_type: str
    nproc: int
    size_class: str
    ds_name: str
    mode: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    deepspeed: str
    offload: bool
    warnings: list[str] = field(default_factory=list)


def resolve_gpu_policy(
    gpu_type: str,
    nproc: int,
    size_class: str,
    ds_name: str,
    mode: str,
    allow_no_offload: bool = False,
) -> GpuPolicy:
    """GPU/데이터셋/모드 조합으로부터 학습 하이퍼파라미터를 결정하는 순수 함수.

    Parameters
    ----------
    gpu_type : {"RTX5090", "A100", "H100"}
    nproc : RTX5090 은 {1,2}, A100/H100 은 {1,2,4,8} 만 허용.
    size_class : {"7-9B", "3-4B"}
    ds_name : 데이터셋 키. ``AndroidControl_EXP03/04/05`` 는 긴 시퀀스 예외로
        ``per_device_train_batch_size`` 를 절반으로 낮춘다 (그 외 값은 자유 문자열).
    mode : {"full", "lora"}
    allow_no_offload : True 면 DeepSpeed offload 를 끈 config 를 반환한다
        (opt-out — 미실측 경고 동반). 기본 False 에서는 항상 offload.

    Raises
    ------
    ValueError
        ``gpu_type`` / ``nproc`` / ``size_class`` / ``mode`` 가 허용 범위를
        벗어나거나, ``GLOBAL_BATCH_SIZE`` 가
        ``per_device_train_batch_size * nproc`` 로 나누어떨어지지 않을 때.
    """
    if gpu_type not in _ALLOWED_GPU_TYPES:
        raise ValueError(
            f"Unsupported gpu_type={gpu_type!r}. Allowed: {_ALLOWED_GPU_TYPES}."
        )

    allowed_nproc = _ALLOWED_NPROC[gpu_type]
    if nproc not in allowed_nproc:
        raise ValueError(
            f"Unsupported nproc={nproc!r} for gpu_type={gpu_type!r}. "
            f"Allowed: {allowed_nproc}."
        )

    if size_class not in _ALLOWED_SIZE_CLASS:
        raise ValueError(
            f"Unsupported size_class={size_class!r}. Allowed: {_ALLOWED_SIZE_CLASS}."
        )

    if mode not in _ALLOWED_MODE:
        raise ValueError(f"Unsupported mode={mode!r}. Allowed: {_ALLOWED_MODE}.")

    per_device = _BASE_PER_DEVICE_BS[gpu_type]
    if ds_name in _HALF_BATCH_DATASETS:
        per_device = max(1, per_device // 2)

    denom = per_device * nproc
    if GLOBAL_BATCH_SIZE % denom != 0:
        raise ValueError(
            f"GLOBAL_BATCH_SIZE({GLOBAL_BATCH_SIZE}) is not divisible by "
            f"per_device_train_batch_size({per_device}) * nproc({nproc}) = {denom}. "
            "Adjust GPU_TYPE/nproc or GLOBAL_BATCH_SIZE."
        )
    grad_accum = GLOBAL_BATCH_SIZE // denom

    warnings: list[str] = []
    if allow_no_offload:
        deepspeed_path = DEEPSPEED_NO_OFFLOAD
        offload = False
        warnings.append(_NO_OFFLOAD_WARNING)
    else:
        deepspeed_path = DEEPSPEED_OFFLOAD
        offload = True

    if size_class == "7-9B" and mode == "full":
        warnings.append(_HOST_RAM_WARNING)

    return GpuPolicy(
        gpu_type=gpu_type,
        nproc=nproc,
        size_class=size_class,
        ds_name=ds_name,
        mode=mode,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        deepspeed=deepspeed_path,
        offload=offload,
        warnings=warnings,
    )


# ============================================================
# === CLI ===
# ============================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GPU 정책 SSoT — llamafactory-cli 런타임 override 인자를 계산한다."
        )
    )
    parser.add_argument("--gpu-type", required=True, choices=_ALLOWED_GPU_TYPES)
    parser.add_argument("--nproc", required=True, type=int)
    parser.add_argument("--size-class", required=True, choices=_ALLOWED_SIZE_CLASS)
    parser.add_argument("--ds", required=True, dest="ds_name")
    parser.add_argument("--mode", required=True, choices=_ALLOWED_MODE)
    # 테스트 전용 opt-out — 프로덕션 호출자 0건 (쉘/노트북/gen_configs 는 allow_no_offload 를 넘기지 않는다).
    # tests/test_gpu_policy.py:205 가 이 인자를 고정한다.
    parser.add_argument(
        "--allow-no-offload",
        action="store_true",
        help="DeepSpeed offload 를 끈다 (opt-out, 미실측 — 기본은 항상 offload).",
    )
    parser.add_argument("--format", choices=("cli", "json"), default="cli")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        policy = resolve_gpu_policy(
            gpu_type=args.gpu_type,
            nproc=args.nproc,
            size_class=args.size_class,
            ds_name=args.ds_name,
            mode=args.mode,
            allow_no_offload=args.allow_no_offload,
        )
    except ValueError as exc:
        print(f"[!] gpu_policy: {exc}", file=sys.stderr)
        return 1

    # warnings 는 stdout 을 오염시키면 override 인자 파싱이 깨지므로 반드시 stderr.
    for w in policy.warnings:
        print(f"[!] gpu_policy: {w}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(asdict(policy), ensure_ascii=False))
    else:
        print(
            f"per_device_train_batch_size={policy.per_device_train_batch_size} "
            f"gradient_accumulation_steps={policy.gradient_accumulation_steps} "
            f"deepspeed={policy.deepspeed}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
