"""Regression tests for scripts/gpu_policy.py — GPU 정책 SSoT.

10 GPU 조합(RTX5090×{1,2}, A100×{1,2,4,8}, H100×{1,2,4,8}) × size_class 2
× ds 5종(AndroidControl_EXP01 + EXP03/04/05/06) × mode 2 = 200 케이스 전수를
파라미터라이즈하며, offload 분기가 **(gpu_type, size_class) 쌍**으로만 갈리는
불변식을 명시적으로 고정한다:

- (A100 | H100) × 3-4B → no-offload + half-batch 예외 면제 (pdbs 안 깎임)
- 그 밖의 모든 조합    → offload

GPU 종류만 보고 갈리던 과거 설계(노트북 Cell 10 의 ``if GPU_TYPE == "RTX5090"``)
로의 회귀를 막기 위해, A100 × **7-9B** 는 여전히 offload 라는 것도 함께 고정한다.

Run:
    pytest tests/test_gpu_policy.py -v
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from gpu_policy import (  # noqa: E402
    DEEPSPEED_NO_OFFLOAD,
    DEEPSPEED_OFFLOAD,
    GLOBAL_BATCH_SIZE,
    resolve_gpu_policy,
)

GPU_POLICY_PY = REPO / "scripts" / "gpu_policy.py"

# --- 200-case matrix ---------------------------------------------------------
GPU_COMBOS = [
    ("RTX5090", 1),
    ("RTX5090", 2),
    ("A100", 1),
    ("A100", 2),
    ("A100", 4),
    ("A100", 8),
    ("H100", 1),
    ("H100", 2),
    ("H100", 4),
    ("H100", 8),
]
SIZE_CLASSES = ["7-9B", "3-4B"]
DS_NAMES = [
    "AndroidControl_EXP01",
    "AndroidControl_EXP03",
    "AndroidControl_EXP04",
    "AndroidControl_EXP05",
    "AndroidControl_EXP06",
]
MODES = ["full", "lora"]

ALL_CASES = [
    (gpu_type, nproc, size_class, ds_name, mode)
    for (gpu_type, nproc) in GPU_COMBOS
    for size_class in SIZE_CLASSES
    for ds_name in DS_NAMES
    for mode in MODES
]


def test_matrix_has_200_cases():
    assert len(ALL_CASES) == 200


# --- 1. 핵심 불변식: offload 는 (gpu_type, size_class, mode) 로 갈린다 ---------

# 80GB GPU 에서 (3-4B | lora) 는 no-offload. 남는 offload 경로는 7-9B×full 과 RTX5090.
LARGE_MEM_GPUS = {"A100", "H100"}


def _expects_no_offload(gpu_type: str, size_class: str, mode: str) -> bool:
    if gpu_type not in LARGE_MEM_GPUS:
        return False
    return size_class == "3-4B" or mode == "lora"


@pytest.mark.parametrize("gpu_type,nproc,size_class,ds_name,mode", ALL_CASES)
def test_offload_split_across_full_matrix(gpu_type, nproc, size_class, ds_name, mode):
    policy = resolve_gpu_policy(gpu_type, nproc, size_class, ds_name, mode)
    if _expects_no_offload(gpu_type, size_class, mode):
        assert policy.offload is False
        assert policy.deepspeed == DEEPSPEED_NO_OFFLOAD
    else:
        assert policy.offload is True
        assert policy.deepspeed == DEEPSPEED_OFFLOAD


def test_a100x2_3b_exp05_full_is_no_offload_and_unhalved():
    """이 정책의 동기가 된 실측 케이스 (EXP05 stage1 full FT / qwen2.5-vl-3b / A100×2).

    offload 를 켜면 165 s/step (약 4 일) 이었다. no-offload + half-batch 면제로
    pdbs=2 / ga=16 이 되며, global batch 64 는 그대로 유지된다.
    """
    policy = resolve_gpu_policy("A100", 2, "3-4B", "AndroidControl_EXP05", "full")
    assert policy.offload is False
    assert policy.deepspeed.endswith("ds_z3_config.json")
    assert policy.per_device_train_batch_size == 2
    assert policy.gradient_accumulation_steps == 16
    assert policy.warnings == []  # 정책 경로 — 미실측 경고가 붙으면 안 된다


def test_a100_7b_lora_is_no_offload_but_full_is_not():
    """7-9B 는 mode 로 갈린다 — lora 는 끄고, full 은 켠다.

    둘을 가르는 것은 optimizer state 의 크기다: lora 는 어댑터만 학습하므로
    GPU 에 올려도 되지만, full 은 모델 상태만 GPU 당 ~77 GiB 라 확정 OOM 이다.
    """
    lora = resolve_gpu_policy("A100", 2, "7-9B", "AndroidControl_EXP05", "lora")
    assert lora.offload is False
    assert lora.deepspeed == DEEPSPEED_NO_OFFLOAD
    assert lora.per_device_train_batch_size == 2  # half-batch 면제
    assert lora.gradient_accumulation_steps == 16
    assert lora.warnings == []

    full = resolve_gpu_policy("A100", 2, "7-9B", "AndroidControl_EXP05", "full")
    assert full.offload is True
    assert full.deepspeed == DEEPSPEED_OFFLOAD
    assert full.per_device_train_batch_size == 1  # half-batch 유지
    assert full.gradient_accumulation_steps == 32


def test_a100_7b_full_stays_offload_gpu_type_alone_does_not_flip_it():
    """GPU 종류만 보고 offload 를 끄던 과거 설계로의 회귀 방지.

    같은 A100 이라도 7-9B × full 은 offload 를 유지해야 한다 (없으면 확정 OOM).
    """
    policy = resolve_gpu_policy("A100", 2, "7-9B", "AndroidControl_EXP01", "full")
    assert policy.deepspeed == DEEPSPEED_OFFLOAD
    assert policy.deepspeed.endswith("ds_z3_offload_config.json")
    assert policy.offload is True
    assert policy.per_device_train_batch_size == 2
    assert policy.gradient_accumulation_steps == 16


# --- 2. pdbs x ga x nproc == 64 — 전 케이스 ----------------------------------


@pytest.mark.parametrize("gpu_type,nproc,size_class,ds_name,mode", ALL_CASES)
def test_global_batch_size_invariant(gpu_type, nproc, size_class, ds_name, mode):
    policy = resolve_gpu_policy(gpu_type, nproc, size_class, ds_name, mode)
    assert (
        policy.per_device_train_batch_size * policy.gradient_accumulation_steps * nproc
        == GLOBAL_BATCH_SIZE
    )


# --- 3. EXP03/04/05/06 절반 규칙 -------------------------------------------------


@pytest.mark.parametrize(
    "gpu_type,expected_base,expected_half",
    [("RTX5090", 1, 1), ("A100", 2, 1), ("H100", 2, 1)],
)
def test_half_batch_rule_concrete(gpu_type, expected_base, expected_half):
    base = resolve_gpu_policy(gpu_type, 2, "7-9B", "AndroidControl_EXP01", "full")
    assert base.per_device_train_batch_size == expected_base
    for half_ds in (
        "AndroidControl_EXP03",
        "AndroidControl_EXP04",
        "AndroidControl_EXP05",
        "AndroidControl_EXP06",
    ):
        halved = resolve_gpu_policy(gpu_type, 2, "7-9B", half_ds, "full")
        assert halved.per_device_train_batch_size == expected_half


def test_half_batch_rule_general():
    """긴 시퀀스 실험군은 pdbs 를 절반으로 — 단 no-offload 조합은 면제 (안 깎인다)."""
    for gpu_type, nproc in GPU_COMBOS:
        for size_class in SIZE_CLASSES:
            for mode in MODES:
                base = resolve_gpu_policy(gpu_type, nproc, size_class, "AndroidControl_EXP01", mode)
                exempt = _expects_no_offload(gpu_type, size_class, mode)
                expected = (
                    base.per_device_train_batch_size
                    if exempt
                    else max(1, base.per_device_train_batch_size // 2)
                )
                for half_ds in (
                    "AndroidControl_EXP03",
                    "AndroidControl_EXP04",
                    "AndroidControl_EXP05",
                    "AndroidControl_EXP06",
                ):
                    halved = resolve_gpu_policy(gpu_type, nproc, size_class, half_ds, mode)
                    assert halved.per_device_train_batch_size == expected, (
                        gpu_type,
                        nproc,
                        size_class,
                        mode,
                        half_ds,
                    )


# --- 4. mode 축: 80GB × 7-9B 에서만 full ≠ lora ------------------------------


@pytest.mark.parametrize("gpu_type,nproc", GPU_COMBOS)
@pytest.mark.parametrize("size_class", SIZE_CLASSES)
@pytest.mark.parametrize("ds_name", DS_NAMES)
def test_mode_only_matters_for_large_mem_7b(gpu_type, nproc, size_class, ds_name):
    """mode 는 **80GB × 7-9B** 에서만 트리오를 가른다 (lora → no-offload).

    그 밖의 조합(RTX5090 전부, 80GB × 3-4B)에서는 full 과 lora 가 같은
    (pdbs, ga, deepspeed) 를 받아야 한다 — mode 가 새는 분기가 없는지 고정한다.
    """
    full = resolve_gpu_policy(gpu_type, nproc, size_class, ds_name, "full")
    lora = resolve_gpu_policy(gpu_type, nproc, size_class, ds_name, "lora")
    full_trio = (full.per_device_train_batch_size, full.gradient_accumulation_steps, full.deepspeed)
    lora_trio = (lora.per_device_train_batch_size, lora.gradient_accumulation_steps, lora.deepspeed)

    mode_splits = gpu_type in LARGE_MEM_GPUS and size_class == "7-9B"
    if mode_splits:
        assert lora.offload is False and full.offload is True
        assert full_trio != lora_trio
    else:
        assert full_trio == lora_trio


# --- 5. baseline no-op: RTX5090x2 는 전 size_class/ds/mode 에서 pdbs=1,ga=32 ---


def test_baseline_rtx5090x2_is_noop_everywhere():
    for size_class in SIZE_CLASSES:
        for ds_name in DS_NAMES:
            for mode in MODES:
                policy = resolve_gpu_policy("RTX5090", 2, size_class, ds_name, mode)
                assert policy.per_device_train_batch_size == 1
                assert policy.gradient_accumulation_steps == 32
                assert policy.offload is True
                assert policy.deepspeed == DEEPSPEED_OFFLOAD


# --- 6. 유효성 -----------------------------------------------------------------


@pytest.mark.parametrize("nproc", [4, 8])
def test_rtx5090_over_2_gpus_rejected(nproc):
    with pytest.raises(ValueError):
        resolve_gpu_policy("RTX5090", nproc, "7-9B", "AndroidControl_EXP01", "full")


def test_unknown_gpu_type_rejected():
    with pytest.raises(ValueError):
        resolve_gpu_policy("RTX4090", 2, "7-9B", "AndroidControl_EXP01", "full")


def test_unknown_size_class_rejected():
    with pytest.raises(ValueError):
        resolve_gpu_policy("A100", 2, "9B", "AndroidControl_EXP01", "full")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        resolve_gpu_policy("A100", 2, "7-9B", "AndroidControl_EXP01", "quantized")


def test_unknown_nproc_rejected_for_a100():
    with pytest.raises(ValueError):
        resolve_gpu_policy("A100", 3, "7-9B", "AndroidControl_EXP01", "full")


# --- 7. opt-out ----------------------------------------------------------------


def test_allow_no_offload_optout_returns_non_offload_with_warning():
    policy = resolve_gpu_policy(
        "H100", 8, "7-9B", "AndroidControl_EXP01", "full", allow_no_offload=True
    )
    assert policy.deepspeed == DEEPSPEED_NO_OFFLOAD
    assert policy.offload is False
    assert any("미실측" in w for w in policy.warnings)


# --- 8. 커밋 코퍼스 대조 게이트 -------------------------------------------------


def test_committed_train_corpus_matches_baseline_policy():
    train_dir = REPO / "configs" / "train"
    if not train_dir.is_dir():
        pytest.skip(f"configs/train not found at {train_dir} — commit corpus gate skipped.")

    yaml_files = sorted(train_dir.rglob("*.yaml"))
    train_yamls = [f for f in yaml_files if "per_device_train_batch_size" in f.read_text()]
    if not train_yamls:
        pytest.skip(
            f"No train YAML with 'per_device_train_batch_size' under {train_dir} "
            "— commit corpus gate skipped."
        )

    pdbs_re = re.compile(r"^per_device_train_batch_size:\s*(\S+)", re.MULTILINE)
    ga_re = re.compile(r"^gradient_accumulation_steps:\s*(\S+)", re.MULTILINE)
    ds_re = re.compile(r"^deepspeed:\s*(\S+)", re.MULTILINE)

    trios: set[tuple[int, int, str]] = set()
    for f in train_yamls:
        text = f.read_text()
        pdbs_m, ga_m, ds_m = pdbs_re.search(text), ga_re.search(text), ds_re.search(text)
        assert pdbs_m and ga_m and ds_m, f"{f}: missing per_device/grad_accum/deepspeed key"
        trios.add((int(pdbs_m.group(1)), int(ga_m.group(1)), Path(ds_m.group(1)).name))

    # as-trained 74 개는 하한이다 — 생성기가 EXP03/04 와 3b/4b 확장분을 추가하므로 코퍼스는 자란다.
    # 개수는 부수적이고, 불변식은 "전 YAML 의 GPU 트리오가 baseline resolve 와 같다" 는 것이다:
    # 그래야 baseline(RTX5090×2)에서 런타임 override 가 no-op 이고, 다른 GPU 조합에서만 값이 바뀐다.
    assert len(train_yamls) >= 74, (
        f"as-trained 74 개가 하한인데 {len(train_yamls)} 개뿐 — 커밋 코퍼스가 유실됐다"
    )
    assert len(trios) == 1, f"non-uniform (pdbs, ga, deepspeed-basename) trio across corpus: {trios}"

    pdbs, ga, ds_basename = next(iter(trios))
    baseline = resolve_gpu_policy("RTX5090", 2, "7-9B", "AndroidControl_EXP01", "full")
    assert pdbs == baseline.per_device_train_batch_size
    assert ga == baseline.gradient_accumulation_steps
    assert ds_basename == Path(baseline.deepspeed).name
    assert baseline.offload is True


# --- CLI ------------------------------------------------------------------------


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GPU_POLICY_PY), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_format_cli_single_line_and_warnings_on_stderr():
    result = _run_cli(
        [
            "--gpu-type",
            "A100",
            "--nproc",
            "2",
            "--size-class",
            "7-9B",
            "--ds",
            "AndroidControl_EXP05",
            "--mode",
            "full",
            "--format",
            "cli",
        ]
    )
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    assert lines[0] == (
        "per_device_train_batch_size=1 gradient_accumulation_steps=32 "
        f"deepspeed={DEEPSPEED_OFFLOAD}"
    )
    # size_class=7-9B, mode=full → host RAM advisory warning must land on stderr only.
    assert result.stderr.strip() != ""
    assert "per_device_train_batch_size=" not in result.stderr


def test_cli_flipped_case_output():
    result = _run_cli(
        [
            "--gpu-type",
            "A100",
            "--nproc",
            "2",
            "--size-class",
            "7-9B",
            "--ds",
            "AndroidControl_EXP01",
            "--mode",
            "full",
            "--format",
            "cli",
        ]
    )
    assert result.returncode == 0
    assert result.stdout.strip() == (
        "per_device_train_batch_size=2 gradient_accumulation_steps=16 "
        f"deepspeed={DEEPSPEED_OFFLOAD}"
    )


def test_cli_format_json_all_fields():
    result = _run_cli(
        [
            "--gpu-type",
            "H100",
            "--nproc",
            "4",
            "--size-class",
            "3-4B",
            "--ds",
            "AndroidControl_EXP01",
            "--mode",
            "lora",
            "--format",
            "json",
        ]
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    # H100 × 3-4B → 정책상 no-offload.
    assert payload["deepspeed"] == DEEPSPEED_NO_OFFLOAD
    assert payload["offload"] is False
    assert payload["per_device_train_batch_size"] == 2
    assert payload["gradient_accumulation_steps"] == 8


def test_cli_invalid_combo_nonzero_exit_with_clear_stderr_error():
    result = _run_cli(
        [
            "--gpu-type",
            "RTX5090",
            "--nproc",
            "4",
            "--size-class",
            "7-9B",
            "--ds",
            "x",
            "--mode",
            "full",
        ]
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "nproc" in result.stderr.lower() or "unsupported" in result.stderr.lower()
