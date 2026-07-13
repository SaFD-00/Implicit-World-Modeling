"""Regression tests for implicit_world_modeling/gen_configs.py — 학습 YAML 생성기 SSoT.

두 가지를 고정한다:

(A) **as-trained 74 개 byte-exact 재현** — 커밋 ``17f49a3`` (LF 워킹트리 흡수:
    74 YAML) 의 내용과 생성기 출력이 바이트 단위로 같아야 한다. 이 74 개는 실제
    학습이 돌아간 설정이므로 생성기가 한 글자라도 바꾸면 재현성이 깨진다.
    경로·필드 순서·주석·공백까지 전부 포함한 비교다.

(B) **신규 확장분의 정책 정합** — EXP03/04 (as-trained YAML 소실) 재구성본 헤더,
    3-4B 모델 확장분의 GPU 트리오(pdbs/ga/deepspeed)가 RTX5090×2 baseline 과
    일치, 실험군별 family 자격 (EXP03/04=Qwen3-VL 전용, EXP05=Qwen3-VL-4B 배제).

또한 노트북에서 제거하기로 한 심볼 (``_YAML_GEN_DS`` allowlist,
``_PER_DEVICE_BS_BY_SIZE``, ``_derive_grad_accum``, ``GPU_TYPE == "RTX5090"``
offload swap) 이 모듈에 **부재** 함을 텍스트로 고정한다.

Run:
    pytest tests/test_gen_configs.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from implicit_world_modeling import gen_configs as gen_mod  # noqa: E402
from implicit_world_modeling import lf_registry as reg_mod  # noqa: E402
from implicit_world_modeling.gen_configs import (  # noqa: E402
    BASELINE_GPU_TYPE,
    BASELINE_NPROC,
    RECONSTRUCTED_DATASETS,
    RECONSTRUCTED_HEADER,
    check_all,
    generate_all,
)
from implicit_world_modeling.lf_registry import (  # noqa: E402
    _MODEL_CONFIG,
    CONFIGS,
    eligible_models,
)

# LF 워킹트리 유일본을 흡수한 커밋 — as-trained 74 YAML 의 정본 스냅샷.
AS_TRAINED_COMMIT = "17f49a3"
AS_TRAINED_COUNT = 74

# 커밋 YAML 의 GPU 트리오 (RTX5090×2 baseline). gpu_policy 와 독립적으로 하드코딩해
# 정책 모듈이 바뀌면 이 테스트가 잡도록 한다.
EXPECTED_PDBS = "per_device_train_batch_size: 1"
EXPECTED_GA = "gradient_accumulation_steps: 32"
EXPECTED_DS = "deepspeed: examples/deepspeed/ds_z3_offload_config.json"

MODULES = (reg_mod, gen_mod)


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    return generate_all()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(scope="module")
def as_trained() -> dict[str, str]:
    """``17f49a3`` 시점의 ``configs/train/**`` — {relpath: content}."""
    try:
        listing = _git(
            "ls-tree", "-r", "--name-only", AS_TRAINED_COMMIT, "--", "configs/train"
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"git 미가용 또는 커밋 {AS_TRAINED_COMMIT} 없음: {exc}")

    # 이 패키지는 git 루트의 서브디렉토리다 (rev-parse --show-prefix 로 접두를 얻는다) —
    # ls-tree 는 cwd 상대 경로를 주지만 `git show <rev>:<path>` 는 루트 상대 경로를 받는다.
    prefix = _git("rev-parse", "--show-prefix").strip()
    files = [ln for ln in listing.splitlines() if ln.endswith(".yaml")]
    return {
        f[len("configs/train/") :]: _git("show", f"{AS_TRAINED_COMMIT}:{prefix}{f}")
        for f in files
    }


# ============================================================
# === (A) as-trained 74 개 byte-exact 재현 ===
# ============================================================


def test_as_trained_snapshot_has_74_files(as_trained: dict[str, str]) -> None:
    assert len(as_trained) == AS_TRAINED_COUNT


def test_as_trained_yaml_reproduced_byte_exact(
    as_trained: dict[str, str], generated: dict[str, str]
) -> None:
    """74 개 전부 — 경로 존재 + 내용 바이트 동일. 허용 diff = 0."""
    missing = sorted(set(as_trained) - set(generated))
    assert not missing, f"생성기가 as-trained YAML 을 만들지 않음: {missing}"

    mismatched = {
        rel: (as_trained[rel], generated[rel])
        for rel in sorted(as_trained)
        if as_trained[rel] != generated[rel]
    }
    assert not mismatched, "as-trained YAML 재현 실패: " + ", ".join(mismatched)


def test_as_trained_gpu_trio_is_baseline(as_trained: dict[str, str]) -> None:
    """as-trained 74 개는 예외 없이 pdbs=1 / ga=32 / offload deepspeed."""
    for rel, content in as_trained.items():
        assert EXPECTED_PDBS in content, rel
        assert EXPECTED_GA in content, rel
        assert EXPECTED_DS in content, rel


# ============================================================
# === (B) 신규 확장분 ===
# ============================================================


def test_check_against_repo_configs_train_passes(generated: dict[str, str]) -> None:
    """``--check`` 등가 — 디스크 산출물과 양방향 일치 (고아 파일 없음)."""
    problems = check_all(REPO / "configs" / "train", generated)
    assert not problems, "\n".join(problems)


def test_generated_gpu_trio_is_baseline(generated: dict[str, str]) -> None:
    """신규 확장분 포함 전 YAML 이 RTX5090×2 baseline 트리오를 쓴다."""
    assert (BASELINE_GPU_TYPE, BASELINE_NPROC) == ("RTX5090", 2)
    for rel, content in generated.items():
        assert EXPECTED_PDBS in content, rel
        assert EXPECTED_GA in content, rel
        assert EXPECTED_DS in content, rel


def test_reconstructed_header_on_exp03_exp04_only(generated: dict[str, str]) -> None:
    """EXP03/EXP04 YAML 은 재구성본 헤더를 갖고, 나머지는 갖지 않는다."""
    exp0304 = [
        rel
        for rel in generated
        if rel.startswith(("IWM-AC_EXP03/", "IWM-AC_EXP04/"))
    ]
    assert exp0304, "EXP03/EXP04 YAML 이 생성되지 않았다"
    for rel in exp0304:
        assert generated[rel].startswith(RECONSTRUCTED_HEADER), rel
    for rel, content in generated.items():
        if rel not in exp0304:
            assert "reconstructed" not in content, rel
    assert RECONSTRUCTED_DATASETS == {
        "AndroidControl_EXP03",
        "AndroidControl_EXP04",
    }


def test_family_eligibility(generated: dict[str, str]) -> None:
    """실험군별 모델 자격 (AGENTS.md 하드 제약) 이 파일 집합에 반영된다."""
    # EXP03/EXP04 (0–1000 정규화) — Qwen3-VL 계열 전용.
    for ds in ("AndroidControl_EXP03", "AndroidControl_EXP04"):
        assert eligible_models(ds) == ["qwen3-vl-8b", "qwen3-vl-4b"]
    for rel in generated:
        if rel.startswith(("IWM-AC_EXP03/", "IWM-AC_EXP04/")):
            assert "qwen2.5" not in rel, f"EXP03/04 에 Qwen2.5-VL 유입: {rel}"

    # EXP05 (절대 픽셀) — qwen3-vl-4b 배제. qwen3-vl-8b 은 as-trained 예외로 유지.
    assert "qwen3-vl-4b" not in eligible_models("AndroidControl_EXP05")
    for rel in generated:
        if rel.startswith("IWM-AC_EXP05/"):
            assert "qwen3-vl-4b" not in rel, f"EXP05 에 qwen3-vl-4b 유입: {rel}"

    # EXP01/EXP02/MC — 등록 4 모델 전부.
    assert len(eligible_models("AndroidControl_EXP01_ratio37")) == 4
    assert len(eligible_models("MonkeyCollection")) == 4


def test_stage2_only_for_stage2_datasets(generated: dict[str, str]) -> None:
    """_STAGE1_ONLY (MC / EXP04 / EXP05) 는 stage2 YAML 을 만들지 않는다."""
    for rel in generated:
        if rel.startswith(("IWM-MC/", "IWM-AC_EXP04/", "IWM-AC_EXP05/")):
            assert "/stage2_" not in rel, rel


def test_diff_loss_flag_only_exp02_exp05(generated: dict[str, str]) -> None:
    """diff loss 플래그는 EXP02/EXP05 stage1 에만 (레지스트리 플래그와 일치)."""
    for rel, content in generated.items():
        has_flag = "use_diff_token_weighted_loss: true" in content
        expected = rel.startswith(("IWM-AC_EXP02/", "IWM-AC_EXP05/")) and (
            "/stage1_" in rel
        )
        assert has_flag == expected, rel


# ============================================================
# === 제거된 심볼·패턴의 부재 ===
# ============================================================


@pytest.mark.parametrize(
    "forbidden",
    [
        "_YAML_GEN_DS",  # EXP05 allowlist — 삭제됨
        "_PER_DEVICE_BS_BY_SIZE",  # → gpu_policy
        "_derive_grad_accum",  # → gpu_policy
        "lf_per_device_bs",  # → gpu_policy
        "GPU_TYPE",  # 모듈 전역 GPU_TYPE (조건부 swap 의 트리거) 없음
    ],
)
@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_removed_symbols_absent_from_namespace(module, forbidden: str) -> None:
    assert not hasattr(module, forbidden), f"{module.__name__}.{forbidden} 잔존"


def test_registry_has_no_gpu_fields() -> None:
    """레지스트리는 GPU 트리오를 들고 있지 않다 (gpu_policy 가 단일 출처)."""
    for mcfg in _MODEL_CONFIG.values():
        assert "stage1_deepspeed" not in mcfg  # 죽은 non-offload 기본값 — 삭제됨
    for ds_cfgs in CONFIGS.values():
        for ds_name, cfg in ds_cfgs.items():
            for field in (
                "deepspeed",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
            ):
                assert field not in cfg, f"{ds_name}.{field}"
            for stage_key in ("stage1_full", "stage1_lora", "stage2"):
                assert "gradient_accumulation_steps" not in cfg[stage_key]


def test_deepspeed_is_offload_for_every_gpu_type() -> None:
    """조건부 offload swap 부재의 행동 증명.

    노트북은 ``GPU_TYPE == "RTX5090"`` 일 때만 offload 로 swap 했다 — 그 분기가
    남아 있다면 A100/H100 생성물은 non-offload config 를 쓸 것이다. 전 GPU 조합에서
    offload 가 나오고, batch/grad_accum 만 정책대로 달라지는지 확인한다.
    """
    a100 = generate_all(gpu_type="A100", nproc=2)
    assert set(a100) == set(generate_all())
    for rel, content in a100.items():
        assert EXPECTED_DS in content, rel
        # A100 base pdbs=2 → ga=16. 단 EXP03/04/05 는 half-batch → pdbs=1, ga=32.
        if rel.startswith(("IWM-AC_EXP03/", "IWM-AC_EXP04/", "IWM-AC_EXP05/")):
            assert "per_device_train_batch_size: 1" in content, rel
            assert "gradient_accumulation_steps: 32" in content, rel
        else:
            assert "per_device_train_batch_size: 2" in content, rel
            assert "gradient_accumulation_steps: 16" in content, rel


def test_generated_count(generated: dict[str, str]) -> None:
    """as-trained 74 + 신규 88 = 162."""
    assert len(generated) == 162
