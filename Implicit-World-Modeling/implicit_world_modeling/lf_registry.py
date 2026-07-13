"""모델·데이터셋 레지스트리 SSoT (노트북 Cell 5 이식본).

노트북 ``implicit-world-modeling.ipynb`` Cell 5 (28KB) 가 유일본이던 레지스트리
(``_MODEL_CONFIG`` / ``_DATASET_CONFIG`` / ``_SIZE_CONFIG_AC`` /
``MODEL_FAMILY_CONFIG`` / ``CONFIGS`` 빌더) 를 임포트 가능한 모듈로 옮긴다.
하이퍼파라미터 값은 **무변경 이식**이다 — 이 모듈이 노트북과 다른 값을 내면
``configs/train/**`` 재현이 깨진다 (``tests/test_gen_configs.py`` 가 고정).

★ 노트북 대비 의도적 제거 (GPU 정책 이관)
--------------------------------------------------------------------------
아래 셋은 **이 모듈에 존재하지 않는다.** 전부 ``scripts/gpu_policy.py``
(``resolve_gpu_policy``) 로 이관됐다:

* ``_PER_DEVICE_BS_BY_SIZE`` / ``lf_per_device_bs`` / ``_derive_grad_accum``
* ``_MODEL_CONFIG[*]["stage1_deepspeed"]`` 하드코드 (한 번도 실행된 적 없는
  non-offload 죽은 기본값이었다)
* EXP03/04/05 의 ``_pd_ds = max(1, _per_device // 2)`` half-batch 예외
  (``gpu_policy._HALF_BATCH_DATASETS`` 가 같은 역할을 한다)

따라서 이 레지스트리의 ``CONFIGS`` 엔트리에는
``per_device_train_batch_size`` / ``gradient_accumulation_steps`` /
``deepspeed`` 필드가 **없다**. 이 셋은 YAML 생성 시점
(``implicit_world_modeling.gen_configs``) 에 ``resolve_gpu_policy`` 로부터
주입된다.

★ 실험군별 모델 자격 (family 자격)
--------------------------------------------------------------------------
``DATASET_MODEL_ELIGIBILITY`` 참조. AGENTS.md 의 하드 제약을 코드화한 것이다 —
EXP03/EXP04 (0–1000 정규화 좌표) 는 Qwen3-VL 계열 전용, EXP05 (절대 픽셀) 는
Qwen2.5-VL 계열 전용.
"""

from __future__ import annotations

import os

# repo 루트 (이 파일: <repo>/implicit_world_modeling/lf_registry.py).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LF_ROOT = os.path.join(BASE_DIR, "LlamaFactory")

# ============================================================
# === Model-family image-pixel configs ===
# Qwen 계열별 vision encoder patch-size (factor) 와 token budget.
# 각 family 의 max/min pixels 는 학습 YAML 의 image_max_pixels /
# image_min_pixels 필드에 그대로 주입된다. factor / merged_tokens /
# vertical_retention 은 데이터 전처리 메타이며 YAML 에는 들어가지 않는다.
#
# Token budget 은 학습 데이터셋에 따라 결정된다 (학습-추론 mismatch 방지).
#   - family default : max_tokens=2048   (모든 학습 DS 공통).
#   family 별 factor 에 따라 max_pixels = max_tokens × factor² 로 환산.
#   Qwen2/2.5-VL  (factor 28): 2048 → 1,605,632
#   Qwen3-VL/3.5  (factor 32): 2048 → 2,097,152
# ============================================================
QWEN2_VL_CONFIG = {
    "max_pixels": 1_605_632,  # 2048 x 28²
    "min_pixels": 3_136,  # 4 x 28²
    "factor": 28,
    "max_tokens": 2048,
    "merged_tokens_at_1080x2400": 510,
    "vertical_retention": 0.782,
}
QWEN2_5_VL_CONFIG = {  # Qwen2-VL 과 동일
    "max_pixels": 1_605_632,
    "min_pixels": 3_136,
    "factor": 28,
    "max_tokens": 2048,
    "merged_tokens_at_1080x2400": 510,
    "vertical_retention": 0.782,
}
QWEN3_VL_CONFIG = {
    "max_pixels": 2_097_152,  # 2048 x 32²
    "min_pixels": 4_096,  # 4 x 32²
    "factor": 32,
    "max_tokens": 2048,
    "merged_tokens_at_1080x2400": 510,
    "vertical_retention": 0.893,
}
QWEN3_5_VL_CONFIG = {  # Qwen3-VL 과 동일 (사용자 지시)
    "max_pixels": 2_097_152,
    "min_pixels": 4_096,
    "factor": 32,
    "max_tokens": 2048,
    "merged_tokens_at_1080x2400": 510,
    "vertical_retention": 0.893,
}

MODEL_FAMILY_CONFIG = {
    "qwen3-vl-8b": QWEN3_VL_CONFIG,
    "qwen3-vl-4b": QWEN3_VL_CONFIG,
    "qwen2.5-vl-7b": QWEN2_5_VL_CONFIG,
    "qwen2.5-vl-3b": QWEN2_5_VL_CONFIG,
}


def _img_cfg(short: str) -> dict:
    f = MODEL_FAMILY_CONFIG[short]
    return {"image_max_pixels": f["max_pixels"], "image_min_pixels": f["min_pixels"]}


# ============================================================
# === Model Registry (4 models) ===
# AndroidControl 계열 학습 DS 에 모델 크기 tier (7-9B / 3-4B 2단) 의 공유
# 하이퍼파라미터 (_SIZE_CONFIG_AC) 를 적용한다.
# MonkeyCollection 은 tier 미적용 — dataset baseline + per-model override 만 적용.
#
# 적용 우선순위 (merge 순서):
#   1. _DATASET_CONFIG[ds].stage{1,2}              (dataset baseline)
#   2. _SIZE_CONFIG_AC[size].stage{1, 1_lora, 2}   (AndroidControl_EXP01~EXP05 일 때만 — MC 는 미적용)
#   3. _MODEL_CONFIG[model].hparam_overrides       (계열 delta)
#
# image_max_pixels / image_min_pixels 는 MODEL_FAMILY_CONFIG 에서 자동 주입.
# cutoff_len: stage1=10000, stage2=10000 (AC_EXP01/02). AC_EXP03/04/05 만 24576 (좌표 표현 무손실).
# NOTE: 노트북에 있던 "stage1_deepspeed" / "stage1_nproc" 필드는 제거됐다 — GPU 정책은
#       scripts/gpu_policy.py 가 단일 출처다.
# ============================================================
_MODEL_CONFIG = {
    "qwen3-vl-8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "short_name": "qwen3-vl-8b",
        "template": "qwen3_vl_nothink",
        "size": "7-9B",
        **_img_cfg("qwen3-vl-8b"),
        "freeze_vision_tower": True,
        "hparam_overrides": {},
    },
    "qwen3-vl-4b": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "short_name": "qwen3-vl-4b",
        "template": "qwen3_vl_nothink",
        "size": "3-4B",
        **_img_cfg("qwen3-vl-4b"),
        "freeze_vision_tower": True,
        "hparam_overrides": {},
    },
    "qwen2.5-vl-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "short_name": "qwen2.5-vl-7b",
        "template": "qwen2_vl",
        "size": "7-9B",
        **_img_cfg("qwen2.5-vl-7b"),
        "freeze_vision_tower": True,
        "hparam_overrides": {},
    },
    "qwen2.5-vl-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "short_name": "qwen2.5-vl-3b",
        "template": "qwen2_vl",
        "size": "3-4B",
        **_img_cfg("qwen2.5-vl-3b"),
        "freeze_vision_tower": True,
        "hparam_overrides": {},
    },
}

# ============================================================
# === Dataset configs (baseline hparams per dataset) ===
# 학습 대상 DS 는 {AC_EXP01, AC_EXP02, AC_EXP03, AC_EXP04, AC_EXP05, MC}. MB 는 평가 전용.
# ============================================================
# AC_EXP01: state_pred / action_pred dual-task test. id/ood × task = 4 test 파일.
_DUAL_TASK_TEST = {
    "AndroidControl_EXP01",
    "AndroidControl_EXP02",
    "AndroidControl_EXP03",
    "AndroidControl_EXP04",
    "AndroidControl_EXP05",
}

# AC_EXP01 ratio (state:action) → split_data.py 가 산출하는 train 파일 stem.
_AC3_RATIO_FILES = {
    "ratio37": "train_3_7",  # state 30% : action 70%
    "ratio55": "train_5_5",
    "ratio73": "train_7_3",
}

_DATASET_CONFIG = {
    # AC_EXP01 (AndroidControl_EXP01) — state_pred + action_pred ratio-mix.
    # ratio 별로 별도 학습 가중치를 산출하므로 "AndroidControl_EXP01_ratio{37,55,73}" 3 개의
    # 가상 DS 로 펼친다 (lf_subfolder/ds_prefix 는 'IWM-AC_EXP01' 공유, output/hf_slug 는 ratio 별).
    # NOTE: AC_EXP01 은 Stage 1 + Stage 2 를 모두 학습한다 (_STAGE1_ONLY 에 없음).
    **{
        f"AndroidControl_EXP01_{_r}": {
            "lf_subfolder": f"IWM-AC_EXP01_{_r}",
            "ds_prefix": "IWM-AC_EXP01",  # dataset_info 의 entry 접두 (test 는 ratio 무관 공유)
            "output_prefix": "AndroidControl_EXP01/",  # ratio 별 산출물을 단일 부모 아래로.
            "model_dir_suffix": f"_{_r}",  # adapters/ / merged/ 모델 디렉토리에 ratio suffix.
            "eval_model_suffix": f"_{_r}",  # eval/ 모델 디렉토리에도 동일 ratio suffix.
            "hf_slug": f"ac-exp01-{_r}-",
            "ac3_ratio": _r,
            "ac3_train_stem": _AC3_RATIO_FILES[_r],
            "stage1": {
                "lr": "1.0e-5",
                "epochs": 3,
                "warmup_ratio": 0.03,
                "save_strategy": "epoch",
                "save_steps": None,
                "eval_strategy": "epoch",
                "eval_steps": None,
                "per_device_eval_batch_size": 4,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "lr_scheduler_type": "cosine",
            },
            # Stage 2 — ratio 3 종이 공유하는 stage2 train/test 로 학습.
            "stage2": {
                "lr": "5.0e-5",
                "epochs": 3,
                "warmup_ratio": 0.03,
                "save_strategy": "epoch",
                "save_steps": None,
                "eval_strategy": "epoch",
                "eval_steps": None,
                "per_device_eval_batch_size": 4,
                "lora_rank": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "lr_scheduler_type": "cosine",
            },
        }
        for _r in _AC3_RATIO_FILES
    },
    # AC_EXP02 — AC_EXP01 ratio73 과 동일한 학습 데이터/하이퍼파라미터.
    # 유일한 차이는 Stage 1 state prediction 에 diff loss (token-weighted SFT) 적용.
    "AndroidControl_EXP02": {
        "lf_subfolder": "IWM-AC_EXP02",
        "ds_prefix": "IWM-AC_EXP02",
        "output_prefix": "AndroidControl_EXP02/",
        "hf_slug": "ac-exp02-",
        "stage1": {
            "lr": "1.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "use_diff_token_weighted_loss": True,  # Stage 1 diff loss (state-pred 가중)
        },
        "stage2": {
            "lr": "5.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
    },
    # AC_EXP03 — AC_EXP01 ratio73 멤버십을 좌표(point) 표현으로 미러한 실험군
    # (index→x,y; scripts/mirror_experiment.py --experiment exp03). diff loss 없음.
    "AndroidControl_EXP03": {
        "lf_subfolder": "IWM-AC_EXP03",
        "ds_prefix": "IWM-AC_EXP03",
        "output_prefix": "AndroidControl_EXP03/",
        "hf_slug": "ac-exp03-",
        "stage1": {
            "lr": "1.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        "stage2": {
            "lr": "5.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
    },
    # AC_EXP04 — EXP03 와 동일 멤버십·좌표 표현이되 stage1 프롬프트만 업그레이드한 실험군
    # (scroll(direction,point) → swipe(start,end), html-style XML role, [SWIPE] 규칙).
    # 학습 hparam/cutoff/batch 은 AC_EXP03 와 동일. Stage 2 는 보류 (_STAGE1_ONLY).
    "AndroidControl_EXP04": {
        "lf_subfolder": "IWM-AC_EXP04",
        "ds_prefix": "IWM-AC_EXP04",
        "output_prefix": "AndroidControl_EXP04/",
        "hf_slug": "ac-exp04-",
        "stage1": {
            "lr": "1.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        # Stage 2 보류 — _STAGE1_ONLY guard 로 등록/YAML 생성 skip.
        "stage2": {
            "lr": "5.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
    },
    # AC_EXP05 — AC_EXP01 ratio73 멤버십을 AndroidWorld 정렬 절대 픽셀 (840×1876) 로
    # 미러한 실험군. Qwen2.5-VL 전용 (factor 28 family). Stage 2 는 보류 (_STAGE1_ONLY).
    "AndroidControl_EXP05": {
        "lf_subfolder": "IWM-AC_EXP05",
        "ds_prefix": "IWM-AC_EXP05",
        "output_prefix": "AndroidControl_EXP05/",
        "hf_slug": "ac-exp05-",
        "stage1": {
            "lr": "1.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "use_diff_token_weighted_loss": True,  # Stage 1 diff loss (state-pred 가중)
        },
        # Stage 2 보류 — _STAGE1_ONLY guard 로 등록/YAML 생성 skip.
        "stage2": {
            "lr": "5.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
    },
    "MonkeyCollection": {
        "lf_subfolder": "IWM-MC",
        "ds_prefix": "IWM-MC",
        "output_prefix": "MC/",
        "hf_slug": "mc-",
        "stage1": {
            "lr": "1.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
        # Stage 2 MC 미지원 — placeholder. `_STAGE1_ONLY` guard 로 skip.
        "stage2": {
            "lr": "5.0e-5",
            "epochs": 3,
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_steps": None,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "per_device_eval_batch_size": 4,
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
        },
    },
}

# EXP04/EXP05 stage2 보류 — 데이터 도입 시 제거.
_STAGE1_ONLY = {"MonkeyCollection", "AndroidControl_EXP04", "AndroidControl_EXP05"}

# ID/OOD split 없이 `implicit-world-modeling_stage{1,2}_test.jsonl` 단일 파일을 쓰는 DS.
# `_STAGE1_ONLY` 와 직교 — MC 는 Stage 1 만 + 단일 test.
_SINGLE_TEST = {"MonkeyCollection"}

_EVAL_ONLY_BENCHMARKS = {
    "MobiBench": {
        "ds_prefix": "IWM-MB",
        "data_dir": os.path.join(BASE_DIR, "data", "MobiBench"),
        "stage1_jsonl": "implicit-world-modeling_stage1.jsonl",
        "stage2_jsonl": "implicit-world-modeling_stage2.jsonl",
        "ds_s1_name": "IWM-MB_stage1",
        "ds_s2_name": "IWM-MB_stage2",
    },
}

# ============================================================
# === 실험군별 모델 자격 (family 자격) ===
# AGENTS.md 의 하드 제약을 코드화한다: Qwen family 의 native 좌표 규약이 세대별로
# 반전되므로 (Qwen2.5-VL = 절대 픽셀, Qwen3-VL = 0–1000 정규화, factor 28 vs 32),
# 실험군의 좌표 표현과 어긋나는 family 를 넣으면 에러 없이 grounding 만 깨진다.
#   - EXP03/EXP04 (0–1000 정규화) : Qwen3-VL 계열 전용
#   - EXP05      (절대 픽셀)      : Qwen2.5-VL 계열 전용
# 명시되지 않은 DS (EXP01/EXP02/MC) 는 등록된 4 모델 전부 자격을 갖는다.
# ============================================================
_QWEN3_VL_FAMILY = ("qwen3-vl-8b", "qwen3-vl-4b")
_QWEN2_5_VL_FAMILY = ("qwen2.5-vl-7b", "qwen2.5-vl-3b")

DATASET_MODEL_ELIGIBILITY: dict[str, frozenset[str]] = {
    "AndroidControl_EXP03": frozenset(_QWEN3_VL_FAMILY),
    "AndroidControl_EXP04": frozenset(_QWEN3_VL_FAMILY),
    # EXP05 는 Qwen2.5-VL 전용이다. 예외 없다.
    #
    # 2026-07-13 까지 qwen3-vl-8b 가 예외로 남아 있었다 (as-trained YAML 이 커밋돼
    # 있어 byte-exact 재현을 위해 자격에 넣어뒀다). 확인해 보니 **그 조합은 한 번도
    # 학습된 적이 없고** (outputs/AndroidControl_EXP05 에 8b 산출물 0건), YAML 은
    # 생성만 되고 쓰이지 않은 것이었다. 보존할 as-trained 가 애초에 없었다.
    # 게다가 이중 mismatch 다 — 좌표 규약(Qwen3-VL = 0~1000 정규화 vs EXP05 = 절대
    # 픽셀)뿐 아니라 image budget 도 어긋난다 (그 YAML 은 2097152, EXP05 데이터는
    # 1605632 = Qwen2.5-VL-3B 기준). 돌아가긴 하지만 grounding 이 조용히 깨진다.
    # → 자격에서 제거하고 YAML 도 삭제했다. AGENTS.md 의 하드 제약과 코드를 일치시킨다.
    "AndroidControl_EXP05": frozenset(_QWEN2_5_VL_FAMILY),
}


def eligible_models(ds_name: str) -> list[str]:
    """``ds_name`` 학습 자격이 있는 모델 키 목록 (_MODEL_CONFIG 등록 순서)."""
    allowed = DATASET_MODEL_ELIGIBILITY.get(ds_name)
    if allowed is None:
        return list(_MODEL_CONFIG)
    return [m for m in _MODEL_CONFIG if m in allowed]


# ============================================================
# === Size-tier shared hparams (AC_EXP01~EXP05) ===
# 정책: EXP01/EXP02 실측 어댑터와 동일조건을 보존해 diff loss 순효과만 비교한다 —
# size-tier 로 baseline 을 덮지 않고 dataset baseline 을 그대로 쓴다 (tier dict 비움).
# ============================================================
_SIZE_CONFIG_AC = {
    "7-9B": {
        "stage1": {},
        "stage1_lora": {},
        "stage2": {},
    },
    "3-4B": {
        "stage1": {},
        "stage1_lora": {},
        "stage2": {},
    },
}

# ============================================================
# === Model ordering for execution cells ===
# ============================================================
MODEL_ORDER = [
    ("qwen3-vl-8b", "Qwen3-VL-8B"),
    ("qwen3-vl-4b", "Qwen3-VL-4B"),
    ("qwen2.5-vl-7b", "Qwen2.5-VL-7B"),
    ("qwen2.5-vl-3b", "Qwen2.5-VL-3B"),
]
DS_ORDER = [("MC", "MonkeyCollection")]

# size-tier 를 적용하는 DS (MC 는 미적용).
_TIERED_DS_PREFIX = "AndroidControl_EXP"

# 긴 표현(좌표/절대픽셀) 실험군은 시퀀스가 ~2.5x 길어 cutoff_len 상향(무손실).
_LONG_CUTOFF_DS = (
    "AndroidControl_EXP03",
    "AndroidControl_EXP04",
    "AndroidControl_EXP05",
)


# ============================================================
# === Build CONFIGS: CONFIGS[model_key][ds_name] ===
# ============================================================
def build_configs() -> dict[str, dict[str, dict]]:
    """노트북 Cell 5 의 ``CONFIGS`` 빌더 이식본 (GPU/batch 필드 제외).

    반환 dict 에는 ``per_device_train_batch_size`` /
    ``gradient_accumulation_steps`` / ``deepspeed`` 가 **없다** —
    ``scripts.gpu_policy.resolve_gpu_policy`` 가 그 셋의 단일 출처다.
    """
    configs: dict[str, dict[str, dict]] = {}
    for model_key, mcfg in _MODEL_CONFIG.items():
        configs[model_key] = {}
        overrides = mcfg.get("hparam_overrides", {})
        size = mcfg["size"]
        for ds_name, cfg in _DATASET_CONFIG.items():
            c = dict(cfg)
            # image budget: family default 위에 dataset image_overrides 를 덮어쓴다.
            # override 키는 token 단위 ("max_tokens", "min_tokens") 또는 px 단위
            # ("image_max_pixels", "image_min_pixels"). token 키는 family factor² 로 환산.
            factor = MODEL_FAMILY_CONFIG[model_key]["factor"]
            img = dict(_img_cfg(model_key))
            for ok, ov in cfg.get("image_overrides", {}).items():
                if ok == "max_tokens":
                    img["image_max_pixels"] = ov * factor * factor
                elif ok == "min_tokens":
                    img["image_min_pixels"] = ov * factor * factor
                else:
                    img[ok] = ov
            c["image_max_pixels"] = img["image_max_pixels"]
            c["image_min_pixels"] = img["image_min_pixels"]
            c["model_key"] = model_key
            c["model_id"] = mcfg["model_id"]
            c["short_name"] = mcfg["short_name"]
            c["template"] = mcfg["template"]
            c["model_config"] = mcfg
            c["dataset_name"] = ds_name
            c["data_dir"] = os.path.join(BASE_DIR, "data", ds_name)
            c["cutoff_len"] = 24576 if ds_name in _LONG_CUTOFF_DS else 10000

            # AC_EXP01 ratio variant 들은 동일한 data/AndroidControl_EXP01 디렉토리를 공유.
            if ds_name.startswith("AndroidControl_EXP01"):
                c["data_dir"] = os.path.join(BASE_DIR, "data", "AndroidControl_EXP01")

            # dual-task DS: train 은 ratio 별 분기, test 는 task × split 4 파일.
            if ds_name in _DUAL_TASK_TEST or ds_name.startswith("AndroidControl_EXP01"):
                r = cfg.get("ac3_ratio", "")
                c["ds_s1_train"] = (
                    f"{c['ds_prefix']}_stage1_train_{r}"
                    if r
                    else f"{c['ds_prefix']}_stage1_train"
                )
                c["ds_s1_test_id_state"] = f"{c['ds_prefix']}_stage1_test_id_state"
                c["ds_s1_test_ood_state"] = f"{c['ds_prefix']}_stage1_test_ood_state"
                c["ds_s1_test_id_action"] = f"{c['ds_prefix']}_stage1_test_id_action"
                c["ds_s1_test_ood_action"] = f"{c['ds_prefix']}_stage1_test_ood_action"
                # Backwards-compat aliases: test_id 는 state 를 default 로 노출.
                c["ds_s1_test_id"] = c["ds_s1_test_id_state"]
                c["ds_s1_test_ood"] = c["ds_s1_test_ood_state"]
                c["ds_s1_test"] = c["ds_s1_test_id"]
            elif ds_name in _SINGLE_TEST:
                c["ds_s1_train"] = f"{c['ds_prefix']}_stage1_train"
                c["ds_s1_test"] = f"{c['ds_prefix']}_stage1_test"
            else:
                c["ds_s1_train"] = f"{c['ds_prefix']}_stage1_train"
                c["ds_s1_test_id"] = f"{c['ds_prefix']}_stage1_test_id"
                c["ds_s1_test_ood"] = f"{c['ds_prefix']}_stage1_test_ood"
                c["ds_s1_test"] = c["ds_s1_test_id"]

            c["ds_s2_train"] = f"{c['ds_prefix']}_stage2_train"
            if ds_name in _SINGLE_TEST:
                c["ds_s2_test"] = f"{c['ds_prefix']}_stage2_test"
            else:
                c["ds_s2_test_id"] = f"{c['ds_prefix']}_stage2_test_id"
                c["ds_s2_test_ood"] = f"{c['ds_prefix']}_stage2_test_ood"
                c["ds_s2_test"] = c["ds_s2_test_id"]

            c["hf_s1_model_full"] = (
                f"SaFD-00/{mcfg['short_name']}-{c['hf_slug']}stage1-full-world-model"
            )
            c["hf_s1_model_lora"] = (
                f"SaFD-00/{mcfg['short_name']}-{c['hf_slug']}stage1-lora-world-model"
            )
            c["hf_s1_model"] = c["hf_s1_model_full"]

            c["hf_s2_base"] = f"SaFD-00/{mcfg['short_name']}-{c['hf_slug']}stage2-base"
            c["hf_s2_world_full"] = (
                f"SaFD-00/{mcfg['short_name']}-{c['hf_slug']}stage2-full-world-model"
            )
            c["hf_s2_world_lora"] = (
                f"SaFD-00/{mcfg['short_name']}-{c['hf_slug']}stage2-lora-world-model"
            )
            c["hf_s2_world"] = c["hf_s2_world_full"]

            ds_code = c["output_prefix"].rstrip("/")
            mshort = mcfg["short_name"]
            # AC_EXP01 ratio variant: model dir 에 ratio suffix 를 붙여 충돌 방지.
            mshort_dir = mshort + c.get("model_dir_suffix", "")

            c["save_s1_full"] = (
                f"../outputs/{ds_code}/adapters/{mshort_dir}_stage1_full_world-model"
            )
            c["save_s1_lora"] = (
                f"../outputs/{ds_code}/adapters/{mshort_dir}_stage1_lora_world-model"
            )
            c["out_s1_merged_full"] = (
                f"../outputs/{ds_code}/merged/{mshort_dir}_stage1_full_world-model"
            )
            c["out_s1_merged_lora"] = (
                f"../outputs/{ds_code}/merged/{mshort_dir}_stage1_lora_world-model"
            )
            c["save_s1"] = c["save_s1_full"]
            c["out_s1_merged"] = c["out_s1_merged_full"]

            for m2 in ("full", "lora"):
                c[f"save_s2_{m2}_base"] = (
                    f"../outputs/{ds_code}/adapters/{mshort_dir}_stage2_{m2}_base"
                )
                c[f"save_s2_{m2}_world_from_full"] = (
                    f"../outputs/{ds_code}/adapters/{mshort_dir}_stage2_{m2}"
                    "_world-model_from_full-ep__STAGE1_EPOCH__"
                )
                c[f"save_s2_{m2}_world_from_lora"] = (
                    f"../outputs/{ds_code}/adapters/{mshort_dir}_stage2_{m2}"
                    "_world-model_from_lora-ep__STAGE1_EPOCH__"
                )
                c[f"out_s2_merged_{m2}_base"] = (
                    f"../outputs/{ds_code}/merged/{mshort_dir}_stage2_{m2}_base"
                )
                c[f"out_s2_merged_{m2}_world_from_full"] = (
                    f"../outputs/{ds_code}/merged/{mshort_dir}_stage2_{m2}"
                    "_world-model_from_full-ep__STAGE1_EPOCH__"
                )
                c[f"out_s2_merged_{m2}_world_from_lora"] = (
                    f"../outputs/{ds_code}/merged/{mshort_dir}_stage2_{m2}"
                    "_world-model_from_lora-ep__STAGE1_EPOCH__"
                )
            c["save_s2_base"] = c["save_s2_lora_base"]
            c["save_s2_world_from_full"] = c["save_s2_lora_world_from_full"]
            c["save_s2_world_from_lora"] = c["save_s2_lora_world_from_lora"]
            c["out_s2_merged_base"] = c["out_s2_merged_lora_base"]
            c["out_s2_merged_world_from_full"] = c["out_s2_merged_lora_world_from_full"]
            c["out_s2_merged_world_from_lora"] = c["out_s2_merged_lora_world_from_lora"]
            c["save_s2_world"] = c["save_s2_world_from_full"]
            c["out_s2_merged_world"] = c["out_s2_merged_world_from_full"]

            tier = (
                _SIZE_CONFIG_AC[size]
                if ds_name.startswith(_TIERED_DS_PREFIX)
                else None
            )

            s1_full = dict(c["stage1"])
            if tier is not None:
                s1_full.update(tier.get("stage1", {}))
            s1_full.update(overrides.get("stage1", {}))

            s1_lora = dict(s1_full)
            if tier is not None:
                s1_lora.update(tier.get("stage1_lora", {}))
            s1_lora.update(overrides.get("stage1_lora", {}))

            s2 = dict(c["stage2"])
            if tier is not None:
                s2.update(tier.get("stage2", {}))
            s2.update(overrides.get("stage2", {}))

            c["stage1_full"] = s1_full
            c["stage1_lora"] = s1_lora
            c["stage1"] = s1_full
            c["stage2"] = s2
            c["stage1_only"] = ds_name in _STAGE1_ONLY
            c["size"] = size

            configs[model_key][ds_name] = c

    return configs


CONFIGS = build_configs()
