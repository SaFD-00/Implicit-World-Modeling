# GUI-Model

모바일 GUI World Modeling 이 Action Prediction 성능에 미치는 영향을 검증하는 **2-stage fine-tuning 파이프라인**. 8 개 Qwen 계열 Vision-Language 모델 × 3 개 학습 데이터셋 매트릭스를 단일 conda env (`gui-model`) + 저장소 내부 [`LlamaFactory/`](./LlamaFactory) 백엔드로 운영한다. 노트북 [`gui-model.ipynb`](./gui-model.ipynb) 가 기준 실행 경로이고 [`scripts/`](./scripts) 가 반복 실행용 자동화 레이어다.

## 개요

- **Stage 1** — `screenshot + UI XML + action -> next UI XML` (World Modeling)
- **Stage 2** — `screenshot + UI XML + task -> action JSON` (Action Prediction)
- **비교 실험**
  - `base` — base model 의 zero-shot 성능
  - `stage2` — base model → Stage 2 SFT
  - `stage1+stage2` — Stage 1 merged (world model) → Stage 2 SFT
- **파이프라인 흐름** (Stage 1 / Stage 2 공통): `train → merge → eval`. eval 은 HF Hub 에 push 된 merged repo 만 pull 하므로 학습 머신이 아닌 환경에서도 재실행 가능하다.

## 지원 모델 (8 개, 모두 Qwen 계열 multimodal)

| # | model_id | short_name | template | size |
|---|----------|------------|----------|------|
| 1 | `Qwen/Qwen2-VL-2B-Instruct` | `qwen2-vl-2b` | `qwen2_vl` | 2B |
| 2 | `Qwen/Qwen2-VL-7B-Instruct` | `qwen2-vl-7b` | `qwen2_vl` | 7-9B |
| 3 | `Qwen/Qwen2.5-VL-3B-Instruct` | `qwen2.5-vl-3b` | `qwen2_vl` | 3-4B |
| 4 | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2.5-vl-7b` | `qwen2_vl` | 7-9B |
| 5 | `Qwen/Qwen3-VL-4B-Instruct` | `qwen3-vl-4b` | `qwen3_vl_nothink` | 3-4B |
| 6 | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3-vl-8b` | `qwen3_vl_nothink` | 7-9B |
| 7 | `Qwen/Qwen3.5-4B-Base` | `qwen3.5-4b-base` | `qwen3_5_nothink` | 3-4B |
| 8 | `Qwen/Qwen3.5-9B-Base` | `qwen3.5-9b-base` | `qwen3_5_nothink` | 7-9B |

> Qwen3.5-Base 는 LlamaFactory 가 multimodal `hf_model_type=qwen3_5` 로 인식하며 (Qwen3-VL 과 동일 그룹), `template=qwen3_5_nothink` 로 학습한다. 추론 시 `vllm_infer.py` 에 `--enable_thinking False` 가 자동 주입된다.

### 모델 family 별 image budget

학습 YAML 의 `image_max_pixels` / `image_min_pixels` 는 vision encoder patch-size (factor) 와 **학습 데이터셋** 으로 결정된다 (`gui-model.ipynb` Cell 5 의 `MODEL_FAMILY_CONFIG` + `_DATASET_CONFIG[ds]["image_overrides"]`). 학습된 모델은 어떤 ds 를 평가하든 **학습 시 budget** 을 그대로 사용한다 (학습-추론 mismatch 방지).

| family | patch | merge | factor | min_tokens | min_pixels |
|--------|-------|-------|--------|-----------|------------|
| Qwen2-VL · Qwen2.5-VL | 14 | 2 | 28 | 4 | 3,136 |
| Qwen3-VL · Qwen3.5 | 16 | 2 | 32 | 4 | 4,096 |

| 학습 DS | max_tokens | Qwen2/2.5-VL `max_pixels` | Qwen3-VL · Qwen3.5 `max_pixels` |
|---|---|---|---|
| AC, MC | 2,048 (family default) | 1,605,632 (= 2048 × 28²) | 2,097,152 (= 2048 × 32²) |
| AC_2 | **5,400 (dataset override)** | 4,233,600 (= 5400 × 28²) | 5,529,600 (= 5400 × 32²) |

> 평가측 `scripts/_common.sh::build_infer_cmd` 가 `TRAIN_DATASET` 환경변수로 학습 DS 를 식별해 동일 budget 을 적용한다. 즉 AC_2 로 학습한 모델은 AC / MC / MB 평가에도 5400-token 예산을 쓴다.

## 디렉토리 구조

```
GUI-Model/
├── gui-model.ipynb               # 단일 노트북 — 파이프라인 기준 실행 경로
├── scripts/
│   ├── _common.sh                # 공통 path / dataset 매핑 / 모델 레지스트리 / HF repo id 헬퍼
│   ├── stage{1,2}_{train,eval,merge}.sh
│   ├── _hungarian_eval.py        # Stage 1 metric (single-pair / ID·OOD 모드)
│   ├── _action_eval.py           # Stage 2 metric (single-pair / ID·OOD 모드, Step Accuracy)
│   ├── eval_viewer.py            # Stage 1/2 산출물 비교 HTML 빌더
│   ├── split_data.py             # AC/MC split (AC_2 는 사전 분할이라 미처리)
│   ├── extract_androidcontrol_metadata.py   # AC episodes_meta.jsonl 생성
│   └── extract_androidcontrol_images.py     # AC GCS TFRecord → PNG (TF 의존 없음)
├── data/                         # AC / AC_2 / MC / MB
├── LlamaFactory/                 # 학습/추론 엔진 (clone)
│   ├── examples/custom/GUI-Model-{AC,AC_2,MC}/  # 노트북이 생성한 학습 YAML
│   └── scripts/vllm_infer.py     # 추론 도구
├── gui_model/                    # 배포용 스텁 (핵심 로직 없음)
├── tests/test_action_eval.py     # Stage 2 메트릭 회귀 테스트 (48 케이스)
├── pyproject.toml                # 공통 deps + extras["llamafactory"] (uv.lock 으로 재현)
├── uv.lock                       # uv sync 가 생성/관리하는 잠금 파일 (커밋 대상)
├── .python-version               # 3.12 (uv 가 자동 선택)
├── .env.example                  # HF_TOKEN, NPROC_PER_NODE, GPU_TYPE
├── README.md                     # (this file)
├── ARCHITECTURE.md               # 시스템 레퍼런스
└── AGENTS.md                     # 에이전트 작업 지침
```

## 환경 설치

단일 `.venv` (uv 관리) 에 `pyproject.toml` 의 공통 deps + `[project.optional-dependencies] llamafactory` 를 설치한다. 서브프로젝트 `./LlamaFactory` 는 `[tool.uv.sources]` 의 editable path source 로 함께 해소되며 (별도 단계 불필요), `transformers` 는 우리 extras pin (`>=4.57.1,<4.58`) 과 LlamaFactory 의 `<=5.2.0` 가 겹치는 4.57.x 구간에서 한 번에 풀린다.

```bash
# 1) uv 설치 (한 번만)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) GUI-Model 설치
cd /path/to/GUI-Model
uv venv --python 3.12          # .python-version 따라 자동 선택
uv sync --extra llamafactory   # LlamaFactory editable + extras 함께 해소

# 3) 활성화 (scripts/*.sh 실행 전 필수)
source .venv/bin/activate
```

deepspeed · vllm · bitsandbytes 모두 단일 `.venv` 에 설치된다. 학습/export 엔진은 `llamafactory-cli train` / `llamafactory-cli export`.

`uv.lock` 은 함께 커밋한다. CI / 다른 머신에서는 `uv sync --frozen --extra llamafactory` 로 동일 환경을 복구한다.

### `.env` 변수

`.env.example` 를 복사해 `.env` 를 만든다. 노트북 Cell 5 와 `scripts/_common.sh` 가 이 파일을 source 한다.

| 변수 | 기본 | 허용값 | 설명 |
|------|------|--------|------|
| `HF_TOKEN` | — | (string) | HF Hub push/pull |
| `NPROC_PER_NODE` | `2` | `1`, `2`, `4`, `8` | node 당 GPU 수 (single node, torchrun world size) |
| `GPU_TYPE` | `H100` | `RTX5090`, `A100`, `H100` | GPU 종류. (모델 size × GPU) 표로 `per_device_train_batch_size` 결정 |

#### `per_device_train_batch_size` (size × GPU)

| 모델 size | RTX5090 (32GB) | A100 (80GB) | H100 (80GB) |
|-----------|----------------|-------------|-------------|
| 2B        | 4              | 8           | 8           |
| 3-4B      | 2              | 4           | 4           |
| 7-9B      | 1              | 2           | 2           |

`GLOBAL_BATCH_SIZE = 64` 를 유지하기 위해 `gradient_accumulation_steps = 64 / (per_device × NPROC_PER_NODE)` 가 자동 역계산된다 (정수가 아니면 `ValueError`). 위 표 값은 `NPROC ∈ {1, 2, 4, 8}` 모두에서 정수가 된다.

> `.env` 의 `GPU_TYPE` / `NPROC_PER_NODE` 를 수정한 뒤에는 노트북 Cell 5 (CONFIGS) 와 Stage 1/2 YAML 생성 셀 (Cell 8 / 10) 을 다시 실행해야 새 값이 YAML 에 반영된다.

### 전제

- Python 3.11 이상, 3.13 미만 (`.python-version` 은 3.12 고정 — LlamaFactory 서브프로젝트가 `>=3.11.0` 을 요구하므로 우리도 동일 하한)
- bash 4+ (`scripts/_common.sh` 기준)
- `transformers>=4.57.1,<4.58` 로 고정 — 변경 시 [`pyproject.toml`](./pyproject.toml) 의 `dependencies` / `[project.optional-dependencies] llamafactory` 와 주석을 함께 갱신한다. **서브프로젝트 `LlamaFactory/pyproject.toml` 은 건드리지 않는다.**

### PATH 정책

`scripts/_common.sh` 는 활성 Python env 의 `bin/` 을 PATH 최상단에 고정해 user-site 의 낡은 CLI (`accelerate` 등) 를 가린다. 우선순위는 `VIRTUAL_ENV` (uv `.venv`) → `CONDA_PREFIX` (conda fallback) 순. 둘 다 미활성 상태에서 `scripts/*.sh` 를 실행하면 즉시 중단된다 — 먼저 `source .venv/bin/activate`.

## 데이터 준비

학습 대상 DS 는 **AC**, **AC_2**, **MC** 세 가지. **MB** 는 평가 전용이므로 split 하지 않는다.

```
data/
├── AndroidControl/                 # 학습 + 평가 (Stage 1 + 2, ID/OOD)
├── AndroidControl_2/                # 학습 + 평가 (Stage 1 + 2, 단일 test, 사전 분할)
│   # NOTE: images/ 디렉토리 없음 — JSONL `images` 가 "AndroidControl/images/..." 참조
├── MonkeyCollection/                # Stage 1 학습 + 평가 (Stage 2 자동 skip)
└── MobiBench/                       # 평가 전용 (단일 파일 stage{1,2}.jsonl)
```

### 1) AndroidControl 준비

```bash
# (a) GCS TFRecord 에서 스크린샷 추출 (TF 의존 없음, urllib 만 사용)
python scripts/extract_androidcontrol_images.py \
    --output data/AndroidControl/images/ --skip-existing --verbose

# (b) episodes_meta.jsonl 생성 (primary_app = 전경 앱 package_name)
#     pip install android-env 필요 (proto 디코딩 용)
python scripts/extract_androidcontrol_metadata.py \
    --output data/AndroidControl/episodes_meta.jsonl

# (c) Stage 1 + Stage 2 ID/OOD split (단일 partition 공유)
python scripts/split_data.py --dataset AndroidControl
#   → gui-model_stage{1,2}_{train,test_id,test_ood}.jsonl
```

### 2) AndroidControl_2 준비

사전 분할 데이터로 제공되므로 `split_data.py` 를 다시 돌리지 않는다. `data/AndroidControl_2/gui-model_stage{1,2}_{train,test}.jsonl` 이 이미 있어야 한다. JSONL `images` 가 `AndroidControl/images/...` 를 참조하므로 **AC images 가 살아 있어야 한다.**

### 2.5) AndroidControl_3 준비 (Stage 1 state+action ratio mix + Stage 2 ID/OOD split)

```bash
# (선행) mm-expanded length > cutoff_len 인 샘플을 제거해 _filtered.jsonl 산출.
#       split 은 항상 _filtered 만 입력으로 사용한다 (Qwen3-VL get_rope_index broadcast 회피).
python scripts/filter_long_samples.py --dataset AC_3

# Stage 1 ratio mix + Stage 2 ID/OOD split 을 한 번에 산출.
python scripts/split_data.py --dataset AC_3 --ac3-ratios 3:7,5:5,7:3 --ac3-train-total 50000
# Stage 1 → gui-model_stage1_train_{3_7,5_5,7_3}.jsonl
#         + gui-model_stage1_test_{id,ood}_{state,action}_pred.jsonl
# Stage 2 → gui-model_stage2_{train,test_id,test_ood}.jsonl   (15K / 3K / 3K, Stage 1 action_pred app partition 공유)
```

ratio 별로 **별개의 학습 가중치** 가 산출되므로 train/merge/eval 모두 `--dataset AC_3` 한 번에 ratio 3 종을 sweep 한다 (`--ac3-ratios r55,r73` 로 부분 실행 가능). 평가는 ratio 단일 (`--ac3-ratio r55` 기본) 로 고정 + state/action **두 task 를 각각 채점** (`_hungarian_eval.py` / `_action_eval.py`). Stage 2 split 파일이 산출되지만 **노트북/스크립트의 `_STAGE1_ONLY` 가드가 `AndroidControl_3_r{37,55,73}` 3 키를 포함하고 있어 AC_3 의 Stage 2 학습/평가 파이프라인은 현재 skip 된다** — 데이터만 디스크에 존재하는 상태.

### 3) MonkeyCollection 준비

```bash
python scripts/split_data.py --dataset MonkeyCollection
# → gui-model_stage1_{train,test}.jsonl  (random 95:5, Stage 2 자동 skip)
```

### 4) MobiBench 준비

split 불필요. `data/MobiBench/gui-model_stage{1,2}.jsonl` 두 단일 파일만 있으면 평가가 성립한다. `dataset_info.json` 등록은 `scripts/_common.sh::ensure_eval_only_dataset_info()` 가 `stage{1,2}_eval.sh` source 시점에 idempotent 하게 보장한다.

### 분할 규칙 요약

- **AC App partition**: 앱 집합을 셔플 → OOD 버킷이 `--stage2-test-ood-size` 를 채울 때까지 먼저 할당, 나머지는 ID 버킷. 같은 (id_apps, ood_apps) 를 Stage 1 / Stage 2 가 공유.
- **Stage 1 (AC)**: 위 partition 으로 entries 라우팅 → 각 풀에서 random sample (action-type stratification 미적용).
- **Stage 2 (AC)**: 같은 partition 에서 action-type **stratified** 샘플링 (largest-remainder).
- **Stage 1 (MC)**: 메타 없음 → random split (`--stage1-ratio`, 기본 0.95).
- **Stage 1 (AC_3)**: state_pred (random) + action_pred (action-type stratified) 를 비율 (`--ac3-ratios`, 기본 `7:3,3:7,5:5` — 콜론 구분) 로 혼합. id/ood 는 task 별로 분리된 4 파일.
- **Stage 2 (AC_3)**: Stage 1 의 action_pred app partition 을 그대로 재사용해 action_type stratified 샘플링 (`--stage2-train-size` 기본 15K, `--stage2-test-{id,ood}-size` 기본 3K). split 결과는 디스크에만 남고 학습/평가는 `_STAGE1_ONLY` 가드로 skip.
- **AC_2 / MB**: split 불필요.

## 실행 방법

### 1. 노트북 경로

`gui-model` env 에서 [`gui-model.ipynb`](./gui-model.ipynb) 를 섹션 순서대로 실행한다.

| Section | 내용 |
|---------|------|
| 0 | 환경, dataset / 모델 / family / size config 정의, Stage 1 · Stage 2 학습 YAML 일괄 생성 |
| 1-2 | `dataset_info.json` 등록 (AC: 6 entry, AC_2: 4 entry, MC: 2 entry, MB: eval-only 2 entry) |
| 3 | Stage 1 SFT 학습 (`qwen3-vl-8b` + Full FT, AC · AC_2 · AC_3) |
| 4 | Stage 1 merge (`qwen3-vl-8b` + Full FT, 모든 epoch local merge + HF Hub push) |
| 5 | Stage 1 평가 (HF Hub merged sweep, EVAL_DS = AC, AC_2, MC, MB) |
| 6 | Stage 2 SFT 학습 (`qwen3-vl-8b` + LoRA, AC · AC_2) |
| 7 | Stage 2 merge (`qwen3-vl-8b` + LoRA, variant × 모든 epoch + HF push) |
| 8 | Stage 2 평가 (ID + OOD 동시 sweep, `action_metrics.json` 3 섹션) |

> Section 3 / 4 / 6 / 7 은 **단일 변형 walkthrough** 다 — Stage 1 = `qwen3-vl-8b` + `full`, Stage 2 = `qwen3-vl-8b` + `lora`. 다른 모델 / 모드 / 데이터셋은 cell 을 추가하지 말고 shell 호출에서 `--model` / `--stage1-mode` / `--stage2-mode` / `--dataset` 인자만 바꿔 실행한다 (등록된 모델은 [`scripts/_common.sh::MODELS`](./scripts/_common.sh) 참조). 매트릭스 sweep 은 `--model all` / `--dataset all` 사용. Section 5 / 8 의 평가 cell 은 base + variant matrix 정의와 plot 코드를 보존한다.

### 2. shell script 경로

shell script 는 노트북에서 한 번 생성된 **학습 YAML** 과 `LlamaFactory/data/dataset_info.json` 이 이미 있다는 전제에서 동작한다. **Stage 1 eval 은 YAML 을 사용하지 않고 HF Hub merged repo 를 직접 sweep 한다.**

#### 학습 / merge

```bash
# Stage 1 Full FT — train → merge → eval (AC 학습, 4 DS 교차 평가)
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC --eval-datasets AC,AC_2,MC,MB \
     --variants base,full_world_model --epochs 1,2,3

# Stage 1 LoRA — MC 학습
bash scripts/stage1_train.sh --model qwen3-vl-4b --dataset MC --stage1-mode lora
bash scripts/stage1_merge.sh --model qwen3-vl-4b --dataset MC --stage1-mode lora
bash scripts/stage1_eval.sh  --model qwen3-vl-4b --train-dataset MC --eval-datasets AC,MC,MB \
     --stage1-mode lora --variants base,lora_world_model --epochs 1,2,3

# Stage 1 — AC_2 학습 (token budget 5400 자동 적용)
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_2 --stage1-mode lora
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_2 --stage1-mode lora
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC_2 --eval-datasets AC_2,MB \
     --stage1-mode lora --variants base,lora_world_model --epochs 1,2,3

# Stage 1 — AC_3 학습 (state_pred + action_pred ratio 3 종 자동 sweep; Stage 2 데이터는 split 산출되지만 학습은 `_STAGE1_ONLY` 게이트로 skip)
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_3 --stage1-mode full
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_3 --stage1-mode full
# 평가는 ratio 1 개 단위 (state, action 각각 채점). r55 기본.
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC_3 --ac3-ratio r55 \
     --eval-datasets AC_3 --variants base,full_world_model --epochs 1,2,3
# 부분 ratio sweep + 외부 벤치마크 교차 평가
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_3 --ac3-ratios r55,r73 --stage1-mode lora
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC_3 --ac3-ratio r73 \
     --eval-datasets AC_3,AC,MB

# 전체 모델 × 전체 DS 일괄 sweep (--model / --dataset 생략)
bash scripts/stage1_train.sh --stage1-mode full   # 8 모델 × {AC, AC_2, MC} (AC_3 는 명시적)
bash scripts/stage1_merge.sh --stage1-mode full   # 학습 안 된 슬롯은 [WARN] SKIP

# Stage 2 — AC, AC_2 지원 (MC 는 Stage 2 데이터/YAML 없음 → 거절)
bash scripts/stage2_train.sh --model qwen3-vl-8b --dataset AC \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora
bash scripts/stage2_merge.sh --model qwen3-vl-8b --dataset AC \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora
bash scripts/stage2_eval.sh  --model qwen3-vl-8b --train-dataset AC --eval-datasets AC,AC_2,MB \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
     --variants base,lora_base,lora_world_model --epochs 1,2,3
```

#### 평가 — `--train-dataset` 으로 HF repo, `--eval-datasets` 로 test 셋을 분리

| 스크립트 | `--train-dataset` 허용 | `--eval-datasets` 허용 | 기본 variants |
|---|---|---|---|
| `stage1_eval.sh` | `AC` / `AC_2` / `AC_3` / `MC` | `AC` / `AC_2` / `AC_3` / `MC` / `MB` | `base, full_world_model, lora_world_model` |
| `stage2_eval.sh` | `AC` / `AC_2` (MC, AC_3 거절) | `AC` / `AC_2` / `MB` | `base, full_base, lora_base, full_world_model, lora_world_model` |

> **AC_3 평가 분기**: `--eval-datasets AC_3` 는 state_pred / action_pred 두 task 를 각각 독립 채점한다.
> 산출 디렉토리는 `on-AC_3-state/hungarian_metrics.json` (state) + `on-AC_3-action/action_metrics.json` (action) 으로 분리되며 각자 독립 skip marker 를 갖는다. AC_3 모델 평가는 ratio 가 정확히 한 개여야 하므로 `--ac3-ratio {r37|r55|r73}` (기본 `r55`) 로 골라야 한다.

`--epochs LIST` (콤마 구분, 기본 `1,2,3`) · `--variants LIST` (콤마 구분) · `--stage1-mode {full|lora}` / `--stage2-mode {full|lora}` / `--stage1-epoch N` (world-model variant 의 상류 Stage 1 epoch).

> **재실행 시 skip**: 각 unit 의 marker (`hungarian_metrics.json` / `action_metrics.json`) 가 이미 존재하면 `[=] ... skip (already done): ...` 로그만 남긴다. 강제 재평가는 해당 marker 를 `rm` 후 재실행. Stage 1 의 `without_open_app` 산출물은 정규 metric 과 별도 marker 로 독립 skip.

#### Stage 1/2 메트릭 정의

- **Stage 1**: Hungarian F1 (UI element matching) + BLEU + ROUGE-L. 산출 파일 `hungarian_metrics.json`.
- **Stage 2**: Step Accuracy `correct = parse_ok ∧ type==gt.type ∧ field_match(type)`. 산출 파일 `action_metrics.json` (`overall` / `in_domain` / `out_of_domain` 3 섹션 또는 `overall` 1 섹션).
- 자세한 정의는 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §6.

### 평가 결과 시각 비교 (`scripts/eval_viewer.py`)

`stage{1,2}_eval.sh` 산출물을 행 정렬된 HTML 로 비교한다.

```bash
python scripts/eval_viewer.py
python scripts/eval_viewer.py --stages 1 --datasets on-AC
python scripts/eval_viewer.py --data-dir AC_2 --model qwen3-vl-8b qwen2.5-vl-7b
python scripts/eval_viewer.py --stages 1 --variants base full_world_model/epoch-3
```

`--data-dir {AC|AC_2}` 로 데이터/산출물 루트를 선택 (기본 `AC`). 산출 위치: `outputs/{AC|AC_2}/eval/{MODEL}/stage{1,2}_eval/{pairs_on-AC.html, pairs_on-MB.html, pairs_summary.md}`.

## 산출물

```
GUI-Model/outputs/{AC|AC_2|AC_3|MC}/      # AC_3 의 ratio (r37/r55/r73) 는 별도 디렉토리가 아니라 아래 model dir 의 suffix 로 운반된다.
├── adapters/
│   ├── {model}_stage1_{full,lora}_world-model/                                  # Stage 1 체크포인트
│   ├── {model}_stage2_{full,lora}_base/                                         # Stage 2 base
│   └── {model}_stage2_{full,lora}_world-model_from_{full,lora}-ep{E1}/          # Stage 2 world-model (E1=stage1 upstream epoch)
├── eval/{model}/
│   ├── stage1_eval/{base, {full,lora}_world-model/epoch-{E}}/   # 각 variant 안에 on-{EVAL_DS}/ + on-{EVAL_DS}-without-open_app/
│   └── stage2_eval/{base,
│                    {full,lora}_base/epoch-{E},
│                    {full,lora}_world-model_from_{full,lora}-ep{E1}/epoch-{E2}}/
└── merged/
    ├── {model}_stage1_{full,lora}_world-model/epoch-{E}/
    ├── {model}_stage2_{full,lora}_base/epoch-{E}/
    └── {model}_stage2_{full,lora}_world-model_from_{full,lora}-ep{E1}/epoch-{E2}/
```

`BEST_CHECKPOINT` 파일은 더 이상 생성되지 않는다 — `trainer_state.json.epoch` 으로 epoch 번호를 결정하고, 어떤 epoch 을 Stage 2 에 쓸지는 사용자가 결과를 보고 `--stage1-epoch` 로 직접 지정한다.

### HuggingFace 업로드 ID 패턴

| Stage / variant | 패턴 |
|-------|------|
| Stage 1 | `SaFD-00/{short}-{slug}world-model-stage1-{full,lora}-epoch{E}` |
| Stage 2 base | `SaFD-00/{short}-{slug}base-stage2-{full,lora}-epoch{E2}` |
| Stage 2 world | `SaFD-00/{short}-{slug}world-model-stage1-{M1}-epoch{E1}-stage2-{M2}-epoch{E2}` |

`{slug}` 는 `ac-` (AC) / `ac-2-` (AC_2) / `ac-3-r{37,55,73}-` (AC_3 ratio 별) / `mc-` (MC). `{E}` 는 `trainer_state.json.epoch` 의 `int(round(...))`. 조립은 `scripts/_common.sh::hf_repo_id_stage1` / `hf_repo_id_stage2_base` / `hf_repo_id_stage2_world_model` 헬퍼에 단일화.

## 모델 추가 방법

새 모델 추가 시 다음을 동기화한다 (자세한 절차는 [`AGENTS.md`](./AGENTS.md)):

1. `gui-model.ipynb` Cell 5 의 `_MODEL_CONFIG` 에 모델 항목 추가 (`MODEL_FAMILY_CONFIG` 에 family 가 이미 등록돼 있는지 확인).
2. `scripts/_common.sh` 의 `MODEL_ID`, `MODEL_TEMPLATE`, `ALL_MODELS` 에 동일 항목 추가.
3. 노트북 Section 0 의 "Stage {1,2} YAML 일괄 생성" 셀 (Cell 8 / 10) 재실행 → YAML 자동 생성. MC 는 Stage 1 전용이라 Stage 2 YAML 은 MC 에 대해 생성되지 않는다 (`_STAGE1_ONLY` guard).

## 테스트 실행

Stage 2 Step Accuracy 채점 로직 (`scripts/_action_eval.py`) 회귀 테스트:

```bash
cd GUI-Model
pytest tests/test_action_eval.py -v   # 48 케이스
```

## 코드 읽기 시작점

- [`gui-model.ipynb`](./gui-model.ipynb) — 전체 파이프라인 기준
- [`scripts/_common.sh`](./scripts/_common.sh) — path / dataset / model / logging 규약, HF repo id 헬퍼
- [`scripts/split_data.py`](./scripts/split_data.py) — split 규칙 (AC / MC)
- [`scripts/_hungarian_eval.py`](./scripts/_hungarian_eval.py) — Stage 1 metric
- [`scripts/_action_eval.py`](./scripts/_action_eval.py) — Stage 2 metric
- [`scripts/eval_viewer.py`](./scripts/eval_viewer.py) — Stage 1/2 평가 결과 HTML 비교 뷰어

구조 설명은 [`ARCHITECTURE.md`](./ARCHITECTURE.md), 작업 규칙은 [`AGENTS.md`](./AGENTS.md) 를 본다.
