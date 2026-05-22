# GUI-Model

모바일 GUI World Modeling 이 Action Prediction 성능에 미치는 영향을 검증하는 **2-stage fine-tuning 파이프라인**. 8 개 Qwen 계열 Vision-Language 모델 × 3 개 학습 데이터셋 매트릭스를 단일 conda env (`gui-model`) + 저장소 내부 [`LlamaFactory/`](./LlamaFactory) 백엔드로 운영한다. 노트북 [`gui-model.ipynb`](./gui-model.ipynb) 가 기준 실행 경로이고 [`scripts/`](./scripts) 가 반복 실행용 자동화 레이어다.

## 개요

- **Stage 1** — `screenshot + UI XML + action -> next UI XML` (World Modeling)
- **Stage 2** — `screenshot + UI XML + task -> action JSON` (Action Prediction)
- **비교 실험**
  - `base` — base model 의 zero-shot 성능
  - `stage2` — base model → Stage 2 SFT
  - `stage1+stage2` — Stage 1 merged (world model) → Stage 2 SFT
- **파이프라인 흐름** (Stage 1 / Stage 2 공통): `train → merge → eval`. merge 는 `--no-hf-upload` 로 local export 만 수행할 수 있다. 현재 eval 은 HF Hub 에 push 된 merged repo 만 pull 하므로 학습 머신이 아닌 환경에서도 재실행 가능하다.

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
├── tests/test_action_eval.py     # Stage 2 메트릭 회귀 테스트 (52 케이스)
├── pyproject.toml                # 공통 deps + extras["llamafactory"] (pip editable 설치)
├── .env.example                  # HF_TOKEN, NPROC_PER_NODE, GPU_TYPE
├── README.md                     # (this file)
├── ARCHITECTURE.md               # 시스템 레퍼런스
└── AGENTS.md                     # 에이전트 작업 지침
```

## 환경 설치

conda env (`gui-model`) 에 `pyproject.toml` 의 공통 deps + `[project.optional-dependencies] llamafactory` 를 pip editable 로 설치한다. 서브프로젝트 `./LlamaFactory` 는 `pip install -e ./LlamaFactory` 로 함께 editable 설치하며, `transformers` 는 우리 extras pin (`>=4.57.1,<4.58`) 과 LlamaFactory 의 `<=5.2.0` 가 겹치는 4.57.x 구간에서 한 번에 풀린다.

```bash
# GUI-Model 설치
cd /path/to/GUI-Model
conda create -n gui-model python=3.12 -y
conda activate gui-model              # scripts/*.sh 실행 전 필수
pip install -e ".[llamafactory]"      # 공통 deps + extras
pip install -e ./LlamaFactory         # editable 서브프로젝트
```

deepspeed · vllm · bitsandbytes 모두 conda env `gui-model` 에 설치된다. 학습/export 엔진은 `llamafactory-cli train` / `llamafactory-cli export`.

CI / 다른 머신에서는 동일 절차로 재설치한다.

> **CUDA mismatch 주의**: 기존 `gui-model` env 에 `torch 2.11+cu130` 등 cu13 빌드가 남아 있으면 RTX5090 CUDA 가드(`scripts/_common.sh`)가 `nvcc 12.8` 불일치로 실행을 차단한다. pyproject 는 `torch>=2.4.0,<2.9` 로 pin 하므로 `pip install -e ".[llamafactory]"` 가 PyPI 기본 cu12.8 wheel(torch 2.8.x)로 교체해 정합시킨다. 잔존 시 강제 교정: `pip install --force-reinstall "torch>=2.4,<2.9" "torchvision<0.24"`.

### `.env` 변수

`.env.example` 를 복사해 `.env` 를 만든다. 노트북 Cell 5 와 `scripts/_common.sh` 가 이 파일을 source 한다.

| 변수 | 기본 | 허용값 | 설명 |
|------|------|--------|------|
| `HF_TOKEN` | — | (string) | HF Hub push/pull. merge 를 `--no-hf-upload` 로만 수행할 때는 불필요 |
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

- Python 3.11 이상, 3.13 미만 (conda env 는 3.12 권장 — LlamaFactory 서브프로젝트가 `>=3.11.0` 을 요구하므로 우리도 동일 하한)
- bash 4+ (`scripts/_common.sh` 기준)
- `transformers>=4.57.1,<4.58` 로 고정 — 변경 시 [`pyproject.toml`](./pyproject.toml) 의 `dependencies` / `[project.optional-dependencies] llamafactory` 와 주석을 함께 갱신한다. **서브프로젝트 `LlamaFactory/pyproject.toml` 은 건드리지 않는다.**

### PATH 정책

`scripts/_common.sh` 는 활성 conda env 의 `bin/` 을 PATH 최상단에 고정해 user-site 의 낡은 CLI (`accelerate` 등) 를 가린다. `CONDA_PREFIX` 미설정 상태에서 `scripts/*.sh` 를 실행하면 즉시 중단된다 — 먼저 `conda activate gui-model`.

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
#       Stage 1 (state_pred / action_pred) + Stage 2 모두 사전 필터 (3 파일).
#       split 은 항상 _filtered 만 입력으로 사용한다 (Qwen3-VL get_rope_index broadcast 회피).
#       Default image_max_pixels=2097152 는 Qwen3-VL family 기준 (Qwen2/2.5-VL 학습 시 --image-max-pixels 1605632 등 override).
#       이미 만들어진 _filtered.jsonl 이 있으면 --skip-existing 으로 누락된 source 만 처리 가능.
python scripts/filter_long_samples.py --dataset AC_3

# Stage 1 ratio mix + Stage 2 ID/OOD split 을 한 번에 산출.
python scripts/split_data.py --dataset AC_3 --ac3-ratios 3:7,5:5,7:3 --ac3-train-total 50000
# Stage 1 → gui-model_stage1_train_{3_7,5_5,7_3}.jsonl
#         + gui-model_stage1_test_{id,ood}_{state,action}_pred.jsonl
# Stage 2 → gui-model_stage2_{train,test_id,test_ood}.jsonl   (15K / 3K / 3K, _filtered 풀에서 산출, Stage 1 action_pred app partition 공유)
```

ratio 별로 **별개의 학습 가중치** 가 산출되므로 train/merge/eval 모두 `--dataset AC_3` 한 번에 ratio 3 종을 sweep 한다 (`--ac3-ratios r55,r73` 로 부분 실행 가능). 평가는 ratio 단일 (`--ac3-ratio r55` 기본) 로 고정. **Stage 1** 은 state/action **두 task 를 각각 채점** (`_hungarian_eval.py` / `_action_eval.py`). **Stage 2** 도 같은 ratio sweep 으로 활성 — Stage 1 ratio merged 를 base 로 같은 `gui-model_stage2_{train,test_id,test_ood}.jsonl` 을 학습. stage2 데이터 자체는 ratio 와 무관 (3 ratio 공유) 이며 ratio 차원은 stage1 → stage2 base 계보로만 흐른다 (산출 디렉토리/HF slug 는 ratio 별 분리).

### 2.6) AndroidControl_4 준비 (AC_3 r73 + Stage 1 diff loss 실험군)

`AndroidControl_4` 는 **AC_3 r73 과 동일한 학습 데이터·하이퍼파라미터**로 학습하되, Stage 1 state prediction 에 **diff loss (token-weighted SFT)** 를 적용하는 실험군이다. AC_3 r73 (기존 균일 cross-entropy) 대조군과 직접 비교해 diff loss 의 순효과를 측정한다.

- **diff loss**: current→future HTML diff 를 헝가리안 매칭으로 분류 (ADDED/MODIFIED/UNCHANGED) 하고, future HTML 의 변화 토큰에 더 큰 loss 가중치 (ADDED=2.0 / MODIFIED=2.0 / UNCHANGED=1.0) 를 준다. action prediction 샘플은 assistant 가 JSON 이라 diff element 가 0 개 → `token_weights` 가 전부 1.0 → 기존 cross-entropy 와 수학적으로 동일 (**action 은 기존 loss 유지**).
- **데이터**: train 만 별도 — `scripts/diff_loss/preprocess_dataset.py` 가 AC_3 r73 train 에 `token_weights` 필드를 추가. test / Stage 2 데이터는 AC_3 에서 복사 (동일 평가셋 — 공정 비교).
- **LlamaFactory 패치**: diff loss 는 LF 6 파일 수정에 의존 (`use_diff_token_weighted_loss` 인자 + `diff_token_weighted_loss_func`). LF 는 gitignore 된 별도 repo 라 패치를 git 에 담지 않고 `scripts/diff_loss/apply_llamafactory_patch.py` 가 멱등 재적용한다.

준비는 노트북 Section 0 의 **"AC_4 diff loss 환경 준비"** 셀이 한 번에 수행 (패치 적용 + 데이터 전처리 + AC_3 복사). 학습/merge/eval 은 다른 DS 와 동일하게 `--dataset AC_4` 로 호출:

```bash
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_4 --stage1-mode full
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC_4 --eval-datasets AC_4
```

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
- **Stage 2 (AC_3)**: Stage 1 의 action_pred app partition 을 그대로 재사용해 action_type stratified 샘플링 (`--stage2-train-size` 기본 15K, `--stage2-test-{id,ood}-size` 기본 3K). stage2 데이터는 ratio 무관 (3 ratio 공유) — train/merge/eval 은 stage1 ratio merged 를 base 로 ratio 별 sweep.
- **AC_4 (diff loss 실험군)**: split 불필요 — AC_3 r73 train 에 `token_weights` 부여 (`scripts/diff_loss/preprocess_dataset.py`) + test/Stage 2 는 AC_3 복사. 노트북 Section 0 환경 세팅 셀이 일괄 수행.
- **AC_2 / MB**: split 불필요.

## 실행 방법

### 1. 노트북 경로

`gui-model` env 에서 [`gui-model.ipynb`](./gui-model.ipynb) 를 섹션 순서대로 실행한다.

| Section | 내용 |
|---------|------|
| 0 | 환경, dataset / 모델 / family / size config 정의, Stage 1 · Stage 2 학습 YAML 일괄 생성 |
| 1-2 | `dataset_info.json` 등록 (AC: 6 entry, AC_2: 4 entry, MC: 2 entry, MB: eval-only 2 entry) |
| 3 | Stage 1 SFT 학습 (`qwen3-vl-8b` + Full FT, AC · AC_2 · AC_3) |
| 4 | Stage 1 merge (`qwen3-vl-8b` + Full FT, 모든 epoch local merge + 선택적 HF Hub push; `--no-hf-upload` 지원) |
| 5 | Stage 1 평가 (HF Hub merged sweep, EVAL_DS = AC, AC_2, AC_3, MC, MB) |
| 6 | Stage 2 SFT 학습 (`qwen3-vl-8b` + LoRA, AC · AC_2 · AC_3 ratio sweep) |
| 7 | Stage 2 merge (`qwen3-vl-8b` + LoRA, variant × 모든 epoch local merge + 선택적 HF push; `--no-hf-upload` 지원) |
| 8 | Stage 2 평가 (ID + OOD 동시 sweep, `action_metrics.json` 3 섹션, AC_3 ratio sweep 포함) |

> Section 3 / 4 / 6 / 7 은 **단일 변형 walkthrough** 다 — Stage 1 = `qwen3-vl-8b` + `full`, Stage 2 = `qwen3-vl-8b` + `lora`. 다른 모델 / 모드 / 데이터셋은 cell 을 추가하지 말고 shell 호출에서 `--model` / `--stage1-mode` / `--stage2-mode` / `--dataset` 인자만 바꿔 실행한다 (등록된 모델은 [`scripts/_common.sh::MODELS`](./scripts/_common.sh) 참조). 매트릭스 sweep 은 `--model all` / `--dataset all` 사용. Section 5 / 8 의 평가 cell 은 base + variant matrix 정의와 plot 코드를 보존한다.

### 2. shell script 경로

shell script 는 노트북에서 한 번 생성된 **학습 YAML** 과 `LlamaFactory/data/dataset_info.json` 이 이미 있다는 전제에서 동작한다. **Stage 1 eval 은 YAML 을 사용하지 않고 HF Hub merged repo 를 직접 sweep 한다.**

`stage1_merge.sh`, `stage2_merge.sh` 는 `--no-hf-upload` 를 받는다. 이 옵션은 로컬 `outputs/.../merged/...` 만 만들고 HF Hub push 를 생략한다. 현재 `stage1_eval.sh`, `stage2_eval.sh` 는 여전히 HF Hub merged repo 기준으로만 동작한다.

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

# Stage 1 — AC_3 학습 (state_pred + action_pred ratio 3 종 자동 sweep)
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_3 --stage1-mode full
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_3 --stage1-mode full
# 업로드 없이 로컬 merged dir 만 생성
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_3 --stage1-mode lora \
     --ac3-ratios r37,r73 --no-hf-upload
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

# Stage 2 — AC, AC_2, AC_3 지원 (MC 는 Stage 2 데이터/YAML 없음 → 거절)
bash scripts/stage2_train.sh --model qwen3-vl-8b --dataset AC \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora
bash scripts/stage2_merge.sh --model qwen3-vl-8b --dataset AC \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora
# 업로드 없이 local merge 만 수행
bash scripts/stage2_merge.sh --model qwen3-vl-8b --dataset AC \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora --no-hf-upload
bash scripts/stage2_eval.sh  --model qwen3-vl-8b --train-dataset AC --eval-datasets AC,AC_2,MB \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
     --variants base,lora_base,lora_world_model --epochs 1,2,3

# Stage 2 — AC_3 ratio sweep (Stage 1 ratio merged 를 base 로 같은 stage2 데이터를 ratio 별로 학습)
bash scripts/stage2_train.sh --model qwen3-vl-8b --dataset AC_3 \
     --stage1-mode full --stage1-epoch 1 --stage2-mode lora --ac3-ratios r37,r55,r73
bash scripts/stage2_merge.sh --model qwen3-vl-8b --dataset AC_3 \
     --stage1-mode full --stage1-epoch 1 --stage2-mode lora --ac3-ratios r37,r55,r73
# AC_3 평가는 ratio 1 개 (학습 ratio 와 일치). MB 교차 평가도 가능.
bash scripts/stage2_eval.sh  --model qwen3-vl-8b --train-dataset AC_3 --ac3-ratio r55 \
     --eval-datasets AC_3 --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
     --variants base,full_base,lora_base,full_world_model,lora_world_model --epochs 1,2,3
```

#### 평가 — `--train-dataset` 으로 HF repo, `--eval-datasets` 로 test 셋을 분리

| 스크립트 | `--train-dataset` 허용 | `--eval-datasets` 허용 | 기본 variants |
|---|---|---|---|
| `stage1_eval.sh` | `AC` / `AC_2` / `AC_3` / `MC` | `AC` / `AC_2` / `AC_3` / `MC` / `MB` | `base, full_world_model, lora_world_model` |
| `stage2_eval.sh` | `AC` / `AC_2` / `AC_3` (MC 거절) | `AC` / `AC_2` / `AC_3` / `MB` | `base, full_base, lora_base, full_world_model, lora_world_model` |

> **AC_3 Stage 1 평가 분기**: `stage1_eval.sh --eval-datasets AC_3` 는 state_pred / action_pred 두 task 를 각각 독립 채점한다.
> 산출 디렉토리는 `on-AC_3-state/hungarian_metrics.json` (state) + `on-AC_3-action/action_metrics.json` (action) 으로 분리되며 각자 독립 skip marker 를 갖는다.
>
> **AC_3 Stage 2 평가 분기**: `stage2_eval.sh --train-dataset AC_3 --eval-datasets AC_3` 는 일반 action prediction 으로 ID + OOD 두 파일을 함께 채점 (`action_metrics.json` 3 섹션). AC_3 모델 평가는 ratio 가 정확히 한 개여야 하므로 stage1/stage2 모두 `--ac3-ratio {r37|r55|r73}` (기본 `r55`) 로 골라야 한다.

`--epochs LIST` (콤마 구분, 기본 `1,2,3`) · `--variants LIST` (콤마 구분) · `--stage1-mode {full|lora}` / `--stage2-mode {full|lora}` / `--stage1-epoch N` (world-model variant 의 상류 Stage 1 epoch).

> **stage2 `--epochs` 에 `0` 포함 (opt-in)**: `{full|lora}_world_model` 은 epoch-0 = stage1 merged repo (`SaFD-00/{short}-{slug}world-model-stage1-{STAGE1_MODE}-epoch{STAGE1_EPOCH}`, stage2 미학습 베이스라인) 를 평가한다 — `hf_repo_id_stage1` 로 해석. `{full|lora}_base` 는 stage1 계보가 없어 epoch-0 = 원본 base 모델(= `base` variant)과 중복이므로 경고 후 skip. `base` variant 는 epoch 무관(원본 모델 zero-shot). 기본값 `1,2,3` 에는 `0` 이 없어 기존 실행에 영향 없음.

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
GUI-Model/outputs/{AC|AC_2|AC_3|MC}/      # AC_3 의 ratio (r37/r55/r73) 는 별도 디렉토리가 아니라 아래 {model}{SFX} 의 SFX 로 운반된다 (AC/AC_2/MC: SFX="", AC_3: SFX=_r37/_r55/_r73).
├── adapters/
│   ├── {model}{SFX}_stage1_{full,lora}_world-model/                              # Stage 1 체크포인트
│   ├── {model}{SFX}_stage2_{full,lora}_base/                                     # Stage 2 base
│   └── {model}{SFX}_stage2_{full,lora}_world-model_from_{full,lora}-ep{E1}/      # Stage 2 world-model (E1=stage1 upstream epoch)
├── eval/{model}{SFX}/
│   ├── stage1_eval/{base, {full,lora}_world-model/epoch-{E}}/   # 각 variant 안에 on-{EVAL_DS}/ + on-{EVAL_DS}-without-open_app/
│   └── stage2_eval/{base,
│                    {full,lora}_base/epoch-{E},
│                    {full,lora}_world-model_from_{full,lora}-ep{E1}/epoch-{E2}}/   # E2∈{0,1,2,3}: epoch-0 = stage1 merged (stage2 미학습)
└── merged/
    ├── {model}{SFX}_stage1_{full,lora}_world-model/epoch-{E}/
    ├── {model}{SFX}_stage2_{full,lora}_base/epoch-{E}/
    └── {model}{SFX}_stage2_{full,lora}_world-model_from_{full,lora}-ep{E1}/epoch-{E2}/
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
3. 노트북 Section 0 의 "Stage {1,2} YAML 일괄 생성" 셀 (Cell 8 / 10) 재실행 → YAML 자동 생성. MC 는 Stage 1 전용이라 Stage 2 YAML 은 MC 에 대해 생성되지 않는다 (`_STAGE1_ONLY = {"MonkeyCollection"}` guard).

## 테스트 실행

Stage 2 Step Accuracy 채점 로직 (`scripts/_action_eval.py`) 회귀 테스트:

```bash
cd GUI-Model
pytest tests/test_action_eval.py -v   # 52 케이스
```

## 코드 읽기 시작점

- [`gui-model.ipynb`](./gui-model.ipynb) — 전체 파이프라인 기준
- [`scripts/_common.sh`](./scripts/_common.sh) — path / dataset / model / logging 규약, HF repo id 헬퍼
- [`scripts/split_data.py`](./scripts/split_data.py) — split 규칙 (AC / MC)
- [`scripts/_hungarian_eval.py`](./scripts/_hungarian_eval.py) — Stage 1 metric
- [`scripts/_action_eval.py`](./scripts/_action_eval.py) — Stage 2 metric
- [`scripts/eval_viewer.py`](./scripts/eval_viewer.py) — Stage 1/2 평가 결과 HTML 비교 뷰어

구조 설명은 [`ARCHITECTURE.md`](./ARCHITECTURE.md), 작업 규칙은 [`AGENTS.md`](./AGENTS.md) 를 본다.
