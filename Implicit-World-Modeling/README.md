# Implicit-World-Modeling

모바일 GUI World Modeling 이 Action Prediction 성능에 미치는 영향을 검증하는 **2-stage fine-tuning 파이프라인**. Qwen 계열 Vision-Language 모델 × 학습 데이터셋 매트릭스를 단일 conda env (`implicit-world-modeling`) + 저장소 내부 [`LlamaFactory/`](./LlamaFactory) 백엔드로 운영한다. 노트북 [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) 가 기준 실행 경로이고 [`scripts/`](./scripts) 가 반복 실행용 자동화 레이어다.

## 개요

- **Stage 1** — `screenshot + UI XML + action -> next UI XML` (World Modeling)
- **Stage 2** — `screenshot + UI XML + task -> action JSON` (Action Prediction)
- **비교 실험**
  - `base` — base model 의 zero-shot 성능
  - `stage2` — base model → Stage 2 SFT
  - `stage1+stage2` — Stage 1 merged (world model) → Stage 2 SFT
- **파이프라인 흐름** (Stage 1 / Stage 2 공통): `train → merge → eval`. merge 는 `--no-hf-upload` 로 local export 만 수행할 수 있다. eval 은 **로컬 `outputs/.../merged/.../epoch-{E}/` 가 있으면 우선 사용하고 없을 때만 HF Hub merged repo 를 pull** (`_common.sh::resolve_eval_model_path`) — `--no-hf-upload` 만 한 워크플로우와 HF push 후 다른 머신에서 재실행하는 워크플로우 모두에서 동작한다.

## 지원 모델 (2 개, 모두 7-9B tier)

| # | model_id | short_name | template | size |
|---|----------|------------|----------|------|
| 1 | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3-vl-8b` | `qwen3_vl_nothink` | 7-9B |
| 2 | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2.5-vl-7b` | `qwen2_vl` | 7-9B |

> Qwen3-VL 의 `qwen3_vl_nothink` template 은 추론 시 `vllm_infer.py` 에 `--enable_thinking False` 가 자동 주입된다. Qwen2.5-VL 의 `qwen2_vl` template 은 thinking 트리거가 없어 해당 플래그를 주입하지 않는다.

### 모델 family 별 image budget

학습 YAML 의 `image_max_pixels` / `image_min_pixels` 는 vision encoder patch-size (factor) 와 **학습 데이터셋** 으로 결정된다 (`implicit-world-modeling.ipynb` Cell 5 의 `MODEL_FAMILY_CONFIG` + `_DATASET_CONFIG[ds]["image_overrides"]`). 학습된 모델은 어떤 ds 를 평가하든 **학습 시 budget** 을 그대로 사용한다 (학습-추론 mismatch 방지).

| family | patch | merge | factor | min_tokens | min_pixels |
|--------|-------|-------|--------|-----------|------------|
| Qwen3-VL | 16 | 2 | 32 | 4 | 4,096 |
| Qwen2.5-VL | 14 | 2 | 28 | 4 | 3,136 |

| 학습 DS | max_tokens | Qwen3-VL `max_pixels` | Qwen2.5-VL `max_pixels` |
|---|---|---|---|
| AC_EXP01, AC_EXP02, AC_EXP03, MC | 2,048 (family default) | 2,097,152 (= 2048 × 32²) | 1,605,632 (= 2048 × 28²) |

> 평가측 `scripts/_common.sh::build_infer_cmd` 가 `TRAIN_DATASET` 환경변수로 학습 DS 를 식별해 동일 budget 을 적용한다.

## 디렉토리 구조

```
Implicit-World-Modeling/
├── implicit-world-modeling.ipynb               # 단일 노트북 — 파이프라인 기준 실행 경로
├── scripts/
│   ├── _common.sh                # 공통 path / dataset 매핑 / 모델 레지스트리 / HF repo id 헬퍼
│   ├── stage{1,2}_{train,eval,merge}.sh
│   ├── _hungarian_eval.py        # Stage 1 metric (single-pair / ID·OOD 모드)
│   ├── _action_eval.py           # Stage 2 metric (single-pair / ID·OOD 모드, Step Accuracy)
│   ├── eval_viewer.py            # Stage 1/2 산출물 비교 HTML 빌더
│   ├── split_data.py             # AC_EXP01/MC split (원본 AndroidControl/ source → EXP01/MC 산출)
│   ├── mirror_exp03.py           # AC_EXP03 = AC_EXP01 ratio73 멤버십 좌표(point) 표현 미러
│   ├── extract_androidcontrol_metadata.py   # AndroidControl episodes_meta.jsonl 생성
│   └── extract_androidcontrol_images.py     # AndroidControl GCS TFRecord → PNG (TF 의존 없음)
├── data/                         # AndroidControl (원본 source) / AC_EXP01 / AC_EXP02 / AC_EXP03 / MC / MB
├── LlamaFactory/                 # 학습/추론 엔진 (clone)
│   ├── examples/custom/IWM-{AC_EXP01,AC_EXP02,AC_EXP03,MC}/  # 노트북이 생성한 학습 YAML
│   └── scripts/vllm_infer.py     # 추론 도구
├── implicit_world_modeling/                    # 배포용 스텁 (핵심 로직 없음)
├── tests/test_action_eval.py     # Stage 2 메트릭 회귀 테스트 (52 케이스)
├── pyproject.toml                # 공통 deps + extras["llamafactory"] (pip editable 설치)
├── .env.example                  # HF_TOKEN, NPROC_PER_NODE, GPU_TYPE
├── README.md                     # (this file)
├── ARCHITECTURE.md               # 시스템 레퍼런스
└── AGENTS.md                     # 에이전트 작업 지침
```

## 환경 설치

conda env (`implicit-world-modeling`) 에 `pyproject.toml` 의 공통 deps + `[project.optional-dependencies] llamafactory` 를 pip editable 로 설치한다. 서브프로젝트 `./LlamaFactory` 는 `pip install -e ./LlamaFactory` 로 함께 editable 설치하며, `transformers` 는 우리 extras pin (`>=4.57.1,<4.58`) 과 LlamaFactory 의 `<=5.2.0` 가 겹치는 4.57.x 구간에서 한 번에 풀린다.

```bash
# Implicit-World-Modeling 설치
cd /path/to/Implicit-World-Modeling
conda create -n implicit-world-modeling python=3.12 -y
conda activate implicit-world-modeling              # scripts/*.sh 실행 전 필수
pip install -e ".[llamafactory]"      # 공통 deps + extras
pip install -e ./LlamaFactory         # editable 서브프로젝트
```

deepspeed · vllm · bitsandbytes 모두 conda env `implicit-world-modeling` 에 설치된다. 학습/export 엔진은 `llamafactory-cli train` / `llamafactory-cli export`.

CI / 다른 머신에서는 동일 절차로 재설치한다.

> **CUDA mismatch 주의**: 기존 `implicit-world-modeling` env 에 `torch 2.11+cu130` 등 cu13 빌드가 남아 있으면 RTX5090 CUDA 가드(`scripts/_common.sh`)가 `nvcc 12.8` 불일치로 실행을 차단한다. pyproject 는 `torch>=2.4.0,<2.9` 로 pin 하므로 `pip install -e ".[llamafactory]"` 가 PyPI 기본 cu12.8 wheel(torch 2.8.x)로 교체해 정합시킨다. 잔존 시 강제 교정: `pip install --force-reinstall "torch>=2.4,<2.9" "torchvision<0.24"`.

### `.env` 변수

`.env.example` 를 복사해 `.env` 를 만든다. 노트북 Cell 5 와 `scripts/_common.sh` 가 이 파일을 source 한다.

| 변수 | 기본 | 허용값 | 설명 |
|------|------|--------|------|
| `HF_TOKEN` | — | (string) | HF Hub push/pull. merge 를 `--no-hf-upload` 로만 수행하고 eval 도 local merged dir hit 로만 끝낼 때는 불필요 (HF fallback 발생 시에만 필요) |
| `NPROC_PER_NODE` | `2` | `1`, `2`, `4`, `8` | node 당 GPU 수 (single node, torchrun world size) |
| `GPU_TYPE` | `H100` | `RTX5090`, `A100`, `H100` | GPU 종류. (모델 size × GPU) 표로 `per_device_train_batch_size` 결정 |

#### `per_device_train_batch_size` (size × GPU)

| 모델 size | RTX5090 (32GB) | A100 (80GB) | H100 (80GB) |
|-----------|----------------|-------------|-------------|
| 7-9B      | 1              | 2           | 2           |

`GLOBAL_BATCH_SIZE = 64` 를 유지하기 위해 `gradient_accumulation_steps = 64 / (per_device × NPROC_PER_NODE)` 가 자동 역계산된다 (정수가 아니면 `ValueError`). 위 표 값은 `NPROC ∈ {1, 2, 4, 8}` 모두에서 정수가 된다.

> **AC_EXP03 예외**: 좌표(point) 표현으로 시퀀스가 ~2.5x (`cutoff_len 24576`) 길어 메모리가 커지므로, EXP03 만 `per_device_train_batch_size` 를 절반 (H100/A100 2→1, RTX5090 1 유지) 으로 낮추고 `gradient_accumulation_steps` 로 보정해 `GLOBAL_BATCH_SIZE=64` 를 유지한다 (EXP01 과 global batch 동일 → 표현 차이만 공정 비교).

> `.env` 의 `GPU_TYPE` / `NPROC_PER_NODE` 를 수정한 뒤에는 노트북 Cell 5 (CONFIGS) 와 Stage 1/2 YAML 생성 셀 (Cell 8 / 10) 을 다시 실행해야 새 값이 YAML 에 반영된다.

### 전제

- Python 3.11 이상, 3.13 미만 (conda env 는 3.12 권장 — LlamaFactory 서브프로젝트가 `>=3.11.0` 을 요구하므로 우리도 동일 하한)
- bash 4+ (`scripts/_common.sh` 기준)
- `transformers>=4.57.1,<4.58` 로 고정 — 변경 시 [`pyproject.toml`](./pyproject.toml) 의 `dependencies` / `[project.optional-dependencies] llamafactory` 와 주석을 함께 갱신한다. **서브프로젝트 `LlamaFactory/pyproject.toml` 은 건드리지 않는다.**

### PATH 정책

`scripts/_common.sh` 는 활성 conda env 의 `bin/` 을 PATH 최상단에 고정해 user-site 의 낡은 CLI (`accelerate` 등) 를 가린다. `CONDA_PREFIX` 미설정 상태에서 `scripts/*.sh` 를 실행하면 즉시 중단된다 — 먼저 `conda activate implicit-world-modeling`.

## 데이터 준비

학습 대상 DS 는 **AC_EXP01**, **AC_EXP02**, **AC_EXP03**, **MC**. **MB** 는 평가 전용이므로 split 하지 않는다. **AndroidControl (AC) 자체는 학습/평가 대상이 아니라 AC_EXP01 / AC_EXP02 / AC_EXP03 의 원본 source 자산** (이미지 + 원본 jsonl + `episodes_meta.jsonl` + 좌표 표현 `*_xy.jsonl`) 으로만 보존된다.

```
data/
├── AndroidControl/                 # 원본 source (이미지 + jsonl + episodes_meta + *_xy.jsonl) — EXP01/EXP02/EXP03 가 read
├── AndroidControl_EXP01/           # state+action ratio-mix (Stage 1 + 2, ID/OOD) — split 산출물
├── AndroidControl_EXP02/           # AC_EXP01 ratio73 + diff loss 실험군 (Stage 1 + 2) — 산출물
├── AndroidControl_EXP03/           # AC_EXP01 ratio73 좌표(point) 표현 미러 (Stage 1 + 2) — 산출물
├── MonkeyCollection/                # Stage 1 학습 + 평가 (Stage 2 자동 skip)
└── MobiBench/                       # 평가 전용 (단일 파일 stage{1,2}.jsonl)
```

### 1) AndroidControl 원본 자산 준비

`AndroidControl` 폴더는 학습/평가 entry 가 아니다 — AC_EXP01 / AC_EXP02 / AC_EXP03 가 source 로 read 한다 (AC_EXP03 은 좌표 표현 `implicit-world-modeling_stage{1_action,1_state,2}_xy.jsonl` 을 추가로 둔다).

```bash
# (a) GCS TFRecord 에서 스크린샷 추출 (TF 의존 없음, urllib 만 사용)
python scripts/extract_androidcontrol_images.py \
    --output data/AndroidControl/images/ --skip-existing --verbose

# (b) episodes_meta.jsonl 생성 (primary_app = 전경 앱 package_name)
#     pip install android-env 필요 (proto 디코딩 용)
python scripts/extract_androidcontrol_metadata.py \
    --output data/AndroidControl/episodes_meta.jsonl
```

### 2) AndroidControl_EXP01 준비 (Stage 1 state+action ratio mix + Stage 2 ID/OOD split)

```bash
# (선행) mm-expanded length > cutoff_len 인 샘플을 제거해 _filtered.jsonl 산출.
#       Stage 1 (state_pred / action_pred) + Stage 2 모두 사전 필터 (3 파일).
#       split 은 항상 _filtered 만 입력으로 사용한다 (Qwen3-VL get_rope_index broadcast 회피).
#       Default image_max_pixels=2097152 는 Qwen3-VL family 기준 (Qwen2/2.5-VL 학습 시 --image-max-pixels 1605632 등 override).
#       이미 만들어진 _filtered.jsonl 이 있으면 --skip-existing 으로 누락된 source 만 처리 가능.
python scripts/filter_long_samples.py --dataset AC_EXP01

# Stage 1 ratio mix + Stage 2 ID/OOD split 을 한 번에 산출. source = 원본 data/AndroidControl/, output = data/AndroidControl_EXP01/.
python scripts/split_data.py --dataset AC_EXP01 --exp01-ratios 3:7,5:5,7:3 --exp01-train-total 50000
# Stage 1 → data/AndroidControl_EXP01/implicit-world-modeling_stage1_train_{3_7,5_5,7_3}.jsonl
#         + data/AndroidControl_EXP01/implicit-world-modeling_stage1_test_{id,ood}_{state,action}_pred.jsonl
# Stage 2 → data/AndroidControl_EXP01/implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl   (15K / 3K / 3K)
```

ratio 별로 **별개의 학습 가중치** 가 산출되므로 train/merge/eval 모두 `--dataset AC_EXP01` 한 번에 ratio 3 종을 sweep 한다 (`--exp01-ratios ratio55,ratio73` 로 부분 실행 가능). 평가는 ratio 단일 (`--exp01-ratio ratio55` 기본) 로 고정. **Stage 1** 은 state/action **두 task 를 각각 채점** (`_hungarian_eval.py` / `_action_eval.py`). **Stage 2** 도 같은 ratio sweep 으로 활성 — Stage 1 ratio merged 를 base 로 같은 `implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl` 을 학습. stage2 데이터 자체는 ratio 와 무관 (3 ratio 공유) 이며 ratio 차원은 stage1 → stage2 base 계보로만 흐른다 (산출 디렉토리/HF slug 는 ratio 별 분리).

### 3) AndroidControl_EXP02 준비 (AC_EXP01 ratio73 + Stage 1 diff loss 실험군)

`AndroidControl_EXP02` 는 **AC_EXP01 ratio73 과 동일한 학습 데이터·하이퍼파라미터**로 학습하되, Stage 1 state prediction 에 **diff loss (token-weighted SFT)** 를 적용하는 실험군이다. AC_EXP01 ratio73 (기존 균일 cross-entropy) 대조군과 직접 비교해 diff loss 의 순효과를 측정한다.

- **diff loss**: current→future HTML diff 를 헝가리안 매칭으로 분류 (ADDED/MODIFIED/UNCHANGED) 하고, future HTML 의 변화 토큰에 더 큰 loss 가중치 (ADDED=2.0 / MODIFIED=2.0 / UNCHANGED=1.0) 를 준다. action prediction 샘플은 assistant 가 JSON 이라 diff element 가 0 개 → `token_weights` 가 전부 1.0 → 기존 cross-entropy 와 수학적으로 동일 (**action 은 기존 loss 유지**).
- **데이터**: train 만 별도 — `scripts/diff_loss/preprocess_dataset.py` 가 AC_EXP01 ratio73 train 에 `token_weights` 필드를 추가. test / Stage 2 데이터는 AC_EXP01 에서 복사 (동일 평가셋 — 공정 비교).
- **LlamaFactory 패치**: diff loss 는 LF 6 파일 수정에 의존 (`use_diff_token_weighted_loss` 인자 + `diff_token_weighted_loss_func`). LF 는 gitignore 된 별도 repo 라 패치를 git 에 담지 않고 `scripts/diff_loss/apply_llamafactory_patch.py` 가 멱등 재적용한다.

준비는 노트북 Section 0 의 **"AC_EXP02 diff loss 환경 준비"** 셀이 한 번에 수행 (패치 적용 + 데이터 전처리 + AC_EXP01 복사). 학습/merge/eval 은 다른 DS 와 동일하게 `--dataset AC_EXP02` 로 호출:

```bash
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP02 --stage1-mode full
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP02 --eval-datasets AC_EXP02
```

### 4) AndroidControl_EXP03 준비 (AC_EXP01 ratio73 좌표(point) 표현 미러)

`AndroidControl_EXP03` 는 **AC_EXP01 ratio73 과 동일한 (episode, step) 멤버십**을 좌표(`point`) 표현으로 미러한 실험군이다. UI 트리가 `index="N"` 대신 `bounds="[x1,y1][x2,y2]" point="[cx,cy]"` 를, 액션이 `index` 대신 `point=[x,y]` (0–1000 정규화) 를 사용한다 — index→좌표 추론 효과만 AC_EXP01 ratio73 (대조군) 과 비교한다 (diff loss 없음).

- **원천**: `data/AndroidControl/implicit-world-modeling_stage{1_action,1_state,2}_xy.jsonl` 3 종 (좌표 표현, 이미 필터됨).
- **미러**: `scripts/mirror_exp03.py` 가 EXP01 의 각 산출 파일(`stage1_train_7_3` + dual-task test + stage2)을 한 줄씩 읽어 `(episode, step)` 키로 대응 좌표 레코드를 골라 동일 순서로 write. 본문은 좌표 표현, 이미지 경로는 EXP01 레코드의 것(`AndroidControl/images/...`)을 채택. stage1 train 은 EXP02 스타일 단일 `implicit-world-modeling_stage1_train.jsonl` 로 출력.
- **누락 제외**: EXP03 원천에 없는 `(episode, step)` 키(~0.8–1.7%)는 제외 → 각 파일이 EXP01 대비 소폭 작다 (train ~49,596 / stage2_train ~14,881). 각 레코드는 EXP01 과 `(episode, step)` 1:1 대응 (표현만 좌표).
- **cutoff_len 24576 (무손실)**: 좌표 표현은 시퀀스가 ~2~2.5x (최대 20k+) 길어 공통 `cutoff_len=10000` 에서 ~10% 잘림·~0.3% 크래시가 난다. EXP03 멤버십이 EXP01 ratio73 (index 기준 ≤10000 필터됨) 의 미러라 팽창 상한이 묶여 있어 **필터 없이 cutoff 만 24576 으로 올려 잘림/크래시 0·데이터 손실 0** 으로 EXP01 과 표현 차이만 공정 비교한다 (학습·평가 모두 24576). 긴 시퀀스 메모리 보전을 위해 EXP03 만 `per_device_train_batch_size` 절반·`gradient_accumulation_steps` 보정 (global batch 64 유지).

```bash
python scripts/mirror_exp03.py
# → data/AndroidControl_EXP03/implicit-world-modeling_stage1_train.jsonl
#   + stage1_test_{id,ood}_{state,action}_pred(+_without_open_app) + stage2_{train,test_id,test_ood}

# (선택) 24576 이 전 샘플을 덮는지 측정 — over-threshold=0 이면 무손실 (필터링은 하지 않음).
python scripts/filter_long_samples.py --dataset AC_EXP03 --threshold 24576 --report-only
```

학습/merge/eval 은 다른 DS 와 동일하게 `--dataset AC_EXP03` / `--train-dataset AC_EXP03` 로 호출.

### 5) MonkeyCollection 준비

```bash
python scripts/split_data.py --dataset MC
# → implicit-world-modeling_stage1_{train,test}.jsonl  (random 95:5, Stage 2 자동 skip)
```

### 6) MobiBench 준비

split 불필요. `data/MobiBench/implicit-world-modeling_stage{1,2}.jsonl` 두 단일 파일만 있으면 평가가 성립한다. `dataset_info.json` 등록은 `scripts/_common.sh::ensure_eval_only_dataset_info()` 가 `stage{1,2}_eval.sh` source 시점에 idempotent 하게 보장한다.

### 분할 규칙 요약

- **App partition (AC_EXP01)**: 원본 `data/AndroidControl/episodes_meta.jsonl` 의 `primary_app` 으로 앱 집합을 셔플 → OOD 버킷이 `--stage2-test-ood-size` 를 채울 때까지 먼저 할당, 나머지는 ID 버킷. 같은 (id_apps, ood_apps) 를 Stage 1 / Stage 2 가 공유.
- **Stage 1 (MC)**: 메타 없음 → random split (`--stage1-ratio`, 기본 0.95).
- **Stage 1 (AC_EXP01)**: state_pred (random) + action_pred (action-type stratified) 를 비율 (`--exp01-ratios`, 기본 `7:3,3:7,5:5` — 콜론 구분) 로 혼합. id/ood 는 task 별로 분리된 4 파일.
- **Stage 2 (AC_EXP01)**: Stage 1 의 action_pred app partition 을 그대로 재사용해 action_type stratified 샘플링 (`--stage2-train-size` 기본 15K, `--stage2-test-{id,ood}-size` 기본 3K). stage2 데이터는 ratio 무관 (3 ratio 공유) — train/merge/eval 은 stage1 ratio merged 를 base 로 ratio 별 sweep.
- **AC_EXP02 (diff loss 실험군)**: split 불필요 — AC_EXP01 ratio73 train 에 `token_weights` 부여 (`scripts/diff_loss/preprocess_dataset.py`) + test/Stage 2 는 AC_EXP01 복사. 노트북 Section 0 환경 세팅 셀이 일괄 수행.
- **AC_EXP03 (좌표 미러 실험군)**: split 불필요 — `scripts/mirror_exp03.py` 가 AC_EXP01 ratio73 의 stage1 train + dual-task test + Stage 2 멤버십을 좌표 표현으로 미러 (없는 키 제외). 표현(index→x,y) 효과만 비교.
- **MB**: split 불필요.

## 실행 방법

### 1. 노트북 경로

`implicit-world-modeling` env 에서 [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) 를 섹션 순서대로 실행한다.

| Section | 내용 |
|---------|------|
| 0 | 환경, dataset / 모델 / family / size config 정의, Stage 1 · Stage 2 학습 YAML 일괄 생성 |
| 1-2 | `dataset_info.json` 등록 (AC_EXP01 / AC_EXP02 / AC_EXP03 / MC / MB) |
| 3 | Stage 1 SFT 학습 (`qwen3-vl-8b` + Full FT, AC_EXP01 · AC_EXP02 · AC_EXP03) |
| 4 | Stage 1 merge (`qwen3-vl-8b` + Full FT, 모든 epoch local merge + 선택적 HF Hub push; `--no-hf-upload` 지원) |
| 5 | Stage 1 평가 (local merged 우선 + HF Hub fallback sweep, EVAL_DS = AC_EXP01, AC_EXP02, AC_EXP03, MC, MB) |
| 6 | Stage 2 SFT 학습 (`qwen3-vl-8b` + LoRA, AC_EXP01 ratio sweep · AC_EXP02 · AC_EXP03) |
| 7 | Stage 2 merge (`qwen3-vl-8b` + LoRA, variant × 모든 epoch local merge + 선택적 HF push; `--no-hf-upload` 지원) |
| 8 | Stage 2 평가 (ID + OOD 동시 sweep, `action_metrics.json` 3 섹션, AC_EXP01 ratio sweep 포함) |

> Section 3 / 4 / 6 / 7 은 **단일 변형 walkthrough** 다 — Stage 1 = `qwen3-vl-8b` + `full`, Stage 2 = `qwen3-vl-8b` + `lora`. 다른 모델 / 모드 / 데이터셋은 cell 을 추가하지 말고 shell 호출에서 `--model` / `--stage1-mode` / `--stage2-mode` / `--dataset` 인자만 바꿔 실행한다 (등록된 모델은 [`scripts/_common.sh::MODELS`](./scripts/_common.sh) 참조). `--dataset` 은 명시적 선택 필수 (`--dataset all` 은 지원하지 않음). Section 5 / 8 의 평가 cell 은 base + variant matrix 정의와 plot 코드를 보존한다.

### 2. shell script 경로

shell script 는 노트북에서 한 번 생성된 **학습 YAML** 과 `LlamaFactory/data/dataset_info.json` 이 이미 있다는 전제에서 동작한다. **Stage 1 eval 은 YAML 을 사용하지 않고 (variant, epoch) 단위로 merged 모델을 직접 sweep 한다.**

`stage1_merge.sh`, `stage2_merge.sh` 는 `--no-hf-upload` 를 받는다. 이 옵션은 로컬 `outputs/.../merged/...` 만 만들고 HF Hub push 를 생략한다. `stage1_eval.sh`, `stage2_eval.sh` 는 **로컬 `outputs/.../merged/.../epoch-{E}/` 가 존재하면 그것을 우선 사용하고, 없을 때만 HF Hub merged repo 로 fallback** 한다 (`_common.sh::resolve_eval_model_path`). 따라서 `--no-hf-upload` 만으로 merge 한 머신에서도 같은 머신 안에서 eval 까지 이어 돌 수 있다.

#### 학습 / merge

```bash
# Stage 1 LoRA — MC 학습 (Stage 1 전용)
bash scripts/stage1_train.sh --model qwen2.5-vl-7b --dataset MC --stage1-mode lora
bash scripts/stage1_merge.sh --model qwen2.5-vl-7b --dataset MC --stage1-mode lora
bash scripts/stage1_eval.sh  --model qwen2.5-vl-7b --train-dataset MC --eval-datasets MC,MB \
     --stage1-mode lora --variants base,lora_world_model --epochs 1,2,3

# Stage 1 — AC_EXP01 학습 (state_pred + action_pred ratio 3 종 자동 sweep)
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP01 --stage1-mode full
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_EXP01 --stage1-mode full
# 업로드 없이 로컬 merged dir 만 생성
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_EXP01 --stage1-mode lora \
     --exp01-ratios ratio37,ratio73 --no-hf-upload
# 평가는 ratio 1 개 단위 (state, action 각각 채점). ratio55 기본.
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP01 --exp01-ratio ratio55 \
     --eval-datasets AC_EXP01 --variants base,full_world_model --epochs 1,2,3
# 부분 ratio sweep + 외부 벤치마크 교차 평가
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP01 --exp01-ratios ratio55,ratio73 --stage1-mode lora
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP01 --exp01-ratio ratio73 \
     --eval-datasets AC_EXP01,MB

# Stage 2 — AC_EXP01, AC_EXP02 지원 (MC 는 Stage 2 데이터/YAML 없음 → 거절)
# Stage 2 — AC_EXP02 (diff loss 실험군)
bash scripts/stage2_train.sh --model qwen3-vl-8b --dataset AC_EXP02 \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora
bash scripts/stage2_merge.sh --model qwen3-vl-8b --dataset AC_EXP02 \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora
bash scripts/stage2_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP02 --eval-datasets AC_EXP02,MB \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
     --variants base,lora_base,lora_world_model --epochs 1,2,3

# Stage 2 — AC_EXP01 ratio sweep (Stage 1 ratio merged 를 base 로 같은 stage2 데이터를 ratio 별로 학습)
bash scripts/stage2_train.sh --model qwen3-vl-8b --dataset AC_EXP01 \
     --stage1-mode full --stage1-epoch 1 --stage2-mode lora --exp01-ratios ratio37,ratio55,ratio73
bash scripts/stage2_merge.sh --model qwen3-vl-8b --dataset AC_EXP01 \
     --stage1-mode full --stage1-epoch 1 --stage2-mode lora --exp01-ratios ratio37,ratio55,ratio73
# AC_EXP01 평가는 ratio 1 개 (학습 ratio 와 일치). MB 교차 평가도 가능.
bash scripts/stage2_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP01 --exp01-ratio ratio55 \
     --eval-datasets AC_EXP01 --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
     --variants base,full_base,lora_base,full_world_model,lora_world_model --epochs 1,2,3
```

#### 평가 — `--train-dataset` 으로 HF repo, `--eval-datasets` 로 test 셋을 분리

| 스크립트 | `--train-dataset` 허용 | `--eval-datasets` 허용 | 기본 variants |
|---|---|---|---|
| `stage1_eval.sh` | `AC_EXP01` / `AC_EXP02` / `AC_EXP03` / `MC` | `AC_EXP01` / `AC_EXP02` / `AC_EXP03` / `MC` / `MB` | `base, full_world_model, lora_world_model` |
| `stage2_eval.sh` | `AC_EXP01` / `AC_EXP02` / `AC_EXP03` (MC 거절) | `AC_EXP01` / `AC_EXP02` / `AC_EXP03` / `MB` | `base, full_base, lora_base, full_world_model, lora_world_model` |

> **AC_EXP01 Stage 1 평가 분기**: `stage1_eval.sh --eval-datasets AC_EXP01` 는 state_pred / action_pred 두 task 를 각각 독립 채점한다.
> 산출 디렉토리는 `on-AC_EXP01-state/hungarian_metrics.json` (state) + `on-AC_EXP01-action/action_metrics.json` (action) 으로 분리되며 각자 독립 skip marker 를 갖는다.
>
> **AC_EXP01 Stage 2 평가 분기**: `stage2_eval.sh --train-dataset AC_EXP01 --eval-datasets AC_EXP01` 는 일반 action prediction 으로 ID + OOD 두 파일을 함께 채점 (`action_metrics.json` 3 섹션). AC_EXP01 모델 평가는 ratio 가 정확히 한 개여야 하므로 stage1/stage2 모두 `--exp01-ratio {r37|r55|r73}` (기본 `r55`) 로 골라야 한다.

`--epochs LIST` (콤마 구분, 기본 `1,2,3`) · `--variants LIST` (콤마 구분) · `--stage1-mode {full|lora}` / `--stage2-mode {full|lora}` / `--stage1-epoch N` (world-model variant 의 상류 Stage 1 epoch).

> **stage2 `--epochs` 에 `0` 포함 (opt-in)**: `{full|lora}_world_model` 은 epoch-0 = stage1 merged repo (`SaFD-00/{short}-{slug}world-model-stage1-{STAGE1_MODE}-epoch{STAGE1_EPOCH}`, stage2 미학습 베이스라인) 를 평가한다 — `resolve_eval_model_path stage1 …` 로 해석되어 local merged 우선 + HF fallback. `{full|lora}_base` 는 stage1 계보가 없어 epoch-0 = 원본 base 모델(= `base` variant)과 중복이므로 경고 후 skip. `base` variant 는 epoch 무관(원본 모델 zero-shot). 기본값 `1,2,3` 에는 `0` 이 없어 기존 실행에 영향 없음.

> **재실행 시 skip**: 각 unit 의 marker (`hungarian_metrics.json` / `action_metrics.json`) 가 이미 존재하면 `[=] ... skip (already done): ...` 로그만 남긴다. 강제 재평가는 해당 marker 를 `rm` 후 재실행. Stage 1 의 `without_open_app` 산출물은 정규 metric 과 별도 marker 로 독립 skip.

#### Stage 1/2 메트릭 정의

- **Stage 1**: Hungarian F1 (UI element matching) + BLEU + ROUGE-L. 산출 파일 `hungarian_metrics.json`.
- **Stage 2**: Step Accuracy `correct = parse_ok ∧ type==gt.type ∧ field_match(type)`. 산출 파일 `action_metrics.json` (`overall` / `in_domain` / `out_of_domain` 3 섹션 또는 `overall` 1 섹션).
- 자세한 정의는 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §6.

### 평가 결과 시각 비교 (`scripts/eval_viewer.py`)

`stage{1,2}_eval.sh` 산출물을 행 정렬된 HTML 로 비교한다. (EXP, MODEL) 쌍 1 개를 `--include` 로 주면 그 EXP 의 eval/ 디렉토리에, 2 개 이상이면 `outputs/_compare/` 에 cross-EXP 비교 HTML 을 산출한다.

```bash
# 단일 EXP — 결과는 outputs/{data_dir}/eval/{MODEL}/stage{N}_eval/pairs_*.html
python scripts/eval_viewer.py --include AC_EXP02:qwen3-vl-8b
python scripts/eval_viewer.py --include AC_EXP01:qwen3-vl-8b_ratio73 --stages 2

# 다중 EXP 동급 stage cross-compare — 결과는 outputs/_compare/stage{N}_eval/pairs_*.html
python scripts/eval_viewer.py --include AC_EXP01:qwen3-vl-8b_ratio73 AC_EXP02:qwen3-vl-8b

# 데이터셋/variant 필터
python scripts/eval_viewer.py --include AC_EXP02:qwen3-vl-8b \
    --datasets on-AC-state-id on-AC-action-id \
    --variants "lora_world-model/epoch-1"
```

- `--include EXP:MODEL` (필수, 1 개 이상): `EXP ∈ {AC_EXP01, AC_EXP02, AC_EXP03, MC}`, `MODEL` 은 `outputs/<DS_DATADIR(EXP)>/eval/` 아래 디렉토리 명 (AC_EXP01 ratio variant 는 `qwen3-vl-8b_ratio{37,55,73}`, AC_EXP02 / AC_EXP03 은 `qwen3-vl-8b`).
- `--stages {1,2}` (기본 둘 다), `--datasets` 는 **logical key** (예: `on-AC-state-id`, `on-AC-state-id-without-open_app`, `on-AC-action-id`, `on-MB`, `on-MC`; Stage 2 는 `on-AC-id`, `on-AC-ood`, `on-MB`).
- multi-EXP 모드 variant 라벨은 `[EXP] MODEL/variant_path` 로 컬럼/메트릭 행에 노출되며, in-page checkbox 로 토글한다.
- 동일 logical key 에 대한 prediction row count 는 모든 spec 에서 일치해야 한다 (AC_EXP01 ↔ AC_EXP02 의 test 데이터는 byte-identical copy — `data/AndroidControl_EXP02/` 가 AC_EXP01 에서 복사됨). **AC_EXP03 은 좌표 표현 + 누락 제외로 행 수가 EXP01/EXP02 와 다르므로 cross-compare 대상이 아니다 — 단독 `--include AC_EXP03:...` 조회만 지원**한다.

## 산출물

```
Implicit-World-Modeling/outputs/{AndroidControl_EXP01|AndroidControl_EXP02|AndroidControl_EXP03|MC}/      # AC_EXP01 의 ratio (ratio37/ratio55/ratio73) 는 별도 디렉토리가 아니라 아래 {model}{SFX} 의 SFX 로 운반된다 (AC_EXP02/AC_EXP03/MC: SFX="", AC_EXP01: SFX=_ratio37/_ratio55/_ratio73).
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

`{slug}` 는 `ac-exp01-ratio{37,55,73}-` (AC_EXP01 ratio 별) / `ac-exp02-` (AC_EXP02) / `ac-exp03-` (AC_EXP03) / `mc-` (MC). `{E}` 는 `trainer_state.json.epoch` 의 `int(round(...))`. HF repo id 조립은 `scripts/_common.sh::hf_repo_id_stage1` / `hf_repo_id_stage2_base` / `hf_repo_id_stage2_world_model` 헬퍼에 단일화되어 있고, eval 시 사용할 model path 는 `resolve_eval_model_path {stage1|stage2_base|stage2_world}` 가 **local merged dir (`outputs/.../merged/.../epoch-{E}/`) 우선 + 위 HF repo id fallback** 으로 해석한다.

## 모델 추가 방법

새 모델 추가 시 다음을 동기화한다 (자세한 절차는 [`AGENTS.md`](./AGENTS.md)):

1. `implicit-world-modeling.ipynb` Cell 5 의 `_MODEL_CONFIG` 에 모델 항목 추가 (`MODEL_FAMILY_CONFIG` 에 family 가 이미 등록돼 있는지 확인).
2. `scripts/_common.sh` 의 `MODEL_ID`, `MODEL_TEMPLATE`, `ALL_MODELS` 에 동일 항목 추가.
3. 노트북 Section 0 의 "Stage {1,2} YAML 일괄 생성" 셀 (Cell 8 / 10) 재실행 → YAML 자동 생성. MC 는 Stage 1 전용이라 Stage 2 YAML 은 MC 에 대해 생성되지 않는다 (`_STAGE1_ONLY = {"MonkeyCollection"}` guard).

## 테스트 실행

Stage 2 Step Accuracy 채점 로직 (`scripts/_action_eval.py`) 회귀 테스트:

```bash
cd Implicit-World-Modeling
pytest tests/test_action_eval.py -v   # 52 케이스
```

## 코드 읽기 시작점

- [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) — 전체 파이프라인 기준
- [`scripts/_common.sh`](./scripts/_common.sh) — path / dataset / model / logging 규약, HF repo id 헬퍼
- [`scripts/split_data.py`](./scripts/split_data.py) — split 규칙 (AC_EXP01 / MC)
- [`scripts/mirror_exp03.py`](./scripts/mirror_exp03.py) — AC_EXP03 좌표 표현 미러 (AC_EXP01 ratio73 멤버십)
- [`scripts/_hungarian_eval.py`](./scripts/_hungarian_eval.py) — Stage 1 metric
- [`scripts/_action_eval.py`](./scripts/_action_eval.py) — Stage 2 metric
- [`scripts/eval_viewer.py`](./scripts/eval_viewer.py) — Stage 1/2 평가 결과 HTML 비교 뷰어

구조 설명은 [`ARCHITECTURE.md`](./ARCHITECTURE.md), 작업 규칙은 [`AGENTS.md`](./AGENTS.md) 를 본다.
