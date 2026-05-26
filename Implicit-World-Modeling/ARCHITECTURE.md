# Implicit-World-Modeling Architecture

`Implicit-World-Modeling` 은 모바일 GUI World Modeling 이 Action Prediction 성능에 주는 영향을 검증하는 2-stage fine-tuning 파이프라인이다. **2 개 Qwen3-VL Vision-Language 모델** (`Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`) 을 지원하며, conda env (`implicit-world-modeling`) + 노트북 [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) 가 오케스트레이션을 담당하고, [`scripts/`](./scripts) 가 반복 실행용 자동화 레이어다. 학습/export 는 conda env 에 `pip install -e ./LlamaFactory` 로 editable 설치된 LlamaFactory 가 수행한다. 모든 stage 의 흐름은 **`train → merge → eval`** 로 통일되며, merge 는 `--no-hf-upload` 로 local export 만 수행할 수 있다. eval 은 **local merged dir (`outputs/.../merged/.../epoch-{E}/`) 우선 + HF Hub merged repo fallback** (`_common.sh::resolve_eval_model_path`) — local merge 한 머신에서도 같은 머신 안에서 바로 eval 까지 이어 돌 수 있다.

---

## 0. Runtime Stack

```
conda env       notebook            엔진                              모델
─────────────   ─────────────────   ──────────────────────────────   ─────────────────────────────────
implicit-world-modeling  implicit-world-modeling.ipynb   llamafactory-cli train/export    Qwen3-VL-4B-Instruct
                                                          YAML: LlamaFactory/examples/      Qwen3-VL-8B-Instruct
                                                                custom/IWM-{DS}/
                                                                stage{1,2}_{full,lora}
```

- `pip install -e ".[llamafactory]"` + `pip install -e ./LlamaFactory` (editable 서브프로젝트)
- `transformers>=4.57.1,<4.58` (vllm 0.11.2 의 `transformers<5` 제약 + LlamaFactory 서브프로젝트 `<=5.2.0` 와의 교집합)
- `deepspeed`, `vllm`, `bitsandbytes` 모두 conda env `implicit-world-modeling` 에 설치된다.
- 평가 파이프라인 (`scripts/stage{1,2}_eval.sh`) 은 `LlamaFactory/scripts/vllm_infer.py` 가 HF 표준 safetensors / PEFT adapter 를 그대로 로드한다.

---

## 1. 실행 구조

### 핵심 엔트리포인트

- [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) — conda `implicit-world-modeling`, 2 개 Qwen3-VL 모델
  - 환경 설치, `_MODEL_CONFIG` (2) + `MODEL_FAMILY_CONFIG` + `_DATASET_CONFIG` (3 학습 DS) + `_SIZE_CONFIG_AC` (3 tier) 정의
  - Stage 1/2 학습 YAML 자동 생성 (Cell 8 = Stage 1, Cell 10 = Stage 2; full / lora 분리)
  - `LlamaFactory/data/dataset_info.json` 등록
  - Stage 1/2 학습·평가·merge 셀에서 `scripts/*.sh` 호출
- [`scripts/`](./scripts)
  - `stage{1,2}_{train,eval,merge}.sh`: `--model MODEL --dataset DS` 플래그 방식 CLI
  - 노트북에서 한 번 생성한 YAML 과 `dataset_info.json` 등록 결과를 재사용하는 반복 실행 경로
- [`LlamaFactory/`](./LlamaFactory) — `llamafactory-cli train` / `llamafactory-cli export`
  - [`LlamaFactory/scripts/vllm_infer.py`](./LlamaFactory/scripts/vllm_infer.py) (eval 스크립트가 `cd "$LF_ROOT" && python scripts/vllm_infer.py …` 로 호출)

### 노트북 섹션 매핑

| Section | 셀 인덱스 (대표) | 내용 |
|---------|-----------------|------|
| 0 | 0–10 | 환경, dataset / 모델 / family / size config, Stage 1·Stage 2 학습 YAML 일괄 생성 |
| 1 | 11–13 | Stage 1 ShareGPT 변환 + `dataset_info.json` 등록 |
| 2 | 14–17 | Stage 2 ShareGPT 변환 + ID/OOD app 분할 + 등록 |
| 3 | 18–84 | Stage 1 SFT (2 모델 × 3 DS × {full, lora}) — explicit per-cell |
| 4 | 85–151 | Stage 1 merge (모든 epoch local merge + 선택적 HF Hub push; `--no-hf-upload` 지원) |
| 5 | 152–159 | Stage 1 평가 — local merged 우선 + HF Hub fallback sweep, Hungarian metric |
| 6 | 160–186 | Stage 2 SFT (2 모델 × {AC_EXP01, AC_EXP02}) |
| 7 | 187–213 | Stage 2 merge (variant × 모든 epoch local merge + 선택적 HF push; `--no-hf-upload` 지원) |
| 8 | 214–218 | Stage 2 평가 — ID + OOD 동시 sweep, `action_metrics.json` 3 섹션 |

> **Stage 2 머지 YAML 은 사전 생성하지 않는다** — `scripts/stage{1,2}_merge.sh` 가 runtime 에 임시 YAML 을 만든다. `BEST_CHECKPOINT` 개념은 제거됐고 epoch 번호는 `trainer_state.json.epoch` 으로 결정된다.

---

## 2. 모델 설정

### 모델 레지스트리

`implicit-world-modeling.ipynb` Cell 5 의 `_MODEL_CONFIG` (2 모델) 와 `scripts/_common.sh` 의 `MODEL_ID` / `MODEL_TEMPLATE` / `ALL_MODELS` 가 동기화되어야 한다.

| short_name | model_id | template | size |
|------------|----------|----------|------|
| qwen3-vl-4b | Qwen/Qwen3-VL-4B-Instruct | qwen3_vl_nothink | 3-4B |
| qwen3-vl-8b | Qwen/Qwen3-VL-8B-Instruct | qwen3_vl_nothink | 7-9B |

> Qwen3-VL 의 `qwen3_vl_nothink` template 은 `vllm_infer.py` 호출 시 `_common.sh::build_infer_cmd` 가 `--enable_thinking False` 를 자동 주입해 thinking 트리거를 끈다.

### 모델 family 별 image budget

노트북 Cell 5 의 `MODEL_FAMILY_CONFIG` (factor / max_tokens / min_tokens) 와 `_DATASET_CONFIG[ds]["image_overrides"]` 의 token 단위 override 로 관리된다. token 예산은 **학습 데이터셋** 으로 결정 — 학습된 모델은 평가 데이터셋과 무관하게 학습 시 budget 으로 추론한다 (학습-추론 mismatch 방지).

| 학습 DS | max_tokens | Qwen3-VL (factor 32) |
|---|---|---|
| AC_EXP01, AC_EXP02, MC | 2,048 (family default) | 2,097,152 / 4,096 |

`min_tokens=4` 는 family 공통. YAML 의 `image_max_pixels` / `image_min_pixels` 는 CONFIGS 빌더가 family default 에 dataset override 를 token-aware 로 덮어써 자동 주입한다. 평가측 `scripts/_common.sh::build_infer_cmd` 는 `TRAIN_DATASET` 글로벌 (parse_args 에서 set) 로 학습 DS 를 식별해 동일 budget 을 적용한다.

> vLLM `gpu_memory_utilization` 은 `build_infer_cmd` 내부에서 기본 `0.80`, 환경변수 `VLLM_GPU_MEM_UTIL` 로 호출 단위 override. `build_infer_cmd` 는 `stage{1,2}_eval.sh` 양쪽에서 공통으로 호출되므로 stage1/2 모두 동일하게 적용된다 (예: 동일 GPU 에서 학습 병행 / OOM 마진 확보 시 `VLLM_GPU_MEM_UTIL=0.6 bash ./scripts/stage2_eval.sh ...`). 미설정 시 0.80 그대로.

`cutoff_len` 은 **AC_EXP01 / AC_EXP02 Stage 1 / Stage 2 모두 10000** 이다 (2026-05-13 16384 → 10000 하향). state+action ratio-mix 로 frame 이 다수 포함돼 Qwen3-VL multimodal RoPE position 길이가 8192 를 초과 (관측: 8521) 하는 샘플이 있어 학습이 첫 step 에서 shape mismatch 로 실패한다 — 10000 은 실측 분포 기준 multi-frame 안전 마진과 메모리/throughput 사이 tradeoff 를 잡은 운영 기준이다. 노트북 Cell 8 의 Stage 1/2 inline YAML 과 `LlamaFactory/examples/custom/IWM-AC_EXP01_*` yaml 모두 10000 으로 통일한다. (`scripts/filter_long_samples.py --threshold` 의 default 도 10000 으로 동기 — 사전 필터와 학습 cutoff 를 같은 기준으로 통일.)

### 하이퍼파라미터 — 3 단 머지 구조

CONFIGS 빌더가 다음 순서로 `dict.update()` 한다:

1. `_DATASET_CONFIG[ds].stage{1,2}` — 데이터셋 공통 baseline (AC_EXP01 / AC_EXP02 / MC).
2. `_SIZE_CONFIG_AC[size].stage{1, 1_lora, 2}` — **AC_EXP01 / AC_EXP02 공유** 모델 크기 공유값 (2B / 3-4B / 7-9B). MC 에는 적용되지 않는다.
3. `_MODEL_CONFIG[model].hparam_overrides` — 모델별 delta.

각 모델은 `_MODEL_CONFIG[model]["size"]` (`"2B" | "3-4B" | "7-9B"`) 필드로 tier 를 지정한다. MB 는 평가 전용이라 학습 하이퍼파라미터 해석에서 제외.

#### `_SIZE_CONFIG_AC` 값

**Stage 1 (full FT)** — dataset baseline 대비 다른 필드만:

| 구간 | lr | warmup_ratio | max_grad_norm |
|---|---|---|---|
| 2B | 1.5e-5 | 0.08 | 0.5 |
| 3-4B | 1.2e-5 | 0.06 | 0.5 |
| 7-9B | (baseline 유지: 1.0e-5 / 0.03 / 1.0) | | |

**Stage 1 LoRA** — `stage1_full` 위에 덮어쓰기:

| 구간 | lr | LoRA r / α | dropout |
|---|---|---|---|
| 2B | 1.5e-4 | 8 / 16 | 0.05 |
| 3-4B | 1.2e-4 | 12 / 24 | 0.05 |
| 7-9B | 1.0e-4 | 16 / 32 | 0.05 |

LoRA 모드의 `deepspeed` 필드는 `GPU_TYPE` 에 따라 분기된다 (Stage 1: Cell 9, Stage 2: Cell 13 — 동일 정책):

| GPU_TYPE | LoRA deepspeed config | 근거 |
|---|---|---|
| `RTX5090` (32GB) | `examples/deepspeed/ds_z3_offload_config.json` | 단일 GPU + 32GB VRAM 에서 7-9B OOM 회피 (ZeRO-3 + CPU offload) |
| `A100` / `H100` (80GB) | `examples/deepspeed/ds_z3_config.json` (= `stage1_deepspeed` 기본값) | 80GB 면 offload 불필요, 통신 오버헤드만 추가됨 |

Full FT (Stage 1 / Stage 2 양쪽) 는 분기 없이 모델별 `stage1_deepspeed` (기본 `ds_z3_config.json`) 그대로 사용. 분기 로직은 `lora` 모드에만 적용.

**Stage 2 (LoRA)** — dataset baseline 대비 다른 필드만:

| 구간 | lr | LoRA r / α | dropout | warmup_ratio |
|---|---|---|---|---|
| 2B | 6.0e-5 | 16 / 32 | 0.05 | 0.05 |
| 3-4B | 5.0e-5 | 24 / 48 | 0.05 | 0.04 |
| 7-9B | 4.0e-5 | (baseline: 32 / 64) | 0.05 | (baseline: 0.03) |

설계 근거: 과거 `outputs/AC/eval/qwen{2.5-vl-7b,3-vl-8b}/stage2_eval` 실측 (AC 학습 시기) 에서 lr 5e-5 가 7-9B 상단 경계, dropout 0.10 이 저빈도 action type 을 불안정하게 만듦. 2B / 3-4B 는 Stage 1 크기 규칙을 Stage 2 에 이식한 외삽.

#### 계열 delta (`_MODEL_CONFIG[model].hparam_overrides`)

| 계열 | stage1 / stage2 에 추가 |
|---|---|
| Qwen 계열 (8 모델) | (empty — 전부 tier 값 그대로) |

#### `per_device_train_batch_size` (size × GPU)

`_PER_DEVICE_BS_BY_SIZE[size][GPU_TYPE]` (Cell 5).

| 모델 size | RTX5090 (32GB) | A100 (80GB) | H100 (80GB) |
|-----------|----------------|-------------|-------------|
| 2B        | 4              | 8           | 8           |
| 3-4B      | 2              | 4           | 4           |
| 7-9B      | 1              | 2           | 2           |

#### `gradient_accumulation_steps` 불변식

```
global_batch = per_device_train_batch_size * gradient_accumulation_steps * NPROC_PER_NODE
            == GLOBAL_BATCH_SIZE  (기본 64)

gradient_accumulation_steps = GLOBAL_BATCH_SIZE / (per_device * NPROC_PER_NODE)
```

`NPROC_PER_NODE ∈ {1, 2, 4, 8}` 와 `GPU_TYPE ∈ {RTX5090, A100, H100}` 만 허용 — 다른 값은 `ValueError`. Cell 5 의 `_derive_grad_accum()` 이 역계산해 CONFIGS 의 `stage{1,2}.gradient_accumulation_steps` 에 주입한다. 위 표 값이 모든 (size, GPU, NPROC) 조합에서 64 로 나누어떨어지므로 silent rounding 은 발생하지 않는다.

---

## 3. 데이터와 설정 계약

### 데이터 디렉토리

```
data/
├── AndroidControl/                   # 원본 source 자산 (학습/평가 entry 아님 — AC_EXP01 / AC_EXP02 가 read)
│   ├── implicit-world-modeling_stage1.jsonl                                # 원천 (전체)
│   ├── implicit-world-modeling_stage1_state_pred.jsonl                     # 원천: state-transition task
│   ├── implicit-world-modeling_stage1_action_pred.jsonl                    # 원천: action-prediction task
│   ├── implicit-world-modeling_stage1_{state,action}_pred_filtered.jsonl   # filter_long_samples.py --dataset AC_EXP01 산출 (mm-expanded length > cutoff_len 제거)
│   ├── implicit-world-modeling_stage2.jsonl                                # 원천: Stage 2 action-prediction
│   ├── implicit-world-modeling_stage2_filtered.jsonl                       # filter_long_samples.py 산출
│   ├── episodes_meta.jsonl                 # primary_app = 전경 앱 package_name
│   └── images/                              # AC_EXP01 / AC_EXP02 가 JSONL `images` 필드로 공유 참조
├── AndroidControl_EXP01/                 # Stage 1 ratio mix 학습 + Stage 2 ratio sweep (split_data.py --dataset AC_EXP01 산출)
│   ├── implicit-world-modeling_stage1_train_{3_7,5_5,7_3}.jsonl            # ratio 별 (state:action) train
│   ├── implicit-world-modeling_stage1_test_{id,ood}_state_pred.jsonl       # state task, app-level partition
│   ├── implicit-world-modeling_stage1_test_{id,ood}_action_pred.jsonl      # action task, app-level partition
│   ├── implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl       # split_data.py 가 원본 _filtered 풀에서 산출, 15K / 3K / 3K (action_type stratified, Stage 1 action_pred app partition 공유)
│   # NOTE: images/ + episodes_meta.jsonl 은 원본 AndroidControl/ 의 것을 그대로 참조 (JSONL `images` 가 "AndroidControl/images/..." prefix).
├── AndroidControl_EXP02/                 # AC_EXP01 ratio73 + Stage 1 state-pred diff loss 실험군 (대조군 = AC_EXP01 ratio73)
│   ├── implicit-world-modeling_stage1_train.jsonl                          # AC_EXP01 ratio73 train + token_weights (scripts/diff_loss/preprocess_dataset.py)
│   ├── implicit-world-modeling_stage1_test_{id,ood}_{state,action}_pred.jsonl  # AC_EXP01 에서 복사 (동일 평가셋 — 공정 비교)
│   ├── implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl       # AC_EXP01 에서 복사 (Stage 2 는 diff loss 미적용)
│   # NOTE: images/ + episodes_meta.jsonl 은 원본 AndroidControl/ 의 것을 공유 참조.
├── MonkeyCollection/                 # Stage 1 전용 학습 + 평가
│   ├── implicit-world-modeling_stage1.jsonl              # 약 100K
│   ├── implicit-world-modeling_stage1_{train,test}.jsonl # split_data.py --dataset MC (95:5)
│   └── images/
└── MobiBench/                        # 평가 전용 벤치마크
    ├── implicit-world-modeling_stage1.jsonl              # stage1_eval.sh --eval-datasets MB
    ├── implicit-world-modeling_stage2.jsonl              # stage2_eval.sh --eval-datasets MB (single-pair)
    └── images/
```

### 분할 규칙

- **App partition (AC_EXP01 / AC_EXP02 공유 — 원본 `data/AndroidControl/episodes_meta.jsonl`)**: `compute_app_partition` 이 Stage 2 행 수를 budget 으로 (id_apps, ood_apps) 를 한 번 계산하고, Stage 1 은 같은 partition 으로 entries 를 라우팅. Stage 2 OOD 앱이 Stage 1 train 에도 포함되지 않아 world-modeling 학습이 OOD 앱을 한 번도 보지 않는다.
- **Stage 1 (MC)**: 메타 없음 → 자동 random split (`--stage1-ratio`, 기본 0.95). `_STAGE1_ONLY` guard 로 Stage 2 자동 skip.
- **AC_EXP01 (Stage 1 ratio mix + Stage 2 ratio sweep)**: split_data.py 의 source 는 항상 원본 `data/AndroidControl/` 이고, 산출물은 `data/AndroidControl_EXP01/` 에 쓰여진다. 선행으로 `scripts/filter_long_samples.py --dataset AC_EXP01` 가 mm-expanded length > `cutoff_len` 인 row 를 제거해 **원본 폴더 안에** `implicit-world-modeling_stage1_{state,action}_pred_filtered.jsonl` + `implicit-world-modeling_stage2_filtered.jsonl` (3 파일) 을 만든다 (Qwen3-VL `get_rope_index` broadcast 회피용). `run_exp01_split` 은 항상 Stage 1/Stage 2 모두 `_filtered` 만 입력으로 사용 — Stage 2 source 누락 시 hard-fail. 그 위에서 `state_pred` (random) + `action_pred` (action-type stratified) 두 풀을 ID/OOD 앱 partition 으로 라우팅 후 ratio (state:action ∈ {7:3, 3:7, 5:5}, default `7:3,3:7,5:5`) 로 혼합한 Stage 1 train 3 종 + (id, ood) × (state, action) 4 test 를 산출. 같은 (id_apps, ood_apps) 를 재사용해 Stage 2 split (`implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl`, 기본 15K / 3K / 3K, action_type stratified) 까지 함께 산출 — Stage 1 ↔ Stage 2 OOD app 집합 일치. **Stage 2 파이프라인은 ratio sweep 으로 활성** (`_STAGE1_ONLY = {"MonkeyCollection"}` 만 유지) — stage2 데이터는 ratio 무관 (3 ratio 공유) 이며, ratio 차원은 stage1 → stage2 base 계보 (Stage 1 ratio merged 가 Stage 2 world-model variant 의 base) 로만 흐른다. 산출 디렉토리/HF slug 는 ratio 별 분리 (`outputs/AndroidControl_EXP01/{adapters,merged,eval}/{MODEL}_ratio{37,55,73}_stage2_*`, `SaFD-00/{short}-ac-exp01-ratio{37,55,73}-...`). ratio 별로 학습 가중치가 다르므로 `--exp01-ratios` 가 sweep 단위, `--exp01-train-total` 이 Stage 1 train 합계 (기본 50K). Stage 2 학습 데이터의 last-message wrapping (`<thought>…</thought>\n<action>{...}</action>`) 은 `_parse_action_payload` regex helper 가 분리.
- **AC_EXP02 (Stage 1 state-pred diff loss 실험군)**: split 불필요. `scripts/diff_loss/preprocess_dataset.py` 가 AC_EXP01 ratio73 train (`implicit-world-modeling_stage1_train_7_3.jsonl`) 의 future HTML 토큰에 diff 가중치를 부여한 `token_weights` 필드를 추가 — current→future HTML diff 를 헝가리안 매칭으로 ADDED/MODIFIED/UNCHANGED 분류 (가중치 2.0/2.0/1.0). action_pred 샘플은 assistant 가 JSON 이라 diff element 0 개 → `token_weights` 전부 1.0 → 기존 cross-entropy 와 동치 (action 은 기존 loss). test / Stage 2 데이터는 AC_EXP01 에서 복사 (`DS_DATADIR[AC_EXP02]=AndroidControl_EXP02`, 노트북 환경 세팅 셀이 일괄 수행). diff loss 는 LlamaFactory 6 파일 패치 (`use_diff_token_weighted_loss` 인자 + `diff_token_weighted_loss_func` + collator 의 labels 기반 token_weights 복원) 에 의존 — LF 는 gitignore 된 별도 repo 라 `scripts/diff_loss/apply_llamafactory_patch.py` 가 멱등 재적용한다. (`scripts/diff_loss/` 의 `hungarian_metric.py` 는 채점용 `scripts/_hungarian_eval.py` 와 의도적으로 분리된 학습 전처리용 사본.)
- **MB**: split 없음. 평가 전용.

### `episodes_meta.jsonl` 스키마 (원본 `data/AndroidControl/`)

```jsonl
{"episode_id": 0, "goal": "...", "primary_app": "com.zoho.meeting", "actions": ["..."], "step_instructions": [...]}
```

- `episode_id` 는 **int** (0, 1, 2, ...). 원본 이미지 경로는 zero-padded string. `split_data.py::_norm_ep` 가 `str(int(...))` 로 정규화해 매칭.
- `primary_app` 은 각 step 의 `accessibility_trees` (`AndroidAccessibilityForest` proto) 에서 전경 `TYPE_APPLICATION` window 의 root `package_name` 을 뽑아 다수결로 정한 값. 시스템/런처 패키지는 다수결에서 제외.
- 전경 window 미검출 시 `primary_app=None`. 해당 에피소드는 train 풀에만 합류, test 분할에서 제외 (`--stage2-exclude-null-app` 으로 완전 제외 가능).
- 메타 추출: `scripts/extract_androidcontrol_metadata.py` (TFRecord → 다수결, `pip install android-env` 별도 필요). 스크린샷은 `extract_androidcontrol_images.py` 가 GCS REST API 로 pull (TF 의존 없음).

### 데이터셋 이름 규약

| 용도 | AndroidControl | AndroidControl_EXP01 | AndroidControl_EXP02 | MonkeyCollection | MobiBench |
|------|----------------|-------------------|-------------------|-------------------|-----------|
| `data/` 실제 디렉토리 | `AndroidControl` (원본 source-only) | `AndroidControl_EXP01` | `AndroidControl_EXP02` | `MonkeyCollection` | `MobiBench` |
| shell 단축 코드 | — (학습/평가 entry 없음) | `AC_EXP01` (ratio 별 가상 키 `AC_EXP01_ratio{37,55,73}` 으로 expand) | `AC_EXP02` | `MC` | `MB` (eval 전용) |
| LF dataset prefix | — | `IWM-AC_EXP01` (test 공유) + `..._train_ratio{37,55,73}` | `IWM-AC_EXP02` | `IWM-MC` | `IWM-MB` |
| `outputs/` 최상위 | — | `AndroidControl_EXP01` (ratio 는 아래 model dir 의 `_ratio{37,55,73}` suffix 로 운반) | `AndroidControl_EXP02` | `MC` | — (TRAIN_DS 산하 `on-MB/`) |
| test split | — | (id, ood) × (state, action) 4 파일 | (id, ood) × (state, action) 4 파일 (AC_EXP01 복사) | 단일 test | 단일 파일 |
| Stage 2 지원 | — | ✓ (ID/OOD 3 섹션, ratio sweep — stage1 → stage2 계보) | ✓ (ID/OOD 3 섹션, diff loss 미적용) | ✗ (데이터 없음, `_STAGE1_ONLY`) | ✓ (single-pair overall) |

### LLaMA-Factory 등록

- 노트북 Section 1-2 가 `LlamaFactory/data/dataset_info.json` 을 갱신 — 등록 분기는 세 직교 플래그 (`_STAGE1_ONLY`, `_SINGLE_TEST`, `_DUAL_TASK_TEST`) 로 결정:
  - **AC_EXP01** (`_DUAL_TASK_TEST`, ratio 변형 3 종): Stage 1 = `IWM-AC_EXP01_stage1_train_{r37,r55,r73}` (3) + `IWM-AC_EXP01_stage1_test_{id,ood}_{state,action}` (4) — test 4 파일은 ratio 변형 간 공유. Stage 2 = `IWM-AC_EXP01_stage2_{train,test_id,test_ood}` (3) — ratio 무관 공유. 총 10 entry (ratio variant 3 회 등록 시 train_{rXX} 만 ratio 별로 다름).
  - **AC_EXP02** (`_DUAL_TASK_TEST`, diff loss 실험군): AC_EXP01 와 동일 구조 — Stage 1 = `IWM-AC_EXP02_stage1_train` (1) + `IWM-AC_EXP02_stage1_test_{id,ood}_{state,action}` (4), Stage 2 = `IWM-AC_EXP02_stage2_{train,test_id,test_ood}` (3). 총 8 entry. train JSONL 만 `token_weights` 필드 포함 — `columns` 등록은 불필요 (converter 가 raw 필드를 `_token_weights` 로 직접 전달).
  - **MC** (`_STAGE1_ONLY` + `_SINGLE_TEST`): `IWM-MC_stage1_{train,test}` 2 entry. `_STAGE1_ONLY = {"MonkeyCollection"}` — AC_EXP01 는 더 이상 게이트되지 않는다.
  - **MB**: `_EVAL_ONLY_BENCHMARKS` 루프가 `IWM-MB_stage{1,2}` 단일 파일 entry 등록. `scripts/_common.sh::ensure_eval_only_dataset_info()` 가 source 시점에 idempotent 하게도 보장 → 노트북 미실행 환경에서도 MB 평가 성립.
- JSONL 파일 경로는 `../../data/{DATASET_NAME}/...` 형태의 **상대 경로** 로 등록.
- JSONL 내부 `images` 값은 `{DATASET_NAME}/images/...` 형태의 **상대 경로** 를 유지.
- `vllm_infer.py` 호출 시 `--dataset_dir` 에 **절대 경로** (`$LF_ROOT/data`) 를 전달해야 한다 — 상대 경로 사용 시 HF datasets 캐시 오염으로 `FileNotFoundError` 발생 가능.

---

## 4. 파이프라인 컴포넌트

### 로컬 오케스트레이션 레이어

- [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) — 전체 실험 실행의 기준 경로
- [`scripts/_common.sh`](./scripts/_common.sh) — 공통 path, `.env`, dataset 매핑, 모델 레지스트리, bash 4+ 가드, logging, HF repo id 헬퍼, `build_infer_cmd`
- [`scripts/split_data.py`](./scripts/split_data.py) — split (AC_EXP01 source=AndroidControl/ → output=AndroidControl_EXP01/, MC random)
- [`scripts/_hungarian_eval.py`](./scripts/_hungarian_eval.py) — Stage 1 metric (`score` 서브커맨드)
- [`scripts/_action_eval.py`](./scripts/_action_eval.py) — Stage 2 metric, ID/OOD/overall 3 섹션 산출 (`score` 서브커맨드)
- [`scripts/eval_viewer.py`](./scripts/eval_viewer.py) — Stage 1/2 산출물 비교 HTML 빌더. `--include EXP:MODEL` 다중 spec 으로 단일 EXP 자체 비교 (산출: `outputs/{DS_DATADIR(EXP)}/eval/{MODEL}/stage{N}_eval/`) 와 EXP 간 동급 stage cross-compare (산출: `outputs/_compare/stage{N}_eval/`) 를 단일 CLI 로 처리. logical key 별 `pairs_<key>.html` + `pairs_summary.md` 생성 (Stage 1: `on-AC-state-{id,ood}[-without-open_app]`, `on-AC-action-{id,ood}`, `on-MB`, `on-MB-without-open_app`, `on-MC`; Stage 2: `on-AC-{id,ood}`, `on-MB`).

### Stage 1 automation

`--stage1-mode {full|lora}` 로 finetuning 방식 선택 (기본: `full`). 모드별로 YAML 경로 · adapter 경로 · merged 경로 · HF Hub ID 가 모두 접미사로 분리되어 공존한다.

- **`scripts/stage1_train.sh`**
  - YAML: `examples/custom/IWM-{DS}/stage1_${MODE}/{MODEL}_world-model.yaml`
  - 실행: `FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=${NPROC_PER_NODE}` + `llamafactory-cli train`
  - full YAML 은 `finetuning_type: full`, lora YAML 은 `finetuning_type: lora` + `lora_rank/alpha/target/dropout` 블록 포함
- **`scripts/stage1_merge.sh`**
  - `outputs/{DS}/adapters/{MODEL}_stage1_${MODE}_world-model/checkpoint-*` 전수 loop. 각 ckpt 에서 `trainer_state.json.epoch` 을 `int(round(...))` 로 추출
  - 임시 merge YAML 생성 → `llamafactory-cli export` (`--no-hf-upload` 시 `export_hub_model_id` 생략)
  - 산출 (epoch 별): `outputs/{DS}/merged/{MODEL}_stage1_${MODE}_world-model/epoch-{E}/` + 선택적 HF Hub push `SaFD-00/...stage1-{MODE}-world-model-epoch{E}` (헬퍼: `_common.sh::hf_repo_id_stage1`)
  - **Skip 동작**: checkpoint 가 없는 슬롯은 `[WARN]` SKIP, 다음 슬롯 진행. 요약에 `merged / skipped / failed` 카운트.
- **`scripts/stage1_eval.sh`**
  - Phase A (baseline zero-shot) + Phase B (`--epochs` 정수 리스트로 **merged sweep**, 기본 `1,2,3`)
  - 각 (variant, epoch) 의 model path 는 `_common.sh::resolve_eval_model_path` 가 결정 — local merged dir (`outputs/{OUT_DS}/merged/{M}{SFX}_stage1_${MODE}_world-model/epoch-{E}/`) 이 존재하면 그것을, 없으면 HF Hub repo id (`hf_repo_id_stage1` 출력) 를 반환. `vllm_infer.py --model_name_or_path <local dir | HF repo id>` 둘 다 그대로 받는다 (merged 이므로 adapter 인자 / `max_lora_rank` 불필요).
  - 결과: `outputs/{DS}/eval/{MODEL}/stage1_eval/{base, ${MODE}_world-model/epoch-{E}}/{on-{EVAL_DS}, on-{EVAL_DS}-without-open_app}/`
  - 각 sweep 결과에 `_hungarian_eval.py score` → `hungarian_metrics.json` 저장
  - **without_open_app 자동 산출**: 정규 score 직후 추론 재실행 없이 `_hungarian_eval.py score --exclude-action open_app --filtered-test-dir data/{DATADIR} --filtered-pred-dir on-{EVAL_DS}-without-open_app/` 가 한 번 더 호출되어 GT `open_app` 행을 양쪽에서 동시 drop 한 메트릭 + 필터된 jsonl + `predict_results.json` 을 sibling 디렉토리에 idempotent 저장. 필터 test JSONL 은 `data/{DATADIR}/{prefix}_stage1{,_test{_id,_ood}}_without_open_app.jsonl` 에 영구 보존.
  - **재실행 시 skip**: marker `hungarian_metrics.json` 존재 unit 은 정규/필터 각각 독립 skip.
  - `--variants` 로 특정 variant 만 평가 가능 (기본 `base, full_world_model, lora_world_model`).

### Stage 2 automation

`--stage2-mode {full|lora}` (기본 `lora`) 로 학습 방식, `--stage1-mode {full|lora}` + `--stage1-epoch N` 으로 world-model variant 의 상류 소스 결정. base variant 는 Stage 1 무관.

- **`scripts/stage2_train.sh`**
  - YAML: `examples/custom/IWM-${DS}/stage2_${STAGE2_MODE}/{MODEL}_{base,world-model-full,world-model-lora}.yaml` (Cell 10 자동 생성)
  - **`FORCE_TORCHRUN` 미사용** (Stage 1 과 의도적으로 다름)
  - world-model variant: `--stage1-epoch N` 으로 지정된 local `merged/{M}_stage1_${STAGE1_MODE}_world-model/epoch-${N}/` 을 base 로 사용 (YAML `model_name_or_path` 런타임 sed 치환). 동시에 YAML `output_dir` 의 `__STAGE1_EPOCH__` 플레이스홀더가 `${N}` 으로 치환되어 stage2 결과가 `..._world-model_from_${STAGE1_MODE}-ep${N}/` 으로 분리 저장. 디렉토리 미존재 시 hard-fail.
- **`scripts/stage2_merge.sh`**
  - 각 variant 의 `adapters/{M}_stage2_${STAGE2_MODE}_{base|world-model_from_${STAGE1_MODE}-ep${STAGE1_EPOCH}}/checkpoint-*` 전수 loop
  - Full FT: checkpoint 자체가 전체 모델 → merge YAML 의 `model_name_or_path` 에 직접 전달 (adapter 블록 없음)
  - LoRA: `model_name_or_path: {base}` + `adapter_name_or_path: {ckpt}` + `finetuning_type: lora`
  - `--no-hf-upload` 시 merge YAML 에서 `export_hub_model_id` 를 생략해 local export 만 수행
  - HF 네이밍 (`_common.sh`):
    - base: `hf_repo_id_stage2_base(MODEL, DS, STAGE2_MODE, E2)` → `...base-stage2-{M2}-epoch{E2}`
    - world: `hf_repo_id_stage2_world_model(MODEL, DS, STAGE1_MODE, STAGE1_EPOCH, STAGE2_MODE, E2)` → `...world-model-stage1-{M1}-epoch{E1}-stage2-{M2}-epoch{E2}`
- **`scripts/stage2_eval.sh`**
  - `--variants` 로 `base`, `{full|lora}_base`, `{full|lora}_world_model` (CLI 토큰) 중 선택 평가. world-model variant 는 `--stage1-epoch` 로 계보 번호 주입 (local merged suffix + HF repo 이름 양쪽에 동일 적용). 출력 경로는 `..._world-model_from_{M1}-ep{E1}/epoch-{E2}/` (path 표기는 hyphen 정규화). 각 (variant, epoch) 의 model path 는 `_common.sh::resolve_eval_model_path` (`stage2_base` | `stage2_world`) 가 local merged dir 우선 + HF Hub fallback 으로 결정.
  - **`--epochs` 에 `0` 포함 시 (opt-in)**: `{full|lora}_world_model` 의 epoch-0 은 stage2 미학습 베이스라인 = stage1 merged 와 동일 모델로 해석 (`resolve_eval_model_path stage1 MODEL DS STAGE1_MODE STAGE1_EPOCH` → local `merged/{M}{SFX}_stage1_{M1}_world-model/epoch-{E1}/` 우선 + HF `...world-model-stage1-{M1}-epoch{E1}` fallback). full/lora 는 동일 모델이나 variant별 디렉토리(`epoch-0/`)를 각각 산출한다. `{full|lora}_base` 는 stage1 계보가 없어 epoch-0 = `base` variant 와 중복 → 경고 후 skip. 기본 `1,2,3` 에는 미포함.
  - `--train-dataset {AC_EXP01|AC_EXP02}` (MC 거절. AC_EXP01 는 `--exp01-ratio` 로 단일 ratio 지정) + `--eval-datasets LIST` (`AC_EXP01, AC_EXP02, MB`). EVAL_DS 별 분기:
    - **AC_EXP01 / AC_EXP02**: ID + OOD 두 test 파일 함께 추론 → `_action_eval.py score --test-id ... --pred-id ... --test-ood ... --pred-ood ...` 가 **overall / in_domain / out_of_domain** 3 섹션 기록.
    - **MB**: 단일 파일 1 회 추론 → single-pair `overall` 1 섹션.
  - 결과: `outputs/{OUT_DS}/eval/{MODEL}{SFX}/stage2_eval/{variant_path}[_from_{M1}-ep{E1}]/epoch-{E2}/on-{EVAL_DS}/` (AC_EXP02: OUT_DS=AndroidControl_EXP02, SFX=""; AC_EXP01: OUT_DS=AndroidControl_EXP01, SFX=`_ratio{37,55,73}`. variant_path 는 CLI VARIANT 의 `world_model` → `world-model` 치환).
  - **재실행 시 skip**: marker `action_metrics.json` 존재 unit 은 variant × EVAL_DS 조합 별로 독립 skip.

### Shell script CLI

```bash
# 학습/merge — --dataset {AC_EXP01|AC_EXP02|MC}. MB 거절. Stage 2 는 MC 미지원 (데이터 없음). --dataset all 은 지원하지 않음 (명시적 선택 필수). AC_EXP01 는 Stage 1/2 모두 ratio sweep.
bash scripts/stage1_train.sh --model qwen3-vl-4b --dataset MC --stage1-mode lora
bash scripts/stage1_merge.sh --model qwen3-vl-4b --dataset MC --stage1-mode lora --no-hf-upload
# AC_EXP01 는 ratio 3 종 자동 sweep (--exp01-ratios ratio55,ratio73 로 부분 실행).
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP01 --stage1-mode full
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_EXP01 --stage1-mode full
# AC_EXP02 (diff loss 실험군)
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP02 --stage1-mode full
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_EXP02 --stage1-mode full
# Stage 2 AC_EXP01 ratio sweep (Stage 1 ratio merged 를 base 로 stage2 데이터 학습)
bash scripts/stage2_train.sh --model qwen3-vl-8b --dataset AC_EXP01 \
     --stage1-mode full --stage1-epoch 1 --stage2-mode lora --exp01-ratios ratio37,ratio55,ratio73
bash scripts/stage2_merge.sh --model qwen3-vl-8b --dataset AC_EXP01 \
     --stage1-mode full --stage1-epoch 1 --stage2-mode lora --exp01-ratios ratio37,ratio55,ratio73

# 평가 — --train-dataset 로 HF repo, --eval-datasets 로 test 셋 (교차 평가).
# AC_EXP01 학습 모델은 --exp01-ratio 단일 (Stage 1: state + action 두 task 채점, Stage 2: id/ood 3 섹션).
bash scripts/stage1_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP01 --exp01-ratio ratio55 \
     --eval-datasets AC_EXP01,MB --epochs 1,2,3
bash scripts/stage2_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP01 --exp01-ratio ratio55 \
     --eval-datasets AC_EXP01,MB --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
     --variants base,full_world_model,lora_world_model --epochs 1,2,3
bash scripts/stage2_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP02 --eval-datasets AC_EXP02,MB \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
     --variants base,lora_base,lora_world_model --epochs 1,2,3
```

플래그:

**학습/merge (`stage{1,2}_{train,merge}.sh`)**: `--dataset {AC_EXP01|AC_EXP02|MC}` (MB 거절. `--dataset all` 은 지원하지 않음 — 명시적 DS 선택 필수) · `--stage1-mode {full|lora}` (기본 full) · `--stage2-mode {full|lora}` (stage2 전용, 기본 lora) · `--stage1-epoch N` (stage2 world-model 전용) · `--exp01-ratios LIST` (AC_EXP01 sweep, 기본 `r37,r55,r73`).

**평가 (`stage{1,2}_eval.sh`)**: `--train-dataset` (stage1: AC_EXP01|AC_EXP02|MC, stage2: AC_EXP01|AC_EXP02) · `--eval-datasets LIST` (stage1: AC_EXP01, AC_EXP02, MC, MB; stage2: AC_EXP01, AC_EXP02, MB. 기본 = train-dataset 단일값) · `--epochs LIST` (기본 1,2,3) · `--variants LIST` · `--exp01-ratio {r37|r55|r73}` (AC_EXP01 train 단일 ratio, 기본 r55).

---

## 5. 실행 데이터 흐름

```
raw JSONL + screenshots  (AndroidControl: 원본 source-only, AC_EXP01: Stage 1 ratio mix + Stage 2 ID/OOD, AC_EXP02: AC_EXP01 ratio73 diff-loss train + AC_EXP01 test 복사, MC: Stage1 전용, MB: eval-only 단일 파일)
  -> extract_androidcontrol_images.py     (data/AndroidControl/images/ : GCS TFRecord → PNG)
  -> extract_androidcontrol_metadata.py   (data/AndroidControl/episodes_meta.jsonl : primary_app = 전경 앱)
  -> filter_long_samples.py --dataset AC_EXP01   (data/AndroidControl/ 원본에 _filtered.jsonl 산출)
  -> split_data.py                        (AC_EXP01: source=data/AndroidControl/ → output=data/AndroidControl_EXP01/ : Stage1 ratio mix + Stage2 ID/OOD, 공통 partition | MC: source=output=data/MonkeyCollection/, Stage1 random)
                                          (MB: split 없음 / AC_EXP02: split_data.py 미지원, 별도 diff_loss preprocess 가 산출)
  -> dataset_info.json registration       (AC_EXP01: stage1 7 + stage2 3, AC_EXP02: stage1 5 + stage2 3, MC: 2 entry, MB: eval-only 2 entry)
  -> [per model] Stage 1 train  (mode1 ∈ {full, lora}, 학습 DS ∈ {AC_EXP01 × ratio, AC_EXP02, MC})
       → adapters/{OUT_DS}/{M}{SFX}_stage1_{mode1}_world-model/checkpoint-*/   (AC_EXP02/MC: SFX=""; AC_EXP01: SFX=_ratio{37,55,73})
  -> [per model] Stage 1 merge (모든 epoch 각각)
       → merged/{OUT_DS}/{M}{SFX}_stage1_{mode1}_world-model/epoch-{E1}/  +  HF Hub ...{slug}world-model-stage1-{mode1}-epoch{E1}
  -> [per model] Stage 1 eval (local merged 우선 + HF Hub fallback × cross-dataset)
       → eval/{OUT_DS}/{M}{SFX}/stage1_eval/{mode1}_world-model/epoch-{E1}/on-{EVAL_DS}/hungarian_metrics.json
       (EVAL_DS ∈ {AC_EXP01, AC_EXP02, MC, MB} — AC_EXP01/AC_EXP02 는 state/action dual-task ID/OOD, MC/MB 는 단일 파일)
       (user picks an epoch E1 → passes as --stage1-epoch to Stage 2)
  -> [per model] Stage 2 train  (mode2 ∈ {full, lora},  variant ∈ {base, world-model-{mode1}}, 학습 DS ∈ {AC_EXP01 × ratio, AC_EXP02})
       world-model base = merged/{OUT_DS}/{M}{SFX}_stage1_{mode1}_world-model/epoch-{E1}/   (local)
       → adapters/{OUT_DS}/{M}{SFX}_stage2_{mode2}_{base|world-model_from_{mode1}-ep{E1}}/checkpoint-*/
  -> [per model] Stage 2 merge (variant × 전 epoch)
       → merged/{OUT_DS}/{M}{SFX}_stage2_{mode2}_{variant}/epoch-{E2}/
       + HF Hub:
          base : ...{slug}base-stage2-{mode2}-epoch{E2}
          world: ...{slug}world-model-stage1-{mode1}-epoch{E1}-stage2-{mode2}-epoch{E2}
  -> [per model] Stage 2 eval (local merged 우선 + HF Hub fallback × cross-dataset)
       → eval/{OUT_DS}/{M}{SFX}/stage2_eval/.../epoch-{E2}/on-{EVAL_DS}/action_metrics.json
          EVAL_DS=AC_EXP01 / AC_EXP02: { overall, in_domain, out_of_domain }   (test_id + test_ood)
          EVAL_DS=MB:        { overall }                              (single-pair)
```

### 산출물 위치

모든 산출물은 `Implicit-World-Modeling/outputs/` 단일 루트 아래 **데이터셋 중심 + category 분리** 구조. merged/eval 은 `epoch-{E}/` 서브디렉토리로 epoch 별 분리. full/lora 산출물은 경로 접미사로 분리되어 공존.

```
Implicit-World-Modeling/outputs/{OUT_DS}/             # OUT_DS = AndroidControl_EXP01 | AndroidControl_EXP02 | MC. AC_EXP01 ratio (ratio37/ratio55/ratio73) 는 아래 {model}{SFX} 의 SFX 로 운반 (AC_EXP02/MC: SFX="", AC_EXP01: SFX=_ratio37/_ratio55/_ratio73)
├── adapters/
│   ├── {model}{SFX}_stage1_{full,lora}_world-model/
│   ├── {model}{SFX}_stage2_{full,lora}_base/
│   └── {model}{SFX}_stage2_{full,lora}_world-model_from_{full,lora}-ep{E1}/
├── eval/{model}{SFX}/
│   ├── stage1_eval/                                              # 각 variant 안에 on-{EVAL_DS}/ + on-{EVAL_DS}-without-open_app/ 쌍
│   │   ├── base/
│   │   ├── full_world-model/epoch-{E}/
│   │   └── lora_world-model/epoch-{E}/
│   └── stage2_eval/
│       ├── base/
│       ├── {full,lora}_base/epoch-{E}/
│       └── {full,lora}_world-model_from_{full,lora}-ep{E1}/epoch-{E2}/
└── merged/
    ├── {model}{SFX}_stage1_{full,lora}_world-model/epoch-{E}/
    ├── {model}{SFX}_stage2_{full,lora}_base/epoch-{E}/
    └── {model}{SFX}_stage2_{full,lora}_world-model_from_{full,lora}-ep{E1}/epoch-{E2}/
```

`BEST_CHECKPOINT` / `BEST_CHECKPOINT.json` 파일은 더 이상 생성되지 않는다.

### HuggingFace 업로드 ID 패턴 (epoch 별 개별 repo)

| Stage / variant | 패턴 |
|-------|------|
| Stage 1 (full FT) | `SaFD-00/{short}-{slug}world-model-stage1-full-epoch{E}` |
| Stage 1 (LoRA)    | `SaFD-00/{short}-{slug}world-model-stage1-lora-epoch{E}` |
| Stage 2 base      | `SaFD-00/{short}-{slug}base-stage2-{M2}-epoch{E2}` |
| Stage 2 world     | `SaFD-00/{short}-{slug}world-model-stage1-{M1}-epoch{E1}-stage2-{M2}-epoch{E2}` |

`{slug}` 는 `ac-exp01-ratio37-` · `ac-exp01-ratio55-` · `ac-exp01-ratio73-` (AC_EXP01 ratio 별) / `ac-exp02-` (AC_EXP02) / `mc-` (MC). MB slug `mb-` 는 학습 대상이 아니므로 dormant. `{E}` 는 `trainer_state.json.epoch` 의 `int(round(...))`. HF repo id 조립은 `_common.sh::hf_repo_id_stage1` / `hf_repo_id_stage2_base` / `hf_repo_id_stage2_world_model` 헬퍼에 단일화되어 있고, eval 시 model path 는 `resolve_eval_model_path {stage1|stage2_base|stage2_world}` 가 **local merged dir (`outputs/.../merged/.../epoch-{E}/`) 우선 + 위 HF repo id fallback** 으로 해석한다.

---

## 6. 메트릭

자동 winner 선정은 없다. Stage 1/2 모두 `score` 서브커맨드로 평가 결과를 JSON 으로 저장하고, 사용자가 결과를 보고 Stage 2 에 쓸 Stage 1 epoch 을 `--stage1-epoch` 로 지정한다.

### Stage 1

- baseline: zero-shot (variant `base`)
- 변형: `full_world_model`, `lora_world_model`
- metric: `avg_hungarian_f1`, `avg_bleu`, `avg_rouge_l` 등
- 저장: `outputs/{DS}/eval/{MODEL}/stage1_eval/{variant}[/epoch-{E}]/on-{EVAL_DS}/hungarian_metrics.json`
- single-pair (`--test/--pred`) 와 ID/OOD (`--test-id/--pred-id/--test-ood/--pred-ood`) 모드 모두 지원 — ID/OOD 모드는 `overall` / `in_domain` / `out_of_domain` 3 섹션 기록.
- **AC_EXP01 / AC_EXP02 dual-task 분기 (Stage 1 한정)**: Stage 1 의 EVAL_DS=AC_EXP01 / AC_EXP02 일 때는 state_pred / action_pred 두 task 를 각각 독립 채점하여 `on-{DS}-state/hungarian_metrics.json` (Stage1 채점, `_hungarian_eval.py`) + `on-{DS}-action/action_metrics.json` (Stage2 채점, `_action_eval.py`) 두 산출물을 만든다. AC_EXP01 ratio 차원은 학습 산출물 (TRAIN_DS=`AC_EXP01_ratio{37,55,73}`) 에 박혀있고 test 4 파일은 ratio 와 무관. without_open_app sibling 은 state branch 만 (action 채점기 미지원). Stage 2 의 EVAL_DS=AC_EXP01 / AC_EXP02 는 dual-task 가 아니라 일반 action prediction 으로 ID + OOD 두 파일 (`implicit-world-modeling_stage2_test_{id,ood}.jsonl`) 을 함께 채점 — 3 섹션 (`overall` / `in_domain` / `out_of_domain`).

### Stage 2

- baseline: zero-shot (variant `base`)
- 변형: `{full|lora}_base`, `{full|lora}_world_model` (world-model 은 `--stage1-epoch` 로 상류 epoch 지정)
- 평가 파일: EVAL_DS=AC_EXP01 / AC_EXP02 는 `implicit-world-modeling_stage2_test_{id,ood}.jsonl` (3 섹션, AC_EXP01 는 ratio 무관 공유, AC_EXP02 는 AC_EXP01 복사본). MB 는 `implicit-world-modeling_stage2.jsonl` (single-pair).
- metric (3 섹션 또는 1 섹션): `action_metrics.json` 내부 `overall` / `in_domain` / `out_of_domain` 각각에 `step_accuracy`, `macro_step_accuracy`, `parse_rate`, `type_accuracy`, `cond_{index,dir,app,text}_acc`, `per_type[]` 포함.

#### `action_metrics.json` 스키마 예시

```json
{
  "overall": {
    "total": 6000,
    "parse_rate": 0.97,
    "type_accuracy": 0.81,
    "step_accuracy": 0.63,
    "macro_step_accuracy": 0.55,
    "cond_index_acc": 0.62,
    "cond_dir_acc": 0.73,
    "cond_app_acc": 0.59,
    "cond_text_acc": 0.48,
    "per_type": {
      "click":    {"count": 3337, "type_acc": 0.89, "step_acc": 0.61},
      "scroll":   {"count": 708,  "type_acc": 0.92, "step_acc": 0.73},
      "open_app": {"count": 365,  "type_acc": 0.78, "step_acc": 0.59},
      "input_text": {"count": 401, "type_acc": 0.71, "step_acc": 0.48},
      "finish":   {"count": 987,  "type_acc": 0.72, "step_acc": 0.72}
    }
  },
  "in_domain":     { "total": 3000, "step_accuracy": 0.68, "...": "..." },
  "out_of_domain": { "total": 3000, "step_accuracy": 0.58, "...": "..." }
}
```

`overall` 은 id + ood 를 단순 concat 해 재집계한 결과 — `overall.total == in_domain.total + out_of_domain.total` 이 항상 성립. ID/OOD gap (`in_domain.step_accuracy - out_of_domain.step_accuracy`) 이 앱 일반화 정도를 나타낸다.

#### Step Accuracy (SA) 정의

AndroidControl 데이터셋은 GT 에 `bounds` 필드가 영구 부재하고 element-index 기반 grounding 을 사용한다. IoU 기반 채점은 구조적으로 0 이 되므로, Stage 2 평가는 다음 정의를 따른다.

```
SA = (1/N) · Σ correct_i

correct_i = 1 iff (parse_ok ∧ type==gt.type ∧ field_match(type))
         = 0 otherwise
```

GT 의 `action_type` 키로 type 판정 (구 `type` 키 fallback 유지). GT last-message
가 `<thought>…</thought>\n<action>{...}</action>` 로 래핑되므로 GT 도
`parse_action` 으로 `<action>` JSON 을 추출한 뒤 채점한다 (pred 와 동일 경로).

| GT action_type | field_match 조건 |
|---|---|
| `navigate_back`, `navigate_home`, `wait` | (검증 필드 없음) → 항상 통과 |
| `finish` | (status / answer 비교 안 함) → 항상 통과 |
| `click`, `long_press` | `str(pred.index) == str(gt.index)` |
| `scroll` | `norm(direction)` 일치 (index 무시) |
| `open_app` | `norm(app_name)` 일치 (params 평탄화 fallback 허용) |
| `input_text` | `norm(text)` 일치 (index 무시) |

`norm(s) = str(s or '').strip().lower()` — 모든 string field 통일.

`action_metrics.json` 각 섹션의 키:
- 1차: `step_accuracy`
- 보조: `macro_step_accuracy` (등장 action_type 평균, 데이터셋 9 type), `parse_rate`, `type_accuracy`, `cond_index_acc` / `cond_dir_acc` / `cond_app_acc` / `cond_text_acc`, `per_type[t] = {count, type_acc, step_acc}`

Reference baselines (해석용):
- `action_type` random baseline: 1/9 ≈ 11.1%
- `scroll` majority baseline (`down`): 79.0%
- `finish` constant baseline: 100% (type-only 정책이라 해석 무의미)

정본은 `scripts/_action_eval.py` 이며, 노트북 Section 8 의 마지막 reference cell 이 이 파일과 글자 단위 동치를 유지한다 (디버깅 reference 용). 회귀 테스트 `tests/test_action_eval.py` 52 케이스 — `parse_action` (`<thought>/<action>` 래퍼 포함) / `evaluate_single` (AndroidControl_EXP01 `action_type` 스키마 + 구 `type` fallback) / `evaluate_predictions` 분기, unknown type 집계, `cond_*` n=0, `predict`/`output` fallback, ID+OOD 통합 집계 커버.

---

## 7. 중요한 운영 제약

- `implicit_world_modeling/` 패키지에는 핵심 파이프라인 로직이 없다. 변경 작업은 노트북, shell script, custom YAML 경로를 우선 검토.
- merge 스크립트는 `outputs/{DS}/adapters/.../checkpoint-*` 가 하나라도 없으면 `[WARN]` SKIP (전 epoch loop). 실패가 아니라 스킵이므로 sweep 친화.
- Stage 2 train/merge (world-model variant) 는 `--stage1-epoch N` 으로 지정된 로컬 `outputs/{OUT_DS}/merged/{MODEL}{SFX}_stage1_{full|lora}_world-model/epoch-${N}/` 이 반드시 선행돼야 한다 (stage1_train → stage1_merge; AC_EXP01 ratio variant 는 SFX=`_ratio{37,55,73}`). Stage 2 eval 은 local merged dir 우선 + HF Hub merged repo fallback (`_common.sh::resolve_eval_model_path`) 로 model path 를 잡으며, `--stage1-epoch` 값은 world-model 계보 식별자로 양쪽 (local dir suffix + HF repo 이름) 에 동일하게 주입.
- merge / eval 스크립트는 Python `pyyaml` 을 전제한다. `HF_TOKEN` 은 HF Hub push 또는 HF fallback pull 시 필요하며, merge 를 `--no-hf-upload` 로만 수행하고 같은 머신에서 eval (local merged dir hit) 만 한다면 불필요하다.
- shell automation 은 bash 4+ 환경 요구.
- 모델 추가 시 `implicit-world-modeling.ipynb` 의 `_MODEL_CONFIG` 와 `_common.sh` `MODEL_ID` / `MODEL_TEMPLATE` / `ALL_MODELS` 를 동시에 동기화. 새 family 라면 노트북 Cell 5 의 `MODEL_FAMILY_CONFIG` 에 image budget 도 추가.
- **transformers 버전**: `setup.py::EXTRAS["llamafactory"]` 에서 `transformers>=4.56.0,<4.57` 로 고정. `pyproject.toml` 의 주석도 이와 일치. 두 파일을 함께 변경한다. 서브프로젝트 `LlamaFactory/pyproject.toml` 은 수정하지 않는다.
- trl 0.24 / transformers 4.56+ API 매핑: `SFTConfig(max_length=...)`, `SFTTrainer(processing_class=...)` 사용. 구버전 키 (`max_seq_length`, `tokenizer=`, `overwrite_output_dir`) 는 `TypeError`.
- `gradient_checkpointing` 은 모델 로드 단계에서만 적용. `SFTConfig` 에는 전달하지 않는다 (이중 적용 방지).
- Full FT 분기에서 `freeze_vision_tower: true` 면 `vision_tower|vision_model|visual|image_encoder` 키워드를 포함한 named parameter 의 `requires_grad=False` 처리 후 frozen 텐서 수/파라미터 수를 stderr 로 출력.
