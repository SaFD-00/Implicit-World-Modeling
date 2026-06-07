# AGENTS.md

`Implicit-World-Modeling/` 프로젝트에서 작업하는 에이전트를 위한 작업 지침. 사용자 가이드는 [`README.md`](./README.md), 시스템 레퍼런스는 [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## 현재 코드 기준 요약

- 실행 엔트리포인트는 단일 노트북 [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) 와 [`scripts/`](./scripts) 다. 노트북은 conda env (`implicit-world-modeling`, `pip install -e ".[llamafactory]"`) 를 전제로 한다.
- **2 개 Vision-Language 모델 (모두 7-9B tier)**: `Qwen/Qwen3-VL-8B-Instruct` (`template=qwen3_vl_nothink`), `Qwen/Qwen2.5-VL-7B-Instruct` (`template=qwen2_vl`, factor 28 → image budget 1,605,632).
- **3 학습 데이터셋 + 1 평가 전용 벤치마크**: AC_EXP01, AC_EXP02, MC 가 학습 대상, MB 는 평가 전용. AC (AndroidControl) 자체는 학습/평가 entry 가 아니라 AC_EXP01/AC_EXP02 의 **원본 source 자산** (이미지 + 원본 jsonl + `episodes_meta.jsonl`) 으로만 보존된다. AC_EXP01 는 Stage 1 ratio mix (state_pred:action_pred = 3:7, 5:5, 7:3) 3 종을 sweep 해 별개의 가중치를 산출하고, **Stage 2 도 같은 ratio sweep 으로 활성** — Stage 1 ratio merged 를 base 로 같은 stage2 데이터 (`implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl`) 를 학습. stage2 데이터 자체는 ratio 와 무관 (3 ratio 공유) 이며, ratio 차원은 **stage1 → stage2 base 계보** 로만 흐른다. `_STAGE1_ONLY = {"MonkeyCollection"}` 로 축소되어 AC_EXP01 는 Stage 1/2 모두 파이프라인에 참여한다.
- **모든 stage 의 흐름은 `train → merge → eval`** 로 통일. eval 은 **로컬 `outputs/.../merged/.../epoch-{E}/` 가 있으면 그것을 우선 사용하고, 없을 때만 HF Hub merged repo 를 pull** (`_common.sh::resolve_eval_model_path`) — `--no-hf-upload` 로 local merge 만 한 경우와 HF push 후 다른 머신에서 재실행하는 경우 모두 동작한다.
- **GPU-aware `per_device_train_batch_size`**: `.env` 의 `GPU_TYPE` (`RTX5090` / `A100` / `H100`) 와 모델 size 로 `_PER_DEVICE_BS_BY_SIZE` 표를 조회 (Cell 5 의 `lf_per_device_bs(size)` 헬퍼). `NPROC_PER_NODE ∈ {1,2,4,8}` 만 허용. 4 가지 GPU 수 모두에서 `GLOBAL_BATCH_SIZE=64` 가 정수로 나뉘도록 표 값을 유지해야 한다.
- 모델 레지스트리는 두 곳에 있다: 노트북 Cell 5 의 `_MODEL_CONFIG` 와 `scripts/_common.sh` 의 `MODEL_ID` / `MODEL_TEMPLATE` / `ALL_MODELS`. 두 곳을 동시에 수정해야 한다.
- 모델 family 별 image budget 은 노트북 Cell 5 의 `MODEL_FAMILY_CONFIG` (factor / max_tokens / min_tokens) 와 `_DATASET_CONFIG[ds]["image_overrides"]` 의 token 단위 override 로 관리된다. 모든 학습 DS 는 family default `max_tokens=2048` (Qwen3-VL → 2,097,152) 를 쓴다. 평가는 `TRAIN_DATASET` 으로 동일 budget 적용.
- 학습 / export 는 conda env 에 `pip install -e ./LlamaFactory` 로 editable 설치된 `LlamaFactory/` clone + `llamafactory-cli` 가 수행.
- 평가 (`scripts/stage{1,2}_eval.sh`) 는 `vllm_infer.py` 가 HF 표준 safetensors / PEFT adapter 를 그대로 로드.
- [`implicit_world_modeling/`](./implicit_world_modeling) 패키지는 사실상 배포용 스텁이며, 핵심 파이프라인 로직은 여기에 없다.
- **데이터셋 역할 분리**:
  - 학습 대상 DS: `AndroidControl_EXP01` (AC_EXP01), `AndroidControl_EXP02` (AC_EXP02), `MonkeyCollection` (MC). `AndroidControl` (AC) 은 EXP01/EXP02 의 source 자산으로만 사용 (학습/평가 entry 아님).
  - `MobiBench` (MB) 는 **평가 전용 벤치마크**. 학습/merge 스크립트에서 `--dataset MB` 는 `parse_args` 에서 거절된다.
  - MC 는 Stage 1 전용 (Stage 2 데이터 자체가 없음, `_STAGE1_ONLY = {"MonkeyCollection"}`). AC_EXP01 는 Stage 2 도 활성 — `split_data.py::run_ac3_split` 이 Stage 1 action_pred app partition 을 그대로 재사용해 `implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl` (15K/3K/3K) 까지 만들고, stage2 학습은 ratio 별 Stage 1 merged 를 base 로 같은 stage2 데이터를 그대로 사용.
  - **AC_EXP01 는 dual-task** (`_DUAL_TASK_TEST` flag, Stage 1 한정): Stage 1 은 `state_pred` (Stage1 채점, `_hungarian_eval.py`) + `action_pred` (Stage2 채점, `_action_eval.py`) 두 task 를 비율 (3:7, 5:5, 7:3) 로 혼합한 train + (id, ood) × (state, action) 4 test. ratio 별로 **별개의 학습 가중치** 가 산출되며 (`AC_EXP01_ratio37`, `AC_EXP01_ratio55`, `AC_EXP01_ratio73`) HF slug 도 ratio 별 (`ac-exp01-ratio37-` 등). Stage 2 는 dual-task 가 아니라 일반 action prediction (id/ood 2 파일) 이며 ratio 차원은 stage1 → stage2 계보로만 흐른다. 평가 sweep 은 단일 ratio (`--exp01-ratio ratio55` 기본). `--dataset AC_EXP01` 는 ratio 3 종을 자동 expand (`--exp01-ratios ratio55,ratio73` 로 부분 실행).
  - 평가 스크립트는 `--train-dataset {AC_EXP01|AC_EXP02|MC}` + `--eval-datasets AC_EXP01,AC_EXP02,MC,MB` 로 학습 DS 와 평가 DS 를 분리. Stage 2 eval 은 `AC_EXP01 | AC_EXP02` (MC 미지원 — 데이터 없음). AC_EXP01 는 `--exp01-ratio` 로 학습 모델 ratio 를 단일 지정.

## 어디를 수정해야 하는가

### 모델 추가
1. 노트북 Cell 5 `_MODEL_CONFIG` 에 모델 항목 추가 (필드: `model_id`, `short_name`, `template`, `size` (현재 `"7-9B"` 단일 tier), image-pixel 은 `_img_cfg(short)` 헬퍼로 family config 에서 자동 주입).
2. `scripts/_common.sh` 의 `MODEL_ID` / `MODEL_TEMPLATE` / `ALL_MODELS` 에 동일 항목 추가.
3. 새 family 라면 노트북 Cell 5 의 `MODEL_FAMILY_CONFIG` 에 image budget 추가 (`factor` / `max_tokens` / `min_tokens`). family default 는 `max_tokens=2048`, `min_tokens=4`. `scripts/_common.sh::build_infer_cmd` 의 template 분기 (factor / mm_min) 도 함께 갱신.
4. 노트북 Section 0 의 "Stage {1,2} YAML 일괄 생성" 셀 (Cell 8 / Cell 10) 재실행 → Stage 1 YAML **full / lora 두 벌**, Stage 2 YAML **full / lora 두 벌 × {base, world-model-full, world-model-lora}** 자동 생성. shell 스크립트의 `--stage1-mode`, `--stage2-mode` 로 full/lora 분기.

### 하이퍼파라미터 수정
- AC_EXP01 / AC_EXP02 는 `_SIZE_CONFIG_AC[size].stage{1, 1_lora, 2}` 로 **7-9B 단일 tier** 공유값 관리하나, **현재 7-9B 의 세 키는 모두 빈 dict** 이라 dataset baseline 을 그대로 쓴다 (EXP01/EXP02 실측 어댑터와 동일조건 보존). 정본: Stage 1 LoRA `8/16 @1.0e-5 dropout0.05`, Stage 2 LoRA `32/64 dropout0.1 @5.0e-5`. `_MODEL_CONFIG[model].hparam_overrides` 는 모델별 delta 전용. lr / warmup / LoRA rank / dropout 은 `_MODEL_CONFIG` 에 직접 쓰지 말고 `_DATASET_CONFIG` baseline (또는 `_SIZE_CONFIG_AC` tier) 에서 바꾼다.
- MC 는 tier 미적용 — dataset baseline + per-model override 만 적용.
- MB 는 평가 전용이라 학습 하이퍼파라미터 해석에서 제외.
- merge 순서 (단일 진실원: 노트북 Cell 5 CONFIGS 빌더): `_DATASET_CONFIG` baseline → `_SIZE_CONFIG_AC[size]` (AC_EXP01 / AC_EXP02 일 때만) → `hparam_overrides`. 전체 표는 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §2.

### Notebook 실행 순서나 YAML 생성 흐름
- [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) 와 [`scripts/stage1_*.sh`](./scripts/stage1_train.sh) / [`scripts/stage2_*.sh`](./scripts/stage2_train.sh) 를 함께 맞춘다.
- 노트북 Cell 8 (Stage 1 YAML) / Cell 10 (Stage 2 YAML) 이 학습 YAML 생성. **Merge YAML 은 사전 생성하지 않는다** — `stage{1,2}_merge.sh` 가 runtime 에 임시 YAML 을 만든다.
- 노트북 Section 3 / 4 / 6 / 7 은 **단일 변형 walkthrough** 다 — Section 3 / 4 는 `qwen3-vl-8b` + `--stage1-mode full`, Section 6 / 7 은 `qwen3-vl-8b` + `--stage2-mode lora`. 다른 모델 / 모드 / DS 는 cell 을 추가하지 말고 shell 인자 (`--model`, `--stage1-mode`, `--stage2-mode`, `--dataset`) 만 바꿔 실행한다 (모델 레지스트리는 `_common.sh::MODELS`). `--dataset` 은 명시적 선택 필수 (`--dataset all` 은 지원하지 않음). Section 5 / 8 의 평가 cell (variant matrix · plot) 은 그대로 유지한다.

### 데이터 분할 규칙
- [`scripts/split_data.py`](./scripts/split_data.py) 가 기준. AC_EXP01 / AC_EXP02 의 **source 는 항상 원본 `data/AndroidControl/`**, 산출물은 각각 `data/AndroidControl_EXP01/` / `data/AndroidControl_EXP02/` 로 쓰여진다 (`SOURCE_DIR` / `OUTPUT_DIR` 두 dict). MC 는 Stage 1 random split (메타 없음, 자동 fallback), MB 는 split 없음.
- AC 원본 메타데이터: [`scripts/extract_androidcontrol_metadata.py`](./scripts/extract_androidcontrol_metadata.py) 가 `data/AndroidControl/episodes_meta.jsonl` 생성 (`pip install android-env` 별도 필요). 스크린샷은 [`scripts/extract_androidcontrol_images.py`](./scripts/extract_androidcontrol_images.py) 가 GCS REST API 로 `data/AndroidControl/images/` 에 pull.
- AC_EXP01 분할: 선행으로 `python scripts/filter_long_samples.py --dataset AC_EXP01` 가 mm-expanded length > cutoff_len 샘플을 제거해 원본 `data/AndroidControl/` 안에 `implicit-world-modeling_stage1_{state,action}_filtered.jsonl` + `implicit-world-modeling_stage2_filtered.jsonl` (3 파일) 을 만든다 (Qwen3-VL `get_rope_index` broadcast 회피). `--image-max-pixels` 기본값 2097152 는 Qwen3-VL family (factor 32) 기준 — Qwen2/2.5-VL 학습 시 1605632 등으로 override 필요. `--skip-existing` 으로 누락된 source 만 처리 가능. 그 위에서 `split_data.py --dataset AC_EXP01 --exp01-ratios 7:3,3:7,5:5 --exp01-train-total 50000` 가 원본 `data/AndroidControl/` 에서 read → `data/AndroidControl_EXP01/` 에 `implicit-world-modeling_stage1_train_{3_7,5_5,7_3}.jsonl` 3 개 + task × split 4 test (`implicit-world-modeling_stage1_test_{id,ood}_{state,action}.jsonl`) 를 산출하고, 이어서 같은 (id_apps, ood_apps) partition 으로 Stage 2 split `implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl` (15K / 3K / 3K, action_type stratified) 까지 만든다. `state_pred` 는 random, `action_pred` / Stage 2 는 action-type stratified 샘플링.

### Stage 1 평가
- [`scripts/_hungarian_eval.py`](./scripts/_hungarian_eval.py) 가 기준 (`score` 서브커맨드만 유지). single-pair (`--test/--pred`) 와 ID/OOD (`--test-id/--pred-id/--test-ood/--pred-ood`) 두 모드 지원 — ID/OOD 모드는 `hungarian_metrics.json` 에 `overall` / `in_domain` / `out_of_domain` 3 섹션 기록.
- 흐름: **`train → merge → eval`** — `stage1_merge.sh` 가 모든 epoch 를 각각 local merge + HF push (`trainer_state.json.epoch` 파싱), `stage1_eval.sh` 가 `--train-dataset {AC_EXP01|AC_EXP02|MC}` / `--eval-datasets {AC_EXP01,AC_EXP02,MC,MB}` / `--variants` / `--epochs` 로 지정된 merged 모델 (local `outputs/.../merged/.../epoch-{E}/` 우선, 없으면 HF Hub `SaFD-00/{short}-{slug}world-model-stage1-{MODE}-epoch{E}`) 을 사용해 EVAL_DS 별 test JSONL 에 대해 `hungarian_metrics.json` 산출. 경로 해석은 `_common.sh::resolve_eval_model_path` 헬퍼에 단일화.
- EVAL_DS 별 분기:
  - **EVAL_DS=AC_EXP01 / AC_EXP02**: state_pred + action_pred **두 task 를 각각 독립 채점**. 각 task 가 (id, ood) 2 파일을 가지므로 inference 4 회 → state 산출 `on-{DS}-state/hungarian_metrics.json` (Stage1 채점, `_hungarian_eval.py`), action 산출 `on-{DS}-action/action_metrics.json` (Stage2 채점, `_action_eval.py`). without_open_app sibling 은 state branch 만 (action 채점기 미지원).
  - **EVAL_DS=MC**: 단일 파일 `implicit-world-modeling_stage1_test.jsonl` (random split) → single-pair overall.
  - **EVAL_DS=MB**: 단일 파일 `implicit-world-modeling_stage1.jsonl` (벤치마크 단일 파일) → single-pair overall.
- 산출 경로: `outputs/{TRAIN_DS}/eval/{MODEL}/stage1_eval/{variant_path}[/epoch-{E}]/on-{EVAL_DS}/` (variant_path 는 CLI VARIANT 의 `world_model` → `world-model` 치환: 예 `full_world-model`, `lora_world-model`). 어떤 epoch 을 쓸지는 사용자가 결과를 보고 수동 결정 (자동 winner 선정 없음).
- 재실행 시 marker (`hungarian_metrics.json`) 존재 unit 은 skip. 정본은 노트북 Section 5. 시각 비교는 [`scripts/eval_viewer.py`](./scripts/eval_viewer.py) — `--include EXP:MODEL` 다중 spec 으로 단일 EXP 자체 비교 와 EXP 간 동급 stage cross-compare 를 동일 CLI 로 처리하며, cross-compare 산출은 `outputs/_compare/stage{N}_eval/` 로 분리.
- **without_open_app 자동 산출**: 각 `(variant, EVAL_DS)` 마다 정규 score 직후 추론 재실행 없이 `_hungarian_eval.py score --exclude-action open_app --filtered-test-dir data/{DATADIR} --filtered-pred-dir on-{EVAL_DS}-without-open_app/` 가 한 번 더 호출되어 GT `## Action.type=="open_app"` 행을 양쪽에서 동시 drop 한 메트릭 + 필터된 jsonl + `predict_results.json` 을 sibling `on-{EVAL_DS}-without-open_app/` 에 idempotent 저장. 정규 산출과 동일한 파일 구조 (섹션 수, `_id` / `_ood` 분리) 미러링. 필터 test JSONL 은 `data/{DATADIR}/{prefix}_stage1{,_test{_id,_ood}}_without_open_app.jsonl` 로 영구 저장 (idempotent 재사용). skip marker 별도라 정규/필터 각각 독립 idempotent.

### Stage 2 평가
- [`scripts/_action_eval.py`](./scripts/_action_eval.py) 가 기준 (`score` 서브커맨드만 유지, single-pair / ID+OOD 모드 모두 제공). winner / `BEST_CHECKPOINT` 개념 제거.
- 흐름: `stage2_train.sh → stage2_merge.sh → stage2_eval.sh`. TRAIN_DATASET 은 `AC_EXP01 | AC_EXP02` (MC 는 Stage 2 데이터 없음). AC_EXP01 는 `--exp01-ratio {r37|r55|r73}` 단일 ratio 로 학습 모델을 지정.
- EVAL_DS 별 분기:
  - **EVAL_DS=AC_EXP01 / AC_EXP02**: ID + OOD 두 test 파일 (`implicit-world-modeling_stage2_test_{id,ood}.jsonl`) 함께 추론 → `action_metrics.json` 에 `overall` / `in_domain` / `out_of_domain` 3 섹션.
  - **EVAL_DS=MB**: 단일 파일 `implicit-world-modeling_stage2.jsonl` 1 회 추론 → single-pair `overall` 1 섹션.
- HF 네이밍: base variant `SaFD-00/{short}-{slug}base-stage2-{MODE2}-epoch{E2}`, world-model variant `SaFD-00/{short}-{slug}world-model-stage1-{MODE1}-epoch{E1}-stage2-{MODE2}-epoch{E2}`. `{slug}` 는 AC_EXP01 ratio 별로 다름 (`ac-exp01-ratio37-/ac-exp01-ratio55-/ac-exp01-ratio73-`).
- 산출 경로: `outputs/{OUT_DS}/eval/{MODEL}{SFX}/stage2_eval/{variant_path}[_from_{M1}-ep{E1}][/epoch-{E2}]/on-{EVAL_DS}/`. AC_EXP02 / MC 는 OUT_DS=TRAIN_DS, SFX=`""`. AC_EXP01 는 OUT_DS=`AndroidControl_EXP01`, SFX=`_ratio{37,55,73}` (variant_path 는 CLI VARIANT 의 `world_model` → `world-model` 치환).
- Stage 2 world-model train/merge 는 `--stage1-epoch N` 으로 지정된 로컬 `outputs/{OUT_DS}/merged/{MODEL}{SFX}_stage1_${MODE1}_world-model/epoch-${N}/` 를 base 로 사용. 학습 결과는 `outputs/{OUT_DS}/{adapters,merged}/{MODEL}{SFX}_stage2_${MODE2}_world-model_from_${MODE1}-ep${N}/` 에 stage1 epoch 별 분리 저장 (stage2_train.sh 가 YAML `__STAGE1_EPOCH__` 플레이스홀더 sed 치환, `_common.sh::local_merged_epoch_dir` 가 stage1/stage2 양쪽에 ratio suffix 부여).
- 재실행 시 marker (`action_metrics.json`) 존재 unit 은 variant × EVAL_DS 조합 별로 독립 skip.
- 회귀 테스트: [`tests/test_action_eval.py`](./tests/test_action_eval.py) 52 케이스. 메트릭 정의는 [`ARCHITECTURE.md`](./ARCHITECTURE.md) §6.

### shell 실행 공통 규약
- `AC_EXP01` / `AC_EXP02` / `MC` / `MB` 매핑 (`DS_PREFIX` / `HF_SLUG` / `DS_DATADIR`), 모델 레지스트리 → [`scripts/_common.sh`](./scripts/_common.sh).
- 학습/merge 스크립트는 `parse_args`, 평가 스크립트는 `parse_eval_args` (`--train-dataset` + `--eval-datasets`).

### Python 의존성
- [`setup.py`](./setup.py) `EXTRAS["llamafactory"]` 가 실제 설치 기준. `transformers>=4.56.0,<4.57` (vllm 0.11.2 의 `transformers<5` 제약 + LlamaFactory 서브프로젝트 `<=5.2.0` 와의 4.56–4.57.x 교집합 중 실제 pin 은 `<4.57`).

## 작업 시 주의점

- `LlamaFactory/` 내부 파일은 마지막 수단으로만 수정한다. 가능하면 노트북, local shell script, custom YAML (`LlamaFactory/examples/custom/...`), 평가 helper 로 해결.
- transformers 버전을 바꿀 때는 [`pyproject.toml`](./pyproject.toml) 의 주석과 [`setup.py`](./setup.py) `EXTRAS["llamafactory"]` 의 transformers pin 을 함께 수정한다. 현재 `>=4.56.0,<4.57` 로 고정. 서브프로젝트 (`LlamaFactory/pyproject.toml`) 는 건드리지 않는다.
- 문서나 스크립트에서 `outputs/{DS}/{category}/...` 의 `{DS}` 는 `AndroidControl_EXP01` / `AndroidControl_EXP02` / `MC`. AC_EXP01 의 ratio (`ratio37/ratio55/ratio73`) 는 디렉토리가 아니라 그 아래 모델 디렉토리의 suffix 로 운반되며 (Stage 1: `{MODEL}_ratio{37,55,73}_stage1_{MODE}_world-model/`, Stage 2: `{MODEL}_ratio{37,55,73}_stage2_{MODE2}_{base|world-model_from_{MODE1}-ep{E1}}/`), 모든 ratio 산출물이 `outputs/AndroidControl_EXP01/` 단일 부모 아래에 모인다. `{category}` 는 `adapters | eval | merged`. `adapters/` 는 flat 네이밍, `merged/` 는 `{MODEL}{SFX}_{detail}/epoch-{E}/` 로 epoch 별 서브디렉토리 분리. `eval/` 은 `{MODEL}{SFX}/stage{1,2}_eval/.../epoch-{E}/` 중첩 (AC_EXP02 / MC 는 SFX=`""`, AC_EXP01 는 `_ratio{37,55,73}`). `BEST_CHECKPOINT` 파일은 더 이상 생성되지 않는다.
- `data/` 아래 실제 디렉토리명은 `AndroidControl` (원본 source-only — EXP01/EXP02 가 read), `AndroidControl_EXP01`, `AndroidControl_EXP02`, `MonkeyCollection`, `MobiBench` (평가 전용). MobiBench 는 `implicit-world-modeling_stage{1,2}.jsonl` 두 단일 파일만.
- eval script 에서 `vllm_infer.py` 호출 시 `--dataset_dir '$LF_ROOT/data'` (절대 경로) 를 반드시 전달한다. 상대 경로 사용 시 HF datasets 캐시 오염으로 이미지 `FileNotFoundError` 가 발생할 수 있다.
- vLLM `gpu_memory_utilization` 은 `_common.sh::build_infer_cmd` 의 기본 `0.80` 을 환경변수 `VLLM_GPU_MEM_UTIL` 로 호출 단위 override 한다. `build_infer_cmd` 가 `stage{1,2}_eval.sh` 양쪽에서 공통 호출되므로 stage1/2 모두 동일하게 적용 (예: `VLLM_GPU_MEM_UTIL=0.6 bash ./scripts/stage2_eval.sh ...`). 미설정 시 0.80 유지.
- **MobiBench dataset_info 자동 보장**: `_common.sh::ensure_eval_only_dataset_info()` 가 source 시점에 `dataset_info.json` 에 `IWM-MB_stage{1,2}` 단일 파일 엔트리를 idempotent 하게 추가한다.
- **JSONL `images` canonical prefix**: 모든 JSONL 의 `images` 필드는 `{DATASET_NAME}/images/...` 형태여야 한다. 이 contract 는 `LF_ROOT/data/{DATASET_NAME}` symlink + `--dataset_dir $LF_ROOT/data` 조합과 맞물려 있어 prefix 가 없으면 `Image.open()` 이 cwd 기준으로 풀려 실패한다.
- Stage 1/2 merge 는 `outputs/{DS}/adapters/.../checkpoint-*` 가 하나라도 없는 슬롯에서 `[WARN]` SKIP 한다 (`--model all` sweep 친화). 모든 epoch 을 순회해서 local merge + HF push.
- Stage 2 train/merge (world-model variant) 는 `--stage1-epoch N` 으로 지정된 로컬 `outputs/{OUT_DS}/merged/{MODEL}{SFX}_stage1_${STAGE1_MODE}_world-model/epoch-${N}/` 가 선행돼야 한다 (AC_EXP01 ratio variant 는 SFX=`_ratio{37,55,73}`). Stage 2 train 은 YAML 의 `model_name_or_path` 를 런타임에 sed 치환하므로 노트북 YAML 생성 시 placeholder 값 (HF id) 은 무시된다. Stage 2 eval 도 local merged dir 우선 사용 (`_common.sh::resolve_eval_model_path`), 없으면 HF Hub merged repo fallback. `--stage1-epoch` 값은 world-model 계보 식별자로 양쪽 (local dir suffix + HF repo 이름) 에 동일하게 주입된다.
- merged 모델 경로 해석은 `_common.sh::resolve_eval_model_path` 단일 헬퍼 (kind=`stage1` | `stage2_base` | `stage2_world`) 가 담당하며, 내부에서 `local_merged_epoch_dir` + `hf_repo_id_stage{1,2}*` 를 조합한다.
- [`scripts/stage1_train.sh`](./scripts/stage1_train.sh) 는 `FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=${NPROC_PER_NODE}` 를 붙여 실행하지만, [`scripts/stage2_train.sh`](./scripts/stage2_train.sh) 는 의도적으로 torchrun prefix 를 붙이지 않는다. `NPROC_PER_NODE` 와 `GPU_TYPE` 은 `.env` 에서 관리 (기본 `NPROC_PER_NODE=2`, `GPU_TYPE=H100`). 노트북 Cell 5 가 (size, GPU_TYPE) 표에서 `per_device_train_batch_size` 를 정하고, YAML 생성 셀이 그 값으로 `gradient_accumulation_steps` 를 역계산 (`64 / (per_device * NPROC_PER_NODE)`) 해 global batch size 를 64 로 유지한다. **`.env` 변경 후에는 노트북 Cell 5 + Stage 1/2 YAML 생성 셀을 다시 실행해야 새 값이 YAML 에 반영된다.** `NPROC_PER_NODE` 는 `{1, 2, 4, 8}` 중 하나, `GPU_TYPE` 은 `{RTX5090, A100, H100}` 중 하나여야 하며 다른 값은 `ValueError` 로 거부된다.
- [`scripts/split_data.py`](./scripts/split_data.py) 는 Stage 1 + Stage 2 분할을 모두 담당. **AC_EXP01**: 원본 `data/AndroidControl/` 의 `_filtered.jsonl` 에서 read → `episodes_meta.jsonl.primary_app` 기반 app-level ID/OOD split (Stage 1 ratio mix train + Stage 1 dual-task ID/OOD test + Stage 2 ID/OOD), 산출물은 `data/AndroidControl_EXP01/` 에 write. Stage 1↔Stage 2 partition 공유. **MC**: 메타 없음 → Stage 1 random split (`_train` / `_test`) 자동 fallback. MC 는 `_STAGE1_ONLY` 라 Stage 2 자동 skip. **MB**: split 미수행 (평가 전용). AC 원본 메타는 [`scripts/extract_androidcontrol_metadata.py`](./scripts/extract_androidcontrol_metadata.py), 스크린샷은 [`scripts/extract_androidcontrol_images.py`](./scripts/extract_androidcontrol_images.py) 가 생성한다 (모두 `data/AndroidControl/` 에 저장).
- bash 자동화는 bash 4+ 전제.
- shell script CLI 공통 플래그:
  - **학습/merge (`stage{1,2}_{train,merge}.sh`)**: `--model MODEL --dataset {AC_EXP01|AC_EXP02|MC} --stage1-mode {full|lora} --exp01-ratios LIST`. `--dataset MB` 는 거절. `--dataset all` 은 지원하지 않음 (명시적 DS 선택 필수). `stage2_*`: `--stage2-mode {full|lora}` (기본 lora), `--stage1-epoch N` (world-model variant 전용). AC_EXP01 sweep 부분 실행은 `--exp01-ratios ratio55,ratio73`.
  - **평가 (`stage{1,2}_eval.sh`)**: `--model MODEL --train-dataset {AC_EXP01|AC_EXP02|MC} --eval-datasets LIST --stage1-mode ... --stage2-mode ... --stage1-epoch N --epochs LIST --variants LIST --exp01-ratio {r37|r55|r73}`. `--eval-datasets` 는 콤마 구분, 허용 `AC_EXP01,AC_EXP02,MC,MB`, 기본값은 `--train-dataset` 단일값 (AC_EXP01 의 기본 eval 은 raw `AC_EXP01`). Stage 2 eval 은 `--train-dataset {AC_EXP01|AC_EXP02}` 만 (MC 미지원). AC_EXP01 학습 모델 평가는 ratio 정확히 1 개 (`--exp01-ratio ratio55` 기본).
    - Stage 1 variants: `base`, `full_world_model`, `lora_world_model`.
    - Stage 2 variants: `base`, `full_base`, `lora_base`, `full_world_model`, `lora_world_model`.

## 빠른 검증 포인트

- `pytest tests/test_action_eval.py -q` — 52 케이스 (parse / field_match / 집계 / ID-OOD aggregation / single-pair overall)
- `bash scripts/stage{1,2}_{train,merge,eval}.sh --help` — 모든 플래그 표기 확인
- `python scripts/split_data.py --dataset MonkeyCollection --help` (MC: Stage 2 자동 skip)
- `bash scripts/stage1_train.sh --dataset MB 2>&1` — 거절 메시지 확인
- HF naming 단위 검증:
  ```bash
  source scripts/_common.sh && parse_args
  hf_repo_id_stage1 qwen3-vl-8b AC_EXP02 full 3
  # → SaFD-00/qwen3-vl-8b-ac-exp02-world-model-stage1-full-epoch3
  hf_repo_id_stage2_world_model qwen3-vl-8b AC_EXP02 full 3 lora 1
  # → SaFD-00/qwen3-vl-8b-ac-exp02-world-model-stage1-full-epoch3-stage2-lora-epoch1
  ```
- `rg "BEST_CHECKPOINT" scripts/ tests/` — 비어야 함
- `rg "transformers.*<5\b" setup.py pyproject.toml README.md ARCHITECTURE.md AGENTS.md` — 비어야 함 (모두 `<4.57`)
- GPU-aware per-device 검증: `.env` 에 `GPU_TYPE=RTX5090 NPROC_PER_NODE=8` 설정 후 노트북 Cell 5 실행 → "RTX5090 × 8 GPU: 7-9B pd=1 ga=8" 출력 확인. 7-9B × GPU × NPROC ∈ {1,2,4,8} 조합이 64 로 나뉘어야 함.

## 문서 동기화 원칙

- README 는 사용자 실행 순서 기준, ARCHITECTURE 는 실제 디렉토리/산출물 기준, AGENTS 는 "어디를 수정?" 기준으로 유지한다.
- 노트북 section 순서가 바뀌면 README 와 ARCHITECTURE 의 section mapping 도 같이 갱신.
- shell script 전제조건이 바뀌면 README, ARCHITECTURE, AGENTS 를 함께 수정.
- 모델을 추가하면 노트북 `_MODEL_CONFIG`, `_common.sh` 모델 레지스트리, README 모델 테이블, ARCHITECTURE 모델 레지스트리 테이블을 모두 갱신한다.
- 새 family 추가 시 노트북 Cell 5 의 `MODEL_FAMILY_CONFIG` 에 `factor` / `max_tokens` / `min_tokens` 명시 (`max_pixels = max_tokens × factor²`, `min_pixels = min_tokens × factor²`). family default 는 `max_tokens=2048`, `min_tokens=4`. `scripts/_common.sh::build_infer_cmd` 의 template 분기 (factor / mm_min) 도 함께 갱신.
- 새 dataset 의 image budget 이 family default 와 달라야 하면 `_DATASET_CONFIG[ds]["image_overrides"]` 에 token 단위로 둔다 (`{"max_tokens": N}`) — 빌더가 family `factor²` 로 환산. 평가측은 `scripts/_common.sh::build_infer_cmd` 의 `TRAIN_DATASET` 분기를 갱신해 학습 DS 와 동일 budget 사용.
