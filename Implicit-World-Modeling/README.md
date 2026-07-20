# Implicit-World-Modeling

모바일 GUI World Modeling (Stage 1) 이 Action Prediction (Stage 2) 성능에 주는 영향을 검증하는 2-stage fine-tuning 파이프라인. 두 stage 모두 **`train → merge → eval`** 로 통일된다.

- **Stage 1** — `screenshot + UI XML + action → next UI XML` (World Modeling)
- **Stage 2** — `screenshot + UI XML + task → action JSON` (Action Prediction)
- **비교 variant** — `base` (zero-shot) / `stage2` / `stage1+stage2`

> **이 문서는 사용법만 싣는다.** 수치·표·근거 (좌표 규약, image budget, GPU 정책, 모델 자격, 메트릭 정의, 산출물 레이아웃, 데이터 계보) 의 정본은 [`ARCHITECTURE.md`](./ARCHITECTURE.md) 다. **작업 규칙 (하지 말아야 할 것) 은 [`AGENTS.md`](./AGENTS.md) 를 먼저 읽어라** — 어기면 에러 없이 조용히 깨지는 것들이 있다. 현재 진행 상태는 [`../docs/ROADMAP.md`](../docs/ROADMAP.md).

---

## 1. 설치

```bash
cd /path/to/Implicit-World-Modeling
conda create -n implicit-world-modeling python=3.12 -y
conda activate implicit-world-modeling
pip install -e ".[llamafactory]"

# LlamaFactory 부트스트랩 — clone → pin → patch → editable 설치 → MANIFEST 검증 (멱등)
bash scripts/setup_llamafactory.sh --install --verify
```

> **`conda activate implicit-world-modeling` 는 필수 전제다.** 이 문서의 모든 `python` / `bash scripts/*.sh` 커맨드가 이 env 를 가정한다. `scripts/_common.sh` 는 `CONDA_PREFIX` 가 없으면 즉시 중단하고, 활성 env 의 `bin/` 을 PATH 최상단에 고정해 user-site 의 낡은 CLI 를 가린다.

`LlamaFactory/` 는 gitignore 된 서드파티 clone 이라 `setup_llamafactory.sh` 가 **유일한 부트스트랩 경로**다 — pin 커밋으로 detach 하고 `patches/llamafactory/*.patch` 를 순서대로 적용한다. 재클론해도 같은 커맨드 한 번이면 복원된다. **LF working tree 를 직접 고치지 마라** — in-place 수정은 재클론 한 번에 증발한다 ([ARCHITECTURE §0](./ARCHITECTURE.md#0-runtime-stack)).

### `.env`

`.env.example` 를 복사해 `.env` 를 만든다 (`scripts/_common.sh` 가 source). `GPU_TYPE` / `NPROC_PER_NODE` 는 **호출 시점 환경변수가 `.env` 값을 이긴다**:

```bash
cp .env.example .env
GPU_TYPE=A100 NPROC_PER_NODE=4 bash scripts/stage1_train.sh --model qwen2.5-vl-7b --dataset AC_EXP05 --stage1-mode full
```

| 변수 | 용도 |
|---|---|
| `HF_TOKEN` | HF Hub push / fallback pull. `--no-hf-upload` 로 merge 하고 로컬 merged dir 로 eval 하면 불필요 |
| `NPROC_PER_NODE` · `GPU_TYPE` | GPU 트리오 (batch / grad_accum / deepspeed) 결정. 허용 조합은 [ARCHITECTURE §2 "GPU 정책"](./ARCHITECTURE.md#2-모델-설정) — 밖이면 학습 진입 전 중단 |
| `REMOTE_*` · `IWM_*` | `scripts/remote_launch.sh` 원격 제출용 — 실행 이력·검증 상태는 아래 "EXP05 는 로컬에서 학습할 수 없다" 절 참고 |

> **GPU 를 바꿔도 YAML 을 재생성하지 않는다** — 위 예시처럼 환경변수만 주면 된다. 왜 그런지는 [ARCHITECTURE §2 함정 8](./ARCHITECTURE.md#2-모델-설정).

### CUDA mismatch

기존 env 에 cu13 빌드 (`torch 2.11+cu130` 등) 가 남아 있으면 `_common.sh` 의 CUDA/nvcc 가드가 실행을 차단한다. 강제 교정:

```bash
pip install --force-reinstall "torch>=2.4,<2.9" "torchvision<0.24"
```

CPUAdam 이 미리 빌드된 이미지라면 `LF_CUDA_GUARD_SKIP=1` 로 가드를 우회한다.

---

## 2. 데이터 준비

학습 대상 DS 는 `AC_EXP01` · `AC_EXP02` · `AC_EXP03` · `AC_EXP04` · `AC_EXP05` · `AC_EXP06` · `MC`. `MB` (MobiBench) 는 평가 전용이다. 원본 `AndroidControl/` 은 학습/평가 entry 가 아니라 **source 자산**이다. 계보와 각 실험군의 설계 의도는 [ARCHITECTURE §3 "데이터 계보"](./ARCHITECTURE.md#3-데이터와-설정-계약).

행수·파일 목록은 문서가 아니라 디스크에서 센다: `wc -l data/AndroidControl_EXP05/*.jsonl`

### AndroidControl 원본 자산

```bash
# (a) 스크린샷 — GCS TFRecord → PNG (TF 의존 없음)
python scripts/extract_androidcontrol_images.py --output data/AndroidControl/images/ --skip-existing --verbose

# (b) episodes_meta.jsonl (primary_app = 전경 앱 package_name) — pip install android-env 필요
python scripts/extract_androidcontrol_metadata.py --output data/AndroidControl/episodes_meta.jsonl
```

### AC_EXP01 — Stage 1 ratio mix + Stage 2 ID/OOD split

```bash
# (선행 필수) cutoff 초과 샘플 제거 → 원본 폴더에 _filtered.jsonl. split 은 _filtered 만 입력으로 쓴다.
python scripts/filter_long_samples.py --dataset AC_EXP01

python scripts/split_data.py --dataset AC_EXP01 --exp01-ratios 3:7,5:5,7:3 --exp01-train-total 50000
```

산출: `data/AndroidControl_EXP01/` 에 stage1 train 3 종 (ratio37/55/73) + dual-task test + stage2 `{train,test_id,test_ood}`. **stage2 데이터는 3 ratio 가 공유**하고, ratio 는 "Stage 1 merged 가 Stage 2 의 base" 라는 계보로만 갈라진다 (산출 디렉토리는 ratio suffix 로 분리).

### AC_EXP02 — diff loss v1 (Stage 1)

데이터 생성의 정본 경로는 **노트북 Section 0 의 "AC_EXP02 diff loss 데이터 준비" 셀**이다 (멱등). 이 셀이 `scripts/diff_loss/preprocess_dataset.py` (**v1**) 로 AC_EXP01 ratio73 train 에 `token_weights` 를 부여하고, test / Stage 2 파일은 AC_EXP01 에서 복사한다 (동일 평가셋 → 공정 비교).

> v1/v2 를 섞으면 조용히 깨진다 — 규칙은 [AGENTS 하드 제약 8·9](./AGENTS.md), 근거는 [ARCHITECTURE §3 "diff loss"](./ARCHITECTURE.md#3-데이터와-설정-계약).

### AC_EXP03 — 좌표(point) 표현 미러

```bash
python scripts/mirror_experiment.py --experiment exp03

# (선택) cutoff 24576 이 전 샘플을 덮는지 측정만 (필터링은 하지 않는다)
python scripts/filter_long_samples.py --dataset AC_EXP03 --threshold 24576 --report-only
```

### AC_EXP04 — ⛔ 현재 준비도 실행도 불가

**AC_EXP04 는 지금 재빌드도 실행도 되지 않는다.** 무엇이 막혔는지는 [ROADMAP "EXP04 — 차단"](../docs/ROADMAP.md), 실측 근거는 [ARCHITECTURE §2 "⚠️ 미해결 — EXP04 데이터 좌표계"](./ARCHITECTURE.md#2-모델-설정).

등록 상태 확인 (빈 목록이면 여전히 차단):

```bash
python -c "import json;d=json.load(open('configs/lf_dataset/dataset_info.json'));print(sorted(k for k in d if 'EXP04' in k))"
```

### AC_EXP05 — 절대 픽셀 좌표 + diff loss v2 (Stage 1 전용)

```bash
# 1) 소스 2 파일을 Google Drive '0711_버젼' 폴더에서 받아 canonical 이름으로 배치 (예: gdown --folder <folder-url>)
#    → data/AndroidControl/implicit-world-modeling_stage1_{action,state}_xy_pixel-aligned.jsonl

# 2) 빌드 정본 — mirror → diff-loss v2 가중치 → 원자 교체 + sidecar. 기본 fail-closed.
python scripts/build_exp05_data.py
```

실측 분포는 문서가 아니라 sidecar 에서 읽는다 (tokenizer / revision / 가중 상수 / 집계):

```bash
cat data/AndroidControl_EXP05/implicit-world-modeling_stage1_train.jsonl.meta.json
```

> 본실험 전 선결 쟁점 (액션 라벨 좌표 범위이탈 · OOD 평가셋 오염 · `wait` 퍼지) 은 [ARCHITECTURE §3 "EXP05 데이터 쟁점"](./ARCHITECTURE.md#3-데이터와-설정-계약).

### AC_EXP06 — 비증강 대조군 (EXP05 대비, Stage 2 전용)

EXP05 와 **동일한 절대 픽셀 xy 좌표/budget 계약** (840×1876, budget 1,605,632, Qwen2.5-VL family, factor 28, `cutoff_len` 24576) 을 승계하되 **데이터 증강을 적용하지 않은** 대조군이다 — "EXP05(증강 O) vs EXP06(증강 X)" 로 증강 효과를 본다. **Stage 2 전용** (stage2 train/test_id/test_ood 만 존재, stage1 없음 — `lf_registry._STAGE2_ONLY` 가 stage1 YAML 렌더를 skip 한다). `base` variant 학습 완료: qwen2.5-vl-3b LoRA base, epoch 1/2/3. merged 3 에폭 HF 업로드: `SaFD-00/qwen2.5-vl-3b-ac-exp06-base-stage2-lora-epoch{1,2,3}`. 평가 채점은 EXP05 처럼 `--coord-mode xy` (`stage2_eval.sh` 가 `AC_EXP06` 에 자동 주입).

`lf_registry.py::DATASET_MODEL_ELIGIBILITY` 에 **등록 완료**(EXP05 와 동일 자격 — Qwen2.5-VL 계열 전용, `require_model_eligible()` 이 코드로 강제). `world-model-{full,lora}` variant 는 base 로 쓸 stage1 체크포인트가 EXP06 자체엔 없어 **EXP05 stage1 을 그대로 승계**한다 (`stage1_hf_slug: "ac-exp05-"` + 셸 `ds_stage1_source()`; stage2 산출물 네이밍은 `ac-exp06-` 그대로). 예:

```bash
bash scripts/stage2_train.sh --model qwen2.5-vl-3b --dataset AC_EXP06 \
  --stage1-mode full --stage1-epoch 3 --stage2-mode lora
```

이 커맨드의 `world-model-full` variant 는 `../outputs/AndroidControl_EXP05/merged/qwen2.5-vl-3b_stage1_full_world-model/epoch-3` 를 base 로 삼는다 (EXP06 자신의 stage1 이 아니라 EXP05 것 — DRY_RUN 로 확인됨).

### MC / MB

```bash
python scripts/split_data.py --dataset MC     # random 95:5, Stage 2 자동 skip
```

MB 는 split 불필요 — `data/MobiBench/implicit-world-modeling_stage{1,2}.jsonl` 두 파일만 있으면 평가가 성립한다. `IWM-MB_stage{1,2}` 는 커밋 정본에 정적 등록돼 있고 `_common.sh::verify_dataset_info()` 가 source 시점에 **검증만** 한다 (런타임에 심지 않는다).

### 학습 YAML

YAML 은 손으로 쓰지 않는다 — 레지스트리 + GPU 정책에서 렌더된다.

```bash
python -m implicit_world_modeling.gen_configs --check   # 커밋본과 byte 대조 + orphan 검출 (CI 게이트)
python -m implicit_world_modeling.gen_configs --write   # 재생성
```

---

## 3. 실행 — train / merge / eval

shell script 는 커밋된 학습 YAML (`configs/train/`) 과 dataset_dir 정본 (`configs/lf_dataset/`) 만 쓴다 — **노트북을 한 번도 돌리지 않은 머신에서도 성립한다.**

커맨드에 넣는 모델 short_name 은 코드에서 확인한다 (레지스트리 표: [ARCHITECTURE §2 모델 레지스트리](./ARCHITECTURE.md#2-모델-설정)):

```bash
python -c "from implicit_world_modeling.lf_registry import _MODEL_CONFIG; print(list(_MODEL_CONFIG))"
```

> **⚠️ 모델 × 실험군 자격은 코드 가드다** (문서 규약이 아니다) — `_common.sh::require_model_eligible()` 이 `llamafactory-cli train` 호출 **전에** `exit 1` 하고, 자격 밖 조합은 `gen_configs` 가 YAML 조차 만들지 않는다. 매트릭스와 근거는 [ARCHITECTURE §2 "실험군별 모델 자격"](./ARCHITECTURE.md#2-모델-설정). 어떤 모델이 되는지는 코드에 묻는다:
>
> ```bash
> python -c "from implicit_world_modeling.lf_registry import eligible_models as e; print(e('AndroidControl_EXP05'))"
> ```

### Stage 1

```bash
# train — --dataset 은 필수 (--dataset all 미지원). MB 는 거절.
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP02 --stage1-mode full
bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP01 --stage1-mode full   # ratio 3 종 자동 sweep

# merge — checkpoint 전수 loop, epoch 별 merged dir. --no-hf-upload 면 local export 만.
bash scripts/stage1_merge.sh --model qwen3-vl-8b --dataset AC_EXP02 --stage1-mode full --no-hf-upload

# eval — --train-dataset 이 모델 계보, --eval-datasets 가 test 셋 (교차 평가)
bash scripts/stage1_eval.sh --model qwen3-vl-8b --train-dataset AC_EXP02 --eval-datasets AC_EXP02,MB \
     --stage1-mode full --variants base,full_world_model --epochs 1,2,3
```

- `--dataset AC_EXP01` 은 ratio 3 종을 자동 sweep 한다 (`--exp01-ratios ratio55,ratio73` 로 부분 실행). **평가는 ratio 단일** — `--exp01-ratio ratio55` (기본).
- eval 은 로컬 `outputs/.../merged/.../epoch-{E}/` 를 **우선** 쓰고 없을 때만 HF Hub 로 fallback 한다 → `--no-hf-upload` 로 merge 한 머신에서 그대로 eval 까지 이어 돌 수 있다 (`HF_TOKEN` 불필요).

### Stage 2

**`AC_EXP01` / `AC_EXP02` / `AC_EXP03` 만** 받는다 (나머지는 Stage 2 YAML 자체가 없어 중단된다 — 이유는 [ARCHITECTURE §4 CLI 계약](./ARCHITECTURE.md#4-파이프라인-컴포넌트)).

```bash
bash scripts/stage2_train.sh --model qwen3-vl-8b --dataset AC_EXP02 \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora
bash scripts/stage2_merge.sh --model qwen3-vl-8b --dataset AC_EXP02 \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora
bash scripts/stage2_eval.sh  --model qwen3-vl-8b --train-dataset AC_EXP02 --eval-datasets AC_EXP02,MB \
     --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
     --variants base,lora_base,lora_world_model --epochs 1,2,3
```

- `world-model` variant 는 `--stage1-epoch N` 의 **로컬 merged dir 이 반드시 선행**돼야 한다 (없으면 hard-fail → 먼저 `stage1_train` + `stage1_merge`). 어떤 epoch 을 쓸지는 Stage 1 eval 결과를 보고 **사용자가 고른다** — 자동 winner 선정은 없다.
- 각 스크립트의 세부 동작 (merge 의 checkpoint SKIP, `--epochs 0` opt-in 베이스라인 등) 은 [ARCHITECTURE §4](./ARCHITECTURE.md#4-파이프라인-컴포넌트).

### 공통 플래그

| 플래그 | 값 |
|---|---|
| `--model` | short_name \| `all` (기본 `all`) |
| `--dataset` (train/merge) | `AC_EXP01`\|`AC_EXP02`\|`AC_EXP03`\|`AC_EXP04`\|`AC_EXP05`\|`MC` — **필수** |
| `--train-dataset` / `--eval-datasets` (eval) | 위 + `--eval-datasets` 에만 `MB` 추가 가능 |
| `--stage1-mode` / `--stage2-mode` | `full` \| `lora` (기본 `full` / `lora`) |
| `--epochs` (기본 `1,2,3`) / `--variants` / `--exp01-ratios` / `--exp01-ratio` | 콤마 구분 리스트 |
| `--no-hf-upload` | merge 전용 — HF push 생략 |

각 스크립트의 `--help` 가 정본이다.

### 실행 옵션 (환경변수)

```bash
DRY_RUN=1 bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP02   # 최종 커맨드만 출력하고 종료
SMOKE=1   bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP02   # max_samples=8 max_steps=1
VLLM_GPU_MEM_UTIL=0.6 bash scripts/stage2_eval.sh --model qwen3-vl-8b --train-dataset AC_EXP02   # vLLM 메모리 (기본 0.80)
```

### 재실행 시 skip

각 unit 의 marker (`hungarian_metrics.json` / `action_metrics.json`) 가 이미 있으면 `[=] ... skip (already done)` 로그만 남는다. 강제 재평가는 해당 marker 를 `rm` 후 재실행.

### ⚠️ EXP05 는 로컬에서 학습할 수 없다

**로컬 학습 불가.** 본 학습은 원격 A100/H100 에서 수행한다 (`scripts/remote_launch.sh` — **UNVALIDATED, 실행 이력 0**). 근거는 [ARCHITECTURE §7 함정 19](./ARCHITECTURE.md#7-중요한-운영-제약).

커맨드 자체는 유효하다 (80GB GPU 전제):

```bash
bash scripts/stage1_train.sh --model qwen2.5-vl-3b --dataset AC_EXP05 --stage1-mode lora
bash scripts/stage1_train.sh --model qwen2.5-vl-7b --dataset AC_EXP05 --stage1-mode lora
```

### ⚠️ YAML 이 있다고 돌릴 수 있는 게 아니다

`configs/train/IWM-AC_EXP0{3,4}/` 의 YAML 은 **as-trained 가 아니라 생성기 재구성본**이고 (`grep -rl reconstructed configs/train`), **EXP04 는 YAML 이 있어도 돌지 않는다.** 무엇이 검증됐고 무엇이 막혔는지는 [ROADMAP 재현성 경고](../docs/ROADMAP.md), 근거는 [ARCHITECTURE §7 함정 20](./ARCHITECTURE.md#7-중요한-운영-제약).

---

## 4. 결과 보기

### 메트릭 파일

| Stage | 파일 | 1차 지표 |
|---|---|---|
| Stage 1 | `hungarian_metrics.json` | `avg_hungarian_f1` |
| Stage 2 | `action_metrics.json` | `step_accuracy` (`overall` / `in_domain` / `out_of_domain` 3 섹션) |

파일을 어디서 찾는지 (경로 규약) 는 [ARCHITECTURE §5 산출물 레이아웃](./ARCHITECTURE.md#5-실행-데이터-흐름과-산출물), 지표 정의 (Step Accuracy · Hungarian F1 · dual-task 분기 · EXP05 xy 좌표 채점) 는 [§6 메트릭](./ARCHITECTURE.md#6-메트릭).

```bash
find outputs -name '*_metrics.json'
```

### HTML 비교 뷰어

```bash
# 단일 EXP — 결과는 outputs/{OUT_DS}/eval/{MODEL}/stage{N}_eval/pairs_*.html
python scripts/eval_viewer.py --include AC_EXP02:qwen3-vl-8b

# 다중 EXP 동급 stage cross-compare — 결과는 outputs/_compare/stage{N}_eval/
python scripts/eval_viewer.py --include AC_EXP01:qwen3-vl-8b_ratio73 AC_EXP02:qwen3-vl-8b

# 필터 — --stages {1,2}, --datasets 는 logical key, --variants 는 variant_path
python scripts/eval_viewer.py --include AC_EXP02:qwen3-vl-8b --stages 2 \
    --datasets on-AC-action-id --variants "lora_world-model/epoch-1"
```

`--include EXP:MODEL` 의 `MODEL` 은 `outputs/<DS_DATADIR(EXP)>/eval/` 아래 실제 디렉토리 명이다 (AC_EXP01 은 `qwen3-vl-8b_ratio73` 처럼 ratio suffix 포함).

> **cross-compare 는 화이트리스트로 막지 않는다** — `eval_viewer` 는 등록된 EXP 라면 뭐든 다중 spec 에 받고, 유일한 게이트는 **런타임 행 수 일치 검사**다 (`eval_viewer.py:483-489` — 행 수가 다르면 `SystemExit`). AC_EXP03/04/05 는 EXP01/EXP02 와 행 수가 달라 그 조합은 실패하므로 **사실상 단독 조회로 쓴다.** AC_EXP05 는 Stage 1 전용이라 `--stages 2` 로는 아무것도 나오지 않는다.

---

## 5. 테스트

```bash
pytest tests/ -v
```

`gen_configs --check` 는 커밋된 YAML 이 레지스트리 + GPU 정책으로부터 재현되는지 대조하는 CI 게이트다 — 레지스트리를 고쳤으면 `--write` 후 커밋한다.

---

## 6. 노트북

[`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) 는 **thin wrapper** 다 — 정본 로직은 전부 코드에 있다. Section 매핑과 주의사항 (Section 2 등록 셀의 미마이그레이션 잔재 포함) 은 [ARCHITECTURE §1 "노트북"](./ARCHITECTURE.md#1-실행-구조).

다른 모델/모드/DS 는 **cell 을 추가하지 말고 shell 인자만 바꾼다.**

---

## 7. 코드 읽기 시작점

| 파일 | 역할 |
|---|---|
| [`implicit_world_modeling/lf_registry.py`](./implicit_world_modeling/lf_registry.py) | 모델·데이터셋·자격 레지스트리 SSoT |
| [`implicit_world_modeling/gen_configs.py`](./implicit_world_modeling/gen_configs.py) | 학습 YAML 생성기 (`--write` / `--check`) |
| [`scripts/gpu_policy.py`](./scripts/gpu_policy.py) | GPU 정책 SSoT (batch / grad_accum / deepspeed) |
| [`scripts/_common.sh`](./scripts/_common.sh) | path·모델 레지스트리·자격/등록 가드·`build_infer_cmd`·HF repo id |
| [`scripts/split_data.py`](./scripts/split_data.py) · [`mirror_experiment.py`](./scripts/mirror_experiment.py) · [`build_exp05_data.py`](./scripts/build_exp05_data.py) | 데이터 생성 |
| [`scripts/_hungarian_eval.py`](./scripts/_hungarian_eval.py) · [`_action_eval.py`](./scripts/_action_eval.py) | Stage 1 / Stage 2 채점기 |

구조·근거는 [`ARCHITECTURE.md`](./ARCHITECTURE.md), 작업 규칙은 [`AGENTS.md`](./AGENTS.md), 진행 상태는 [`../docs/ROADMAP.md`](../docs/ROADMAP.md).
