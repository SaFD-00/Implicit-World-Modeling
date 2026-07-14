# Implicit-World-Modeling Architecture

모바일 GUI World Modeling 이 Action Prediction 성능에 주는 영향을 검증하는 2-stage fine-tuning 파이프라인. 모든 stage 는 **`train → merge → eval`** 로 통일된다.

> **이 문서는 수치·표·근거의 유일 정본이다.** 좌표계/image budget, GPU 정책, 모델 자격 매트릭스, 메트릭 정의, 산출물 레이아웃, 데이터 계보가 여기 있다.
> [`AGENTS.md`](./AGENTS.md) 는 규칙만, [`README.md`](./README.md) 는 사용법만 싣고 근거는 이 문서로 링크한다. 같은 사실을 두 문서가 서술하지 않는다.

> **파생 수치는 여기 손으로 적지 않는다.** YAML 개수·데이터 행수·등록 키 개수처럼 코드/파일에서 파생되는 값은 **재확인 커맨드**로 대체했다 (드리프트하면 문서가 거짓말이 되므로). **설계 상수** (840×1876, 1,605,632, factor 28/32, `cutoff_len` 24576, `GLOBAL_BATCH_SIZE=64`) 는 규약이므로 본문에 남는다.

---

## 0. Runtime Stack

| 레이어 | 정본 |
|---|---|
| conda env | `implicit-world-modeling` — `pip install -e ".[llamafactory]"` |
| LF 부트스트랩 | `bash scripts/setup_llamafactory.sh --install --verify` (clone → pin `99464b3d034fd19fa73486f05e3b64b963e1b423` → `patches/llamafactory/*.patch` 적용 → editable 설치 → `MANIFEST.sha256` 검증; 멱등) |
| 학습/export | `llamafactory-cli train` / `llamafactory-cli export` |
| 추론 | `LlamaFactory/scripts/vllm_infer.py` (HF safetensors / PEFT adapter 를 그대로 로드) |
| 오케스트레이션 | [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) (walkthrough) + [`scripts/`](./scripts) (반복 실행) |

- `transformers>=4.57.1,<4.58` — vllm 0.11.2 의 `transformers<5` 제약 ∩ LF 서브프로젝트 `<=5.2.0`, 그리고 Qwen3-VL processor 가 4.57+ 도입. 정본은 `pyproject.toml` 의 `[project.optional-dependencies] llamafactory`.
- `deepspeed` · `vllm` · `bitsandbytes` 모두 같은 env 에 설치된다.

> ⚠️ **함정 1 — LF working tree 직접 수정 금지.** `LlamaFactory/` 는 gitignore 된 서드파티 clone 이라 **in-place 수정은 재클론 한 번에 증발한다.** LF 소스를 고쳐야 하면 `patches/llamafactory/` 에 `.patch` 를 추가한다 (`setup_llamafactory.sh` 가 pin 위에 순서대로 적용). 현재 패치: `0001-diff-loss.patch` (diff loss), `0002-double-ce-fix.patch` (이중 CE 수정).

---

## 1. 실행 구조

### 핵심 엔트리포인트

| 파일 | 역할 |
|---|---|
| [`implicit_world_modeling/lf_registry.py`](./implicit_world_modeling/lf_registry.py) | **레지스트리 SSoT** — `_MODEL_CONFIG` · `MODEL_FAMILY_CONFIG` · `_DATASET_CONFIG` · `_SIZE_CONFIG_AC` · `DATASET_MODEL_ELIGIBILITY` · `CONFIGS` 빌더. GPU 트리오(batch/grad_accum/deepspeed)는 **여기 없다**. |
| [`implicit_world_modeling/gen_configs.py`](./implicit_world_modeling/gen_configs.py) | **학습 YAML 생성기** — `--write` (재생성) / `--check` (커밋본 byte 대조 + orphan 검출, CI 게이트). 산출: `configs/train/IWM-{DS}/stage{1,2}_{full,lora}/`. |
| [`scripts/gpu_policy.py`](./scripts/gpu_policy.py) | **GPU 정책 SSoT** — `resolve_gpu_policy(gpu_type, nproc, size_class, ds_name, mode)` → batch / grad_accum / deepspeed. |
| [`scripts/_common.sh`](./scripts/_common.sh) | 공통 path·DS 매핑·모델 레지스트리·자격/등록 가드·`build_infer_cmd`·HF repo id 헬퍼. |
| [`scripts/stage{1,2}_{train,merge,eval}.sh`](./scripts) | `--model MODEL --dataset DS` 플래그 CLI. 커밋된 YAML (`configs/train/`) 과 dataset_dir 정본 (`configs/lf_dataset/`) 을 소비 — 노트북 실행 이력에 의존하지 않는다. |

학습 YAML 의 개수·내용 정합은 문서가 아니라 생성기가 보증한다:

```bash
python -m implicit_world_modeling.gen_configs --check   # 커밋본과 byte 대조 + orphan 검출
```

> **orphan 도 실패다.** 생성기가 만들지 않는데 디스크에 있는 YAML 은 **자격 테이블이 틀렸다는 신호**이므로 `--check` 가 exit 1 한다.

### 노트북

노트북은 **thin wrapper** 다 — 정본 로직은 전부 코드에 있다. **Section 매핑의 정본은 여기다** (README·AGENTS 는 이 절로 링크만 한다).

| Section | 노트북 markdown 헤더 | 하는 일 |
|---|---|---|
| 0 | Environment Setup | 환경/설정 + **AC_EXP02 diff loss 데이터 준비** (v1 — EXP02 데이터의 유일 생성기, §3 함정 10) |
| **1** | `## 1. Stage 1 Data Preparation` | `configs/lf_dataset/dataset_info.json` **대조 (read-only — 아무것도 쓰지 않는다)** + 통계 출력 |
| **2** | `## 2. Stage 2 Data Preparation` | Stage 2 등록 셀 — **미마이그레이션 잔재** (아래 참조) + 통계 출력 |
| 3–5 | Stage 1 | SFT → merge → eval |
| 6–8 | Stage 2 | SFT → merge → eval |

Section 3/4/6/7 은 **단일 변형 walkthrough** (`qwen3-vl-8b`) 이고, 다른 모델/모드/DS 는 cell 을 추가하지 말고 shell 인자만 바꾼다.

> ⚠️ **Section 2 의 Stage 2 등록 셀은 아직 마이그레이션되지 않았다.** 그 셀은 여전히 `LlamaFactory/data/dataset_info.json` (LF clone 안의 **죽은 사본**) 에 in-place 기록한다 — 정본 `configs/lf_dataset/dataset_info.json` 은 건드리지 않으므로 **무해하지만 실행할 이유도 없다** (그 파일은 이제 아무도 읽지 않는다 — §3 함정 13). 노트북 Section 1 의 markdown 이 스스로 이 경고를 달고 있다.

- **학습 YAML 생성은 노트북이 하지 않는다** — `gen_configs` 소관.
- **`dataset_info.json` 은 런타임 등록이 아니라 `configs/lf_dataset/` 의 커밋 정본이다** (§3).
- **Merge YAML 은 사전 생성하지 않는다** — `stage{1,2}_merge.sh` 가 runtime 에 임시 YAML 을 만든다. `BEST_CHECKPOINT` 개념은 제거됐고 epoch 번호는 `trainer_state.json.epoch` 의 `int(round(...))` 로 결정된다.

---

## 2. 모델 설정

### 모델 레지스트리

정본은 `lf_registry.py::_MODEL_CONFIG` 이며 `scripts/_common.sh` 의 `MODEL_ID` / `MODEL_TEMPLATE` / `ALL_MODELS` 와 **동시에** 동기화해야 한다 (두 곳이다).

| short_name | model_id | template | size tier |
|---|---|---|---|
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3_vl_nothink` | 7-9B |
| `qwen3-vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` | `qwen3_vl_nothink` | 3-4B |
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2_vl` | 7-9B |
| `qwen2.5-vl-3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | `qwen2_vl` | 3-4B |

현재 등록 목록 확인: `python -c "from implicit_world_modeling.lf_registry import _MODEL_CONFIG; print(list(_MODEL_CONFIG))"`

> `qwen3_vl_nothink` template 은 `_common.sh::build_infer_cmd` 가 `--enable_thinking False` 를 자동 주입해 thinking 트리거를 끈다. `qwen2_vl` 은 트리거가 없어 주입되지 않는다 (template 분기로 자동 처리).
>
> `qwen3-vl-4b` 는 **자격만 있고 학습 이력이 없다** (2026-07-13 복원). 아래 자격 매트릭스는 family 자격이지 학습 이력이 아니다.

### 좌표 규약 — 이 절이 전 실험군 모델 자격의 단일 근거다

Qwen 계열은 **세대마다 native 좌표 규약이 반전됐다** (Qwen2-VL 정규화 → Qwen2.5-VL 절대 픽셀 → Qwen3-VL 다시 정규화). 게다가 vision encoder 의 `factor` (patch × merge) 도 다르다.

| model family | native 좌표 규약 | factor | 해상도 의존성 |
|---|---|---|---|
| **Qwen3-VL** (8B/4B) | **0–1000 정규화** | 32 | resolution-independent (rescale 분모 999, backward-compat) |
| **Qwen2.5-VL** (7B/3B) | **절대 픽셀** (입력 이미지 실제 해상도) | 28 | 이미지 해상도에 종속 |

**메커니즘 (이 문서에서 한 번만 서술한다):** 데이터의 좌표 표현과 어긋나는 family 를 쓰면 **에러가 나지 않는다 — grounding 만 조용히 깨진다.** 두 축이 동시에 어긋나기 때문이다. ① 좌표 규약 (정규화 ↔ 픽셀), ② factor 로 결정되는 **리사이즈된 이미지의 실제 픽셀 크기** (아래 image budget 절). 학습은 정상 종료되고 loss 도 내려간다. 성능만 떨어진다.

출처 — [Qwen3-VL Spatial Understanding & 2D Grounding (DeepWiki)](https://deepwiki.com/QwenLM/Qwen3-VL/5.2-spatial-understanding-and-2d-grounding) · [Qwen2.5-VL Visual Grounding issue #866](https://github.com/QwenLM/Qwen2.5-VL/issues/866).

### 실험군별 모델 자격 (`DATASET_MODEL_ELIGIBILITY`) — 코드 가드다

| 실험군 | 좌표 표현 | 자격 모델 |
|---|---|---|
| AC_EXP01 / AC_EXP02 / MC | element `index` (좌표 아님) | 등록된 전 모델 |
| **AC_EXP03 / AC_EXP04** | 좌표 `point=[x,y]` | **Qwen3-VL 계열 전용** (`qwen3-vl-8b`, `qwen3-vl-4b`) |
| **AC_EXP05** | **절대 픽셀 840×1876** | **Qwen2.5-VL 계열 전용** (`qwen2.5-vl-7b`, `qwen2.5-vl-3b`) |

재확인: `python -c "from implicit_world_modeling.lf_registry import eligible_models as e; print(e('AndroidControl_EXP05'))"`

> ⚠️ **함정 2 — 이 자격은 문서 규약이 아니라 코드 가드다** (커밋 `fd4fd77`, 2026-07-13). `lf_registry.py::DATASET_MODEL_ELIGIBILITY` 가 정본이고, **`scripts/_common.sh::require_model_eligible()` 이 `llamafactory-cli train` 호출 전에 `exit 1`** 한다 (`stage{1,2}_train.sh` 가 호출). `--model qwen3-vl-8b --dataset AC_EXP05` 는 **학습이 시작되지 않고 중단된다.** `gen_configs` 도 `eligible_models()` 를 거쳐 자격 밖 조합의 YAML 을 아예 만들지 않는다.
>
> 단 가드는 **학습 entry 에만** 걸린다 (eval 스크립트는 검사하지 않는다 — 학습이 막혀 체크포인트가 생기지 않으므로 실질적으로 무의미).
>
> "3B/8B 모두" 요청의 **8B 는 `qwen2.5-vl-7b` 로 해석**한다 — Qwen2.5-VL family 에 8B 체크포인트가 없고, 유일한 8B 인 `qwen3-vl-8b` 는 factor·좌표 규약 이중 mismatch 로 EXP05 부적합.

> ⚠️ **함정 3 — EXP05 에 Qwen3-VL 을 쓰면 이중으로 어긋난다.** ① 좌표 규약 (Qwen3-VL native = 정규화 vs EXP05 = 절대 픽셀), ② image budget (Qwen3-VL 은 factor 32 라 1,605,632 를 줘도 832×1888 로 정렬돼 840×1876 과 어긋난다). 2026-07-13 까지 `qwen3-vl-8b × EXP05` 가 "as-trained 보존" 예외로 남아 있었으나 **그 조합은 한 번도 학습된 적이 없어** 보존할 것이 없었다 — 자격에서 제거하고 YAML 도 삭제했다.

---

> ## ⚠️ 미해결 — EXP04 데이터 좌표계가 문서 전제와 모순된다 (2026-07-13 실측)
>
> 위 자격 표는 EXP03/EXP04 를 "0–1000 정규화 → Qwen3-VL native 와 일치" 로 정당화한다. **그러나 디스크의 EXP04 데이터는 그 전제를 만족하지 않는다.** `data/AndroidControl_EXP0{3,4,5}/…_stage1_train.jsonl` 의 XML `bounds="[x1,y1][x2,y2]"` 를 직접 파싱한 결과 (train 2,000 행 표본):
>
> | 실험군 | bounds 최대값 | >1000 인 행 | 판정 |
> |---|---|---|---|
> | EXP03 | **1000** | **0%** | 0–1000 정규화 ✓ |
> | **EXP04** | **3120** | **100%** | **원본 디바이스 픽셀** (`bounds="[0,136][1080,2337]"`, 1080×2400 급) ✗ |
> | EXP05 | **1876** | (의도됨) | 840×1876 절대 픽셀 ✓ |
>
> **이것이 버그인지 의도인지 아직 판정되지 않았다.** 그러므로:
> - **"EXP04 는 0–1000 정규화" 라고 다시 쓰지 마라** — 실측과 어긋난다.
> - **동시에 "원본 픽셀이 EXP04 의 스펙" 이라고 못박지도 마라** — EXP04 는 EXP03 의 *프롬프트만* 업그레이드한 변형이고 **멤버십·좌표 표현 동일이 설계 전제**다. 좌표계만 달라질 이유가 없으므로 **데이터 생성 버그일 공산이 크다.** 원본 픽셀이라면 Qwen3-VL native (정규화) 와도, Qwen2.5-VL native (840×1876 리사이즈 픽셀) 와도 맞지 않는다.
>
> **재판정·재빌드 경로가 둘 다 끊겨 있다:**
> 1. **실행 불가** — `configs/lf_dataset/dataset_info.json` 에 `IWM-AC_EXP04_*` 키가 **하나도 없어** `require_dataset_registered` 가 학습 진입을 막는다 (§3). 확인: 아래 §3 등록 커맨드.
> 2. **재빌드 불가** — `mirror_experiment.py --experiment exp04` 의 원천 `data/AndroidControl/implicit-world-modeling_stage1_{action,state}_xy_prompt-enhanced.jsonl` 2 개가 **디스크에 없다** (남은 것은 `*_xy.jsonl` (EXP03) 와 `*_xy_pixel-aligned.jsonl` (EXP05) 뿐).
>
> **선결 순서: 좌표 규약 확정 → (원천 확보 후) 재빌드 → dataset_info 등록.**
> 위 자격 서술 (Qwen3-VL 전용) 자체는 **코드가 데이터 내용과 무관하게 강제**하므로 유효하다 — 흔들리는 것은 자격의 *근거로 적힌 좌표계 전제*다.

---

### 모델 family 별 image budget

**이 절이 image budget 의 단일 진실원이다.** `lf_registry.py::MODEL_FAMILY_CONFIG` 와 `_DATASET_CONFIG[ds]["image_overrides"]` 로 관리된다.

**키 이름을 헷갈리지 마라 — 두 dict 의 스키마가 다르다:**

| dict | 코드가 읽는 키 | 읽는 곳 |
|---|---|---|
| `MODEL_FAMILY_CONFIG[short]` | **`max_pixels` / `min_pixels`** (px 단위) | `_img_cfg()` — 이 둘이 없으면 `KeyError` |
| 〃 | **`factor`** | `build_configs()` — image_overrides 의 token→px 환산용 |
| 〃 | `max_tokens` · `merged_tokens_at_1080x2400` · `vertical_retention` | **아무도 읽지 않는다 (provenance 전용)** |
| `_DATASET_CONFIG[ds]["image_overrides"]` | **`max_tokens` / `min_tokens`** (token 단위) 또는 `image_{max,min}_pixels` (px 단위) | `build_configs()` — token 키는 family `factor²` 로 환산 |

`MODEL_FAMILY_CONFIG` 에 **`min_tokens` 키는 없다** (`min_tokens` 은 `image_overrides` 전용 override 키다). 아래 표의 `max_pixels = max_tokens × factor²` / `min_pixels = min_tokens × factor²` 는 값의 **유도 근거이지 런타임 계산이 아니다** — 코드는 이미 환산된 `max_pixels`/`min_pixels` 를 그대로 읽는다.

모든 학습 DS 가 family default (`max_tokens` 2048 · `min_tokens` 4 상당) 를 쓴다 — **dataset `image_overrides` 는 현재 하나도 없다.**

> ⚠️ **함정 — 평가측 budget 은 학습 DS 를 따라가지 않는다. `template` (모델 family) 만 본다.**
> `_common.sh::build_infer_cmd` 는 `mm_max = 2048 × factor²` 를 **오직 `template` 에서** 계산한다 (`_factor=28`, `qwen3_vl*`/`qwen3_5*` 이면 `32`). `TRAIN_DATASET` 은 그 함수 안에서 **cutoff_len 분기에만** 쓰이고 budget 계산에는 관여하지 않는다 — 함수 상단의 "*정책: token 예산은 학습 데이터셋(TRAIN_DATASET) 으로 결정한다*" 주석은 **코드가 실제로 하는 일이 아니다.**
> 지금 학습 budget 과 평가 budget 이 일치하는 것은 **모든 학습 DS 가 family default 를 쓰고 `image_overrides` 가 0 개이기 때문인 우연**이다. **DS 별 `image_overrides` 를 추가하는 순간 이 등식이 깨진다** — 학습은 override 된 budget 으로, 평가는 family default 로 돌아 리사이즈 픽셀 크기가 달라지고 **grounding 이 에러 없이 조용히 깨진다.** `image_overrides` 를 넣는다면 `build_infer_cmd` 를 **반드시 함께** 고쳐야 한다.

| family | factor | max_tokens | **max_pixels** | min_pixels |
|---|---|---|---|---|
| Qwen3-VL (8B/4B) | **32** | 2,048 | **2,097,152** (= 2048 × 32²) | 4,096 |
| Qwen2.5-VL (7B/3B) | **28** | 2,048 | **1,605,632** (= 2048 × 28²) | 3,136 |

**budget 이 이미지를 실제로 어떤 크기로 만드는가** — base 스크린샷 **1080×2400 (W×H)** 를 `smart_resize` 한 결과:

| 설정 | factor | max_pixels | → 리사이즈 (W×H) | visual tokens |
|---|---|---|---|---|
| **Qwen2.5-VL (우리 값)** | 28 | **1,605,632** | **840 × 1876** | **2,010** |
| **Qwen3-VL (우리 값)** | 32 | **2,097,152** | **960 × 2144** | **2,010** |
| Qwen3-VL 에 EXP05 budget 을 억지로 준 경우 | 32 | 1,605,632 | 832 × 1888 | 1,534 |
| (참고) Qwen 공식 기본값 — Qwen2.5-VL | 28 | 12,845,056 | 1092 × 2408 | 3,354 |

재확인: `python -c "from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as s; print(s(2400,1080,factor=28,min_pixels=3136,max_pixels=1605632))"` (반환은 `(H, W)`)

> ⚠️ **함정 4 — "토큰 수가 같으니 괜찮겠지".** 두 family 모두 `max_tokens=2048` 이라 **visual token 수는 2,010 개로 동일하다.** 다른 것은 **이미지의 실제 픽셀 크기** (840×1876 vs 960×2144) 다. EXP05 데이터 (840×1876 절대 픽셀 좌표) 를 Qwen3-VL 로 학습하면 모델이 보는 이미지는 960×2144 인데 좌표는 840×1876 이라 **grounding 이 조용히 깨진다.** 토큰 수 일치는 아무것도 보장하지 않는다.

> ⚠️ **함정 5 — 1,605,632 는 Qwen 공식 기본값이 아니라 우리의 의도적 override 다.** Qwen 공식 기본 `max_pixels` 는 **12,845,056** (→ 1092×2408 / 3,354 tokens) 인데, 우리는 `max_tokens=2048` 로 낮춰 잡았다 (위 표).
>
> **budget 일치 규칙: 데이터가 전제한 리사이즈 픽셀 크기와 학습·평가의 `image_max_pixels` 는 반드시 같은 budget 에서 나와야 한다.** 어긋나면 좌표계 전체가 무효다 — 데이터의 좌표는 budget B 로 리사이즈된 이미지 기준인데 모델은 budget B' 로 리사이즈된 이미지를 보게 되기 때문이다.
>
> **데이터 생성기에는 image budget 인자가 없다.** EXP05 의 840×1876 좌표는 원천 jsonl (`*_xy_pixel-aligned.jsonl`) 에 **이미 박혀 들어온다** — `build_exp05_data.py` / `mirror_experiment.py` 는 좌표를 재계산하지 않는다. 즉 EXP05 는 **원천 데이터가 전제한 budget 1,605,632 가 우리 Qwen2.5-VL family default 와 정확히 같아서** `image_overrides` 없이 자동으로 일치한다. (길이 사전측정용 `filter_long_samples.py` 의 `--image-max-pixels` 는 별개 도구의 인자다 — §4.)

> vLLM `gpu_memory_utilization` 은 `build_infer_cmd` 기본 `0.80`, 환경변수 `VLLM_GPU_MEM_UTIL` 로 호출 단위 override (stage1/2 eval 공통). 예: `VLLM_GPU_MEM_UTIL=0.6 bash scripts/stage2_eval.sh …`

### `cutoff_len`

| 실험군 | 학습 `cutoff_len` | 평가 `--cutoff_len` |
|---|---|---|
| AC_EXP01 / AC_EXP02 / MC | **10000** | 8192 |
| **AC_EXP03 / AC_EXP04 / AC_EXP05** | **24576** | **24576** |

정본은 `lf_registry._LONG_CUTOFF_DS`, 평가측 분기는 `_common.sh::build_infer_cmd` (`IWM-AC_EXP0{3,4,5}*` prefix).

> ⚠️ **함정 6 — cutoff 를 내리면 학습이 첫 step 에서 죽는다.** state+action ratio-mix 는 frame 이 다수 포함돼 Qwen3-VL multimodal RoPE position 길이가 8192 를 초과하는 샘플이 있고 (관측: 8521), 그러면 `get_rope_index` **shape mismatch 로 크래시**한다. 10000 은 실측 분포 기준 multi-frame 안전 마진과 메모리/throughput 의 tradeoff 다 (2026-05-13 16384 → 10000 하향). `scripts/filter_long_samples.py --threshold` default 도 10000 으로 동기 — 사전 필터와 학습 cutoff 를 같은 기준으로 묶는다.

**왜 좌표 실험군만 24576 인가**: 같은 전이를 `index="N"` 대신 `point=[x,y]` 로 적어 시퀀스가 ~2~2.5x 길다. `cutoff_len=10000` 에서는 ~10% 가 잘리고 ~0.3% 가 위 크래시를 냈다 (EXP01 ratio73 은 max 9059, 잘림 0%). EXP03 멤버십은 EXP01 ratio73 (index 기준 ≤10000 으로 이미 필터된 집합) 의 좌표 미러라 팽창 상한이 묶여 있어, **필터 없이 cutoff 만 24576 으로 올리면 잘림·크래시·데이터 손실 0** 이 성립한다. EXP04 ⊆ EXP03, EXP05 도 동일 멤버십이라 이 성질을 승계한다. 평가측도 24576 으로 맞춰 입력 truncation 0 을 보장한다 (vLLM `max_model_len = cutoff + max_new_tokens` 증가 → KV cache 메모리↑·throughput↓; 필요 시 `VLLM_GPU_MEM_UTIL`).

측정 (필터링은 하지 않고 측정만):
```bash
python scripts/filter_long_samples.py --dataset AC_EXP03 --threshold 24576 --report-only
```

### GPU 정책 — `scripts/gpu_policy.py` 가 단일 진실원

`resolve_gpu_policy(gpu_type, nproc, size_class, ds_name, mode)` → `per_device_train_batch_size` / `gradient_accumulation_steps` / `deepspeed`. **이 세 값은 `lf_registry` 에 없다.**

**허용 매트릭스** (밖이면 `ValueError` → 학습 진입 전 중단):

| GPU_TYPE | 허용 nproc | `per_device_train_batch_size` | 좌표 실험군 (EXP03/04/05) |
|---|---|---|---|
| RTX5090 (32GB) | {1, 2} | 1 | 1 (이미 최소 — 추가 축소 불가) |
| A100 (80GB) | {1, 2, 4, 8} | 2 | **1** (`_HALF_BATCH_DATASETS` 반감) |
| H100 (80GB) | {1, 2, 4, 8} | 2 | **1** (반감) |

**global batch 불변식:**
```
per_device_train_batch_size × gradient_accumulation_steps × NPROC_PER_NODE == GLOBAL_BATCH_SIZE (64)
gradient_accumulation_steps = 64 / (per_device × nproc)      ← resolve_gpu_policy 가 역계산
```
좌표 실험군의 half-batch 는 grad_accum 재계산으로 보정돼 **`GLOBAL_BATCH=64` 가 유지된다** — EXP01 과 global batch 가 같아 비교가 공정하다. 위 표의 모든 조합이 64 로 나누어떨어져 silent rounding 이 없다 (`tests/test_gpu_policy.py` 가 전 조합을 고정). `_common.sh::resolve_overrides` 도 주입 직전에 `pdbs × ga × nproc == 64` 를 한 번 더 검증한다. RTX5090 은 per_device 가 이미 최소(1) 라 추가 축소가 불가능하다 → ZeRO-3 offload + gradient_checkpointing 에 의존 (OOM 시 NPROC↑).

> ⚠️ **함정 7 — deepspeed 는 GPU 무관 항상 offload. "80GB 면 offload 불필요" 는 틀린 추론이다.**
> `resolve_gpu_policy` 는 `allow_no_offload=True` 없이는 `GPU_TYPE`·`nproc`·`mode` 와 무관하게 항상 `ds_z3_offload_config.json` 을 반환한다. 근거 셋:
> 1. **커밋 YAML 전수가 offload 다** — 예외 0. 재확인: `grep -rh '^deepspeed:' configs/train | sort | uniq -c`
> 2. A100/H100 에서 offload 를 빼면 **EXP05 7B full FT 는 확정 OOM** (모델 상태 fp32 param+grad+Adam m/v 만 GPU 당 ~77 GiB).
> 3. **OOM peak 을 지배하는 항은 lm_head logits** (시퀀스 길이 × vocab) 이며 파라미터 샤딩이나 GPU 증설로 줄어들지 않는다 — "GPU 가 크니까/많으니까 offload 없이도 될 것" 은 성립하지 않는다.
>
> **조건부 offload 분기 (`if GPU_TYPE == "RTX5090": …`) 는 금지된 패턴이다.** 예전 노트북 Cell 10 이 바로 그 패턴이었고, RTX5090 이 아닌 조합에서는 no-offload 기본값 (`ds_z3_config.json`) 으로 조용히 divergence 했다 — **한 번도 실행된 적 없는 죽은 값이었다.** 되살리지 말 것.
>
> `DEEPSPEED_NO_OFFLOAD` / `--allow-no-offload` 는 **테스트 전용 opt-out 이다 — 프로덕션 호출자 0 건** (쉘·노트북·`gen_configs` 어느 것도 넘기지 않는다). 미실측 경고를 동반한다.
>
> 파생: `_common.sh` 의 CUDA/nvcc 가드는 `GPU_TYPE == RTX5090` 이 아니라 **offload 사용 여부**를 조건으로 건다 (A100/H100 도 CPUAdam JIT 빌드를 탄다). 탈출구: `LF_CUDA_GUARD_SKIP=1`.

> ⚠️ **함정 8 — 커밋 YAML 은 GPU-불변 baseline 이다. 하드웨어가 바뀌어도 YAML 을 재생성하지 않는다.**
> `configs/train/**` 는 **RTX5090×2 프로필** (`per_device=1`, `grad_accum=32`) 로 고정 emit 된다. 다른 GPU 조합은 **런타임 override** 로 주입한다 (`llamafactory-cli train cfg.yaml key=value` — LF `hparams/parser.py` 의 OmegaConf merge). `stage{1,2}_train.sh` 가 `_common.sh::resolve_overrides` 로 GPU 트리오 + `dataset_dir` / `media_dir` 절대경로를 붙인다:
> ```bash
> GPU_TYPE=A100 NPROC_PER_NODE=4 bash scripts/stage1_train.sh --model qwen2.5-vl-7b --dataset AC_EXP05 --stage1-mode full
> ```

### 하이퍼파라미터 — 3 단 머지

`lf_registry::build_configs()` 가 이 순서로 `dict.update()` 한다:

1. `_DATASET_CONFIG[ds].stage{1,2}` — dataset baseline
2. `_SIZE_CONFIG_AC[size].stage{1, 1_lora, 2}` — size tier (`AndroidControl_EXP*` 에만 적용, MC 제외)
3. `_MODEL_CONFIG[model].hparam_overrides` — 모델별 delta

**현재 (2) 와 (3) 은 전부 빈 dict 다** — 두 tier (`7-9B`/`3-4B`) 의 세 키가 모두 비었고 등록된 모델들의 `hparam_overrides` 도 비었다. 즉 **모든 모델이 dataset baseline 을 그대로 받는다** (EXP01/EXP02 실측 어댑터와 동일조건 보존 → diff loss 순효과만 비교). MB 는 평가 전용이라 하이퍼파라미터 해석에서 제외.

**dataset baseline (전 실험군 공통):**

| | lr | epochs | LoRA r/α | dropout | warmup | weight_decay | max_grad_norm | scheduler |
|---|---|---|---|---|---|---|---|---|
| **Stage 1 (full)** | 1.0e-5 | 3 | — | — | 0.03 | 0.01 | 1.0 | cosine |
| **Stage 1 (LoRA)** | 1.0e-5 | 3 | **8 / 16** | 0.05 | 0.03 | 0.01 | 1.0 | cosine |
| **Stage 2 (LoRA)** | **5.0e-5** | 3 | **32 / 64** | **0.1** | 0.03 | 0.01 | 1.0 | cosine |
| **Stage 2 (full)** | **1.5e-5** | 3 | — | — | 0.03 | 0.01 | 1.0 | cosine |

> Stage 2 full 의 `1.5e-5` 는 `_DATASET_CONFIG` 가 아니라 `gen_configs.render_stage2()` 안에 하드코드돼 있다 (LoRA 대비 안정화).
>
> 설계 근거 (dropout 0.1 유지): 과거 실측에서 dropout 0.10 이 저빈도 action type 을 불안정하게 만든다는 관측이 있으나, **EXP01 기존 stage2 어댑터 (d0.1 / 5e-5) 와의 동일조건 비교를 우선**해 baseline 을 유지한다 — 그래야 EXP02 stage2 만 재학습해도 EXP01 전체 (학습/eval) 를 보존할 수 있다.

`freeze_vision_tower: true` 는 등록된 모델 전부. Full FT 분기에서 `vision_tower|vision_model|visual|image_encoder` 키워드를 포함한 named parameter 를 `requires_grad=False` 처리한 뒤 frozen 텐서/파라미터 수를 stderr 로 출력한다.

---

## 3. 데이터와 설정 계약

### 데이터 계보 — 모든 실험군은 AC_EXP01 에서 파생된다

```
data/AndroidControl/              # 원본 source 자산 — 학습/평가 entry 아님 (DS_DATADIR 미등재)
  ├── implicit-world-modeling_stage1_{state,action}.jsonl               # 원천
  ├── implicit-world-modeling_stage1_{state,action}_filtered.jsonl      # filter_long_samples.py 산출
  ├── implicit-world-modeling_stage2{,_filtered}.jsonl
  ├── implicit-world-modeling_stage1_{action,state}_xy.jsonl            # EXP03 원천 (좌표)
  ├── implicit-world-modeling_stage2_xy.jsonl                           # EXP03 stage2 원천
  ├── implicit-world-modeling_stage1_{action,state}_xy_pixel-aligned.jsonl  # EXP05 원천 (절대 픽셀, Drive '0711_버젼')
  ├── episodes_meta.jsonl         # primary_app = 전경 앱 package_name
  └── images/                     # ★ 유일한 이미지 디렉토리 — EXP01~EXP05 전부가 "AndroidControl/images/..." 로 참조
      ⚠ EXP04 원천 (*_xy_prompt-enhanced.jsonl) 은 디스크에 없다 → 재빌드 불가 (§2 경고 블록)

  AC_EXP01  = filter_long_samples → split_data.py       (Stage1 ratio mix 3 종 + Stage2 ID/OOD)
     │
     ├── AC_EXP02 = ratio73 train + diff loss v1 token_weights    (test/Stage2 는 EXP01 복사)
     ├── AC_EXP03 = ratio73 멤버십의 좌표(point) 미러             (mirror_experiment.py --experiment exp03)
     │      └── AC_EXP04 = EXP03 + stage1 프롬프트 업그레이드     (--experiment exp04, stage1-only)
     └── AC_EXP05 = ratio73 멤버십의 절대 픽셀 미러 + diff loss v2 (build_exp05_data.py, stage1-only)

  MonkeyCollection  = Stage 1 전용 (random split 0.95)
  MobiBench         = 평가 전용 (stage{1,2}.jsonl 단일 파일)
```

행수·파일 목록은 파생값이다 — 디스크에서 직접 센다: `wc -l data/AndroidControl_EXP05/*.jsonl`

| 실험군 | Stage 2 | Stage 1 test | 좌표 표현 | diff loss |
|---|---|---|---|---|
| AC_EXP01 (ratio 37/55/73) | ✓ (ratio sweep) | (id,ood) × (state,action) 4 파일 | index | — |
| AC_EXP02 | ✓ (diff loss 미적용) | EXP01 복사 (4) | index | **v1** (stage1) |
| AC_EXP03 | ✓ | EXP01 좌표 미러 (4) | point | — |
| **AC_EXP04** | ✗ `_STAGE1_ONLY` | EXP03 미러 + 프롬프트 (4 + without_open_app 2) | **§2 경고 참조** | — |
| **AC_EXP05** | ✗ `_STAGE1_ONLY` | 4 + without_open_app 2 | **절대 픽셀 840×1876** | **v2** (stage1) |
| MC | ✗ (데이터 없음) | 단일 test | — | — |
| MB | 평가 전용 | 단일 파일 | — | — |

`_STAGE1_ONLY = {MonkeyCollection, AndroidControl_EXP04, AndroidControl_EXP05}` — Stage 2 YAML/등록/eval 을 전부 skip. `_DUAL_TASK_TEST` 는 AC_EXP01~EXP05 전부 (Stage 1 한정, §6). `_SINGLE_TEST = {MonkeyCollection}` — 세 플래그는 **직교**한다.

- **EXP03 미러**: EXP01 ratio73 산출 파일을 한 줄씩 읽어 `(episode, step)` 키로 좌표 원천의 대응 레코드를 골라 **동일 순서로** write. UI 트리는 `index="N"` 대신 `bounds="[x1,y1][x2,y2]" point="[cx,cy]"`, 액션은 `point=[x,y]`. **본문만 좌표이고 이미지 경로는 EXP01 것을 채택**한다. 원천에 없는 키 (~0.8–1.7%) 는 제외 → EXP01 과 `(episode, step)` 1:1 대응이나 행 수는 소폭 작다.
- **EXP04 미러**: EXP03 와 **동일 멤버십·좌표 표현**, 프롬프트만 업그레이드 (action space `scroll(direction, point)` → `swipe(start, end)`, role 문구 "represented as html-style XML", `[SWIPE]` 규칙). **EXP04 pool ⊆ EXP03 pool** 이라 멤버십 drift 가 없다.
- **EXP05 미러**: 절대 픽셀 좌표. 출력 이미지 경로는 EXP01 의 `AndroidControl/images/...` 재사용 (source 의 `myset/images/...` 는 매칭 키 추출용).

### 분할 규칙 (`scripts/split_data.py`)

- **App partition** (`compute_app_partition`, 원본 `episodes_meta.jsonl` 기준): Stage 2 행 수를 budget 으로 `(id_apps, ood_apps)` 를 **한 번** 계산하고 Stage 1 이 같은 partition 을 재사용한다. → **Stage 2 OOD 앱은 Stage 1 train 에 한 번도 등장하지 않는다** (world-modeling 이 OOD 앱을 보지 못함). EXP02~05 는 EXP01 산출 멤버십을 미러하므로 이 partition 을 그대로 승계한다 (별도 계산 없음).
- **AC_EXP01**: source 는 항상 원본 `data/AndroidControl/`, 산출은 `data/AndroidControl_EXP01/`. 선행 `filter_long_samples.py --dataset AC_EXP01` 가 mm-expanded length > cutoff 인 row 를 제거해 **원본 폴더 안에** `_filtered.jsonl` 3 개를 만든다 (Qwen3-VL `get_rope_index` broadcast 회피). split 은 **`_filtered` 만 입력으로 쓴다** (Stage 2 source 누락 시 hard-fail). `state_pred` 는 random, `action_pred` / Stage 2 는 **action_type stratified**. ratio (state:action ∈ {7:3, 5:5, 3:7}) 로 혼합한 stage1 train 3 종 + 같은 partition 의 Stage 2 split (기본 15K / 3K / 3K) 을 한 번에 산출. `--exp01-train-total` 이 Stage 1 train 합계 (기본 50K).
  - **ratio 차원은 데이터가 아니라 계보로 흐른다**: stage2 데이터는 3 ratio 가 **공유**하고, ratio 는 "Stage 1 ratio merged 가 Stage 2 의 base" 라는 계보로만 갈라진다. 산출물은 `outputs/AndroidControl_EXP01/…/{MODEL}_ratio{37,55,73}_…` suffix 로 분리된다.
  - Stage 2 학습 데이터의 last-message wrapping (`<thought>…</thought>\n<action>{...}</action>`) 은 `_parse_action_payload` regex helper 가 분리한다.
- **MC**: 메타 없음 → random split (`--stage1-ratio`, 기본 0.95).
- **AC_EXP02~05 / MB**: split 없음 (미러·전처리 산출 또는 평가 전용).

`episodes_meta.jsonl` — `{"episode_id": 0, "goal": "...", "primary_app": "com.zoho.meeting", "actions": [...], "step_instructions": [...]}`.
`episode_id` 는 **int** 인데 이미지 경로는 zero-padded string 이라 `split_data.py::_norm_ep` 가 `str(int(...))` 로 정규화해 매칭한다. `primary_app` 은 각 step 의 `accessibility_trees` (`AndroidAccessibilityForest` proto) 에서 전경 `TYPE_APPLICATION` window 의 root `package_name` 다수결 (시스템/런처 제외). 미검출 시 `None` → train 풀에만 합류하고 test 분할에서 제외 (`--stage2-exclude-null-app` 으로 완전 제외 가능).

### diff loss — v1 (EXP02) 과 v2 (EXP05) 는 병존한다

`scripts/diff_loss/` 에 `{hungarian_metric, hungarian_diff, token_weight_builder, preprocess_dataset}.py` (v1) 와 `..._v2.py` (v2) 가 나란히 있다.

| | **v1 (AC_EXP02)** | **v2 (AC_EXP05)** |
|---|---|---|
| 위치 매칭 cost | DOM `index` | bounds 중심점 거리 (`W_POS=0.4`, `BOUNDS_NORM=2050.0` = 840×1876 대각선, `BOUNDS_TAU=50.0`) |
| 텍스트 수집 | 자손 텍스트 흡수 포함 | direct text + 자체 속성 (description/id/text/aria-label) 만 |
| `MATCH_THRESHOLD` | 1.5 | 1.7 |
| element 키 | index | index → bounds 폴백 |
| metric key | `hungarian_idx` | `hungarian_pos` |
| **diff 토큰 가중치** | ADDED/MODIFIED **2.0** / UNCHANGED **1.0** | diff **1.0** / non-diff **0.25** |
| action_pred 샘플 | uniform 1.0 (diff element 0 개라 자동 동치) | uniform 1.0 (**명시적 분기** — 판별은 `images` 개수: 1=state_pred, 2=action_pred) |
| 토큰↔element 경계 | 토큰 **시작점만** 검사 (비대칭 버그) | **interval overlap** (`tok_cs < char_end and tok_ce > char_start`, zero-length offset 제외), 중첩 span 은 **max** |
| 실패 처리 | fail-open (uniform 폴백을 성공으로 집계) | `--on-error {fail,uniform,skip}`, 기본 **fail-closed** |
| 출력 쓰기 | 최종 경로 직접 스트리밍 (부분 산출물 가능) | sibling temp → **`os.replace` 원자 교체**, `--input == --output` 거부 |
| 재현성 | tokenizer revision 미고정 | `--revision` 고정 + `<out>.meta.json` sidecar (model/revision/가중 상수/집계) |

> ⚠️ **함정 9 — EXP05 에 v1 을 쓰면 diff loss 가 조용히 무력화된다.** **EXP05 HTML 에는 `index` 속성이 아예 없다** (실측: index 0 개, bounds 48 개). v1 builder 는 `index="..."` 를 regex 로 **필수 요구**하므로, v1 을 EXP05 에 쓰면 에러 없이 모든 토큰이 baseline 으로 방치된다.

> ⚠️ **함정 10 — v1 4 파일을 삭제하지 마라. 데드 코드가 아니다.** 노트북 Cell 7 이 **EXP02 데이터의 유일한 생성기**이고, v1 재실행은 저장된 `token_weights` 와 **40/40 일치**한다 (v2 로 돌리면 17/40). v1 의 경계 비대칭 버그도 **EXP02 재현성 보존을 위해 의도적으로 고치지 않는다.**

> ⚠️ **함정 11 — baseline skip 이 diff 토큰을 삼킨다.** `token_weight_builder` 의 baseline 은 `[1.0]*n_asst` 이고 `if weight == 1.0: continue` 로 기본값을 스킵하는 구조다. **신규 체계에서는 diff weight 가 바로 그 1.0** 이라, 스킵 때문에 diff 토큰이 baseline (0.25) 에 방치되는 함정이 있었다. v2 는 baseline 을 `wmap["UNCHANGED"]` 에서 유도하고 스킵 조건을 `if weight == base` 로 바꿔 해결했다. **action 샘플의 uniform-1.0 분기가 빠지면 "diff 없음 → 전부 0.25" 로 잘못 처리된다.**

- **v2 경계 교정의 실측 영향**: 시작점만 보던 규칙은 element 왼쪽 경계를 걸친 토큰 (앞 텍스트와 합쳐 토크나이즈된 것) 을 놓치면서 오른쪽으로 넘치는 토큰에는 가중치를 줬다. interval overlap 으로 바꾸자 state 출력 토큰의 **0.65% 가 바뀌었고 전부 `0.25 → 1.0` 상향** (하향 0 건) — 버그 서명과 정확히 일치한다.
- **신규 가중 체계의 순효과**: 배수 자체는 2.0 → 1.0 으로 줄었지만 non-diff 도 1.0 → 0.25 로 함께 낮아져 **diff 가 non-diff 대비 실질 4 배** 강해진다 (근거: 2026-07-11 Slack DM + Google Meet 회의록).
- **배선**: EXP05 stage1 YAML 전부 `use_diff_token_weighted_loss: true`. `token_weights` 는 train jsonl **인라인 필드**이며 `dataset_info.json` 의 `columns` 등록이 **불필요**하다 (`converter.py:226` 이 raw jsonl 에서 직접 읽는다). 확인: `grep -rl use_diff_token_weighted_loss configs/train`
- diff loss 는 LF 6 파일 패치 (`use_diff_token_weighted_loss` 인자 + `diff_token_weighted_loss_func` + collator 의 labels 기반 token_weights 복원) 에 의존한다 → `patches/llamafactory/0001-diff-loss.patch` (§0 함정 1). 예전의 anchor 문자열 치환기 `scripts/diff_loss/apply_llamafactory_patch.py` 는 은퇴했다.
- **이중 CE 버그는 수정 완료** (`patches/llamafactory/0002-double-ce-fix.patch`): labels 를 pop 하지 않아 HF 내부 CE 가 같이 돌고 버려지던 것 + logits 전량 fp32 upcast → activation peak **20.87 → 10.72 GiB (48.6% 절감, bit-exact)**.

  > ⚠️ **함정 — 청킹은 per-token loss 를 concat 한 뒤 *한 번에* reduce 해야 한다.** 메모리를 더 아끼려고 **청크마다 분자·분모(sum/count)를 따로 누적하면 합산 순서가 바뀌어 bit-exact 가 깨진다.** 패치가 전 토큰 loss 를 리스트에 들고 있는 것은 낭비가 아니라 **의도**다. bit-exactness 는 EXP02 재현성과 이 패치의 검증 근거라, 깨지면 조용히 재현성을 잃는다 (`tests/test_diff_loss_double_ce.py` 가 chunked ↔ unchunked 를 대조한다 — 이 테스트를 "고치는" 방향으로 가지 마라).
- `scripts/diff_loss/hungarian_metric.py` 는 채점용 `scripts/_hungarian_eval.py` 와 **의도적으로 분리된** 학습 전처리용 사본이다.

### EXP05 데이터 쟁점 (원천 확인 필요 — 본실험 전 선결)

> ⚠️ **함정 12 — EXP05 액션 라벨 좌표 중 11 키가 840×1876 범위를 벗어난다** (0710·0711 공통 — 기존 원천 버그). 값이 `[1682, 975]` 로 반복되며 **1682 ≈ 840×2** 라 스케일링 실수로 의심된다. 그중 **10 행이 EXP05 산출물에 실려 있고** (train 7 + `test_ood_{action,state,state_without_open_app}` 각 1), **`(12571,0) = [421,1979]` 는 OOD 평가셋 3 파일 전부를 오염**시킨다. XML 의 `bounds`/`point` 자체는 정확히 x_max 840 / y_max 1876 이다 — 어긋난 것은 **액션 라벨의 `coordinate` 필드**뿐이다.

0711 수정본 적용 시 관측된 나머지 3 건 (판정 대기):

| 쟁점 | 실측 |
|---|---|
| **`wait` 액션 전량 퍼지** | 0710 의 `wait` 4,958 건이 전부 빠지고 다른 위치에 400 건이 새로 들어왔는데 **399 건이 빈 current state** → `wait` 이 "빈 화면이면 wait" 라는 degenerate shortcut 이 되고 정상 맥락의 `wait` 을 학습하지 못한다 |
| **train 6.1% 축소** | 47,556 → 44,670. 퍼지분에 EXP01 **train 멤버십 2,548 · test 멤버십 598** 포함 → 오염 제거가 아니라 **커버리지 축소** |
| **action / state 키 대칭 붕괴** | 400 차이 |

빌드 정본은 [`scripts/build_exp05_data.py`](./scripts/build_exp05_data.py) (mirror → diff-loss 가중치 → 원자 교체) 이며 tokenizer / revision / 가중 상수 / 집계를 `<train>.meta.json` sidecar 에 기록한다. 실측 분포는 문서가 아니라 sidecar 에서 읽는다:
```bash
cat data/AndroidControl_EXP05/implicit-world-modeling_stage1_train.jsonl.meta.json
```

### LLaMA-Factory 등록 (`configs/lf_dataset/dataset_info.json`)

> ⚠️ **함정 13 — dataset_info 는 커밋된 정본이다. 런타임에 쓰지 않는다.** LF 안의 `data/dataset_info.json` 을 런타임 변조하던 방식은 은퇴했다 — **재클론 한 번에 증발했기 때문이다.** 스크립트는 `--dataset_dir <repo>/configs/lf_dataset` 를 **절대경로**로 넘기고, 같은 디렉토리의 **상대 심링크** (`{DATASET_NAME}` → `../../data/{DATASET_NAME}`) 가 이미지 경로를 푼다. `_common.sh::verify_dataset_info()` 는 source 시점에 정본과 `IWM-MB_stage{1,2}` 키의 존재를 **검증만** 하고 없으면 죽는다 (`ensure_eval_only_dataset_info()` 의 in-place 추가는 은퇴) → 노트북 미실행 환경에서도 MB 평가가 성립한다.

등록 분기는 세 직교 플래그 (`_STAGE1_ONLY`, `_SINGLE_TEST`, `_DUAL_TASK_TEST`) 로 결정된다:

아래는 **스키마**다 — 실제 키 개수는 파생값이므로 문서가 아니라 정본 파일에서 센다 (확인 커맨드는 표 아래).

| DS | 등록 키 스키마 |
|---|---|
| AC_EXP01 | stage1 train (ratio 별) + dual-task test (id/ood × state/action) (**test 는 ratio 간 공유**) + stage2 (train/test_id/test_ood) |
| AC_EXP02 / AC_EXP03 | stage1 train + dual-task test + stage2 (train/test_id/test_ood) |
| **AC_EXP04** | **미등록 — 키가 하나도 없다. §2 경고 블록 참조.** |
| AC_EXP05 | stage1 train + dual-task test. `_STAGE1_ONLY` 라 **stage2 키 없음** |
| MC | `IWM-MC_stage1_{train,test}` (`_SINGLE_TEST`) |
| MB | `IWM-MB_stage{1,2}` (평가 전용, 정적 등록) |

현재 등록 상태 확인 (**이것이 개수의 정본이다**):
```bash
python -c "import json;d=json.load(open('configs/lf_dataset/dataset_info.json'));print(sorted(d))"

# EXP04 만:
python -c "import json;d=json.load(open('configs/lf_dataset/dataset_info.json'));print(sorted(k for k in d if 'EXP04' in k))"
```

> ⚠️ **함정 14 — 가드는 YAML 유무가 아니라 등록 여부를 본다.** `require_dataset_registered()` 는 YAML 의 `dataset:` 키가 정본에 등록돼 있고 그 파일이 실재하는지 확인한다. EXP04 는 데이터 파일도 있고 학습 YAML 도 커밋돼 있지만 **등록 키가 하나도 없어 `stage1_train.sh --dataset AC_EXP04` 는 학습 진입 전에 죽는다.** (과거 EXP03 stage2 가 데이터·YAML 은 있는데 등록만 안 돼 LF 안쪽 깊은 곳에서 죽은 적이 있다 — 그때 이 가드가 없었다. 커밋 `3917446`.) EXP04 를 돌리려면 stage1 키 (`IWM-AC_EXP04_stage1_{train,test_id_state,test_ood_state,test_id_action,test_ood_action}`) 를 AC_EXP03 와 동일한 sharegpt 스키마로 추가·커밋해야 한다 — **단 §2 경고 블록의 좌표 규약 확정이 선결이다.**

- JSONL **파일 경로**는 `../../data/{DATASET_NAME}/...` **상대 경로**로 등록한다.
- ⚠️ **함정 15 — JSONL 내부 `images` 값은 `{DATASET_NAME}/images/...` prefix 를 반드시 유지한다.** 이 contract 는 위 심링크 + 절대 `--dataset_dir` 조합과 맞물려 있어, prefix 가 없으면 `Image.open()` 이 cwd 기준으로 풀려 실패한다. 그리고 **`--dataset_dir` 에 상대 경로를 주면** HF datasets 캐시 오염으로 `FileNotFoundError` 가 날 수 있다.
- 정본에는 과거 스키마의 사문화된 키 (`IWM-AC_*`, `IWM-AC_2_*`) 가 남아 있고 그 파일들은 실재하지 않는다. `require_dataset_registered` 는 **키 존재만** 보므로 무해하다 — 쓰지 말 것.

### 이름 규약

| | AC_EXP01 | AC_EXP02~05 | MC | MB |
|---|---|---|---|---|
| `data/` 디렉토리 | `AndroidControl_EXP01` | `AndroidControl_EXP0{2..5}` | `MonkeyCollection` | `MobiBench` |
| shell 코드 | `AC_EXP01` (→ `AC_EXP01_ratio{37,55,73}` expand) | `AC_EXP0{2..5}` | `MC` | `MB` (eval 전용) |
| LF prefix | `IWM-AC_EXP01` | `IWM-AC_EXP0{2..5}` | `IWM-MC` | `IWM-MB` |
| `outputs/` 최상위 | `AndroidControl_EXP01` (ratio 는 model dir suffix) | `AndroidControl_EXP0{2..5}` | `MC` | — (TRAIN_DS 산하 `on-MB/`) |
| HF slug | `ac-exp01-ratio{37,55,73}-` | `ac-exp0{2..5}-` | `mc-` | `mb-` (dormant) |

원본 `AndroidControl` (AC) 은 **학습/평가 entry 가 아니다** — `DS_DATADIR` 에 등재되지 않으며 source 자산으로만 쓰인다. 메타 추출은 `scripts/extract_androidcontrol_metadata.py` (TFRecord → 다수결, `pip install android-env` 필요), 스크린샷은 `scripts/extract_androidcontrol_images.py` (GCS REST API, TF 의존 없음).

---

## 4. 파이프라인 컴포넌트

| 스크립트 | 역할 |
|---|---|
| [`scripts/split_data.py`](./scripts/split_data.py) | AC_EXP01 split (source=`AndroidControl/` → output=`AndroidControl_EXP01/`), MC random |
| [`scripts/filter_long_samples.py`](./scripts/filter_long_samples.py) | mm-expanded length > cutoff 제거 (`--report-only` 로 측정만). `--image-max-pixels` 기본값 2097152 는 **Qwen3-VL 기준** — Qwen2.5-VL 대상이면 1605632 로 override |
| [`scripts/mirror_experiment.py`](./scripts/mirror_experiment.py) | `--experiment {exp03,exp04,exp05}` 통합 미러 (EXP01 ratio73 멤버십 → 좌표 표현) |
| [`scripts/build_exp05_data.py`](./scripts/build_exp05_data.py) | **EXP05 빌드 정본** (mirror → diff-loss 가중 → 원자 교체 + sidecar) |
| [`scripts/_hungarian_eval.py`](./scripts/_hungarian_eval.py) | Stage 1 metric (`score` 서브커맨드) |
| [`scripts/_action_eval.py`](./scripts/_action_eval.py) | Stage 2 metric, ID/OOD/overall 3 섹션 |
| [`scripts/eval_viewer.py`](./scripts/eval_viewer.py) | 비교 HTML 빌더. `--include EXP:MODEL` 다중 spec — 단일 EXP 자체 비교 + EXP 간 동급 stage cross-compare (산출 `outputs/_compare/stage{N}_eval/`) |
| [`scripts/remote_launch.sh`](./scripts/remote_launch.sh) + `configs/remote/run.template.yaml` | 원격 클러스터 제출 — **제공자 중립** (코드에 플랫폼 이름 없음, 제출 커맨드는 `.env` 의 `REMOTE_SUBMIT_CMD` 템플릿). **UNVALIDATED — 실행 이력 0** |

### Stage 1

`--stage1-mode {full|lora}` (기본 `full`). 모드별로 YAML · adapter · merged · HF ID 경로가 접미사로 분리돼 공존한다.

- **`stage1_train.sh`** — YAML `configs/train/IWM-{DS}/stage1_${MODE}/{MODEL}_world-model.yaml` + `FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=…` + GPU 트리오·`dataset_dir`/`media_dir` **런타임 override**.
- **`stage1_merge.sh`** — `adapters/…/checkpoint-*` **전수 loop**. 각 ckpt 의 `trainer_state.json.epoch` → `epoch-{E}/`. `--no-hf-upload` 시 `export_hub_model_id` 를 생략해 local export 만. **checkpoint 없는 슬롯은 `[WARN]` SKIP** (실패가 아니라 스킵 — sweep 친화). 요약에 `merged / skipped / failed` 카운트.
- **`stage1_eval.sh`** — Phase A (zero-shot `base`) + Phase B (`--epochs`, 기본 `1,2,3` merged sweep). model path 는 `resolve_eval_model_path` 가 **local merged dir 우선 + HF Hub fallback** 으로 결정 (merged 이므로 adapter 인자·`max_lora_rank` 불필요). marker (`hungarian_metrics.json`) 존재 unit 은 skip.
  - **without_open_app 자동 산출**: 정규 score 직후 **추론 재실행 없이** `--exclude-action open_app` 로 한 번 더 채점해 GT `open_app` 행을 양쪽에서 동시 drop 한 sibling 디렉토리 (`on-{EVAL_DS}-without-open_app/`) 를 idempotent 하게 만든다. 필터 test JSONL 은 `data/{DATADIR}/` 에 영구 보존한다. **state branch 만** — action 채점기는 미지원.

    > ⚠️ **함정 — 이 필터는 지금 전 실험군에서 무동작이다 (2026-07-13 실측).** `_hungarian_eval.py::_gt_action_type` 이 `ACTION_MARKER = "## Action\n"` 를 찾는데 데이터가 그 마커를 그렇게 담고 있지 않아 **항상 `None`** 을 돌려준다 → drop 되는 행이 **0** 이다 (EXP01 state test 3,000 행 전수 확인: non-None 0건). 따라서 `on-{DS}-without-open_app/hungarian_metrics.json` 은 **정규 메트릭과 수치가 같다.** 이걸 "open_app 제외 성능" 으로 읽으면 **거짓 결론**이 나온다 — 필터를 고치기 전까지 비교 근거로 쓰지 마라. (열린 이슈: [`docs/ROADMAP.md`](../docs/ROADMAP.md))

### Stage 2

`--stage2-mode {full|lora}` (기본 `lora`), `--stage1-mode` + `--stage1-epoch N` 으로 world-model variant 의 상류 계보 결정. `base` variant 는 Stage 1 무관.

- **`stage2_train.sh`** — YAML `…/stage2_${MODE2}/{MODEL}_{base,world-model-full,world-model-lora}.yaml`. 실행 env 는 **Stage 1 과 동일**하다 — `FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=…` (`stage2_train.sh:101` ↔ `stage1_train.sh:43`). 유일한 차이는 stage1 만 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 추가로 export 한다는 것뿐이다. world-model variant 는 local `merged/…/epoch-${N}/` 을 base 로 쓰도록 `model_name_or_path` 를 런타임 sed 치환하고 (그래서 YAML 의 HF id placeholder 는 무시된다), YAML 의 `__STAGE1_EPOCH__` 플레이스홀더가 `${N}` 으로 치환돼 결과가 `…_world-model_from_{M1}-ep{N}/` 로 분리 저장된다. 디렉토리 미존재 시 hard-fail.
- **`stage2_merge.sh`** — Full FT 는 checkpoint 자체가 전체 모델 (adapter 블록 없음), LoRA 는 `model_name_or_path: {base}` + `adapter_name_or_path: {ckpt}` + `finetuning_type: lora`.
- **`stage2_eval.sh`** — `--variants` 로 `base` / `{full|lora}_base` / `{full|lora}_world_model` 선택. EVAL_DS=AC_EXP01/02/03 은 ID+OOD 동시 추론 → 3 섹션, MB 는 single-pair 1 섹션. marker (`action_metrics.json`) 존재 unit 은 variant × EVAL_DS 별 독립 skip.
  - **`--epochs` 에 `0` 포함 (opt-in)**: `{full|lora}_world_model` 의 epoch-0 = stage2 미학습 베이스라인 (= stage1 merged 와 동일 모델). `{full|lora}_base` 는 stage1 계보가 없어 epoch-0 이 `base` 와 중복 → 경고 후 skip. 기본 `1,2,3` 에는 미포함.

### CLI 계약 — 어떤 인자가 무엇을 결정하는가

**실행법 (복붙 커맨드·플래그 기본값) 은 [README §3](./README.md#3-실행--train--merge--eval) 이 정본이다.** 여기에는 인자 → 산출물 경로·계보의 **계약**만 둔다.

| 인자 | 무엇을 결정하는가 |
|---|---|
| `--model` | `{model}` — 산출물 경로 전 구간 + HF repo id 의 `{short}` |
| `--dataset` (train/merge) | `{OUT_DS}` (outputs 최상위) + 소비할 YAML 디렉토리 (`configs/train/IWM-{DS}/`) + 등록 키. **MB 는 거절** (평가 전용), `--dataset all` 미지원 (명시적 선택 필수) |
| `--train-dataset` / `--eval-datasets` (eval) | 전자 = **모델 계보** (`{OUT_DS}` + budget·cutoff 결정), 후자 = **test 셋** (`on-{EVAL_DS}/`) → 둘이 갈리면 교차 평가 |
| `--stage1-mode` / `--stage2-mode` | 경로 접미사 `stage{1,2}_{full,lora}` — 모드별 산출물이 **공존**한다 |
| `--stage1-epoch N` | Stage 2 world-model variant 의 **상류 계보** — `…_world-model_from_{M1}-ep{N}/` 로 분리 저장 |
| `--epochs` | merged sweep 대상 epoch. `0` 은 opt-in (위 Stage 2 참조) |
| `--exp01-ratios` (학습 sweep) / `--exp01-ratio` (평가 단일) | AC_EXP01 의 `{SFX} = _ratio{37,55,73}` — ratio 는 데이터가 아니라 **계보**로 흐른다 (§3) |

- **Stage 2 대상 DS 는 `_STAGE1_ONLY` 가 결정한다** (§3) — `MC` · `AC_EXP04` · `AC_EXP05` 는 **셋 다 `_STAGE1_ONLY`** 라 (MC 는 stage2 데이터 자체가 없다) Stage 2 YAML 이 생성되지 않고 `require_yaml` 에서 중단된다. 즉 Stage 2 는 `AC_EXP01 | AC_EXP02 | AC_EXP03` 뿐이다.
- 각 스크립트의 `--help` 가 플래그의 최종 정본이다.

> ⚠️ **함정 16 — 유령 스크립트를 인용하지 마라.** tmux 실행 스케줄 같은 일회성 스크립트는 `.gitignore` 의 `scripts/tmux_*.sh` 로 **추적 제외**된다 (커밋 `0303167`). 저장소에 존재하지 않으므로 문서가 복붙 커맨드로 제시하면 안 된다 — 실행은 위 `stage{1,2}_*.sh` 를 직접 호출한다.

> ⚠️ **함정 17 — `SMOKE=1` override 의 따옴표.** `save_strategy` 값은 반드시 따옴표를 살려 넘긴다. OmegaConf 는 YAML 1.1 규칙으로 따옴표 없는 `no` 를 boolean `False` 로 파싱하고, HF 가 `"False is not a valid SaveStrategy"` 로 죽는다.

---

## 5. 실행 데이터 흐름과 산출물

```
raw JSONL + screenshots
  → extract_androidcontrol_images.py     (data/AndroidControl/images/ : GCS REST → PNG, TF 의존 없음)
  → extract_androidcontrol_metadata.py   (episodes_meta.jsonl : primary_app 다수결)
  → filter_long_samples.py --dataset AC_EXP01     (원본 폴더에 _filtered.jsonl 3 개)
  → split_data.py  /  mirror_experiment.py  /  build_exp05_data.py     (§3 계보)
  → configs/lf_dataset/dataset_info.json          (커밋 정본 — 런타임 등록 아님)

  → Stage 1 train  (mode1 ∈ {full, lora})
       → outputs/{OUT_DS}/adapters/{M}{SFX}_stage1_{mode1}_world-model/checkpoint-*/
  → Stage 1 merge  (모든 epoch 각각)
       → outputs/{OUT_DS}/merged/{M}{SFX}_stage1_{mode1}_world-model/epoch-{E1}/  (+ 선택적 HF push)
  → Stage 1 eval   (local merged 우선 + HF fallback × cross-dataset)
       → outputs/{OUT_DS}/eval/{M}{SFX}/stage1_eval/{mode1}_world-model/epoch-{E1}/on-{EVAL_DS}/hungarian_metrics.json
       ★ 사용자가 결과를 보고 epoch E1 을 고른다 (자동 winner 선정 없음) → --stage1-epoch 로 Stage 2 에 전달

  → Stage 2 train  (mode2 ∈ {full, lora}, variant ∈ {base, world-model-{mode1}})
       world-model base = merged/…/stage1_{mode1}_world-model/epoch-{E1}/   (local, 선행 필수)
       → adapters/{M}{SFX}_stage2_{mode2}_{base | world-model_from_{mode1}-ep{E1}}/checkpoint-*/
  → Stage 2 merge → merged/…/epoch-{E2}/  (+ HF push)
  → Stage 2 eval  → eval/…/epoch-{E2}/on-{EVAL_DS}/action_metrics.json
       EVAL_DS=AC_EXP01/02/03 : { overall, in_domain, out_of_domain }   (test_id + test_ood)
       EVAL_DS=MB             : { overall }                             (single-pair)
```

### 산출물 레이아웃

`outputs/` 단일 루트 아래 **데이터셋 중심 + category 분리**. merged/eval 은 `epoch-{E}/` 서브디렉토리로 분리하고 full/lora 는 경로 접미사로 공존한다.

```
outputs/{OUT_DS}/                # AndroidControl_EXP0{1..5} | MC.  AC_EXP01 의 ratio 는 디렉토리가 아니라 {SFX} 로 운반
├── adapters/                    #   SFX = _ratio{37,55,73} (AC_EXP01) | "" (그 외)
│   ├── {model}{SFX}_stage1_{full,lora}_world-model/
│   ├── {model}{SFX}_stage2_{full,lora}_base/
│   └── {model}{SFX}_stage2_{full,lora}_world-model_from_{full,lora}-ep{E1}/
├── merged/                      # 같은 이름 + /epoch-{E}/
└── eval/{model}{SFX}/
    ├── stage1_eval/{base | {full,lora}_world-model/epoch-{E}}/on-{EVAL_DS}[-without-open_app]/
    └── stage2_eval/{base | {full,lora}_base/epoch-{E} | {full,lora}_world-model_from_{M1}-ep{E1}/epoch-{E2}}/on-{EVAL_DS}/
```

`BEST_CHECKPOINT` / `BEST_CHECKPOINT.json` 은 더 이상 생성되지 않는다. eval 경로의 `variant_path` 는 CLI VARIANT 의 `world_model` → `world-model` 치환이다.

### HF 업로드 ID (epoch 별 개별 repo)

| Stage / variant | 패턴 |
|---|---|
| Stage 1 | `SaFD-00/{short}-{slug}world-model-stage1-{M1}-epoch{E1}` |
| Stage 2 base | `SaFD-00/{short}-{slug}base-stage2-{M2}-epoch{E2}` |
| Stage 2 world | `SaFD-00/{short}-{slug}world-model-stage1-{M1}-epoch{E1}-stage2-{M2}-epoch{E2}` |

조립은 `_common.sh::hf_repo_id_stage1` / `hf_repo_id_stage2_base` / `hf_repo_id_stage2_world_model` 로 단일화. eval 의 model path 해석은 `resolve_eval_model_path {stage1|stage2_base|stage2_world}` 가 **local merged dir 우선 + HF fallback** 으로 처리한다 → local merge 한 머신에서 같은 머신 안에서 바로 eval 까지 이어 돌 수 있다 (`HF_TOKEN` 불필요).

---

## 6. 메트릭

**자동 winner 선정은 없다.** 두 stage 모두 `score` 서브커맨드로 JSON 을 저장하고, 사용자가 결과를 보고 Stage 2 에 쓸 Stage 1 epoch 을 `--stage1-epoch` 로 지정한다.

### Stage 1 — `hungarian_metrics.json`

- variant: `base` (zero-shot) / `full_world_model` / `lora_world_model`
- metric: `avg_hungarian_f1` (1차), `avg_bleu`, `avg_rouge_l` 등
- single-pair (`--test/--pred`) 와 ID/OOD (`--test-id/--pred-id/--test-ood/--pred-ood`) 모두 지원 — ID/OOD 모드는 `overall` / `in_domain` / `out_of_domain` 3 섹션.

> **dual-task 분기 (Stage 1 한정)**: EVAL_DS 가 AC_EXP01~AC_EXP05 (`_DUAL_TASK_TEST`) 이면 `state_pred` 와 `action_pred` 를 **각각 독립 채점**한다 — `on-{DS}-state/hungarian_metrics.json` (Stage1 채점기) + `on-{DS}-action/action_metrics.json` (**Stage2 채점기**). 각 task 가 (id, ood) 2 파일을 가지므로 inference 는 4 회. without_open_app sibling 은 state branch 만.
> **Stage 2 의 EVAL_DS 는 dual-task 가 아니다** — 일반 action prediction 으로 `stage2_test_{id,ood}.jsonl` 을 함께 채점해 3 섹션을 낸다. AC_EXP01 의 test 4 파일은 ratio 와 무관하다 (ratio 차원은 학습 산출물에만 박힌다).

### Stage 2 — `action_metrics.json`

각 섹션 (`overall` / `in_domain` / `out_of_domain`) 에 `step_accuracy` (1차), `macro_step_accuracy` (등장 action_type 평균), `parse_rate`, `type_accuracy`, `cond_{index,dir,app,text}_acc`, `per_type[t] = {count, type_acc, step_acc}`.

`overall` 은 id + ood 를 단순 concat 해 재집계한 결과이므로 **`overall.total == in_domain.total + out_of_domain.total` 이 항상 성립**한다. **ID/OOD gap** (`in_domain.step_accuracy - out_of_domain.step_accuracy`) 이 앱 일반화 정도를 나타낸다.

#### Step Accuracy (SA) 정의

> **왜 IoU 가 아닌가**: AndroidControl 은 GT 에 `bounds` 필드가 **영구 부재**하고 element-index 기반 grounding 을 쓴다. IoU 기반 채점은 구조적으로 0 이 된다.

```
SA = (1/N) · Σ correct_i
correct_i = 1  iff  (parse_ok ∧ type == gt.type ∧ field_match(type))
```

GT 의 `action_type` 키로 type 판정 (구 `type` 키 fallback 유지). GT last-message 가 `<thought>…</thought>\n<action>{...}</action>` 로 래핑되므로 **GT 도 `parse_action` 으로 `<action>` JSON 을 추출한 뒤 채점한다** (pred 와 동일 경로).

| GT action_type | field_match 조건 |
|---|---|
| `navigate_back`, `navigate_home`, `wait` | (검증 필드 없음) → 항상 통과 |
| `finish` | (status/answer 비교 안 함) → 항상 통과 |
| `click`, `long_press` | `str(pred.index) == str(gt.index)` |
| `scroll` | `norm(direction)` 일치 (index 무시) |
| `open_app` | `norm(app_name)` 일치 (params 평탄화 fallback 허용) |
| `input_text` | `norm(text)` 일치 (index 무시) |

`norm(s) = str(s or '').strip().lower()` — 모든 string field 통일.

해석용 reference baseline: `action_type` random ≈ **1/9 = 11.1%** (데이터셋 9 type) · `scroll` majority (`down`) **79.0%** · `finish` constant 100% (type-only 정책이라 해석 무의미).

정본은 `scripts/_action_eval.py`, 회귀 테스트는 `tests/test_action_eval.py` (`parse_action` / `evaluate_single` / `evaluate_predictions` 분기, unknown type 집계, `cond_*` n=0, `predict`/`output` fallback, ID+OOD 통합 집계 커버).

### xy 좌표 스페이스 채점 (EXP05 전용, opt-in)

EXP05 는 액션 스페이스가 xy 좌표로 통일돼 **GT 스키마가 바뀐다** — `<action>{"action":"click","coordinate":[x,y]}</action>` (키가 `action`, 구 `action_type` 과 다름; swipe 는 `coordinate1`/`coordinate2`).

EXP01~04 채점 결과가 **불변**이도록 **opt-in 플래그**로 구현했다: `_action_eval.py --coord-mode {index,xy}` (기본 `index`), `_hungarian_eval.py --match-mode {index,pos}` (기본 `index`). `stage1_eval.sh` 는 **EVAL_DS=AC_EXP05 일 때만** 전달한다.

| action | xy 모드 채점 규칙 |
|---|---|
| click / long_press | pred 좌표가 **GT 좌표가 속한 element 의 bbox 안**이면 정답 (포함 element 없으면 오답 + `no_bbox_n` 별도 집계) |
| scroll / swipe | xy1→xy2 벡터의 **주 방향** (`|dx| >= |dy|` → left/right, else up/down) 이 GT 와 일치하면 정답 |
| input_text / type | 좌표 무관 |

> ⚠️ **함정 18 — bbox 채점은 pred 가 GT 와 같은 840×1876 절대 픽셀 공간임을 가정한다.** 모델이 다른 좌표 공간으로 답하면 (§2 함정 4) 채점이 조용히 전부 오답이 된다.

---

## 7. 중요한 운영 제약

- **`implicit_world_modeling/` 패키지에 학습·평가 실행 로직은 없다** — 레지스트리 SSoT (`lf_registry.py`) 와 YAML 생성기 (`gen_configs.py`) 뿐이다. 실행 로직은 `scripts/` 와 노트북에 있다.
- **모델 추가 시**: `lf_registry._MODEL_CONFIG` + `_common.sh` 의 `MODEL_ID`/`MODEL_TEMPLATE`/`ALL_MODELS` 를 **동시에** 고치고 `gen_configs --write` 로 YAML 을 재생성해 커밋한다. 새 family 라면 `MODEL_FAMILY_CONFIG` 에 image budget 을 **먼저** 등록해야 하고 (`_img_cfg` 가 `KeyError` 를 낸다), `build_infer_cmd` 의 template 분기 (factor / mm_min) 도 함께 갱신한다.
- **하이퍼파라미터는 `_MODEL_CONFIG` 에 직접 쓰지 마라** — `_DATASET_CONFIG` baseline (또는 `_SIZE_CONFIG_AC` tier) 에서 바꾼다. `hparam_overrides` 는 모델별 delta 전용.
- Stage 2 world-model variant 는 `--stage1-epoch N` 의 로컬 merged dir 이 **반드시 선행**돼야 한다 (stage1_train → stage1_merge).
- merge/eval 은 Python `pyyaml` 을 전제한다. `HF_TOKEN` 은 HF push 또는 HF fallback pull 시에만 필요하다.
- shell automation 은 **bash 4+** 요구.
- **transformers 버전**은 `pyproject.toml` 의 `llamafactory` extra 에서 `>=4.57.1,<4.58` 로 고정 — 값과 그 위 주석을 함께 바꾼다. **서브프로젝트 `LlamaFactory/pyproject.toml` 은 수정하지 않는다.**
  - trl 0.24 / transformers 4.56+ API: `SFTConfig(max_length=…)`, `SFTTrainer(processing_class=…)`. 구버전 키 (`max_seq_length`, `tokenizer=`, `overwrite_output_dir`) 는 `TypeError`.
  - `gradient_checkpointing` 은 모델 로드 단계에서만 적용 — `SFTConfig` 에 전달하지 않는다 (이중 적용 방지).

> ⚠️ **함정 19 — EXP05 는 로컬에서 학습할 수 없다 (실측).** 로컬 2×RTX5090 에서 EXP05 3B Full FT 는 **CUDA OOM** (step 3 에서 8.92 GiB 할당 실패) + **157~168 s/it → 총 97~104 시간 (약 4 일)**. 원인은 `cutoff_len 24576` + `max_pixels 1,605,632` 로 시퀀스가 극단적으로 길어진 것과 RTX5090 에 강제되는 ZeRO-3 CPU offload 다. **본 학습은 원격 A100/H100 에서 수행한다** — 제출 스펙 (`configs/remote/run.template.yaml` + `scripts/remote_launch.sh`) 은 저장소에 있으나 **UNVALIDATED (실행 이력 0)** 이다. 참고: `gpu_policy` 는 7-9B full FT 조합에 host RAM 경고를 붙인다 (steady ~154GB/노드, 체크포인트 저장 시 ~175GB+).

> ⚠️ **함정 20 — YAML 이 있다고 돌릴 수 있는 게 아니다.**
> - **왜 재구성본인가**: EXP03/EXP04 의 as-trained YAML 은 **소실됐다.** 지금 `configs/train/IWM-AC_EXP0{3,4}/` 에 있는 것은 현행 레지스트리 + GPU 정책으로 `gen_configs` 가 **다시 렌더한 재구성본**이며 (`# [reconstructed 2026-07-13]` 헤더), 실제 학습 설정과 다를 수 있다. `--check` 는 "커밋본이 **현행 생성기**와 일치하는가" 만 보증하지 "**as-trained 와 일치하는가**" 는 보증하지 못한다 — 그래서 헤더를 남겼다. 확인: `grep -rl reconstructed configs/train`
> - **EXP04 는 YAML 이 있어도 돌지 않는다** (등록 키 없음 — 함정 14, §2 경고 블록).
>
> 재구성 이후 무엇이 실제 학습으로 확인됐는가 (**상태**) 는 [`../docs/ROADMAP.md`](../docs/ROADMAP.md) 의 재현성 경고를 본다.

현재 상태 (무엇이 됐고 무엇이 막혔나) 는 [`../docs/ROADMAP.md`](../docs/ROADMAP.md) 를 본다.
