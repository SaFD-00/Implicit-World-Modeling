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
| 오케스트레이션 | [`scripts/`](./scripts) — 파이프라인 전 단계가 shell/py 스크립트다 (walkthrough 노트북은 2026-07-26 은퇴). |

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
| [`scripts/stage{1,2}_{train,merge,eval}.sh`](./scripts) | `--model MODEL --dataset DS` 플래그 CLI. 커밋된 YAML (`configs/train/`) 과 dataset_dir 정본 (`configs/lf_dataset/`) 만 소비 — 특정 실행 이력에 의존하지 않는다 (갓 clone 한 머신에서도 성립). |

학습 YAML 의 개수·내용 정합은 문서가 아니라 생성기가 보증한다:

```bash
python -m implicit_world_modeling.gen_configs --check   # 커밋본과 byte 대조 + orphan 검출
```

> **orphan 도 실패다.** 생성기가 만들지 않는데 디스크에 있는 YAML 은 **자격 테이블이 틀렸다는 신호**이므로 `--check` 가 exit 1 한다.

### 실행 오케스트레이션 (노트북 은퇴)

walkthrough 노트북(`implicit-world-modeling.ipynb`)은 **2026-07-26 은퇴했다.** 정본 로직은 전부 코드에 있었고, 노트북은 `scripts/*.sh` 예시 호출 + 읽기전용 통계 셀을 묶은 thin wrapper 였다. 파이프라인 단계 → 정본 스크립트 매핑:

| 단계 | 정본 스크립트 |
|---|---|
| 환경 셋업 | `scripts/setup_llamafactory.sh --install --verify` |
| 데이터 생성 | 실험군별 `scripts/build_exp0N_data.py` (EXP02/EXP05/EXP07) — EXP02 는 diff loss v1 (§3 함정 10) |
| 학습 YAML 대조 | `python -m implicit_world_modeling.gen_configs --check` |
| Stage 1/2 | `scripts/stage{1,2}_{train,merge,eval}.sh --model M --dataset DS --stage{1,2}-mode {full|lora}` |

다른 모델/모드/DS 는 **shell 인자만 바꾼다** (예전엔 cell 을 추가하지 말라고 했던 그 규칙).

은퇴하며 함께 사라진 노트북 전용 잔재:
- **AC_EXP02 데이터 생성 셀(구 Cell 7)** → `scripts/build_exp02_data.py` 로 이관 (유일 생성기 지위 유지, §3 함정 10).
- **읽기전용 통계/등록 검증 셀** → 드롭. `dataset_info.json` 정본은 `configs/lf_dataset/` 커밋본이고 대조는 `_common.sh::verify_dataset_info()` 가 source 시점에 한다 (§3 함정 13).
- **eval 리포트/플롯 + `BEST_CHECKPOINT` 자동 선정 셀** → 드롭 (EXP01/02 전용 레거시). `BEST_CHECKPOINT` 개념은 이미 merge 경로에서 제거됐고 epoch 번호는 `trainer_state.json.epoch` 의 `int(round(...))` 로 결정된다. 현행 채점 정본은 `scripts/_hungarian_eval.py` · `_action_eval.py` · `thought_eval.py` (stage{1,2}_eval.sh 경유).
- **구 Stage 2 등록 셀**은 LF clone 안 죽은 사본에 쓰던 무해·무의미 잔재였다 — 은퇴로 자연 소멸.

- **학습 YAML 생성은 `gen_configs` 소관이다** (스크립트/사람이 손으로 만들지 않는다).

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
| **AC_EXP06** | **절대 픽셀 840×1876** | **Qwen2.5-VL 계열 전용** (EXP05 와 동일 — 아래 각주) |
| **AC_EXP07** | **절대 픽셀 840×1876** | **`qwen2.5-vl-3b` 단독** (아래 각주) |

재확인: `python -c "from implicit_world_modeling.lf_registry import eligible_models as e; print(e('AndroidControl_EXP05'))"`

> ⓘ **AC_EXP06 은 EXP05 의 비증강 Stage-2 대조군**이며 `DATASET_MODEL_ELIGIBILITY` 에 **등록 완료**됐다(2026-07-20 — EXP05 와 동일하게 `frozenset(_QWEN2_5_VL_FAMILY)`). 따라서 EXP06 도 함정 2 의 코드 가드 대상이다 — 다른 family 는 `require_model_eligible()` 이 학습 진입 전 `exit 1` 한다. **단 EXP06 자체엔 stage1 학습 데이터·체크포인트가 없다** (`lf_registry._STAGE2_ONLY` — stage1 YAML 도 렌더되지 않는다). stage2 world-model variant 는 base 로 **EXP05 stage1 체크포인트를 그대로 승계**한다 — `_DATASET_CONFIG["AndroidControl_EXP06"]["stage1_hf_slug"] = "ac-exp05-"`(HF fallback repo id 조립용) + 셸 `scripts/_common.sh::ds_stage1_source()`(`AC_EXP06 → AC_EXP05`, `stage2_{train,merge,eval}.sh` 의 stage1 참조 지점에 관통) 가 이를 구현한다. **stage2 산출물(경로/HF id)은 EXP06 정체성을 그대로 유지**한다 — 승계되는 것은 stage1 참조뿐이다.

> ⓘ **AC_EXP07 은 `qwen2.5-vl-3b` 단독** stage1 world-modeling 실험군이다 — EXP05/06 이 3B·7B 둘 다 자격인 것과 달리 **7B 를 자격에서 뺐다** (`eligible_models('AndroidControl_EXP07') == ['qwen2.5-vl-3b']`). **좌표계는 EXP05 와 같은 절대 픽셀 840×1876 이지만 데이터 계보는 EXP05 파생이 아니다** — 별도 0725 myset 소스에서 자체 빌드하며 (§3 계보), 같은 좌표계라서 채점도 EXP05 와 동일하게 `--coord-mode xy` 로 돈다 (`stage1_eval.sh` · `stage2_eval.sh` 가 `AC_EXP07` 에 자동 주입). **EXP06 과 달리 EXP07 은 자체 stage1 을 가진다** (`ds_stage1_source` 미매핑, `stage1_hf_slug` 미설정) — stage1 train 50K 를 자기 데이터로 학습한다. 대신 stage2 에는 **merge X 변형(`world-model-adapter`)** 이 EXP07 한정으로 opt-in 돼 있다 (아래 §4 Stage 2 · §5 HF 업로드 각주). `_LONG_CUTOFF_DS` 에 편입돼 학습·평가 cutoff 24576 을 승계한다.

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
> 2. **재빌드 불가** — `mirror_experiment.py --experiment exp04` 의 원천 `data/AndroidControl/EXP04_stage1_{action,state}.jsonl` 2 개가 **디스크에 없다** (남은 것은 `EXP03_*` 와 `EXP05_*` 뿐).
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
> **데이터 생성기에는 image budget 인자가 없다.** EXP05 의 840×1876 좌표는 원천 jsonl (`EXP05_stage1_{action,state}.jsonl`) 에 **이미 박혀 들어온다** — `build_exp05_data.py` / `mirror_experiment.py` 는 좌표를 재계산하지 않는다. 즉 EXP05 는 **원천 데이터가 전제한 budget 1,605,632 가 우리 Qwen2.5-VL family default 와 정확히 같아서** `image_overrides` 없이 자동으로 일치한다. (길이 사전측정용 `filter_long_samples.py` 의 `--image-max-pixels` 는 별개 도구의 인자다 — §4.)

> vLLM `gpu_memory_utilization` 은 `build_infer_cmd` 기본 `0.80`, 환경변수 `VLLM_GPU_MEM_UTIL` 로 호출 단위 override (stage1/2 eval 공통). 예: `VLLM_GPU_MEM_UTIL=0.6 bash scripts/stage2_eval.sh …`

### `cutoff_len`

| 실험군 | 학습 `cutoff_len` | 평가 `--cutoff_len` |
|---|---|---|
| AC_EXP01 / AC_EXP02 / MC | **10000** | 8192 |
| **AC_EXP03 / AC_EXP04 / AC_EXP05 / AC_EXP06** | **24576** | **24576** |

정본은 `lf_registry._LONG_CUTOFF_DS` (`EXP03/04/05/06` — `gen_configs` 학습 cutoff), 평가측 분기는 `_common.sh::build_infer_cmd` (`IWM-AC_EXP0{3,4,5}*` / `IWM-AC_EXP06*` prefix). **EXP06 도 2026-07-20 에 `_LONG_CUTOFF_DS` 에 편입됐다** — 이제 학습측 24576 도 생성기가 렌더하며(등록 이전엔 hand-authored `configs/train/IWM-AC_EXP06/stage2_lora/*.yaml` 의 `cutoff_len` 값이었다), 평가측은 그대로 `build_infer_cmd` 의 `IWM-AC_EXP06*` 분기에서 온다.

> ⚠️ **함정 6 — cutoff 를 내리면 학습이 첫 step 에서 죽는다.** state+action ratio-mix 는 frame 이 다수 포함돼 Qwen3-VL multimodal RoPE position 길이가 8192 를 초과하는 샘플이 있고 (관측: 8521), 그러면 `get_rope_index` **shape mismatch 로 크래시**한다. 10000 은 실측 분포 기준 multi-frame 안전 마진과 메모리/throughput 의 tradeoff 다 (2026-05-13 16384 → 10000 하향). `scripts/filter_long_samples.py --threshold` default 도 10000 으로 동기 — 사전 필터와 학습 cutoff 를 같은 기준으로 묶는다.

**왜 좌표 실험군만 24576 인가**: 같은 전이를 `index="N"` 대신 `point=[x,y]` 로 적어 시퀀스가 ~2~2.5x 길다. `cutoff_len=10000` 에서는 ~10% 가 잘리고 ~0.3% 가 위 크래시를 냈다 (EXP01 ratio73 은 max 9059, 잘림 0%). EXP03 멤버십은 EXP01 ratio73 (index 기준 ≤10000 으로 이미 필터된 집합) 의 좌표 미러라 팽창 상한이 묶여 있어, **필터 없이 cutoff 만 24576 으로 올리면 잘림·크래시·데이터 손실 0** 이 성립한다. EXP04 ⊆ EXP03, EXP05 도 동일 멤버십이라 이 성질을 승계한다. EXP06 (EXP05 의 비증강 Stage-2 대조군) 도 같은 절대 픽셀 좌표 규약과 24576/24576 cutoff 를 승계한다. 평가측도 24576 으로 맞춰 입력 truncation 0 을 보장한다 (vLLM `max_model_len = cutoff + max_new_tokens` 증가 → KV cache 메모리↑·throughput↓; 필요 시 `VLLM_GPU_MEM_UTIL`).

> ⚠️ **함정 14 — EXP07 은 bounded membership 이 아니다. 빌더가 직접 길이 필터를 해야 한다.** EXP03~06 이 "필터 없이 cutoff 24576 이면 손실 0" 인 건 멤버십이 EXP01 ratio73 (index ≤10000 사전필터본) 의 미러라 팽창 상한이 묶여서다. **EXP07 은 별도 0725 myset 소스라 그 상한이 없다** — NEXT_STATE_PREDICTION 의 gpt 출력(전체 UI XML)이 최대 51,852 토큰까지 나온다. 초기 `build_exp07_data.py` 는 필터가 없어, `cutoff_len` 초과 샘플이 학습 dataloader 에서 **잘리기 전 `image_grid_thw` vs 잘린 input_ids 불일치**로 죽었다 (Qwen2.5-VL: `Image features and image tokens do not match: tokens 2010, features 4020` — 배치 pdbs=2 의 한 샘플에서 이미지 placeholder 2010 개가 통째로 truncation, 이미지는 전부 1080×2400 동일이라 순수 길이 문제다; Qwen3-VL 이면 함정 6 의 `get_rope_index` shape mismatch 로 같은 원인이 다르게 뜬다). **수정(2026-07-26)**: `build_exp07_data.py` 가 `filter_long_samples.build_length_fn` 을 재사용해 **샘플링 전** state/downstream 풀에서 mm-expanded>cutoff 를 제외한다 (누출 0·40000/10000/15000 유지, 실현치 sidecar `length_filter`). `filter_long_samples.py` 본체는 AC_EXP01/02/03 만 대상이라 EXP07 은 빌더 내장 필터로 처리한다 — **새 소스로 실험군을 추가할 때 이 필터를 잊으면 학습이 죽는다.**

측정 (필터링은 하지 않고 측정만):
```bash
python scripts/filter_long_samples.py --dataset AC_EXP03 --threshold 24576 --report-only
```

### GPU 정책 — `scripts/gpu_policy.py` 가 단일 진실원

`resolve_gpu_policy(gpu_type, nproc, size_class, ds_name, mode)` → `per_device_train_batch_size` / `gradient_accumulation_steps` / `deepspeed`. **이 세 값은 `lf_registry` 에 없다.**

**허용 매트릭스** (밖이면 `ValueError` → 학습 진입 전 중단). **batch 도 deepspeed 도 `(GPU_TYPE, size_class, mode)` 3 축으로 갈린다 — GPU 종류만 보면 안 된다:**

| GPU_TYPE | 허용 nproc | size_class | mode | `per_device_train_batch_size` | 좌표 실험군 (EXP03/04/05/06) | deepspeed |
|---|---|---|---|---|---|---|
| RTX5090 (32GB) | {1, 2} | 7-9B / 3-4B | full / lora | 1 | 1 (이미 최소 — 추가 축소 불가) | **offload** |
| A100 / H100 (80GB) | {1, 2, 4, 8} | **7-9B** | **full** | 2 | **1** (`_HALF_BATCH_DATASETS` 반감) | **offload** |
| A100 / H100 (80GB) | {1, 2, 4, 8} | **7-9B** | **lora** | **2** | **2** (반감 **면제**) | **no-offload** |
| A100 / H100 (80GB) | {1, 2, 4, 8} | **3-4B** | full / lora | **2** | **2** (반감 **면제**) | **no-offload** |

아래 두 행이 no-offload 경로다 (`_is_no_offload_combo`): **80GB × (3-4B 이거나 lora)**. 그 조합은 optimizer state 를 GPU 에 올려도 들어가므로 offload 를 끄고 half-batch 예외도 면제한다 — 두 결정이 같은 메모리 실측(함정 7)에서 나오므로 코드도 한 술어로 함께 판정한다. **남는 offload 경로는 `7-9B × full` 과 RTX5090 전부** 뿐이며, 이 둘을 가르는 것은 정확히 optimizer state 의 크기다 (lora 는 어댑터만 학습 → 작다, full 7B 는 모델 상태만 GPU 당 ~77 GiB → 확정 OOM).

**global batch 불변식:**
```
per_device_train_batch_size × gradient_accumulation_steps × NPROC_PER_NODE == GLOBAL_BATCH_SIZE (64)
gradient_accumulation_steps = 64 / (per_device × nproc)      ← resolve_gpu_policy 가 역계산
```
좌표 실험군의 half-batch 도, no-offload 조합의 반감 면제도 grad_accum 재계산으로 보정돼 **`GLOBAL_BATCH=64` 가 유지된다** — 어느 GPU 조합·GPU 대수에서 돌리든 EXP01 과 global batch 가 같아 비교가 공정하다 (예: A100×2 × 3-4B × EXP05 → `pdbs 2 × ga 16 × 2 = 64`). 위 표의 모든 조합이 64 로 나누어떨어져 silent rounding 이 없다 (`tests/test_gpu_policy.py` 가 EXP06 편입 이후 200 조합 전수를 고정 — 10 GPU 조합 × size_class 2 × ds 5종(EXP01+EXP03/04/05/06) × mode 2). `_common.sh::resolve_overrides` 도 주입 직전에 `pdbs × ga × nproc == 64` 를 한 번 더 검증한다. RTX5090 은 per_device 가 이미 최소(1) 라 추가 축소가 불가능하다 → ZeRO-3 offload + gradient_checkpointing 에 의존 (OOM 시 NPROC↑).

> ⚠️ **함정 7 — offload 는 `(GPU_TYPE, size_class, mode)` 3 축으로 갈린다. "80GB 면 무조건 offload 불필요" 도, "언제나 offload" 도 둘 다 틀리다.**
> `resolve_gpu_policy` 는 **(A100|H100) × (3-4B 이거나 lora)** 에서 `ds_z3_config.json` (no-offload) 을, 그 밖의 조합 — **7-9B × full** 과 **RTX5090 전부** — 에서는 `ds_z3_offload_config.json` 을 반환한다.
>
> **끄는 근거** (2026-07-14 실측 — EXP05 stage1 full FT / qwen2.5-vl-3b / A100×2):
> 1. offload 를 켜면 **165 s/step** → 2094 step 에 **약 4 일**. GPU util 은 97 % 인데 메모리는 80GB 중 **23~26 GB** 만 쓰고, 두 GPU 전력이 **135 W 대 378 W** 로 벌어진다 — 계산이 아니라 CPU↔GPU 전송이 step 을 지배한다는 신호. 끄면 **138 s/step**, 메모리 64~73 GB.
> 2. **3-4B** 는 optimizer state 를 GPU 에 올려도 들어간다: 파라미터 3.09 B 의 fp32 master + Adam m/v 를 2 GPU 로 샤딩하면 GPU 당 ~25 GB, 여기에 위 activation/logits 를 더해도 80 GB 안이다.
> 3. **lora** 는 size_class 무관으로 끈다 — 학습 대상이 어댑터뿐이라 optimizer state 가 base weight 대비 무시할 만큼 작다 (7-9B lora 포함).
>
> **켜야 하는 근거** (아래 둘은 여전히 offload 다):
> 4. **7-9B × full**: A100/H100 에서 offload 를 빼면 **EXP05 7B full FT 는 확정 OOM** (모델 상태 fp32 param+grad+Adam m/v 만 GPU 당 ~77 GiB). **OOM peak 을 지배하는 항은 lm_head logits** (시퀀스 길이 × vocab) 이며 파라미터 샤딩이나 GPU 증설로 줄어들지 않는다 — "GPU 가 많으니 괜찮을 것" 은 성립하지 않는다. **3 이 lora 를 풀어준다고 full 까지 풀리지 않는다** — 둘을 가르는 것이 정확히 optimizer state 의 크기다.
> 5. **RTX5090 (32GB)**: 모델 크기·모드와 무관하게 offload 없이는 들어가지 않는다.
>
> **GPU 종류만 보고 갈리는 분기 (`if GPU_TYPE == "RTX5090": …`) 는 여전히 금지된 패턴이다.** 예전 노트북 Cell 10 이 그 패턴이었고, RTX5090 이 아닌 조합 전부가 no-offload 죽은 기본값으로 조용히 divergence 했다 (**7-9B full 까지 포함해서 — 그래서 위험했다**). 지금 분기는 **size_class 와 mode 를 함께 보고**, 각 조합이 (1)~(5) 의 메모리 실측에 묶여 있다. 회귀 방지: `tests/test_gpu_policy.py::test_a100_7b_full_stays_offload_gpu_type_alone_does_not_flip_it` · `::test_a100_7b_lora_is_no_offload_but_full_is_not`.
>
> `DEEPSPEED_NO_OFFLOAD` / `--allow-no-offload` 는 **정책이 offload 를 켜는 조합(7-9B full / RTX5090)에서 강제로 끄는 opt-out** 이다 (미실측 경고 동반). 80GB × (3-4B | lora) 는 이 플래그 없이도 기본이 no-offload 다.
>
> 파생: `_common.sh` 의 CUDA/nvcc 가드는 offload 여부와 무관하게 **항상** 건다 — 정책상 offload 를 쓰는 조합(7-9B / RTX5090)이 CPUAdam JIT 빌드를 타기 때문이다. 80GB×3-4B 처럼 offload 를 끄는 경로에서는 불필요하지만, 가드는 조합을 모른 채 source 시점에 돌므로 그대로 통과시켜야 한다. 탈출구: `LF_CUDA_GUARD_SKIP=1`. `CUDA_HOME` 이 torch 와 다른 cu 버전이면 여기서 막힌다 (`CUDA_HOME=<cu12.8 toolkit>` 로 지정).
>
> 2026-07-26 실측 — **이 가드는 conda env 활성화에 의존한다**: 가드의 기본값 `/usr/local/cuda` 는 이 머신에 **존재하지 않고**, `conda activate implicit-world-modeling` 이 `CUDA_HOME` 을 env prefix 로 세팅해 주기 때문에(그 안의 `nvcc` 가 12.8 = `torch.version.cuda` 12.8) 통과한다. env 를 활성화하지 않고 스크립트를 부르면 `_common.sh` 가 먼저 "conda env 가 활성화되지 않았습니다" 로 죽고, `CUDA_HOME` 만 지운 채로 부르면 이 nvcc 가드가 죽는다 — **둘 다 정상 동작**이며 학습 실패가 아니다.

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

**dataset baseline (EXP01–06 공통 · EXP07 은 아래 예외):**

| | lr | epochs | LoRA r/α | dropout | warmup | weight_decay | max_grad_norm | scheduler |
|---|---|---|---|---|---|---|---|---|
| **Stage 1 (full)** | 1.0e-5 | 3 | — | — | 0.03 | 0.01 | 1.0 | cosine |
| **Stage 1 (LoRA)** | 1.0e-5 | 3 | **8 / 16** | 0.05 | 0.03 | 0.01 | 1.0 | cosine |
| **Stage 2 (LoRA)** | **5.0e-5** | 3 | **32 / 64** | **0.1** | 0.03 | 0.01 | 1.0 | cosine |
| **Stage 2 (full)** | **1.5e-5** | 3 | — | — | 0.03 | 0.01 | 1.0 | cosine |

> **EXP07 은 이 baseline 을 `_DATASET_CONFIG["AndroidControl_EXP07"]` 에서 벗어난다** (size/model tier 는 여전히 빈 dict — DS baseline 만 다르다). Stage 1 = lr 1.0e-5 · **1 epoch** · LoRA **64 / 128** · dropout 0.05 · `save_strategy: steps` + `save_steps: 0.25` (→ fractional 체크포인트, 아래 각주). Stage 2 (LoRA) = lr 5.0e-5 · 3 epoch · LoRA **8 / 16** (rank/alpha, 사용자 지시 2026-07-28 — 64→16→8 순으로 하향; stage1 rank 와 독립) · dropout 0.1. Stage 2 (full) 은 baseline 과 동일 (1.5e-5 · 3 epoch). 값 재확인: `python -c "from implicit_world_modeling.lf_registry import _DATASET_CONFIG as D; import json; print(json.dumps({k:D['AndroidControl_EXP07'][k] for k in ('stage1','stage2')}, default=str, ensure_ascii=False))"`

> Stage 2 full 의 `1.5e-5` 는 `_DATASET_CONFIG` 가 아니라 `gen_configs.render_stage2()` 안에 하드코드돼 있다 (LoRA 대비 안정화).
>
> ⚠️ **함정 15 — EXP07 은 v1/v2 두 버전이다 (2026-07-29). 위 `AndroidControl_EXP07` 키는 이제 `AndroidControl_EXP07_v1`/`_v2` 다.** 버전은 experiment/데이터 디렉토리가 아니라 **model 아티팩트 이름 끝 trailing `_v1`/`_v2`**(HF는 끝 `-v1`/`-v2`)에 붙는다. **공유(버전 없음)**: output 부모 `AndroidControl_EXP07`, 데이터 디렉토리 `data/AndroidControl_EXP07`, config subfolder `configs/train/IWM-AC_EXP07`, **모든 test 파일**(v1≡v2 공통). **버전별 차이는 데이터뿐**: v1 = state 40K(diff `W_UNCHANGED`=0.2) + down 10K, v2 = state 24K(`W_UNCHANGED`=0.05) + down 6K (8:2 유지) + **95% 복사-편향 필터**(state 예측에서 next≈current, diff v2 UNCHANGED비율 ≥0.95 제거 ≈12%; train+공통test). 구현: lf_registry `model_run_suffix`/`config_version_suffix` 등 4필드, `_common.sh` `ds_version_suffix`/`ds_config_subfolder`(v→공유 IWM-AC_EXP07) + 학습/merge/eval 스크립트 배선, `build_exp07_data.py --version {v1,v2}`. `eligible_models('AndroidControl_EXP07_v1')`/`_v2` 각각 `['qwen2.5-vl-3b']`. `gen_configs --check` = **202**(v1 9 + v2 9). 값 재확인 시 위 명령의 키를 `AndroidControl_EXP07_v1` 로.
>
> ⚠️ **함정 16 — stage eval 의 생성 토큰 예산 (2026-07-29, 커밋 6a4b59e).** `build_infer_cmd` 가 `--max_new_tokens` 를 안 넘겨 `vllm_infer` 기본 **1024** 로 생성되던 탓에, **state prediction(전체 UI XML) 출력이 1024 토큰에서 하드컷**됐다 (state test 라벨 max=11218 / p50=1664 → **69% 초과**, Hungarian F1 무효). action(라벨 max 441)은 무영향. 수정: task 별 예산 — state/hungarian(dual-task state, MC/MB) **12288**, action **2048**. 이 버그 이전의 EXP07 state F1(0.32~0.58)은 전부 무효이며 재-inference 필요하다 (action step_acc 는 유효).
>
> 설계 근거 (dropout 0.1 유지): 과거 실측에서 dropout 0.10 이 저빈도 action type 을 불안정하게 만든다는 관측이 있으나, **EXP01 기존 stage2 어댑터 (d0.1 / 5e-5) 와의 동일조건 비교를 우선**해 baseline 을 유지한다 — 그래야 EXP02 stage2 만 재학습해도 EXP01 전체 (학습/eval) 를 보존할 수 있다.

`freeze_vision_tower: true` 는 등록된 모델 전부. Full FT 분기에서 `vision_tower|vision_model|visual|image_encoder` 키워드를 포함한 named parameter 를 `requires_grad=False` 처리한 뒤 frozen 텐서/파라미터 수를 stderr 로 출력한다.

---

## 3. 데이터와 설정 계약

### 데이터 계보 — EXP02~EXP05 는 AC_EXP01 파생, EXP07 은 별도 소스

```
data/AndroidControl/              # 원본 source 자산 — 학습/평가 entry 아님 (DS_DATADIR 미등재)
  ├── implicit-world-modeling_stage1_{state,action}.jsonl   # 원천 (이름 그대로 유지 — 어느 EXP 소속도 아님)
  ├── implicit-world-modeling_stage2.jsonl                  # 원천
  ├── EXP01_stage1_{state,action}.jsonl                     # filter_long_samples.py 산출 (EXP01 계보 입력)
  ├── EXP01_stage2.jsonl
  ├── EXP03_stage1_{action,state}.jsonl                     # EXP03 원천 (0–1000 정규화 좌표)
  ├── EXP03_stage2.jsonl                                    # EXP03 stage2 원천
  ├── EXP05_stage1_{action,state}.jsonl                     # EXP05 원천 (절대 픽셀, Drive '0711_버젼')
  ├── EXP07_stage1_state.jsonl                              # ★ EXP07 원천 — 계보 다름 (0725 myset, EXP01 파생 아님)
  ├── EXP07_stage2.jsonl                                    #   downstream(with_history). stage2 15K 가 주 소비자,
  │                                                         #   stage1 down 10K·stage1 action test 도 같은 풀 (규칙 3)
  ├── EXP07_open_aug.jsonl                                  #   open 증강 (home.jpg) — s1/s2 에 50 씩 균등 → stage 중립명
  ├── episodes_meta.jsonl         # primary_app = 전경 앱 package_name
  └── images/                     # ★ 유일한 이미지 디렉토리 — EXP01~EXP07 전부가 "AndroidControl/images/..." 로 참조
      ⚠ EXP04 원천 (EXP04_stage1_{action,state}.jsonl) 은 디스크에 없다 → 재빌드 불가 (§2 경고 블록)
      ⚠ 이 디렉토리의 파일이 전부 위 원천 3 종에서 파생된 건 아니다 — EXP01/03/05 계열은 원천 파생이지만
        **EXP07_\* 는 0725 myset 이라는 별도 provenance 스트림**이다 (아래 계보도 참조).

  AC_EXP01  = filter_long_samples → split_data.py       (Stage1 ratio mix 3 종 + Stage2 ID/OOD)
     │
     ├── AC_EXP02 = ratio73 train + diff loss v1 token_weights    (test/Stage2 는 EXP01 복사)
     ├── AC_EXP03 = ratio73 멤버십의 좌표(point) 미러             (mirror_experiment.py --experiment exp03)
     │      └── AC_EXP04 = EXP03 + stage1 프롬프트 업그레이드     (--experiment exp04, stage1-only)
     └── AC_EXP05 = ratio73 멤버십의 절대 픽셀 미러 + diff loss v2 (build_exp05_data.py; stage2 는 2026-07-15 별도 도입 → build_exp05_stage2_data.py)

  AC_EXP07  = EXP01 계보 밖 — AndroidControl/EXP07_*.jsonl (0725 myset) 에서 자체 빌드 (build_exp07_data.py, seed 7)
     ├── train  : stage1 50K (state 40K 가중 + downstream 10K 이미지제거) + stage2 15K   ← 전부 myset 원천
     ├── test   : 자체 6 종 (stage1 state 2 · stage1 action 2 · stage2 2)
     └── EXP05 의존은 test 정의뿐 — EXP05 test 의 (episode, step) 키를 myset 포맷으로
         재현해 자체 파일로 굽고 id/ood 배정을 승계한다. train 내용은 EXP05 와 무관하며
         그 test 키 union 을 train 두 풀에서 전량 제외해 train ∩ test = 0 을 만든다.
         이미지는 myset → AndroidControl/images/ 로 경로 문자열 remap (파일 복사·링크 없음).

  MonkeyCollection  = Stage 1 전용 (random split 0.95)
  MobiBench         = 평가 전용 (stage{1,2}.jsonl 단일 파일)
```

행수·파일 목록은 파생값이다 — 디스크에서 직접 센다: `wc -l data/AndroidControl_EXP05/*.jsonl`

#### 데이터 파일명 규칙 (2026-07-25 확정)

1. **디렉토리가 실험을 말하면 파일은 침묵한다.** `AndroidControl_EXP{NN}/` · `MonkeyCollection/` · `MobiBench/` 안에서는 `stage{N}_<role>.jsonl` 을 쓴다 (`implicit-world-modeling_` 접두 없음). 예: `stage1_train.jsonl`, `stage1_test_ood_action.jsonl`, `stage2_test_id.jsonl`.
2. **공유 디렉토리의 변형본만 `EXP{NN}_` 접두를 쓴다.** `data/AndroidControl/` 은 여러 실험군이 함께 쓰는 폴더라 파일이 소속을 스스로 말해야 한다 — `EXP01_stage1_state.jsonl`, `EXP03_stage2.jsonl`, `EXP05_stage1_action.jsonl`. **표현 서픽스(`_xy` / `_pixel-aligned` / `_filtered`)는 쓰지 않는다** — EXP 소속이 이미 좌표 표현·필터 여부를 함의하고, 그 정의는 이 문서(§2 자격 매트릭스 · 위 계보)가 기록한다.
3. **멀티-exp 공유는 단일 소유 + 계보 기록.** 한 파일을 여러 실험군이 읽어도 접두는 **직접 소비하는 최초 EXP 하나**만 붙인다 (누가 재사용하는지는 문서가 기록한다). 태생부터 공동 소유로 신설할 때만 `EXP05+07_` 처럼 `+` 로 잇는다.
4. **유지되는 서픽스·sidecar**: 평가 파생본의 `_without_open_app`, train 의 sidecar `<train>.jsonl.meta.json` 은 규칙 그대로다. EXP01 ratio 파일은 dataset key 표기와 통일해 `stage1_train_ratio{37,55,73}.jsonl`.
5. **빌더 입력 원천도 예외가 아니다 — `*_src` 전용 디렉토리를 만들지 마라 (2026-07-26 확정).** 벤더가 준 원천이라도 특정 EXP 의 변형본이면 규칙 2 를 그대로 받는다: `data/AndroidControl/` 에 `EXP{NN}_` 접두로 **평탄하게** 놓고 벤더 파일명(`all_samples_*_filtered_v2` 류)은 버린다. 전용 디렉토리는 등록 키가 0 이라 `_common.sh` 의 심링크 자기치유에 걸려 유령 심링크를 만들고, 규칙 2 를 비껴가는 유일한 예외를 남긴다. 한 원천 파일이 **여러 stage 를 먹이면** 규칙 3(단일 소유 + 계보 문서화)을 stage 축으로 확장 적용해 **최대 소비자**의 이름을 쓰고 나머지 소비는 이 문서가 기록한다 (예: `EXP07_stage2.jsonl` 은 stage2 15K 가 최대 소비자지만 stage1 down 10K·stage1 action test 도 같은 풀에서 나온다). 다만 **stage 간 균등 분배라 최대 소비자가 없으면** stage 이름을 붙이는 순간 거짓이 되므로 **stage 중립명**을 쓴다 (예: `EXP07_open_aug.jsonl` — `AUG_PER_STAGE=50` 으로 s1/s2 에 정확히 반씩). 이때 `open`·`aug` 는 내용·역할이지 좌표 표현이 아니므로 규칙 2 가 금지한 표현 서픽스에 해당하지 않는다.

원본 3 종 (`implicit-world-modeling_stage1_{state,action}.jsonl` · `implicit-world-modeling_stage2.jsonl`) 과 `episodes_meta.jsonl` · `images/` 는 **어느 EXP 소속도 아닌 원본**이라 이름을 바꾸지 않는다.

| 실험군 | Stage 2 | Stage 1 test | 좌표 표현 | diff loss |
|---|---|---|---|---|
| AC_EXP01 (ratio 37/55/73) | ✓ (ratio sweep) | (id,ood) × (state,action) 4 파일 | index | — |
| AC_EXP02 | ✓ (diff loss 미적용) | EXP01 복사 (4) | index | **v1** (stage1) |
| AC_EXP03 | ✓ | EXP01 좌표 미러 (4) | point | — |
| **AC_EXP04** | ✗ `_STAGE1_ONLY` | EXP03 미러 + 프롬프트 (4 + without_open_app 2) | **§2 경고 참조** | — |
| **AC_EXP05** | ✓ (2026-07-15 도입) | 4 + without_open_app 2 | **절대 픽셀 840×1876** | **v2** (stage1) |
| **AC_EXP07** | ✓ (자체 train 15K) | **자체 4** (id,ood) × (state,action) | **절대 픽셀 840×1876** | **v2** (stage1 state, 인라인 1:0.2) |
| MC | ✗ (데이터 없음) | 단일 test | — | — |
| MB | 평가 전용 | 단일 파일 | — | — |

`_STAGE1_ONLY = {MonkeyCollection, AndroidControl_EXP04}` — Stage 2 YAML/등록/eval 을 전부 skip (**EXP05 는 2026-07-15 stage2 도입으로 이 집합에서 제거됨**). `_DUAL_TASK_TEST` 는 AC_EXP01~EXP05 + AC_EXP07 (Stage 1 한정, §6). `_SINGLE_TEST = {MonkeyCollection}` — 세 플래그는 **직교**한다.

> ✅ **해소된 함정 (2026-07-25) — EXP07 은 `_DUAL_TASK_TEST` 에 편입됐다.** 그 전까지 EXP07 은 디스크·`dataset_info` 에 dual-task test 4 종을 다 갖고도 세트에서 빠져 있어, `build_configs()` 가 else 분기를 타 **미등록 키** `IWM-AC_EXP07_stage1_test_{id,ood}` 를 `ds_s1_test_*` 에 담았다 (실제 등록명은 `…_test_{id,ood}_{state,action}` 4 개다).
> - **학습·평가는 그때도 무증상**이었다 — 셸이 이름을 직접 조립하기 때문이다 (`stage1_eval.sh` 의 `${eval_prefix}_stage1_test_{id,ood}_${task}`), 그리고 stage1 YAML 은 `dataset:` (train) 만 참조한다. 그래서 편입 후에도 **커밋된 YAML 은 한 글자도 안 바뀐다** (`gen_configs` 는 `ds_s1_train`/`ds_s2_train` 만 읽고, EXP07 은 `ac3_ratio` 가 없어 두 분기의 렌더 결과가 문자열 동일).
> - **증상이 있던 유일한 곳은 (은퇴한) 노트북의 등록 검증 셀**이었다 — `_expected_keys()` 가 같은 `_DUAL_TASK_TEST` 분기를 미러해 EXP07 에 대해 없는 bare 키를 "기대" 하고 실재하는 4 키를 누락으로 봤다. 그 셀은 2026-07-26 노트북 은퇴로 사라졌고, `_DUAL_TASK_TEST` 편입으로 근본 원인도 해소돼 **이 함정은 이중으로 소멸했다.**

- **EXP03 미러**: EXP01 ratio73 산출 파일을 한 줄씩 읽어 `(episode, step)` 키로 좌표 원천의 대응 레코드를 골라 **동일 순서로** write. UI 트리는 `index="N"` 대신 `bounds="[x1,y1][x2,y2]" point="[cx,cy]"`, 액션은 `point=[x,y]`. **본문만 좌표이고 이미지 경로는 EXP01 것을 채택**한다. 원천에 없는 키 (~0.8–1.7%) 는 제외 → EXP01 과 `(episode, step)` 1:1 대응이나 행 수는 소폭 작다.
- **EXP04 미러**: EXP03 와 **동일 멤버십·좌표 표현**, 프롬프트만 업그레이드 (action space `scroll(direction, point)` → `swipe(start, end)`, role 문구 "represented as html-style XML", `[SWIPE]` 규칙). **EXP04 pool ⊆ EXP03 pool** 이라 멤버십 drift 가 없다.
- **EXP05 미러**: 절대 픽셀 좌표. 출력 이미지 경로는 EXP01 의 `AndroidControl/images/...` 재사용 (source 의 `myset/images/...` 는 매칭 키 추출용).
- **EXP07 빌드**: 빌드 정본 [`scripts/build_exp07_data.py`](./scripts/build_exp07_data.py) — **EXP07 train/test jsonl 의 유일한 커밋된 생성 경로**다. 원천은 `data/AndroidControl/` 의 0725 myset 필터링본 3 파일 (`EXP07_stage1_state` state-pred · `EXP07_stage2` with_history downstream · `EXP07_open_aug` open 증강) 이고, 같은 디렉토리의 EXP05 파일은 **읽지 않는다** (test 키 추출 예외 — 아래). 산출 (train 2 · test 6 · stage1 sidecar 1) 전부 `data/AndroidControl_EXP07/` 안의 **실파일**이다 — 데이터 파일 심링크는 없다. (헷갈리지 말 것: `configs/lf_dataset/AndroidControl_EXP07 → ../../data/AndroidControl_EXP07` **dataset_dir 심링크는 다른 것이고 반드시 필요하다** — `_common.sh` 가 만든다.)
  - **train 구성** (빌더 상수): stage1 50K = state 40K (이미지 유지, diff v2 가중) + downstream 10K (**이미지 제거**), stage2 15K = downstream (이미지 유지, 무가중). 두 downstream 풀은 **비중복** (with_history 풀을 disjoint 분할, action 비율 largest-remainder) 이고, open 증강과 answer-terminate 행을 각 stage 에 **우선 배치**한다 (가용분·쿼터에 맞춘 동적 절반 — 실현치는 sidecar 의 `priority` 참조).
  - **누출 0 (설계 불변식)**: EXP07 test 는 EXP05 test 의 `(episode, step)` 키를 myset 포맷으로 재현해 만든 **자체 파일**이고 (id/ood 배정은 EXP05 승계), 그 test 키 union 을 **train 두 풀에서 전량 제외**한다 → 교차목적을 포함해 `EXP07 train ∩ EXP07 test = 0`. 빌더의 `verify()` 가 이 불변식을 fail-closed 로 검사한다.
  - **이미지 remap**: `myset/images/episode_{N}_step_{M}.jpg` → `AndroidControl/images/episode_{N:06d}_step_{M}.jpg` (zero-pad 6), `home.jpg` → `AndroidControl/images/home.jpg`. 즉 EXP07 도 이미지는 공용 `AndroidControl/images/` 를 참조한다 (하드 제약 12 의 prefix 규약 그대로).
  - **diff v2 1:0.2 는 데이터-빌드 시점 인라인이라 실험군 격리**다: `ADDED 1.0 / MODIFIED 1.0 / UNCHANGED 0.2` 가중이 EXP07 train jsonl 에만 박혀 EXP02(v1)·EXP05(v2 1:0.25) 에 영향이 없다. 학습측 배선은 EXP05 와 동일하게 stage1 YAML 의 `use_diff_token_weighted_loss: true` 뿐이다 (§3 diff loss 절).
  - 재빌드는 `--source-dir` / `--seed` 로 한다 (`W_UNCHANGED=0.2` · metric v2 는 불변식으로 고정). 과거 잠정본이 남긴 EXP05 심링크는 빌더의 `cleanup_legacy_symlinks()` 가 제거한다 — 그 시절의 `--links-only` 플래그는 **없어졌다**.

### MC 데이터 상태 (2026-07-14 — 프로덕션 코퍼스 아님)

레지스트리 (`lf_registry.py` `"MonkeyCollection"`) · `dataset_info.json` (`IWM-MC_stage1_{train,test}`) ·
`configs/train/IWM-MC/` · `split_data.py --dataset MC` 는 **전부 이전부터 존재**했으나, 이 파이프는
2026-07-14 에 **처음 실행됐다**. 현재 `data/MonkeyCollection/` 의 164 examples (train 155 / test 9) 는
Monkey-Collector 의 **실험 잔여물** (musicplayer + calendar) 이며 **프로덕션 코퍼스가 아니다** — osmand 는
page 지문 오염 (S-9) 으로 의도적으로 제외했고, 수집기의 page_graph 에는 **교차-앱 병합 오염**이 남아 있다
(수집기 `AGENTS.md` 「알려진 한계」). **이 데이터로 낸 학습 결과를 코퍼스 품질의 근거로 쓰지 마라** (규칙:
[AGENTS.md](./AGENTS.md) 「인용 금지」). 또한 `--stage1-ratio` 기본값 0.95 를 164 행에 적용하면 **test 가
9 개**뿐이라 통계적으로 무의미하므로 실제 학습 전에 비율을 다시 정해야 한다.

**MC 브리지 end-to-end 실행은 보류됐다 (사용자 결정, 2026-07-14 — 지금은 핸드오프만).** 배선·정적 관통은
확인됐으니 남은 것은 원격 GPU 박스에서 `stage1_train.sh --dataset MC` 를 실제로 한 번 돌려보는 것뿐이다.
경로: `setting:claude-code-remote` 로 원격 서버에 붙거나 `scripts/remote_launch.sh`. 맥에서 검증 가능한
최대치는 **정적 관통 확인** (jsonl 스키마 + 이미지 경로 해석) 이다 — `stage1_train.sh` 자체는 `_common.sh`
가 요구하는 `CONDA_PREFIX` (conda env) · bash 4+ 와 체크아웃에 없는 `LlamaFactory/` 때문에 맥에서
실행되지 않는다.

행수 재확인: `wc -l data/MonkeyCollection/*.jsonl`

### 분할 규칙 (`scripts/split_data.py`)

- **App partition** (`compute_app_partition`, 원본 `episodes_meta.jsonl` 기준): Stage 2 행 수를 budget 으로 `(id_apps, ood_apps)` 를 **한 번** 계산하고 Stage 1 이 같은 partition 을 재사용한다. → **Stage 2 OOD 앱은 Stage 1 train 에 한 번도 등장하지 않는다** (world-modeling 이 OOD 앱을 보지 못함). EXP02~05 는 EXP01 산출 멤버십을 미러하므로 이 partition 을 그대로 승계한다 (별도 계산 없음).
- **AC_EXP01**: source 는 항상 원본 `data/AndroidControl/`, 산출은 `data/AndroidControl_EXP01/`. 선행 `filter_long_samples.py --dataset AC_EXP01` 가 mm-expanded length > cutoff 인 row 를 제거해 **원본 폴더 안에** `EXP01_stage1_{state,action}.jsonl` + `EXP01_stage2.jsonl` 3 개를 만든다 (Qwen3-VL `get_rope_index` broadcast 회피). split 은 **이 `EXP01_*` 만 입력으로 쓴다** (Stage 2 source 누락 시 hard-fail). `state_pred` 는 random, `action_pred` / Stage 2 는 **action_type stratified**. ratio (state:action ∈ {7:3, 5:5, 3:7}) 로 혼합한 stage1 train 3 종 + 같은 partition 의 Stage 2 split (기본 15K / 3K / 3K) 을 한 번에 산출. `--exp01-train-total` 이 Stage 1 train 합계 (기본 50K).
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

> ⚠️ **함정 10 — v1 4 파일을 삭제하지 마라. 데드 코드가 아니다.** `scripts/build_exp02_data.py` (구 노트북 Cell 7 이관본) 가 이 v1 (`scripts/diff_loss/preprocess_dataset.py`) 로 **EXP02 데이터를 생성하는 유일 경로**이고, v1 재실행은 저장된 `token_weights` 와 **40/40 일치**한다 (v2 로 돌리면 17/40). v1 의 경계 비대칭 버그도 **EXP02 재현성 보존을 위해 의도적으로 고치지 않는다** (빌더는 `preprocess_dataset_v2.py` 를 절대 쓰지 않는다).

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
cat data/AndroidControl_EXP05/stage1_train.jsonl.meta.json
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
| AC_EXP05 | stage1 train + dual-task test + **stage2 (train/test_id/test_ood)** (2026-07-15 등록, `_STAGE1_ONLY` 에서 제거) |
| **AC_EXP07** | **8키 전부 자체 경로** = stage1 train + stage1 test_{id,ood}_{state,action} 4 + stage2 train/test_id/test_ood — 모든 `file_name` 이 `../../data/AndroidControl_EXP07/…` 를 가리킨다 (EXP05 포인터·심링크 없음). **이 8 키는 손으로 넣은 게 아니라 `build_exp07_data.py::register_dataset_info()` 가 빌드할 때 정본에 써넣는다** — 유일한 예외적 쓰기 경로다 (함정 13 의 "런타임 변조 금지" 는 학습·평가 런타임을 말한다; 빌드 타임 등록은 결과를 커밋한다). |
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
- 과거 스키마의 사문화된 키 (`IWM-AC_stage*` 6 · `IWM-AC_2_*` 4) 는 **제거 완료 (2026-07-25)** — 파일이 실재하지 않는 등록은 정본에 하나도 없다. 이제 전 키가 실파일로 resolve 한다.

### 이름 규약

| | AC_EXP01 | AC_EXP02~07 | MC | MB |
|---|---|---|---|---|
| `data/` 디렉토리 | `AndroidControl_EXP01` | `AndroidControl_EXP0{2..7}` | `MonkeyCollection` | `MobiBench` |
| shell 코드 | `AC_EXP01` (→ `AC_EXP01_ratio{37,55,73}` expand) | `AC_EXP0{2..7}` | `MC` | `MB` (eval 전용) |
| LF prefix | `IWM-AC_EXP01` | `IWM-AC_EXP0{2..7}` | `IWM-MC` | `IWM-MB` |
| `outputs/` 최상위 | `AndroidControl_EXP01` (ratio 는 model dir suffix) | `AndroidControl_EXP0{2..7}` | `MC` | — (TRAIN_DS 산하 `on-MB/`) |
| HF slug | `ac-exp01-ratio{37,55,73}-` | `ac-exp0{2..7}-` | `mc-` | `mb-` (dormant) |

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

> ⓘ **fractional epoch 라벨 (EXP07 stage1).** EXP07 stage1 은 `save_strategy: steps` + `save_steps: 0.25` 라 epoch 경계가 아닌 **0.25 단위**로 체크포인트를 남긴다 → `ckpt_epoch_from_dir` (`trainer_state.json.epoch` 을 float 로 읽음) 이 라벨을 **0.25 / 0.5 / 0.75 / 1** 로 확장하고, `epoch-{E}/` 디렉토리·eval `--epochs` 도 소수를 허용한다. **EXP01–06 은 정수 epoch 표기·동작 그대로**다 (회귀 byte-diff 0 검증). 이 라벨이 `--stage1-epoch` 로 Stage 2 world-model variant 의 상류 계보에 그대로 전달된다.
- **`stage1_eval.sh`** — Phase A (zero-shot `base`) + Phase B (`--epochs`, 기본 `1,2,3` merged sweep). model path 는 `resolve_eval_model_path` 가 **local merged dir 우선 + HF Hub fallback** 으로 결정 (merged 이므로 adapter 인자·`max_lora_rank` 불필요). marker (`hungarian_metrics.json`) 존재 unit 은 skip.
  - **without_open_app 자동 산출**: 정규 score 직후 **추론 재실행 없이** `--exclude-action open_app` 로 한 번 더 채점해 GT `open_app` 행을 양쪽에서 동시 drop 한 sibling 디렉토리 (`on-{EVAL_DS}-without-open_app/`) 를 idempotent 하게 만든다. 필터 test JSONL 은 `data/{DATADIR}/` 에 영구 보존한다. **state branch 만** — action 채점기는 미지원.

    > ⚠️ **함정 — 이 필터는 지금 전 실험군에서 무동작이다 (2026-07-13 실측).** `_hungarian_eval.py::_gt_action_type` 이 `ACTION_MARKER = "## Action\n"` 를 찾는데 데이터가 그 마커를 그렇게 담고 있지 않아 **항상 `None`** 을 돌려준다 → drop 되는 행이 **0** 이다 (EXP01 state test 3,000 행 전수 확인: non-None 0건). 따라서 `on-{DS}-without-open_app/hungarian_metrics.json` 은 **정규 메트릭과 수치가 같다.** 이걸 "open_app 제외 성능" 으로 읽으면 **거짓 결론**이 나온다 — 필터를 고치기 전까지 비교 근거로 쓰지 마라. (열린 이슈: [`docs/ROADMAP.md`](../docs/ROADMAP.md))

### Stage 2

`--stage2-mode {full|lora}` (기본 `lora`), `--stage1-mode` + `--stage1-epoch N` 으로 world-model variant 의 상류 계보 결정. `base` variant 는 Stage 1 무관.

- **`stage2_train.sh`** — YAML `…/stage2_${MODE2}/{MODEL}_{base,world-model-full,world-model-lora}.yaml`. 실행 env 는 **Stage 1 과 동일**하다 — `FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=…` (`stage2_train.sh:101` ↔ `stage1_train.sh:43`). 유일한 차이는 stage1 만 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 추가로 export 한다는 것뿐이다. world-model variant 는 local `merged/…/epoch-${N}/` 을 base 로 쓰도록 `model_name_or_path` 를 런타임 sed 치환하고 (그래서 YAML 의 HF id placeholder 는 무시된다), YAML 의 `__STAGE1_EPOCH__` 플레이스홀더가 `${N}` 으로 치환돼 결과가 `…_world-model_from_{M1}-ep{N}/` 로 분리 저장된다. 디렉토리 미존재 시 hard-fail.
- **`stage2_merge.sh`** — Full FT 는 checkpoint 자체가 전체 모델 (adapter 블록 없음), LoRA 는 `model_name_or_path: {base}` + `adapter_name_or_path: {ckpt}` + `finetuning_type: lora`.
- **`stage2_eval.sh`** — `--variants` 로 `base` / `{full|lora}_base` / `{full|lora}_world_model` 선택. EVAL_DS=AC_EXP01/02/03/05 는 ID+OOD 동시 추론 → 3 섹션, MB 는 single-pair 1 섹션. **AC_EXP05 는 xy 통일 액션 스페이스라 `_action_eval.py score --coord-mode xy` 로 채점**한다 (stage1_eval 의 EXP05 분기와 동일; 나머지 EXP 는 플래그 없이 index 채점). marker (`action_metrics.json`) 존재 unit 은 variant × EVAL_DS 별 독립 skip.
  - **`--epochs` 에 `0` 포함 (opt-in)**: `{full|lora}_world_model` 의 epoch-0 = stage2 미학습 베이스라인 (= stage1 merged 와 동일 모델). `{full|lora}_base` 는 stage1 계보가 없어 epoch-0 이 `base` 와 중복 → 경고 후 skip. 기본 `1,2,3` 에는 미포함.

> ⓘ **merge X 변형 (`world-model-adapter`) — EXP07 한정 opt-in.** 일반 world-model variant 는 stage1 어댑터를 **merge 한** local merged dir 을 base 로 삼지만(merge O), EXP07 은 `_DATASET_CONFIG["AndroidControl_EXP07"]["stage2_adapter_variant"] = True` 로 `stage2_lora` 에 **merge 하지 않는** 변형을 하나 더 렌더한다. 그 YAML (`…/stage2_lora/{MODEL}_world-model-adapter.yaml`) 은 `model_name_or_path` 를 **원본 base 그대로** 두고 `adapter_name_or_path: __STAGE1_ADAPTER__` placeholder 만 둔다 — `stage2_train.sh` 가 런타임에 stage1 LoRA 어댑터 checkpoint 경로로 sed 치환해 base 위에 얹어 이어학습한다 (merge 스텝 없음). 산출은 `…_world-model_from_adapter-ep{E1}/` 로 `from_full`/`from_lora` 와 대칭 분리되고, HF slug 는 `_common.sh::hf_repo_id_stage2_world_model_adapter` 가 조립한다 (`…world-model-stage1-lora-epoch{E1}-stage2-lora-adapter-epoch{E2}`). eval 변형 `lora_world_model_adapter` 는 `STAGE2_EXTRA_VARIANTS` 라 **기본 sweep 에 미포함** — `--variants lora_world_model_adapter` 로 명시할 때만 채점된다.

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

- **Stage 2 대상 DS 는 `_STAGE1_ONLY` 가 결정한다** (§3) — `MC` · `AC_EXP04` 는 `_STAGE1_ONLY` 라 (MC 는 stage2 데이터 자체가 없다) Stage 2 YAML 이 생성되지 않고 `require_yaml` 에서 중단된다. 즉 Stage 2 는 `AC_EXP01 | AC_EXP02 | AC_EXP03 | AC_EXP05` 다 (**EXP05 는 2026-07-15 stage2 도입**).
- 각 스크립트의 `--help` 가 플래그의 최종 정본이다.

> ⚠️ **함정 16 — 유령 스크립트를 인용하지 마라.** tmux 실행 스케줄 같은 일회성 스크립트는 `.gitignore` 의 `scripts/tmux_*.sh` 로 **추적 제외**된다 (커밋 `0303167`). 저장소에 존재하지 않으므로 문서가 복붙 커맨드로 제시하면 안 된다 — 실행은 위 `stage{1,2}_*.sh` 를 직접 호출한다.

> ⚠️ **함정 17 — `SMOKE=1` override 의 따옴표.** `save_strategy` 값은 반드시 따옴표를 살려 넘긴다. OmegaConf 는 YAML 1.1 규칙으로 따옴표 없는 `no` 를 boolean `False` 로 파싱하고, HF 가 `"False is not a valid SaveStrategy"` 로 죽는다.

---

## 5. 실행 데이터 흐름과 산출물

```
raw JSONL + screenshots
  → extract_androidcontrol_images.py     (data/AndroidControl/images/ : GCS REST → PNG, TF 의존 없음)
  → extract_androidcontrol_metadata.py   (episodes_meta.jsonl : primary_app 다수결)
  → filter_long_samples.py --dataset AC_EXP01     (원본 폴더에 EXP01_*.jsonl 3 개)
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
| Stage 2 world (merge X, EXP07) | `SaFD-00/{short}-{slug}world-model-stage1-lora-epoch{E1}-stage2-lora-adapter-epoch{E2}` |

조립은 `_common.sh::hf_repo_id_stage1` / `hf_repo_id_stage2_base` / `hf_repo_id_stage2_world_model` (+ EXP07 merge X 는 `hf_repo_id_stage2_world_model_adapter`) 로 단일화. eval 의 model path 해석은 `resolve_eval_model_path {stage1|stage2_base|stage2_world}` 가 **local merged dir 우선 + HF fallback** 으로 처리한다 → local merge 한 머신에서 같은 머신 안에서 바로 eval 까지 이어 돌 수 있다 (`HF_TOKEN` 불필요).

---

## 6. 메트릭

**자동 winner 선정은 없다.** 두 stage 모두 `score` 서브커맨드로 JSON 을 저장하고, 사용자가 결과를 보고 Stage 2 에 쓸 Stage 1 epoch 을 `--stage1-epoch` 로 지정한다.

### Stage 1 — `hungarian_metrics.json`

- variant: `base` (zero-shot) / `full_world_model` / `lora_world_model`
- metric: `avg_hungarian_f1` (1차), `avg_bleu`, `avg_rouge_l` 등
- single-pair (`--test/--pred`) 와 ID/OOD (`--test-id/--pred-id/--test-ood/--pred-ood`) 모두 지원 — ID/OOD 모드는 `overall` / `in_domain` / `out_of_domain` 3 섹션.

> **dual-task 분기 (Stage 1 한정)**: EVAL_DS 가 AC_EXP01~AC_EXP05 · AC_EXP07 (`_DUAL_TASK_TEST`) 이면 `state_pred` 와 `action_pred` 를 **각각 독립 채점**한다 — `on-{DS}-state/hungarian_metrics.json` (Stage1 채점기) + `on-{DS}-action/action_metrics.json` (**Stage2 채점기**). 각 task 가 (id, ood) 2 파일을 가지므로 inference 는 4 회. without_open_app sibling 은 state branch 만.
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

### thought 유사도 메트릭 (`thought_metrics.json`) — Stage 2 자동 hook

Stage 2 pred 의 `<thought>…</thought>` 텍스트가 GT thought 와 얼마나 가까운지를 잰다. **행동 정확도(SA)와 독립**이며, `stage2_eval.sh` 가 action 채점(`action_metrics.json`) **직후 자동으로** `scripts/thought_eval.py` 를 호출해 같은 `on-{EVAL_DS}/` 아래 `thought_metrics.json` 을 남긴다 (별도 추론 없음 — pred/label 재사용). EXP07 이 이 메트릭을 스펙에 명시적으로 넣지만, hook 자체는 Stage 2 eval 전반에 배선돼 있다.

- **GT 소스**: pred 파일의 `label` 필드에서 thought 를 뽑는다. **GT-thought 가 없는 행은 `n` 에서 제외**되고, 대상 행이 하나도 없으면(no-op) 파일을 만들지 않는다 (파이프라인 비차단).
- **구조**: `overall` / `in_domain` / `out_of_domain` 3 섹션 × 각 섹션에 `n`, `missing_thought_{n,ratio}`, 세 지표의 `mean`/`std`:
  - **`cosine`** — 문장 임베딩 코사인 유사도. **1차 지표(정본)**. 기본 임베딩 모델 `all-MiniLM-L6-v2`, `--embed-model` 로 교체 가능.
  - **`rouge_l`** — LCS 기반 ROUGE-L (자체 구현).
  - **`bleu`** — `sacrebleu` (0~1 정규화).

> ⚠️ **함정 — thought eval 이 유의미하려면 실행 env 에 `sentence-transformers`·`sacrebleu` 가 있어야 한다.** 둘 다 `pyproject.toml` 에 등재돼 `.venv` 는 충족하지만, conda env 는 이들이 자동으로 따라오지 않으므로 **새로 만들 때마다 별도 설치**가 필요하다. 미설치 시 `cosine`/`bleu` 는 `null`, `rouge_l` 만 산출되며 **파이프라인은 차단되지 않는다** — "cosine 이 비었다" 를 성능 결론으로 읽지 마라.
>
> **현재 conda env `implicit-world-modeling` 은 충족 상태다** (2026-07-26 설치·실측): `sentence-transformers` 있음, `sacrebleu` **2.6.0** · `portalocker` 3.2.0 · `colorama` 0.4.6 설치됨(`tabulate` 0.10.0 은 이미 충족돼 미변경). EXP07 test 300 쌍 오라클 실행에서 `cosine`/`rouge_l`/`bleu` 모두 산출을 확인했다.
>
> **env 를 재구성할 때 설치 방법은 반드시 `pip install --no-deps sacrebleu portalocker tabulate colorama`** — 의존성을 끌어오지 않는 형태여야 한다. 평범한 `pip install sacrebleu` 는 `numpy` 를 올려 같은 env 의 torch/vLLM 을 깨뜨릴 수 있다. `--no-deps` 로 빠지는 나머지 의존성(`regex`·`lxml`·`numpy`)은 이 env 에 이미 있어 위 4 개만으로 충분했다. 실제로 `--no-deps` 설치 후 `pip freeze` 차분은 **추가 3 줄뿐이고 numpy 2.2.6 / torch 2.8.0+cu128 / vLLM 0.11.0 은 그대로**였다.

정본은 `scripts/thought_eval.py` (회귀 테스트 `tests/test_thought_eval.py`). 실험 결과 수치는 Notion `🧪 Experiments` DB 가 정본이다 — 여기엔 정의만 둔다.

---

## 7. 중요한 운영 제약

- **`implicit_world_modeling/` 패키지에 학습·평가 실행 로직은 없다** — 레지스트리 SSoT (`lf_registry.py`) 와 YAML 생성기 (`gen_configs.py`) 뿐이다. 실행 로직은 전부 `scripts/` 에 있다.
- **모델 추가 시**: `lf_registry._MODEL_CONFIG` + `_common.sh` 의 `MODEL_ID`/`MODEL_TEMPLATE`/`ALL_MODELS` 를 **동시에** 고치고 `gen_configs --write` 로 YAML 을 재생성해 커밋한다. 새 family 라면 `MODEL_FAMILY_CONFIG` 에 image budget 을 **먼저** 등록해야 하고 (`_img_cfg` 가 `KeyError` 를 낸다), `build_infer_cmd` 의 template 분기 (factor / mm_min) 도 함께 갱신한다.
- **하이퍼파라미터는 `_MODEL_CONFIG` 에 직접 쓰지 마라** — `_DATASET_CONFIG` baseline (또는 `_SIZE_CONFIG_AC` tier) 에서 바꾼다. `hparam_overrides` 는 모델별 delta 전용.
- Stage 2 world-model variant 는 `--stage1-epoch N` 의 로컬 merged dir 이 **반드시 선행**돼야 한다 (stage1_train → stage1_merge).
- merge/eval 은 Python `pyyaml` 을 전제한다. `HF_TOKEN` 은 HF push 또는 HF fallback pull 시에만 필요하다.
- shell automation 은 **bash 4+** 요구.
- **transformers 버전**은 `pyproject.toml` 의 `llamafactory` extra 에서 `>=4.57.1,<4.58` 로 고정 — 값과 그 위 주석을 함께 바꾼다. **서브프로젝트 `LlamaFactory/pyproject.toml` 은 수정하지 않는다.**
  - trl 0.24 / transformers 4.56+ API: `SFTConfig(max_length=…)`, `SFTTrainer(processing_class=…)`. 구버전 키 (`max_seq_length`, `tokenizer=`, `overwrite_output_dir`) 는 `TypeError`.
  - `gradient_checkpointing` 은 모델 로드 단계에서만 적용 — `SFTConfig` 에 전달하지 않는다 (이중 적용 방지).

> ⚠️ **함정 19 — EXP05 는 로컬에서 학습할 수 없다. 그리고 80GB 라도 offload 를 켜면 4 일이 걸린다 (둘 다 실측).**
> - **로컬 2×RTX5090**: EXP05 3B Full FT 는 **CUDA OOM** (step 3 에서 8.92 GiB 할당 실패) + **157~168 s/it → 총 97~104 시간 (약 4 일)**. 원인은 `cutoff_len 24576` + `max_pixels 1,605,632` 로 시퀀스가 극단적으로 길어진 것과 RTX5090 에 강제되는 ZeRO-3 CPU offload 다.
> - **A100×2 + offload (2026-07-14 실측)**: OOM 은 안 나지만 **165 s/step → 2094 step 에 약 4 일**로 RTX5090 과 사실상 같다. GPU 메모리는 80GB 중 23~26 GB 만 쓰고 전력이 135 W 대 378 W 로 벌어진다 — 병목이 계산이 아니라 CPU↔GPU 전송이라는 뜻. **그래서 80GB × 3-4B 는 offload 를 끈다** (함정 7). 끄면 GPU 메모리 ~40~57 GB, `pdbs=2`/`ga=16`.
> - **본 학습은 A100/H100 에서 수행한다.** 원격 제출 스펙 (`configs/remote/run.template.yaml` + `scripts/remote_launch.sh`) 은 저장소에 있으나 **UNVALIDATED (실행 이력 0)** 이다 — 위 A100 실측은 원격 제출이 아니라 A100 머신에서 `scripts/stage1_train.sh` 를 직접 호출한 결과다.
> - 참고: `gpu_policy` 는 7-9B full FT 조합에 host RAM 경고를 붙인다 (steady ~154GB/노드, 체크포인트 저장 시 ~175GB+).

> ⚠️ **함정 20 — YAML 이 있다고 돌릴 수 있는 게 아니다.**
> - **왜 재구성본인가**: EXP03/EXP04 의 as-trained YAML 은 **소실됐다.** 지금 `configs/train/IWM-AC_EXP0{3,4}/` 에 있는 것은 현행 레지스트리 + GPU 정책으로 `gen_configs` 가 **다시 렌더한 재구성본**이며 (`# [reconstructed 2026-07-13]` 헤더), 실제 학습 설정과 다를 수 있다. `--check` 는 "커밋본이 **현행 생성기**와 일치하는가" 만 보증하지 "**as-trained 와 일치하는가**" 는 보증하지 못한다 — 그래서 헤더를 남겼다. 확인: `grep -rl reconstructed configs/train`
> - **EXP04 는 YAML 이 있어도 돌지 않는다** (등록 키 없음 — 함정 14, §2 경고 블록).
>
> 재구성 이후 무엇이 실제 학습으로 확인됐는가 (**상태**) 는 [`../docs/ROADMAP.md`](../docs/ROADMAP.md) 의 재현성 경고를 본다.

현재 상태 (무엇이 됐고 무엇이 막혔나) 는 [`../docs/ROADMAP.md`](../docs/ROADMAP.md) 를 본다.
