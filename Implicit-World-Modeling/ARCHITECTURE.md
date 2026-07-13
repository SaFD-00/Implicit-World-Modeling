# Implicit-World-Modeling Architecture

`Implicit-World-Modeling` 은 모바일 GUI World Modeling 이 Action Prediction 성능에 주는 영향을 검증하는 2-stage fine-tuning 파이프라인이다. **4 개 Vision-Language 모델** (`Qwen/Qwen3-VL-8B-Instruct`·`Qwen/Qwen2.5-VL-7B-Instruct` 7-9B tier, `Qwen/Qwen3-VL-4B-Instruct`·`Qwen/Qwen2.5-VL-3B-Instruct` 3-4B tier) 을 지원하며, conda env (`implicit-world-modeling`) + 노트북 [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) 가 오케스트레이션을 담당하고, [`scripts/`](./scripts) 가 반복 실행용 자동화 레이어다. 학습/export 는 conda env 에 `pip install -e ./LlamaFactory` 로 editable 설치된 LlamaFactory 가 수행한다. 모든 stage 의 흐름은 **`train → merge → eval`** 로 통일되며, merge 는 `--no-hf-upload` 로 local export 만 수행할 수 있다. eval 은 **local merged dir (`outputs/.../merged/.../epoch-{E}/`) 우선 + HF Hub merged repo fallback** (`_common.sh::resolve_eval_model_path`) — local merge 한 머신에서도 같은 머신 안에서 바로 eval 까지 이어 돌 수 있다.

---

## 0. Runtime Stack

```
conda env       notebook            엔진                              모델
─────────────   ─────────────────   ──────────────────────────────   ─────────────────────────────────
implicit-world-modeling  implicit-world-modeling.ipynb   llamafactory-cli train/export    Qwen3-VL-8B-Instruct
                                                          YAML: LlamaFactory/examples/      Qwen2.5-VL-7B-Instruct
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

- [`implicit-world-modeling.ipynb`](./implicit-world-modeling.ipynb) — conda `implicit-world-modeling`, 4 개 모델 (Qwen3-VL / Qwen2.5-VL 2 family)
  - 환경 설치, `_MODEL_CONFIG` (4) + `MODEL_FAMILY_CONFIG` + `_DATASET_CONFIG` (8 키 = 6 학습 DS; AC_EXP01 은 ratio 3 종으로 전개) + `_SIZE_CONFIG_AC` (2 tier) 정의
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
| 3 | 18–84 | Stage 1 SFT — `qwen3-vl-8b` Full FT walkthrough (AC_EXP01 · AC_EXP02). 다른 모델/모드는 `--model` / `--stage1-mode` 인자 교체 |
| 4 | 85–151 | Stage 1 merge (모든 epoch local merge + 선택적 HF Hub push; `--no-hf-upload` 지원) |
| 5 | 152–159 | Stage 1 평가 — local merged 우선 + HF Hub fallback sweep, Hungarian metric |
| 6 | 160–186 | Stage 2 SFT (AC_EXP01·AC_EXP02 는 2 모델, AC_EXP03 은 Qwen3-VL 계열 전용 — 좌표계 mismatch) |
| 7 | 187–213 | Stage 2 merge (variant × 모든 epoch local merge + 선택적 HF push; `--no-hf-upload` 지원) |
| 8 | 214–218 | Stage 2 평가 — ID + OOD 동시 sweep, `action_metrics.json` 3 섹션 |

> **Stage 2 머지 YAML 은 사전 생성하지 않는다** — `scripts/stage{1,2}_merge.sh` 가 runtime 에 임시 YAML 을 만든다. `BEST_CHECKPOINT` 개념은 제거됐고 epoch 번호는 `trainer_state.json.epoch` 으로 결정된다.

---

## 2. 모델 설정

### 모델 레지스트리

`implicit-world-modeling.ipynb` Cell 5 의 `_MODEL_CONFIG` (4 모델) 와 `scripts/_common.sh` 의 `MODEL_ID` / `MODEL_TEMPLATE` / `ALL_MODELS` 가 동기화되어야 한다.

| short_name | model_id | template | size |
|------------|----------|----------|------|
| qwen3-vl-8b | Qwen/Qwen3-VL-8B-Instruct | qwen3_vl_nothink | 7-9B |
| qwen3-vl-4b | Qwen/Qwen3-VL-4B-Instruct | qwen3_vl_nothink | 3-4B |
| qwen2.5-vl-7b | Qwen/Qwen2.5-VL-7B-Instruct | qwen2_vl | 7-9B |
| qwen2.5-vl-3b | Qwen/Qwen2.5-VL-3B-Instruct | qwen2_vl | 3-4B |

> Qwen3-VL 의 `qwen3_vl_nothink` template 은 `vllm_infer.py` 호출 시 `_common.sh::build_infer_cmd` 가 `--enable_thinking False` 를 자동 주입해 thinking 트리거를 끈다. Qwen2.5-VL 의 `qwen2_vl` template 은 thinking 트리거가 없어 해당 플래그가 주입되지 않는다 (template 분기로 자동 처리).

### 모델 family 별 native 좌표 규약 (GUI grounding)

두 모델은 grounding/GUI 좌표를 서로 다른 규약으로 pretrain 되어 있다. Qwen 계열은 세대마다 규약이 반전됐다 (Qwen2-VL 정규화 → Qwen2.5-VL 절대 픽셀 → Qwen3-VL 다시 정규화).

| model family | native 좌표 규약 | 표현 | 해상도 의존성 |
|--------------|-----------------|------|---------------|
| Qwen3-VL | **0–1000 정규화** | bbox `[x1,y1,x2,y2]` + point `[x,y]` (정규화) | resolution-independent (rescale 시 분모 999, backward-compat) |
| Qwen2.5-VL | **절대 픽셀** | bbox `[x1,y1,x2,y2]` (입력 이미지 실제 해상도 픽셀) | 이미지 해상도에 종속 |
| (참고) Qwen2-VL | 0–1000 정규화 | — | — |

**우리 좌표 실험군 (EXP03/EXP04) 데이터는 0–1000 정규화로 통일** — XML root div 가 `[0,0][1000,1000]` (`point="[500,500]"`), 전 좌표 min 0 / max 1000, action 의 `point` 는 노드 `point` 속성 verbatim 복사 (`[POINT]` / `[NO HALLUCINATION]` 규칙). 원천은 `data/AndroidControl/implicit-world-modeling_stage1_{action,state}_xy.jsonl` (EXP03) / `*_xy_prompt-enhanced.jsonl` (EXP04) — 둘 다 정규화. **픽셀 좌표가 아님.**

> **EXP03/EXP04 는 Qwen3-VL 계열 전용 실험군 (좌표계 정합성)**: EXP03/EXP04 의 0–1000 정규화는 **Qwen3-VL native 와 일치** 하지만 **Qwen2.5-VL native (절대 픽셀) 와는 어긋난다**. 이 mismatch 때문에 **EXP03/EXP04 는 Qwen3-VL 계열 (`qwen3-vl-4b`/`qwen3-vl-8b`) 만 학습·평가** 하고 Qwen2.5-VL 은 사용하지 않는다 (지금까지 실제로 학습한 것은 `qwen3-vl-8b` 뿐이며, `qwen3-vl-4b` 는 자격만 있고 학습 이력은 없다) (Qwen2.5-VL 은 정규화 좌표를 native 로 쓰지 않아 자기 pretrain 규약과 충돌). 좌표 표현(index→point) 효과는 동일 Qwen3-VL 계열의 EXP01 ratio73 대조군과 비교한다. 출처 — [Qwen3-VL Spatial Understanding & 2D Grounding (DeepWiki)](https://deepwiki.com/QwenLM/Qwen3-VL/5.2-spatial-understanding-and-2d-grounding), [Qwen2.5-VL Visual Grounding issue #866 (QwenLM)](https://github.com/QwenLM/Qwen2.5-VL/issues/866).

> **EXP05 는 대칭으로 Qwen2.5-VL 전용 실험군 (절대 픽셀)**: EXP05 (AndroidWorld 해상도 정렬) 데이터는 **절대 픽셀 840×1876** — base 1080×2400 을 budget 1,605,632 / factor 28 로 smart_resize 한 이미지의 실제 픽셀 좌표 (min 3,136). 이는 **Qwen2.5-VL native 와 일치** 하고 Qwen3-VL native (0–1000 정규화) 와는 어긋난다. 따라서 **EXP05 는 `qwen2.5-vl-3b`/`qwen2.5-vl-7b` 만 학습·평가** 하고 **Qwen3-VL 계열 (`qwen3-vl-4b`/`qwen3-vl-8b`) 은 제외** 한다 (factor 32 라 1,605,632 를 줘도 832×1888 로 정렬돼 840×1876 과 어긋나고, 0–1000 정규화 native 와도 이중 mismatch — EXP03/04 의 정확한 대칭). 이 배제는 코드 가드가 아니라 **문서 규약** 이다 — `--model` 에 Qwen3-VL 계열을 넣어도 에러 없이 학습이 돌아가고 grounding 만 조용히 깨진다. 원천은 `data/AndroidControl/implicit-world-modeling_stage1_{action,state}_xy_pixel-aligned.jsonl` (Google Drive **'0711_버젼'** 제공, 빌드 정본 `scripts/build_exp05_data.py` = mirror + diff-loss 가중치). 좌표 실측: **XML 의 `bounds`/`point` 는 정확히 x_max 840 / y_max 1876** 이다. 다만 **액션 라벨(`coordinate`) 필드에는 범위를 벗어나는 키가 11 개** 있고 (0710·0711 동일 — 기존 원천 버그; 값이 `[1682, 975]` 로 반복되며 1682 ≈ 840×2 라 스케일링 의심), 그중 **10 행이 EXP05 산출물에 실려 있다** (train 7 + `test_ood_{action,state,state_without_open_app}` 각 1 → OOD 평가셋 오염). 상세는 [`docs/EXP05_DIFF_LOSS_PLAN.md`](../docs/EXP05_DIFF_LOSS_PLAN.md) §6. "3B/8B 모두" 요청의 8B 는 `qwen2.5-vl-7b` 로 해석 (Qwen2.5-VL 에 8B 부재).

### 모델 family 별 image budget

**이 절이 image budget 의 단일 진실원이다** (AGENTS / README 는 여기를 참조한다). 노트북 Cell 5 의 `MODEL_FAMILY_CONFIG` (factor / max_tokens / min_tokens) 와 `_DATASET_CONFIG[ds]["image_overrides"]` 의 token 단위 override 로 관리된다. token 예산은 **학습 데이터셋** 으로 결정 — 학습된 모델은 평가 데이터셋과 무관하게 학습 시 budget 으로 추론한다 (학습-추론 mismatch 방지). `max_pixels = max_tokens × factor²`, `min_pixels = min_tokens × factor²` 이며 모든 학습 DS 가 family default `max_tokens=2048` / `min_tokens=4` 를 쓴다 (dataset 별 `image_overrides` 없음).

| family | patch | merge | factor | max_tokens | **max_pixels** | min_tokens | min_pixels |
|---|---|---|---|---|---|---|---|
| Qwen3-VL (8B) | 16 | 2 | **32** | 2,048 | **2,097,152** (= 2048 × 32²) | 4 | 4,096 |
| Qwen2.5-VL (3B/7B) | 14 | 2 | **28** | 2,048 | **1,605,632** (= 2048 × 28²) | 4 | 3,136 |

**budget 이 이미지를 실제로 어떤 크기로 만드는가** — base 스크린샷 **1080×2400 (W×H)** 를 각 설정으로 `smart_resize` 한 결과:

| 설정 | factor | max_pixels | → 리사이즈 (W×H) | visual tokens |
|---|---|---|---|---|
| **Qwen2.5-VL (우리 값)** | 28 | **1,605,632** | **840 × 1876** | **2,010** |
| **Qwen3-VL (우리 값)** | 32 | **2,097,152** | **960 × 2144** | **2,010** |
| Qwen3-VL 에 EXP05 budget 을 억지로 준 경우 | 32 | 1,605,632 | 832 × 1888 | 1,534 |
| (참고) Qwen 공식 기본값 — Qwen2.5-VL | 28 | 12,845,056 | 1092 × 2408 | 3,354 |

> **함정 — "토큰 수가 같으니 괜찮겠지"**: 두 family 모두 `max_tokens=2048` 기준이라 **visual token 수는 2,010 개로 동일** 하다. 다른 것은 **이미지의 실제 픽셀 크기** 다 (840×1876 vs 960×2144). 그래서 EXP05 데이터 (840×1876 절대 픽셀 좌표) 를 Qwen3-VL 로 학습하면 모델이 보는 이미지는 960×2144 인데 좌표는 840×1876 이라 **grounding 이 조용히 깨진다** — 토큰 수 일치는 아무것도 보장하지 않는다. 이것이 EXP05 를 Qwen2.5-VL 전용으로 묶는 두 번째 이유다 (첫 번째는 위 좌표 규약 mismatch).

> **1,605,632 는 "기본값" 이 아니라 의도적 override**: Qwen 공식 기본 `max_pixels` 는 **12,845,056** 이다. 1080×2400 을 공식 기본값으로 돌리면 1092×2408 / 3,354 tokens 가 나온다. 우리는 이를 명시적으로 `max_tokens=2048` (Qwen2.5-VL → 1,605,632) 로 낮춰 잡았고, **데이터 생성 시 `--image-budget` 과 학습 프로세서 `image_max_pixels` 가 반드시 같은 값이어야 한다**. EXP05 는 데이터 생성 budget (`--image-budget 1605632`) 이 Qwen2.5-VL family default 와 정확히 같아 `image_overrides` 를 두지 않아도 자동으로 일치한다 (Slack 요구 "두 값 일치" 충족).

YAML 의 `image_max_pixels` / `image_min_pixels` 는 CONFIGS 빌더가 family default 에 dataset override 를 token-aware 로 덮어써 자동 주입한다. 평가측 `scripts/_common.sh::build_infer_cmd` 는 `TRAIN_DATASET` 글로벌 (parse_args 에서 set) 로 학습 DS 를 식별해 동일 budget 을 적용한다.

> vLLM `gpu_memory_utilization` 은 `build_infer_cmd` 내부에서 기본 `0.80`, 환경변수 `VLLM_GPU_MEM_UTIL` 로 호출 단위 override. `build_infer_cmd` 는 `stage{1,2}_eval.sh` 양쪽에서 공통으로 호출되므로 stage1/2 모두 동일하게 적용된다 (예: 동일 GPU 에서 학습 병행 / OOM 마진 확보 시 `VLLM_GPU_MEM_UTIL=0.6 bash ./scripts/stage2_eval.sh ...`). 미설정 시 0.80 그대로.

`cutoff_len` 은 **AC_EXP01 / AC_EXP02 는 Stage 1 / Stage 2 모두 10000**, **AC_EXP03 / AC_EXP04 / AC_EXP05 (좌표 표현) 만 24576** 이다 — AC_EXP03 은 Stage 1 / Stage 2 / 평가 모두, AC_EXP04 는 Stage 2 보류라 Stage 1 / 평가에 적용 (10000 은 2026-05-13 16384 → 10000 하향). state+action ratio-mix 로 frame 이 다수 포함돼 Qwen3-VL multimodal RoPE position 길이가 8192 를 초과 (관측: 8521) 하는 샘플이 있어 학습이 첫 step 에서 shape mismatch 로 실패한다 — 10000 은 실측 분포 기준 multi-frame 안전 마진과 메모리/throughput 사이 tradeoff 를 잡은 운영 기준이다. 노트북 Cell 8 의 Stage 1/2 inline YAML 과 `LlamaFactory/examples/custom/IWM-AC_EXP01_*` yaml 모두 10000 으로 통일한다. (`scripts/filter_long_samples.py --threshold` 의 default 도 10000 으로 동기 — 사전 필터와 학습 cutoff 를 같은 기준으로 통일.)

**AC_EXP03 cutoff_len = 24576 (좌표 표현 무손실·EXP01 공정 비교)**: AC_EXP03 는 같은 전이를 `index="N"` 대신 `point=[x,y]` 좌표로 적어 시퀀스가 ~2~2.5x (최대 20k+) 길다 — `cutoff_len=10000` 에서는 ~10% 가 잘리고 ~0.3% 가 위 `get_rope_index` shape mismatch 로 크래시했다 (EXP01 ratio73 은 max 9059, 잘림 0%). EXP03 멤버십은 EXP01 ratio73 (index 기준 ≤10000 으로 이미 필터된 집합) 의 좌표 미러라 팽창 상한이 묶여 있어, EXP01 원본의 39K long-tail 과 달리 **필터 없이 cutoff 만 24576 으로 올리면 잘림/크래시 0·데이터 손실 0** 이 성립한다 (`python scripts/filter_long_samples.py --dataset AC_EXP03 --threshold 24576 --report-only` 측정: stage1_train max=20272 / stage2_train max=20697 / over-threshold=0 — EXP03 는 필터링하지 않고 측정만 한다; 24576 은 ~3.9k 여유). 평가(`scripts/_common.sh::build_infer_cmd`)도 `IWM-AC_EXP03*` 데이터셋이면 `--cutoff_len 24576` 으로 분기해 입력 truncation 0 을 맞춘다 (vLLM `max_model_len = cutoff + max_new_tokens` 증가 → KV cache 메모리↑·throughput↓, 필요 시 `VLLM_GPU_MEM_UTIL`). 긴 시퀀스로 학습 메모리가 늘어 **EXP03 만 `per_device_train_batch_size` 를 절반** 으로 낮추고 `gradient_accumulation_steps` 로 보정해 `GLOBAL_BATCH_SIZE=64` 를 유지한다 (EXP01 과 global batch 동일 → 공정). **AC_EXP04 는 AC_EXP03 좌표 pool 의 프롬프트 업그레이드본(같은 (episode,step) 멤버십·좌표 표현)** 이라 `cutoff_len 24576` 을 그대로 공유한다 — 좌표 시퀀스가 ~2.5x 길어도 무손실이며, EXP04 멤버십 ⊆ EXP03 이라 잘림/크래시 0 도 승계된다. `IWM-AC_EXP04*` 평가도 `build_infer_cmd` 가 24576 으로 분기한다 (Stage 2 는 보류라 stage1·평가에만 적용). **AC_EXP05 도 동일** — 절대 픽셀 좌표(point) 표현이라 시퀀스가 길어 `cutoff_len 24576` + half-batch 를 공유하고, `IWM-AC_EXP05*` 평가도 24576 으로 분기한다 (Stage 2 보류).

### 하이퍼파라미터 — 3 단 머지 구조

CONFIGS 빌더가 다음 순서로 `dict.update()` 한다:

1. `_DATASET_CONFIG[ds].stage{1,2}` — 데이터셋 공통 baseline (AC_EXP01 / AC_EXP02 / AC_EXP03 / AC_EXP04 / AC_EXP05 / MC). AC_EXP04 · AC_EXP05 는 stage1 만 사용 (Stage 2 보류 — `_STAGE1_ONLY`).
2. `_SIZE_CONFIG_AC[size].stage{1, 1_lora, 2}` — **AC_EXP01 / AC_EXP02 / AC_EXP03 / AC_EXP04 / AC_EXP05 공유** 모델 크기 공유값 (7-9B / 3-4B 2 단 tier). **현재 두 tier 모두 세 키가 빈 dict** 이라 baseline 을 그대로 쓴다 (EXP01/EXP02 실측 어댑터와 동일조건 보존). MC 에는 적용되지 않는다.
3. `_MODEL_CONFIG[model].hparam_overrides` — 모델별 delta.

각 모델은 `_MODEL_CONFIG[model]["size"]` 필드로 tier 를 지정한다 — `"7-9B"` (`qwen3-vl-8b`, `qwen2.5-vl-7b`) / `"3-4B"` (`qwen3-vl-4b`, `qwen2.5-vl-3b`). MB 는 평가 전용이라 학습 하이퍼파라미터 해석에서 제외.

#### `_SIZE_CONFIG_AC` 값 (7-9B / 3-4B)

> 등록된 두 tier (`7-9B`, `3-4B`) 는 `stage1` / `stage1_lora` / `stage2` 가 **모두 빈 dict** 이라, tier 와 무관하게 dataset baseline 이 그대로 적용된다. 아래 표는 그 baseline 을 7-9B 기준으로 적은 것이며, 3-4B 모델도 (delta 가 없으므로) 동일한 값을 받는다.

**Stage 1 (full FT)** — dataset baseline 유지:

| 구간 | lr | warmup_ratio | max_grad_norm |
|---|---|---|---|
| 7-9B | (baseline 유지: 1.0e-5 / 0.03 / 1.0) | | |

**Stage 1 LoRA** — **baseline(8/16 @1.0e-5) 유지** (tier 비움 — EXP01/EXP02 실측 어댑터와 동일조건 보존):

| 구간 | lr | LoRA r / α | dropout |
|---|---|---|---|
| 7-9B | 1.0e-5 | 8 / 16 | 0.05 |

LoRA 모드의 `deepspeed` 필드는 `GPU_TYPE` 에 따라 분기된다 (Stage 1: Cell 9, Stage 2: Cell 13 — 동일 정책):

| GPU_TYPE | LoRA deepspeed config | 근거 |
|---|---|---|
| `RTX5090` (32GB) | `examples/deepspeed/ds_z3_offload_config.json` | 단일 GPU + 32GB VRAM 에서 7-9B OOM 회피 (ZeRO-3 + CPU offload) |
| `A100` / `H100` (80GB) | `examples/deepspeed/ds_z3_config.json` (= `stage1_deepspeed` 기본값) | 80GB 면 offload 불필요, 통신 오버헤드만 추가됨 |

Full FT (Stage 1 / Stage 2 양쪽) 는 분기 없이 모델별 `stage1_deepspeed` (기본 `ds_z3_config.json`) 그대로 사용. 분기 로직은 `lora` 모드에만 적용.

**Stage 2 (LoRA)** — baseline 그대로 (tier 비움):

| 구간 | lr | LoRA r / α | dropout | warmup_ratio |
|---|---|---|---|---|
| 7-9B | 5.0e-5 | 32 / 64 | 0.1 | 0.03 |

설계 근거: 과거 `outputs/AC/eval/qwen{2.5-vl-7b,3-vl-8b}/stage2_eval` 실측에서 dropout 0.10 이 저빈도 action type 을 불안정하게 만든다는 관측이 있으나, **EXP01 기존 stage2 어댑터(d0.1 / 5e-5)와의 동일조건 비교를 우선**해 정본을 baseline(0.1 / 5e-5) 으로 둔다 — diff loss 순효과 측정을 위해 EXP02 stage2 만 재학습하면 EXP01 전체(학습/eval)를 보존할 수 있다. `_DATASET_CONFIG` baseline 의 stage2 dropout 도 0.1.

#### 계열 delta (`_MODEL_CONFIG[model].hparam_overrides`)

| 계열 | stage1 / stage2 에 추가 |
|---|---|
| Qwen 계열 (4 모델) | (empty — 전부 tier 값 그대로) |

#### `per_device_train_batch_size` (size × GPU)

`_PER_DEVICE_BS_BY_SIZE[size][GPU_TYPE]` (Cell 5).

| 모델 size | RTX5090 (32GB) | A100 (80GB) | H100 (80GB) |
|-----------|----------------|-------------|-------------|
| 7-9B      | 1              | 2           | 2           |

#### `gradient_accumulation_steps` 불변식

```
global_batch = per_device_train_batch_size * gradient_accumulation_steps * NPROC_PER_NODE
            == GLOBAL_BATCH_SIZE  (기본 64)

gradient_accumulation_steps = GLOBAL_BATCH_SIZE / (per_device * NPROC_PER_NODE)
```

`NPROC_PER_NODE ∈ {1, 2, 4, 8}` 와 `GPU_TYPE ∈ {RTX5090, A100, H100}` 만 허용 — 다른 값은 `ValueError`. Cell 5 의 `_derive_grad_accum()` 이 역계산해 CONFIGS 의 `stage{1,2}.gradient_accumulation_steps` 에 주입한다. 위 표 값이 모든 (size, GPU, NPROC) 조합에서 64 로 나누어떨어지므로 silent rounding 은 발생하지 않는다.

> **AC_EXP03 / AC_EXP04 / AC_EXP05 override**: 좌표 표현으로 시퀀스가 ~2.5x (cutoff_len 24576) 길어 활성화 메모리가 커지므로, EXP03 / EXP04 / EXP05 만 Cell 5 에서 `per_device_train_batch_size` 를 절반 (`max(1, per_device // 2)`; H100/A100 2→1, RTX5090 1 유지) 으로 낮추고 `_derive_grad_accum` 으로 재계산 (H100/A100 grad_accum 16→32) 해 `GLOBAL_BATCH_SIZE=64` 를 그대로 유지한다 — EXP01 과 global batch 가 같아 비교가 공정하다. RTX5090 은 per_device 최소(1) 라 추가 축소 불가 → ZeRO-3 offload + gradient_checkpointing 에 의존 (OOM 시 NPROC↑ 또는 수동 조정). **AC_EXP04 stage1 YAML 은 EXP03 stage1 YAML 을 복사** 한다 (노트북 Cell 10 미사용 — EXP03 single-H100 hand-fix 보존: full→`ds_z3_offload`, lora→`ds_z0`). GPU 0,1 (NPROC=2) 2-GPU 로 `per_device_train_batch_size 1` + grad_accum 보정 (lora 32 → global 64, full 16 → global 32) 해 global batch 를 유지하며, 실제 학습 대상은 **`qwen3-vl-8b` 단일 모델** stage1 LoRA (EXP03/EXP04 는 Qwen3-VL 전용 — 좌표계 mismatch 로 `qwen2.5-vl-7b` 제외), tmux 스케줄은 `scripts/tmux_exp04_stage1.sh`.

---

## 3. 데이터와 설정 계약

### 데이터 디렉토리

```
data/
├── AndroidControl/                   # 원본 source 자산 (학습/평가 entry 아님 — AC_EXP01 / AC_EXP02 / AC_EXP03 / AC_EXP04 가 read)
│   ├── implicit-world-modeling_stage1.jsonl                                # 원천 (전체)
│   ├── implicit-world-modeling_stage1_state.jsonl                     # 원천: state-transition task
│   ├── implicit-world-modeling_stage1_action.jsonl                    # 원천: action-prediction task
│   ├── implicit-world-modeling_stage1_{state,action}_filtered.jsonl   # filter_long_samples.py --dataset AC_EXP01 산출 (mm-expanded length > cutoff_len 제거)
│   ├── implicit-world-modeling_stage2.jsonl                                # 원천: Stage 2 action-prediction
│   ├── implicit-world-modeling_stage2_filtered.jsonl                       # filter_long_samples.py 산출
│   ├── implicit-world-modeling_stage1_action_xy.jsonl                      # AC_EXP03 원천: action-pred 좌표(point) 표현
│   ├── implicit-world-modeling_stage1_state_xy.jsonl                       # AC_EXP03 원천: state-pred 좌표(point) 표현
│   ├── implicit-world-modeling_stage2_xy.jsonl                             # AC_EXP03 원천: Stage 2 좌표(point) 표현
│   ├── implicit-world-modeling_stage1_action_xy_prompt-enhanced.jsonl      # AC_EXP04 원천: action-pred 좌표(point) + 프롬프트 업그레이드 (swipe / html-style XML / [SWIPE])
│   ├── implicit-world-modeling_stage1_state_xy_prompt-enhanced.jsonl       # AC_EXP04 원천: state-pred 좌표(point) + 프롬프트 업그레이드 (Stage 2 원천 없음 — 보류)
│   ├── implicit-world-modeling_stage1_action_xy_pixel-aligned.jsonl        # AC_EXP05 원천: action-pred 절대 픽셀 좌표(840×1876, Qwen2.5-VL) — Drive '0711_버젼' raw stage1_0711_action_pred.jsonl (74,350 행)
│   ├── implicit-world-modeling_stage1_state_xy_pixel-aligned.jsonl         # AC_EXP05 원천: state-pred 절대 픽셀 좌표 (73,950 행; Stage 2 원천 없음 — 보류)
│   ├── episodes_meta.jsonl                 # primary_app = 전경 앱 package_name
│   └── images/                              # AC_EXP01 / AC_EXP02 / AC_EXP03 / AC_EXP04 / AC_EXP05 가 JSONL `images` 필드로 공유 참조
├── AndroidControl_EXP01/                 # Stage 1 ratio mix 학습 + Stage 2 ratio sweep (split_data.py --dataset AC_EXP01 산출)
│   ├── implicit-world-modeling_stage1_train_{3_7,5_5,7_3}.jsonl            # ratio 별 (state:action) train
│   ├── implicit-world-modeling_stage1_test_{id,ood}_state.jsonl       # state task, app-level partition
│   ├── implicit-world-modeling_stage1_test_{id,ood}_action.jsonl      # action task, app-level partition
│   ├── implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl       # split_data.py 가 원본 _filtered 풀에서 산출, 15K / 3K / 3K (action_type stratified, Stage 1 action_pred app partition 공유)
│   # NOTE: images/ + episodes_meta.jsonl 은 원본 AndroidControl/ 의 것을 그대로 참조 (JSONL `images` 가 "AndroidControl/images/..." prefix).
├── AndroidControl_EXP02/                 # AC_EXP01 ratio73 + Stage 1 state-pred diff loss 실험군 (대조군 = AC_EXP01 ratio73)
│   ├── implicit-world-modeling_stage1_train.jsonl                          # AC_EXP01 ratio73 train + token_weights (scripts/diff_loss/preprocess_dataset.py)
│   ├── implicit-world-modeling_stage1_test_{id,ood}_{state,action}.jsonl  # AC_EXP01 에서 복사 (동일 평가셋 — 공정 비교)
│   ├── implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl       # AC_EXP01 에서 복사 (Stage 2 는 diff loss 미적용)
│   # NOTE: images/ + episodes_meta.jsonl 은 원본 AndroidControl/ 의 것을 공유 참조.
├── AndroidControl_EXP03/                 # AC_EXP01 ratio73 멤버십 좌표(point) 표현 미러 (scripts/mirror_experiment.py --experiment exp03 산출)
│   ├── build_exp05_data.py        # AC_EXP05 학습 데이터 정본 빌더 (mirror → diff-loss token_weights → 원자 교체 + sidecar)
│   ├── implicit-world-modeling_stage1_train.jsonl                          # AC_EXP01 train_7_3 미러 (좌표 표현, ~49,596)
│   ├── implicit-world-modeling_stage1_test_{id,ood}_{state,action}.jsonl  # AC_EXP01 test 멤버십 미러 (좌표 표현)
│   ├── implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl       # AC_EXP01 Stage 2 멤버십 미러 (좌표 표현)
│   # NOTE: 본문은 bounds/point 좌표, 이미지 경로는 EXP01 채택 ("AndroidControl/images/..."). 원천에 없는 키(~0.8–1.7%) 제외.
├── AndroidControl_EXP04/                 # EXP03 멤버십 + stage1 프롬프트 업그레이드 미러 (scripts/mirror_experiment.py --experiment exp04 산출 — stage1-only)
│   ├── implicit-world-modeling_stage1_train.jsonl                          # EXP03 train 미러 (좌표 + 프롬프트 업그레이드, 49,276)
│   ├── implicit-world-modeling_stage1_test_{id,ood}_{state,action}.jsonl  # EXP03 dual-task test 미러 (4 파일, 좌표 + 프롬프트)
│   ├── implicit-world-modeling_stage1_test_{id,ood}_state_without_open_app.jsonl  # state without_open_app sibling (2 파일) — train 포함 총 7 파일
│   # NOTE: 본문 = 좌표(EXP03 와 동일 멤버십·좌표) + 프롬프트 업그레이드 (action scroll→swipe, "html-style XML" 명시, [SWIPE] 규칙), 이미지 경로는 EXP01 채택. Stage 2 보류 (`_STAGE1_ONLY`). EXP04 pool ⊆ EXP03 pool → drop train 320 / 전체 stage1 450 (0.67%).
├── AndroidControl_EXP05/                 # AC_EXP01 ratio73 멤버십 + 절대 픽셀 좌표 미러 (scripts/mirror_experiment.py --experiment exp05 산출 — stage1-only, Qwen2.5-VL 전용). 좌표계·budget 은 §2 참조
│   ├── implicit-world-modeling_stage1_train.jsonl                          # EXP01 train 미러 (44,670 — 입력 50,000 / drop 5,330) + diff-loss token_weights 인라인
│   ├── implicit-world-modeling_stage1_test_{id,ood}_{state,action}.jsonl  # dual-task test 미러 (4 파일)
│   ├── implicit-world-modeling_stage1_test_{id,ood}_state_without_open_app.jsonl  # state without_open_app sibling (2 파일) — train 포함 총 7 파일, 60,717 행
│   # NOTE: 이미지 경로는 EXP01 채택 ("AndroidControl/images/..."). Stage 2 보류 (`_STAGE1_ONLY`). 소스는 Drive '0711_버젼'.
│   # 빌드 정본: scripts/build_exp05_data.py (mirror → diff-loss 가중치 → 원자 교체). sidecar: <train>.meta.json
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

- **App partition (AC_EXP01 / AC_EXP02 공유 — 원본 `data/AndroidControl/episodes_meta.jsonl`)**: `compute_app_partition` 이 Stage 2 행 수를 budget 으로 (id_apps, ood_apps) 를 한 번 계산하고, Stage 1 은 같은 partition 으로 entries 를 라우팅. Stage 2 OOD 앱이 Stage 1 train 에도 포함되지 않아 world-modeling 학습이 OOD 앱을 한 번도 보지 않는다. (AC_EXP03 / AC_EXP04 는 EXP01 산출 멤버십을 그대로 미러하므로 동일 partition 을 승계 — 별도 partition 계산 없음.)
- **Stage 1 (MC)**: 메타 없음 → 자동 random split (`--stage1-ratio`, 기본 0.95). `_STAGE1_ONLY` guard 로 Stage 2 자동 skip.
- **AC_EXP01 (Stage 1 ratio mix + Stage 2 ratio sweep)**: split_data.py 의 source 는 항상 원본 `data/AndroidControl/` 이고, 산출물은 `data/AndroidControl_EXP01/` 에 쓰여진다. 선행으로 `scripts/filter_long_samples.py --dataset AC_EXP01` 가 mm-expanded length > `cutoff_len` 인 row 를 제거해 **원본 폴더 안에** `implicit-world-modeling_stage1_{state,action}_filtered.jsonl` + `implicit-world-modeling_stage2_filtered.jsonl` (3 파일) 을 만든다 (Qwen3-VL `get_rope_index` broadcast 회피용). `run_exp01_split` 은 항상 Stage 1/Stage 2 모두 `_filtered` 만 입력으로 사용 — Stage 2 source 누락 시 hard-fail. 그 위에서 `state_pred` (random) + `action_pred` (action-type stratified) 두 풀을 ID/OOD 앱 partition 으로 라우팅 후 ratio (state:action ∈ {7:3, 3:7, 5:5}, default `7:3,3:7,5:5`) 로 혼합한 Stage 1 train 3 종 + (id, ood) × (state, action) 4 test 를 산출. 같은 (id_apps, ood_apps) 를 재사용해 Stage 2 split (`implicit-world-modeling_stage2_{train,test_id,test_ood}.jsonl`, 기본 15K / 3K / 3K, action_type stratified) 까지 함께 산출 — Stage 1 ↔ Stage 2 OOD app 집합 일치. **Stage 2 파이프라인은 ratio sweep 으로 활성** (`_STAGE1_ONLY = {"MonkeyCollection", "AndroidControl_EXP04"}` 만 Stage 1 전용) — stage2 데이터는 ratio 무관 (3 ratio 공유) 이며, ratio 차원은 stage1 → stage2 base 계보 (Stage 1 ratio merged 가 Stage 2 world-model variant 의 base) 로만 흐른다. 산출 디렉토리/HF slug 는 ratio 별 분리 (`outputs/AndroidControl_EXP01/{adapters,merged,eval}/{MODEL}_ratio{37,55,73}_stage2_*`, `SaFD-00/{short}-ac-exp01-ratio{37,55,73}-...`). ratio 별로 학습 가중치가 다르므로 `--exp01-ratios` 가 sweep 단위, `--exp01-train-total` 이 Stage 1 train 합계 (기본 50K). Stage 2 학습 데이터의 last-message wrapping (`<thought>…</thought>\n<action>{...}</action>`) 은 `_parse_action_payload` regex helper 가 분리.
- **AC_EXP02 (Stage 1 state-pred diff loss 실험군)**: split 불필요. `scripts/diff_loss/preprocess_dataset.py` 가 AC_EXP01 ratio73 train (`implicit-world-modeling_stage1_train_7_3.jsonl`) 의 future HTML 토큰에 diff 가중치를 부여한 `token_weights` 필드를 추가 — current→future HTML diff 를 헝가리안 매칭으로 ADDED/MODIFIED/UNCHANGED 분류 (가중치 2.0/2.0/1.0). action_pred 샘플은 assistant 가 JSON 이라 diff element 0 개 → `token_weights` 전부 1.0 → 기존 cross-entropy 와 동치 (action 은 기존 loss). test / Stage 2 데이터는 AC_EXP01 에서 복사 (`DS_DATADIR[AC_EXP02]=AndroidControl_EXP02`, 노트북 환경 세팅 셀이 일괄 수행). diff loss 는 LlamaFactory 6 파일 패치 (`use_diff_token_weighted_loss` 인자 + `diff_token_weighted_loss_func` + collator 의 labels 기반 token_weights 복원) 에 의존 — LF 는 gitignore 된 별도 repo 라 `scripts/diff_loss/apply_llamafactory_patch.py` 가 멱등 재적용한다. (`scripts/diff_loss/` 의 `hungarian_metric.py` 는 채점용 `scripts/_hungarian_eval.py` 와 의도적으로 분리된 학습 전처리용 사본.)
- **diff loss v1 / v2 구조 병존 (AC_EXP02 vs AC_EXP05)**: `scripts/diff_loss/` 에는 `{hungarian_metric,hungarian_diff,token_weight_builder,preprocess_dataset}.py` (v1, AC_EXP02 전용) 와 `..._v2.py` (v2, AC_EXP05 전용) 가 나란히 존재한다 — v1 은 그대로 두고 v2 를 새로 추가해 EXP02 재현성을 보존한다.

  | | v1 (AC_EXP02) | v2 (AC_EXP05) |
  |---|---|---|
  | 위치 매칭 cost | DOM `index` | bounds 중심점 거리 (`W_POS=0.4`, `BOUNDS_NORM=2050.0`=840×1876 대각선, `BOUNDS_TAU=50.0`) |
  | 텍스트 수집 | 자손 텍스트 흡수 포함 | direct text + 자체 속성(description/id/text/aria-label)만 |
  | `MATCH_THRESHOLD` | 1.5 | 1.7 |
  | element 키 | index | index → bounds 폴백 |
  | metric key | `hungarian_idx` | `hungarian_pos` |
  | diff 토큰 가중치 | ADDED/MODIFIED **2.0** / UNCHANGED 1.0 | diff(ADDED/MODIFIED) **1.0** / non-diff(UNCHANGED) **0.25** |
  | action_pred 샘플 | uniform 1.0 (diff element 0개라 자동 동치) | uniform 1.0 (명시적 분기 — 판별은 `images` 개수: 1개=state_pred, 2개=action_pred) |
  | 토큰↔element 경계 | 토큰 **시작점만** 검사 (왼쪽 경계를 걸친 토큰 누락 — 비대칭) | **interval overlap** (`tok_cs < char_end and tok_ce > char_start`, zero-length offset 제외). 중첩 span 은 **max 가중치** |
  | 실패 처리 | fail-open (uniform 폴백을 성공으로 집계) | `--on-error {fail,uniform,skip}`, 기본 **fail-closed**. 폴백은 성공으로 집계하지 않음 |
  | 출력 쓰기 | 최종 경로 직접 스트리밍 (부분 산출물 가능) | sibling temp → **`os.replace` 원자 교체**, `--input == --output` 거부 |
  | 재현성 | tokenizer revision 미고정·미기록 | `--revision` 고정 + `<out>.meta.json` sidecar 에 model/revision/가중 상수/집계 기록 |

  > **v2 의 경계 교정(실측 영향)**: 시작점만 보던 규칙은 element 왼쪽 경계를 걸친 토큰(앞 텍스트와 합쳐 토크나이즈된 것)을 놓치면서 오른쪽으로 넘치는 토큰에는 가중치를 줬다. interval overlap 으로 바꾸자 state 출력 토큰의 **0.65% 가 바뀌었고 전부 `0.25 → 1.0` 상향**(하향 0건) — 버그 서명과 정확히 일치한다. v1 (`token_weight_builder.py`) 에도 같은 비대칭이 있으나 **EXP02 재현성 보존을 위해 고치지 않는다**.

  **EXP05 에 v2 가 필수인 이유**: EXP05 HTML 에는 `index` 속성이 없다 (실측: index 0개, bounds 48개). v1 builder 는 `index="..."` 를 regex 로 필수 요구하므로 v1 을 EXP05 에 쓰면 모든 토큰이 baseline 으로 방치되어 diff loss 가 조용히 무력화된다.

  **신규 가중 체계의 순효과**: 배수 자체(2.0→1.0)는 줄었지만 non-diff 도 1.0→0.25 로 함께 낮아져 diff 가 non-diff 대비 실질 **4 배** 강해진다 (근거: 2026-07-11 Slack DM + Google Meet 회의록).

  **가중치 적용 함정**: `token_weight_builder` 의 baseline 은 `[1.0]*n_asst` 이고 `if weight == 1.0: continue` 로 기본값을 스킵하는 구조다. 신규 체계에서는 diff weight 가 바로 그 1.0 이라, 스킵 때문에 diff 토큰이 baseline(0.25) 에 방치되는 함정이 있었다. v2 는 baseline 을 `wmap["UNCHANGED"]` 에서 유도하고 스킵 조건을 `if weight == base` 로 바꿔 해결했다.

  **EXP05 배선**: EXP05 stage1 YAML 6 개 전부 `use_diff_token_weighted_loss: true`. `token_weights` 는 train jsonl 인라인 필드이며 `dataset_info.json` 컬럼 등록은 불필요 (`converter.py:226` 이 raw jsonl 에서 직접 읽음). 실측: train **44,670** 행 = state 31,221 + action 13,449, weight 값 분포는 state `{0.25, 1.0}` / action `{1.0}` (state 출력 토큰의 53.4% 가 0.25 배 감쇠). 빌드 정본은 [`scripts/build_exp05_data.py`](./scripts/build_exp05_data.py) 이며 tokenizer/revision/가중 상수를 `<train>.meta.json` sidecar 에 기록한다.
- **AC_EXP03 (좌표(point) 표현 미러 실험군)**: split 불필요. `scripts/mirror_experiment.py --experiment exp03` 가 AC_EXP01 ratio73 의 산출 파일(`stage1_train_7_3` + dual-task test + Stage 2)을 한 줄씩 읽어 `(episode, step)` 키로 `data/AndroidControl/implicit-world-modeling_stage{1_action,1_state,2}_xy.jsonl` (좌표 표현 원천) 의 대응 레코드를 골라 동일 순서로 `data/AndroidControl_EXP03/` 에 write. UI 트리는 `index="N"` 대신 `bounds="[x1,y1][x2,y2]" point="[cx,cy]"`, 액션은 `point=[x,y]` (0–1000 정규화). 본문만 좌표, 이미지 경로는 EXP01 레코드의 것(`AndroidControl/images/...`)을 채택. 원천에 없는 `(episode, step)` 키(~0.8–1.7%)는 제외 — 각 레코드는 EXP01 과 `(episode, step)` 1:1 대응이나 행 수는 소폭 작다 (train ~49,596 / stage2_train ~14,881). stage1 train 은 EXP02 스타일 단일 `implicit-world-modeling_stage1_train.jsonl` 로 출력. diff loss 미적용 — index→좌표 표현 효과만 AC_EXP01 ratio73 대조군과 비교.
- **AC_EXP04 (좌표 표현 + stage1 프롬프트 업그레이드 실험군)**: split 불필요. EXP03 의 **stage1 프롬프트 업그레이드** 변형으로, EXP03 와 동일한 `(episode, step)` 멤버십·좌표(point) 표현을 유지하되 프롬프트만 바뀐다 — action space `scroll(direction, point)` → `swipe(start, end)`, role 문구 "represented as html-style XML" 명시, `[SWIPE]` 규칙 추가. `scripts/mirror_experiment.py --experiment exp04` 가 AC_EXP01 ratio73 산출 파일(EXP03 와 동일 멤버십)을 한 줄씩 읽어 `(episode, step)` 키로 `data/AndroidControl/implicit-world-modeling_stage1_{action,state}_xy_prompt-enhanced.jsonl` (EXP03 좌표 pool 의 프롬프트 업그레이드본) 의 대응 레코드를 골라 동일 순서로 `data/AndroidControl_EXP04/` 에 write — 본문만 좌표/프롬프트, 이미지 경로는 EXP01 채택. **EXP04 pool ⊆ EXP03 pool** 이라 멤버십 = EXP03 ∩ EXP04 pool 이고, EXP03 출력 대비 drop 은 train 320 / 전체 stage1 450 (0.67%) (EXP03 train 49,596 → EXP04 49,276). **Stage 2 는 보류** — `_STAGE1_ONLY` 에 `AndroidControl_EXP04` 를 포함해 stage2 YAML/등록/eval 을 skip 하며, stage1 train + dual-task test(4) + state without_open_app sibling(2) 7 파일만 산출. stage1 train 은 EXP02 스타일 단일 `implicit-world-modeling_stage1_train.jsonl` 로 출력.
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

| 용도 | AndroidControl | AndroidControl_EXP01 | AndroidControl_EXP02 | AndroidControl_EXP03 | AndroidControl_EXP04 | MonkeyCollection | MobiBench |
|------|----------------|-------------------|-------------------|-------------------|-------------------|-------------------|-----------|
| `data/` 실제 디렉토리 | `AndroidControl` (원본 source-only) | `AndroidControl_EXP01` | `AndroidControl_EXP02` | `AndroidControl_EXP03` | `AndroidControl_EXP04` | `MonkeyCollection` | `MobiBench` |
| shell 단축 코드 | — (학습/평가 entry 없음) | `AC_EXP01` (ratio 별 가상 키 `AC_EXP01_ratio{37,55,73}` 으로 expand) | `AC_EXP02` | `AC_EXP03` | `AC_EXP04` | `MC` | `MB` (eval 전용) |
| LF dataset prefix | — | `IWM-AC_EXP01` (test 공유) + `..._train_ratio{37,55,73}` | `IWM-AC_EXP02` | `IWM-AC_EXP03` | `IWM-AC_EXP04` | `IWM-MC` | `IWM-MB` |
| `outputs/` 최상위 | — | `AndroidControl_EXP01` (ratio 는 아래 model dir 의 `_ratio{37,55,73}` suffix 로 운반) | `AndroidControl_EXP02` | `AndroidControl_EXP03` | `AndroidControl_EXP04` | `MC` | — (TRAIN_DS 산하 `on-MB/`) |
| test split | — | (id, ood) × (state, action) 4 파일 | (id, ood) × (state, action) 4 파일 (AC_EXP01 복사) | (id, ood) × (state, action) 4 파일 (AC_EXP01 좌표 미러) | (id, ood) × (state, action) 4 파일 (EXP03 미러 + 프롬프트 업그레이드) | 단일 test | 단일 파일 |
| Stage 2 지원 | — | ✓ (ID/OOD 3 섹션, ratio sweep — stage1 → stage2 계보) | ✓ (ID/OOD 3 섹션, diff loss 미적용) | ✓ (ID/OOD 3 섹션, 좌표 표현) | ✗ (보류, `_STAGE1_ONLY` — stage1 만) | ✗ (데이터 없음, `_STAGE1_ONLY`) | ✓ (single-pair overall) |

### LLaMA-Factory 등록

- 노트북 Section 1-2 가 `LlamaFactory/data/dataset_info.json` 을 갱신 — 등록 분기는 세 직교 플래그 (`_STAGE1_ONLY`, `_SINGLE_TEST`, `_DUAL_TASK_TEST`) 로 결정:
  - **AC_EXP01** (`_DUAL_TASK_TEST`, ratio 변형 3 종): Stage 1 = `IWM-AC_EXP01_stage1_train_{r37,r55,r73}` (3) + `IWM-AC_EXP01_stage1_test_{id,ood}_{state,action}` (4) — test 4 파일은 ratio 변형 간 공유. Stage 2 = `IWM-AC_EXP01_stage2_{train,test_id,test_ood}` (3) — ratio 무관 공유. 총 10 entry (ratio variant 3 회 등록 시 train_{rXX} 만 ratio 별로 다름).
  - **AC_EXP02** (`_DUAL_TASK_TEST`, diff loss 실험군): AC_EXP01 와 동일 구조 — Stage 1 = `IWM-AC_EXP02_stage1_train` (1) + `IWM-AC_EXP02_stage1_test_{id,ood}_{state,action}` (4), Stage 2 = `IWM-AC_EXP02_stage2_{train,test_id,test_ood}` (3). 총 8 entry. train JSONL 만 `token_weights` 필드 포함 — `columns` 등록은 불필요 (converter 가 raw 필드를 `_token_weights` 로 직접 전달).
  - **AC_EXP03** (`_DUAL_TASK_TEST`, 좌표 표현 실험군): AC_EXP02 와 동일 구조 — Stage 1 = `IWM-AC_EXP03_stage1_train` (1) + `IWM-AC_EXP03_stage1_test_{id,ood}_{state,action}` (4), Stage 2 = `IWM-AC_EXP03_stage2_{train,test_id,test_ood}` (3). 총 8 entry. 본문이 좌표(bounds/point) 표현이라는 점 외 등록 구조는 동일 (diff loss·token_weights 없음).
  - **AC_EXP04** (`_DUAL_TASK_TEST` + `_STAGE1_ONLY`, 좌표 + stage1 프롬프트 업그레이드 실험군): AC_EXP03 와 같은 dual-task 구조이나 **Stage 2 보류** — Stage 1 = `IWM-AC_EXP04_stage1_train` (1) + `IWM-AC_EXP04_stage1_test_{id,ood}_{state,action}` (4) 의 **stage1 5 키만** 등록한다. `_STAGE1_ONLY` 에 `AndroidControl_EXP04` 가 포함돼 stage2 YAML/등록/eval 이 skip 되므로 AC_EXP03 의 8 키(stage1 5 + stage2 3)와 달리 stage2 3 키는 없다. 본문이 좌표(bounds/point) + 프롬프트 업그레이드(swipe / html-style XML / `[SWIPE]`)라는 점 외 stage1 등록 구조는 동일 (diff loss·token_weights 없음).
  - **MC** (`_STAGE1_ONLY` + `_SINGLE_TEST`): `IWM-MC_stage1_{train,test}` 2 entry. `_STAGE1_ONLY = {"MonkeyCollection", "AndroidControl_EXP04"}` — AC_EXP01 는 더 이상 게이트되지 않고, AC_EXP04 는 Stage 2 보류로 게이트된다 (stage2 YAML/등록/eval skip; AC_EXP04 자체는 `_DUAL_TASK_TEST` 라 stage1 test 는 4 파일 dual-task).
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
  - `--train-dataset {AC_EXP01|AC_EXP02|AC_EXP03}` (MC 거절. AC_EXP01 는 `--exp01-ratio` 로 단일 ratio 지정) + `--eval-datasets LIST` (`AC_EXP01, AC_EXP02, AC_EXP03, MB`). EVAL_DS 별 분기:
    - **AC_EXP01 / AC_EXP02**: ID + OOD 두 test 파일 함께 추론 → `_action_eval.py score --test-id ... --pred-id ... --test-ood ... --pred-ood ...` 가 **overall / in_domain / out_of_domain** 3 섹션 기록.
    - **MB**: 단일 파일 1 회 추론 → single-pair `overall` 1 섹션.
  - 결과: `outputs/{OUT_DS}/eval/{MODEL}{SFX}/stage2_eval/{variant_path}[_from_{M1}-ep{E1}]/epoch-{E2}/on-{EVAL_DS}/` (AC_EXP02: OUT_DS=AndroidControl_EXP02, SFX=""; AC_EXP01: OUT_DS=AndroidControl_EXP01, SFX=`_ratio{37,55,73}`. variant_path 는 CLI VARIANT 의 `world_model` → `world-model` 치환).
  - **재실행 시 skip**: marker `action_metrics.json` 존재 unit 은 variant × EVAL_DS 조합 별로 독립 skip.

### Shell script CLI

```bash
# 학습/merge — --dataset {AC_EXP01|AC_EXP02|AC_EXP03|AC_EXP04|MC}. MB 거절. Stage 2 는 MC·AC_EXP04 미지원 (데이터 없음 / 보류). --dataset all 은 지원하지 않음 (명시적 선택 필수). AC_EXP01 는 Stage 1/2 모두 ratio sweep.
bash scripts/stage1_train.sh --model qwen2.5-vl-7b --dataset MC --stage1-mode lora
bash scripts/stage1_merge.sh --model qwen2.5-vl-7b --dataset MC --stage1-mode lora --no-hf-upload
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

**학습/merge (`stage{1,2}_{train,merge}.sh`)**: `--dataset {AC_EXP01|AC_EXP02|AC_EXP03|AC_EXP04|MC}` (MB 거절. `--dataset all` 은 지원하지 않음 — 명시적 DS 선택 필수. AC_EXP04 는 stage1 만 — Stage 2 는 `_STAGE1_ONLY` 로 자동 skip) · `--stage1-mode {full|lora}` (기본 full) · `--stage2-mode {full|lora}` (stage2 전용, 기본 lora) · `--stage1-epoch N` (stage2 world-model 전용) · `--exp01-ratios LIST` (AC_EXP01 sweep, 기본 `r37,r55,r73`).

**평가 (`stage{1,2}_eval.sh`)**: `--train-dataset` (stage1: AC_EXP01|AC_EXP02|AC_EXP03|AC_EXP04|MC, stage2: AC_EXP01|AC_EXP02|AC_EXP03) · `--eval-datasets LIST` (stage1: AC_EXP01, AC_EXP02, AC_EXP03, AC_EXP04, MC, MB; stage2: AC_EXP01, AC_EXP02, AC_EXP03, MB. 기본 = train-dataset 단일값) · `--epochs LIST` (기본 1,2,3) · `--variants LIST` · `--exp01-ratio {r37|r55|r73}` (AC_EXP01 train 단일 ratio, 기본 r55).

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
       (EVAL_DS ∈ {AC_EXP01, AC_EXP02, AC_EXP03, AC_EXP04, MC, MB} — AC_EXP01/AC_EXP02/AC_EXP03/AC_EXP04 는 state/action dual-task ID/OOD, MC/MB 는 단일 파일)
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
- **AC_EXP01 / AC_EXP02 / AC_EXP03 / AC_EXP04 dual-task 분기 (Stage 1 한정)**: Stage 1 의 EVAL_DS=AC_EXP01 / AC_EXP02 / AC_EXP03 / AC_EXP04 일 때는 state_pred / action_pred 두 task 를 각각 독립 채점하여 `on-{DS}-state/hungarian_metrics.json` (Stage1 채점, `_hungarian_eval.py`) + `on-{DS}-action/action_metrics.json` (Stage2 채점, `_action_eval.py`) 두 산출물을 만든다. AC_EXP01 ratio 차원은 학습 산출물 (TRAIN_DS=`AC_EXP01_ratio{37,55,73}`) 에 박혀있고 test 4 파일은 ratio 와 무관. without_open_app sibling 은 state branch 만 (action 채점기 미지원). Stage 2 의 EVAL_DS=AC_EXP01 / AC_EXP02 / AC_EXP03 는 dual-task 가 아니라 일반 action prediction 으로 ID + OOD 두 파일 (`implicit-world-modeling_stage2_test_{id,ood}.jsonl`) 을 함께 채점 — 3 섹션 (`overall` / `in_domain` / `out_of_domain`).
- **xy 좌표 스페이스 채점 (EXP05, opt-in)**: EXP05 는 액션 스페이스가 xy 좌표로 통일되어 GT 스키마가 `<action>{"action":"click","coordinate":[x,y]}</action>` (키가 `action`, 구 `action_type` 과 다름; swipe 는 `coordinate1`/`coordinate2`) 로 바뀐다. EXP01~04 채점 결과가 불변이도록 **opt-in 플래그** 로 구현했다 — `_action_eval.py --coord-mode {index,xy}` (기본 `index`), `_hungarian_eval.py --match-mode {index,pos}` (기본 `index`). `stage1_eval.sh` 는 **EVAL_DS=AC_EXP05 일 때만** 이 플래그들을 전달한다. xy 모드 채점 규칙: click/long-press 는 pred 좌표가 GT 좌표가 속한 element 의 bbox 안이면 정답 (포함 element 가 없으면 오답 + `no_bbox_n` 으로 별도 집계), scroll/swipe 는 xy1→xy2 벡터의 주 방향(`|dx|>=|dy|` 이면 left/right, 아니면 up/down) 이 GT 와 일치하면 정답, input_text/type 은 좌표 무관. **알려진 제약**: bbox 채점은 pred 가 GT 와 같은 840×1876 절대 픽셀 공간임을 가정한다.

### Stage 2

- baseline: zero-shot (variant `base`)
- 변형: `{full|lora}_base`, `{full|lora}_world_model` (world-model 은 `--stage1-epoch` 로 상류 epoch 지정)
- 평가 파일: EVAL_DS=AC_EXP01 / AC_EXP02 / AC_EXP03 는 `implicit-world-modeling_stage2_test_{id,ood}.jsonl` (3 섹션, AC_EXP01 는 ratio 무관 공유, AC_EXP02 는 AC_EXP01 복사본, AC_EXP03 은 AC_EXP01 좌표 미러). MB 는 `implicit-world-modeling_stage2.jsonl` (single-pair).
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
- **transformers 버전**: `pyproject.toml` 의 `[project.optional-dependencies] llamafactory` 에서 `transformers>=4.57.1,<4.58` 로 고정 (§ 위 설치 절과 일치). 값과 그 위 주석을 함께 변경한다. 서브프로젝트 `LlamaFactory/pyproject.toml` 은 수정하지 않는다.
- trl 0.24 / transformers 4.56+ API 매핑: `SFTConfig(max_length=...)`, `SFTTrainer(processing_class=...)` 사용. 구버전 키 (`max_seq_length`, `tokenizer=`, `overwrite_output_dir`) 는 `TypeError`.
- `gradient_checkpointing` 은 모델 로드 단계에서만 적용. `SFTConfig` 에는 전달하지 않는다 (이중 적용 방지).
- Full FT 분기에서 `freeze_vision_tower: true` 면 `vision_tower|vision_model|visual|image_encoder` 키워드를 포함한 named parameter 의 `requires_grad=False` 처리 후 frozen 텐서 수/파라미터 수를 stderr 로 출력.
- **EXP05 로컬 학습 불가 (실측)**: 로컬 2×RTX5090 에서 EXP05 3B Full FT 는 **CUDA OOM** (step 3 에서 8.92GiB 할당 실패) + **157~168 s/it → 총 97~104시간(약 4일)**. 원인은 `cutoff_len 24576` + `max_pixels 1,605,632` 의 비전 토큰으로 시퀀스가 극단적으로 길어진 것과, RTX5090 에 강제되는 ZeRO-3 CPU offload. **본 학습은 Vessl A100/H100 에서 수행한다** — 저장소에는 Vessl 파이프라인 스크립트가 없다 (운영 지식).
