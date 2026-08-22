# AGENTS.md

`Implicit-World-Modeling/` 에서 작업하는 에이전트를 위한 **규칙** 문서.

> **이 문서는 수치를 싣지 않는다.** 좌표계·image budget·GPU 매트릭스·하이퍼파라미터·메트릭 정의·산출물 레이아웃의 정본은 [`ARCHITECTURE.md`](./ARCHITECTURE.md) 하나뿐이다. 여기에는 **"무엇을 하면 안 되는가 / 어디를 고쳐야 하는가"** 만 있고 근거는 전부 §N 링크다. 사용법은 [`README.md`](./README.md), 현재 상태는 [`../docs/ROADMAP.md`](../docs/ROADMAP.md).

## 제1원칙

**코드가 진실원천이고 문서는 피고다.** 문서 서술이 코드와 어긋나면 코드가 옳다 — 문서를 고쳐라. SSoT 는 네 파일이다:

| 대상 | 정본 |
|---|---|
| 모델·데이터셋·자격·하이퍼파라미터 | `implicit_world_modeling/lf_registry.py` |
| 학습 YAML | `implicit_world_modeling/gen_configs.py` (`--write` / `--check`) |
| batch / grad_accum / deepspeed | `scripts/gpu_policy.py` |
| shell 경로·가드·추론 커맨드 | `scripts/_common.sh` |

**파생 수치를 문서에 손으로 쓰지 마라** (YAML 개수, 데이터 행수, 등록 키 개수, 모델 개수, 테스트 케이스 수). 드리프트하면 문서가 거짓말이 된다 — 아래 "검증 커맨드" 를 가리켜라.

---

## 하드 제약 — 어기면 **에러 없이** 조용히 깨진다

각 줄이 규칙이고, 근거·수치는 링크에 있다.

1. **LF working tree 를 in-place 수정하지 마라.** `LlamaFactory/` 는 gitignore 된 clone 이라 재클론 한 번에 증발한다 → `patches/llamafactory/` 에 `.patch` 를 추가한다. [§0 함정 1](./ARCHITECTURE.md#0-runtime-stack)
2. **좌표 실험군에 다른 family 모델을 쓰지 마라** (EXP03/04 ↔ EXP05 는 좌표 규약이 정반대다). 이건 문서 규약이 아니라 **코드 가드**다 — `DATASET_MODEL_ELIGIBILITY` 가 정본이고 `require_model_eligible()` 이 학습 진입 전에 `exit 1` 한다. [§2 자격 매트릭스 · 함정 2/3](./ARCHITECTURE.md#2-모델-설정)
3. **"visual token 수가 같으니 괜찮다" 는 틀렸다.** 두 family 는 token 수가 같아도 **리사이즈된 이미지의 실제 픽셀 크기가 다르다** → 절대 픽셀 좌표 데이터를 다른 family 에 먹이면 grounding 만 조용히 깨진다. [§2 함정 4](./ARCHITECTURE.md#2-모델-설정)
4. **budget 일치 규칙**: **데이터가 전제한 리사이즈 픽셀 크기**와 학습·평가 `image_max_pixels` 는 반드시 **같은 budget** 에서 나와야 한다 (EXP05 = 1,605,632 → 840×1876). 어긋나면 좌표계 전체가 무효다. 우리 budget 은 Qwen 공식 기본값이 아니라 의도적 override 이며, **데이터 생성 스크립트에는 budget 인자가 없다** (좌표는 원천 jsonl 에 이미 박혀 온다). [§2 함정 5](./ARCHITECTURE.md#2-모델-설정)
5. **offload 분기를 `GPU_TYPE` 만 보고 짜지 마라 — `(GPU_TYPE, size_class, mode)` 3 축이다.** 옛 `if GPU_TYPE == "RTX5090": …` 처럼 GPU 종류만 보는 분기를 **되살리지 마라** — `7-9B × full` 까지 no-offload 로 넘겨 조용히 OOM 낸다. `--allow-no-offload` 는 offload 를 켜는 조합에서 강제로 끄는 opt-out 이다. 조합별 매트릭스와 메모리 실측 근거는 [§2 GPU 정책 · 함정 7](./ARCHITECTURE.md#2-모델-설정).
6. **하드웨어가 바뀌어도 YAML 을 재생성하지 마라.** 커밋 YAML 은 GPU-불변 baseline 이고, GPU 트리오는 **런타임 override** 로 주입된다 (`GPU_TYPE=… NPROC_PER_NODE=… bash scripts/stage1_train.sh …`). [§2 함정 8](./ARCHITECTURE.md#2-모델-설정)
7. **`cutoff_len` 을 내리지 마라.** 좌표 실험군에서 내리면 multimodal RoPE position 길이 초과로 **첫 step 에서 크래시**한다. 사전 필터 `--threshold` 와 학습 cutoff 는 같은 기준으로 묶여 있다. [§2 함정 6](./ARCHITECTURE.md#2-모델-설정)
8. **EXP05 에 diff loss v1 을 쓰지 마라.** v1 builder 는 `index` 속성을 필수 요구하는데 EXP05 HTML 에는 그 속성이 없다 → **에러 없이** 전 토큰이 baseline 으로 방치돼 diff loss 가 무력화된다. v2 (`*_v2.py`) 를 쓴다. [§3 함정 9](./ARCHITECTURE.md#3-데이터와-설정-계약)
9. **`scripts/diff_loss/` 의 v1 4 파일을 삭제하지 마라. 데드 코드가 아니다** — EXP02 데이터의 유일한 생성기이고, v1 의 경계 비대칭 버그도 EXP02 재현성 보존을 위해 **의도적으로 고치지 않는다.** [§3 함정 10](./ARCHITECTURE.md#3-데이터와-설정-계약)
10. **diff loss 가중치를 만질 때 baseline skip 과 action-uniform 분기를 함께 보라.** baseline 을 `wmap["UNCHANGED"]` 에서 유도하지 않으면 diff 토큰이 baseline 에 방치되고, action 샘플의 uniform 분기가 빠지면 "diff 없음 → 전부 최저 가중치" 로 잘못 처리된다. [§3 함정 11](./ARCHITECTURE.md#3-데이터와-설정-계약)
11. **`configs/lf_dataset/dataset_info.json` 은 커밋된 정본이다 — 런타임에 쓰지 마라.** LF 안의 `data/dataset_info.json` 을 변조하던 방식은 재클론에 증발해서 은퇴했다. [§3 함정 13](./ARCHITECTURE.md#3-데이터와-설정-계약)
12. **JSONL 의 `images` 값은 `{DATASET_NAME}/images/...` prefix 를 유지하고, `--dataset_dir` 에는 절대경로를 넘겨라.** prefix 가 빠지면 `Image.open()` 이 cwd 기준으로 풀려 실패하고, 상대 `--dataset_dir` 은 HF datasets 캐시 오염으로 `FileNotFoundError` 를 낸다. [§3 함정 15](./ARCHITECTURE.md#3-데이터와-설정-계약)
13. **EXP05 bbox 채점은 pred 가 GT 와 같은 절대 픽셀 공간임을 가정한다.** 모델이 다른 좌표 공간으로 답하면 채점이 조용히 전부 오답이 된다. [§6 함정 18](./ARCHITECTURE.md#6-메트릭)
13b. **state 예측을 `hungarian_f1` 만으로 판단하지 마라 — "예측"과 "입력 복사"가 구분되지 않는다.** next-state 는 current state 와 대부분 겹치므로 current 를 그대로 베껴도 f1 이 높다 (2026-08-01 실측: EXP05 zero-shot base 는 `copy_rate_pred` 0.986·`added_recall` 0.011 인데도 f1 0.19 가 나온다). `state_diff_metrics.json` 의 **`copy_excess`** 와 함께 읽는다. `copy_rate` 단독도 금지 — 완벽한 예측도 GT 자체가 겹치는 만큼 높게 나온다. 절단(1024) leaf 에는 산출하지 않는다 (copy_rate 가 한쪽으로 과소평가된다). [§6 state_diff_metrics.json · 함정 19](./ARCHITECTURE.md#6-메트릭)
14. **YAML 이 있다고 돌릴 수 있는 게 아니다.** 가드는 YAML 유무가 아니라 **dataset_info 등록 여부**를 본다 (`require_yaml` 이 내부에서 `require_dataset_registered` 를 호출한다 — [§3 함정 14](./ARCHITECTURE.md#3-데이터와-설정-계약)). EXP03/04 YAML 은 as-trained 가 아니라 **생성기 재구성본**이다 — [§7 함정 20](./ARCHITECTURE.md#7-중요한-운영-제약).
15. **EXP05 는 로컬에서 학습할 수 없다** (OOM + 수일 단위 소요, 실측). 본 학습은 원격 A100/H100 — 단 제출 스크립트는 **UNVALIDATED** 다. [§7 함정 19](./ARCHITECTURE.md#7-중요한-운영-제약)
15b. **EXP06 = EXP05 의 비증강 Stage-2 대조군**이다 (좌표/budget/`--coord-mode xy` 규약을 EXP05 에서 승계). `lf_registry` 에 **등록 완료**됐고(2026-07-20) 모델 자격 코드 가드 (하드 제약 2) 가 EXP05 와 동일하게 **Qwen2.5-VL 계열 전용**으로 걸린다 — 다른 family 는 `require_model_eligible()` 이 학습 진입 전 `exit 1` 한다. **EXP06 자체엔 stage1 체크포인트가 없다** — stage2 world-model variant 는 `stage1_hf_slug: "ac-exp05-"` override + 셸 `ds_stage1_source()` 로 **EXP05 stage1 을 그대로 승계**한다 (stage2 산출물 네이밍은 `ac-exp06-` 그대로 유지). [§2 자격 매트릭스 각주](./ARCHITECTURE.md#2-모델-설정)
15c. **EXP07 = `qwen2.5-vl-3b` 단독 stage1 world-modeling 실험군**이다 (좌표계·budget·cutoff 는 EXP05 와 같되 **자격을 3B 로 좁혔다** — 7B 제외, `eligible_models('AndroidControl_EXP07') == ['qwen2.5-vl-3b']`). 모델 자격 코드 가드 (하드 제약 2) 가 EXP07 에도 걸린다 — `qwen2.5-vl-3b` 외 조합은 `require_model_eligible()` 이 학습 진입 전 `exit 1` 한다. **데이터는 EXP05 파생이 아니다** — 별도 0725 소스(`data/AndroidControl/EXP07_*.jsonl` 3 파일)에서 자체 빌드한 train 2 (stage1 50K / stage2 15K) + **자체 test 6 종**이고, EXP05 를 향하는 데이터 심링크·포인터는 **없다** (과거 잠정본의 심링크는 빌더가 제거한다 — `configs/lf_dataset/` 의 dataset_dir 심링크는 별개로 필요). **EXP06 과 달리 EXP07 은 자체 stage1 을 가진다** (`ds_stage1_source` 미매핑, `stage1_hf_slug` 미설정). stage2 에는 **merge X 변형(`world-model-adapter`)** 이 EXP07 한정 opt-in 이다. [§2 자격 매트릭스 각주](./ARCHITECTURE.md#2-모델-설정) · [§3 계보](./ARCHITECTURE.md#3-데이터와-설정-계약)
15d. **AC_EXP08 = 3-포맷 관측성 분할(anti-copy) stage1 실험군**이다 (좌표계·budget·cutoff 는 EXP05/07 과 같고 자격은 `qwen2.5-vl-3b`·`qwen2.5-vl-7b`). 데이터는 EXP01/05/07 파생이 **아니고** `data/AndroidControl/EXP08_*.jsonl` (0822 Cerebra 파서 재추출) 에서 자체 빌드한다 — 다만 이미지는 공용 `AndroidControl/images/` 를 그대로 참조한다 (zero-pad remap, 전수 매핑 확인). **누출 차단은 에피소드 단위 홀드아웃**이고 **ID/OOD 가 없다** — 원본에 앱 파티션 메타가 없다. state test 는 `full`/`masked`/`dropped` 3 파일이며 **같은 500 원본에서 나온다** (sample_id 집합 동일이 불변식). 빌드 정본은 `scripts/build_exp08_data.py`. [§3 계보](./ARCHITECTURE.md#3-데이터와-설정-계약)
15e. **EXP08 에 diff loss v2 를 쓰지 마라 — `_v2c` 를 써라.** EXP08 XML 은 Cerebra 스키마(`data-bbox="x1 y1 x2 y2"` / `aria-label`)라 v2 의 `bounds` 파서가 **에러 없이** 전부 실패해 위치 신호가 죽고 element 집합이 쪼그라든다 (하드 제약 8 과 같은 실패 유형). 반대로 **`_v2c` 로 기존 실험군을 재생성하지도 마라** — element 키 축 우선순위가 달라 EXP02/05/06/07 재현이 깨진다. `build_diff_targets.py` 만 `_v2c` 를 import 하고 `preprocess_dataset_v2.py` 는 `_v2` 를 쓴다. [§3 v2c 절](./ARCHITECTURE.md#3-데이터와-설정-계약)
15f. **Cerebra 스키마를 채점하려면 `--xml-schema cerebra` 를 줘라.** `scripts/_hungarian_eval.py` / `_state_diff_eval.py` 의 기본값은 `android` 이고, 기본값으로 EXP08 을 채점하면 **에러 없이** 지표가 무너진다. 반대로 기존 실험군에 `cerebra` 를 주지 마라 — 채점 기준이 바뀐다 (opt-in 규약: 기존 실험군의 채점 결과는 불변이어야 한다). [§6 메트릭](./ARCHITECTURE.md#6-메트릭)
16. **`SMOKE=1` override 의 따옴표를 살려라.** OmegaConf 가 따옴표 없는 값을 boolean 으로 파싱해 HF 가 죽는다. [§4 함정 17](./ARCHITECTURE.md#4-파이프라인-컴포넌트)
17. **데이터 파일명 규칙을 지켜라 — 디렉토리가 실험을 말하면 파일은 침묵한다.** `AndroidControl_EXP{NN}/`·`MonkeyCollection/`·`MobiBench/` 안은 `stage{N}_<role>.jsonl` 이고, 공유 폴더 `data/AndroidControl/` 의 변형본만 `EXP{NN}_` 접두를 쓴다 (표현 서픽스 `_xy`/`_pixel-aligned`/`_filtered` 는 **쓰지 않는다** — EXP 소속이 표현을 함의한다). 멀티-exp 공유 파일은 **직접 소비하는 최초 EXP 하나**만 소유하고 재사용 관계는 문서가 기록한다. 원본 3 종 (`implicit-world-modeling_*`) 과 `episodes_meta.jsonl`·`images/` 는 이름을 바꾸지 않는다. **빌더 입력 원천도 예외가 아니다 — `*_src` 전용 디렉토리를 만들지 마라** (벤더 파일명을 버리고 공유 폴더에 `EXP{NN}_` 접두로 평탄하게 놓는다; 한 파일이 여러 stage 를 먹이면 **최대 소비자** 이름, 균등 분배라 최대 소비자가 없으면 **stage 중립명**). [§3 데이터 파일명 규칙](./ARCHITECTURE.md#3-데이터와-설정-계약)

### 인용 금지 (유령 참조)

- **`scripts/tmux_*.sh` 를 커맨드로 제시하지 마라.** `.gitignore` 대상이라 저장소에 존재하지 않는다 — 실행은 `scripts/stage{1,2}_{train,merge,eval}.sh` 를 직접 호출한다. [§4 함정 16](./ARCHITECTURE.md#4-파이프라인-컴포넌트)
- `docs/EXP05_DIFF_LOSS_PLAN.md`, `docs/research/gui-exploration-world-model.md` 는 **삭제됐다** — 링크하지 마라.
- **`MC` (MonkeyCollection) 데이터셋은 배선만 검증됐고 프로덕션 코퍼스가 아니다** — 이 데이터 학습 결과를
  코퍼스 품질의 근거로 쓰지 마라. 실제 학습 전 `--stage1-ratio` 재조정이 선행돼야 한다. 데이터 상태·오염
  경위는 [ARCHITECTURE §3 "MC 데이터 상태"](./ARCHITECTURE.md#3-데이터와-설정-계약).
- **`scripts/stage1_train.sh` 는 개발용 맥에서 실행되지 않는다** — `_common.sh` 가 `CONDA_PREFIX`(conda env
  `implicit-world-modeling`)와 **bash 4+** 를 요구하고, `LlamaFactory/` 도 체크아웃에 없다. **DRY_RUN 조차
  안 된다.** 이 스크립트들은 **원격 리눅스 GPU 박스** 전용이다. 맥에서 검증 가능한 최대치는 **정적 관통
  확인**(jsonl 스키마 + 이미지 경로 해석 + `python -m implicit_world_modeling.gen_configs --check`)이다.
  - **MC 브리지 end-to-end 실행은 보류됐다 (사용자 결정, 2026-07-14 — 지금은 핸드오프만).** 배선·정적 관통은
    확인됐으니 남은 것은 원격 GPU 박스에서 `stage1_train.sh --dataset MC` 를 실제로 한 번 돌려보는 것뿐이다.
    경로: `setting:claude-code-remote` 로 원격 서버에 붙거나 `scripts/remote_launch.sh`.

---

## ⚠ EXP04 — 손대기 전에 반드시 읽어라

**EXP04 데이터의 좌표계가 문서 전제와 모순된다 (2026-07-13 실측, 미해결).** 규칙:

- **"EXP04 는 0–1000 정규화" 라고 다시 쓰지 마라** — 실측과 어긋난다 (거짓).
- **동시에 "원본 픽셀이 EXP04 의 스펙" 이라고 못박지도 마라** — EXP03 의 프롬프트만 바꾼 변형이라 **데이터 생성 버그일 공산이 크다.**
- 선결 순서: **좌표 규약 확정 → (원천 확보 후) 재빌드 → dataset_info 등록.** 등록부터 하지 마라.

현재 무엇이 막혀 있는가 (**상태**) 는 [`../docs/ROADMAP.md`](../docs/ROADMAP.md) 를 본다 — 규칙은 하드 제약 14 (YAML 이 있다고 돌릴 수 있는 게 아니다) 가 이미 커버한다.

실측값·상세는 [ARCHITECTURE §2 경고 블록](./ARCHITECTURE.md#2-모델-설정). 모델 자격 서술 자체는 코드가 데이터 내용과 무관하게 강제하므로 유효하다 — 흔들리는 것은 자격의 *근거로 적힌 좌표계 전제*다.

---

## 어디를 고쳐야 하는가

### 모델 추가

1. **새 family 라면 `lf_registry.py::MODEL_FAMILY_CONFIG` 를 먼저** 등록한다 — 코드가 실제로 읽는 키는 **`max_pixels` / `min_pixels`** (`_img_cfg` 가 이 둘을 읽는다) 와 **`factor`** (`build_configs` 의 image_overrides token→px 환산용) 다. 순서를 어기면 `_img_cfg` 가 `KeyError` 를 낸다. **`min_tokens` 는 이 dict 의 키가 아니다** (`image_overrides` 전용 override 키다 — 아래 "새 데이터셋 추가" 참조). 스키마 표: [§2 image budget](./ARCHITECTURE.md#2-모델-설정)
2. `lf_registry.py::_MODEL_CONFIG` 에 항목 추가 (`model_id` · `short_name` · `template` · `size` tier). image-pixel 값은 `_img_cfg(short)` 가 family config 에서 **자동 주입**하므로 손으로 쓰지 않는다.
3. `scripts/_common.sh` 의 `MODEL_ID` / `MODEL_TEMPLATE` / `ALL_MODELS` 에 **동일 항목** 추가 — **레지스트리는 두 곳이다. 동시에 고쳐야 한다.**
4. 새 family 라면 `_common.sh::build_infer_cmd` 의 template 분기 (factor / mm_min / thinking 플래그) 도 갱신한다.
5. 좌표 실험군에서 쓰려면 `lf_registry.py::DATASET_MODEL_ELIGIBILITY` 에 넣어야 한다. **넣지 않아 학습이 막히는 것은 정상 동작이다** (하드 제약 2) — 좌표 규약이 맞는지 먼저 확인하고 넣어라.
6. `python -m implicit_world_modeling.gen_configs --write` 로 YAML 재생성 → **생성물을 커밋한다.**
7. 문서는 **[ARCHITECTURE §2 모델 레지스트리 표](./ARCHITECTURE.md#2-모델-설정) 한 곳만** 갱신한다 — README·AGENTS 에는 모델 목록이 없다 (README 는 `_MODEL_CONFIG` 를 출력하는 확인 커맨드만 싣는다).

### 하이퍼파라미터 수정

- **`_MODEL_CONFIG` 에 lr / warmup / LoRA rank / dropout 을 직접 쓰지 마라.** `hparam_overrides` 는 **모델별 delta 전용**이다 → 값은 `_DATASET_CONFIG[ds].stage{1,2}` baseline 에서 바꾼다.
- `_SIZE_CONFIG_AC` (size tier) 는 **의도적으로 비어 있다** — EXP01/EXP02 실측 어댑터와의 동일조건 비교를 보존하기 위해서다. 채우면 그 비교가 깨진다.
- merge 순서 (dataset baseline → size tier → model delta) 와 실제 값 표: [§2 하이퍼파라미터](./ARCHITECTURE.md#2-모델-설정).
- **Stage 2 full 의 lr 은 `_DATASET_CONFIG` 가 아니라 `gen_configs.render_stage2()` 안에 하드코드돼 있다** — 거기서 고친다.
- **batch / grad_accum / deepspeed 는 레지스트리에 없다** → `scripts/gpu_policy.py`. 손대면 `GLOBAL_BATCH_SIZE` 불변식이 깨지지 않는지 `pytest tests/test_gpu_policy.py` 로 확인한다.
- 고친 뒤 반드시 `gen_configs --write` + 커밋. 안 하면 `--check` 가 CI 에서 exit 1 한다.

### 데이터 분할 · 새 데이터셋 추가

- 분할 정본은 [`scripts/split_data.py`](./scripts/split_data.py) (AC_EXP01 전용; MC 는 random fallback). 좌표 파생 실험군은 [`scripts/mirror_experiment.py`](./scripts/mirror_experiment.py), EXP05 는 [`scripts/build_exp05_data.py`](./scripts/build_exp05_data.py) (stage2 는 [`build_exp05_stage2_data.py`](./scripts/build_exp05_stage2_data.py)), **EXP07 은 [`scripts/build_exp07_data.py`](./scripts/build_exp07_data.py)** 가 정본이다 — EXP07 은 EXP01 계보 밖에서 자체 소스로 빌드하며 train ∩ test = 0 을 불변식으로 검사한다. 계보·분할 규칙: [§3](./ARCHITECTURE.md#3-데이터와-설정-계약).
- **App partition 을 재계산하지 마라** — Stage 2 budget 으로 한 번 계산한 `(id_apps, ood_apps)` 를 Stage 1 이 재사용해야 "OOD 앱은 Stage 1 train 에 한 번도 등장하지 않는다" 가 성립한다. 파생 실험군은 EXP01 멤버십을 미러할 뿐 별도 계산을 하지 않는다.
- **AC_EXP01 split 은 `EXP01_*.jsonl` (filter 산출) 만 입력으로 쓴다** (선행: `filter_long_samples.py`). 원본 `implicit-world-modeling_*` 을 직접 먹이지 마라.
- 새 DS 를 추가하면 **다음을 전부** 손대야 한다 (하나라도 빠지면 학습 진입 전에 죽거나 조용히 잘못 돈다):
  - `lf_registry.py` — `_DATASET_CONFIG`, 세 **직교** 플래그 (`_STAGE1_ONLY` / `_DUAL_TASK_TEST` / `_SINGLE_TEST`), `_LONG_CUTOFF_DS`, `DATASET_MODEL_ELIGIBILITY`
  - `scripts/gpu_policy.py` — `_HALF_BATCH_DATASETS` (좌표 실험군이면)
  - `scripts/_common.sh` — `DS_PREFIX` / `HF_SLUG` / `DS_DATADIR`, `parse_args` · `parse_eval_args` 허용 목록, `build_infer_cmd` 의 cutoff 분기
  - `configs/lf_dataset/dataset_info.json` — **등록 키 + 상대 심링크** (빠뜨리면 하드 제약 14)
  - `scripts/filter_long_samples.py` · `scripts/eval_viewer.py` 인식
  - `python -m implicit_world_modeling.gen_configs --write` → 커밋
- image budget 이 family default 와 달라야 하면 `_DATASET_CONFIG[ds]["image_overrides"]` 에 **token 단위** (`{"max_tokens": N}`) 로 둔다 — 빌더가 family `factor²` 로 환산한다. 평가측 `build_infer_cmd` 도 같은 budget 을 쓰도록 함께 갱신한다 (하드 제약 4).
- 채점기를 바꿔야 하면 **opt-in 플래그로** 추가한다 (EXP05 의 `--coord-mode` / `--match-mode` 가 선례) — 기존 실험군의 채점 결과는 불변이어야 한다.

### GPU / 하드웨어 변경

**아무것도 고치지 않는다.** 호출 시점 환경변수만 준다 (`GPU_TYPE` / `NPROC_PER_NODE` — `.env` 기본값보다 프로세스 환경값이 이긴다). 허용 밖 조합은 학습 진입 전에 거부된다. YAML 재생성 금지 (하드 제약 6). 매트릭스: [§2 GPU 정책](./ARCHITECTURE.md#2-모델-설정).

### LlamaFactory 소스 수정

working tree 를 직접 고치지 말고 `patches/llamafactory/` 에 `.patch` 를 추가한다 (하드 제약 1). `LlamaFactory/pyproject.toml` 은 **건드리지 않는다** — transformers pin 은 우리 `pyproject.toml` 의 `[project.optional-dependencies] llamafactory` 에서 값과 그 위 주석을 함께 바꾼다.

### 실행 오케스트레이션

**노트북은 은퇴했다 (2026-07-26) — 되살리지 마라.** 파이프라인은 전부 `scripts/` 로 실행한다: 다른 모델/모드/DS 는 shell 인자 (`--model` / `--dataset` / `--stage{1,2}-mode`) 만 바꾼다. 데이터는 실험군별 `scripts/build_exp0N_data.py`, 학습 YAML 은 `gen_configs` (손으로 만들지 마라), Merge YAML 은 merge 스크립트가 runtime 에 만든다. 단계 → 정본 스크립트 매핑: [§1 실행 오케스트레이션](./ARCHITECTURE.md#1-실행-구조).

### 문서

**같은 사실을 두 문서가 서술하면 안 된다.** ARCHITECTURE 가 정본이고 나머지는 링크다 — README 는 사용법, AGENTS 는 규칙, ROADMAP 은 상태, `docs/CHANGELOG.md`·`docs/DEVLOG.md` 는 append-only 연대기다. 수치를 옮겨 적지 말고 §N 을 걸어라.

---

## 검증 커맨드

숫자를 문서에서 읽지 말고 여기서 확인한다.

```bash
# 레지스트리/정책 ↔ 커밋 YAML 정합 (byte 대조 + orphan 검출). orphan 도 실패다.
python -m implicit_world_modeling.gen_configs --check

# 자격·등록 실태
python -c "from implicit_world_modeling.lf_registry import eligible_models as e; print(e('AndroidControl_EXP05'))"
python -c "import json;d=json.load(open('configs/lf_dataset/dataset_info.json'));print(sorted(d))"

# 테스트 — 2026-07-26 부터 `pytest tests` 가 **두 env 모두에서 플래그 없이** 돈다
# (--ignore 불필요). 무거운 선택 의존성은 importorskip/skipif 로 가려지므로 카운트가
# env 마다 다른 것이 정상이다:
#   conda `implicit-world-modeling` (학습·평가 env) : 691 passed / 9 skipped
#   .venv                          (lint·test env) : 681 passed / 10 skipped  ← llamafactory 부재 → double_ce 모듈 skip
# conda 의 9 skip 은 전부 test_mirror_experiment 의 **무조건 skip**(원본 삭제됨)이라
# 양 env 에서 동일하게 빠진다. 즉 **conda 그린 = 조건부 skip 0 = 전량 검증**이고,
# `.venv` 그린만 부분 검증(diff-loss 미검증)이다. diff-loss 를 건드렸으면 conda 에서
# 반드시 한 번 더 돌려라. (2026-07-26 conda 에 sacrebleu 설치 전에는 conda 도
# BLEU 6건이 빠져 685/15 였다 — 그 시기 로그를 지금 값과 혼동하지 말 것.)
pytest tests -q
pytest tests/test_gpu_policy.py tests/test_gen_configs.py -q          # GPU 매트릭스 + always-offload 불변식
pytest tests/test_action_eval.py tests/test_action_eval_xy.py -q      # Stage 2 채점 (index / xy 모드)
pytest tests/test_state_diff_eval.py -q                               # Stage 1 copy-bias 진단 (배선 self-test + 층 분해 불변식)
pytest tests/test_copy_baseline_eval.py -q                            # Stage 1 복사기 기준선 (정본 채점기 위임 + recall 층 항등식 + gain 교집합 분모)
pytest tests/test_diff_loss_v2.py tests/test_diff_loss_double_ce.py tests/test_mirror_experiment.py -q

# LF 부트스트랩 (pin + 패치 적용 상태)
bash scripts/setup_llamafactory.sh --verify

# CLI 계약 — nvcc/CUDA 가드는 _common.sh 의 source 시점에 돌아 parse_args 에 도달하기도 전에 exit 1 한다.
# 학습 머신 (nvcc 가 torch 와 같은 cu 버전) 이 아니면 --help 조차 안 나오므로 가드를 우회한다.
LF_CUDA_GUARD_SKIP=1 bash scripts/stage1_train.sh --help
LF_CUDA_GUARD_SKIP=1 bash scripts/stage1_train.sh --dataset MB 2>&1   # 거절돼야 정상 (MB 는 평가 전용)

# transformers pin 이 5.x 를 허용하면 안 된다 (비어야 함)
rg '"transformers[^"]*,<[5-9]' pyproject.toml
```
