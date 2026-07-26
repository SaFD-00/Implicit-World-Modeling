# Roadmap — 상태

> **이 문서는 상태만 싣는다.** 실험군별로 무엇이 완료·진행·차단인가, 차단이면 **정확히 어디서 막히는가**.
> 수치·표·근거·메커니즘은 전부 [`ARCHITECTURE.md`](../Implicit-World-Modeling/ARCHITECTURE.md) 가 정본이고 여기서는 링크만 한다.
> 실험 **결과 지표**는 Notion `🧪 Experiments` DB 가 정본이다 (메트릭 정의는 [§6 메트릭](../Implicit-World-Modeling/ARCHITECTURE.md#6-메트릭)).
>
> **포맷 규약**: 실험군 8종(EXP01–07·MC)은 아래에서 **동일 포맷**으로 싣는다 — 한 줄 상태(요약 표) → 실험군별 `완료 / 남은 것 / 차단·쟁점` 섹션.

---

## 완료 판정 규칙 — 먼저 읽어라

> ⚠️ **`outputs/` 가 비어 있다고 미착수가 아니다.** 학습 산출물은 HF Hub (`SaFD-00/…`) 에 있고 로컬 `outputs/` 는 머신마다 비어 있을 수 있다. eval 은 `resolve_eval_model_path` 가 **local merged 우선 + HF fallback** 으로 푼다 ([§5](../Implicit-World-Modeling/ARCHITECTURE.md#5-실행-데이터-흐름과-산출물)) — 로컬이 비어도 평가는 돈다.
> **EXP03 이 실제 사례다**: 로컬엔 빈 eval 디렉토리 하나뿐인데 HF 에는 stage1·stage2 산출물이 다 있다. 로컬만 보고 "미착수" 로 판정하면 **틀린다.**

**무엇이 학습됐는지 확인하는 정본 커맨드:**

```bash
# 학습 산출물 (정본)
python -c "from huggingface_hub import HfApi; print(*sorted(m.id for m in HfApi().list_models(author='SaFD-00')), sep='\n')"

# 로컬 캐시 (보조 — 비어 있어도 무의미)
find outputs -name '*_metrics.json'
```

HF slug 규약은 [§3 이름 규약](../Implicit-World-Modeling/ARCHITECTURE.md#이름-규약), repo id 조립은 [§5](../Implicit-World-Modeling/ARCHITECTURE.md#5-실행-데이터-흐름과-산출물).
`ac-2-` slug 는 **구 스키마 사문화분**이다 — 현행 실험군이 아니다 (대응 등록 키 `IWM-AC_2_*` 는 2026-07-25 정본에서 제거됐다) ([§3 LF 등록](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약)).

---

## 실험군 상태 — 한 줄 요약

| 실험군 | 상태 | 한 줄 요약 |
|---|---|---|
| **AC_EXP01** | ✅ 완료 (공백 1) | stage1+stage2 완료. **ratio55 만 미학습** |
| **AC_EXP02** | ✅ 완료 | diff loss **v1**. stage1+stage2 완료 |
| **AC_EXP03** | ✅ 완료 | stage1+stage2 완료 — **자격 모순 1건 미판정** |
| **AC_EXP04** | ⛔ **차단** | **3중 차단** — 좌표계 모순 · 재빌드 소스 부재 · 등록 0 키 |
| **AC_EXP05** | ✅ **eval 완주** | `qwen2.5-vl-3b` stage1 full FT + stage2(full·lora world-model·base) **전부 eval 완주** (2026-07-21). **데이터 쟁점 4건 미판정** |
| **AC_EXP06** | 🔄 **merge/업로드** | EXP05 비증강 Stage-2 대조군. `base` variant 완료·업로드, **world-model variant 학습 미착수** |
| **AC_EXP07** | 🧱 **데이터·인프라 완비** | `qwen2.5-vl-3b` 단독 stage1 world-modeling. 등록·YAML·빌더 완비, **0725 실데이터 빌드 완료(누출 0)**, **학습 이력 0** |
| **MC** | ⬜ 미착수 | 데이터·등록·YAML 완비, 자격 제한 없음. **프로덕션 코퍼스 아님** |
| **MB** | ⬜ 미사용 | 평가 전용. `on-MB*` 산출물 0 |

---

## ✅ EXP01 — 완료 (공백 1)

**완료**: `qwen3-vl-8b` (ratio37 · ratio73) + `qwen2.5-vl-7b` (ratio73) 로 stage1 LoRA → stage2 LoRA (`base` / `world-model`) 학습·평가 완료. 지표는 Notion `🧪 Experiments` DB 정본.

**남은 것**: `ratio55` 학습 — 평가 기본값이 `--exp01-ratio ratio55` 이니 주의 ([§4 CLI](../Implicit-World-Modeling/ARCHITECTURE.md#4-파이프라인-컴포넌트)).

**차단·쟁점**: ⚠️ **`ratio55` 는 학습된 적이 없다** (HF·로컬 모두 산출물 0). ratio sweep 3종 중 하나가 비어 있으므로 **"ratio 매트릭스 완주" 는 아직 거짓이다.**

---

## ✅ EXP02 — 완료 (diff loss v1)

**완료**: diff loss **v1**. `qwen3-vl-8b` · `qwen2.5-vl-7b` stage1 LoRA → stage2 LoRA 완료.

**남은 것**: 없음. EXP02 재실행이 필요해지면 그때 판단한다 (v1 동결 주의).

**차단·쟁점**: v1 은 EXP02 재현성 때문에 **의도적으로 동결**돼 있다 — 경계 비대칭 버그도 고치지 않는다. **v1 4파일 삭제 금지** ([§3 함정 10](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약)).

---

## ✅ EXP03 — 완료 (자격 모순 1건 미판정)

**완료**: `qwen3-vl-8b` · `qwen2.5-vl-7b` stage1 LoRA → stage2 LoRA 완료 (산출물은 HF. 로컬 `outputs/` 는 비어 있다 — 위 완료 판정 규칙 참조).

**남은 것**: 없음 (기존 HF 산출물의 **평가는 되고 재학습만 막힌다** — 아래 자격 모순).

**차단·쟁점**: ⚠️ **`qwen2.5-vl-7b` × EXP03 자격 모순** — HF 에 as-trained `ac-exp03-` 산출물이 있는데 현행 `eligible_models('AndroidControl_EXP03')` 는 Qwen3-VL 계열만 허용한다 → `require_model_eligible()` 이 **재학습을 막는다** (학습 당시엔 없던 가드). 열린 판정 2 참조. 커밋 YAML 은 재구성본이다 (아래 재현성 경고).

---

## ⛔ EXP04 — 차단 (3중)

**완료**: 데이터 파일과 stage1 YAML 은 디스크에 있다. **있다는 사실이 돌아간다는 뜻이 아니다** — `require_dataset_registered` 가 `llamafactory-cli` 진입 전에 죽인다 ([§7 함정 20](../Implicit-World-Modeling/ARCHITECTURE.md#7-중요한-운영-제약)).

**남은 것**: 선결 순서대로 — **좌표 규약 확정 → (원천 확보 후) 재빌드 → dataset_info 등록.** Stage 2 는 `_STAGE1_ONLY` 라 애초에 대상 아님. HF 에 EXP04 산출물 0.

**차단·쟁점**: 3중 차단 — **순서대로** 풀어야 한다:

1. **좌표계 모순 (선결)** — 디스크의 EXP04 데이터가 문서 전제(0–1000 정규화)를 **만족하지 않는다**. **버그인지 의도인지 아직 판정되지 않았다** — 어느 쪽으로도 단정하지 말 것. 이게 안 풀리면 아래 둘을 풀어도 **틀린 좌표계를 등록하게 된다**. 실측·근거는 [§2 EXP04 경고 블록](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정).
2. **재빌드 불가** — `mirror_experiment.py --experiment exp04` 의 원천 `data/AndroidControl/EXP04_stage1_{action,state}.jsonl` 이 **디스크에 없다** → **원천 확보가 물리적 선결**.
3. **등록 0 키** — `configs/lf_dataset/dataset_info.json` 에 `IWM-AC_EXP04_*` 키가 없다. 가드는 YAML 유무가 아니라 **등록 여부**를 본다 ([§3 함정 14](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약)).

```bash
# 등록 상태 재확인 (빈 목록이면 여전히 차단)
python -c "import json;d=json.load(open('configs/lf_dataset/dataset_info.json'));print(sorted(k for k in d if 'EXP04' in k))"
```

---

## ✅ EXP05 — eval 완주 (`qwen2.5-vl-3b`)

절대 픽셀 좌표 실험군. 자격 밖 모델은 **코드 가드가 막는다** — 매트릭스는 [§2 자격 매트릭스 · 함정 3](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정).

**완료**:
- **데이터 빌드** — 0711 수정본 + diff loss **v2** 가중, 등록 완료. 빌드 정본 [`scripts/build_exp05_data.py`](../Implicit-World-Modeling/scripts/build_exp05_data.py).
- **stage1 full FT eval 완주** — state F1 ep2 정점 **0.5963** / action ep3 **0.7292** (A100×2 에서 학습, base→ep 추이 확인). **현행 실험군 최초의 완주 full FT 경로**다 (나머지 완료 실험군은 전부 LoRA).
- **Stage 2 도입 (2026-07-15)** — drive stage2 3 jsonl(train 15000 / test_id 3000 / test_ood 3000)을 저장소 이미지 경로 관례로 변환(`myset/images/…` → `AndroidControl/images/episode_<6자리>_step_<S>.jpg`, `home.png` → `home.jpg`)해 3키 등록 + `_STAGE1_ONLY` 에서 제거 + stage2 YAML **12개** 생성. 빌드 정본 [`scripts/build_exp05_stage2_data.py`](../Implicit-World-Modeling/scripts/build_exp05_stage2_data.py).
- **stage2 full·lora world-model + base 전부 eval 완주** (2026-07-21, RTX5090×2). 재채점 `step_acc`: **full-wm ep3 0.6950** / lora-wm ep3 0.6570 / lora-base ep3 0.6627 / base 0.4053.
- ⚠️ **terminate 채점 버그 수정** (`_action_eval.py`, 커밋 `dd17426`) — GT `terminate` 가 xy no-field 채점서 누락돼 stage2 18.6% 오채점 → 재추론 없이 재채점.

**남은 것**: 데이터 쟁점 4건 판정 후 산출물 유효성 재확인. `qwen2.5-vl-7b` stage2 는 **사용자 지시로 스킵** (감시 스크립트 `exp05_guard` 가 7b 학습 진입 직전 kill — 0바이트 로그·GPU idle·프로세스 없음으로 확인, 실제 GPU 연산 없음).

**차단·쟁점**:
- ⚠️ **데이터 쟁점 4건 — 조병웅님 판정 대기 (본실험 전 선결)**: `wait` 액션 전량 퍼지 · train 축소 · action/state 키 대칭 붕괴 · **좌표 범위이탈**(OOD 평가셋 오염). 실측·상세는 [§3 EXP05 데이터 쟁점](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약).
- 분석: [`.claude/analysis/2026-07-21_00-32-13/`](../.claude/analysis/2026-07-21_00-32-13/README.md) — **동일 예산(LoRA)서 world modeling 사전학습 이득 미관찰** (single-run).

---

## 🔄 EXP06 — merge/업로드 (world-model variant 학습 미착수)

EXP05 의 **비증강(증강 X) Stage-2 대조군**. 좌표/budget/`--coord-mode xy` 규약을 EXP05 에서 승계하며 모델 자격도 EXP05 와 동일하게 **Qwen2.5-VL 계열 전용**이다 ([§2 자격 매트릭스 각주](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정)).

**완료**:
- **마이그레이션** (2026-07-18) — `AC_NOTAUG` 에서 표준 네이밍으로.
- **lf_registry 등록 완료** (2026-07-20) — `DATASET_MODEL_ELIGIBILITY`/`_STAGE2_ONLY`/`_LONG_CUTOFF_DS`/half-batch 편입. stage1 계보는 `stage1_hf_slug: "ac-exp05-"` override + 셸 `ds_stage1_source()` 로 **EXP05 stage1 체크포인트를 그대로 승계** (stage2 산출물 네이밍은 `ac-exp06-` 유지).
- **stage2 YAML 12종** — EXP05 stage2 매트릭스 완전 미러 (base/world-model-full/world-model-lora × `qwen2.5-vl-{3b,7b}` × stage2 {full,lora}).
- **`base` variant** — 학습 완료 (`qwen2.5-vl-3b` LoRA base ep1/2/3) · merged 3에폭 HF 업로드 (`SaFD-00/…ac-exp06-…epoch{1,2,3}`).

**남은 것**: **world-model variant (EXP05 stage1 full/lora 계승) 학습·평가 실행** — YAML 만 있고 학습 이력 0. `scripts/stage2_train.sh` 가 `ds_stage1_source` 로 EXP05 local merged 를 base 로 삼는다. eval 보류 (데이터 준비됨, 각 test 3000).

**차단·쟁점**: 없음 — 배선 완비, 실행만 남았다.

---

## 🧱 EXP07 — 데이터·인프라 완비 (학습 미착수)

`qwen2.5-vl-3b` **단독** stage1 world-modeling 실험군. 자격 밖 모델은 EXP05 와 동일하게 코드 가드가 막는다 ([§2 자격 매트릭스](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정)).

**스펙 요약**: 3B 전용 · **stage1** 1ep · `save_steps 0.25` (fractional 체크포인트 라벨 0.25/0.5/0.75/1) · train **50K** = state 40K + downstream 10K(이미지 제거) · LoRA **64/128** · diff v2 **1:0.2** (ADDED 1.0 / MODIFIED 1.0 / UNCHANGED 0.2, state 분량에 인라인). **stage2** 3ep · train **15K** · **merge O/X** 둘 다 지원 · rank 64 · thought 유사도 메트릭. 수치 정본은 [§2 하이퍼파라미터](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정) · [§6 메트릭](../Implicit-World-Modeling/ARCHITECTURE.md#6-메트릭), 데이터 계약은 [§3](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약).

**완료 (인프라 + 데이터)**:
- **등록** — 자격 `qwen2.5-vl-3b` 단독, `dataset_info` **8키 전부 자체 경로**(stage1 train + stage1 test_{id,ood}_{state,action} + stage2 train/test_id/test_ood — EXP05 포인터 없음), `_LONG_CUTOFF_DS` 편입, stage2 **merge X 변형(`world-model-adapter`)** 을 EXP07 한정 opt-in.
- **YAML 9종** — stage1 {full,lora} 2 + stage2_full 3 + stage2_lora 4. `gen_configs --check` 통과.
- **데이터 빌드 (2026-07-25 실데이터)** — 빌드 정본 [`scripts/build_exp07_data.py`](../Implicit-World-Modeling/scripts/build_exp07_data.py). 원천은 0725 myset 필터링본 3 파일(`data/AndroidControl/EXP07_{stage1_state,stage2,open_aug}.jsonl` — 2026-07-26 에 `AndroidControl_EXP07_src/` 에서 공유 원본 디렉토리로 이동·개명)이고 **EXP05 파생이 아니다**. train 2(stage1 50K / stage2 15K) + **자체 test 6종** 전부 실파일 — EXP05 심링크·포인터는 폐기됐다. **누출 0**: EXP05 test 의 `(episode, step)` 키를 재현해 test 를 굽고 그 union 을 train 두 풀에서 전량 제외 → `train ∩ test = 0` (빌더 `verify()` 가 fail-closed 검사). 두 downstream 풀(stage1 10K / stage2 15K)도 **비중복**. 행수·분포는 sidecar 에서 읽는다: `cat data/AndroidControl_EXP07/stage1_train.jsonl.meta.json`.
- **thought 유사도 메트릭 배선** — `stage2_eval.sh` 가 action 채점 직후 자동 hook 으로 산출 ([§6](../Implicit-World-Modeling/ARCHITECTURE.md#6-메트릭)).

**남은 것**: **학습 실행** — 데이터·인프라 완비, **학습 이력 0**. 2026-07-26 에 `DRY_RUN=1` 로 stage1(full/lora)·stage2(base/world-model/adapter)·merge 경로를 관통 검증했고, 채점기 3종(hungarian `--match-mode pos` / action `--coord-mode xy` / thought)은 EXP07 test 실데이터 300 쌍 **오라클**(GT 를 예측으로 그대로 투입)로 **실행 검증**(parse 100%, 전 지표 1.0)했다 — 검증된 것은 채점 로직 절반뿐이고, 실제 예측을 만드는 `vllm_infer.py` 는 GPU 를 요구해 **미실행**이다. 남은 것은 GPU 연산(학습 + vLLM 추론) 전부다.

**차단·쟁점**:
- **로컬 GPU 가 없다 (2026-07-26 현재, 학습을 막는 유일한 요인)**: RTX 5090 2 장을 **다른 사용자(`byeongung.cho`)의 sglang 서버 2 개**가 상시 점유 중이라 (PID 7318·7328, 33 시간 가동, 카드당 여유 **3.2/3.1 GiB**) qwen2.5-vl-3b LoRA 조차 올라가지 않는다 — `cutoff_len 24576` × vocab 151936 의 lm_head logits 만 bf16 로 ~7.5 GiB 다. ZeRO-3·CPU offload 로도 못 줄이고 `SMOKE=1` 은 `max_samples/max_steps` 만 줄여 cutoff 를 그대로 두므로 역시 OOM 이다. **남의 프로세스를 죽이지 말 것** — 카드가 비기를 기다리거나(협의) 원격으로 가야 한다.
- 재빌드는 `python scripts/build_exp07_data.py --seed 7` 로 한다 — `--source-dir` 기본값이 **공유 원본 디렉토리 `data/AndroidControl/`** 이라 보통 넘길 필요가 없다 (`W_UNCHANGED=0.2`·metric v2 는 빌더 불변식으로 고정). 소스가 바뀌면 test 도 함께 다시 구워지므로 **누출 0 불변식은 빌더가 매번 재검사**한다. **`--revision` 을 반드시 고정하라** (sidecar 의 `revision_resolved`, 현재 `66285546…`) — 기본값 `None` 은 Hub HEAD 를 다시 해석하므로 토크나이저가 바뀌면 `token_weights` 가 조용히 달라진다.
- ⚠️ **다음 재빌드에서 sidecar 는 바뀐다 — 손상이 아니다.** train/test 8 파일은 2026-07-26 에 bit-identical 재현성이 증명됐지만, sidecar 의 `exp07_sampling.source_dir` 에는 **빌드 당시 경로**가 박힌다. 디스크의 현 sidecar 는 아직 이동 전 경로(`AndroidControl_EXP07_src`)를 담고 있으므로, 재빌드하면 그 문자열 하나가 갱신된 diff 가 나온다. 파일 내용·md5 와 무관하다.
- ~~thought eval 의존성 (`sentence-transformers`·`sacrebleu`) 미충족~~ → **2026-07-26 해소**. conda env 에 `pip install --no-deps sacrebleu portalocker tabulate colorama` 로 설치(sacrebleu 2.6.0 · portalocker 3.2.0 · colorama 0.4.6, tabulate 는 이미 충족). `pip freeze` 차분 추가 3 줄뿐 — numpy 2.2.6 / torch 2.8.0+cu128 / vLLM 0.11.0 불변. EXP07 test 300 쌍 오라클에서 cosine/rouge_l/**bleu** 모두 산출 확인, conda `pytest tests` 691 passed / 9 skipped (조건부 skip 0). **env 재구성 시에도 반드시 `--no-deps`** — 평범한 `pip install` 은 numpy 를 올려 torch/vLLM 을 깨뜨릴 수 있다.

---

## ⬜ MC — 미착수 (차단 아님)

**완료**: 배선·정적 관통 확인 (jsonl 스키마 + 이미지 경로 해석). 레지스트리·`dataset_info`·`configs/train/IWM-MC/`·`split_data.py --dataset MC` 전부 존재.

**남은 것**: 원격 GPU 박스에서 `stage1_train.sh --dataset MC` 를 실제로 한 번 실행 (핸드오프만, **사용자 결정 2026-07-14 로 보류**). `--stage1-ratio` 기본값 0.95 를 164 행에 적용하면 test 가 9개뿐이라 실제 학습 전 비율을 다시 정해야 한다.

**차단·쟁점**: 현재 `data/MonkeyCollection/` 는 **프로덕션 코퍼스가 아니다** (Monkey-Collector 실험 잔여물 164 examples, 교차-앱 병합 오염). **이 데이터로 낸 학습 결과를 코퍼스 품질의 근거로 쓰지 마라** ([§3 MC 데이터 상태](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약)).

---

## 열린 판정 (착수 전에 사람이 결정해야 하는 것)

1. **EXP04 좌표계 — 버그인가 스펙인가.** EXP04 차단 해제의 선결 조건. → [§2 경고 블록](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정)
2. **`qwen2.5-vl-7b` × EXP03 자격 모순** — HF 에 **as-trained `ac-exp03-` 산출물이 있는데**, 현행 `eligible_models('AndroidControl_EXP03')` 는 Qwen3-VL 계열만 허용한다 → `require_model_eligible()` 이 **재현을 막는다**. 학습 당시엔 없던 가드다.
   가드는 **학습 entry 에만** 걸리므로 (eval 은 검사하지 않는다) 기존 HF 산출물의 **평가는 되고 재학습만 막힌다**.
   판정 필요: 그 산출물을 (a) 좌표 규약 불일치로 폐기할지, (b) 자격을 넓힐지. **어느 쪽도 아직 정해지지 않았다** — 성급히 "깨진 모델" 로 단정하지 말 것. 메커니즘은 [§2 좌표 규약 · 자격 매트릭스](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정).
   ```bash
   python -c "from implicit_world_modeling.lf_registry import eligible_models as e; print(e('AndroidControl_EXP03'))"
   ```
3. **EXP05 데이터 쟁점 4건** — 조병웅님. → [§3](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약)
4. **EXP05 7:3 분할 비율** — 교수님 최종 확인.
5. **`without_open_app` 필터가 전 실험군에서 무동작** (2026-07-13 실측) — `_hungarian_eval.py::_gt_action_type` 이 GT 의 action type 을 뽑지 못해 **항상 `None`** 을 돌려주고, 그 결과 `--exclude-action open_app` 이 **0 행을 drop** 한다 (EXP01 state test 3,000 행 전수: non-None 0건). 산출되는 `on-{DS}-without-open_app/` 메트릭은 **정규 메트릭과 동일한 수치**라 "open_app 제외 성능" 으로 읽으면 거짓 결론이 된다. **EXP05 의 `Action:` 마커만 고치는 것으로는 해결되지 않는다** — EXP01–EXP04 도 똑같이 0 행 drop 이다. 채점 규약을 학습·평가가 공유하므로 수정 전 확인 필요.
6. **`extract_elements` 의 aria-label 누락** — 포함 조건이 `description` 단독이라 `<div aria-label="...">` 류가 element 집합에서 빠진다. 포함시키면 element 집합이 커져 **pos 메트릭 값이 바뀐다**(= 채점 기준 변경) → 확인 필요. 근거 주석은 `scripts/_hungarian_eval.py` 의 `is_described` 위에 있다.

---

## 미착수 (차단 아님 — 그냥 안 돌렸다)

- **`qwen3-vl-4b`** — 2026-07-13 레지스트리 복원으로 EXP01–EXP04 **자격만** 생겼다. **학습·평가 이력 0.** 위 EXP01–03 완료 표시는 전부 `qwen3-vl-8b`·`qwen2.5-vl-7b` 기준이다. EXP05/06/07 은 자격 밖 ([§2 모델 레지스트리](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정)).
- **MB (MobiBench)** — 평가 전용. 등록돼 있으나 `on-MB*` eval 산출물 0.
- **Full FT 경로** — **EXP05 3B stage1 full FT 가 A100×2 에서 완주해 eval 까지 마쳤다** (현행 실험군 최초의 완주 full FT 경로 — 위 EXP05 참조). 로컬 RTX5090 시도는 OOM 으로 죽었다. 나머지 완료 실험군(EXP01–03)은 전부 LoRA 다.

---

## 마일스톤

- [x] 문서 트리오 정비 (README · ARCHITECTURE · AGENTS) + SSoT 재배치
- [x] 2-stage 파이프라인 자동화 (`scripts/stage{1,2}_{train,merge,eval}.sh`)
- [x] 학습 설정 정본화 — YAML 생성기(`gen_configs`) + 커밋된 `dataset_info.json` + 코드 가드(자격·등록)
- [ ] **실험 매트릭스 완주** (모델 × 데이터셋 × {base / stage2 / stage1+stage2}) — EXP04 차단, EXP06 world-model·EXP07 학습 미착수, ratio55·`qwen3-vl-4b`·MC 공백
- [ ] 결과 종합 및 논문화 (AAAI/ICLR 2027 트랙)
- [ ] (추후) Obsidian 동기화 — Vault 있는 환경에서 `/project-sync init` 재실행

---

## 재현성 경고

> ⚠️ **EXP03/EXP04 의 커밋 YAML 은 `# [reconstructed 2026-07-13]` 재구성본이다 — as-trained 가 아니다** (원본 소실). 위 EXP03 "완료" 산출물이 이 YAML 로 학습됐다는 보장이 없다. 상세 [§7 함정 20](../Implicit-World-Modeling/ARCHITECTURE.md#7-중요한-운영-제약).
> 재구성 이후 **실제 학습으로 확인된 경로는 EXP02 3B LoRA 스모크 하나뿐이다.**

```bash
python -m implicit_world_modeling.gen_configs --check   # YAML 정합 (byte 대조 + orphan 검출)
grep -rl reconstructed configs/train                    # 재구성본 식별
```

<!-- project-sync: task/계획 진척 시 - [ ] / - [x] 상태와 항목만 갱신. -->
