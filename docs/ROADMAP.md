# Roadmap — 상태

> **이 문서는 상태만 싣는다.** 실험군별로 무엇이 완료·진행·차단인가, 차단이면 **정확히 어디서 막히는가**.
> 수치·표·근거·메커니즘은 전부 [`ARCHITECTURE.md`](../Implicit-World-Modeling/ARCHITECTURE.md) 가 정본이고 여기서는 링크만 한다.
> 실험 **결과 지표**는 Notion `🧪 Experiments` DB 가 정본이다 (메트릭 정의는 [§6 메트릭](../Implicit-World-Modeling/ARCHITECTURE.md#6-메트릭)).

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
`ac-2-` slug 와 `IWM-AC_2_*` 등록 키는 **구 스키마 사문화분**이다 — 현행 실험군이 아니다 ([§3 LF 등록](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약)).

---

## 실험군 상태

| 실험군 | 상태 | 한 줄 요약 |
|---|---|---|
| **AC_EXP01** | ✅ 완료 (공백 1) | stage1+stage2 완료. **ratio55 만 미학습** |
| **AC_EXP02** | ✅ 완료 | diff loss **v1**. stage1+stage2 완료 |
| **AC_EXP03** | ✅ 완료 | stage1+stage2 완료 — **단 자격 모순 1건 미판정** |
| **AC_EXP04** | ⛔ **차단** | **3중 차단** — 좌표계 모순 · 재빌드 소스 부재 · 등록 0 키 |
| **AC_EXP05** | 🔄 **학습 중** | stage1 full FT **A100×2 진행 중** (2026-07-14~) · **stage2 도입** (2026-07-15 — YAML 12 + 등록 3키, `qwen2.5-vl-3b` LoRA E2E loop 진입). **데이터 쟁점 4건 미판정 · 완주 산출물 0** |
| **MC** | ⬜ 미착수 | 데이터·등록·YAML 다 있고 자격 제한 없음 — 그냥 안 돌렸다 |
| **MB** | ⬜ 미사용 | 평가 전용. `on-MB*` 산출물 0 |

---

## ⛔ EXP04 — 차단 (3중)

**돌리려 하면 `require_dataset_registered` 가 `llamafactory-cli` 진입 전에 죽인다.** YAML 이 있어도 돌지 않는다 ([§7 함정 20](../Implicit-World-Modeling/ARCHITECTURE.md#7-중요한-운영-제약)).

차단 지점 셋 — **순서대로** 풀어야 한다:

1. **좌표계 모순 (선결)** — 디스크의 EXP04 데이터가 문서 전제(0–1000 정규화)를 **만족하지 않는다**. 실측값과 판정 보류 근거는 [§2 EXP04 경고 블록](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정).
   **버그인지 의도인지 아직 판정되지 않았다** — 어느 쪽으로도 단정하지 말 것. 이게 안 풀리면 아래 둘을 풀어도 의미가 없다 (**틀린 좌표계를 등록하게 된다**).
2. **재빌드 불가** — `mirror_experiment.py --experiment exp04` 의 원천 `implicit-world-modeling_stage1_{action,state}_xy_prompt-enhanced.jsonl` 이 **디스크에 없다**. 좌표계를 고쳐 재빌드하려 해도 소스가 없다 → **원천 확보가 물리적 선결**.
3. **등록 0 키** — `configs/lf_dataset/dataset_info.json` 에 `IWM-AC_EXP04_*` 키가 없다. 가드는 YAML 유무가 아니라 **등록 여부**를 본다 ([§3 함정 14](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약)).

```bash
# 등록 상태 재확인 (빈 목록이면 여전히 차단)
python -c "import json;d=json.load(open('configs/lf_dataset/dataset_info.json'));print(sorted(k for k in d if 'EXP04' in k))"
```

데이터 파일과 stage1 YAML 은 디스크에 있다 — **있다는 사실이 돌아간다는 뜻이 아니다.** Stage 2 는 `_STAGE1_ONLY` 라 애초에 대상 아님. HF 에 EXP04 산출물 0.

---

## 🔄 EXP05 — stage1 학습 중 (A100×2, 2026-07-14~)

절대 픽셀 좌표 실험군. 자격 밖 모델은 **코드 가드가 막는다** — 매트릭스는 [§2 자격 매트릭스 · 함정 3](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정).

- [x] **데이터 빌드 완료** — 0711 수정본 + diff loss **v2** 가중. 빌드 정본 [`scripts/build_exp05_data.py`](../Implicit-World-Modeling/scripts/build_exp05_data.py). 등록도 되어 있다.
- [ ] **데이터 쟁점 4건 — 조병웅님 판정 대기 (본실험 전 선결)**: `wait` 액션 전량 퍼지 · train 축소 · action/state 키 대칭 붕괴 · **좌표 범위이탈**(OOD 평가셋 오염). 실측·상세는 [§3 EXP05 데이터 쟁점](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약).
  ⚠️ **아래 stage1 학습은 이 4건이 미판정인 채로 진행 중이다** — 판정 결과에 따라 산출물의 유효성이 달라질 수 있다.
- [ ] **학습 — A100×2 에서 stage1 full FT 진행 중** (`qwen2.5-vl-3b`, 2026-07-14 재시작). GPU 정책이 80GB × 3-4B 에서 offload 를 끄면서 **138 s/step · ETA 약 3.3 일**이 됐다 (재시작 시점 확인값: step 15/2094, loss 0.1735, OOM 없음). 정책·실측 근거는 [§2 GPU 정책 · 함정 7](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정) · [§7 함정 19](../Implicit-World-Modeling/ARCHITECTURE.md#7-중요한-운영-제약).
  - **아직 완주하지 않았다** — 체크포인트·HF 산출물 **0**. 메모리 여유가 **~7 GB** 뿐이라 **OOM 가능성이 남아 있다** (감시 중).
  - **로컬 2×RTX5090 은 여전히 불가** (OOM + 비현실적 소요시간, [§7 함정 19](../Implicit-World-Modeling/ARCHITECTURE.md#7-중요한-운영-제약)). 로컬에 남은 건 OOM 로그(`trainer_log.jsonl.oom_0711.bak`) 뿐이다.
- [ ] **원격 실행 경로 확보** — 제출 스펙(`scripts/remote_launch.sh` + `configs/remote/run.template.yaml`)은 저장소에 있으나 **UNVALIDATED (실행 이력 0)** ([§4](../Implicit-World-Modeling/ARCHITECTURE.md#4-파이프라인-컴포넌트)). org/project/cluster + 데이터 업로드 방식 미정. **위 A100×2 학습은 원격 제출 경로가 아니라 A100 머신에서 `scripts/stage1_train.sh` 를 직접 호출한 것이다** — 제출 스펙 검증과는 무관하다.
- [ ] **평가** — xy 좌표 채점은 구현·배선 완료 (EXP05 일 때만 opt-in). 규칙과 함정은 [§6 xy 좌표 스페이스 채점](../Implicit-World-Modeling/ARCHITECTURE.md#6-메트릭).
- [x] **Stage 2 도입 (2026-07-15)** — drive-download stage2 3개 jsonl(train 15000 / test_id 3000 / test_ood 3000)을 저장소 이미지 경로 관례로 변환(`myset/images/…` → `AndroidControl/images/episode_<6자리>_step_<S>.jpg`, `home.png` → `home.jpg` 주입)해 `dataset_info.json` 에 3키 등록하고, `_STAGE1_ONLY` 에서 EXP05 를 제거해 stage2 YAML **12개**(`stage2_full` 6 + `stage2_lora` 6, `qwen2.5-vl-{3b,7b}` × {base, world-model-full, world-model-lora})를 정식 생성했다. 빌드 정본 [`scripts/build_exp05_stage2_data.py`](../Implicit-World-Modeling/scripts/build_exp05_stage2_data.py). `qwen2.5-vl-3b` LoRA **E2E 학습 loop 진입 확인**(Num examples 15,000 · 3 epoch · 705 step · trainable 59.8M · finite loss) — **아직 완주 전, 체크포인트 0**. 상세는 [devlog 2026-07-15](../.claude/devlog/2026-07-15_08-51-11_exp05-stage2-apply.md). 산출 데이터·`home.jpg` 는 관례상 gitignored.

HF 에 EXP05 산출물 0 — stage1·stage2 모두 아직 완주 전이다 (stage2 는 2026-07-15 도입, 위 참조).

---

## ✅ 완료 — EXP01 / EXP02 / EXP03

지표는 Notion `🧪 Experiments` DB 가 정본. 학습된 변형은 위 HF 커맨드로 확인한다.

- **EXP01** — `qwen3-vl-8b` (ratio37 · ratio73) + `qwen2.5-vl-7b` (ratio73) 로 stage1 LoRA → stage2 LoRA (`base` / `world-model`) 학습·평가 완료.
  - ⚠️ **`ratio55` 는 학습된 적이 없다** (HF·로컬 모두 산출물 0). ratio sweep 3종 중 하나가 비어 있으므로 **"ratio 매트릭스 완주" 는 아직 거짓이다.** 평가 기본값이 `--exp01-ratio ratio55` 라는 점에 주의 ([§4 CLI](../Implicit-World-Modeling/ARCHITECTURE.md#4-파이프라인-컴포넌트)).
- **EXP02** — diff loss **v1**. `qwen3-vl-8b` · `qwen2.5-vl-7b` stage1 LoRA → stage2 LoRA 완료.
  - v1 은 EXP02 재현성 때문에 **의도적으로 동결**돼 있다 — 경계 비대칭 버그도 고치지 않는다. **v1 4파일 삭제 금지** ([§3 함정 10](../Implicit-World-Modeling/ARCHITECTURE.md#3-데이터와-설정-계약)). EXP02 재실행이 필요해지면 그때 판단한다.
- **EXP03** — `qwen3-vl-8b` · `qwen2.5-vl-7b` stage1 LoRA → stage2 LoRA 완료 (산출물은 HF. 로컬 `outputs/` 는 비어 있다 — 위 판정 규칙 참조).

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

- **`qwen3-vl-4b`** — 2026-07-13 레지스트리 복원으로 EXP01–EXP04 **자격만** 생겼다. **학습·평가 이력 0.** 위 EXP01–03 완료 표시는 전부 `qwen3-vl-8b`·`qwen2.5-vl-7b` 기준이다. EXP05 는 자격 밖 ([§2 모델 레지스트리](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정)).
- **MC (MonkeyCollection)** — stage1 전용. 데이터·등록·YAML 다 있고 자격 제한 없음. 산출물 0.
- **MB (MobiBench)** — 평가 전용. 등록돼 있으나 `on-MB*` eval 산출물 0.
- **Full FT 경로** — 현행 실험군(EXP01–05)에서 **완주한** stage1 full FT 는 아직 **없다** (완료된 실험군은 전부 LoRA). EXP05 3B full FT 가 **A100×2 에서 처음으로 진행 중**이다 (위 EXP05 참조). 로컬 RTX5090 시도는 OOM 으로 죽었다.

---

## 마일스톤

- [x] 문서 트리오 정비 (README · ARCHITECTURE · AGENTS) + SSoT 재배치
- [x] 2-stage 파이프라인 자동화 (`scripts/stage{1,2}_{train,merge,eval}.sh`)
- [x] 학습 설정 정본화 — YAML 생성기(`gen_configs`) + 커밋된 `dataset_info.json` + 코드 가드(자격·등록)
- [ ] **실험 매트릭스 완주** (모델 × 데이터셋 × {base / stage2 / stage1+stage2}) — EXP04 차단, EXP05 stage1 학습 중, ratio55·`qwen3-vl-4b`·MC 공백
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
