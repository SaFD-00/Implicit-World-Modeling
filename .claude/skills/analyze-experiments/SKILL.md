---
name: analyze-experiments
description: IWM 2-stage 실험(base vs stage2 vs stage1+stage2)의 eval 결과를 **데이터 구성→Stage1 학습→Stage2 학습→평가·스코어링→ID/OOD 일반화 5단계**로 나눠 실제 metric 파일·로그·코드로 진단하고, 각 실패모드에 근거 verdict 를 부착해 "world model(Stage1)이 action prediction(Stage2)을 돕는가"를 정량 분해, 우선순위 개선책을 산출하는 분석 — 핵심 SSoT 는 eval leaf dir 의 `hungarian_metrics.json`(state·avg_hungarian_f1)·`action_metrics.json`(action·step_accuracy) **직접 카운트**(자동 요약값 불신). `.claude/issues/<날짜_시간>/` 스냅샷 + `/workflow:project-sync` 위임으로 Obsidian(날짜-우선)·Notion(Experiments DB 1행) 미러. --exp 대상 실험, --stage 단계 부분집합, --no-obsidian/--no-notion 미러 생략.
argument-hint: "[--exp EXP01,EXP02,EXP03,EXP04|all] [--model qwen3-vl-8b|qwen2.5-vl-7b] [--eval-ds AC_EXPnn,MB,MC] [--stage data|stage1|stage2|eval|idood|all] [--no-obsidian] [--no-notion] [--no-workflow]"
---

# Analyze Experiments (IWM)

IWM 2-stage 파이프라인의 **핵심 가설** — *World Modeling(Stage 1)을 먼저 학습하면 Action Prediction(Stage 2)이
좋아지는가* — 을, `base` vs `stage2` vs `stage1+stage2` 세 변형의 eval metric 으로 정량 분해하고, 어디서
격차가 생기는지를 **데이터→Stage1→Stage2→평가→ID/OOD 5단계**로 진단해 우선순위 개선책을 산출하는 분석 스킬.
산출물은 `.claude/issues/<날짜_시간>/` 스냅샷 + Obsidian·Notion 미러(둘 다 `/workflow:project-sync` 위임).

> 이 스킬은 **분석·권고 문서만 생성**한다 — 코드·학습 스크립트는 바꾸지 않는다(실제 수정은 별도 신규 이슈로 개시).
> ⛔ **기존 수정 로그 금지**: 과거 실험의 수정·적용·revert 이력을 산출 문서에 삽입하지 않는다. 진단은 오직
> `--exp` 대상 실험의 metric 데이터로만 수행한다.
> 참고: 이 스킬은 MobileGPT-V2 의 `analyze-exploration`(Explore 5단계 진단)을 IWM 실험-매트릭스 도메인에 맞춰 옮긴 것.

## ⛔ 최우선 원칙 — SSoT 게이트 (metric 파일 직접 카운트)

**모든 정량 주장은 eval leaf dir 의 metric JSON 을 직접 읽어 나온 값이어야 한다.** 자동 요약 도구의
per-cell 값·과거 문서의 stale 값을 그대로 베끼지 말 것. SSoT:

- **Stage 1 / state(World Model)** → `hungarian_metrics.json`. 헤드라인 = **`avg_hungarian_f1`**(+ `exact_match_rate`·`avg_bleu`·`avg_rouge_l`). 정의: `scripts/_hungarian_eval.py`.
- **Stage 2 / action(Action Prediction)** → `action_metrics.json`. 헤드라인 = **`step_accuracy`**(+ `type_accuracy`·`macro_step_accuracy`·`parse_rate`·`cond_*_acc`). 정의: `scripts/_action_eval.py`.
- **AC(AndroidControl) eval 은 3-섹션** `overall`/`in_domain`/`out_of_domain`. MC/MB 는 flat(단일). ID/OOD 표는 `in_domain`/`out_of_domain` 섹션에서 뽑는다.
- **집계 로직은 `scripts/eval_viewer.py` 를 재사용**: `load_metrics()`(파일·섹션 → flat dict 병합), `discover_variants()`(변형 경로 자동탐지), `EVAL_DATASETS`(어느 metric 파일·키를 어느 셀에 쓰나). `eval_viewer.py --include <EXP>:<MODEL> …` 로 표를 재유도하고, `outputs/{DS}/eval/.../pairs_summary.md`·`outputs/_compare/stage{N}_eval/` 를 교차검산.
- **metric 키 폐쇄집합**(표 헤더): `STATE_METRIC_KEYS`·`ACTION_METRIC_KEYS`(`eval_viewer.py` 상단 정의)만 쓴다. 임의 키 신설 금지.

> ⚠️ **outputs 는 학습 머신(Linux `/data/seungwoo/Implicit-World-Modeling`)에서 채워진다.** 이 checkout 이 macOS 이면
> `outputs/` 가 **없을 수 있다** — 그러면 metric 카운트는 "N/A(미materialize)"로 표기하고 `eval_viewer.py` 처럼 **조용히 skip**,
> 그 사실을 산출 문서에 명시한다(없는 값을 지어내지 말 것). 코드·스크립트·config 진단은 outputs 없이도 가능.

## ⛔ 최우선 원칙 — 내부 비교 (No External Baseline)

**IWM 의 비교 기준은 하드코딩된 외부 baseline 이 아니라 내부 변형 대조다.** 진단·개선책은 아래 대조축으로만
근거화한다(외부 시스템 숫자를 끌어와 근거 삼지 말 것 — `.claude/references/`(LLM-Explorer·mobilevlm)는 설계 참고용 코드지 metric 표가 아니다):

1. **변형 대조** — `base`(zero-shot) vs `stage2`(base→Stage2 SFT) vs `stage1+stage2`(WorldModel→Stage2 SFT). **핵심 가설 = stage1+stage2 − stage2 의 부호·크기.**
2. **ID vs OOD** — 매 AC eval 의 `in_domain` vs `out_of_domain` 격차(app-partition split).
3. **cross-dataset** — 한 모델을 자기 EXP + MB + MC 로 평가(`on-{EVAL_DS}` 차원).
4. **EXP vs EXP** — EXP01(ratio-mix) / EXP02(diff-loss) / EXP03(coordinate rep) / EXP04(prompt-upgrade, Stage1-only)의 처치 효과.
5. **without_open_app** — Stage1 state 의 `on-{DS}-without-open_app/`(trivially-easy `open_app` GT 제거 후 재스코어) vs 일반 → metric 팽창 격리.

각 실패모드·개선책에 **verdict**: `hypothesis-bearing`(가설 대조축) / `id-ood-gap` / `data-artifact` / `train-config` / `eval-artifact` / `single-cell(→일반화 필요)` / `factually-wrong`.

## Options

`$ARGUMENTS` 파싱:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--exp` | `all` | 분석 대상 실험. `EXP01`/`EXP02`/`EXP03`/`EXP04`(쉼표구분) 또는 `all`. EXP04·MC 는 Stage1-only. |
| `--model` | (전체) | `qwen3-vl-8b`(Qwen3-VL-8B-Instruct) / `qwen2.5-vl-7b`(Qwen2.5-VL-7B-Instruct). 미지정 시 존재하는 전 모델. |
| `--eval-ds` | (해당 EXP 전체) | 평가 데이터셋 부분집합(`AC_EXP0n`,`MB`,`MC`). `on-{EVAL_DS}` 셀만 분석. |
| `--stage` | `all` | 진단 단계 부분집합. `data`(S1) / `stage1`(S2) / `stage2`(S3) / `eval`(S4) / `idood`(S5) / `all`. |
| `--no-obsidian` | false | Obsidian 미러 생략(repo 스냅샷만). |
| `--no-notion` | false | Notion 미러 생략(기본은 project-sync 경유 Experiments DB 1행). |
| `--no-workflow` | false | 멀티에이전트 워크플로 없이 인라인 분석(가벼운 점검용). 기본은 워크플로로 적대적 검증까지. |

## 실험 매트릭스 (참조)

`scripts/_common.sh`·README·ROADMAP 기준:

- **Models**: `qwen3-vl-8b`(template `qwen3_vl_nothink`) · `qwen2.5-vl-7b`(template `qwen2_vl`).
- **Datasets**: 학습 = `AC_EXP01`·`AC_EXP02`·`AC_EXP03`·`AC_EXP04`·`MC`; eval-only 벤치마크 = `MB`(MobiBench). `AC` 자체는 source-only.
  - **EXP01** = state:action ratio-mix(`_ratio37/55/73`, 별도 weight) · Stage1+2 · ID/OOD.
  - **EXP02** = EXP01 ratio73 + **Stage1 diff-loss**(token-weighted SFT, `scripts/diff_loss/`).
  - **EXP03** = EXP01 ratio73 을 **coordinate(point) 표현**으로 mirror(`mirror_experiment.py --experiment exp03`, `cutoff_len 24576`).
  - **EXP04** = EXP03 + **Stage1 prompt-upgrade**(swipe action space) · **Stage1-only(LoRA)**, Stage2 held.
- **변형 경로**(`discover_variants`): `base` · `full_base` · `lora_base` · `full_world-model` · `lora_world-model`
  (stage2 world 변형은 Stage1 lineage `_from_{mode}-ep{E1}` 부착) × epoch 1/2/3.
- **eval 출력 경로 문법**(`stage{1,2}_eval.sh`):
  `outputs/{OUT_DS}/eval/{MODEL}{SFX}/stage{1,2}_eval/{variant}[/epoch-{E}]/on-{EVAL_DS}[-state|-action][-without-open_app]/`
  (`OUT_DS` = `AndroidControl_EXP0n`|`MonkeyCollection`, `SFX` = `_ratio37/55/73`(EXP01만) 또는 ``). metric JSON 존재 = 그 셀 완료(skip-marker).

## 격차 분해 — 5단계

진단을 학습→평가 파이프라인 단계로 나눈다. 각 실패모드에 위 verdict 를 부착. 실패모드 코드는 단계 풀네임 기반
(`DATA-N`/`STAGE1-N`/`STAGE2-N`/`EVAL-N`/`IDOOD-N`).

### 🧱 Stage 1 — DATA 구성 *(architectural → 전 셀 공통)*

코드: `scripts/filter_long_samples.py`(cutoff_len 초과 drop) · `split_data.py`(EXP01 ratio-mix + ID/OOD app-partition) ·
`mirror_experiment.py --experiment exp03/exp04`(coordinate·prompt mirror). **목적: 학습·평가 데이터가 무결·정렬됐나.**

- **DATA-1 sample drop** — cutoff 초과·`(episode,step)` 키 누락으로 표본 소실(EXP03 ~0.8–1.7%, EXP04 ~0.67%). drop 카운트·비율을 EXP별로.
- **DATA-2 row-count mismatch** — EXP01/02/03 간 행수 불일치로 cross-compare 붕괴(`eval_viewer.py` SystemExit line 443). 어느 쌍에서 발생하는지.
- **DATA-3 split leakage / OOD 오분류** — `episodes_meta.jsonl` `primary_app` → ID/OOD 버킷 배정 오류(같은 앱이 양쪽에). 앱-파티션 정합성.
- verdict 가이드: DATA-* 는 구성상 전 셀 공통이라 architectural; drop 편차가 특정 EXP 처치와 얽히면 `data-artifact`.

### 🌍 Stage 2 — STAGE1 학습(World Model) *(변형·epoch 대조)*

코드: `scripts/stage1_train.sh` → `stage1_merge.sh`(epoch별 local merge + HF push), diff-loss = `scripts/diff_loss/`.
**목적: World Model SFT 가 실제로 next-UI-XML 예측을 학습했나(state metric).**

- **STAGE1-1 world-model 미학습/열화** — `full_world-model`/`lora_world-model` 의 state `avg_hungarian_f1`·`exact_match_rate` 가 `base` 대비 개선 없음/악화. epoch 1/2/3 추세.
- **STAGE1-2 diff-loss 효과 부재(EXP02)** — token-weighted SFT 가 state metric 을 못 올림(EXP02 vs EXP01 ratio73 state 대조).
- **STAGE1-3 coordinate/prompt 처치 부작용(EXP03/04)** — 표현 변경이 state 학습을 해침(EXP03/04 vs EXP01 state; EXP04 는 Stage1-only 라 여기서 판정).
- verdict 가이드: 처치 효과는 EXP-vs-EXP 대조에서 부호가 재현될 때 `hypothesis-bearing`.

### 🎯 Stage 3 — STAGE2 학습(Action Prediction) *(핵심 가설)*

코드: `scripts/stage2_train.sh` → `stage2_merge.sh`. 변형: `base` · `{full,lora}_base` · `{full,lora}_world-model`(Stage1 lineage).
**목적: World Model 선학습(stage1+stage2)이 action `step_accuracy` 를 base→stage2 대비 더 올리나 — 논문의 핵심 주장.**

- **STAGE2-1 world-model 이득 부재/음수(핵심)** — `stage1+stage2`(world-model 변형) `step_accuracy` − `stage2`(base 변형) `step_accuracy` ≤ 0. **가설 반증 신호.** EXP·model·eval-DS별로 부호·크기 표.
- **STAGE2-2 base 대비 SFT 이득 미미** — `stage2` − `base` 자체가 작음(Stage2 SFT 효용 의심).
- **STAGE2-3 lora vs full 격차** — LoRA 변형이 full 대비 크게 뒤처짐(용량 병목).
- **STAGE2-4 lineage epoch 민감** — `_from_{mode}-ep{E1}` Stage1 epoch 선택에 step_accuracy 가 과민.
- verdict 가이드: STAGE2-1 은 반드시 **여러 EXP·model·eval-DS 에서 부호가 일관**해야 가설-bearing; 단일 셀은 `single-cell(→일반화 필요)`.

### 📏 Stage 4 — EVAL·스코어링 *(measurement → 전 셀 공통)*

코드: `scripts/stage{1,2}_eval.sh` → `_hungarian_eval.py`(state) / `_action_eval.py`(action), `vllm_infer.py`.
**목적: metric 자체가 신뢰 가능한가(파싱·완결·팽창).**

- **EVAL-1 parse_rate < 1** — action JSON 파싱 실패로 `step_accuracy` 저평가(모델 산출 형식 문제 vs 채점기 문제 구분). 어느 셀에서 몇 %.
- **EVAL-2 missing metric JSON** — 셀이 아예 안 돌아 표에 구멍(skip-marker 부재). 매트릭스 커버리지 표로.
- **EVAL-3 open_app 팽창** — 일반 vs `without-open_app` state metric 격차가 큼 → trivially-easy 액션이 헤드라인을 부풀림(state 한정; action 은 woa 미지원).
- **EVAL-4 predict_results 스케일/키 혼동** — `predict_bleu-4`/`predict_rouge-l`(×100 스케일) 과 hungarian 키 혼용 방지.
- verdict 가이드: EVAL-* 는 채점기 코드 차원이라 measurement; 특정 EXP 처치와 무관하게 재현되면 전 셀 공통.

### 🔀 Stage 5 — ID/OOD 일반화 *(generalization → 핵심 진단축)*

코드: split = `split_data.py` app-partition, 3-섹션 metric = `_hungarian_eval.py`/`_action_eval.py` `overall`/`in_domain`/`out_of_domain`.
**목적: 학습이 미학습 앱(OOD)으로 일반화되나 — ID−OOD 격차와 그 변형 의존성.**

- **IDOOD-1 큰 ID−OOD 격차** — `in_domain` ≫ `out_of_domain`(과적합). EXP·변형별 격차 표.
- **IDOOD-2 world-model 이 OOD 를 더 돕나** — `stage1+stage2` 의 OOD 이득이 `stage2` 보다 큰지(가설의 강한 형태: WM 이 일반화에 기여). OOD 한정 STAGE2-1 재계산.
- **IDOOD-3 EXP 처치의 OOD 편향** — coordinate(EXP03)·prompt(EXP04)·diff-loss(EXP02)가 OOD 에 유·불리 편향.
- verdict 가이드: IDOOD-* 는 `in_domain`/`out_of_domain` 섹션 직접 카운트로만; 특정 앱 1개 사례는 illustrative.

## 개선책 산출 (S 시리즈)

원칙: **싸고 정보량 큰 것 먼저 → 데이터·설정 → 학습 레시피 → 근본 재설계.** 각 항목에 **`[stage:...]` 태그**
(`[data]`/`[stage1]`/`[stage2]`/`[eval]`/`[idood]`)와 verdict 를 부착. 표준 골격:

| 우선 | 분류 | stage | 예 |
|------|------|-------|----|
| P1 | 평가 | `[eval]` | parse_rate 회복(action JSON 포맷 가드) · missing 셀 재실행으로 매트릭스 구멍 메움 |
| P1 | 데이터 | `[data]` | ID/OOD split 누수 교정 · drop 편차 정합화(EXP간 row-count 일치) |
| P2 | 학습 | `[stage2]` | world-model lineage epoch 선택 · lora→full 승격(용량 병목) |
| P2 | 학습 | `[stage1]` | diff-loss/coordinate 처치의 state 이득 검증 후 채택 |
| P3 | 근본 | `[stage2]`/`[idood]` | World Model 목적함수/데이터 재설계 — 장기(가설이 음수로 재현될 때만) |

**정직한 결론**도 함께: 현재 데이터로 "stage1+stage2 > stage2"(핵심 가설)가 **성립하는가/부호가 무엇인가**를
EXP·model·eval-DS·ID/OOD 별로 표로 명시하고, 성립 안 하면 그대로 보고(가설 옹호 금지).

## 실행 절차

1. **카운트**: `--exp`/`--model`/`--eval-ds` 대상 각 셀의 metric JSON 직접 읽기(`load_metrics` 로직) → `base`/`stage2`/`stage1+stage2` × ID/OOD 표. `eval_viewer.py`·`pairs_summary.md` 와 **교차검산**. outputs 부재 셀은 `N/A(미materialize)` 명시.
2. **5단계 분해**: `--stage` 지정 단계(기본 all)를 DATA→STAGE1→STAGE2→EVAL→IDOOD 순회. 각 실패모드에 verdict. 단일-셀 주장은 여러 EXP·model·eval-DS 로 일반화하거나 illustrative 로 강등.
3. **검증**(기본, `--no-workflow` 아니면): `Workflow` 로 단계별 분류 → **적대적 검증**(가설-bearing 주장은 부호 일관성 재확인; metric 은 파일에서 재유도; ID/OOD 는 3-섹션 직접 카운트) → 종합.
4. **개선책**: S 시리즈 + ROI 순서 + 핵심 가설 정직한 결론(각 항목 `[stage:...]` 태그).
5. **산출**: 아래 7종(repo 스냅샷) + **`/workflow:project-sync` 위임**으로 Obsidian Issue 노트(날짜-우선)·Notion Experiments DB 1행 미러.

## 산출물

`.claude/issues/<오늘날짜_HH-MM-SS>/` 에 7종(시간은 `date +%H-%M-%S`; 이 디렉터리 관례는 이 스킬이 신설):

| 파일 | 내용 |
|------|------|
| `README.md` | 허브: 한 줄 요약 + 7문서 인덱스 + **stage × verdict 핵심 표** + 핵심 가설 결론(부호) |
| `_matrix.md` | 전 셀 metric 직접 카운트 표(base/stage2/stage1+stage2 × ID/OOD × eval-DS), 미materialize 셀 `N/A` |
| `gap-stage1-data.md` | DATA-1~3 + verdict + SSoT 증거(파일 경로) |
| `gap-stage2-stage1train.md` | STAGE1-1~3 + verdict + SSoT 증거 |
| `gap-stage3-stage2train.md` | STAGE2-1~4(핵심 가설) + verdict + SSoT 증거 |
| `gap-stage4-eval.md` | EVAL-1~4 + verdict + SSoT 증거 |
| `gap-stage5-idood.md` | IDOOD-1~3 + verdict + SSoT 증거 |
| `solutions.md` | S 시리즈(각 항목 `[stage:...]` 태그) + ROI + 핵심 가설 정직한 결론 |

## Obsidian 미러 — project-sync 위임

기본(`--no-obsidian` 아니면): 기록은 **`/workflow:project-sync` (obsidian)** 로 수행한다. vault 경로·frontmatter·
명명/링크/no-split 규칙의 **SoT 는 `~/.claude/skills/utility/project-sync/references/obsidian.md`**(이 스킬은 인라인하지
않는다). analyze-experiments 는 project-sync 에 **스냅샷**(`category=experiment` + 아래 Issue 허브 노트 본문)을 넘겨
`Projects/Implicit-World-Modeling/Issues/` 에 **런당 1개 이슈 노트**로 적재하게 한다.

- **파일명·title·H1 은 날짜-우선**(project-sync 규약): `Projects/Implicit-World-Modeling/Issues/<YYYY-MM-DD HH-MM-SS> Experiment Analysis (요약).md`,
  H1 `# <YYYY-MM-DD> Experiment Analysis`. (Vault 접근 불가/headless — Linux 학습머신 등 — 이면 Obsidian 미러 생략하고 그 사실을 결과에 명시. `.project-sync.json` 은 Obsidian 을 macOS 에서만 활성.)
- **허브 노트 본문**: TL;DR + **핵심 가설 결론(stage1+stage2 − stage2 부호)** + stage×verdict 표 → 🧱 DATA → 🌍 STAGE1 → 🎯 STAGE2 → 📏 EVAL → 🔀 ID/OOD → 개선책(`[stage]` 태그). **런당 노트 1개**(H2 앵커, 분할 금지).
- **개념노트 wikilink**(있으면): `[[World Model vs Action Prediction]]`·`[[ID-OOD 일반화]]` 같은 IWM 전용 개념은 `Projects/Implicit-World-Modeling/Concepts/`, 여러 프로젝트 공유 개념만 `Wiki/`(LLM Wiki 승격 기준).
- **관련 문서 갱신은 project-sync 담당**: `log.md`·`Projects/index.md`·개요 갱신은 project-sync 가 `references/obsidian.md` 규약대로.

## Notion 미러 — project-sync 위임 (카테고리 DB 단일 기록)

기본(`--no-notion` 아니면): 기록은 **`/workflow:project-sync` (notion)** 로 수행한다. DB 라우팅·속성·기록 규칙의
**SoT 는 `~/.claude/skills/utility/project-sync/references/notion.md`**, DB ID 는 `.project-sync.json` `notion.databases`
에서 읽는다(하드코딩 안 함). 권한 없음/headless 면 Notion 미러 생략.

**기록 정책** — 카테고리 DB 한 곳에만 1행:
- 실험 분석이므로 **`experiment` 카테고리 → 🧪 Experiments DB 1행**(`data_source` 는 config 에서). **Dev Log·Timeline 중복 행 미생성.**
- project-sync 에 넘기는 스냅샷: `category=experiment`, `summary=Experiment Analysis <YYYY-MM-DD HH-MM-SS> (요약)`,
  `experiment.id`=대상 EXP, `experiment.model`=모델, `experiment.stage`=`base|stage2|stage1+stage2`, `experiment.ratio`=(EXP01) `3:7/5:5/7:3`,
  `experiment.metric`=**Overall 헤드라인**(action=`step_accuracy`·state=`avg_hungarian_f1`), `experiment.output_path`=`.claude/issues/<날짜_시간>/`,
  `details`=README 허브 요약 + `_matrix.md` ID/OOD 표.
- project-sync 는 Experiments DB 스키마(`Experiment`/`Model`/`Stage`/`Ratio`/`Status`/`날짜`/`Metric`/`Output path`)를 **조회해** 실제 속성명에 맞춘다. 새 DB·속성 생성 금지, 옛 행 삭제 금지(비파괴).

> 여러 셀(EXP×model×변형)을 한 번에 기록해야 하면, 대표 헤드라인 1행 + 본문 표로 남긴다(행 폭증 방지). 세부 정본은 `.claude/issues/` 스냅샷.

## 정직성 규칙 (Honesty)

- metric JSON 직접 카운트가 SSoT. stale 값·자동 요약 분모를 그대로 베끼지 말고 현재 파일로 검산.
- **outputs 부재(macOS)** 는 `N/A(미materialize)` 로 정직히 표기 — 없는 metric 을 지어내지 말 것. 코드·config 진단은 그래도 수행.
- **핵심 가설 옹호 금지**: `stage1+stage2 > stage2` 가 데이터로 성립 안 하면 부호 그대로 보고. 단일-셀 우연을 일반 결론으로 승격 금지.
- **내부 비교만**: 외부 시스템 숫자를 근거로 끌어오지 말 것(references/ 는 설계 참고 코드).
- verdict 미통과(single-cell·factually-wrong 등) 진술이 남으면 산출 문서에 **명시 플래그**(숨기지 말 것).
