# EXP05 Diff Loss 개편 작업 계획

> 작성일: 2026-07-11 · 근거: Slack DM(조병웅↔백승우) 2026-07-11 12:33–13:21 KST + Google Meet 회의록(12:35 KST)
> 상태: 계획 확정, 구현 착수

---

## 1. 배경 — 무엇이 바뀌었나

조병웅님이 diff loss의 **가중치 체계**와 **매칭 로직**을 동시에 바꿨고, xy 좌표 통일에 따라 **채점 기준**도 바뀐다.
EXP05는 지금까지 diff loss 없이 학습돼 왔으므로(YAML에 플래그 부재 실측 확인), 이번에 **diff loss를 적용**해 재학습한다.

### 1-a. 가중치 체계 (핵심 변경)

| 구분 | 기존 | 신규 (Qwen Agent World 방식) |
|---|---|---|
| state transition — **diff 토큰** | 2.0배 | **1.0배** |
| state transition — **non-diff 토큰** | 1.0배 | **0.25배** |
| intermediate **action** 출력 | (가중 없음) | **전부 1.0배 — 가중치 주지 않음** |

배수 자체는 줄었지만 diff가 non-diff 대비 **실질 4배** 강해진다. 조병웅님 원문: *"배수는 줄긴 했는데 사실상 diff가 아닌 토큰에 비해 거의 네 배가 강하게 되는 거죠"*, *"인터미디어트 액션 프레딕션은 1배수를 그대로 준다. 액션 폼의 출력은 따로 가중치를 주지 않는다."*

### 1-b. 매칭 로직 (`new_diff_loss.zip`, Slack 12:49 첨부)

v2가 **DOM index를 버리고 bounds(좌표) 기반**으로 전환했다.

| 항목 | v1 | v2 |
|---|---|---|
| 위치 cost | DOM index 거리 (`W_INDEX=0.2`) | **bounds 중심점 거리** (`W_POS=0.4`, `BOUNDS_NORM=2050.0` = 840×1876 대각선) |
| 텍스트 수집 | 자손 텍스트까지 흡수 | **direct text + 자체 속성만**(description/id/text/aria-label) |
| 매칭 임계 | `MATCH_THRESHOLD=1.5` | **1.7** (같은 자리 텍스트 변경도 MODIFIED로 인정) |
| element 키 | `index` 필수 | **index → bounds 폴백** |
| metric key | `hungarian_idx` | **`hungarian_pos`** (px 거리 기반) |

### 1-c. 채점 기준 (회의록 확정)

액션 스페이스가 **모든 액션이 xy1(start)/xy2(end)를 출력**하도록 통일됐다. 기존 eval은 index 기반이라 못 쓴다.

- **click / long-press**: GT에 bbox가 없고 클릭 좌표만 있음 → 예측 좌표가 **GT 좌표가 속한 element의 bbox 안**에 들어가면 정답 (OS-Atlas AndroidControl 방식)
- **scroll / swipe**: xy1→xy2 벡터의 **방향만** 매칭(up/down/left/right). 대각선은 **주 성분** 기준
- **input_text**: 좌표 없음 → 액션 타입/텍스트 fuzzy 매칭만

### 1-d. 그 외 확정 사항

- **데이터 분할 비율 7:3 확정** — EXP05는 이미 AC_EXP01 **ratio73** 멤버십을 미러하므로 **추가 작업 불필요** (교수님 최종 확인만 남음)
- **`max_pixels: 1605632` 항상 유지** (조병웅님 강조 — 이걸 안 맞추면 grounding이 조용히 깨짐)
- 15일부터 7B 본실험, 그 전까지 3B 테스트런

---

## 2. 조사에서 드러난 정정 사항 3건 ⚠️

계획을 세우며 저장소를 직접 확인한 결과, 처음 가정과 다른 사실 3가지가 나왔다. **셋 다 계획을 바꾼다.**

### (1) EXP05 HTML에는 `index` 속성이 아예 없다
train 1행 실측: `index` 0개, `bounds` 48개, `point` 48개.
→ v1 builder(`get_element_char_spans`가 `index="..."` regex를 필수로 요구)를 EXP05에 쓰면 **pos_map이 비어 모든 토큰이 baseline으로 방치**된다.
**v2의 bounds 폴백은 "개선"이 아니라 EXP05의 필수 요건이다.**

### (2) zip의 `preprocess_dataset_v2.py`에 import 버그
`from token_weight_builder import build_token_weights` — **v1 builder를 import**한다(v2 아님).
그대로 넣으면 v1이 조용히 로드되어 bounds 키가 무시된다. → 반드시 `token_weight_builder_v2`로 고쳐야 한다.

### (3) Vessl 파이프라인이 저장소에 없다
`scripts/` 전체 grep/find 결과 Vessl 스크립트·스펙 파일 전무, `vessl` CLI 미설치, `~/.vessl` 없음.
README 254행과 `docs/DEVLOG.md:17`의 서술이 전부다.
→ 저장소 측 산출물은 **"가중 데이터 + YAML + 로컬 스모크 통과"까지**이고, **실제 Vessl 잡 제출(org/project/cluster/데이터 업로드 경로)은 사용자 확인이 필요**하다.

---

## 3. 학습 인프라 — 로컬 불가 판정 (실측)

로컬 2×RTX5090에서 EXP05 3B Full FT를 실제로 돌려 측정한 결과:

| 항목 | 실측값 |
|---|---|
| 총 스텝 | 2,232 (47,556 samples × 3 epoch — 0710 기준 실측. 0711 적용 후 train 은 44,670) |
| 속도 | **157.8 ~ 168.2 s/it** |
| 추정 총 시간 | **97~104시간 (약 4일)** |
| 결과 | **step 3에서 CUDA OOM** (GPU0 31.36 GiB 중 8.92 GiB 추가 할당 실패) |

- OOM 원인: `per_device_batch_size=1`인데도 터짐 — `cutoff_len 24576` + `max_pixels 1,605,632`의 비전 토큰으로 시퀀스가 극단적으로 길다.
- 157 s/it 원인: RTX5090에 강제되는 **ZeRO-3 + CPU offload**.

**→ 결론: Vessl A100/H100에서 학습한다.** (회의록에서 조병웅님도 *"일단 A에 돌리는 게 좋을 것 같은데, 패스를 충전했으니까"*라고 제안 — 실측이 이를 뒷받침)

---

## 4. 작업 단위

| 유닛 | 내용 | 복잡도/리스크 | 담당 | 실행 |
|---|---|---|---|---|
| **W1** | diff loss v2 통합 + 신규 가중 체계 + EXP05 train 데이터 재생성 | complex / **high** | claude/opus | 병렬 |
| **W2** | 노트북 SSoT + EXP05 YAML 6종에 `use_diff_token_weighted_loss` 적용 | simple / medium | claude/sonnet | 병렬 |
| **W3** | eval 채점 기준 변경 (xy/bbox/방향 매칭) | substantial / medium | claude/opus | 병렬 |
| **W4** | 학습 스모크 검증 + Vessl hand-off | simple / medium | claude/sonnet | W1·W2 완료 후 |

W1/W2/W3은 파일이 완전히 분리되어 동시 실행 가능(worktree 불필요). 검증(tier-2)은 전 유닛 **codex**가 담당(구현자가 claude이므로 검증자 플랫폼을 반대편으로).

---

### W1 — diff loss v2 통합 + 신규 가중 체계 + 데이터 재생성

**목표**: `data/AndroidControl_EXP05/..._stage1_train.jsonl` 전체에 신규 체계 `token_weights`를 인라인 부여. (0711 수정본 적용 후 **44,670행** — 0710 기준 47,556행이었다.)

**핵심 함정 — 단순 상수 교체가 아니다.**
현재 `token_weight_builder.py`는 baseline이 `[1.0]*n_asst`이고 `if weight == 1.0: continue`로 기본값을 스킵한다.
신규 체계에선 **diff weight가 1.0**이라, 이 스킵 때문에 **diff 토큰이 baseline(0.25)에 방치**된다.
→ baseline을 `wmap["UNCHANGED"]`(=0.25)에서 유도하고, 스킵 조건도 `if weight == base`로 바꿔야 한다.

**action 샘플 분기 필수.**
샘플 판별 규약은 **이미지 개수**(`mirror_experiment.py --experiment exp05`): `images 1개 = state_pred`, `2개 = action_pred`.
action 샘플은 diff 계산을 건너뛰고 **uniform 1.0**을 줘야 한다. 이 분기가 없으면 action이 "diff 없음 → 전부 0.25"로 잘못 처리된다.
(구 체계에선 baseline이 1.0이라 우연히 맞았지만, 신규 체계에선 치명적)

**작업**:
- v1 파일 4개는 **건드리지 않는다**(EXP02 재현성 보존). `scripts/diff_loss/`에 v2 4파일 신규 추가.
- `token_weight_builder_v2.py`: baseline 유도 + 스킵 조건 수정 + `WEIGHT_MAP = {ADDED:1.0, MODIFIED:1.0, UNCHANGED:0.25}`
- `preprocess_dataset_v2.py`: import 버그 수정 + action 분기 추가 + CLI 기본값 변경
- 전체 재생성 후 원자 교체 (`scripts/build_exp05_data.py` 가 mirror → 가중치 → `os.replace` 를 수행)

**검증**: state 샘플은 weight 집합이 `{0.25, 1.0}` 부분집합, action 샘플은 정확히 `{1.0}`. 행수는 mirror 산출과 일치해야 한다.

> zip의 `weighted_v2_bd.jsonl`(100행)은 가중치가 구 체계(1.0/2.0/3.0)이므로 **목표값이 아니라 포맷 레퍼런스**로만 쓴다.

---

### W2 — diff loss 플래그 활성화

**목표**: EXP05 학습이 실제로 diff loss를 켜고 돌게 한다. **지금은 꺼져 있다.**

- 노트북 Cell 5의 `AndroidControl_EXP05.stage1` dict에 `"use_diff_token_weighted_loss": True` 추가 (SSoT)
- EXP05 YAML 6개(`stage1_{full,lora}` × 3모델)에 `use_diff_token_weighted_loss: true` 반영
- Cell 10은 이미 이 키를 YAML로 주입하는 로직 보유 → **수정 불필요**
- `dataset_info.json` 변경 불필요 — `token_weights`는 train jsonl 인라인이고 `converter.py:226`이 직접 읽음 (EXP02 선례)
- `max_pixels`·`cutoff_len` 등 기존 값 **일절 변경 금지**

---

### W3 — eval 채점 기준

**하위호환 우선**: opt-in 모드 플래그로 구현해 EXP01~04 채점은 **바이트 단위로 불변** 유지.

- `_action_eval.py`: `--coord-mode {index,xy}` (기본 index)
- `_hungarian_eval.py`: `--match-mode {index,pos}` (기본 index)
- `stage1_eval.sh`: DS가 `AC_EXP05`일 때만 신규 플래그 전달

**xy 모드 규칙**:
- click/long-press: test user content의 `Current UI State:` XML에서 bounds 수집 → GT 좌표를 포함하는 **최소 면적** element를 GT bbox로 → pred 좌표가 그 안이면 정답. 포함 element가 없으면 오답 처리하되 `no_bbox_n`으로 **별도 집계**(임의 폴백 발명 금지)
- scroll: `|dx| ≥ |dy|` → left/right, 아니면 up/down. 방향 일치 시 정답
- input_text: 좌표 무관, 텍스트 매칭만

**pos 모드**: `_hungarian_eval.py`에 내장된 v1 metric을 v2 상수·bounds 중심점 cost로 포팅, 출력 key `hungarian_pos`.

---

### W4 — 스모크 + Vessl hand-off

- 로컬 Full FT는 OOM이므로 스모크는 **LoRA + `max_steps: 3`**으로 (repo YAML 수정 없이 scratchpad 복사본에서)
- 통과 기준: 3 step loss 산출 + `use_diff_token_weighted_loss` 분기 통과 + 크래시 없음
- Vessl hand-off 패키지: 가중 train jsonl 경로, YAML 경로, `data/AndroidControl/images/` 의존성, env 요구사항
- **사용자 확인 필요**: Vessl org/project/cluster/데이터 업로드 경로 (저장소에 없음)

---

## 5. 데이터 상태 — **0711 수정본 적용 완료 (2026-07-13)**

Drive `0711_버젼` 폴더에 수정본이 모두 도착해 (action 파일이 2026-07-12 23:24 로 마지막 업로드) 재다운로드 → mirror → 전처리를 **재실행 완료**했다.

### 0711 이 바꾼 것 — `wait` 액션 클래스 교체가 전부다 (전수 확인)

- 나머지 7개 액션(click 49,282 / swipe 10,606 / type 5,787 / open 5,208 / navigate_back 2,879 / long_press 160 / navigate_home 28)은 **건수까지 완전 동일**.
- 기존 `wait` **4,958건을 전량 퍼지**하고, **전혀 다른 (episode, step) 위치에 `wait` 400건을 action 쪽에만 재생성**. `78,908 − 4,958 + 400 = 74,350` 으로 산술이 닫힌다.
- 공통 레코드 73,950건의 변화는 **프롬프트·JSON 포맷 정리뿐**이며 GT 의미는 불변 (state 의 예측 대상 XML 은 전건 동일; action GT 는 compact→spaced 재포맷).
  - action_pred: `[Screenshot]` → `[Current UI Screenshot]` / `[Next UI Screenshot]`
  - state_pred: system prompt 재작성 (1,269→1,575자; `Current Action` 입력 명시 + `# Output Format (STRICT)` 신설)

### 실측 (재생성 결과)

| | 0710 | 0711 |
|---|---:|---:|
| source pool (action / state) | 78,908 / 78,908 | **74,350 / 73,950** |
| mirror train (입력 50,000) | 47,556 (drop 2,444) | **44,670 (drop 5,330)** |
| mirror 총계 (7 파일) | 64,787 | **60,717** |
| 가중 train 구성 | state 33,285 + action 14,271 | **state 31,221 + action 13,449** |
| state 출력 토큰 0.25배 감쇠 | 54.1% | **53.4%** |

### ⚠️ 조병웅님 확인이 필요한 사항 3건

1. **`wait` 전량 퍼지**가 의도인가? 퍼지된 4,958건 중 **EXP01 train 멤버십 2,548건 · test 멤버십 598건** 이 포함돼 있어 **테스트셋 오염 제거가 아니라 커버리지 축소**다 (train −6.1%).
2. **빈 current state `wait` 400건 재생성** — 399/400 이 Current UI State 가 빈 노드(`<node bounds="[0,0][0,0]" point="[0,0]"/>`)다. 0710 엔 빈 current state 가 0건이었다. `wait` 이 **"빈 화면이면 wait" 이라는 degenerate shortcut** 이 됐고, 이 중 **74건이 실제로 train 에 유입**됐다. (Drive 6월 폴더명이 *"…비어있는 state 샘플 삭제"* 였던 점에서 재유입 의심.)
3. **action / state 키 대칭 붕괴** — 0710 은 두 pool 의 키집합이 같았으나 (78,908) 0711 은 action(74,350) ⊋ state(73,950), 차이가 정확히 위 400건이다.

---

## 6. 미결 / 확인 필요

1. **Vessl 실행 경로** — org/project/cluster, 데이터 업로드 방식 (저장소에 스크립트 없음)
2. ~~**수정본 stage1 데이터** — 조병웅님 업로드 대기 중~~ → **해소 (2026-07-13)**. 0711 수정본 적용 완료. 단, §5 의 **확인 필요 3건**(wait 전량 퍼지 / 빈-state wait 400건 / 키 대칭 붕괴)이 새로 열렸다.
3. **7:3 비율 교수님 최종 확인** — 회의록의 백승우님 action item
4. **structural 토큰(태그 문법 등)의 가중치** — 이진 표현상 0.25가 충실한 해석이나, 애매하면 조병웅님 확인
5. **`extract_elements` 의 aria-label 누락** — 포함 조건이 `description` 단독이라 `<div aria-label="Home">` 류가 제외된다. 0711 pool 실측 **약 88%** 의 레코드가 aria-label 을 갖는다 (action 65,355/74,350 · state 65,097/73,950) → 영향 범위가 크다. 학습/평가 공유 규약이라 조병웅님 확인 필요.
6. **`without_open_app` 필터 무동작** — 필터가 `## Action` 마커를 찾는데 프롬프트 포맷은 `Action:` 을 쓴다 (0711 state 100% = 73,950/73,950 이 `Action:`, `## Action` 은 0건). 그래서 `test_*_state` 와 `test_*_state_without_open_app` 의 행수가 같다. **EXP03 에도 있는 기존 이슈.**
7. **좌표 범위이탈 11키 (원천 데이터 버그)** — 0710·0711 **양쪽 동일하게** `coordinate`(액션 라벨) 필드에서 840×1876 을 벗어나는 키가 11개다 (`bounds`/`point` 는 깨끗). 값이 `[1682, 975]` 로 반복되고 **1682 ≈ 840×2** 라 스케일링 버그로 보인다. **현재 EXP05 산출물에 10행이 실려 있고** (train 7 + `test_ood_{action,state,state_without_open_app}` 각 1), `(12571,0) = [421,1979]` 는 **OOD 평가셋 3개 파일 전부를 오염**시킨다. 조병웅님 확인 필요.
