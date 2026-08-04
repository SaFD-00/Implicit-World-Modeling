# Notion 반영 대기분 — `change_f1` 두 기준선 (2026-08-04)

> **왜 이 파일이 있나**: 2026-08-04 작업 시점에 Notion MCP 서버가 연결되지 않아
> (`claude.ai Notion is not connected`) 표를 직접 갱신하지 못했다. 연결 복구 후
> 아래 내용을 그대로 반영하면 된다. 반영이 끝나면 이 파일은 지운다.
>
> 대상 페이지: Experiments — Offline 표 (`264dbb15…`)

## 반영 시 지켜야 할 것 (기존에 데인 자리)

- `update_content` 의 `old_str` 은 **페이지 전체에서 유일**해야 한다. 지표명처럼 여러 행에
  반복되는 문자열을 쓰면 엉뚱한 행이 바뀐다.
- **나눠 호출하지 말 것.** 한 표를 여러 번에 걸쳐 고치면 중간 상태가 diff 로 남아 오염된다.
  한 번의 호출로 블록 전체를 교체한다.
- 출처 host / seed callout 은 기존 규칙대로 유지 (seed 42 · `--match-mode pos`).

---

## 1. Metrics 정의 표 — `change` 축 항목 (신규/교체)

| 지표 | 정의 | 읽는 법 |
|---|---|---|
| `change_prec` / `change_recall` / `change_f1` | `current` 대비 바뀐 항목을 pred/gt 양쪽에서 **같은 절차**로 뽑아 집합 비교. `C = ADDED ∪ MODIFIED ∪ DELETED`. hit 은 ADDED/MODIFIED 면 정본 pred↔gt 매칭에서 짝이 맞고 그 짝의 `text_sim ≥ τ(0.9)` 일 때, DELETED 면 같은 `current` 요소를 양쪽 다 지웠을 때. | **`change_f1_null` 과 반드시 같이 읽는다.** 단독 인용 금지. |
| `change_f1_null` | 같은 행에서 **빈 예측**이 받는 점수 = 이 축의 퇴화 바닥. `(current, gt)` 만의 함수라 **test set 당 상수**이고 예측과 무관하다. | 이 값을 **넘어야** 비로소 "변화를 예측했다"고 말할 수 있다. |

### 이 축의 두 기준선

| 퇴화 전략 | 점수 | 비고 |
|---|---|---|
| 복사기 (`pred = current`) | **0.0** | 설계상 상수. 변화를 하나도 주장하지 않음 |
| delete-all (`pred = ""`) | **데이터 의존** — EXP01 0.378 · EXP03 0.377 · EXP05 0.245~0.250 · EXP07 0.261 | 빈 예측은 `current` 전체를 지운 것으로 분류되고, 화면 전환은 실제로 상당 부분을 지우므로 교집합이 공짜 hit 이 된다 |

> **0 을 바닥으로 읽으면 미학습 base 가 학습된 모델보다 높게 나온다.** 그건 결과가 아니라
> 눈금 없이 읽은 것이다. 실측: EXP07v1 lora ep1(학습됨·`unclosed_root` 0.092) `change_f1`
> **0.114** < 바닥 **0.261**. 전 **32 leaf 중 바닥을 넘긴 것은 4개**(EXP05 Full FT ep2·ep3,
> 각각 정규/woa)뿐이다.

## 2. 열 분리 — index 모드와 pos 계열

`change` 축은 매칭 임계에 걸리므로 두 모드의 값을 한 열에 섞지 말 것.

- **index 모드** (EXP01 / EXP02 / EXP03 / EXP04 / MB): `MATCH_THRESHOLD` 1.5
- **pos 계열** (EXP05 / EXP06 / EXP07): `MATCH_THRESHOLD_POS` 1.7

텍스트가 통째로 바뀌면 cost 가 정확히 `W_TEXT`(1.5)라 index 는 매칭이 **떨어져 ADDED**,
pos 는 **붙어서 MODIFIED** 가 된다. 층 크기(`n_gt_added` 등)가 갈리므로 diff 유형 분포를
두 모드에 걸쳐 비교하면 안 된다.

## 3. Offline 표에 추가할 수치 (state 계열)

`change_f1` 열 옆에 `null` 열을 같이 둔다. Δ 가 음수면 **퇴화 바닥 미달**이다.

| EXP | 대상 | `change_f1` | `null` | Δ |
|---|---|---|---|---|
| EXP01 | 7b ratio73 base | 0.352 | 0.378 | −0.027 |
| EXP01 | 7b ratio73 lora ep1 | 0.188 | 0.396 | −0.208 |
| EXP03 | 7b base | 0.364 | 0.377 | −0.013 |
| EXP05 | 3b base | 0.208 | 0.242 | −0.034 |
| EXP05 | 3b full ep1 | 0.233 | 0.250 | −0.016 |
| EXP05 | 3b full ep2 | **0.281** | 0.249 | **+0.032** |
| EXP05 | 3b full ep3 | **0.293** | 0.249 | **+0.044** |
| EXP07 v1 | 3b base | 0.231 | 0.260 | −0.029 |
| EXP07 v1 | 3b lora ep0.25 → ep1 | 0.038 → 0.114 | 0.261 | −0.223 → −0.147 |
| EXP07 v2 | 3b lora ep0.25 → ep1 | 0.055 → 0.070 | 0.261 | −0.206 → −0.191 |

(ID 섹션 기준. woa sibling 도 같은 방향이며 차이는 ≤0.005.)

## 4. 망각 프로브(`probe_forget`) — 신규 행

stage2 체크포인트를 **stage1 state test** 로 재평가. 평가면이 stage1 leaf 와 같으므로
같은 표에 놓고 읽을 수 있다. (ID n=2032, `--match-mode pos`)

| 대상 | `diff_recall` | `added_recall` | `copy_excess` | `change_f1` (바닥 0.261) |
|---|---|---|---|---|
| stage1 lora ep1 *(기준선)* | 0.682 | **0.237** | +0.063 | 0.114 |
| → mergeO s2ep1 | 0.592 | **0.045** | +0.184 | 0.062 |
| → mergeO s2ep3 | 0.600 | **0.042** | +0.192 | 0.041 |
| onlyS2 ep3 *(stage1 없음)* | 0.469 | 0.038 | +0.219 | 0.111 |

**stage2 학습이 world-model 능력을 복사 편향 쪽으로 되돌린다** — `added_recall` −81%,
`copy_excess` 3배. `diff_recall` 은 덜 떨어지는데 UNCHANGED 위주로 유지되기 때문이라
그 축만 보면 놓친다. stage1 없는 `onlyS2` 가 `diff_recall` 0.469 로 `mergeO`(0.59~0.60)
보다 낮아 **stage1 의 잔여 효과는 남는다**.

> 빈 3셀(mergeO ep2 · onlyS2 ep1 · ep2) 추론이 끝나면 6셀 전수로 epoch 단조성까지
> 채워 넣을 것. 산출 경로는 `outputs/AndroidControl_EXP07/probe_forget/<cell>/`.
