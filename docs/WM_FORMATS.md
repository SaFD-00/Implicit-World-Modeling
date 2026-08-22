# Mobile GUI World-Model 데이터 포맷 분할 파이프라인

> **이 문서는 조병웅님이 전달한 벤더 설계 문서다 (2026-08-22 diff_loss_bundle_v2).**
> 원문을 보존하되, 이 저장소에 실제로 배선된 형태는 아래와 같이 다르다. 충돌 시 **이 표가 정본**이다.
>
> | 문서가 말하는 것 | 이 저장소의 실제 |
> |---|---|
> | `build_wm_formats.py` (프로젝트 루트) | `scripts/build_wm_formats.py` |
> | `validate_wm_formats.py` (루트) | `scripts/validate_wm_formats.py` |
> | `inspect_weighted.py` (루트) | `scripts/diff_loss/inspect_weighted.py` (경로를 CLI 인자로 받도록 수정) |
> | `hungarian_metric_v2.py` / `token_weight_builder_v2.py` **교체** | 교체하지 않았다. `*_v2c.py` 복제본에만 Cerebra 확장을 넣었다 (AGENTS 하드 제약 9 — v2 는 EXP02/05/06/07 재현 경로라 불가침). `build_diff_targets.py` 만 `_v2c` 를 import 한다 |
> | UNCHANGED 가중치 `0.25` | 동일 (0.25 확정). 참고로 지난 EXP07 v2 는 0.05, v1 은 0.2 였다 |
> | element 집합 = interactive/content/clickable/described | **+ 구조축**: `data-bbox` 가 있는 `div` 도 채택한다. 확장 전에는 div 여는-태그의 52% 가 어떤 span 에도 안 덮여 새로 생긴 컨테이너가 diff weight 를 못 받았다 (2026-08-22 실측) |
> | element 의 가중 구간 = 서브트리 전체 | **자식 element 가 있으면 여는-태그 구간만** 가중한다. 서브트리를 주면 문서 전체를 감싼 래퍼 하나의 판정이 나머지를 덮어써서 diff 신호가 소멸한다 (200 샘플 중 6건이 토큰 100% 가 1.0 이 됐다) |
> | 데이터 빌드 | `scripts/build_exp08_data.py` 가 AC_EXP08 의 정본 빌더다 (샘플링·필터·3포맷·가중치를 한 번에 수행) |
>
> 실험군 배선·계보는 [`../Implicit-World-Modeling/ARCHITECTURE.md`](../Implicit-World-Modeling/ARCHITECTURE.md) §3, 규칙은 `AGENTS.md` 를 본다.


`NEXT_STATE_PREDICTION` 학습 데이터를 세 가지 **관측성(observability) 포맷**으로
분할·변환하여, 월드 모델 사전학습에서 발생하는 **복사 편향(copy shortcut)** 을 억제한다.

| 파일 | 역할 |
|---|---|
| `build_wm_formats.py` | 분할 + 마스킹 + 프롬프트 재조립 + 2파일 출력 |
| `validate_wm_formats.py` | 출력물 불변식 11종 검증 (학습 전 필수 통과) |
| `scripts/diff_loss/build_diff_targets.py` | raw 로 헝가리안 diff 계산 → applied 에 `token_weights` 부착 (학습 입력 생성) |

```bash
# 1. 세 포맷으로 분할
python build_wm_formats.py --input data.jsonl --outdir out --seed 42

# 2. 불변식 검증
python validate_wm_formats.py \
    --raw     out/data_split_raw.jsonl \
    --applied out/data_split_applied.jsonl

# 3. raw 로 diff 계산해서 applied 에 token_weights 붙이기 (최종 학습 입력)
python scripts/diff_loss/build_diff_targets.py \
    --raw     out/data_split_raw.jsonl \
    --applied out/data_split_applied.jsonl \
    --output  out/data_split_weighted.jsonl \
    --model   Qwen/Qwen2.5-VL-7B-Instruct
```

---

## 1. 배경: 왜 이 파이프라인이 필요한가

### 1.1 문제

`(current XML + screenshot + action) -> next XML` 로 50k 규모 학습을 돌린 결과,
모델이 **입력 XML을 그대로 복사하는 전략**에 수렴했다.

### 1.2 원인 (실측)

200 샘플 분석 결과:

| 지표 | 값 |
|---|---|
| "입력 XML을 그대로 출력" 했을 때의 토큰 일치율 (평균) | **0.508** |
| 같은 지표 p90 | 0.987 |
| 일치율 > 0.9 인 샘플 비율 | 19% |

**아무것도 이해하지 않고 베끼기만 해도 토큰의 절반을 맞힌다.** 이것이
학습 초반 수백 스텝 안에 도달 가능한 강력한 국소 최적이고, 여기서 빠져나오려면
"스크린샷을 파싱해서 XML을 새로 쓰는" 비싼 능력이 필요하다.

추가 원인:
- **무변화 전이** 4~8% (액션을 했는데 화면이 안 바뀜) — 복사가 *진짜* 정답인 샘플
- Transformer의 **identity mapping bias** (residual connection + tied embedding)

### 1.3 처방

**입력에서 베낄 원본을 물리적으로 제거한다.** 손실 가중치(별도 파이프라인)와
상보적으로 작동한다.

| 슬라이스 | anti-copy 장치 |
|---|---|
| `full` 25% | diff 손실 가중치 (별도 파이프라인) |
| `masked` 55% | **입력 부분 제거** + diff 손실 |
| `dropped` 20% | **입력 완전 제거** (복사 원천 불가) |

마스킹은 anti-copy 장치이면서 동시에 **데이터 효율 장치**다. 50k × ~2,400토큰
≈ 1.2억 토큰은 UI-Oceanus(32억 토큰) 대비 4% 규모라, 모든 gradient 스텝을
어려운 능력에 쓰게 만드는 것이 중요하다.

---

## 2. 입출력 명세

### 2.1 입력

LLaMA-Factory ShareGPT 형식 JSONL. 샘플 1건:

```json
{
  "messages": [
    {"from": "system", "value": "# Mode: NEXT_STATE_PREDICTION\n..."},
    {"from": "human",  "value": "Current UI State:\n<div ...>\n\n[Screenshot]\n<image>\n\nAction:\n<action>{...}</action>"},
    {"from": "gpt",    "value": "<div data-bbox=...>...</div>"}
  ],
  "images": ["myset/images/episode_4065_step_0001.jpg"]
}
```

**전제 조건** (어기면 해당 샘플을 skip 하고 리포트에 기록):
- `human` 턴이 `Current UI State:` → `[Screenshot]` → `Action:` 순서
- `<image>` 플레이스홀더 1개
- XML은 `data-bbox` 속성을 가진 html-style 태그 트리

### 2.2 출력

#### `*_split_raw.jsonl` — 원본 보존본

```json
{"messages": [...원본 그대로...], "images": [...],
 "sample_id": 9, "content_hash": "a3f2c1d4e5b6", "fmt": "masked"}
```

**용도**: diff-loss(헝가리안) 파이프라인의 입력. 원본 XML 좌표계가 필요하므로
반드시 이 파일을 쓴다.

#### `*_split_applied.jsonl` — 학습용

```json
{"messages": [...변환됨...], "images": [...],
 "sample_id": 9, "content_hash": "a3f2c1d4e5b6", "fmt": "masked",
 "meta": {
   "image_first": true,
   "xml_len_orig": 1312, "xml_len_out": 832,
   "assistant_char_len": 3934,
   "mask_ratio_target": 0.364, "mask_ratio_actual": 0.3659,
   "n_whole": 7, "n_text": 2, "n_aria": 2,
   "tier_used": {"1": 11, "3": 1},
   "masked_elements": [
     {"group": 0, "kind": "whole", "tag": "button",
      "bbox": "758 116 840 214", "key": "", "src_span": [113, 176]},
     {"group": 3, "kind": "text",  "tag": "p",
      "bbox": "49 466 495 519", "key": "Thu, Sep 28", "src_span": [486, 511]},
     {"group": 3, "kind": "aria",  "tag": "p",
      "bbox": "49 466 495 519", "key": "Thursday, September 28", "src_span": [401, 452]}
   ]}}
```

두 파일은 **같은 순서, 같은 `sample_id`** 로 1:1 대응한다.

---

## 3. 처리 단계

### Step 0 — XML 스캔 파서

라인 단위 파싱은 쓰지 않는다. 실측상 inner text 안에 개행이 들어간 요소가
존재하기 때문이다(200 샘플 중 61줄).

```xml
<p data-bbox="117 1324 795 1358">✍️How to create tasks?
⏰How to set reminders?
✅How to complete tasks?</p>
```

대신 정규식으로 태그 경계를 훑어 트리를 만든다. 속성 값 안의 `>` 도 안전하게
처리하기 위해 큰따옴표 구간을 통째로 소비한다.

```python
TAG_RE = re.compile(r'<\s*(/?)\s*([A-Za-z][\w:-]*)((?:"[^"]*"|[^<>"])*?)(/?)\s*>')
```

각 `Node` 는 **원본 XML 문자열 기준 문자 구간**을 갖는다.

| 필드 | 의미 |
|---|---|
| `span_full` | 서브트리 전체 (여는 태그 ~ 닫는 태그) |
| `span_text` | inner text 구간. **자식 태그가 없는 요소에만** 부여 |
| `span_aria` | `aria-label` 값(따옴표 안) 구간 |

`span_text` 를 리프에만 주는 이유: 컨테이너의 "내용"은 자식 요소이므로
텍스트 마스킹 대상이 아니다.

### Step 1 — 25 / 55 / 20 분할

- `sample_id` = **입력 파일의 0-based 순서 인덱스**. 입력을 재정렬하면 안 된다.
- `content_hash` = `sha1(assistant_value)[:12]`. 조인은 `sample_id` 로 하고
  해시는 검증용 assert 에만 쓴다.
- 분할 방식: 해시 모듈러가 아니라 **셔플-절단**. 정확한 개수를 보장한다
  (50k → 정확히 12,500 / 27,500 / 10,000).
- 샘플별 마스킹 난수는 `Random(f"{seed}:{sample_id}")` 로 독립 시딩 →
  일부만 재처리해도 결과가 동일.

### Step 2 — 마스킹 (`masked` 55%)

#### 후보 티어

의미 없는 레이아웃 컨테이너를 먼저 가리면 예산만 먹고 학습 신호가 없다.
실측상 `div` 2,143개가 속성 없는 순수 컨테이너다.

| Tier | 조건 |
|---|---|
| 1 (우선) | `tag ∈ {button, p, input}` **AND** (inner text 있음 OR aria-label 있음) |
| 2 | `tag ∈ {button, p, input, img}` (속성·텍스트 없는 아이콘) |
| 3 (최후) | 나머지 (`div` 등 컨테이너) |

Tier 1 → 2 → 3 순으로 소진한다.

#### 마스킹 방식 선택 — 누수 방지가 핵심

`inner text` 와 `aria-label` 은 대개 같은 정보를 담는다. 한쪽만 가리면
다른 쪽에서 답이 샌다:

```xml
<!-- 잘못된 마스킹: aria-label 이 답을 그대로 노출 -->
<p aria-label="Thursday, September 28, Next alarm: Thu 7:00 PM"><MASK/></p>
```

| 요소가 가진 것 | 허용 방식 |
|---|---|
| text + aria(마스킹 허용) | `whole` 0.60 / **`text+aria` 동시** 0.40 |
| text + aria(마스킹 불허) | **`whole` 만** |
| text 만 | `whole` 0.70 / `text` 0.30 |
| aria(허용) 만 | `whole` 0.70 / `aria` 0.30 |
| aria(불허) 만 / 아무것도 없음 | `whole` 만 |

`text+aria` 는 **원자적으로** 적용된다(한쪽만 잡히면 취소). 두 span 은 같은
`group` 번호로 묶여 메타에 기록된다.

#### aria-label 마스킹 허용 필터

`aria-label` 값의 약 21%는 스크린샷에서 복원 불가능하다. 이런 값을 가리면
**hallucination을 훈련시키는 셈**이 된다.

| | 예시 |
|---|---|
| 허용 (실측 79%) | `Back`, `Search`, `Settings`, `Emoji button`, `Q`, `Reviews`, `1 Color` |
| 불허 (21%) | `Inbox, tab 4 out of 5`, `8c451ae4-7d9d-4cb8-...`, `Mail Delivery Subsystem, Yesterday, unread...`, `Add to search query,nike shoes size 11` |

```python
def aria_maskable(v):
    return (len(v) <= 25 and len(v.split()) <= 3
            and ',' not in v and '@' not in v and '&#' not in v
            and not re.search(r'\.\w{2,4}\b', v))     # 파일 확장자
```

허용 aria-label은 XML 문자의 **8.2%** 를 차지한다. inner text 8.1%와 합쳐
부분 마스킹만으로 약 16%의 예산을 확보할 수 있다.

> **참고**: inner text만으로는 XML의 8%밖에 안 되므로, "text 40% / 요소 60%"
> 같은 고정 분할로는 목표 비율 0.2~0.8을 **물리적으로 달성할 수 없다.**
> 그래서 `r` 을 독립변수로 두고 방식을 섞는 구조를 택했다.

#### 예산 루프

```
r      = 0.2 + 0.6 × Beta(2, 1.6)        # 평균 ≈ 0.56 (--ratio-dist uniform 로 변경 가능)
budget = r × len(xml)

1차: Tier 1 → 2 → 3 순회, 각 노드에 방식 확률 배정 후 적용
2차: 예산이 남으면 큰 컨테이너부터 흡수(subsume)하여 목표에 접근
```

**흡수(subsumption)** 가 핵심이다. 단순 "겹치면 skip" 방식으로는 리프를 먼저
가린 뒤 그 부모 컨테이너를 영영 가릴 수 없어 달성률이 0.78에 그쳤다.
새 span이 기존 span들을 **완전히 포함**하면 기존 것을 제거하고 흡수하도록
바꿔 달성률이 **median 1.000 / p10 0.954** 로 올라갔다.

기타 안전장치:
- `min_depth=2`: 루트와 최상위 자식은 `whole` 마스킹 금지 (골격 보존)
- `overshoot=1.3`: 남은 예산의 1.3배를 넘는 span은 건너뜀
- 치환은 span 시작 오프셋 순으로 정렬해 한 번에 재조립 (오프셋 밀림 없음)

### Step 3 — 전체 드롭 (`dropped` 20%)

XML 본문을 `(none)` 으로 대체한다. 섹션 헤더는 남겨 다른 포맷과 레이아웃을
평행하게 유지한다.

### Step 4 — 프롬프트 재조립

#### (a) 이미지를 XML 앞으로 (`--image-first`, 기본 True)

원본은 `XML → Screenshot → Action` 순이었다. **causal attention에서 XML 토큰은
이미지를 볼 수 없다.** `<MASK/>` 의 표현이 계산되는 시점에 스크린샷이 컨텍스트에
없으므로, "가려진 부분을 이미지에서 복원하라"는 전제가 인코딩 단계에서
구조적으로 성립하지 않는다.

```
[Screenshot]
<image>

Current UI State (PARTIAL — hidden content is marked <MASK/> / [MASK]):
<div data-bbox="0 0 840 1876">
  ...
</div>

Action:
<action>{"action": "click", "coordinate": [420, 1531]}</action>
```

> **주의**: 다운스트림(`NEXT_ACTION_PREDICTION`) 포맷도 함께 바꿔야 입력 정합이
> 유지된다. 기존 실험과의 비교가 필요하면 `--no-image-first` 로 원본 순서 유지.
>
> **검증 필수**: 학습 프레임워크가 `<image>` 위치를 존중하는지 확인할 것.
> 일부 구현은 이미지를 항상 prefix로 강제 이동시킨다. 전처리 후 실제
> `input_ids` 를 덤프해 이미지 토큰이 XML 토큰보다 앞에 있는지 확인한다.

#### (b) 인라인 마커

`# Observability` 는 시스템 프롬프트 안에 있어 실제 `<MASK/>` 등장 지점과
수천 토큰 떨어져 있다. 마스킹 지점 바로 앞에 한 줄을 붙여 지역성을 확보한다.

| fmt | 마커 |
|---|---|
| `full` | `Current UI State (FULL):` |
| `masked` | `Current UI State (PARTIAL — hidden content is marked <MASK/> / [MASK]):` |
| `dropped` | `Current UI State (NOT PROVIDED):` |

**마스크 개수는 절대 쓰지 않는다** (`8 subtrees hidden` 같은 것은 정보 누수).

#### (c) `# Observability` 섹션

`# Action Space` 와 `# Output Format (STRICT)` 사이에 삽입한다.
**`# Output Format (STRICT)` 이후 구간은 세 포맷에서 바이트 단위로 동일**하다
(검증 C5).

```
# Observability
[masked] The current UI state is given as a screenshot plus a PARTIAL XML.
         Hidden content is replaced by <MASK/> (an element or its inner text)
         or [MASK] (an attribute value). These are placeholders, never real
         screen content. Resolve each placeholder from the screenshot first,
         then predict the resulting state. Never emit <MASK/> or [MASK]
         in your output.
```

부정 지시("do not copy")보다 **긍정 절차 지시**로 작성했다. `Never emit <MASK/>`
는 실제로 발생하는 실패 유형이므로 반드시 유지한다.

`# Given` 의 한 줄도 세 포맷 공통으로 교체된다:

```
- Current UI State is provided as html-style XML and a screenshot.
→ Current UI State is provided as a screenshot, optionally accompanied by
  html-style XML (see # Observability).
```

> **한계 인정**: `[full]` 의 "The XML describes the state BEFORE the action" 같은
> 문구가 복사를 막지는 못한다. 프롬프트 지시로 손실 지형은 바뀌지 않는다.
> `full` 슬라이스의 실제 방어는 diff 손실 가중치다.

### Step 5 — 센티넬 규약

| 위치 | 센티넬 | 이유 |
|---|---|---|
| 요소 전체 / inner text | `<MASK/>` | 요소 위치이므로 태그 형태가 자연스러움 |
| 속성 값 내부 | `[MASK]` | `aria-label="<MASK/>"` 는 well-formed 위반 |

새 special token을 vocab에 추가하지 **않는다**. 임베딩 리사이즈가 필요하고
초기값이 랜덤이라 초반 학습이 불안정해지며, LoRA 사용 시 `embed_tokens` 까지
학습 대상에 넣어야 한다. 평범한 문자열 센티넬로 충분하다.

---

## 4. diff-loss 파이프라인과의 연계

### 4.1 반드시 지킬 것: 저장은 assistant 단독 토크나이즈 기준

**diff 위치를 절대 시퀀스(prefix+assistant) 전체 토큰 인덱스로 저장하면 안 된다.**

diff 는 `raw` 에서 계산하는데 학습은 `applied` 로 한다. 마스킹 때문에 프롬프트
길이가 달라지므로(예: XML 4,188자 → 1,968자) 전체 시퀀스에서 assistant 턴의
시작 위치가 밀린다. raw 기준 전체 시퀀스 토큰 인덱스를 applied 에 그대로 쓰면
가중치가 통째로 어긋난다.

```
저장 형식:  sample.token_weights = list[float]
            길이 = tokenizer(assistant_text).input_ids 의 길이
```

**성립하는 이유**:
- 타깃(next XML = assistant) 은 세 포맷에서 바이트 동일하다 (검증 C2).
- `token_weight_builder_v2.build_token_weights()` 는 `future_html` 을 **standalone
  으로** 토크나이즈해서 assistant 부분 배열을 만든다 (prefix 내용에 무관).
- 따라서 raw 로 계산한 assistant 토큰 배열은 applied 에도 그대로 유효하다.

**LlamaFactory 통합 지점**: `patches/llamafactory/0001-diff-loss.patch` 가
`token_weights` 필드를 읽어 `labels != -100` 마스크(=assistant 구간) 경계로
배치 텐서에 정렬한다. 저장된 배열의 앞/뒤 오프셋 처리는 collator 가 담당한다
(`n_asst >= n_w` 이면 뒤에서 정렬, `n_asst < n_w` 이면 앞에서 자름 — chat template
prefix 나 truncation 대응).

### 4.2 2단 가중치 구성

헝가리안 파이프라인은 **raw** current XML 과 next XML 을 매칭해 next XML 의 각 요소를
ADDED / MODIFIED / UNCHANGED 중 하나로 분류한다. 가중치는 이 축 하나로만 결정한다:

```python
if diff_type in {"ADDED", "MODIFIED"}:  w = 1.0    # 학습 신호 집중
else:                                    w = 0.25   # UNCHANGED baseline (span 밖 토큰도 동일)
```

포맷과 무관하게 동일한 표를 쓴다. 타깃(next XML) 은 세 포맷에서 바이트 동일하므로(C2)
diff 판정과 assistant 토큰 배열이 세 포맷에서 동일하게 나오고, 조인 대상은
`sample_id` 하나다.

| fmt | diff | non-diff | 비고 |
|---|---|---|---|
| `full` | 1.0 | 0.25 | current XML 그대로 → Hungarian 자연스러움 |
| `masked` | 1.0 | 0.25 | current XML 이 masked 라 raw 로 계산해야 정확 |
| `dropped` | 1.0 | 0.25 | current XML 이 `(none)` 이라 raw 로 계산해야 정확 |

세 포맷 모두 안전하다:
- `full`: raw 와 applied 의 current 가 동일 → 어느 쪽으로 계산해도 같은 결과.
- `masked`: applied 로 계산하면 `<MASK/>` 로 사라진 요소가 next 에서 ADDED 로 오분류.
  → 항상 raw 로 계산해서 UNCHANGED 로 정확히 잡는다.
- `dropped`: applied 로 계산하면 current 가 비어 있어 next 전체가 ADDED → 학습 신호 균일화.
  → raw 로 계산하면 실제 변경 요소만 1.0 을 받는다.

**설계 결정**: 원래 3단 (0.4 / 0.2 로 입력 마스킹 여부에 따라 UNCHANGED 를 세분) 을
고려했으나 v2 헝가리안 파이프라인의 기본 2단으로 먼저 검증하기로 했다.
`meta.masked_elements` 는 가중치 계산에는 쓰이지 않는다 (검증 C8/C9 용 메타데이터).
확장이 필요해지면 `sample_id` 로 raw diff 결과와 applied 의 `meta.masked_elements` 를
붙일 수 있는 구조라 파일 포맷은 유지된다.

### 4.3 가중치 하한에 대한 경고

baseline 을 **0.05 로 내리는 것은 권하지 않는다.** 실측상 XML 토큰의 40.5%가
`data-bbox` 좌표이고, UNCHANGED 좌표는 "베끼면 되는 쓰레기"가 아니라
다음 화면의 렌더 골격이자 다운스트림 그라운딩의 근원이다. `0.25` 에서 시작해
**렌더/파싱 실패율을 보면서** 내릴 것.

### 4.4 매칭 임계값이 최대 리스크

헝가리안 매칭 임계값에 따라 diff 비율이 크게 흔들린다 (실측):

| 임계값 | diff 중앙값 | 매칭률 |
|---|---|---|
| 0.25 | 0.881 | 0.25 |
| **0.40** | **0.78** | **0.33** |
| 0.55 | 0.545 | 0.65 |
| 0.65 | 0.427 | 0.72 |

**0.35~0.45 구간을 쓰고, 무작위 30샘플의 매칭 결과를 육안 검증할 것.**
자동 지표로는 검증되지 않는다. 이것이 이 프로젝트에서 수익률이 가장 높은 30분이다.

### 4.5 `build_diff_targets.py` — 실행 방법

`build_wm_formats.py` 의 두 파일을 `sample_id` 로 조인해서 학습 입력을 만든다.

```bash
python scripts/diff_loss/build_diff_targets.py \
    --raw     out/data_split_raw.jsonl \
    --applied out/data_split_applied.jsonl \
    --output  out/data_split_weighted.jsonl \
    --model   Qwen/Qwen2.5-VL-7B-Instruct \
    --w-added 1.0 --w-modified 1.0 --w-unchanged 0.25 \
    --on-error fail
```

**동작**:
1. 두 파일을 로드하고 길이·`sample_id` 집합·`content_hash` 정합성 검증.
2. 각 샘플에 대해:
   - raw 의 user 메시지에서 원본 current XML 추출
   - `hungarian_diff_v2.classify_diff(current, next)` → 요소별 diff_type
   - `token_weight_builder_v2.build_token_weights(...)` → assistant 토큰 배열
   - prefix 부분 잘라내고 assistant 부분만 유지
3. applied 샘플 dict 에 `token_weights` 필드를 추가해서 출력.

**출력 스키마** (`*_split_weighted.jsonl`):
```json
{
  "messages": [...applied 그대로 (masked/dropped 반영됨)...],
  "images": [...],
  "sample_id": 9,
  "content_hash": "a3f2c1d4e5b6",
  "fmt": "masked",
  "meta": {...applied 그대로...},
  "token_weights": [1.0, 1.0, 0.25, 0.25, ...],  ← assistant 토큰 수와 길이 동일
  "_diff_counts": {"ADDED": 3, "MODIFIED": 2, "UNCHANGED": 15}
}
```

**부가 산출물**: `*.meta.json` sidecar — 모델·템플릿·가중치·집계 카운트 기록.

**옵션**:
- `--template qwen|llama3|default` — 모델명 자동 감지 결과를 오버라이드
- `--revision <sha>` — tokenizer commit SHA 고정 (재현성)
- `--on-error fail|uniform|skip`
  - `fail` (기본): 어떤 실패도 실행 중단 (fail-closed)
  - `uniform`: 실패 샘플에 균일 1.0 폴백, 성공 카운트에는 넣지 않음
  - `skip`: 실패 샘플을 출력에서 제외

**LlamaFactory 학습 설정**:
- `patches/llamafactory/0001-diff-loss.patch` 가 이미 적용되어 있어야 한다.
- `configs/lf_dataset/dataset_info.json` 에 이 파일을 가리키는 엔트리 추가.
- 학습 YAML 에 `use_diff_token_weighted_loss: true`.

---

## 5. 이 파이프라인이 하지 않는 것

| 항목 | 담당 |
|---|---|
| **무변화 전이 제거** (d < 0.02, 실측 4~8%) | **전처리 — 최우선. 여기서 안 함** |
| 헝가리안 매칭 / diff 요소 산출 | `build_diff_targets.py` (§4.5) |
| assistant 토큰 배열 → 배치 텐서 정렬 | LlamaFactory collator (`0001-diff-loss.patch`) |
| per-token weighted CE loss | LlamaFactory trainer (`0001-diff-loss.patch`) |
| `diff_ratio` 조건부 포맷 배정 | 미구현 (아래 확장 참고) |

### ⚠️ 무변화 전이 제거는 반드시 선행할 것

200 샘플 중 **완전 동일 7건(3.5%), 변경비율 < 0.05 인 것 16건(8%)** 이 있다.
액션 분포를 보면 `click` 12건, `swipe` 3건, `type` 1건 — 눌렀는데 아무 일도
일어나지 않은 케이스다. 50k 기준 2,000~4,000건.

이 샘플들은 **복사가 진짜 정답**이므로 모델에게 identity mapping을 직접
가르친다. 나머지 92%에서는 복사가 안 통하므로, **이 8%가 복사 편향의 주된
원인일 가능성이 높다.** 제거하거나 `no_op` 플래그로 분리한 뒤 이 파이프라인에
넣을 것.

---

## 6. 실행 옵션

```
--input               입력 JSONL (필수)
--outdir              출력 디렉터리 (필수)
--prefix              출력 파일 접두사 (기본: 입력 파일명)
--seed                난수 시드 (기본 42)

--ratio-full          기본 0.25
--ratio-masked        기본 0.55
--ratio-dropped       기본 0.20    (합이 1이 아니면 에러)

--ratio-lo            샘플별 마스킹 비율 하한 (기본 0.2)
--ratio-hi            상한 (기본 0.8)
--ratio-dist          beta | uniform (기본 beta = Beta(2,1.6) 스케일링)

--min-depth           whole 마스킹 허용 최소 depth (기본 2, 루트=0)
--overshoot           남은 예산 대비 허용 초과 배수 (기본 1.3)

--image-first         스크린샷을 XML 앞에 배치 (기본값)
--no-image-first      원본 순서(XML → 스크린샷) 유지
```

---

## 7. 검증

`validate_wm_formats.py` 는 11개 불변식을 검사한다. **학습 전 반드시 통과시킬 것.**

| ID | 검사 |
|---|---|
| C1 | 두 파일 1:1 대응 (길이·순서·`sample_id`·`content_hash`) |
| C2 | **타깃이 두 파일에서 바이트 동일** — diff 조인의 전제 |
| C3 | 포맷 비율이 설정값과 일치 (허용 오차 ±0.02) |
| C4 | human 레이아웃: `image_first` 설정대로 배치, 인라인 마커 존재 |
| C5 | system: `# Observability` 가 `# Output Format` 앞, 이후 구간은 3포맷 동일 |
| C6 | `dropped` 에 XML 잔존 없음 |
| C7 | `masked` XML 이 well-formed |
| C8 | `src_span` 이 raw 원본 XML 기준으로 정확 |
| C9 | 같은 요소의 text만 가리고 aria를 남긴 누수 없음 |
| C10 | 타깃에 `<MASK/>` / `[MASK]` 유출 없음 |
| C11 | 마스킹 목표 대비 달성률 (median ≥ 0.90) |

### 200 샘플 실행 결과

```
written 200 / input 200  (skipped 0)
fmt        : {'full': 50, 'masked': 110, 'dropped': 40}
mask kind  : {'whole': 1839, 'text': 132, 'aria': 101}
tier used  : {'1': 2310, '2': 379, '3': 165}
mask ratio : p10 0.340  p50 0.533  p90 0.701  mean 0.527
xml len    : 2499 -> 904 (median chars)
layout_fail=0  parse_fail=0  wellformed_fail=0  mask_leak=0

C1~C11  ALL PASS
C8  src_span 정합: 2072건 중 불일치 0
C11 마스킹 달성률 median 1.000 / p10 0.954
```

Tier 1이 전체 마스킹의 81%를 차지한다 — 의미 있는 요소 우선 정책이 의도대로 작동.

---

## 8. 학습 시 모니터링 지표

| 지표 | 목표 |
|---|---|
| **복사율** `token_match(pred, cur_xml)` | **평균 0.5 근처** (GT 자체가 0.508) |
| 포맷별 검증 손실 (`full` vs `dropped`) | 격차가 학습 중 **좁혀질 것** |
| 렌더/파싱 실패율 | 가중치 하한 0.2가 적절한지 판정 |
| 저-diff hard set (`d < 0.3`) 성능 | **진짜 지표** — 전체 평균에 묻히지 않게 분리 |
| 타깃 `<MASK/>` 유출률 | 0 |

**복사율이 0.8이면 여전히 복사 중이고, 0.2로 너무 낮으면 반대로 유지돼야 할
chrome까지 바꾸는 hallucination이다.** GT 분포와 같아지는 것이 정답이다.

---

## 9. 확장 아이디어 (미구현)

### 9.1 `diff_ratio` 조건부 포맷 배정

실측상 변경 비율 분포가 이봉형이다 (p25 = 0.46, p50 = 0.81, p75 = 0.91).
고-diff 샘플은 복사가 애초에 안 통하므로 전체 XML을 줘도 안전하고, 오히려
"직전에 어느 앱 어느 화면이었나"라는 유용한 문맥이 된다.

```python
if   d < 0.30: p_format = [0.10, 0.60, 0.30]   # 마스킹 집중
elif d < 0.70: p_format = [0.25, 0.55, 0.20]   # 기본
else:          p_format = [0.40, 0.45, 0.15]   # 전체 XML 관대하게
```

diff_ratio를 전처리에서 계산해 입력 JSONL에 넣어두면 조건문 하나로 구현된다.

### 9.2 다중 시드 마스킹

마스킹이 확률적이므로 같은 55% 슬라이스에 서로 다른 시드로 2~3벌을 만들면
epoch 간 다양성이 확보된다. 1.2억 토큰 규모에서는 사실상 공짜 증강이다.
(온라인 dataloader 마스킹이 더 낫지만, 오프라인 구조라면 이 방법이 실용적.)

```bash
for s in 42 43 44; do
  python build_wm_formats.py --input data.jsonl --outdir out_s$s --seed $s --prefix seed$s
done
```

### 9.3 액션 드롭아웃 (CFG용)

무변화 전이를 버리지 말고 `action: NONE` 으로 재라벨링해 5~8% 슬라이스로 두면,
추론 시 classifier-free guidance 가 가능해진다.

```
logits_final = logits(s, a) + w × [ logits(s, a) − logits(s, ∅) ]
```

`logits(s, ∅)` 는 정확히 "복사"이므로, 빼면 복사 성분이 상쇄되고 액션 특이적
변화만 증폭된다. 학습 데이터 추가 비용이 거의 0이라 문을 열어둘 만하다.
다만 자기회귀 LM에 대한 CFG는 GUI 월드 모델에서 검증된 사례가 없으므로
실험적 옵션으로 취급할 것.

### 9.4 look-ahead reasoning trace

XML을 쓰기 전에 "무엇이 바뀌는지"를 먼저 출력하게 하면 복사 경로가 출력 순서상
차단된다. gWorld는 이 분해가 5개 벤치마크 전부에서 우세했다고 보고했고,
WebWorld는 1,000개 CoT 샘플만으로 충분하다고 보고했다.

데이터 규모가 작을수록 문제 분해의 이득이 크므로, 마스킹만으로 부족하면
다음 후보로 고려할 것.

---

## 10. 알려진 제약

- **파싱 실패 시 `full` 로 강등**한다 (데이터 손실 방지). 리포트의
  `parse_fail_downgraded_to_full` 로 추적. 200 샘플 기준 0건.
- `key` 필드는 80자로 절단된다 (조인용 보조 키이므로 충분).
- `whole` 마스킹은 `min_depth=2` 이상에서만 허용되므로, 트리가 매우 얕은
  샘플에서는 목표 비율 달성률이 떨어질 수 있다. C11로 감지된다.
- 이미지 순서 변경의 효과 크기는 **원리에서 나온 가설**이며,
  GUI 월드 모델에서 ablation된 사례를 찾지 못했다. `--no-image-first` 와
  A/B 비교를 권한다.
