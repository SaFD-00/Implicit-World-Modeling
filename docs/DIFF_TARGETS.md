# diff-loss 학습 입력 생성 파이프라인

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


`build_wm_formats.py` 의 두 출력 파일을 헝가리안 파이프라인과 이어 붙여
**LlamaFactory 학습용 최종 JSONL** 을 만드는 단계에 대한 문서.

- **대상 스크립트**: `Implicit-World-Modeling/scripts/diff_loss/build_diff_targets.py` (신규)
- **관련 문서**: `README_wm_formats.md` §4 (설계 배경), `patches/llamafactory/0001-diff-loss.patch` (학습 통합)

---

## 1. 왜 이 스크립트가 필요한가

### 1.1 앞 단계에서 남긴 문제

`build_wm_formats.py` 는 하나의 원본 JSONL 을 세 가지 관측성 포맷(full/masked/dropped)
으로 분할해 두 파일을 낸다:

- `*_split_raw.jsonl`  — 원본 그대로 (헝가리안 diff 계산용)
- `*_split_applied.jsonl` — 마스킹·드롭 적용본 (실제 학습에 넣을 것)

이 시점에서 아직 **학습이 안 되는 상태**다. 두 가지 문제가 남아 있다.

**① diff 위치를 어디서 계산할 것인가**

anti-copy 학습의 핵심은 "변화한 요소(ADDED/MODIFIED)에 loss weight 를 몰아 주는 것"인데,
헝가리안 매칭은 `current XML` 과 `next XML` 을 비교해야 한다.

- `applied` 의 masked current 로 계산하면: `<MASK/>` 로 사라진 요소가 next 에서
  ADDED 로 오분류. 실제로는 UNCHANGED 인데 full weight (1.0) 를 받는다 → 학습 신호가
  오염됨.
- `applied` 의 dropped current 로 계산하면: current 가 `(none)` 이라 next 요소 전부가
  ADDED. 결국 균일 1.0 이 되어 anti-copy 신호가 소멸.

→ **diff 는 반드시 raw 의 원본 current 로 계산해야 한다.**

**② 계산한 diff 를 어떻게 applied 에 이식할 것인가**

학습은 applied 로 해야 anti-copy 훈련이 성립한다 (모델이 마스킹된 프롬프트를 보고
next 를 예측하도록 강제). 그런데 raw 와 applied 는 프롬프트 길이가 다르다:

```
raw       : system(500) + user(1200 XML) + assistant(300) = 2000 토큰
applied   : system(520 Observability 추가) + user(600 masked) + assistant(300) = 1420 토큰
                                                                    ↑ 여기부터가 학습 대상
```

raw 기준 "전체 시퀀스 토큰 인덱스 1750~1800에 가중치 1.0" 을 저장하면 applied 에
그대로 쓸 수 없다. assistant 시작 위치가 다르기 때문.

→ **저장은 assistant 를 standalone 으로 토크나이즈한 배열 (prefix 무관)** 로 해야 한다.

### 1.2 성립 조건

- `next XML` (= assistant 답) 은 세 포맷에서 **바이트 동일** (`build_wm_formats` 검증 C2).
- 그러므로 `next XML` 을 standalone 으로 토크나이즈한 assistant 토큰 배열은
  raw / applied / full / masked / dropped 어디에서 만들어도 동일하다.
- `token_weight_builder_v2.build_token_weights()` 는 이미 그렇게 동작한다 —
  `future_html` 을 독립적으로 토크나이즈해서 배열을 만든다.

이 성질이 파이프라인 전체를 가능하게 한다.

---

## 2. 이 스크립트가 하는 일 (한 문단 요약)

`_split_raw.jsonl` 과 `_split_applied.jsonl` 을 `sample_id` 로 조인해서, **raw 의
원본 XML 로 헝가리안 diff 를 계산**하고, 그 결과로 만든 **assistant 토큰 가중치
배열을 applied 샘플에 `token_weights` 필드로 붙여** 최종 학습 JSONL 을 낸다.
가중치는 2단 (ADDED/MODIFIED=1.0, UNCHANGED=0.25). LlamaFactory 의 diff-loss 패치가
이 필드를 자동으로 읽어 배치 텐서에 정렬하고 per-token weighted CE loss 를 계산한다.

---

## 3. 파이프라인 전체 그림

```
                     ┌──────────────────────────┐
  all_samples.jsonl  │ build_wm_formats.py      │
  (원본 ShareGPT)    │   • 25/55/20 분할        │
                     │   • 마스킹·프롬프트 재조립 │
                     └─────────┬────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
       *_split_raw.jsonl            *_split_applied.jsonl
       (원본 그대로)                  (masked/dropped 반영)
                 │                           │
                 └──────────┬────────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ validate_wm_formats.py     │
              │   C1~C11 불변식 검증        │
              └─────────────┬──────────────┘
                            │ (통과 필수)
                            ▼
              ┌────────────────────────────┐
              │ build_diff_targets.py ★    │
              │   • sample_id 로 조인       │
              │   • raw 로 Hungarian diff  │
              │   • assistant 토큰 가중치   │
              │   • applied 에 부착         │
              └─────────────┬──────────────┘
                            │
                            ▼
                *_split_weighted.jsonl
                (학습 최종 입력)
                            │
                            ▼
              ┌────────────────────────────┐
              │ LlamaFactory 학습          │
              │   patch: token_weights 자동 │
              │   처리, weighted CE loss   │
              └────────────────────────────┘
```

---

## 4. 무엇이 추가/수정되었나

### 4.1 신규 파일

| 경로 | 라인수 | 설명 |
|---|---|---|
| `Implicit-World-Modeling/scripts/diff_loss/build_diff_targets.py` | ~350 | 조인 + diff + token_weights 부착 |

**설계 원칙**:
- v2 파이프라인(`hungarian_diff_v2`, `token_weight_builder_v2`)은 **import 로 재사용**만 하고 손대지 않는다 — 하드 제약 9 (재현성 때문에 v1/v2 불가침).
- 실패 처리는 fail-closed 기본 (`--on-error fail`). 부분 산출물 방지 위해 tmp 파일 → atomic rename.
- 재현성을 위해 sidecar `*.meta.json` 에 모델·템플릿·가중치·집계 카운트 기록.

### 4.2 문서 수정

**`README_wm_formats.md`**:
- 상단 파일 표에 `build_diff_targets.py` 한 줄 추가.
- §4.1 오프셋 규약 정정: "assistant 턴 내부 문자 오프셋 저장" → "assistant standalone 토큰 배열 저장". 실제 v2 구현과 LF collator 통합 방식에 맞춤.
- §4.2 2단 가중치 표에 fmt 별 안전성 설명 추가 (full/masked/dropped 세 포맷 모두 raw diff 로 정확히 판정됨을 명시).
- §4.5 신설: `build_diff_targets.py` 실행 방법·옵션·출력 스키마.
- §5 담당 표 정정: 헝가리안·collator·loss 를 "별도 파이프라인" → 실제 파일 경로로 명시.

### 4.3 손대지 않은 것

| 파일 | 이유 |
|---|---|
| `hungarian_diff_v2.py` / `hungarian_metric_v2.py` / `token_weight_builder_v2.py` | v2 불가침 |
| `preprocess_dataset_v2.py` | 기존 단일 파일 워크플로우로 유효 |
| `patches/llamafactory/0001-diff-loss.patch` | 이미 `token_weights` 필드를 완벽히 처리 |
| `configs/lf_dataset/dataset_info.json` | 컨버터가 `token_weights` 를 필드명으로 직접 읽음 (columns 매핑 불필요) |

---

## 5. 사용법

### 5.1 실행 순서

**0단계 — (권장, 선행 필수) 무변화 전이 제거**

`build_wm_formats.py` **이전에** 무변화 전이(액션 후 화면이 안 바뀐 샘플) 를 원본 JSONL
에서 제거하거나 `no_op` 플래그로 분리한다. 실측 4~8% 존재하며, 이 8% 가 복사 편향의
주된 원인이다. `README_wm_formats.md` §5 참고. 이 스크립트에서 다루지 않는다 — 데이터셋
전처리 단계에서 해결해야 한다.

**1단계 — 세 포맷으로 분할**
```bash
python build_wm_formats.py \
    --input all_samples_state_pred.jsonl \
    --outdir out/ \
    --seed 42
```
→ `out/all_samples_state_pred_split_raw.jsonl` + `out/all_samples_state_pred_split_applied.jsonl`

**2단계 — 불변식 검증 (필수)**
```bash
python validate_wm_formats.py \
    --raw     out/all_samples_state_pred_split_raw.jsonl \
    --applied out/all_samples_state_pred_split_applied.jsonl
```
→ C1~C11 모두 PASS 여야 다음으로.

**3단계 — diff 계산 + token_weights 부착**
```bash
python Implicit-World-Modeling/scripts/diff_loss/build_diff_targets.py \
    --raw     out/all_samples_state_pred_split_raw.jsonl \
    --applied out/all_samples_state_pred_split_applied.jsonl \
    --output  out/all_samples_state_pred_split_weighted.jsonl \
    --model   Qwen/Qwen2.5-VL-7B-Instruct
```
→ `out/all_samples_state_pred_split_weighted.jsonl` (학습 최종 입력) + `.meta.json` (재현 메타)

### 5.2 CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--raw` | (필수) | `*_split_raw.jsonl` 경로 |
| `--applied` | (필수) | `*_split_applied.jsonl` 경로 |
| `--output` | (필수) | 출력 JSONL 경로 (raw/applied 와 같으면 거부) |
| `--model` | (필수) | HuggingFace 모델명 (토크나이저 로드용) |
| `--template` | 자동 감지 | `qwen` / `llama3` / `default` 중 강제 지정 |
| `--w-added` | 1.0 | ADDED 요소 토큰 가중치 |
| `--w-modified` | 1.0 | MODIFIED 요소 토큰 가중치 |
| `--w-unchanged` | 0.25 | UNCHANGED 및 span 밖 토큰 baseline |
| `--revision` | None | tokenizer commit SHA / 태그 고정 (재현성) |
| `--on-error` | `fail` | `fail` \| `uniform` \| `skip` |

**`--on-error` 처리**:
- `fail` (기본, 권장): 어떤 실패도 실행 중단. 부분 산출물 남기지 않음.
- `uniform`: 실패 샘플에 균일 1.0 폴백. 성공 카운트에 포함하지 않음. diff 강조 없이 그냥 SFT 로 학습되므로 **로그에서 fallback 건수를 반드시 확인**해야 한다.
- `skip`: 실패 샘플을 출력에서 제외. 데이터 손실이 발생하므로 신중하게 사용.

### 5.3 출력 스키마

```json
{
  "messages": [
    {"from": "system", "value": "...Observability 포함..."},
    {"from": "human",  "value": "...masked/dropped XML..."},
    {"from": "gpt",    "value": "...next XML (raw/applied 동일)..."}
  ],
  "images": ["..."],
  "sample_id": 9,
  "content_hash": "a3f2c1d4e5b6",
  "fmt": "masked",
  "meta": { ... build_wm_formats 의 masked_elements 등 ... },

  "token_weights": [1.0, 1.0, 0.25, 0.25, 1.0, ...],
  "_diff_counts": {"ADDED": 3, "MODIFIED": 2, "UNCHANGED": 15}
}
```

- **`token_weights`**: 길이 = `tokenizer(assistant_text).input_ids` 길이. LlamaFactory collator 가 배치 텐서에 자동 정렬.
- **`_diff_counts`**: 진단용. 학습 전 제거하거나 그대로 둬도 무방.

### 5.4 부가 산출물: `*.meta.json`

```json
{
  "raw_input": ".../data_split_raw.jsonl",
  "applied_input": ".../data_split_applied.jsonl",
  "output": ".../data_split_weighted.jsonl",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "template": "qwen",
  "metric_version": "v2",
  "weight_map": {"ADDED": 1.0, "MODIFIED": 1.0, "UNCHANGED": 0.25},
  "on_error": "fail",
  "counts": {"total": 50000, "ok": 49998, "action": 0, "written": 49998, ...},
  "fmt_counts": {"full": 12500, "masked": 27500, "dropped": 10000},
  "diff_totals": {"ADDED": 132450, "MODIFIED": 89320, "UNCHANGED": 421150}
}
```

---

## 6. 학습에 넣기 (LlamaFactory)

### 6.1 사전 조건
- LlamaFactory 워킹트리에 `patches/llamafactory/0001-diff-loss.patch` 가 적용되어 있어야 한다. 미적용이면 `token_weights` 필드가 무시되고 그냥 SFT 로 학습된다.

### 6.2 데이터셋 등록

`data/dataset_info.json` (LlamaFactory 쪽) 에 엔트리 추가:
```json
"IWM-STATE_weighted": {
  "file_name": "../../data/all_samples_state_pred_split_weighted.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system"
  }
}
```
> **주의**: `token_weights` 는 `columns` 매핑에 넣지 않는다. 패치된 컨버터가 필드명으로 직접 읽는다.

### 6.3 학습 YAML

```yaml
dataset: IWM-STATE_weighted
use_diff_token_weighted_loss: true
# ... 나머지 SFT 설정 ...
```

### 6.4 학습 시 동작 (내부)

1. `SharegptDatasetConverter` 가 `token_weights` 를 `_token_weights` 로 통과시킴.
2. `SupervisedDatasetProcessor` 가 `model_inputs["token_weights"]` 에 적재.
3. `MultiModalDataCollatorForSeq2Seq` 가 배치 텐서 생성 시:
   - `labels != -100` 마스크로 assistant 구간을 찾음.
   - `n_asst >= n_w`: 뒤에서 정렬 (chat template `<think>` prefix 등 대응).
   - `n_asst < n_w`: 앞에서 자름 (cutoff truncation 대응).
4. `CustomSeq2SeqTrainer.compute_loss()` 가 `diff_token_weighted_loss_func()` 호출:
   - shift 후 per-token cross-entropy 계산.
   - `token_weights` 를 곱함.
   - `valid_mask` (labels != -100) 로 정규화.

---

## 7. 세 포맷의 처리

**최종 산출물은 파일 3개**: `*_split_raw.jsonl`, `*_split_applied.jsonl`, `*_split_weighted.jsonl`.
세 포맷(full/masked/dropped) 은 각 파일 안에 `fmt` 필드로 섞여 있다.

**세 포맷 모두 정확한 diff 를 받는다**:

| fmt | applied 의 current | 이 스크립트의 처리 |
|---|---|---|
| `full` | 원본 XML 그대로 | raw 로 계산 (raw==applied 이므로 결과 동일) |
| `masked` | `<MASK/>` 섞임 | raw 로 계산 → UNCHANGED 요소가 오분류되지 않음 |
| `dropped` | `(none)` | raw 로 계산 → 실제 변경 요소만 1.0 |

fmt 별로 분리하고 싶다면 사후에 `jq` 로 필터링 가능:
```bash
jq -c 'select(.fmt == "masked")' out/data_split_weighted.jsonl > out/masked_only.jsonl
```

---

## 8. 검증 체크리스트 (학습 전)

1. **0단계 선행 확인**: 원본 JSONL 에서 무변화 전이가 제거/분리되었는가.
2. **1단계 산출물**: `_report.json` 의 `RESULT: OK` 확인.
3. **2단계 산출물**: `validate_wm_formats.py` 출력에서 C1~C11 ALL PASS 확인.
4. **3단계 산출물**:
   - `_split_weighted.jsonl` 라인 수 == applied 라인 수.
   - `.meta.json` 의 `counts.written` == `counts.total` (fallback 0건).
   - `.meta.json` 의 `fmt_counts` 가 build_wm_formats 의 비율과 일치.
   - 임의 샘플 하나를 뽑아 `len(token_weights) == len(tokenizer(assistant_text).input_ids)` 확인.
5. **학습 시 첫 배치**: `token_weights` 텐서가 배치에 들어오는지, `labels==-100` 구간이 0 으로 마스킹되었는지 확인.

---

## 9. 알려진 제약과 향후 확장

### 9.1 제약

- **매칭 임계값 리스크**: `hungarian_metric_v2.MATCH_THRESHOLD` 값에 따라 diff 비율이 크게 흔들린다. 0.35~0.45 를 쓰고 무작위 30샘플 육안 검증 필수 (`README_wm_formats.md` §4.4 참고).
- **action 샘플 폴백**: build_wm_formats 는 state_pred 전용이지만, 이미지 개수 규약 위반이나 assistant 가 `<action>` 으로 시작하면 균일 1.0 처리 (안전망).
- **tokenizer offset_mapping 의존**: `return_offsets_mapping=True` 를 지원하는 fast tokenizer 여야 한다. Slow tokenizer 는 `token_weight_builder_v2` 가 실패한다.

### 9.2 확장 (필요해지면)

- **3단 가중치**: 원래 설계 (README §4.2 결정 노트) 는 UNCHANGED 를 masked(0.4) / visible(0.2) 로 세분하는 것이었다. 지금은 v2 기반 2단으로 갔지만, applied 의 `meta.masked_elements` 는 그대로 남아 있어 `sample_id` 로 조인하면 언제든 재구성 가능하다. 파일 포맷은 유지된다.
- **v3 유도성 축**: `hungarian_diff_v3.classify_derivability()` 를 얹어 NON_DERIVABLE 콘텐츠 (서버가 내려준 상품명·가격) 를 감쇠. 현재 build_diff_targets 는 v2 전용.
- **다중 시드 마스킹**: 오프라인 데이터 증강용. `build_wm_formats` 를 시드 여러 개로 돌리고, 각 시드마다 이 스크립트를 다시 돌리면 된다.
