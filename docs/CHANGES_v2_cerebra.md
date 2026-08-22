# diff_loss/ 변경 이력 — Cerebra 스키마 지원

이 디렉터리의 v2 파이프라인이 원래 **Android UI 스키마**만 지원했으나,
Cerebra/Monkey-Collector 데이터셋(`data-bbox`, `aria-label`) 도 처리하도록 확장했다.

- 원본 v2 는 "하드 제약 9: v1/v2 불가침"이 있으나, 사용자 지시에 따라 **하위 호환을
  유지한 채** 두 스키마 모두 지원하도록 수정.
- v3 는 별도 metric 파일(`hungarian_metric_v3.py`) 을 쓰므로 영향 없음.

---

## 파일 요약

| 파일 | 상태 | 라인 변경 |
|---|---|---|
| `hungarian_metric_v2.py` | **수정** | `_collect_texts`, `extract_elements`, `_parse_bounds_center`, `_bounds_dist` |
| `token_weight_builder_v2.py` | **수정** | `_scan_tag_spans`(내부), `get_element_char_spans`, `_make_el_key` |
| `build_diff_targets.py` | **신규** | 전체 (약 350 라인) |

---

## 1. `hungarian_metric_v2.py` (수정)

### 배경

v2 는 매칭용 element 를 뽑을 때 다음을 가정했다:

```python
INTERACTIVE_TAGS = {"button", "input", "a", "select", "textarea"}
CONTENT_TAGS = {"p", "img", "span"}
CLICKABLE_ATTRS = {"clickable", "long-clickable"}
is_described = bool(el.get("description"))
```

- 위치 축: `bounds="[x1,y1][x2,y2]"` (Android dumpsys 형식)
- 텍스트 축: `description`, `id`, `text`, `aria-label`
- element 채택: 위 태그 세트 + `clickable` 또는 `description` 있는 것

Cerebra 데이터의 실제 스키마 (`structured_xml_parser_v2_htmllike.py` 참조):

- 태그: `div`, `button`, `input`, `p`, `img`
- 위치 축: **`data-bbox="x1 y1 x2 y2"`** (공백 구분, 대괄호 없음)
- 텍스트 축: **`aria-label`** (Android `content-desc` 정규화), `alt` (img), `placeholder`/`value` (input)
- `clickable` 속성 없음 — button 태그 자체가 클릭 가능성을 함의

이 미스매치로 인해:
1. `bounds` 파싱 실패 → 위치 신호가 0 이 되어 매칭 정확도 저하 (텍스트만으로 매칭)
2. `_collect_texts` 가 `alt`/`placeholder`/`value` 를 못 뽑아 img/input 신호 누락

### 변경 (3곳)

**(a) `_collect_texts` — 텍스트 원천 추가**

```diff
 add(el.get("description"))
 add(el.get("id"))
 add(el.get("text"))
 add(el.get("aria-label"))
+add(el.get("alt"))
+add(el.get("placeholder"))
+add(el.get("value"))
```

`token_weight_builder_v2` 의 `has_text` 판정도 이 규약을 그대로 따르도록 함께
수정 (아래 §2 참조).

**(b) `extract_elements` — 채택 조건 확장 + element dict 에 두 위치 필드 저장**

```diff
-is_described = bool(el.get("description"))
+is_described = bool(el.get("description")) or bool(el.get("aria-label"))

 elements.append({
     "tag": tag,
     "text": text,
     "bounds": el.get("bounds", "") or "",
+    "data-bbox": el.get("data-bbox", "") or "",
+    "index": el.get("index", "") or "",
 })
```

`data-bbox` / `index` 는 이후 `_make_el_key` (token_weight_builder) 가 매칭 키로 쓴다.

**(c) `_parse_bounds_center` — 좌표 파서 두 형식 지원**

```diff
 def _parse_bounds_center(s):
     if not s:
         return None
     m = re.findall(r"\[(-?\d+),(-?\d+)\]", s)   # Android
     if len(m) >= 2:
         ...
-    return None
+    # Cerebra: "x1 y1 x2 y2"
+    parts = s.split()
+    if len(parts) == 4:
+        try:
+            x1, y1, x2, y2 = (int(p) for p in parts)
+            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
+        except ValueError:
+            return None
+    return None
```

**(d) `_pos_key` — 위치 축 헬퍼 추가** (신규)

```python
def _pos_key(el):
    return el.get("bounds", "") or el.get("data-bbox", "") or ""
```

`_bounds_dist` 가 이 헬퍼를 통해 bounds/data-bbox 를 자동으로 흡수한다.

### 하위 호환

- Android 데이터는 여전히 `bounds` / `description` 을 갖고 있고, 위 확장은
  **추가 조건**이므로 기존 동작을 그대로 유지한다.
- 다만 `_collect_texts` 에 `alt`/`placeholder`/`value` 가 추가되어 Android 에서도
  이 속성을 가진 요소는 텍스트 신호가 커진다 (Android 원본은 대개 이 속성이 없어
  실질적 영향은 없음).

---

## 2. `token_weight_builder_v2.py` (수정)

### 배경

이 모듈은 세 가지 역할을 한다:

1. **원본 XML 스캔** (`_scan_tag_spans` 내부 정규식) — element 별 char (start, end) 추출
2. **BeautifulSoup element 필터링** (`get_element_char_spans`) — 어떤 element 를 매칭 대상으로 삼을지
3. **element key 생성** (`_make_el_key`) — diff_result 와 char_span 을 매핑할 키

이 세 지점 모두 스키마 종속이었다. 특히 원본 스캔 정규식은
`\bindex="..."` 또는 `\bbounds="..."` 만 검색해서 Cerebra 의 `data-bbox` 를 놓치고
있었고, element key 도 `bounds` 없으면 `(tag, "")` 로 붕괴하여 여러 element 가
하나로 합쳐졌다.

### 변경 (3곳)

**(a) 원문 스캔 (`get_element_char_spans` 내부 Step 1) — `data-bbox` 지원**

```diff
 idx_m = re.search(r'\bindex="(-?\d+)"', attrs_str)
 bounds_m = re.search(r'\bbounds="([^"]+)"', attrs_str)
+dbbox_m = re.search(r'(?:^|\s)data-bbox="([^"]+)"', attrs_str)

-if idx_m is not None:
-    key = (tag_name, int(idx_m.group(1)))
-elif bounds_m is not None:
-    key = (tag_name, bounds_m.group(1))
+if idx_m is not None:
+    key = (tag_name, "i:" + idx_m.group(1))
+elif bounds_m is not None:
+    key = (tag_name, "b:" + bounds_m.group(1))
+elif dbbox_m is not None:
+    key = (tag_name, "d:" + dbbox_m.group(1))
 else:
     continue
```

키에 **접두** (`i:` / `b:` / `d:`) 를 붙여 서로 다른 축의 값이 우연히 문자열 충돌하지
않게 한다. `\b` 는 하이픈 경계에서 잘리므로 `data-bbox` 는 명시적으로 `(?:^|\s)`
로 경계를 잡는다.

**(b) BeautifulSoup 필터링 (Step 2) — 스키마 확장**

```diff
 has_text = (
     bool(text)
     or bool(el.get("description"))
     or bool(el.get("id"))
     or bool(el.get("text"))
+    or bool(el.get("aria-label"))
+    or bool(el.get("alt"))
+    or bool(el.get("placeholder"))
+    or bool(el.get("value"))
 )
 is_content = (tag in CONTENT_TAGS) and has_text
-is_described = bool(el.get("description"))
+is_described = bool(el.get("description")) or bool(el.get("aria-label"))

 # 키 생성 (Step 1 과 동일한 규약)
+dbbox = el.get("data-bbox", "") or ""
 if idx != -1:
-    key = (tag, idx)
+    key = (tag, "i:" + str(idx))
 elif bounds:
-    key = (tag, bounds)
+    key = (tag, "b:" + bounds)
+elif dbbox:
+    key = (tag, "d:" + dbbox)
 else:
     continue
```

element info dict 에도 `data-bbox` 필드 추가.

**(c) `_make_el_key` — 3단 폴백 통일**

```diff
 def _make_el_key(el):
-    if el.get("index", -1) != -1:
-        return (el["tag"], el["index"])
-    if el.get("bounds"):
-        return (el["tag"], el["bounds"])
-    return (el["tag"], el["text"])
+    idx = el.get("index", -1)
+    if isinstance(idx, int) and idx != -1:
+        return (el["tag"], "i:" + str(idx))
+    if isinstance(idx, str) and idx.strip():
+        return (el["tag"], "i:" + idx.strip())
+    if el.get("bounds"):
+        return (el["tag"], "b:" + el["bounds"])
+    if el.get("data-bbox"):
+        return (el["tag"], "d:" + el["data-bbox"])
+    return (el["tag"], "t:" + el.get("text", ""))
```

- `extract_elements` (`hungarian_metric_v2`) 는 `index` 를 빈 문자열로 채우므로
  int / str 두 형태를 모두 방어.
- 접두 규약이 Step 1/Step 2 와 완벽히 일치해야 diff_result → char_span 매칭이 성립.

### 하위 호환

- Android 데이터는 `index` 또는 `bounds` 축을 갖는다. 두 축의 접두 규약은
  **키 문자열 구조만 바뀌었을 뿐** diff_result 와 char_span 이 **같은 규약**을
  쓰므로 매칭은 그대로 성립.
- 즉 Android 데이터에서 element 채택 개수·매칭 결과는 변화 없음 (텍스트 원천만 넓어짐).

---

## 3. `build_diff_targets.py` (신규)

### 역할

`build_wm_formats.py` 의 두 출력 파일 (`*_split_raw.jsonl`, `*_split_applied.jsonl`)
을 `sample_id` 로 조인해서, 다음을 수행한 최종 학습 JSONL 을 만든다:

1. raw 의 원본 XML 로 헝가리안 diff 계산 (`hungarian_diff_v2.classify_diff`)
2. next XML 을 standalone 으로 토크나이즈해 assistant 토큰 가중치 배열 생성
   (`token_weight_builder_v2.build_token_weights`)
3. applied 샘플 dict 에 `token_weights` 필드를 부착

### 왜 필요한가

`build_wm_formats.py` 는 세 fmt (full/masked/dropped) 로 데이터를 분할했지만,
diff 계산이 안 된 상태다. 그리고:

- **diff 는 raw 로 계산해야 한다** — masked/dropped 의 current XML 로 계산하면
  UNCHANGED 요소가 사라진 채로 매칭되어 ADDED 로 오분류.
- **학습은 applied 로 해야 한다** — 그게 anti-copy 훈련의 요점.

이 둘을 연결하는 것이 이 스크립트다. 성립하는 이유는 next XML 이 두 파일에서
바이트 동일하고 (C2 불변식), `build_token_weights` 는 assistant 를 standalone 으로
토크나이즈하므로 raw 로 계산한 token 배열이 applied 에도 그대로 유효하기 때문.

### 주요 로직 흐름

```
raw_sample  ─┐
            ├── sample_id 조인 (정합성 검증: sample_id + content_hash)
applied_sample ─┘
             │
             ▼
raw user 에서 원본 current XML 추출  (build_wm_formats 레이아웃 규약: "Current UI State:" ~ "[Screenshot]")
             │
             ▼
hungarian_diff_v2.classify_diff(current, next)   → 요소별 ADDED/MODIFIED/UNCHANGED
             │
             ▼
token_weight_builder_v2.build_token_weights(...)
    - prefix_text 로 raw system+user 를 만들고 (assistant 부분에는 무관)
    - future_html 을 standalone 토크나이즈
    - 요소별 weight 를 토큰 인덱스로 이식
             │
             ▼
prefix 부분 (0.0) 잘라내고 assistant 부분만 남김
             │
             ▼
applied sample dict 에 `token_weights` 필드 추가 → 출력
```

### CLI

```bash
python build_diff_targets.py \
    --raw     out/data_split_raw.jsonl \
    --applied out/data_split_applied.jsonl \
    --output  out/data_split_weighted.jsonl \
    --model   Qwen/Qwen2.5-VL-7B-Instruct \
    --w-added 1.0 --w-modified 1.0 --w-unchanged 0.25 \
    --on-error fail
```

### 안전장치

- **정합성 사전 검증**: 두 파일의 라인 수·`sample_id` 집합·per-sample `content_hash`.
- **`--on-error fail` 기본**: 어떤 실패도 실행 중단 (부분 산출물 방지). tmp → atomic rename.
- **`--on-error uniform`**: 실패 샘플에 균일 1.0 폴백 (성공 카운트 미포함, 로그 필수).
- **action 샘플 안전망**: `images` 개수가 1이 아니거나 assistant 가 `<action>` 으로
  시작하면 균일 1.0 (build_wm_formats 은 state_pred 전용이지만 방어).
- **부가 산출물**: `<output>.meta.json` — 모델·템플릿·가중치·집계 카운트 기록.

### 의존성

- 프로젝트 내부: `hungarian_diff_v2`, `hungarian_metric_v2`, `token_weight_builder_v2`
- 외부: `transformers` (fast tokenizer with `return_offsets_mapping`), `beautifulsoup4`,
  `scipy` 또는 `munkres` (선택)

---

## 스키마 확장이 v2 매칭 결과에 미친 영향 (실측)

**Cerebra smoke 2000 샘플** 재실행 결과:

| 지표 | 스키마 확장 전 | 확장 후 |
|---|---|---|
| ADDED (총) | 21,767 | 24,053 |
| MODIFIED (총) | **18,957** | **23,909** ← 위치 신호 살아남 |
| UNCHANGED (총) | 24,048 | 21,148 |
| token_weights 1.0 배정 비율 | **0.0%** (조용한 실패) | **58~60%** ✓ |
| char span 매칭 성공 | 0건 | 정상 |

MODIFIED 가 크게 증가한 이유: 이전에는 위치 신호가 죽어서 (bounds 파싱 실패)
같은 자리에 있는 요소를 매칭하지 못하고 ADDED/UNCHANGED 로 흩어졌으나, 확장 후
좌표 거리 축이 살아나 같은 위치의 텍스트 변경을 MODIFIED 로 정확히 잡음.

---

## 관련 파일

- 상위 문서: `../../../README_diff_targets.md`, `../../../README_wm_formats.md`
- LlamaFactory 통합: `../../patches/llamafactory/0001-diff-loss.patch`
- 이전 세대 (참고용, v2 fallback 로직 확인 시): `hungarian_metric.py`, `token_weight_builder.py` (v1)
- 유도성 축 확장 (별도 파이프라인): `hungarian_metric_v3.py`, `hungarian_diff_v3.py`, `token_weight_builder_v3.py`
