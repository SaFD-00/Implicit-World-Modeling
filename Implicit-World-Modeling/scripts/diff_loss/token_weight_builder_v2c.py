"""
token_weight_builder.py
────────────────────────────────────────────────────────────────
classify_diff() 결과를 실제 토큰 위치의 가중치 배열로 변환하는 모듈.

핵심 흐름:
  future_html (문자열)
      ↓  get_element_char_spans()   [★ search_start 방식으로 중복 element 버그 수정]
  (char_start, char_end, element_info) 리스트
      ↓  tokenizer offset_mapping
  token 인덱스 범위
      ↓  diff_type → WEIGHT_MAP
  token_weights 배열  [prefix: 0.0, assistant: diff_type별 가중치]

외부에서 호출하는 함수:
  build_token_weights(tokenizer, system, user, future_html, diff_result, prefix_text)
      -> list[float]

[v2c = v2 + Cerebra 스키마]  이 파일은 `token_weight_builder_v2.py` 의 **복제본**이다.
`token_weight_builder_v2.py` 자체는 EXP02/05/06/07 학습 데이터의 생성기라 재현성을 위해 불가침이므로
(AGENTS 하드 제약 9, 2026-08-21 결정 #2), Cerebra(`data-bbox`/`aria-label`) 확장은
이 `_v2c` 파일에만 넣는다. AC_EXP08 파이프라인(`build_diff_targets.py`)만 이걸 import 하고,
기존 `preprocess_dataset_v2.py` 경로는 그대로 `_v2` 를 쓴다.
변경 내역: `docs/CHANGES_v2_cerebra.md`
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

# ── diff_type별 기본 가중치 ────────────────────────────────────────────────
# 필요에 따라 preprocess_dataset.py에서 오버라이드 가능
WEIGHT_MAP: dict[str, float] = {
    "ADDED": 1.0,  # 새로 생긴 element → full loss
    "MODIFIED": 1.0,  # 변경된 element    → full loss
    "UNCHANGED": 0.25,  # 동일한 element    → 감쇠 (baseline)
}

INTERACTIVE_TAGS = {"button", "input", "a", "select", "textarea"}
CONTENT_TAGS = {"p", "img", "span"}
CLICKABLE_ATTRS = {"clickable", "long-clickable"}

# [v2c 구조축] hungarian_metric_v2c 의 STRUCTURE_TAGS / ADOPT_STRUCTURE 와 **반드시 동일**해야
# 한다. 어긋나면 diff_result 의 element key 와 char_span 의 key 가 안 맞아 에러 없이 전부
# baseline 으로 방치된다. 구조축으로만 채택된 요소는 서브트리가 아니라 **여는-태그 구간**에만
# 가중치를 준다 (루트 div 의 서브트리 = 문서 전체라 diff 신호가 소멸한다).
STRUCTURE_TAGS = {"div"}
ADOPT_STRUCTURE = True


# ── char offset 추출 ──────────────────────────────────────────────────────


def _find_closing_tag_end(html: str, tag: str, after: int) -> int:
    """매칭되는 closing tag(</tag>)의 끝 위치를 반환. 같은 태그 중첩 처리."""
    open_pat = re.compile(rf"<{re.escape(tag)}\b[^>]*?(?<!/)>")
    close_pat = re.compile(rf"</{re.escape(tag)}\s*>")

    depth = 1
    pos = after
    while depth > 0:
        next_open = open_pat.search(html, pos)
        next_close = close_pat.search(html, pos)

        if next_close is None:
            return after  # closing tag 없음 → opening tag 끝 반환

        if next_open is not None and next_open.start() < next_close.start():
            depth += 1
            pos = next_open.end()
        else:
            depth -= 1
            pos = next_close.end()

    return pos


def get_element_char_spans(html: str) -> list[tuple[int, int, dict]]:
    """
    future HTML에서 각 interactive/content/clickable element의
    (char_start, char_end) 오프셋을 추출한다.

    ★ regex 기반 위치 탐색으로 BeautifulSoup str(el) 재구성 문제 회피
    ─────────────────────────────────────────────────────────────────
    문제:
        html.parser가 self-closing 태그(<tag ... />)를 <tag ...></tag>로
        변환하고 속성 순서도 재배치하기 때문에, str(el)이 원본 HTML과 달라져
        html.find(str(el))가 -1을 반환하는 버그가 있었다.

    해결:
        원본 HTML에서 regex로 index 속성을 가진 태그의 위치를 직접 탐색.
        (tag, index) 쌍이 고유하므로 정확한 위치를 찾을 수 있다.
        BeautifulSoup은 element 필터링(interactive/content/clickable)에만 사용.
    ─────────────────────────────────────────────────────────────────

    Returns:
        [(char_start, char_end, {"tag": str, "index": int, "text": str}), ...]
        char_start 기준 정렬됨.
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Step 1: regex로 원본 HTML에서 (tag, id_key) → 위치 맵 구축 ──────
    # [Cerebra 확장] `data-bbox` 도 위치 축으로 인정한다. `\b` 는 하이픈에서 잘리므로
    # 명시적으로 태그명 뒤 공백/따옴표 경계로 매칭한다.
    pos_map: dict[tuple, tuple[int, int]] = {}
    open_map: dict[tuple, tuple[int, int]] = {}  # 여는-태그 구간만 (구조축 전용)

    for m in re.finditer(r"<(\w+)\b([^>]*?)\s*(/?)>", html):
        tag_name = m.group(1)
        attrs_str = m.group(2)
        self_closing = m.group(3) == "/"

        idx_m = re.search(r'\bindex="(-?\d+)"', attrs_str)
        bounds_m = re.search(r'\bbounds="([^"]+)"', attrs_str)
        dbbox_m = re.search(r'(?:^|\s)data-bbox="([^"]+)"', attrs_str)

        # index → bounds → data-bbox 순으로 폴백. 두 스키마를 하나의 pos_map 에 흡수.
        if idx_m is not None:
            key = (tag_name, "i:" + idx_m.group(1))
        elif bounds_m is not None:
            key = (tag_name, "b:" + bounds_m.group(1))
        elif dbbox_m is not None:
            key = (tag_name, "d:" + dbbox_m.group(1))
        else:
            continue  # 식별 불가

        if key in pos_map:
            continue  # 첫 번째 출현만 사용

        start = m.start()
        if self_closing:
            end = m.end()
        else:
            end = _find_closing_tag_end(html, tag_name, m.end())

        pos_map[key] = (start, end)
        open_map[key] = (start, m.end())

    # ── Step 2: BeautifulSoup으로 대상 element 필터링 후 위치 매핑 ─────
    spans: list[tuple[int, int, dict]] = []

    for el in soup.find_all(True):
        tag = el.name

        is_interactive = tag in INTERACTIVE_TAGS
        is_clickable = any(el.get(a) for a in CLICKABLE_ATTRS)
        text = el.get_text(strip=True)
        # hungarian_metric._collect_texts 와 동일한 텍스트 원천 규약을 따른다.
        has_text = (
            bool(text)
            or bool(el.get("description"))
            or bool(el.get("id"))
            or bool(el.get("text"))
            or bool(el.get("aria-label"))
            or bool(el.get("alt"))
            or bool(el.get("placeholder"))
            or bool(el.get("value"))
        )
        is_content = (tag in CONTENT_TAGS) and has_text
        # Cerebra: content-desc 는 aria-label 로 정규화되어 있다.
        is_described = bool(el.get("description")) or bool(el.get("aria-label"))
        is_structure = (
            ADOPT_STRUCTURE
            and tag in STRUCTURE_TAGS
            and bool(el.get("data-bbox") or el.get("bounds"))
        )
        # 자식 element 를 가진 요소는 **여는-태그 구간만** 가중한다 (서브트리 금지).
        # 요소의 "자기 내용"은 태그+속성이고 자식은 각자 element 로 따로 가중치를
        # 받으므로, 서브트리를 주면 문서 전체를 감싼 래퍼 하나의 판정이 나머지 전부를
        # 덮어써서 diff 신호가 소멸한다. Cerebra 스키마는 div 뿐 아니라 button 도
        # 컨테이너로 쓰기 때문에 태그명이 아니라 **자식 유무**로 판정한다.
        # (2026-08-22 실측 200 샘플: 이 규칙 없이는 6건이 루트 래퍼 때문에 토큰
        #  100% 가 1.0 이 됐다 — UNCHANGED 요소가 22개인 샘플 포함.)
        container_span = el.find(True) is not None

        if not (is_interactive or is_content or is_clickable or is_described or is_structure):
            continue

        idx = -1
        try:
            idx = int(el.get("index", -1))
        except (TypeError, ValueError):
            pass

        bounds = el.get("bounds", "") or ""
        dbbox = el.get("data-bbox", "") or ""

        # 위 Step 1 과 동일한 우선순위·prefix 규약으로 키 생성
        if idx != -1:
            key = (tag, "i:" + str(idx))
        elif bounds:
            key = (tag, "b:" + bounds)
        elif dbbox:
            key = (tag, "d:" + dbbox)
        else:
            continue

        if key not in pos_map:
            continue

        char_start, char_end = (open_map if container_span else pos_map)[key]
        spans.append(
            (
                char_start,
                char_end,
                {
                    "tag": tag,
                    "index": idx,
                    "text": text,
                    "bounds": bounds,
                    "data-bbox": dbbox,
                    "container_span": container_span,
                },
            )
        )

    spans.sort(key=lambda x: x[0])
    return spans


# ── element key 빌더 ──────────────────────────────────────────────────────


def _make_el_key(el: dict) -> tuple:
    """
    diff_result 의 element 와 char_span 의 element 를 매핑할 key.

    우선순위: index → bounds → data-bbox → text. 각 축은 서로 다른 접두를 붙여
    문자열이 우연히 충돌하지 않게 한다 (`b:` bounds, `d:` data-bbox).
    """
    idx = el.get("index", -1)
    # extract_elements 는 빈 문자열로 채우므로 정수·문자열 둘 다 방어
    if isinstance(idx, int) and idx != -1:
        return (el["tag"], "i:" + str(idx))
    if isinstance(idx, str) and idx.strip():
        return (el["tag"], "i:" + idx.strip())
    if el.get("bounds"):
        return (el["tag"], "b:" + el["bounds"])
    if el.get("data-bbox"):
        return (el["tag"], "d:" + el["data-bbox"])
    return (el["tag"], "t:" + el.get("text", ""))


# ── 메인 공개 함수 ─────────────────────────────────────────────────────────


def build_token_weights(
    tokenizer,
    system: str,
    user: str,
    future_html: str,
    diff_result: list[dict],
    prefix_text: str | None = None,
    weight_map: dict[str, float] | None = None,
) -> list[float]:
    """
    전체 시퀀스(system + user + assistant)에 대한 token_weights 배열 반환.

    Args:
        tokenizer   : HuggingFace tokenizer (offset_mapping 지원 필요)
        system      : system 메시지 텍스트
        user        : user 메시지 텍스트
        future_html : assistant 답변 (= future state HTML)
        diff_result : classify_diff()의 반환값
        prefix_text : system+user 구간의 완성된 문자열.
                      None이면 내부에서 단순 포맷으로 생성 (모델 템플릿에 맞게 외부에서 주입 권장)
        weight_map  : diff_type별 가중치 오버라이드. None이면 모듈 기본값 사용.

    Returns:
        list[float], 길이 = len(prefix_ids) + len(asst_ids)
        - prefix 구간 : 0.0  (labels=-100이므로 loss에 기여하지 않음)
        - assistant 구간 : diff_type별 wmap 값 (기본값 ADDED=MODIFIED=1.0 / UNCHANGED=0.25)
          element에 해당하지 않는 토큰(태그 문법 등)은 baseline(=wmap["UNCHANGED"]) 유지
    """
    wmap = weight_map if weight_map is not None else WEIGHT_MAP
    # baseline = UNCHANGED 가중치. 하드코딩된 1.0을 쓰면 diff 가중치가 1.0인 체계에서
    # diff 토큰이 스킵되어 baseline에 방치된다.
    base = wmap.get("UNCHANGED", 1.0)

    # ── 1. prefix 텍스트 결정 ───────────────────────────────────────────────
    if prefix_text is None:
        # 실제 사용 시 모델 chat template에 맞는 prefix_text를 외부에서 주입할 것
        prefix_text = f"<|system|>\n{system}<|user|>\n{user}<|assistant|>\n"

    # ── 2. 토크나이즈 ──────────────────────────────────────────────────────
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

    asst_encoding = tokenizer(
        future_html,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    asst_ids = asst_encoding["input_ids"]
    offset_mapping = asst_encoding["offset_mapping"]
    # offset_mapping: [(tok_char_start, tok_char_end), ...] — future_html 기준

    n_prefix = len(prefix_ids)
    n_asst = len(asst_ids)

    # ── 3. 기본 weight 배열 초기화 ─────────────────────────────────────────
    weights = [0.0] * n_prefix + [base] * n_asst

    # ── 4. diff_result를 key 기반 딕셔너리로 변환 ──────────────────────────
    diff_by_key: dict[tuple, str] = {}
    for d in diff_result:
        key = _make_el_key(d["element"])
        # 동일 key가 여러 개이면 더 높은 가중치 우선
        existing = diff_by_key.get(key)
        if existing is None or wmap[d["diff_type"]] > wmap[existing]:
            diff_by_key[key] = d["diff_type"]

    # ── 5. char_spans 추출 (★ search_start 방식으로 중복 버그 수정) ─────────
    char_spans = get_element_char_spans(future_html)

    # ── 6. char offset → token offset 변환 후 weight 적용 ─────────────────
    # 경계 규칙: 토큰 구간 [tok_cs, tok_ce) 와 element 구간 [char_start, char_end) 가
    # 조금이라도 겹치면 가중치를 준다 (interval overlap). 토큰 **시작점만** 보면
    # element 왼쪽 경계를 걸친 토큰(앞 텍스트와 합쳐 토크나이즈된 경우)을 놓치고,
    # 오른쪽으로 넘치는 토큰에는 주면서 왼쪽은 안 주는 비대칭이 생긴다.
    # zero-length offset (special token 등) 은 어디에도 겹치지 않으므로 제외한다.
    for char_start, char_end, el_info in char_spans:
        el_key = _make_el_key(el_info)
        diff_type = diff_by_key.get(el_key, "UNCHANGED")
        weight = wmap[diff_type]

        if weight == base:
            continue  # baseline 그대로이므로 스킵

        for tok_i, (tok_cs, tok_ce) in enumerate(offset_mapping):
            if tok_cs == tok_ce:
                continue  # zero-length (special token 등)
            if tok_cs < char_end and tok_ce > char_start:
                pos = n_prefix + tok_i
                cur = weights[pos]
                # 서로 다른 span 에 동시에 걸친 토큰(중첩 element 경계 등)은
                # **더 큰 가중치**를 채택한다 (명시적 정책 — 마지막 span 이 이기던
                # 기존 덮어쓰기 방식은 순서 의존적이었다).
                weights[pos] = weight if cur == base else max(cur, weight)

    return weights
