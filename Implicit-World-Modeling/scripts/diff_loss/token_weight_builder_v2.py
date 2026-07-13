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

    # ── Step 1: regex로 원본 HTML에서 (tag, index) → 위치 맵 구축 ──────
    pos_map: dict[tuple, tuple[int, int]] = {}

    for m in re.finditer(r"<(\w+)\b([^>]*?)\s*(/?)>", html):
        tag_name = m.group(1)
        attrs_str = m.group(2)
        self_closing = m.group(3) == "/"

        idx_m = re.search(r'\bindex="(-?\d+)"', attrs_str)
        bounds_m = re.search(r'\bbounds="([^"]+)"', attrs_str)

        # index를 우선 키로 사용, 없으면 bounds로 폴백
        if idx_m is not None:
            key = (tag_name, int(idx_m.group(1)))
        elif bounds_m is not None:
            key = (tag_name, bounds_m.group(1))
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

    # ── Step 2: BeautifulSoup으로 대상 element 필터링 후 위치 매핑 ─────
    spans: list[tuple[int, int, dict]] = []

    for el in soup.find_all(True):
        tag = el.name

        is_interactive = tag in INTERACTIVE_TAGS
        is_clickable = any(el.get(a) for a in CLICKABLE_ATTRS)
        text = el.get_text(strip=True)
        # hungarian_metric._collect_texts와 동일하게 description/id/text 속성도 텍스트로 간주
        has_text = (
            bool(text)
            or bool(el.get("description"))
            or bool(el.get("id"))
            or bool(el.get("text"))
        )
        is_content = (tag in CONTENT_TAGS) and has_text
        is_described = bool(el.get("description"))

        if not (is_interactive or is_content or is_clickable or is_described):
            continue

        idx = -1
        try:
            idx = int(el.get("index", -1))
        except (TypeError, ValueError):
            pass

        bounds = el.get("bounds", "") or ""

        # extract 쪽과 동일한 우선순위로 키 생성
        if idx != -1:
            key = (tag, idx)
        elif bounds:
            key = (tag, bounds)
        else:
            continue

        if key not in pos_map:
            continue

        char_start, char_end = pos_map[key]
        spans.append(
            (
                char_start,
                char_end,
                {
                    "tag": tag,
                    "index": idx,
                    "text": text,
                    "bounds": bounds,
                },
            )
        )

    spans.sort(key=lambda x: x[0])
    return spans


# ── element key 빌더 ──────────────────────────────────────────────────────


def _make_el_key(el: dict) -> tuple:
    """
    diff_result의 element와 char_span의 element를 매핑할 key.

    우선순위: index → bounds → text.
    """
    if el.get("index", -1) != -1:
        return (el["tag"], el["index"])
    if el.get("bounds"):
        return (el["tag"], el["bounds"])
    return (el["tag"], el["text"])


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
