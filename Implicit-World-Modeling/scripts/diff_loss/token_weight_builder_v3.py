"""
token_weight_builder_v3.py
────────────────────────────────────────────────────────────────
v2 의 1축(diff_type) 가중치를 **2축 (diff_type × derivability)** 으로 확장한 빌더.

    future_html
        ↓ get_element_char_spans()      # 모든 요소, extract_elements 와 seq_idx 정렬
    (char_start, char_end, {seq_idx,…})
        ↓ tokenizer offset_mapping
    token 인덱스
        ↓ (diff_type, derivability) → WEIGHT_MAP
    token_weights

v2 대비 무엇이 달라졌고 왜인가
──────────────────────────────
[1] **모든 요소가 span 을 갖는다.** v2 의 span 추출은 metric 의 화이트리스트를 그대로
    복제해서, 화이트리스트 밖 요소(EXP05 의 컨테이너 div 대부분)는 char span 을 아예
    얻지 못하고 baseline(0.25)에 방치됐다 — "구조 토큰이 최저 가중치" 라는 결함이다.
    v3 는 필터 없이 스캔한다.

[2] **span↔요소 매핑을 (tag, ident, 등장순번) 으로 한다.** v2 는 (tag, index) 또는
    (tag, bounds) 첫 등장만 썼다. 화이트리스트를 없애면 **동일 bounds 를 가진 중첩
    컨테이너**가 흔해서 첫 등장 규칙이 서로 다른 요소를 같은 span 에 묶는다.
    등장순번을 키에 넣으면 원문 스캔 순서(=문서 순서)와 soup 순서가 1:1 로 맞는다.

[3] **innermost(최협소 span)-wins.** ★ v2 의 max 병합을 그대로 쓸 수 없다.
    화이트리스트를 없애면 **루트 요소의 span 이 문서 전체**이고, 루트는 current/future
    사이에서 태그가 자주 바뀌거나(EXP05 실측: future 루트가 `<node>`, current 루트가
    `<div>`) 자손 변화로 텍스트가 흔들려 ADDED/MODIFIED 로 잡힌다. max 병합이면 루트
    하나가 전 토큰을 full weight 로 덮어써 **NON_DERIVABLE 감쇠가 0건이 된다**
    (EXP05 test id 30문서 실측: 루트 ADDED 2건 / MODIFIED 26건 / UNCHANGED 2건).
    그래서 토큰의 가중치는 **그 토큰을 가장 좁게 감싸는 요소**가 정한다. 중첩이 아닌
    형제 경계를 걸친 토큰(동률 폭)만 v2 처럼 **max** 로 가른다 — 순서 비의존은 유지된다.
    interval-overlap 규칙(v1 의 "토큰 시작점만" 비대칭 버그 수정)은 그대로 간다.

[4] baseline 은 하드코딩하지 않고 `wmap` 의 UNCHANGED 행에서 유도한다 (하드 제약 10).

외부에서 호출하는 함수:
  make_weight_map(w_added, w_modified, w_unchanged, w_non_derivable) -> dict
  build_token_weights(tokenizer, system, user, future_html, diff_result,
                      prefix_text, weight_map, deriv_result) -> list[float]
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right

from hungarian_diff_v3 import DERIVABILITY_LABELS, NON_DERIVABLE
from hungarian_metric_v3 import iter_nodes, parse_soup

DIFF_TYPES = ("ADDED", "MODIFIED", "UNCHANGED")


def make_weight_map(
    w_added: float = 1.0,
    w_modified: float = 1.0,
    w_unchanged: float = 0.25,
    w_non_derivable: float = 0.5,
) -> dict[tuple[str, str], float]:
    """2축 가중치 표 `(diff_type, derivability) -> weight`.

    정책:
      · 유도 **가능**한 ADDED/MODIFIED → full weight (w_added / w_modified).
        "버튼을 눌렀으니 이 패널이 열린다" 는 모델이 맞혀야 하는 것이다.
      · 유도 **불가능**한 ADDED/MODIFIED → `w_non_derivable` 로 감쇠.
        서버가 내려준 상품명·가격·리뷰어 이름을 full weight 로 외우게 하면
        hallucination 을 학습시킨다. 0 으로 죽이지는 않는다 — 그 자리에 "무언가
        텍스트가 온다" 는 형식은 여전히 배워야 하기 때문이다.
      · UNCHANGED 는 유도성과 무관하게 baseline(w_unchanged). 이미 현재 화면에 있는
        것이라 정의상 유도 가능하고, 감쇠 이유가 없다.
    """
    wmap: dict[tuple[str, str], float] = {}
    for diff_type, w in (
        ("ADDED", w_added),
        ("MODIFIED", w_modified),
        ("UNCHANGED", w_unchanged),
    ):
        for label in DERIVABILITY_LABELS:
            attenuate = label == NON_DERIVABLE and diff_type in ("ADDED", "MODIFIED")
            wmap[(diff_type, label)] = w_non_derivable if attenuate else w
    return wmap


#: 기본 가중치 표. preprocess_dataset_v2 에서 CLI 인자로 오버라이드된다.
WEIGHT_MAP: dict[tuple[str, str], float] = make_weight_map()


def baseline_weight(wmap: dict[tuple[str, str], float]) -> float:
    """어떤 요소에도 안 걸린 토큰의 baseline.

    하드 제약 10: **하드코딩 금지**. UNCHANGED 행에서 유도한다. 행 안에서 값이
    갈리면 최소값을 쓴다 — 아무 요소에도 안 걸린 토큰(문서 밖 공백 등)을 "가장 덜
    중요한 변화 등급" 보다 높게 쳐줄 근거가 없다.
    """
    vals = [w for (diff_type, _), w in wmap.items() if diff_type == "UNCHANGED"]
    return min(vals) if vals else 1.0


def _weight_of(
    wmap: dict[tuple[str, str], float], diff_type: str, label: str | None
) -> float:
    if label is None:
        # 유도성 정보 없이 호출된 경우(deriv_result 미제공) — v2 와 같은 1축 거동으로
        # 되돌린다: 해당 diff_type 행의 최대값(=유도 가능 취급)을 쓴다.
        row = [w for (dt, _), w in wmap.items() if dt == diff_type]
        if not row:
            raise KeyError(f"weight_map 에 diff_type={diff_type!r} 행이 없다")
        return max(row)
    try:
        return wmap[(diff_type, label)]
    except KeyError as e:
        raise KeyError(
            f"weight_map 에 ({diff_type!r}, {label!r}) 항목이 없다 — "
            "make_weight_map() 으로 만들면 전 조합이 채워진다"
        ) from e


# ── char offset 추출 ──────────────────────────────────────────────────────

# 태그 스캐너. `<!--`, `<?xml`, `<!DOCTYPE` 는 이름 첫 글자 조건에서 걸러진다.
_TAG_RE = re.compile(r"<(/?)([A-Za-z_][\w.:\-]*)([^>]*?)(/?)\s*>", re.S)
_INDEX_RE = re.compile(r'\bindex="(-?\d+)"')
_BOUNDS_RE = re.compile(r'\bbounds="([^"]*)"')


def _ident_from_attrs(attrs: str) -> str:
    """원문 속성 문자열에서 식별자. index 우선, 없으면 bounds, 둘 다 없으면 빈 문자열."""
    m = _INDEX_RE.search(attrs)
    if m:
        return "i" + m.group(1)
    m = _BOUNDS_RE.search(attrs)
    if m:
        return "b" + m.group(1)
    return ""


def _ident_from_node(node) -> str:
    """soup 노드에서 같은 규칙의 식별자. 원문 스캔 쪽과 **문자열이 정확히 같아야** 한다."""
    idx = node.get("index")
    if idx is not None and str(idx).strip() != "":
        return "i" + str(idx).strip()
    b = node.get("bounds")
    if b is not None and str(b) != "":
        return "b" + str(b)
    return ""


def _scan_tag_spans(html: str) -> list[dict]:
    """원문을 한 번 훑어 여는 태그 순서(=문서 순서)대로 (start, end) 를 만든다.

    스택 방식이라 전체 O(n) 이다. v2 는 요소마다 닫는 태그를 앞으로 재탐색해서
    (`_find_closing_tag_end`) 요소 수에 제곱으로 늘었는데, 화이트리스트를 없앤 v3 에서는
    그 비용이 전처리 전체를 지배한다.

    깨진 마크업 방어: 짝 없는 닫는 태그는 무시하고, 끝까지 안 닫힌 여는 태그는
    **여는 태그 끝**에서 잘라 형제 요소를 삼키지 않게 한다.
    """
    spans: list[dict] = []
    stack: list[tuple[str, int, int]] = []  # (tag, span_idx, opening_tag_end)

    for m in _TAG_RE.finditer(html):
        closing, tag, attrs, self_closing = (
            m.group(1),
            m.group(2).lower(),
            m.group(3),
            m.group(4),
        )
        if closing:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == tag:
                    # k 위의 미닫힌 것들은 자기 여는 태그 끝에서 닫는다
                    while len(stack) - 1 > k:
                        _, si, oe = stack.pop()
                        spans[si]["end"] = oe
                    _, si, _ = stack.pop()
                    spans[si]["end"] = m.end()
                    break
            continue

        spans.append(
            {
                "tag": tag,
                "ident": _ident_from_attrs(attrs),
                "start": m.start(),
                "end": m.end(),
            }
        )
        if not self_closing:
            stack.append((tag, len(spans) - 1, m.end()))

    while stack:
        _, si, oe = stack.pop()
        spans[si]["end"] = oe

    return spans


def get_element_char_spans(html: str) -> list[tuple[int, int, dict]]:
    """모든 요소의 (char_start, char_end, info) — **문서 순서**.

    info["seq_idx"] 는 `hungarian_metric_v3.extract_elements(html)` 의 인덱스와 같다.
    diff/derivability 결과와의 결합은 이 seq_idx 로만 한다 (키 충돌 없음).

    원문 스캔과 soup 순회가 둘 다 문서 순서라, 같은 (tag, ident) 의 n 번째 등장끼리
    맞물린다. 파서가 깨진 태그를 버려 순번이 밀리면 그 (tag, ident) 계열만 해석에
    실패하고 해당 요소는 span 없이 빠진다 (조용히 어긋나지 않게 `span_coverage()` 로
    셀 수 있다).
    """
    scanned = _scan_tag_spans(html)
    by_key: dict[tuple[str, str, int], tuple[int, int]] = {}
    seen: dict[tuple[str, str], int] = {}
    for s in scanned:
        k = (s["tag"], s["ident"])
        n = seen.get(k, 0)
        seen[k] = n + 1
        by_key[(s["tag"], s["ident"], n)] = (s["start"], s["end"])

    nodes = iter_nodes(parse_soup(html))
    out: list[tuple[int, int, dict]] = []
    used: dict[tuple[str, str], int] = {}
    for seq_idx, node in enumerate(nodes):
        tag = node.name.lower()
        ident = _ident_from_node(node)
        n = used.get((tag, ident), 0)
        used[(tag, ident)] = n + 1
        pos = by_key.get((tag, ident, n))
        if pos is None:
            continue
        out.append(
            (
                pos[0],
                pos[1],
                {
                    "seq_idx": seq_idx,
                    "tag": tag,
                    "bounds": node.get("bounds", "") or "",
                },
            )
        )
    return out


def span_coverage(html: str) -> dict[str, int]:
    """진단용 — 요소 수 대비 span 해석 성공 수. 미해석이 있으면 그 요소는 baseline 이다."""
    n_el = len(iter_nodes(parse_soup(html)))
    n_span = len(get_element_char_spans(html))
    return {"elements": n_el, "spans": n_span, "unresolved": n_el - n_span}


# ── 메인 공개 함수 ─────────────────────────────────────────────────────────


def _index_by_seq(result: list[dict] | None, key: str) -> dict[int, str]:
    """diff/deriv 결과를 seq_idx → 값 으로. seq_idx 는 top-level 우선, 없으면 element 안."""
    out: dict[int, str] = {}
    if not result:
        return out
    for d in result:
        idx = d.get("future_seq_idx")
        if idx is None:
            idx = d.get("element", {}).get("seq_idx")
        if idx is None:
            continue
        out[int(idx)] = d[key]
    return out


def build_token_weights(
    tokenizer,
    system: str,
    user: str,
    future_html: str,
    diff_result: list[dict],
    prefix_text: str | None = None,
    weight_map: dict[tuple[str, str], float] | None = None,
    deriv_result: list[dict] | None = None,
) -> list[float]:
    """전체 시퀀스(prefix + assistant)에 대한 token_weights 배열.

    Args:
        diff_result  : hungarian_diff_v3.classify_diff() 반환값
        deriv_result : hungarian_diff_v3.classify_derivability() 반환값.
                       None 이면 유도성 축 없이 v2 와 같은 1축 거동.
        weight_map   : 2축 표. None 이면 모듈 기본값(WEIGHT_MAP).

    Returns:
        list[float], 길이 = len(prefix_ids) + len(asst_ids)
        - prefix 구간    : 0.0 (labels=-100 이라 loss 에 기여하지 않음)
        - assistant 구간 : 토큰을 가장 좁게 감싸는 요소의 (diff_type × derivability) 가중치,
                           어떤 요소에도 안 걸리면 baseline
    """
    wmap = weight_map if weight_map is not None else WEIGHT_MAP
    base = baseline_weight(wmap)

    if prefix_text is None:
        # 실제 사용 시 모델 chat template 에 맞는 prefix_text 를 외부에서 주입할 것
        prefix_text = f"<|system|>\n{system}<|user|>\n{user}<|assistant|>\n"

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    asst_encoding = tokenizer(
        future_html,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    asst_ids = asst_encoding["input_ids"]
    offset_mapping = asst_encoding["offset_mapping"]

    n_prefix = len(prefix_ids)
    n_asst = len(asst_ids)
    weights = [0.0] * n_prefix + [base] * n_asst

    diff_by_seq = _index_by_seq(diff_result, "diff_type")
    deriv_by_seq = _index_by_seq(deriv_result, "derivability")

    # zero-length offset (special token 등) 은 어떤 span 과도 겹치지 않으므로 제외.
    valid = [(cs, ce, i) for i, (cs, ce) in enumerate(offset_mapping) if ce > cs]
    starts = [v[0] for v in valid]
    ends = [v[1] for v in valid]
    # 정상 토크나이저의 offset 은 단조 증가라 이분탐색으로 겹치는 토큰 구간을 O(log n)
    # 에 찾는다. 단조가 아니면(비표준 토크나이저) 조용히 놓치지 말고 전수 스캔으로 내려간다.
    monotonic = all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1)) and all(
        ends[i] <= ends[i + 1] for i in range(len(ends) - 1)
    )

    # 토큰별 (감싸는 span 의 폭, 가중치). 폭이 작을수록(=안쪽 요소) 이긴다.
    best: list[tuple[int, float] | None] = [None] * n_asst

    for char_start, char_end, info in get_element_char_spans(future_html):
        seq_idx = info["seq_idx"]
        diff_type = diff_by_seq.get(seq_idx, "UNCHANGED")
        weight = _weight_of(wmap, diff_type, deriv_by_seq.get(seq_idx))
        width = char_end - char_start

        if monotonic:
            lo = bisect_right(ends, char_start)  # ends[k] > char_start 인 첫 토큰
            hi = bisect_left(starts, char_end)  # starts[k] >= char_end 인 첫 토큰
            candidates = range(lo, max(lo, hi))
        else:
            candidates = range(len(valid))

        for k in candidates:
            cs, ce, tok_i = valid[k]
            if not (cs < char_end and ce > char_start):
                continue  # interval overlap (토큰 시작점만 보던 v1 비대칭 버그 방지)
            cur = best[tok_i]
            if cur is None or width < cur[0] or (width == cur[0] and weight > cur[1]):
                best[tok_i] = (width, weight)

    for tok_i, b in enumerate(best):
        if b is not None:
            weights[n_prefix + tok_i] = b[1]

    return weights
