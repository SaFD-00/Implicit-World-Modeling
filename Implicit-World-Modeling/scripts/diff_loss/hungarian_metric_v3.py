"""
hungarian_metric_v3.py
────────────────────────────────────────────────────────────────
hungarian_metric_v2.py 의 v3 개선판. v1/v2 는 재현성 때문에 불가침이라
개선은 항상 새 파일로 간다 (하드 제약 9).

v2 대비 변경점 — **element 집합**이 유일한 축이다:

  [수정 A] 화이트리스트 제거. v2 는 `INTERACTIVE_TAGS ∪ CONTENT_TAGS ∪ clickable ∪
           description` 만 채택했다. EXP05 XML 에는 `description` 이 0건이고
           컨테이너 `div` 대부분이 아무 조건에도 안 걸려 **element 집합에서 통째로
           빠졌다** — 화면 변화의 상당수가 그 안에서 일어나는데 관측조차 안 됐고,
           가중치 빌더 쪽에서는 그 요소들이 char span 을 못 얻어 baseline 으로
           방치됐다 (= "구조 토큰이 최저 가중치"). v3 는 **모든 요소**를 채택한다.

  [수정 B] `tag = el.name.lower()`. 파서/원본에 따라 대소문자가 섞이면 `_match_cost`
           의 tag 비교가 조용히 어긋나 전부 W_TAG(=매칭 실패) 로 떨어진다.

  [수정 C] 텍스트는 "자체 우선, 비면 자손 흡수". 자체 텍스트가 있으면 v2 와 동일하고
           (자손 흡수 없음), 자체 텍스트가 없는 컨테이너만 자손 텍스트를 흡수해
           매칭 신호를 얻는다. `own_text` 는 별도 키로 함께 실어 보낸다 —
           유도성 분류(hungarian_diff_v3)는 **반드시 own_text 로만** 판정해야
           하기 때문이다 (흡수 텍스트로 판정하면 루트가 화면 전체 텍스트를 갖고
           NON_DERIVABLE 로 오분류된다).

  [수정 D] 파서 폴백. lxml "xml" 파서는 입력이 well-formed XML 이 아니면 예외를
           던지지 않고 **빈 트리를 조용히** 돌려준다 → 요소 0개로 전 지표 0.0.
           비어있지 않은 입력이 요소 0개면 파싱 실패로 보고 html.parser 로 재시도한다
           (`scripts/_hungarian_eval.py` 의 같은 수정과 동일 규칙).

  [수정 E] 할당 솔버는 scipy 우선, 없으면 munkres 폴백 (`_hungarian_eval.py` 와 동일).
           화이트리스트를 없애면 비용 행렬이 커지는데 munkres 는 순수 파이썬이라
           크기에 급격히 느려진다.

비용 함수(W_TAG/W_TEXT/W_POS)·임계값·반환 키는 v2 와 같다 — 이 파일의 변경은
"무엇을 요소로 볼 것인가" 하나뿐이라, v2 대비 차이가 element 집합에서만 나오게 한다.
"""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup, NavigableString

# ── 비용 함수 가중치 (v2 와 동일) ───────────────────────────────────────────
W_TAG = 3.0  # tag 불일치 패널티
W_TEXT = 1.5  # text 불일치
W_POS = 0.4  # bounds 중심점 거리

MATCH_THRESHOLD = 1.7
BOUNDS_NORM = 2050.0  # 화면 대각선 근사값(840x1876) — 거리 정규화용
BOUNDS_TAU = 50.0  # hungarian_pos 계산 시 "위치 정확" 기준 (px)

TEXT_ATTRS = ("description", "id", "text", "aria-label")

_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]")


def _make_solver():
    """Hungarian 할당 솔버 — scipy(C) 우선, 없으면 munkres(순수 파이썬).

    `scripts/_hungarian_eval.py::_make_solver` 와 같은 이유·같은 우선순위다:
    munkres 는 행렬이 커지면 급격히 느려진다(실측 600×600 에서 scipy 대비 1만배).
    v3 는 화이트리스트를 없애 요소 수가 늘어나므로 이 폴백 순서가 v2 보다 중요하다.
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        def solve(padded):
            r, c = linear_sum_assignment(np.asarray(padded, dtype=float))
            return list(zip(r.tolist(), c.tolist()))

        return solve
    except ImportError:  # scipy 가 없는 환경 대비한 안전망
        from munkres import Munkres

        return lambda padded: Munkres().compute(padded)


_solve = None


# ── 파싱 / 요소 추출 ───────────────────────────────────────────────────────


def parse_soup(xml_str: str) -> BeautifulSoup:
    """[수정 D] xml 파서 우선, 빈 트리면 html.parser 로 재시도."""
    try:
        soup = BeautifulSoup(xml_str, "xml")
        if xml_str.strip() and not soup.find_all(True):
            raise ValueError("xml parse produced no elements")
    except Exception:
        soup = BeautifulSoup(xml_str, "html.parser")
    return soup


def iter_nodes(soup: BeautifulSoup) -> list:
    """문서 순서(pre-order) 노드 목록.

    `build_element_records` 와 `hungarian_diff_v3` 의 유도성 분류가 **같은 순서**를
    보게 하는 단일 진입점이다. 두 결과는 seq_idx 로 위치 정렬되므로 이 순서가
    어긋나면 diff × derivability 결합이 조용히 뒤섞인다.
    """
    return soup.find_all(True)


def _own_tokens(el: Any) -> set[str]:
    """요소 자신의 텍스트 토큰 — 자체 속성 + direct text. 자손 흡수 없음."""
    tokens: set[str] = set()
    for k in TEXT_ATTRS:
        v = el.get(k)
        if v and v.strip():
            tokens.add(v.strip())
    for c in el.contents:
        if isinstance(c, NavigableString):
            s = str(c).strip()
            if s:
                tokens.add(s)
    return tokens


def own_text(el: Any) -> str:
    """자체 텍스트만 (자손 흡수 없음). 유도성 분류가 쓰는 정본 텍스트."""
    t = _own_tokens(el)
    return " | ".join(sorted(t)) if t else ""


def build_element_records(nodes: list) -> list[dict]:
    """[수정 A·B·C] 노드 목록 → element dict 목록. **필터 없이 1:1** 대응.

    자손 흡수는 문서 순서 **역순** 한 번의 누적으로 계산한다 (역순이면 자손이
    항상 먼저 확정된다). 노드마다 find_all 을 부르는 방식은 O(n²) 라 깊은 트리에서
    전처리 전체를 느리게 만든다.
    """
    own_map: dict[int, set[str]] = {}
    abs_map: dict[int, set[str]] = {}
    for node in reversed(nodes):
        o = _own_tokens(node)
        own_map[id(node)] = o
        acc = set(o)
        for c in node.children:
            if getattr(c, "name", None):
                acc |= abs_map.get(id(c), set())
        abs_map[id(node)] = acc

    records: list[dict] = []
    for seq_idx, node in enumerate(nodes):
        o = own_map[id(node)]
        own_s = " | ".join(sorted(o)) if o else ""
        if own_s:
            text = own_s
        else:
            a = abs_map[id(node)]
            text = " | ".join(sorted(a)) if a else ""
        records.append(
            {
                "tag": node.name.lower(),
                "text": text,
                "own_text": own_s,
                "bounds": node.get("bounds", "") or "",
                # 요소 동일성 키의 폴백 축. EXP01~04·MobiBench HTML 에는 `bounds` 가
                # 없고 `index` 만 있다 (EXP01 실측 `index=` 307,354건 / `bounds=` 0건).
                # bounds 만으로 "같은 자리" 를 판정하면 그 데이터셋에서 판정이 통째로
                # 죽는다 — `token_weight_builder_v2.py:101-107` 이 같은 문제를 같은
                # index↔bounds 폴백으로 이미 풀어놨고 그 선례를 따른다.
                "index": node.get("index", "") or "",
                "seq_idx": seq_idx,
            }
        )
    return records


def extract_elements(xml_str: str) -> list[dict]:
    """XML/HTML → element dict 목록 (문서 순서, 모든 요소)."""
    return build_element_records(iter_nodes(parse_soup(xml_str)))


# ── 비용 함수 (v2 와 동일) ─────────────────────────────────────────────────


def _text_sim(a: str, b: str) -> float:
    """Jaccard 유사도."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(a.lower().replace("|", "").split())
    sb = set(b.lower().replace("|", "").split())
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def parse_bounds(s: str) -> tuple[int, int, int, int] | None:
    """'[x1,y1][x2,y2]' → (x1, y1, x2, y2). 실패 시 None."""
    if not s:
        return None
    m = _BOUNDS_RE.findall(s)
    if len(m) >= 2:
        return int(m[0][0]), int(m[0][1]), int(m[1][0]), int(m[1][1])
    return None


def _parse_bounds_center(s: str) -> tuple[float, float] | None:
    b = parse_bounds(s)
    if b is None:
        return None
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _bounds_dist(e1: dict, e2: dict) -> float | None:
    c1 = _parse_bounds_center(e1.get("bounds", ""))
    c2 = _parse_bounds_center(e2.get("bounds", ""))
    if c1 is None or c2 is None:
        return None
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _match_cost(e1: dict, e2: dict) -> float:
    if e1["tag"] != e2["tag"]:
        return W_TAG

    tc = W_TEXT * (1.0 - _text_sim(e1["text"], e2["text"]))

    dist = _bounds_dist(e1, e2)
    if dist is None:
        pc = 0.0  # bounds 없으면 위치 신호 없음
    else:
        pc = W_POS * min(dist / BOUNDS_NORM, 1.0)

    return round(tc + pc, 5)


# ── 헝가리안 매칭 ──────────────────────────────────────────────────────────


def _hungarian_match(
    pred: list[dict], gt: list[dict]
) -> tuple[list[tuple], list[list]]:
    global _solve
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return [], []

    if _solve is None:
        _solve = _make_solver()

    matrix: list[list[float]] = [[_match_cost(p, g) for g in gt] for p in pred]

    size = max(n, m)
    padded = [row + [MATCH_THRESHOLD * 2] * (size - len(row)) for row in matrix]
    while len(padded) < size:
        padded.append([MATCH_THRESHOLD * 2] * size)

    indexes = _solve(padded)

    pairs: list[tuple] = []
    for i, j in indexes:
        if i < n and j < m and matrix[i][j] < MATCH_THRESHOLD:
            pairs.append((i, j, matrix[i][j]))

    return pairs, matrix


# ── 메인 공개 함수 ─────────────────────────────────────────────────────────


def compute_hungarian_acc(pred_str: str, gt_str: str) -> dict[str, float]:
    """매칭 기반 평가 메트릭 반환 (키는 v2 와 동일).

    반환 키:
      - hungarian_ea    : Element Accuracy
      - hungarian_f1    : F1
      - hungarian_prec  : Precision
      - hungarian_rec   : Recall
      - hungarian_text  : 매칭된 쌍의 평균 텍스트 유사도
      - hungarian_pos   : 매칭된 쌍의 위치 정확도 (bounds 중심점 거리 ≤ BOUNDS_TAU 비율)
    """
    _zero = {
        "hungarian_ea": 0.0,
        "hungarian_f1": 0.0,
        "hungarian_prec": 0.0,
        "hungarian_rec": 0.0,
        "hungarian_text": 0.0,
        "hungarian_pos": 0.0,
    }

    try:
        pred_els = extract_elements(pred_str)
        gt_els = extract_elements(gt_str)
    except Exception:
        return _zero

    if not gt_els:
        return _zero

    pairs, _ = _hungarian_match(pred_els, gt_els)

    n_pred = len(pred_els)
    n_gt = len(gt_els)
    n_matched = len(pairs)

    ea = n_matched / max(n_pred, n_gt) if max(n_pred, n_gt) > 0 else 0.0
    prec = n_matched / n_pred if n_pred > 0 else 0.0
    rec = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    if pairs:
        text_sims = [
            _text_sim(pred_els[i]["text"], gt_els[j]["text"]) for i, j, _ in pairs
        ]
        dists = [_bounds_dist(pred_els[i], gt_els[j]) for i, j, _ in pairs]
        dists_valid = [d for d in dists if d is not None]

        text_avg = sum(text_sims) / len(text_sims)
        pos_acc = (
            sum(1 for d in dists_valid if d <= BOUNDS_TAU) / len(dists_valid)
            if dists_valid
            else 0.0
        )
    else:
        text_avg = 0.0
        pos_acc = 0.0

    return {
        "hungarian_ea": round(ea, 4),
        "hungarian_f1": round(f1, 4),
        "hungarian_prec": round(prec, 4),
        "hungarian_rec": round(rec, 4),
        "hungarian_text": round(text_avg, 4),
        "hungarian_pos": round(pos_acc, 4),
    }


def aggregate_hungarian_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {k: round(sum(r[k] for r in results) / len(results), 4) for k in keys}
