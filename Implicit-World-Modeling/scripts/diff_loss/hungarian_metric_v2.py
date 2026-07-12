"""
hungarian_metric_v2.py
────────────────────────────────────────────────────────────────
hungarian_metric.py의 v2 개선판. 원본은 그대로 두고 이 파일에 개선사항만 반영.

v1 대비 변경점:
  [수정 2] _collect_texts(): 자손 텍스트 흡수 제거 → 자기 자신의 direct text와
           자체 속성(description/id/text/aria-label)만 사용.

  [수정 3] _match_cost(): index 완전 제거. bounds="[x1,y1][x2,y2]" 좌표의
           중심점 거리만 위치 cost로 사용. bounds가 없으면 위치 신호 없음(0).

  [수정 3b] compute_hungarian_acc의 hungarian_idx → hungarian_pos로 교체.
            DOM index 차이 대신 bounds 중심점 거리(px)로 위치 정확도 측정.

시그니처 스키마는 v1과 대부분 호환. 다만 metric key `hungarian_idx`는
`hungarian_pos`로 이름이 바뀜(의미 자체가 다르므로).
"""

from __future__ import annotations
from typing import Any
import re

from bs4 import BeautifulSoup, NavigableString
from munkres import Munkres

# ── 파서 설정 ──────────────────────────────────────────────────────────────
INTERACTIVE_TAGS = {"button", "input", "a", "select", "textarea"}
CONTENT_TAGS     = {"p", "img", "span"}
CLICKABLE_ATTRS  = {"clickable", "long-clickable"}

# ── 비용 함수 가중치 ────────────────────────────────────────────────────────
W_TAG    = 3.0    # tag 불일치 패널티
W_TEXT   = 1.5    # text 불일치
W_POS    = 0.4    # bounds 중심점 거리 (개선 B: 0.2 → 0.4, 위치 신호 상향)

MATCH_THRESHOLD = 1.7       # 개선 D: 1.5 → 1.7 (같은 자리 텍스트 변경도 MODIFIED로 인정)
BOUNDS_NORM     = 2050.0    # 화면 대각선 근사값(840x1876) — 거리 정규화용
BOUNDS_TAU      = 50.0      # hungarian_pos 계산 시 "위치 정확" 기준 (px)


# ── 요소 추출 ──────────────────────────────────────────────────────────────

def _collect_texts(el: Any) -> str:
    """[수정 2] 요소의 direct text와 자체 속성만 수집.
    자손 요소의 텍스트는 흡수하지 않음."""
    tokens: set[str] = set()

    def add(v: str | None) -> None:
        if v:
            tokens.add(v.strip())

    add(el.get("description"))
    add(el.get("id"))
    add(el.get("text"))
    add(el.get("aria-label"))

    for c in el.contents:
        if isinstance(c, NavigableString):
            s = str(c).strip()
            if s:
                tokens.add(s)

    return " | ".join(sorted(tokens)) if tokens else ""


def extract_elements(xml_str: str) -> list[dict]:
    """XML/HTML에서 의미있는 요소를 평탄화 추출."""
    try:
        soup = BeautifulSoup(xml_str, "xml")
    except Exception:
        soup = BeautifulSoup(xml_str, "html.parser")

    elements: list[dict] = []
    for el in soup.find_all(True):
        tag  = el.name
        text = _collect_texts(el)

        is_interactive = tag in INTERACTIVE_TAGS
        is_content     = (tag in CONTENT_TAGS) and bool(text)
        is_clickable   = any(el.get(a) for a in CLICKABLE_ATTRS)
        is_described   = bool(el.get("description"))

        if is_interactive or is_content or is_clickable or is_described:
            elements.append({
                "tag":    tag,
                "text":   text,
                "bounds": el.get("bounds", "") or "",
            })

    return elements


# ── 비용 함수 ──────────────────────────────────────────────────────────────

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


def _parse_bounds_center(s: str) -> tuple[float, float] | None:
    """'[x1,y1][x2,y2]' → 중심점 (cx, cy). 실패 시 None."""
    if not s:
        return None
    m = re.findall(r'\[(-?\d+),(-?\d+)\]', s)
    if len(m) >= 2:
        x1, y1 = int(m[0][0]), int(m[0][1])
        x2, y2 = int(m[1][0]), int(m[1][1])
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return None


def _bounds_dist(e1: dict, e2: dict) -> float | None:
    c1 = _parse_bounds_center(e1.get("bounds", ""))
    c2 = _parse_bounds_center(e2.get("bounds", ""))
    if c1 is None or c2 is None:
        return None
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _match_cost(e1: dict, e2: dict) -> float:
    """[수정 3] bounds 좌표만으로 위치 cost 계산. index 미사용."""
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

def _hungarian_match(pred: list[dict], gt: list[dict]) -> tuple[list[tuple], list[list]]:
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return [], []

    matrix: list[list[float]] = [
        [_match_cost(p, g) for g in gt]
        for p in pred
    ]

    size   = max(n, m)
    padded = [row + [MATCH_THRESHOLD * 2] * (size - len(row)) for row in matrix]
    while len(padded) < size:
        padded.append([MATCH_THRESHOLD * 2] * size)

    indexes = Munkres().compute(padded)

    pairs: list[tuple] = []
    for i, j in indexes:
        if i < n and j < m and matrix[i][j] < MATCH_THRESHOLD:
            pairs.append((i, j, matrix[i][j]))

    return pairs, matrix


# ── 메인 공개 함수 ─────────────────────────────────────────────────────────

def compute_hungarian_acc(pred_str: str, gt_str: str) -> dict[str, float]:
    """매칭 기반 평가 메트릭 반환.

    반환 키:
      - hungarian_ea    : Element Accuracy
      - hungarian_f1    : F1
      - hungarian_prec  : Precision
      - hungarian_rec   : Recall
      - hungarian_text  : 매칭된 쌍의 평균 텍스트 유사도
      - hungarian_pos   : 매칭된 쌍의 위치 정확도 (bounds 중심점 거리 ≤ BOUNDS_TAU 비율)
    """
    _zero = {
        "hungarian_ea":   0.0,
        "hungarian_f1":   0.0,
        "hungarian_prec": 0.0,
        "hungarian_rec":  0.0,
        "hungarian_text": 0.0,
        "hungarian_pos":  0.0,
    }

    try:
        pred_els = extract_elements(pred_str)
        gt_els   = extract_elements(gt_str)
    except Exception:
        return _zero

    if not gt_els:
        return _zero

    pairs, _ = _hungarian_match(pred_els, gt_els)

    n_pred    = len(pred_els)
    n_gt      = len(gt_els)
    n_matched = len(pairs)

    ea   = n_matched / max(n_pred, n_gt) if max(n_pred, n_gt) > 0 else 0.0
    prec = n_matched / n_pred             if n_pred  > 0           else 0.0
    rec  = n_matched / n_gt               if n_gt    > 0           else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0    else 0.0

    if pairs:
        text_sims = [_text_sim(pred_els[i]["text"], gt_els[j]["text"])
                     for i, j, _ in pairs]
        dists = [_bounds_dist(pred_els[i], gt_els[j]) for i, j, _ in pairs]
        dists_valid = [d for d in dists if d is not None]

        text_avg = sum(text_sims) / len(text_sims)
        pos_acc  = (sum(1 for d in dists_valid if d <= BOUNDS_TAU) / len(dists_valid)
                    if dists_valid else 0.0)
    else:
        text_avg = 0.0
        pos_acc  = 0.0

    return {
        "hungarian_ea":   round(ea,       4),
        "hungarian_f1":   round(f1,       4),
        "hungarian_prec": round(prec,     4),
        "hungarian_rec":  round(rec,      4),
        "hungarian_text": round(text_avg, 4),
        "hungarian_pos":  round(pos_acc,  4),
    }


def aggregate_hungarian_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {k: round(sum(r[k] for r in results) / len(results), 4) for k in keys}
