"""
hungarian_metric.py
────────────────────────────────────────────────────────────────
LLaMA Factory SFT 훈련 시 eval 단계에서 hungarian_acc를 계산하는 모듈.

compute_hungarian_acc(pred_str, gt_str) 하나만 외부에서 호출하면 됨.
반환값: dict
  - hungarian_ea   : Element Accuracy  (매칭수 / max(pred수, gt수))
  - hungarian_f1   : F1-Score
  - hungarian_prec : Precision
  - hungarian_rec  : Recall
  - hungarian_text : TextSimilarity 평균 (매칭된 쌍 기준)
  - hungarian_idx  : index_acc (차이 ≤ τ 비율)

의존 패키지: beautifulsoup4, munkres, lxml
  pip install beautifulsoup4 munkres lxml
"""

from __future__ import annotations
from typing import Any
import re

from bs4 import BeautifulSoup
from munkres import Munkres

# ── 파서 설정 ──────────────────────────────────────────────────────────────
INTERACTIVE_TAGS = {"button", "input", "a", "select", "textarea"}
CONTENT_TAGS     = {"p", "img", "span"}
CLICKABLE_ATTRS  = {"clickable", "long-clickable"}

# ── 비용 함수 가중치 ────────────────────────────────────────────────────────
W_TAG   = 3.0   # tag 불일치 패널티 (가장 강함 — 다른 종류는 절대 매칭 안 되게)
W_TEXT  = 1.5   # text 불일치
W_INDEX = 0.2   # DOM index 거리

MATCH_THRESHOLD = 1.5   # 이 이상이면 매칭 거부 → ADDED / DELETED
INDEX_TAU       = 2     # index_acc: 차이 ≤ τ 이면 위치 정확


# ── 요소 추출 ──────────────────────────────────────────────────────────────

def _collect_texts(el: Any) -> str:
    """요소 자신 + 자식 전체에서 텍스트 토큰 수집 (중복 제거, 알파벳순 join).

    text/description/id 속성과 inner text를 모두 토큰으로 본다.
    """
    tokens: set[str] = set()

    def add(v: str | None) -> None:
        if v:
            tokens.add(v.strip())

    add(el.get("description"))
    add(el.get("id"))
    add(el.get("text"))

    for child in el.find_all(True):
        add(child.get("description"))
        add(child.get("id"))
        add(child.get("text"))
        t = child.get_text(strip=True)
        if t:
            tokens.add(t)

    t = el.get_text(strip=True)
    if t:
        tokens.add(t)

    return " | ".join(sorted(tokens)) if tokens else ""


def extract_elements(xml_str: str) -> list[dict]:
    """
    XML/HTML 문자열에서 의미있는 요소를 평탄화하여 추출.

    포함 기준:
      - INTERACTIVE_TAGS (button, input, a, select, textarea)
      - CONTENT_TAGS (p, img, span) 중 텍스트가 있는 것
      - clickable / long-clickable 속성을 가진 요소
      - description 속성을 가진 요소 (시계/숫자 다이얼처럼
        description 으로만 의미가 표현되는 위젯을 포착)

    각 요소: {"tag": str, "text": str, "index": int}
    """
    try:
        soup = BeautifulSoup(xml_str, "xml")
    except Exception:
        soup = BeautifulSoup(xml_str, "html.parser")

    elements: list[dict] = []
    for el in soup.find_all(True):
        tag  = el.name
        idx  = _safe_int(el.get("index", -1))
        text = _collect_texts(el)

        is_interactive = tag in INTERACTIVE_TAGS
        is_content     = (tag in CONTENT_TAGS) and bool(text)
        is_clickable   = any(el.get(a) for a in CLICKABLE_ATTRS)
        is_described   = bool(el.get("description"))

        if is_interactive or is_content or is_clickable or is_described:
            elements.append({"tag": tag, "text": text, "index": idx})

    return elements


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ── 비용 함수 ──────────────────────────────────────────────────────────────

def _text_sim(a: str, b: str) -> float:
    """Jaccard 유사도 (단어 집합 기준)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(a.lower().replace("|", "").split())
    sb = set(b.lower().replace("|", "").split())
    union = sa | sb
    if not union:
        return 1.0  # 양쪽 다 구분자만 있는 경우
    return len(sa & sb) / len(union)


def _match_cost(e1: dict, e2: dict, max_idx: int) -> float:
    """pred 요소 e1과 gt 요소 e2 사이의 매칭 비용."""
    if e1["tag"] != e2["tag"]:
        return W_TAG
    tc = W_TEXT  * (1.0 - _text_sim(e1["text"], e2["text"]))
    ic = W_INDEX * (abs(e1["index"] - e2["index"]) / max(max_idx, 1))
    return round(tc + ic, 5)


# ── 헝가리안 매칭 ──────────────────────────────────────────────────────────

def _hungarian_match(pred: list[dict], gt: list[dict]) -> tuple[list[tuple], list[list]]:
    """
    헝가리안 알고리즘으로 pred-gt 간 최적 1:1 매칭을 찾는다.

    반환:
        pairs  : [(pred_idx, gt_idx, cost), ...]  — threshold 미만인 쌍만
        matrix : 원본 비용 행렬 (list[list[float]])
    """
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return [], []

    max_idx = max(
        (e["index"] for e in pred + gt if e["index"] >= 0),
        default=1,
    )
    matrix: list[list[float]] = [
        [_match_cost(p, g, max_idx) for g in gt]
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
    """
    모델이 생성한 XML 문자열(pred_str)과 정답 XML 문자열(gt_str)을 비교해
    hungarian 기반 평가 메트릭을 반환한다.
    """
    _zero = {
        "hungarian_ea":   0.0,
        "hungarian_f1":   0.0,
        "hungarian_prec": 0.0,
        "hungarian_rec":  0.0,
        "hungarian_text": 0.0,
        "hungarian_idx":  0.0,
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
        text_sims  = [_text_sim(pred_els[i]["text"], gt_els[j]["text"])
                      for i, j, _ in pairs]
        idx_diffs  = [abs(pred_els[i]["index"] - gt_els[j]["index"])
                      for i, j, _ in pairs]
        text_avg   = sum(text_sims) / len(text_sims)
        idx_acc    = sum(1 for d in idx_diffs if d <= INDEX_TAU) / len(idx_diffs)
    else:
        text_avg = 0.0
        idx_acc  = 0.0

    return {
        "hungarian_ea":   round(ea,       4),
        "hungarian_f1":   round(f1,       4),
        "hungarian_prec": round(prec,     4),
        "hungarian_rec":  round(rec,      4),
        "hungarian_text": round(text_avg, 4),
        "hungarian_idx":  round(idx_acc,  4),
    }


# ── 배치 집계 헬퍼 ─────────────────────────────────────────────────────────

def aggregate_hungarian_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """compute_hungarian_acc 결과 리스트를 받아 키별 평균을 반환."""
    if not results:
        return {}
    keys = results[0].keys()
    return {k: round(sum(r[k] for r in results) / len(results), 4) for k in keys}
